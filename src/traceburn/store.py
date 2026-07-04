"""SQLite storage for traces.

One local file, stdlib sqlite3, WAL mode so a viewer process can read while
an instrumented app writes. No ORM, no server, no account.

Default location is ``./.traceburn/traces.db``, overridable with the
``TRACEBURN_DB`` environment variable or the ``path`` argument.

Large request and response payloads are stored once in a ``blobs`` table,
keyed by content hash, and referenced from span attributes as
``{"$blob": "<sha256>", "bytes": <n>}``. Identical payloads are stored a
single time, which keeps the file small and makes duplicate detection and
replay lookups cheap.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .schema import Session, Span, Trace

SCHEMA_VERSION = 1

DEFAULT_DB_DIR = ".traceburn"
DEFAULT_DB_NAME = "traces.db"

# Attribute keys whose values are externalized to the blobs table.
BLOB_KEYS = ("request", "response")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    start_ns   INTEGER NOT NULL,
    end_ns     INTEGER
);
CREATE TABLE IF NOT EXISTS traces (
    trace_id   TEXT PRIMARY KEY,
    session_id TEXT,
    name       TEXT NOT NULL,
    start_ns   INTEGER NOT NULL,
    end_ns     INTEGER
);
CREATE TABLE IF NOT EXISTS spans (
    span_id      TEXT PRIMARY KEY,
    trace_id     TEXT NOT NULL,
    parent_id    TEXT,
    session_id   TEXT,
    name         TEXT NOT NULL,
    kind         TEXT NOT NULL,
    start_ns     INTEGER NOT NULL,
    end_ns       INTEGER,
    status       TEXT NOT NULL DEFAULT 'ok',
    error        TEXT,
    request_hash TEXT,
    attributes   TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS blobs (
    blob_hash TEXT PRIMARY KEY,
    content   BLOB NOT NULL,
    size      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spans_trace   ON spans (trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_parent  ON spans (parent_id);
CREATE INDEX IF NOT EXISTS idx_spans_session ON spans (session_id);
CREATE INDEX IF NOT EXISTS idx_spans_reqhash ON spans (request_hash);
CREATE INDEX IF NOT EXISTS idx_traces_session ON traces (session_id);
"""


def default_db_path() -> str:
    env = os.environ.get("TRACEBURN_DB")
    if env:
        return env
    return str(Path(DEFAULT_DB_DIR) / DEFAULT_DB_NAME)


_MAX_SANITIZE_DEPTH = 20


def _sanitize(value: Any, seen: set[int] | None = None, depth: int = 0) -> Any:
    """Coerce an arbitrary value into a JSON-safe structure, lossily but safely.

    A span must never be lost because one attribute was not serializable.
    Degradations, all explicit: non-string dict keys become strings, bytes are
    decoded with replacement characters, NaN and infinities become the strings
    "NaN", "Infinity", "-Infinity", sets and tuples become lists, circular
    references become "<circular>", anything else becomes str(value).
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if depth >= _MAX_SANITIZE_DEPTH:
        return "<max-depth>"
    if seen is None:
        seen = set()
    if isinstance(value, dict):
        if id(value) in seen:
            return "<circular>"
        seen.add(id(value))
        out = {str(k): _sanitize(v, seen, depth + 1) for k, v in value.items()}
        seen.discard(id(value))
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        if id(value) in seen:
            return "<circular>"
        seen.add(id(value))
        items = value
        if isinstance(value, (set, frozenset)):
            # Deterministic order so hashes are stable across processes.
            items = sorted(value, key=str)
        out = [_sanitize(v, seen, depth + 1) for v in items]
        seen.discard(id(value))
        return out
    return str(value)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _sanitize(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


class Store:
    """Thread-safe writer and reader over one SQLite file."""

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = str(path) if path is not None else default_db_path()
        if self.path != ":memory:":
            parent = Path(self.path).resolve().parent
            parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- schema ----------------------------------------------------------

    def _migrate(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
            # OR IGNORE keeps concurrent first-time initializers from racing.
            self._conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            found = int(row["value"])
            if found > SCHEMA_VERSION:
                raise RuntimeError(
                    f"trace database {self.path} has schema version {found}, "
                    f"but this package only supports up to {SCHEMA_VERSION}; "
                    "upgrade the package"
                )
            # Future migrations run here, in order, each bumping the version.

    # -- sessions --------------------------------------------------------

    def insert_session(self, session: Session) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, name, start_ns, end_ns) "
                "VALUES (?, ?, ?, ?)",
                (session.session_id, session.name, session.start_ns, session.end_ns),
            )

    def end_session(self, session_id: str, end_ns: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE sessions SET end_ns = ? WHERE session_id = ?",
                (end_ns, session_id),
            )

    def list_sessions(self, limit: int = 50) -> list[Session]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY start_ns DESC LIMIT ?", (limit,)
            ).fetchall()
        return [Session(**dict(r)) for r in rows]

    # -- traces ----------------------------------------------------------

    def insert_trace(self, trace: Trace) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO traces (trace_id, session_id, name, start_ns, end_ns) "
                "VALUES (?, ?, ?, ?, ?)",
                (trace.trace_id, trace.session_id, trace.name, trace.start_ns, trace.end_ns),
            )

    def extend_trace_end(self, trace_id: str, end_ns: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE traces SET end_ns = MAX(COALESCE(end_ns, 0), ?) WHERE trace_id = ?",
                (end_ns, trace_id),
            )

    def get_trace(self, trace_id: str) -> Trace | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        return Trace(**dict(row)) if row else None

    def find_traces(self, trace_id_prefix: str) -> list[Trace]:
        """Traces whose id starts with the given prefix (CLI convenience)."""
        if not trace_id_prefix:
            return []
        escaped = (
            trace_id_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM traces WHERE trace_id LIKE ? ESCAPE '\\' "
                "ORDER BY start_ns DESC",
                (escaped + "%",),
            ).fetchall()
        return [Trace(**dict(r)) for r in rows]

    def count_traces(self, session_id: str) -> int:
        with self._lock:
            (count,) = self._conn.execute(
                "SELECT COUNT(*) FROM traces WHERE session_id = ?", (session_id,)
            ).fetchone()
        return count

    def trace_stats(self, trace_id: str) -> dict:
        """Aggregates for one trace: span counts, errors, tokens, cost."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                  COUNT(*) AS span_count,
                  SUM(CASE WHEN kind = 'llm' THEN 1 ELSE 0 END) AS llm_count,
                  SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_count,
                  SUM(COALESCE(json_extract(attributes, '$.cost_usd'), 0)) AS cost_usd,
                  SUM(COALESCE(json_extract(attributes, '$."gen_ai.usage.input_tokens"'), 0)
                      + COALESCE(json_extract(attributes, '$.cached_input_tokens'), 0)
                      + COALESCE(json_extract(attributes, '$.cache_write_tokens'), 0)) AS input_tokens,
                  SUM(COALESCE(json_extract(attributes, '$."gen_ai.usage.output_tokens"'), 0)) AS output_tokens
                FROM spans WHERE trace_id = ?
                """,
                (trace_id,),
            ).fetchone()
        return dict(row)

    def list_traces(self, limit: int = 50, session_id: str | None = None) -> list[Trace]:
        with self._lock:
            if session_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM traces ORDER BY start_ns DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM traces WHERE session_id = ? ORDER BY start_ns DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
        return [Trace(**dict(r)) for r in rows]

    # -- blobs -----------------------------------------------------------

    def put_blob(self, payload: bytes) -> str:
        blob_hash = hashlib.sha256(payload).hexdigest()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO blobs (blob_hash, content, size) VALUES (?, ?, ?)",
                (blob_hash, payload, len(payload)),
            )
        return blob_hash

    def get_blob(self, blob_hash: str) -> bytes | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT content FROM blobs WHERE blob_hash = ?", (blob_hash,)
            ).fetchone()
        return bytes(row["content"]) if row else None

    def get_blob_json(self, blob_hash: str) -> Any:
        payload = self.get_blob(blob_hash)
        return None if payload is None else json.loads(payload)

    # -- spans -----------------------------------------------------------

    def insert_span(self, span: Span) -> None:
        attributes = _sanitize(dict(span.attributes))
        for key in BLOB_KEYS:
            if key in attributes and attributes[key] is not None:
                payload = _canonical_json(attributes[key])
                blob_hash = self.put_blob(payload)
                attributes[key] = {"$blob": blob_hash, "bytes": len(payload)}
        request_hash = attributes.get("request_hash")
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO spans "
                "(span_id, trace_id, parent_id, session_id, name, kind, start_ns, end_ns, "
                " status, error, request_hash, attributes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    span.span_id,
                    span.trace_id,
                    span.parent_id,
                    span.session_id,
                    span.name,
                    span.kind,
                    span.start_ns,
                    span.end_ns,
                    span.status,
                    span.error,
                    request_hash,
                    json.dumps(attributes, allow_nan=False),
                ),
            )

    def _row_to_span(self, row: sqlite3.Row, hydrate: bool) -> Span:
        attributes = json.loads(row["attributes"])
        if hydrate:
            for key in BLOB_KEYS:
                ref = attributes.get(key)
                if isinstance(ref, dict) and "$blob" in ref:
                    attributes[key] = self.get_blob_json(ref["$blob"])
        return Span(
            span_id=row["span_id"],
            trace_id=row["trace_id"],
            parent_id=row["parent_id"],
            session_id=row["session_id"],
            name=row["name"],
            kind=row["kind"],
            start_ns=row["start_ns"],
            end_ns=row["end_ns"],
            status=row["status"],
            error=row["error"],
            attributes=attributes,
        )

    def get_span(self, span_id: str, hydrate: bool = True) -> Span | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM spans WHERE span_id = ?", (span_id,)
            ).fetchone()
            return self._row_to_span(row, hydrate) if row else None

    def get_spans(self, trace_id: str, hydrate: bool = True) -> list[Span]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_ns", (trace_id,)
            ).fetchall()
            return [self._row_to_span(r, hydrate) for r in rows]

    def find_spans_by_request_hash(
        self, request_hash: str, session_id: str | None = None, hydrate: bool = False
    ) -> list[Span]:
        with self._lock:
            if session_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM spans WHERE request_hash = ? ORDER BY start_ns",
                    (request_hash,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM spans WHERE request_hash = ? AND session_id = ? ORDER BY start_ns",
                    (request_hash, session_id),
                ).fetchall()
            return [self._row_to_span(r, hydrate) for r in rows]
