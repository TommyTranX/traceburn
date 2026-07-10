"""Core data model: spans, traces, sessions, and waste findings.

These dataclasses and their dict forms are a public interface. The span JSON
schema and the Finding JSON schema are versioned with the package; breaking
changes to either are treated as semver-major.

Span kinds:
    llm        one model API call
    tool       one tool or function invocation
    agent      a logical agent step (plan, act, reflect, a whole loop)
    retrieval  a retrieval or search step
    custom     anything else

Attribute conventions for ``llm`` spans follow the OpenTelemetry GenAI
semantic conventions where they exist:

    gen_ai.system                  provider name, for example "openai"
    gen_ai.request.model           model id sent in the request
    gen_ai.response.model          model id the provider reports back
    gen_ai.usage.input_tokens      input tokens billed at the full input rate
    gen_ai.usage.output_tokens     output tokens

plus these extensions:

    cached_input_tokens    input tokens read from a prompt cache
    cache_write_tokens     input tokens written to a prompt cache
    cost_usd               estimated cost of the call in US dollars
    usage_estimated        true when token counts were estimated locally
    stream                 true when the response was streamed
    finish_reason          provider finish or stop reason
    request                normalized request payload (messages and params)
    response               normalized response payload (text or tool calls)
    request_hash           hash of the normalized request, used by replay
                           and duplicate detection

Token accounting note: ``gen_ai.usage.input_tokens`` here counts only tokens
billed at the full input rate. Cache reads and cache writes are reported
separately in ``cached_input_tokens`` and ``cache_write_tokens``. The total
prompt size is the sum of the three. Instrumentation normalizes each
provider's reporting into this shape so cost math stays auditable.
"""

from __future__ import annotations

import secrets
from dataclasses import asdict, dataclass, field
from typing import Any

SPAN_KINDS = ("llm", "tool", "agent", "retrieval", "custom")

STATUS_OK = "ok"
STATUS_ERROR = "error"


def new_span_id() -> str:
    """Return a new 8-byte hex span id (OpenTelemetry compatible width)."""
    return secrets.token_hex(8)


def new_trace_id() -> str:
    """Return a new 16-byte hex trace id (OpenTelemetry compatible width)."""
    return secrets.token_hex(16)


def new_session_id() -> str:
    return secrets.token_hex(8)


@dataclass
class Span:
    span_id: str
    trace_id: str
    name: str
    kind: str = "custom"
    parent_id: str | None = None
    session_id: str | None = None
    start_ns: int = 0
    end_ns: int | None = None
    status: str = STATUS_OK
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ns(self) -> int | None:
        if self.end_ns is None:
            return None
        return self.end_ns - self.start_ns

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Span:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass
class Trace:
    trace_id: str
    name: str
    session_id: str | None = None
    start_ns: int = 0
    end_ns: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass
class Session:
    session_id: str
    name: str
    start_ns: int = 0
    end_ns: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


SEVERITIES = ("info", "low", "medium", "high")


@dataclass
class Finding:
    """One waste-rule result.

    Every finding carries its evidence: the spans it points at and a
    quantified, defensible estimate of what was avoidable. Estimates are
    derived from observed tokens and the dated pricing table, never invented.
    ``confidence`` is one of "low", "medium", "high" and reflects how likely
    the flagged waste is real, not how large it is. ``fix`` is rule-specific,
    machine-readable data for a mechanical fix, populated only when one is
    defensible without seeing the caller's source code; None means the rule
    has no automatic fix. See ``analyze/fix.py`` for how each kind renders.
    """

    rule_id: str
    severity: str
    summary: str
    explanation: str
    trace_id: str
    span_ids: list[str] = field(default_factory=list)
    avoidable_tokens: int | None = None
    avoidable_usd: float | None = None
    avoidable_seconds: float | None = None
    confidence: str = "medium"
    fix: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})
