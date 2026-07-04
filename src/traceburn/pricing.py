"""Cost estimation from a dated, transparent pricing table.

The bundled ``pricing.json`` maps ``provider/model`` keys to prices per
million tokens, taken from public provider pricing pages on the date in its
``as_of`` field. Prices change; every figure this module produces is an
estimate and is labeled as such throughout the tool.

Users can override or extend the table with the ``TRACEBURN_PRICING``
environment variable (a path to a JSON file whose entries overlay the
bundled ones) or programmatically with ``PricingTable.override``.

Cost math, per call:

    cost = input_tokens        * input_price
         + cached_input_tokens * cached_input_price
         + cache_write_tokens  * cache_write_price
         + output_tokens       * output_price

where ``input_tokens`` counts only tokens billed at the full input rate
(see schema.py for the token accounting convention).

Model matching: an exact key match wins. Otherwise the longest key that the
model id extends with a DATE-LIKE suffix matches: a dash followed by a
YYYYMMDD, YYYY-MM, or YYYY-MM-DD stamp. So ``claude-sonnet-5-20250929``
resolves to the ``claude-sonnet-5`` entry and ``gpt-4o-2024-08-06`` to
``gpt-4o``, while ``gpt-4o-mini`` or a new version like ``claude-opus-4-9``
never falls through to a shorter entry. Unknown models return ``None``
rather than a guess.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_BUNDLED_PATH = Path(__file__).parent / "pricing.json"

_DATE_SUFFIX = re.compile(r"^-(\d{8}|\d{4}(-\d{2}){1,2})$")


@dataclass(frozen=True)
class ModelPrice:
    """Prices in USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None


class PricingTable:
    def __init__(self, prices: dict[str, ModelPrice], as_of: str, note: str = ""):
        self._prices = {k.lower(): v for k, v in prices.items()}
        self.as_of = as_of
        self.note = note

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> PricingTable:
        """Load the bundled table, then overlay TRACEBURN_PRICING if set.

        ``path``, when given, is used instead of the environment variable.
        """
        table = cls._from_file(_BUNDLED_PATH)
        override_path = path or os.environ.get("TRACEBURN_PRICING")
        if override_path:
            overlay = cls._from_file(override_path)
            merged = dict(table._prices)
            merged.update(overlay._prices)
            return cls(merged, overlay.as_of or table.as_of, overlay.note or table.note)
        return table

    @classmethod
    def _from_file(cls, path: str | os.PathLike) -> PricingTable:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        prices = {}
        for key, entry in data.get("prices", {}).items():
            prices[key] = ModelPrice(
                input_per_mtok=float(entry["input_per_mtok"]),
                output_per_mtok=float(entry["output_per_mtok"]),
                cached_input_per_mtok=_opt_float(entry.get("cached_input_per_mtok")),
                cache_write_per_mtok=_opt_float(entry.get("cache_write_per_mtok")),
            )
        return cls(prices, data.get("as_of", ""), data.get("note", ""))

    def override(self, key: str, price: ModelPrice) -> None:
        """Set or replace one ``provider/model`` entry at runtime."""
        self._prices[key.lower()] = price

    def cheapest_for_provider(self, provider: str) -> tuple[str, ModelPrice] | None:
        """The provider's lowest-input-rate entry, as (key, price)."""
        prefix = provider.lower() + "/"
        candidates = [
            (key, price) for key, price in self._prices.items() if key.startswith(prefix)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[1].input_per_mtok)

    def lookup(self, provider: str | None, model: str | None) -> ModelPrice | None:
        if not model:
            return None
        candidates = []
        if provider:
            candidates.append(f"{provider}/{model}".lower())
        candidates.append(model.lower())
        for candidate in candidates:
            exact = self._prices.get(candidate)
            if exact is not None:
                return exact
        for candidate in candidates:
            best_key = None
            for key in self._prices:
                if not candidate.startswith(key + "-"):
                    continue
                if not _DATE_SUFFIX.match(candidate[len(key) :]):
                    continue
                if best_key is None or len(key) > len(best_key):
                    best_key = key
            if best_key is not None:
                return self._prices[best_key]
        return None

    def cost(
        self,
        provider: str | None,
        model: str | None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float | None:
        """Estimated USD cost of one call, or None for unknown models.

        Cached and cache-write tokens on a model with no listed cache price
        are billed at the full input rate, which overestimates slightly
        rather than underestimates.
        """
        price = self.lookup(provider, model)
        if price is None:
            return None
        cached_rate = (
            price.cached_input_per_mtok
            if price.cached_input_per_mtok is not None
            else price.input_per_mtok
        )
        write_rate = (
            price.cache_write_per_mtok
            if price.cache_write_per_mtok is not None
            else price.input_per_mtok
        )
        total = (
            input_tokens * price.input_per_mtok
            + cached_input_tokens * cached_rate
            + cache_write_tokens * write_rate
            + output_tokens * price.output_per_mtok
        )
        return total / 1_000_000

    def cost_for_attributes(self, attributes: dict[str, Any]) -> float | None:
        """Compute cost from llm span attributes (see schema.py conventions)."""
        model = attributes.get("gen_ai.response.model") or attributes.get(
            "gen_ai.request.model"
        )
        return self.cost(
            provider=attributes.get("gen_ai.system"),
            model=model,
            input_tokens=int(attributes.get("gen_ai.usage.input_tokens") or 0),
            output_tokens=int(attributes.get("gen_ai.usage.output_tokens") or 0),
            cached_input_tokens=int(attributes.get("cached_input_tokens") or 0),
            cache_write_tokens=int(attributes.get("cache_write_tokens") or 0),
        )


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)
