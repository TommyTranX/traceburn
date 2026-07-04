"""Auto-instrumentation: detect installed clients and patch them.

``traceburn.install()`` is the zero-config entry point. It looks for the
openai and anthropic packages and wraps their call sites so every LLM call
becomes an ``llm`` span. Idempotent: calling it twice patches nothing new.
``traceburn.uninstall()`` restores the original methods.
"""

from __future__ import annotations

import logging

from . import anthropic as anthropic_patcher
from . import openai as openai_patcher

logger = logging.getLogger("traceburn")

_PATCHERS = {
    "openai": openai_patcher,
    "anthropic": anthropic_patcher,
}


def install() -> list[str]:
    """Patch every supported client that is importable.

    Returns the names of the clients newly patched by this call. A patcher
    failure is logged and skipped; it never propagates.
    """
    patched = []
    for name, patcher in _PATCHERS.items():
        try:
            if patcher.is_available() and patcher.patch():
                patched.append(name)
        except Exception:
            logger.warning("failed to instrument %s", name, exc_info=True)
    return patched


def uninstall() -> None:
    """Restore the original client methods."""
    for name, patcher in _PATCHERS.items():
        try:
            if patcher.is_available():
                patcher.unpatch()
        except Exception:
            logger.warning("failed to uninstrument %s", name, exc_info=True)
