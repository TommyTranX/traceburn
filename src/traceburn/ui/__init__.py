"""The local web viewer: a read-only JSON API plus a vendored SPA.

Requires the [ui] extra (starlette, uvicorn). Nothing here is imported by
the package root, so the core stays dependency-free.
"""
