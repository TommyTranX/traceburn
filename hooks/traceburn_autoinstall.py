"""Zero-code activation: ``TRACEBURN=1 python app.py`` instruments the run.

Installed at site-packages root alongside a .pth file that imports this
module at interpreter startup. With TRACEBURN unset this is a single env
check and nothing else; it can never raise into the host program.
"""

import os


def _activate() -> None:
    if os.environ.get("TRACEBURN") != "1":
        return
    try:
        import traceburn

        traceburn.install()
    except Exception:
        pass


_activate()
