"""Thin client to the semhood daemon, with auto-start.

Both the CLI and the MCP server talk to the daemon through this module. The
contract is simple: call :func:`request` with an endpoint and payload; if the
daemon isn't running, it is spawned (detached) and we wait for it to become
healthy before issuing the call. Callers never touch the model or the index
directly — that's the daemon's job, and the reason repeated searches are fast.

Uses only the standard library (urllib) so importing the client never drags in
heavy dependencies; the daemon side owns those.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from semhood.paths import daemon_info_path, semhood_home

logger = logging.getLogger("semhood.client")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = int(os.environ.get("SEMHOOD_DAEMON_PORT", "7711"))

# First start pays the model-load cost (~15s for the local embedder); be patient.
START_TIMEOUT_S = float(os.environ.get("SEMHOOD_DAEMON_START_TIMEOUT", "90"))


class DaemonError(RuntimeError):
    """Raised when the daemon returns an error or can't be reached/started."""


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _read_info() -> dict | None:
    path = daemon_info_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _base_url(info: dict) -> str:
    return f"http://{info.get('host', DEFAULT_HOST)}:{info.get('port', DEFAULT_PORT)}"


def _http(url: str, payload: dict | None, timeout: float) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


def _is_healthy(info: dict, timeout: float = 1.5) -> bool:
    try:
        out = _http(_base_url(info) + "/health", None, timeout)
        return bool(out.get("ok"))
    except (urllib.error.URLError, OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def _spawn_daemon() -> None:
    """Launch the daemon as a detached background process."""
    log_path = semhood_home() / "daemon.log"
    log_file = open(log_path, "a", encoding="utf-8")
    cmd = [sys.executable, "-m", "semhood.daemon"]

    kwargs: dict = {"stdout": log_file, "stderr": log_file, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP — survive the parent exiting.
        kwargs["creationflags"] = 0x00000008 | 0x00000200
    else:
        kwargs["start_new_session"] = True

    subprocess.Popen(cmd, **kwargs)
    logger.info("starting semhood daemon (logs: %s)", log_path)


def ensure_daemon(*, autostart: bool = True) -> dict:
    """Return connection info for a healthy daemon, starting one if needed."""
    info = _read_info()
    if info and _is_healthy(info):
        return info

    if not autostart:
        raise DaemonError(
            "semhood daemon is not running. Start it with `semhood serve`."
        )

    _spawn_daemon()

    deadline = time.monotonic() + START_TIMEOUT_S
    while time.monotonic() < deadline:
        time.sleep(0.4)
        info = _read_info()
        if info and _is_healthy(info):
            return info
    raise DaemonError(
        f"daemon did not become healthy within {START_TIMEOUT_S:.0f}s — "
        f"see {semhood_home() / 'daemon.log'}"
    )


def is_running() -> bool:
    info = _read_info()
    return bool(info and _is_healthy(info))


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------


def request(endpoint: str, payload: dict | None = None, *,
            timeout: float = 600.0, autostart: bool = True) -> dict:
    """POST ``payload`` to the daemon ``endpoint`` and return the JSON result.

    Auto-starts the daemon on first use. Raises :class:`DaemonError` with the
    server's message on a 4xx/5xx so callers (CLI, MCP) can surface it cleanly.
    """
    info = ensure_daemon(autostart=autostart)
    url = _base_url(info) + "/" + endpoint.lstrip("/")
    try:
        # payload is None → GET (health, projects); a dict → POST.
        return _http(url, payload, timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except ValueError:
            pass
        raise DaemonError(f"{endpoint} failed: {detail}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise DaemonError(f"could not reach daemon for {endpoint}: {exc}") from exc


def current_root(start: str | Path | None = None) -> str:
    """The project root for the caller's location — what requests are keyed on."""
    from semhood.paths import find_project_root

    return str(find_project_root(start))
