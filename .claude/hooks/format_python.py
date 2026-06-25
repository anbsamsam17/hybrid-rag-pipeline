#!/usr/bin/env python
"""PostToolUse hook: auto-format the Python file that was just edited/written.

Contract (Claude Code hooks):
  - Reads the hook event JSON from stdin.
  - For an Edit/Write of a .py file, runs `ruff check --fix` then `black` on
    just that one file so the diff stays lint-clean and the adversarial-reviewer
    reviews real logic rather than formatting noise.
  - Degrades gracefully: if ruff/black are not installed (deps not present at
    scaffold time) or anything goes wrong, it exits 0 silently. A PostToolUse
    hook never needs to block here.

Windows-safe: invoked as `python .../format_python.py`; uses `python -m ruff`
and `python -m black` (no reliance on console scripts being on PATH), and falls
back to the bare `ruff`/`black` executables if the module form is unavailable.
No bash-only syntax.

Exit codes:
  0  -> normal; nothing to report (the only exit this hook ever uses).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _read_event() -> dict:
    try:
        raw = sys.stdin.read()
    except Exception:
        return {}
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _extract_file_path(event: dict) -> str | None:
    tool_input = event.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    fp = tool_input.get("file_path")
    if isinstance(fp, str) and fp.strip():
        return fp
    return None


def _build_command(tool: str, target: str) -> list[str] | None:
    """Prefer `python -m <tool>` (robust on Windows); fall back to console script."""
    if tool == "ruff":
        module_args = ["check", "--fix", target]
    elif tool == "black":
        module_args = [target]
    else:
        return None

    # 1) python -m <tool> -- works whenever the tool is importable, even if its
    #    console script isn't on PATH (common on Windows).
    if _module_available(tool):
        return [sys.executable, "-m", tool, *module_args]

    # 2) bare executable on PATH (e.g. ruff.exe / black.exe).
    exe = shutil.which(tool)
    if exe:
        return [exe, *module_args]

    return None


def _module_available(module: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def _run(cmd: list[str], cwd: str | None) -> None:
    try:
        subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=120,
        )
    except Exception:
        # Tool missing / crashed / timed out -> degrade silently.
        pass


def main() -> int:
    event = _read_event()
    file_path = _extract_file_path(event)
    if not file_path:
        return 0

    try:
        path = Path(file_path)
    except Exception:
        return 0

    if path.suffix.lower() != ".py":
        return 0
    if not path.exists():
        return 0

    target = str(path)
    cwd = event.get("cwd")
    cwd = cwd if isinstance(cwd, str) and cwd else None

    # ruff first (autofix lint), then black (canonical formatting).
    for tool in ("ruff", "black"):
        cmd = _build_command(tool, target)
        if cmd is not None:
            _run(cmd, cwd)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # A formatting hook must never break the session.
        sys.exit(0)
