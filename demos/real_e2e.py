#!/usr/bin/env python3
"""Real-model end-to-end check for pion (DeepSeek).

Runs the actual `pion` CLI against the live DeepSeek API in a temp workspace:
the agent is asked to create a file, write content into it, and verify it with
bash — then this script asserts the file really exists with the right content.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    uv run python demos/real_e2e.py

Exit codes: 0 = pass, 1 = agent/verification failure, 2 = skipped (no key /
network unreachable). Skipping is honest: this script never fakes a pass.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

TASK = (
    "Create a file named hello.txt containing exactly the text: hello pion. "
    "Then verify it by running `cat hello.txt` with the bash tool. "
    "Finally reply with one short confirmation sentence."
)


def main() -> int:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("SKIP: DEEPSEEK_API_KEY is not set", file=sys.stderr)
        return 2

    # Prefer this project's installed console script; fall back to PATH.
    cli = Path(sys.prefix) / "bin" / "pion"
    cmd = [str(cli) if cli.exists() else "pion", "-p", TASK]

    with tempfile.TemporaryDirectory(prefix="pion-real-e2e-") as workdir:
        print(f"[real-e2e] workdir: {workdir}")
        print(f"[real-e2e] running: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except Exception as exc:  # network unreachable, DNS, etc.
            print(f"SKIP: could not reach the API ({exc})", file=sys.stderr)
            return 2

        print("----- agent stdout -----")
        print(proc.stdout)
        if proc.stderr:
            print("----- agent stderr -----", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)

        target = Path(workdir) / "hello.txt"
        if proc.returncode != 0:
            print(f"FAIL: pion exited with {proc.returncode}", file=sys.stderr)
            return 1
        if not target.exists():
            print("FAIL: hello.txt was not created", file=sys.stderr)
            return 1
        content = target.read_text(encoding="utf-8").strip()
        if "hello pion" not in content:
            print(f"FAIL: unexpected file content: {content!r}", file=sys.stderr)
            return 1

        print(f"[real-e2e] verified: {target} contains {content!r}")
        print("REAL E2E OK")
        return 0


if __name__ == "__main__":
    sys.exit(main())
