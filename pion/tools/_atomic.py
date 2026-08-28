"""Atomic file write shared by the write/edit tools.

Writing in place (truncate then write) loses the original file if the
process dies or the disk fills mid-write. Write to a sibling temp file,
fsync, then os.replace, so a reader only ever sees the complete old or
new content — mirrors `config.save_config`.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Overwrite `path` with `text` atomically (UTF-8)."""
    data = text.encode("utf-8")
    parent = path.parent
    try:
        # Preserve the existing file's permission bits on overwrite; a fresh
        # file falls back to a sensible default (mkstemp would leave 0o600).
        mode = os.stat(path).st_mode & 0o777
    except OSError:
        mode = 0o644
    fd, tmp_name = tempfile.mkstemp(dir=parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
