"""Dim two-line footer (port of pi-coding-agent's footer.ts)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ...session import estimate_tokens
from ..core.component import Component
from ..core.text_utils import truncate_to_width, visible_width
from ..theme import Theme, get_theme


def _fmt_tokens(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1000:
        return f"{count / 1000:.1f}k"
    return str(count)


def _git_branch(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    branch = result.stdout.strip()
    return branch if result.returncode == 0 and branch else None


def _context_tokens(controller) -> int:  # noqa: ANN001
    """Tokens the next request would send, from real usage when available.

    Both LLM adapters normalize usage so that
    input + cache_read + cache_write equals the total prompt size; fall
    back to the chars//4 estimate before the first turn completes.
    """
    usage = controller.last_usage
    if usage is not None:
        return usage.input + usage.cache_read + usage.cache_write
    return estimate_tokens(controller.agent.messages)


class Footer(Component):
    def __init__(
        self,
        controller,  # AgentSessionController
        session_name: str,
        queued=None,  # Callable[[], int] | None
        theme: Theme | None = None,
    ) -> None:
        self.controller = controller
        self.session_name = session_name
        self.queued = queued or (lambda: 0)
        self._theme = theme
        try:
            cwd = Path.cwd()
            self._cwd_display = str(cwd).replace(str(Path.home()), "~", 1)
        except OSError:
            self._cwd_display = "?"
        self._branch = _git_branch(Path.cwd())

    def render(self, width: int) -> list[str]:
        theme = self._theme or get_theme()
        agent = self.controller.agent

        location = self._cwd_display
        if self._branch:
            location += f" ({self._branch})"
        location += f" • {self.session_name}"
        line1 = theme.fg("dim", truncate_to_width(location, width - 1))

        usage = self.controller.last_usage
        stats = ""
        if usage is not None:
            cost = usage.cost.total or agent.model.compute_cost(usage).total
            stats = (
                f"↑{_fmt_tokens(usage.input)} ↓{_fmt_tokens(usage.output)}"
                f" R{_fmt_tokens(usage.cache_read)}"
                f" W{_fmt_tokens(usage.cache_write)}"
                f" ${cost:.3f}"
            )
        tokens = _context_tokens(self.controller)
        window = agent.model.context_window
        percent = min(100, round(tokens / max(1, window) * 100))
        ctx_token = "dim"
        if percent > 90:
            ctx_token = "error"
        elif percent > 70:
            ctx_token = "warning"
        ctx = f"{percent}%/{_fmt_tokens(window)}"
        left = theme.fg("dim", stats) if stats else ""
        left += (" " if left else "") + theme.fg(ctx_token, ctx)
        queued = self.queued()
        if queued:
            left += "  " + theme.fg("warning", f"⇢{queued} queued")

        right = theme.fg("dim", agent.model.id)
        gap = width - visible_width(left) - visible_width(right)
        if gap < 2:
            left = truncate_to_width(left, max(1, width - visible_width(right) - 2))
            gap = width - visible_width(left) - visible_width(right)
        line2 = left + " " * max(0, gap) + right

        return ["", line1, line2]
