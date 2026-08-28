"""Message queue: steer (next turn) + follow-up (after all work).

Port of pi's message-queue semantics, reduced: Pion's agent loop cannot be
steered mid-turn, so queued messages are sent sequentially when the current
run finishes — steering messages first, then follow-ups.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MessageQueue:
    steer: list[str] = field(default_factory=list)
    followup: list[str] = field(default_factory=list)

    def enqueue_steer(self, text: str) -> None:
        self.steer.append(text)

    def enqueue_followup(self, text: str) -> None:
        self.followup.append(text)

    def pop_next(self) -> str | None:
        if self.steer:
            return self.steer.pop(0)
        if self.followup:
            return self.followup.pop(0)
        return None

    def pop_back(self) -> str | None:
        """Alt+Up: take the most recent queued message back to the editor.

        This mirrors ``pop_next``: since messages are sent steer-first then
        followup, the *last* one that would be sent lives at the tail of
        ``followup`` (or of ``steer`` when there are no follow-ups). Popping in
        that order returns the genuinely most-recently-queued message.
        """
        if self.followup:
            return self.followup.pop()
        if self.steer:
            return self.steer.pop()
        return None

    def __len__(self) -> int:
        return len(self.steer) + len(self.followup)

    def clear(self) -> None:
        self.steer.clear()
        self.followup.clear()
