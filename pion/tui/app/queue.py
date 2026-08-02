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
        """Alt+Up: take the most recent queued message back to the editor."""
        if self.steer:
            return self.steer.pop()
        if self.followup:
            return self.followup.pop()
        return None

    def __len__(self) -> int:
        return len(self.steer) + len(self.followup)

    def clear(self) -> None:
        self.steer.clear()
        self.followup.clear()
