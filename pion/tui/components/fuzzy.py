"""Fuzzy matching (port of pi-tui's fuzzy.ts)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


def fuzzy_match(query: str, candidate: str) -> int | None:
    """Subsequence match score; higher is better, None when no match."""
    if not query:
        return 0
    query = query.lower()
    candidate_lower = candidate.lower()
    score = 0
    pos = -1
    consecutive = 0
    for ch in query:
        found = candidate_lower.find(ch, pos + 1)
        if found == -1:
            return None
        if found == pos + 1:
            consecutive += 1
            score += 10 + consecutive * 5
        else:
            consecutive = 0
            score += 1
        # Bonus at word starts.
        if found == 0 or candidate_lower[found - 1] in " /_-.":
            score += 8
        pos = found
    # Prefer shorter candidates and earlier first matches.
    score -= len(candidate) // 8
    score -= candidate_lower.find(query[0])
    return score


def fuzzy_filter(
    query: str, items: Iterable[T], key: Callable[[T], str]
) -> list[T]:
    scored = []
    for item in items:
        score = fuzzy_match(query, key(item))
        if score is not None:
            scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored]
