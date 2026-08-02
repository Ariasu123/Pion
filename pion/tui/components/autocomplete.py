"""Editor autocomplete: slash commands + @file paths (port of pi-tui's
autocomplete.ts, reduced)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .fuzzy import fuzzy_filter
from .select_list import SelectItem

_IGNORED_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
}
_MAX_FILES = 3000


@dataclass(frozen=True)
class SlashCommand:
    name: str  # without the leading "/"
    description: str = ""
    source: str = ""  # e.g. "extension"


@dataclass
class Suggestion:
    replace_start: int  # index into the text where the token begins
    items: list[SelectItem] = field(default_factory=list)


class FileIndex:
    """Cached recursive file listing rooted at a directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._files: list[str] | None = None

    def files(self) -> list[str]:
        if self._files is None:
            found: list[str] = []
            for dirpath, dirnames, filenames in os.walk(self.root):
                dirnames[:] = [
                    d for d in dirnames if d not in _IGNORED_DIRS and not d.startswith(".")
                ]
                for name in filenames:
                    found.append(str(Path(dirpath, name).relative_to(self.root)))
                    if len(found) >= _MAX_FILES:
                        self._files = found
                        return found
            self._files = sorted(found)
        return self._files

    def invalidate(self) -> None:
        self._files = None


class CombinedAutocompleteProvider:
    """Suggests slash commands at line start and files after `@`."""

    def __init__(self, commands: list[SlashCommand], file_root: Path) -> None:
        self.commands = commands
        self.file_index = FileIndex(file_root)

    def set_commands(self, commands: list[SlashCommand]) -> None:
        self.commands = commands

    def suggest(self, text: str, cursor: int) -> Suggestion | None:
        head = text[:cursor]
        word_start = max(head.rfind(" "), head.rfind("\n")) + 1
        word = head[word_start:]
        if word.startswith("/") and word_start == 0 and " " not in word:
            prefix = word[1:]
            commands = (
                fuzzy_filter(prefix, self.commands, lambda c: c.name)
                if prefix
                else list(self.commands)
            )
            items = [
                SelectItem(
                    value="/" + c.name,
                    label="/" + c.name,
                    description=c.description,
                )
                for c in commands[:10]
            ]
            return Suggestion(word_start, items) if items else None
        if word.startswith("@") and len(word) >= 1:
            prefix = word[1:]
            files = (
                fuzzy_filter(prefix, self.file_index.files(), lambda f: f)
                if prefix
                else self.file_index.files()[:10]
            )
            items = [SelectItem(value="@" + f, label=f) for f in files[:10]]
            return Suggestion(word_start, items) if items else None
        return None
