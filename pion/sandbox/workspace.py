"""Workspace-scoped file access policy for sandbox-aware tools."""

from __future__ import annotations

import fnmatch
import os
import stat
from pathlib import Path, PurePosixPath


class WorkspaceAccessError(PermissionError):
    """Raised when a tool path escapes the workspace or touches a protected path."""


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class WorkspaceGuard:
    """Resolve tool paths beneath one immutable workspace root.

    ``Path.resolve(strict=False)`` resolves every existing symlink component
    and normalizes ``..`` even when the final file does not exist. This keeps
    reads and writes from escaping through either absolute paths or symlinks.
    """

    def __init__(
        self,
        workspace: Path,
        protect_paths: list[str] | None = None,
        write_protect_paths: list[str] | None = None,
    ) -> None:
        self.workspace = workspace.expanduser().resolve()
        if not self.workspace.is_dir():
            raise WorkspaceAccessError(
                f"workspace is not an existing directory: {self.workspace}"
            )
        self.protect_paths = tuple(
            self._validate_pattern(pattern) for pattern in (protect_paths or [])
        )
        self.write_protect_paths = tuple(
            self._validate_pattern(pattern) for pattern in (write_protect_paths or [])
        )

    @staticmethod
    def _validate_pattern(pattern: str) -> str:
        normalized = pattern.strip().replace("\\", "/")
        if not normalized:
            raise WorkspaceAccessError("protected path patterns must not be empty")
        pure = PurePosixPath(normalized)
        if pure.is_absolute() or ".." in pure.parts:
            raise WorkspaceAccessError(
                f"protected path pattern must be workspace-relative: {pattern}"
            )
        return normalized

    def resolve(self, requested: str, operation: str = "access") -> Path:
        """Return a canonical in-workspace path or raise ``WorkspaceAccessError``."""
        raw = Path(requested).expanduser()
        candidate = raw if raw.is_absolute() else self.workspace / raw
        lexical = candidate.absolute()
        if not _is_relative_to(lexical, self.workspace):
            raise WorkspaceAccessError(
                f"{operation} denied: path is outside workspace {self.workspace}: "
                f"{requested}"
            )
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise WorkspaceAccessError(
                f"{operation} denied: could not resolve {requested!r}: {exc}"
            ) from exc

        if not _is_relative_to(resolved, self.workspace):
            raise WorkspaceAccessError(
                f"{operation} denied: path is outside workspace {self.workspace}: "
                f"{requested}"
            )

        requested_rel = lexical.relative_to(self.workspace).as_posix()
        resolved_rel = resolved.relative_to(self.workspace).as_posix()
        write_operation = operation.lower() in {
            "write",
            "edit",
            "delete",
            "create",
        }
        protected = self._is_protected(
            requested_rel, self.protect_paths
        ) or self._is_protected(resolved_rel, self.protect_paths)
        write_protected = write_operation and (
            self._is_protected(requested_rel, self.write_protect_paths)
            or self._is_protected(resolved_rel, self.write_protect_paths)
        )
        if protected or write_protected:
            raise WorkspaceAccessError(
                f"{operation} denied: protected workspace path: {requested}"
            )
        return resolved

    def open_file(
        self,
        requested: str,
        operation: str,
        flags: int,
        *,
        create_parents: bool = False,
        mode: int = 0o666,
    ) -> int:
        """Open a regular file beneath the workspace without symlink races.

        Resolving and then opening by absolute path leaves a TOCTOU window in
        which a container background process could replace an ancestor with a
        symlink. Walking from an already-open workspace directory descriptor
        with ``O_NOFOLLOW`` keeps every component beneath the original root.
        """
        resolved = self.resolve(requested, operation)
        relative = resolved.relative_to(self.workspace)
        if not relative.parts:
            raise IsADirectoryError(f"path is a directory, not a file: {requested}")
        if (
            not hasattr(os, "O_NOFOLLOW")
            or not hasattr(os, "O_DIRECTORY")
            or os.open not in os.supports_dir_fd
            or os.mkdir not in os.supports_dir_fd
        ):
            raise WorkspaceAccessError(
                f"{operation} denied: this platform cannot enforce race-safe "
                "workspace file access"
            )

        close_on_exec = getattr(os, "O_CLOEXEC", 0)
        no_follow = os.O_NOFOLLOW
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | no_follow | close_on_exec
        current_fd = os.open(self.workspace, directory_flags)
        try:
            for component in relative.parts[:-1]:
                try:
                    next_fd = os.open(
                        component,
                        directory_flags,
                        dir_fd=current_fd,
                    )
                except FileNotFoundError:
                    if not create_parents:
                        raise
                    try:
                        os.mkdir(component, mode=0o777, dir_fd=current_fd)
                    except FileExistsError:
                        # A concurrent creator is fine only if the no-follow
                        # open below confirms it created a real directory.
                        pass
                    next_fd = os.open(
                        component,
                        directory_flags,
                        dir_fd=current_fd,
                    )
                os.close(current_fd)
                current_fd = next_fd

            safe_flags = (
                flags | no_follow | close_on_exec | getattr(os, "O_NONBLOCK", 0)
            )
            file_fd = os.open(
                relative.parts[-1],
                safe_flags,
                mode,
                dir_fd=current_fd,
            )
            try:
                file_mode = os.fstat(file_fd).st_mode
            except OSError:
                os.close(file_fd)
                raise
            if not stat.S_ISREG(file_mode):
                os.close(file_fd)
                if stat.S_ISDIR(file_mode):
                    raise IsADirectoryError(
                        f"path is a directory, not a file: {requested}"
                    )
                raise WorkspaceAccessError(
                    f"{operation} denied: path is not a regular file: {requested}"
                )
            return file_fd
        finally:
            os.close(current_fd)

    @staticmethod
    def _is_protected(relative_path: str, patterns: tuple[str, ...]) -> bool:
        pure = PurePosixPath(relative_path)
        for pattern in patterns:
            if "/" in pattern:
                # Prefixing both values with "/" makes PurePath matching
                # workspace-rooted instead of suffix-based, and unlike
                # fnmatch it does not let "*" cross directory separators.
                if PurePosixPath(f"/{relative_path}").match(f"/{pattern}"):
                    return True
            elif any(fnmatch.fnmatchcase(part, pattern) for part in pure.parts):
                return True
        return False

    def protected_existing_paths(self) -> list[Path]:
        """Return existing protected files/directories without following escapes."""
        matches: set[Path] = set()
        for pattern in self.protect_paths:
            iterator = (
                self.workspace.glob(pattern)
                if "/" in pattern
                else self.workspace.rglob(pattern)
            )
            for path in iterator:
                try:
                    resolved = path.resolve(strict=False)
                except (OSError, RuntimeError):
                    continue
                if _is_relative_to(resolved, self.workspace):
                    matches.add(path.absolute())
        return sorted(matches, key=str)
