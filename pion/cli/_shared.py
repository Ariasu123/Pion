"""Shared console objects for the pion CLI package."""

from rich.console import Console

console = Console()
err_console = Console(stderr=True)
