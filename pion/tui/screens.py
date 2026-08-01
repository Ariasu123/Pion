"""Modal screens used by the Pion TUI."""

from __future__ import annotations

from dataclasses import dataclass

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList
from textual.widgets.option_list import Option


@dataclass(frozen=True)
class BranchDecision:
    summarize: bool
    instructions: str | None = None


class BranchScreen(ModalScreen[BranchDecision | None]):
    BINDINGS = [("escape", "cancel", "Cancel")]

    def action_cancel(self) -> None:
        self.dismiss(None)

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-box"):
            yield Label("Switch session branch", classes="modal-title")
            yield Label(
                "The current branch suffix will be left behind. Optionally summarize it."
            )
            yield Input(placeholder="Optional custom summary focus", id="branch-focus")
            with Horizontal(classes="modal-actions"):
                yield Button("No summary", id="branch-no")
                yield Button("Default summary", id="branch-default", variant="primary")
                yield Button("Custom summary", id="branch-custom")
                yield Button("Cancel", id="branch-cancel")

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "branch-no":
            self.dismiss(BranchDecision(False))
        elif event.button.id == "branch-default":
            self.dismiss(BranchDecision(True))
        elif event.button.id == "branch-custom":
            value = self.query_one("#branch-focus", Input).value.strip()
            if value:
                self.dismiss(BranchDecision(True, value))
            else:
                self.query_one("#branch-focus", Input).focus()
                self.notify("Enter custom summary focus first", severity="warning")
        else:
            self.dismiss(None)


class TextInputScreen(ModalScreen[str | None]):
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, title: str, value: str = "", placeholder: str = "") -> None:
        super().__init__()
        self.dialog_title = title
        self.initial_value = value
        self.placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-box compact-modal"):
            yield Label(self.dialog_title, classes="modal-title")
            yield Input(
                value=self.initial_value,
                placeholder=self.placeholder,
                id="text-value",
            )
            with Horizontal(classes="modal-actions"):
                yield Button("Save", id="text-save", variant="primary")
                yield Button("Clear", id="text-clear")
                yield Button("Cancel", id="text-cancel")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Input.Submitted)
    def input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "text-save":
            self.dismiss(self.query_one(Input).value)
        elif event.button.id == "text-clear":
            self.dismiss("")
        else:
            self.dismiss(None)


class ChoiceScreen(ModalScreen[str | None]):
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, title: str, choices: list[tuple[str, str]]) -> None:
        super().__init__()
        self.dialog_title = title
        self.choices = choices

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal-box choice-modal"):
            yield Label(self.dialog_title, classes="modal-title")
            yield OptionList(
                *(Option(label, id=value) for value, label in self.choices),
                id="choice-list",
            )
            yield Label("Enter select · Esc cancel", classes="modal-hint")

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    @on(OptionList.OptionSelected)
    def option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.id))

    def action_cancel(self) -> None:
        self.dismiss(None)
