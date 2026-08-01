"""Pion's fixed dark terminal theme."""

from textual.theme import Theme

PION_DARK = Theme(
    name="pion-dark",
    primary="#D97757",
    secondary="#8B93A3",
    warning="#D7A85B",
    error="#E06C75",
    success="#68B984",
    accent="#D97757",
    foreground="#E7E9EE",
    background="#0D0F12",
    surface="#12151A",
    panel="#171A20",
    boost="#20242C",
    dark=True,
    variables={
        "pion-bg": "#0D0F12",
        "pion-surface": "#12151A",
        "pion-panel": "#171A20",
        "pion-hover": "#20242C",
        "pion-border": "#2A2F39",
        "pion-text": "#E7E9EE",
        "pion-muted": "#8B93A3",
        "pion-faint": "#5E6675",
        "pion-accent": "#D97757",
        "pion-accent-dim": "#3A241D",
        "pion-success": "#68B984",
        "pion-warning": "#D7A85B",
        "pion-error": "#E06C75",
    },
)


__all__ = ["PION_DARK"]
