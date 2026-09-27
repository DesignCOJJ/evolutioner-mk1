"""Evolutioner MK1 TUI theme — cyberpunk palette + unicode gauge primitives."""

from __future__ import annotations

from typing import List

# ---- neon palette (Rich color names) -------------------------------------- #
CYAN = "bright_cyan"
MAGENTA = "magenta"
PURPLE = "dark_violet"
GREEN = "bright_green"
AMBER = "bright_yellow"
RED = "bright_red"
GRAY = "grey58"
DIM = "grey35"
FG = "grey78"

FULL = "█"
SEVEN = "▓"
THREE = "▒"
LIGHT = "░"

WIDTHS = 10


def bar(frac: float, width: int = WIDTHS, fill: str = FULL, empty: str = LIGHT,
        color: str = CYAN) -> str:
    """Unicode block gauge: [████████░░] 82%"""
    frac = max(0.0, min(1.0, frac))
    filled = round(frac * width)
    return f"[{color}]{fill * filled}[/{color}][{DIM}]{empty * (width - filled)}[/{DIM}]"


def pct(frac: float) -> str:
    return f"{max(0.0, min(1.0, frac)) * 100:5.1f}%"


def kv(key: str, value: str, key_color: str = GRAY, width: int = 14) -> str:
    return f"[{key_color}]{key:<{width}}[/{key_color}][{FG}]{value}[/{FG}]"


def badge(label: str, value: str, color: str) -> str:
    return f"[{color}]● {label}: {value}[/{color}]"


BANNER_LINES: List[str] = [
    r" ▗▄▄▖ ▗▄▄▖  ▄▄▄  ▄▄▄▖ ▗▄▖ ▗▄▄▖  ▗▄▄▖  ▄▄▄  ▄▄▄▖ ▗▄▖ ",
    r"▐▛   ▘▐▛ ▜▌█   █ █▄▄▘█▛▜▌▐▛ ▜▌▐▛   ▘█   █ █▄▄▘█▛▜▌",
    r"▐▙▄▄▖▐▛ ▜▌▐▌  █ █▄▄▖█▌ ▐▌▐▙▄▟▌▐▙▄▄▖▐▌  ▐▌█▄▄▖█▌ ▐▌",
    r" ▝▀▀▘▝▘  ▘ ▀▀▀  ▝▀▀ ▝▘ ▝▘ ▝▀▀▘ ▝▀▀▘ ▀▀▀  ▝▀▀ ▝▘ ▝▘",
    r"   TEST-TIME-COMPUTE REASONING HARNESS // RLVR // MCTS-7   ",
]


def render_banner() -> str:
    l1 = "\n".join(f"[{CYAN}]{line}[/{CYAN}]" for line in BANNER_LINES[:4])
    tag = f"[{MAGENTA}]{BANNER_LINES[4]}[/{MAGENTA}]"
    return f"{l1}\n{tag}"
