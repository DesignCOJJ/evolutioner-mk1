"""Mock backend for tests and the built-in selftest (zero model downloads)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .backend import BaseBackend


class MockLLM(BaseBackend):
    """Scripted backend: returns queued responses in order, records calls."""

    name = "mock"

    def __init__(self, responses: List[str] | None = None, default: str = "ok") -> None:
        self.queue: List[str] = list(responses or [])
        self.default = default
        self.calls: List[Dict[str, Any]] = []

    def generate(self, prompt: str, *, max_tokens: int = 256,
                 temperature: float = 0.7, stop: Tuple[str, ...] = ()) -> Tuple[str, Dict[str, Any]]:
        text = self.queue.pop(0) if self.queue else self.default
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "stop": stop})
        return text, {
            "prompt_tokens": max(1, len(prompt) // 4),
            "gen_tokens": max(1, len(text) // 4),
            "duration_s": 0.001,
        }
