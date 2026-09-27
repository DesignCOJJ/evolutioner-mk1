"""Backend protocol shared by the real llama.cpp engine and test mocks."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseBackend(ABC):
    """Anything that can ``generate(prompt) -> (text, stats)``.

    stats = {"prompt_tokens": int, "gen_tokens": int, "duration_s": float}
    """

    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, *, max_tokens: int = 256,
                 temperature: float = 0.7, stop: Tuple[str, ...] = ()) -> Tuple[str, Dict[str, Any]]:
        ...

    def close(self) -> None:  # pragma: no cover - default no-op
        pass
