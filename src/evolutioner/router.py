"""API fallback router — spec: cloud GPU engine as an escape hatch.

If the local backend fails (OOM, missing model, inference error) and a
fallback like ``api:OPENROUTER_API_KEY`` is configured, the request is
re-routed to an OpenAI-compatible chat completions endpoint. Token usage is
estimated and reported through the same stats dict as local calls.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Tuple

import requests

from .backend import BaseBackend


class RouterError(RuntimeError):
    pass


class FallbackRouter(BaseBackend):
    name = "router"

    def __init__(self, primary: BaseBackend, fallback: str | None = None,
                 api_base: str = "https://openrouter.ai/api/v1") -> None:
        self.primary = primary
        self.fallback = fallback
        self.api_base = api_base

    def generate(self, prompt: str, *, max_tokens: int = 256,
                 temperature: float = 0.7, stop: Tuple[str, ...] = ()) -> Tuple[str, Dict[str, Any]]:
        try:
            return self.primary.generate(
                prompt, max_tokens=max_tokens, temperature=temperature, stop=stop
            )
        except Exception as primary_exc:
            if not self.fallback or not self.fallback.startswith("api:"):
                raise
            env_key = self.fallback.split(":", 1)[1]
            api_key = os.environ.get(env_key, "")
            if not api_key:
                raise RouterError(
                    f"primary backend failed ({primary_exc}) and fallback env "
                    f"var {env_key} is not set"
                ) from primary_exc
            return self._call_api(prompt, max_tokens, temperature, api_key)

    def _call_api(self, prompt: str, max_tokens: int, temperature: float,
                  api_key: str) -> Tuple[str, Dict[str, Any]]:
        t0 = time.perf_counter()
        resp = requests.post(
            f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": os.environ.get("EVOLUTIONER_API_MODEL", "openai/gpt-4o-mini"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=60,
        )
        duration = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        stats: Dict[str, Any] = {
            "prompt_tokens": int(usage.get("prompt_tokens", len(prompt) // 4)),
            "gen_tokens": int(usage.get("completion_tokens", len(text) // 4)),
            "duration_s": round(duration, 4),
            "routed": True,
        }
        return text, stats
