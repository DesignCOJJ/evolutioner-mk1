"""Ollama backend — runs the official wallpillar-lm portfolio through Ollama.

Uses the native ``/api/generate`` endpoint (non-streaming) and maps Ollama's
usage counters into the standard stats dict so the UsageTracker records real
token counts (prompt_eval_count / eval_count) for every call.

Also provides ``TieredOllamaBackend`` implementing the spec's portfolio
routing: the 1.5B variant serves ultra-fast MCTS rollouts while the selected
variant (3B default) handles deep reasoning / final answers.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .backend import BaseBackend

DEFAULT_HOST = "http://localhost:11434"

# spec portfolio -> ollama model tag
PORTFOLIO_TAGS: Dict[str, str] = {
    "1.5B": "wallpillar-lm/evolutionermk1-1.5B:latest",
    "2B": "wallpillar-lm/evolutionermk1-2B:latest",
    "3B": "wallpillar-lm/evolutionermk1-3B:latest",
    "5B": "wallpillar-lm/evolutionermk1-5B:latest",
}


class OllamaError(RuntimeError):
    pass


_RESOLVE_CACHE: Dict[Tuple[str, str], str] = {}


def resolve_model_tag(requested: str, host: Optional[str] = DEFAULT_HOST) -> str:
    """Map a requested portfolio tag to a model that is actually installed.

    Tolerates local renames (e.g. ``evolutionermk1-3B`` -> ``evolu-general-3B``)
    by matching on size token (1.5B/2B/3B/5B) plus keyword (coder/general/
    uncens/integrated). Returns ``requested`` unchanged when nothing matches
    or the server is unreachable, so callers fail with their own error.
    """
    host = (host or DEFAULT_HOST).rstrip("/")
    cache_key = (host, requested)
    if cache_key in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[cache_key]
    resolved = requested
    try:
        installed = list_installed(host)
    except Exception:
        installed = []
    if requested in installed:
        pass
    elif installed:
        base = requested.split(":")[0].lower()
        size_m = re.search(r"(1\.5b|2b|3b|5b)$", base)
        size = size_m.group(1) if size_m else None
        kw = next((t for t in ("coder", "general", "uncens", "integrated") if t in base), None)
        best, best_score = None, -1
        for name in installed:
            nl = name.split(":")[0].lower()
            token = nl.rsplit("-", 1)[-1]  # e.g. "general-5b" -> "5b"
            if size:
                if token == size:
                    score = 3  # exact variant token ("1.5b" never matches "5b")
                elif nl.endswith(size) and not (size == "5b" and "1.5b" in nl):
                    score = 1  # fuzzy suffix fallback
                else:
                    continue
            else:
                if token != base and not nl.endswith(base):
                    continue
                score = 0
            if kw:
                score += 2 if kw in nl else -4
            elif any(t in nl for t in ("coder", "uncens", "integrated")):
                score -= 1  # plain variant: prefer un-suffixed/general builds
            if score > best_score:
                best, best_score = name, score
        if best:
            resolved = best
    _RESOLVE_CACHE[cache_key] = resolved
    return resolved


def ollama_ok(host: str = DEFAULT_HOST, timeout: float = 2.0, retries: int = 2) -> bool:
    host = (host or DEFAULT_HOST).rstrip("/")
    if not host.startswith("http"):
        host = f"http://{host}"
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(f"{host}/api/version", timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            if attempt < retries:
                time.sleep(1.5)  # server may still be binding
    return False


def list_installed(host: Optional[str] = DEFAULT_HOST) -> List[str]:
    host = (host or DEFAULT_HOST).rstrip("/")
    if not host.startswith("http"):
        host = f"http://{host}"
    req = urllib.request.Request(f"{host}/api/tags")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m.get("name", "") for m in data.get("models", [])]


class OllamaBackend(BaseBackend):
    """Single-model Ollama backend (native /api/generate)."""

    def __init__(self, model: str, host: str = DEFAULT_HOST,
                 host_env: str = "OLLAMA_HOST") -> None:
        import os

        self.model = model
        self.host = (host or os.environ.get(host_env) or DEFAULT_HOST).rstrip("/")
        if not self.host.startswith("http"):
            self.host = f"http://{self.host}"
        self.model = resolve_model_tag(model, self.host)
        self.name = f"ollama:{self.model}"

    def generate(self, prompt: str, *, max_tokens: int = 256,
                 temperature: float = 0.7, stop: Tuple[str, ...] = ()) -> Tuple[str, Dict[str, Any]]:
        # /api/chat (non-raw) so Ollama applies each model's chat template —
        # correct for Gemma/Qwen/DeepSeek instruct fine-tunes.
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                **({"stop": list(stop)} if stop else {}),
            },
        }
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                f"{self.host}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"cannot reach Ollama at {self.host} ({exc}); is `ollama serve` running?"
            ) from exc
        duration = time.perf_counter() - t0

        if data.get("error"):
            raise OllamaError(f"ollama error: {data['error']}")
        text = (data.get("message") or {}).get("content", data.get("response", ""))
        stats: Dict[str, Any] = {
            "prompt_tokens": int(data.get("prompt_eval_count", len(prompt) // 4)),
            "gen_tokens": int(data.get("eval_count", len(text) // 4)),
            "duration_s": round(duration, 4),
            "eval_duration_s": round(data.get("eval_duration", 0) / 1e9, 4),
        }
        return text, stats


class TieredOllamaBackend(BaseBackend):
    """Spec portfolio routing: 1.5B rollouts + chosen main reasoning model.

    The MCTS harness calls ``generate_rollout`` for cheap candidate steps and
    ``generate`` for everything else; plain ``generate`` (base API) routes to
    the main model so any other caller stays correct.
    """

    name = "ollama:tiered"

    def __init__(self, main_variant: str = "3B", rollout_variant: Optional[str] = "1.5B",
                 host: str = DEFAULT_HOST) -> None:
        if main_variant not in PORTFOLIO_TAGS:
            raise ValueError(f"unknown variant {main_variant!r}; pick from {sorted(PORTFOLIO_TAGS)}")
        self.main = OllamaBackend(PORTFOLIO_TAGS[main_variant], host=host)
        self.rollout = (
            OllamaBackend(PORTFOLIO_TAGS[rollout_variant], host=host)
            if rollout_variant and rollout_variant != main_variant else None
        )
        self.name = f"ollama:tiered({main_variant}+{rollout_variant or '-'})"

    def generate_rollout(self, prompt: str, *, max_tokens: int = 64,
                         temperature: float = 0.7, stop: Tuple[str, ...] = ()) -> Tuple[str, Dict[str, Any]]:
        backend = self.rollout or self.main
        return backend.generate(prompt, max_tokens=max_tokens,
                                temperature=temperature, stop=stop)

    def generate(self, prompt: str, *, max_tokens: int = 256,
                 temperature: float = 0.7, stop: Tuple[str, ...] = ()) -> Tuple[str, Dict[str, Any]]:
        return self.main.generate(prompt, max_tokens=max_tokens,
                                  temperature=temperature, stop=stop)


def build_ollama_backend(config, mock: bool = False) -> BaseBackend:
    """Factory used by the CLI: plain single-model or tiered portfolio backend."""
    if mock:
        from .mock import MockLLM

        return MockLLM(default="Final Answer: 4")
    tiered = config.extra.get("ollama_tiered", True)
    host = config.extra.get("ollama_host", DEFAULT_HOST)
    if tiered:
        return TieredOllamaBackend(
            main_variant=config.variant,
            rollout_variant=config.extra.get("ollama_rollout_variant", "1.5B"),
            host=host,
        )
    return OllamaBackend(PORTFOLIO_TAGS[config.variant], host=host)
