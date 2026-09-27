"""Unified engine: model download + llama.cpp GGUF inference (spec section 1)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple

from .backend import BaseBackend
from .config import Config, VARIANTS


class ModelDownloadError(RuntimeError):
    pass


class EvolutionerEngine(BaseBackend):
    """Real GGUF backend backed by llama-cpp-python.

    Mirrors the spec's ``EvolutionerEngine``: resolves the official
    wallpillar-lm variant, downloads from Hugging Face on first use, then
    initializes the llama.cpp backend (AVX2-friendly, no CUDA required).
    Import is deferred so tests / --list-models never require the package.
    """

    name = "llama.cpp"

    def __init__(self, config: Config) -> None:
        self.config = config
        try:
            from llama_cpp import Llama  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "llama-cpp-python is not installed. Run: bash setup.sh"
            ) from exc
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        spec = VARIANTS[config.variant]
        model_dir = Path(config.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = model_dir / spec.filename

        if not self.model_path.exists():
            print(f"[Engine] Downloading {config.variant} weights from HF: {spec.repo_id} ...")
            try:
                hf_hub_download(
                    repo_id=spec.repo_id,
                    filename=spec.filename,
                    local_dir=str(model_dir),
                )
            except Exception as exc:
                raise ModelDownloadError(
                    f"Could not download {spec.repo_id}/{spec.filename}: {exc}"
                ) from exc

        print(f"[Engine] Initializing llama.cpp backend for {config.variant} ...")
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=config.n_ctx,
            n_threads=config.n_threads,
            use_mmap=config.use_mmap,
            verbose=config.verbose,
        )

    def generate(self, prompt: str, *, max_tokens: int = 256,
                 temperature: float = 0.7, stop: Tuple[str, ...] = ()) -> Tuple[str, Dict[str, Any]]:
        t0 = time.perf_counter()
        out = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=list(stop) if stop else None,
            echo=False,
        )
        duration = time.perf_counter() - t0
        text = out["choices"][0]["text"]
        stats: Dict[str, Any] = {
            "prompt_tokens": int(out["usage"]["prompt_tokens"]),
            "gen_tokens": int(out["usage"]["completion_tokens"]),
            "duration_s": round(duration, 4),
        }
        return text, stats

    def close(self) -> None:
        self.llm = None
