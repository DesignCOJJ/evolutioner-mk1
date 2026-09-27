"""Configuration and the official wallpillar-lm variant registry."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class VariantSpec:
    """One entry of the official model portfolio."""

    repo_id: str
    filename: str
    params: str
    purpose: str


HF_ORG = "wallpillar-lm"

VARIANTS: Dict[str, VariantSpec] = {
    "1.5B": VariantSpec(
        repo_id=f"{HF_ORG}/evolutionermk1-1.5B",
        filename="evolutionermk1-1.5b-q4_k_m.gguf",
        params="1.5B",
        purpose="Ultra-fast rollouts for high-depth MCTS exploration and rapid verification.",
    ),
    "2B": VariantSpec(
        repo_id=f"{HF_ORG}/evolutionermk1-2B",
        filename="evolutionermk1-2b-q4_k_m.gguf",
        params="2.0B",
        purpose="Instruction & structured tasks, schema enforcement, JSON outputs.",
    ),
    "3B": VariantSpec(
        repo_id=f"{HF_ORG}/evolutionermk1-3B",
        filename="evolutionermk1-3b-q4_k_m.gguf",
        params="3.0B",
        purpose="Primary local workhorse: SymPy execution loops, Python AST checking.",
    ),
    "5B": VariantSpec(
        repo_id=f"{HF_ORG}/evolutionermk1-5B",
        filename="evolutionermk1-5b-q4_k_m.gguf",
        params="~5-7B",
        purpose="High-capacity deep multi-step reasoning and complex code synthesis.",
    ),
}


@dataclass
class Config:
    """Runtime configuration, loaded from JSON with spec defaults."""

    variant: str = "3B"
    model_dir: str = "./models"
    n_ctx: int = 2048
    n_threads: int = 4
    use_mmap: bool = True
    verbose: bool = False
    # MCTS
    mcts_iterations: int = 8
    mcts_depth: int = 4
    c_param: float = 1.414
    candidates: int = 3
    # Virtual context / persistent memory
    enable_rag: bool = True
    rag_top_k: int = 4
    rag_dir: str = "./data/memory"
    memories_dir: str = "./data/memories"
    session_dir: str = "./data/sessions"
    # Tracking / routing
    usage_db: str = "./data/usage.db"
    router_fallback: Optional[str] = None  # e.g. "api:OPENROUTER_API_KEY"
    # Generation
    gen_tokens: int = 320
    max_rounds: int = 3
    temperature: float = 0.7
    auto_memory: bool = False
    # Sandbox
    sandbox_timeout_s: float = 5.0
    use_bwrap: Optional[bool] = None  # None = auto-probe
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(
                f"Unknown variant: {self.variant!r}. Supported: {sorted(VARIANTS)}"
            )

    @classmethod
    def load(cls, path: Optional[str] = None, **overrides: Any) -> "Config":
        data: Dict[str, Any] = {}
        if path:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            known = {f.name for f in fields(cls)}
            data = {k: v for k, v in raw.items() if k in known}
            extra = {k: v for k, v in raw.items() if k not in known and k != "extra"}
            if extra:
                data.setdefault("extra", {}).update(extra)
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)
