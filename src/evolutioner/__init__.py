"""Evolutioner MK1 — reasoning harness, RLVR verification and MCTS TTC scaling."""

from .config import Config, VariantSpec, VARIANTS
from .engine import EvolutionerEngine
from .verifier import SandboxVerifier, VerificationResult
from .search import EvolutionerMCTS, MCTSResult
from .context import VirtualContextEngine
from .memory import MemoryStore
from .router import FallbackRouter
from .usage import UsageTracker
from .mock import MockLLM
from .ollama_backend import OllamaBackend, TieredOllamaBackend, PORTFOLIO_TAGS
from .harness import Harness, HarnessResult

__all__ = [
    "Config",
    "VariantSpec",
    "VARIANTS",
    "EvolutionerEngine",
    "SandboxVerifier",
    "VerificationResult",
    "EvolutionerMCTS",
    "MCTSResult",
    "VirtualContextEngine",
    "MemoryStore",
    "FallbackRouter",
    "UsageTracker",
    "MockLLM",
    "OllamaBackend",
    "TieredOllamaBackend",
    "PORTFOLIO_TAGS",
    "Harness",
    "HarnessResult",
]

__version__ = "1.0.0"
