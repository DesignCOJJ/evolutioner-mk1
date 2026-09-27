"""Virtual Context Engine — spec pillar 4.

Rolling context window backed by lightweight hashed TF vectors (no external
embedding model needed) with JSONL persistence, simulating long-term
retrieval without exploding physical RAM limits.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DIM = 256


class VirtualContextEngine:
    def __init__(self, rag_dir: str = "./data/memory", top_k: int = 4) -> None:
        self.rag_dir = Path(rag_dir)
        self.rag_dir.mkdir(parents=True, exist_ok=True)
        self.store_path = self.rag_dir / "vectors.jsonl"
        self.top_k = max(1, top_k)
        self.entries: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------ #
    @staticmethod
    def tokenize(text: str) -> List[str]:
        return _TOKEN_RE.findall((text or "").lower())

    @classmethod
    def vectorize(cls, text: str) -> List[float]:
        vec = [0.0] * _DIM
        counts = Counter(cls.tokenize(text))
        for tok, count in counts.items():
            idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % _DIM
            vec[idx] += 1.0 + math.log(count)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def cosine(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            for line in self.store_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self.entries.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            self.entries = []

    def _persist(self, entry: Dict[str, Any]) -> None:
        with self.store_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def add(self, text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        entry = {
            "text": text,
            "meta": meta or {},
            "ts": time.time(),
            "vector": self.vectorize(text),
        }
        self.entries.append(entry)
        try:
            self._persist({"text": text, "meta": entry["meta"], "ts": entry["ts"],
                           "vector": entry["vector"]})
        except OSError:
            pass  # persistence is best-effort; memory copy still works
        return entry

    def search(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.entries:
            return []
        qv = self.vectorize(query)
        k = k or self.top_k
        scored = [
            {**{ "text": e["text"], "meta": e.get("meta", {}), "ts": e.get("ts", 0)},
             "score": round(self.cosine(qv, e["vector"]), 4)}
            for e in self.entries
        ]
        scored.sort(key=lambda e: e["score"], reverse=True)
        return scored[:k]

    def build_context(self, query: str, budget_chars: int = 1200) -> str:
        """Format top-k retrieved memories as a context block."""
        hits = [h for h in self.search(query) if h["score"] > 0.05]
        if not hits:
            return ""
        lines, used = [], 0
        for h in hits:
            block = f"- {h['text']}"
            if used + len(block) > budget_chars:
                break
            lines.append(block)
            used += len(block)
        return "\n".join(lines)
