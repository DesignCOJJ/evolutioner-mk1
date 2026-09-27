"""Persistent memory store: curated long-term notes as individual JSON files."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class MemoryStore:
    def __init__(self, memories_dir: str = "./data/memories") -> None:
        self.memories_dir = Path(memories_dir)
        self.memories_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def add(self, text: str, tags: Optional[List[str]] = None,
            source: str = "user") -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise ValueError("memory text must be non-empty")
        mid = hashlib.sha1(f"{time.time()}:{text}".encode()).hexdigest()[:12]
        record = {
            "id": mid,
            "ts": time.time(),
            "text": text,
            "tags": list(tags or []),
            "source": source,
        }
        path = self.memories_dir / f"{mid}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    def all(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for path in sorted(self.memories_dir.glob("*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        records.sort(key=lambda r: r.get("ts", 0))
        return records

    def get(self, mid: str) -> Optional[Dict[str, Any]]:
        path = self.memories_dir / f"{mid}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def delete(self, mid: str) -> bool:
        path = self.memories_dir / f"{mid}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def search(self, query: str, limit: int = 3,
               tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Cheap relevance: substring hits on words, tag filter, recency boost."""
        query_l = (query or "").lower()
        words = [w for w in query_l.split() if len(w) > 2]
        scored: List[tuple[float, Dict[str, Any]]] = []
        for rec in self.all():
            if tags and not set(tags) & set(rec.get("tags", [])):
                continue
            text_l = rec["text"].lower()
            score = 0.0
            for w in words:
                if w in text_l:
                    score += 1.0
            if query_l and query_l in text_l:
                score += 2.0
            if score > 0:
                age_days = max(0.0, (time.time() - rec.get("ts", 0)) / 86400)
                score += max(0.0, 0.5 - age_days * 0.01)  # recency boost
                scored.append((score, rec))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [rec for _, rec in scored[:limit]]
