"""Usage tracker — SQLite-backed token/call accounting for every engine call.

Standard library only (sqlite3). Every generation call (MCTS rollouts,
reasoning rounds, final answers, fallback API calls) records prompt tokens,
generated tokens, wall time and derived tok/s statistics.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class UsageTracker:
    def __init__(self, db_path: str = "./data/usage.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    kind TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    gen_tokens INTEGER NOT NULL,
                    duration_s REAL NOT NULL,
                    session_id TEXT,
                    meta TEXT
                )
                """
            )
            self._conn.commit()

    def record(
        self,
        kind: str,
        model: str,
        prompt_tokens: int,
        gen_tokens: int,
        duration_s: float,
        session_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage_events (ts, kind, model, prompt_tokens, gen_tokens,"
                " duration_s, session_id, meta) VALUES (?,?,?,?,?,?,?,?)",
                (
                    time.time(),
                    kind,
                    model,
                    int(prompt_tokens),
                    int(gen_tokens),
                    float(duration_s),
                    session_id,
                    json.dumps(meta or {}),
                ),
            )
            self._conn.commit()

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0),"
                " COALESCE(SUM(gen_tokens),0), COALESCE(SUM(duration_s),0)"
                " FROM usage_events"
            )
            calls, pt, gt, dur = cur.fetchone()
            per_kind: Dict[str, Dict[str, Any]] = {}
            for kind, k_calls, k_pt, k_gt, k_dur in self._conn.execute(
                "SELECT kind, COUNT(*), COALESCE(SUM(prompt_tokens),0),"
                " COALESCE(SUM(gen_tokens),0), COALESCE(SUM(duration_s),0)"
                " FROM usage_events GROUP BY kind"
            ):
                per_kind[kind] = {
                    "calls": k_calls,
                    "prompt_tokens": k_pt,
                    "gen_tokens": k_gt,
                    "duration_s": round(k_dur, 3),
                }
        total_tokens = int(pt) + int(gt)
        return {
            "total_calls": int(calls),
            "prompt_tokens": int(pt),
            "gen_tokens": int(gt),
            "total_tokens": total_tokens,
            "duration_s": round(float(dur), 3),
            "avg_tok_per_s": round(int(gt) / float(dur), 2) if float(dur) > 0 else 0.0,
            "by_kind": per_kind,
        }

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, kind, model, prompt_tokens, gen_tokens, duration_s,"
                " session_id, meta FROM usage_events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "ts": r[0],
                "kind": r[1],
                "model": r[2],
                "prompt_tokens": r[3],
                "gen_tokens": r[4],
                "duration_s": round(r[5], 3),
                "session_id": r[6],
                "meta": json.loads(r[7] or "{}"),
            }
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
