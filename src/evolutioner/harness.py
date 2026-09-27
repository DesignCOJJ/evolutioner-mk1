"""Harness: ties engine, MCTS, verifier, virtual context and memory together.

Pipeline (spec): retrieve memory -> (optional) MCTS think -> answer with
RLVR verification of emitted python -> inject verifier outputs back into
the active context for self-correction -> persist session + usage.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backend import BaseBackend
from .config import Config
from .context import VirtualContextEngine
from .memory import MemoryStore
from .search import EvolutionerMCTS
from .usage import UsageTracker
from .verifier import SandboxVerifier

_CODE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>")
_THINK_INNER_RE = re.compile(r"<think>([\s\S]*?)</think>")


@dataclass
class HarnessResult:
    query: str
    answer: str
    think: str = ""
    steps: List[str] = field(default_factory=list)
    rounds: int = 0
    verification: Dict[str, Any] = field(default_factory=dict)
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class Harness:
    def __init__(
        self,
        config: Config,
        llm: BaseBackend,
        tracker: Optional[UsageTracker] = None,
        context: Optional[VirtualContextEngine] = None,
        memory: Optional[MemoryStore] = None,
        verifier: Optional[SandboxVerifier] = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.tracker = tracker
        self.context = context
        self.memory = memory
        self.verifier = verifier
        self.mcts = EvolutionerMCTS(
            llm=llm,
            verifier=verifier,
            c_param=config.c_param,
            candidates=config.candidates,
            temperature=config.temperature,
        )
        self.session_id = uuid.uuid4().hex[:12]
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def _record(self, kind: str, stats: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> None:
        if self.tracker is None:
            return
        self.tracker.record(
            kind=kind,
            model=self.llm.name,
            prompt_tokens=int(stats.get("prompt_tokens", 0)),
            gen_tokens=int(stats.get("gen_tokens", 0)),
            duration_s=float(stats.get("duration_s", 0.0)),
            session_id=self.session_id,
            meta=meta,
        )

    def _build_prompt(self, query: str, retrieved: str, history: str) -> str:
        parts = ["You are Evolutioner MK1, a precise local reasoning engine."]
        if retrieved:
            parts.append(f"Relevant long-term memory:\n{retrieved}")
        if history:
            parts.append(f"Conversation so far:\n{history}")
        parts.append(
            "Rules: reason step by step. When a computation helps, emit a ```python\n"
            "code block; it will be executed and its output shown back to you.\n"
            "Finish with 'Final Answer:' followed by the answer.\n\n"
            f"User Query: {query}"
        )
        return "\n\n".join(parts)

    def _extract_code(self, text: str) -> Optional[str]:
        match = _CODE_RE.search(text or "")
        if match:
            return match.group(1).strip() or None
        return None

    @staticmethod
    def _clean_answer(text: str) -> str:
        """Robust final-answer extraction for R1-style and plain models.

        Order: text outside think tags -> think-internal content -> raw;
        then the last 'Final Answer:' marker. Never returns empty while
        anything usable remains (truncation right after the marker happens).
        """
        raw = (text or "").strip()
        inner_match = _THINK_INNER_RE.search(raw)
        inner = inner_match.group(1).strip() if inner_match else ""
        outside = _THINK_RE.sub("", raw).strip()
        candidate = outside or inner or raw
        if "Final Answer:" in candidate:
            candidate = candidate.rsplit("Final Answer:", 1)[1].strip()
        if not candidate:
            candidate = (inner or outside or raw).strip()
        return candidate.strip()

    def _persist_session(self, result: HarnessResult) -> None:
        session_dir = Path(self.config.session_dir)
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
            path = session_dir / f"{self.session_id}.json"
            path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        except OSError:
            pass  # session persistence is best-effort

    # ------------------------------------------------------------------ #
    def run(self, query: str, *, use_mcts: bool = True, use_rag: bool = True) -> HarnessResult:
        t0 = time.perf_counter()
        result = HarnessResult(query=query, answer="", session_id=self.session_id)

        # 1. Retrieve from virtual context (RAG) + curated memory.
        retrieved_block = ""
        if use_rag and self.config.enable_rag and self.context is not None:
            retrieved_block = self.context.build_context(query)
            result.retrieved = self.context.search(query, k=3)
        if self.memory is not None:
            mem_hits = self.memory.search(query, limit=3)
            if mem_hits:
                mem_block = "\n".join(f"- {m['text']}" for m in mem_hits)
                retrieved_block = f"{retrieved_block}\n{mem_block}".strip()

        history = "\n".join(
            f"Q: {h['query']}\nA: {h['answer']}" for h in self.history[-4:]
        )
        base_prompt = self._build_prompt(query, retrieved_block, history)

        # 2. Think (MCTS) if enabled — produces the <think> trajectory.
        prompt = base_prompt
        if use_mcts:
            mcts_res = self.mcts.search(
                query, iterations=self.config.mcts_iterations, max_depth=self.config.mcts_depth
            )
            self._record("mcts", {
                "prompt_tokens": mcts_res.prompt_tokens,
                "gen_tokens": mcts_res.gen_tokens,
                "duration_s": mcts_res.duration_s,
            })
            result.think = mcts_res.think
            result.steps = mcts_res.steps
            if mcts_res.steps:  # skip empty trajectories (e.g. instant stops)
                prompt = f"{base_prompt}\n\nCandidate reasoning so far:\n{mcts_res.think}"

        # 3. Reasoning rounds with RLVR self-correction.
        max_rounds = max(1, self.config.max_rounds)
        last_error = ""
        finalized = False
        for round_no in range(max_rounds):
            text, stats = self.llm.generate(
                prompt, max_tokens=self.config.gen_tokens,
                temperature=self.config.temperature,
            )
            self._record("answer", stats, {"round": round_no + 1})
            result.rounds = round_no + 1

            code = self._extract_code(text)
            if code and self.verifier is not None and not finalized:
                ver = self.verifier.verify(code)
                result.verification = {
                    "ok": ver.ok,
                    "sandbox": ver.sandbox,
                    "message": ver.message[:400],
                }
                if not ver.ok:
                    last_error = ver.message
                    prompt = (
                        f"{prompt}\n\nPrevious attempt failed. Your code produced:\n"
                        f"{ver.message}\nFix it and answer again."
                    )
                    continue  # self-correction round
                if "Final Answer:" not in text:
                    # Fold verified output back in (spec: RLVR injection) and
                    # ask for one short finalize round.
                    finalized = True
                    prompt = (
                        f"{prompt}\n\nYour code ran successfully with output:\n"
                        f"{ver.stdout.strip() or '(no output)'}\n"
                        f"Reply with one short line starting with 'Final Answer:'."
                    )
                    continue
                text = f"{text}\n\n[verifier output]\n{ver.stdout.strip()}\n[/verifier]"

            result.answer = self._clean_answer(text)
            break
        else:
            if finalized:
                text, stats = self.llm.generate(
                    prompt + "\nReply with one short line starting with 'Final Answer:'.",
                    max_tokens=self.config.gen_tokens,
                    temperature=self.config.temperature,
                )
                self._record("answer", stats, {"round": "finalize"})
                result.answer = self._clean_answer(text)
            else:
                result.answer = (
                    f"(after {max_rounds} rounds, last verification error: "
                    f"{last_error or 'n/a'})"
                )

        # 4. Persist: session file, rolling context, optional auto-memory.
        result.usage = {"total_s": round(time.perf_counter() - t0, 3), "rounds": result.rounds}
        self.history.append({"query": query, "answer": result.answer})
        self._persist_session(result)
        if self.context is not None:
            self.context.add(f"Q: {query} -> A: {result.answer[:400]}",
                             meta={"session": self.session_id})
        if self.config.auto_memory and self.memory is not None:
            self.memory.add(f"Q: {query} -> A: {result.answer[:400]}", source="auto")
        return result
