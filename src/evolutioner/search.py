"""MCTS & Process Reward Engine — spec pillar 3 (implementation-corrected).

Two deliberate corrections to the spec draft (kept behaviour-compatible):
1. ``MCTSProcessRewardModel.evaluate_step`` extracts fenced python *without*
   the spec regex's bogus ``\\n`` requirement and never raises — any sympy /
   AST failure only lowers the reward.
2. ``search`` backpropagates **once per candidate** (not spec's triple
   visit-inflation) so UCT statistics stay meaningful.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .backend import BaseBackend
from .verifier import SandboxVerifier

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)


class Node:
    __slots__ = ("state", "parent", "action", "children", "visits", "value", "is_terminal")

    def __init__(self, state: str, parent: Optional["Node"] = None, action: str = "") -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List["Node"] = []
        self.visits = 0
        self.value = 0.0
        self.is_terminal = False

    def uct_score(self, c_param: float = 1.414) -> float:
        if self.visits == 0:
            return float("inf")
        if not self.parent or self.parent.visits == 0:
            return self.value / self.visits
        return (self.value / self.visits) + c_param * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def _exploitation(self) -> float:
        return self.value / self.visits if self.visits else 0.0


def uct(node: Node, c_param: float = 1.414) -> float:
    """Standard UCT: exploitation + exploration term."""
    return node.uct_score(c_param)


@dataclass
class MCTSResult:
    think: str
    steps: List[str] = field(default_factory=list)
    iterations: int = 0
    prompt_tokens: int = 0
    gen_tokens: int = 0
    duration_s: float = 0.0


class MCTSProcessRewardModel:
    """Rule-based process reward: executable/verifiable steps score higher."""

    def __init__(self, verifier: Optional[SandboxVerifier] = None) -> None:
        self.verifier = verifier

    def evaluate_step(self, step_text: str) -> float:
        score = 0.5
        if "```python" in step_text or "sympy" in step_text:
            match = _CODE_FENCE_RE.search(step_text)
            code = match.group(1).strip() if match else ""
            if not code:
                return 0.1
            # Static AST sanity (cheap, deterministic).
            try:
                ast.parse(code)
                score += 0.1
            except SyntaxError:
                return 0.0
            # Dynamic verification when a sandbox is available.
            if self.verifier is not None:
                result = self.verifier.verify(code)
                score += 0.3 if result.ok else -0.4
            else:
                try:  # pragma: no cover - sympy optional in this path
                    import sympy

                    sympy.sympify(code)
                    score += 0.4
                except Exception:
                    # No sandbox -> no dynamic verdict; valid-parse code stays
                    # neutral instead of being punished for non-sympy syntax.
                    pass
        return max(0.0, min(1.0, score))


class EvolutionerMCTS:
    def __init__(
        self,
        llm: BaseBackend,
        verifier: Optional[SandboxVerifier] = None,
        c_param: float = 1.414,
        candidates: int = 3,
        temperature: float = 0.7,
    ) -> None:
        self.llm = llm
        self.prm = MCTSProcessRewardModel(verifier)
        self.c_param = c_param
        self.candidates = max(1, candidates)
        self.temperature = temperature
        self.prompt_tokens = 0
        self.gen_tokens = 0
        self.duration_s = 0.0

    # ------------------------------------------------------------------ #
    def _track(self, stats: Dict[str, Any]) -> None:
        self.prompt_tokens += int(stats.get("prompt_tokens", 0))
        self.gen_tokens += int(stats.get("gen_tokens", 0))
        self.duration_s += float(stats.get("duration_s", 0.0))

    def generate_step_candidates(self, context: str, num_candidates: int = 3) -> List[str]:
        # Tiered backends (e.g. Ollama portfolio) expose a cheap rollout path;
        # fall back to the main generate() for plain backends.
        gen = getattr(self.llm, "generate_rollout", None) or self.llm.generate
        candidates: List[str] = []
        for _ in range(max(1, num_candidates)):
            # R1-distill rollouts open with <think> and may newline instantly,
            # so stop only on </think> and take the first meaningful line here.
            text, stats = gen(
                context,
                max_tokens=64,
                temperature=self.temperature,
                stop=("</think>",),
            )
            self._track(stats)
            step = self._extract_step(text)
            if step and step not in candidates:
                candidates.append(step)
        return candidates

    @staticmethod
    def _extract_step(text: str) -> str:
        text = re.sub(r"</?think>", "", text or "")
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line[:200]
        return ""

    def _select(self, root: Node, max_depth: int) -> Tuple[Node, int]:
        node, depth = root, 0
        while node.children and depth < max_depth and not node.is_terminal:
            node = max(node.children, key=lambda c: uct(c, self.c_param))
            depth += 1
        return node, depth

    def search(self, prompt: str, iterations: int = 8, max_depth: int = 4) -> MCTSResult:
        root_prompt = f"<think>\nUser Query: {prompt}\nStep 1:"
        root = Node(state=root_prompt)

        for _ in range(max(1, iterations)):
            node, depth = self._select(root, max_depth)

            if not node.is_terminal and depth < max_depth:
                candidates = self.generate_step_candidates(node.state, self.candidates)
                for cand in candidates:
                    child = Node(state=f"{node.state}\n{cand}", parent=node, action=cand)
                    if "</think>" in cand or "Final Answer:" in cand:
                        child.is_terminal = True
                    node.children.append(child)
                # Backprop per candidate (spec-corrected, see module docstring).
                for child in node.children:
                    reward = self.prm.evaluate_step(child.action)
                    curr: Optional[Node] = child
                    while curr is not None:
                        curr.visits += 1
                        curr.value += reward
                        curr = curr.parent

        best_trajectory: List[str] = []
        curr = root
        while curr.children:
            curr = max(curr.children, key=lambda c: (c.visits, c._exploitation()))
            best_trajectory.append(curr.action)
            if curr.is_terminal:
                break

        think = "<think>\n" + "\n".join(best_trajectory) + "\n</think>"
        return MCTSResult(
            think=think,
            steps=best_trajectory,
            iterations=iterations,
            prompt_tokens=self.prompt_tokens,
            gen_tokens=self.gen_tokens,
            duration_s=round(self.duration_s, 4),
        )
