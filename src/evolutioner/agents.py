"""Tool-using agent system with sub-agent spawning (original spec ask).

Safety model:
- File tools are jailed to a workspace root (path-escape blocked).
- ``shell`` is disabled unless explicitly enabled AND confirmed per call.
- Sub-agents spawn depth-limited, never inherit shell, and run on their own
  portfolio tier (default 2B workers, 3B+ orchestrator).
- Every model call flows through the UsageTracker.
"""

from __future__ import annotations

import ast
import json
import os
import platform
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .backend import BaseBackend
from .config import Config
from .ollama_backend import PORTFOLIO_TAGS, OllamaBackend
from .usage import UsageTracker

_TOOL_RE = re.compile(r"```tool\s*\n(.*?)\n```", re.DOTALL)

MAX_SPAWN_DEPTH = 2
MAX_STEPS = 8


# ----------------------------- tools ------------------------------------- #
def _sysinfo() -> str:
    mem = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines()[:3]:
            k, v = line.split(":", 1)
            mem[k.strip()] = v.strip()
    except OSError:
        pass
    load = os.getloadavg()
    return json.dumps({
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "loadavg": [round(x, 2) for x in load],
        "mem": mem,
        "cwd": os.getcwd(),
        "ollama_host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    }, indent=1)


@dataclass
class ToolContext:
    workspace: Path
    allow_shell: bool = False
    confirm_shell: bool = True  # interactive y/N confirmation per command

    def resolve(self, path: str) -> Path:
        p = (self.workspace / path).resolve()
        if not str(p).startswith(str(self.workspace.resolve())):
            raise PermissionError(f"path escapes workspace jail: {path}")
        return p


def make_tools(ctx: ToolContext, shell_enabled: bool) -> Dict[str, Dict[str, Any]]:
    tools: Dict[str, Dict[str, Any]] = {
        "list_dir": {
            "desc": "List a directory in the workspace. args: {path}",
            "fn": lambda path=".": "\n".join(
                sorted(p.name + ("/" if p.is_dir() else "") for p in ctx.resolve(path).iterdir())
            ) or "(empty)",
        },
        "read_file": {
            "desc": "Read a text file (max 4000 chars). args: {path}",
            "fn": lambda path: ctx.resolve(path).read_text(encoding="utf-8", errors="replace")[:4000],
        },
        "write_file": {
            "desc": "Write a text file. args: {path, content}",
            "fn": lambda path, content="": _write(ctx, path, content),
        },
        "sysinfo": {
            "desc": "Machine telemetry snapshot (cpu/mem/load). args: {}",
            "fn": lambda: _sysinfo(),
        },
    }
    if shell_enabled:
        def _shell(cmd: str, timeout: float = 20.0) -> str:
            if not shutil.which("sh"):
                return "error: no shell available"
            if ctx.confirm_shell:
                answer = input(f"\n[agent shell request] $ {cmd}\nApprove? [y/N] ").strip().lower()
                if answer != "y":
                    return "denied by operator"
            proc = subprocess.run(
                ["sh", "-c", cmd], capture_output=True, text=True,
                timeout=timeout, cwd=str(ctx.workspace),
            )
            out = (proc.stdout + (f"\n[stderr] {proc.stderr}" if proc.stderr else "")).strip()
            return f"exit={proc.returncode}\n{out[:4000]}"
        tools["shell"] = {
            "desc": "Run an approved shell command in the workspace. args: {cmd}",
            "fn": _shell,
        }
    return tools


def _write(ctx: ToolContext, path: str, content: str) -> str:
    p = ctx.resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} bytes to {p.name}"


def tool_catalog(tools: Dict[str, Dict[str, Any]]) -> str:
    lines = ["Available tools (call with a ```tool json block):"]
    for name, spec in tools.items():
        lines.append(f"- {name}: {spec['desc']}")
    lines.append(
        'Example:\n```tool\n{"tool": "list_dir", "args": {"path": "."}}\n```\n'
        "After each tool call you will receive its output. When done, reply with "
        "'Final Report: <summary>' and no tool block."
    )
    return "\n".join(lines)


# --------------------------- sub-agents ---------------------------------- #
@dataclass
class SubAgent:
    name: str
    role: str
    task: str
    tier: str
    depth: int
    llm: BaseBackend
    tools: Dict[str, Dict[str, Any]]
    max_steps: int = MAX_STEPS
    transcript: List[Dict[str, str]] = field(default_factory=list)
    report: str = ""
    tool_calls: int = 0

    def system_header(self) -> str:
        return (
            f"You are {self.name}, a {self.role} sub-agent (tier {self.tier}).\n"
            f"TASK: {self.task}\n\n{tool_catalog(self.tools)}"
        )


class AgentOrchestrator:
    """Runs tool-loop agents; can spawn depth-limited sub-agents."""

    def __init__(self, config: Config, llm: Optional[BaseBackend] = None,
                 tracker: Optional[UsageTracker] = None,
                 workspace: str = "./workspace",
                 shell: bool = False, confirm_shell: bool = True) -> None:
        self.config = config
        self.tracker = tracker
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.ctx = ToolContext(workspace=self.workspace, allow_shell=shell,
                               confirm_shell=confirm_shell)
        self._root_llm = llm  # used only for depth-0 runs; sub-agents get their tier
        self.agents: List[SubAgent] = []

    def _backend_for_tier(self, tier: str) -> BaseBackend:
        if tier in PORTFOLIO_TAGS:
            return OllamaBackend(PORTFOLIO_TAGS[tier], host=self.config.extra.get("ollama_host"))
        raise ValueError(f"unknown tier {tier!r}")

    def _gen(self, llm: BaseBackend, prompt: str, max_tokens: int = 512) -> str:
        text, stats = llm.generate(prompt, max_tokens=max_tokens,
                                   temperature=self.config.temperature)
        if self.tracker:
            self.tracker.record("agent", llm.name, stats.get("prompt_tokens", 0),
                                stats.get("gen_tokens", 0), stats.get("duration_s", 0.0))
        return text

    # ------------------------------------------------------------------ #
    def run_task(self, task: str, role: str = "orchestrator", tier: Optional[str] = None,
                 depth: int = 0) -> SubAgent:
        tier = tier or self.config.variant
        llm = self._root_llm if depth == 0 else None
        if llm is None:
            llm = self._backend_for_tier(tier)
        shell_ok = self.ctx.allow_shell and depth == 0
        tools = make_tools(self.ctx, shell_enabled=shell_ok)

        agent = SubAgent(
            name=f"agent-{uuid.uuid4().hex[:6]}",
            role=role, task=task, tier=tier, depth=depth, llm=llm, tools=tools,
        )
        self.agents.append(agent)

        prompt = agent.system_header()
        for step in range(agent.max_steps):
            text = self._gen(llm, prompt)
            agent.transcript.append({"step": step, "out": text[:2000]})

            block = _TOOL_RE.search(text)
            if not block:
                agent.report = self._final_of(text)
                break
            try:
                raw_call = block.group(1).strip()
                try:
                    call = json.loads(raw_call)
                except json.JSONDecodeError:
                    # Small models emit single-quoted pseudo-JSON; accept
                    # Python dict literals as a fallback.
                    call = ast.literal_eval(raw_call)
                if not isinstance(call, dict) or "tool" not in call:
                    raise ValueError("missing 'tool' key")
                name, args = call.get("tool"), call.get("args", {}) or {}
            except Exception as exc:
                result = (f"tool call parse error: {exc} — emit a ```tool block "
                          f'with strict JSON, e.g. {{"tool": "write_file", "args": '
                          f'{"path": "x.txt", "content": "hi"}}}')
                name = "invalid"
            else:
                if name == "spawn_subagent":
                    if depth + 1 >= MAX_SPAWN_DEPTH:
                        result = "spawn denied: max depth reached"
                    else:
                        sub = self.run_task(
                            task=str(args.get("task", "help the orchestrator")),
                            role=str(args.get("role", "worker")),
                            tier=str(args.get("tier", "2B")) if str(args.get("tier", "2B")) in PORTFOLIO_TAGS else "2B",
                            depth=depth + 1,
                        )
                        result = f"sub-agent {sub.name} report: {sub.report[:1200] or '(no report)'}"
                elif name in tools:
                    try:
                        result = str(tools[name]["fn"](**args))[:4000] or "(ok, no output)"
                    except Exception as exc:
                        result = f"tool error: {type(exc).__name__}: {exc}"
                else:
                    result = f"unknown tool {name!r}; available: {', '.join(tools)}"
            agent.tool_calls += 1
            agent.transcript.append({"step": step, "tool": name, "result": result[:1500]})
            prompt = (
                f"{prompt}\n\n---\nYour last action:\n{text[:800]}\n\nTool result ({name}):\n"
                f"{result}\n\nContinue. Use another tool or reply with 'Final Report: ...'."
            )
        else:
            agent.report = "(max steps reached without Final Report)"
        return agent

    @staticmethod
    def _final_of(text: str) -> str:
        if "Final Report:" in text:
            return text.rsplit("Final Report:", 1)[1].strip()
        return text.strip()[:2000]
