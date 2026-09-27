"""Evolutioner MK1 command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .config import Config, VARIANTS


def _build(config: Config, mock: bool, backend: str = "ollama"):
    from .context import VirtualContextEngine
    from .harness import Harness
    from .memory import MemoryStore
    from .usage import UsageTracker
    from .verifier import SandboxVerifier

    if mock:
        from .mock import MockLLM

        llm = MockLLM(default=(
            "Step: compute the sum.\n```python\nprint(2 + 2)\n```\n"
            "Final Answer: 4"
        ))
    elif backend == "ollama":
        from .ollama_backend import build_ollama_backend, ollama_ok

        if not ollama_ok(config.extra.get("ollama_host") or None):
            raise RuntimeError(
                "Ollama is not reachable. Start it with:  ollama serve"
            )
        llm = build_ollama_backend(config)
    else:
        from .engine import EvolutionerEngine

        llm = EvolutionerEngine(config)

    tracker = UsageTracker(config.usage_db)
    context = VirtualContextEngine(config.rag_dir, top_k=config.rag_top_k) if config.enable_rag else None
    memory = MemoryStore(config.memories_dir)
    verifier = SandboxVerifier(timeout_s=config.sandbox_timeout_s, use_bwrap=config.use_bwrap)
    harness = Harness(config, llm, tracker=tracker, context=context, memory=memory, verifier=verifier)
    return harness, tracker


def _print_result(result) -> None:
    print("=" * 62)
    print(f"session    : {result.session_id}")
    if result.steps:
        print("think      :")
        for i, step in enumerate(result.steps[:8], 1):
            print(f"  {i}. {step[:120]}")
    if result.verification:
        v = result.verification
        print(f"verify     : {'OK' if v.get('ok') else 'FAILED'}"
              f" (sandbox={v.get('sandbox')})")
    print(f"rounds     : {result.rounds}")
    print(f"latency    : {result.usage.get('total_s', 0)}s")
    print("-" * 62)
    print(result.answer)
    print("=" * 62)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="evolutioner",
        description="Evolutioner MK1 — local reasoning harness (MCTS + RLVR + RAG)",
    )
    parser.add_argument("--config", default=None, help="path to config JSON")
    sub = parser.add_subparsers(dest="cmd")

    q = sub.add_parser("query", help="answer a single query")
    q.add_argument("text", nargs="+", help="the query text")
    q.add_argument("--no-mcts", action="store_true", help="skip MCTS think phase")
    q.add_argument("--no-rag", action="store_true", help="disable memory retrieval")
    q.add_argument("--json", action="store_true", dest="as_json", help="print JSON result")
    q.add_argument("--backend", choices=("ollama", "llama"), default="ollama",
                   help="inference backend (default: ollama)")
    q.add_argument("--iters", type=int, help="MCTS iterations override")
    q.add_argument("--depth", type=int, help="MCTS max depth override")
    q.add_argument("--gen", type=int, help="max answer tokens override")
    q.add_argument("--variant", choices=tuple(VARIANTS), help="portfolio variant override")

    sub.add_parser("selftest", help="run built-in mock-backend selftest (no model needed)")
    sub.add_parser("models", help="list official wallpillar-lm variants")
    sub.add_parser("usage", help="show usage statistics")
    sub.add_parser("status", help="show Ollama server status and installed portfolio models")
    sub.add_parser("tui", help="launch the cyberpunk TUI dashboard")
    ag = sub.add_parser("agent", help="run a tool-using agent task (can spawn sub-agents)")
    ag.add_argument("task", nargs="+")
    ag.add_argument("--tier", choices=tuple(VARIANTS), default=None, help="orchestrator tier")
    ag.add_argument("--shell", action="store_true", help="enable shell tool (still per-call confirmed)")
    ag.add_argument("--yes", action="store_true", help="auto-approve shell commands (DANGEROUS)")
    ag.add_argument("--workspace", default="./workspace")
    ag.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args(argv)

    overrides = {}
    if args.cmd == "query" and args.no_rag:
        overrides["enable_rag"] = False
    if getattr(args, "iters", None):
        overrides["mcts_iterations"] = args.iters
    if getattr(args, "depth", None):
        overrides["mcts_depth"] = args.depth
    if getattr(args, "gen", None):
        overrides["gen_tokens"] = args.gen
    if getattr(args, "variant", None):
        overrides["variant"] = args.variant
    config = Config.load(args.config, **overrides)

    if args.cmd == "models":
        print(f"Official model portfolio ({VARIANTS['3B'].repo_id.split('/')[0]}):")
        for key, spec in VARIANTS.items():
            marker = "*" if key == config.variant else " "
            print(f" {marker} {key:<5} {spec.params:<8} {spec.filename}")
            print(f"        {spec.purpose}")
        return 0

    if args.cmd == "tui":
        from .tui_app import EvolutionerTUI

        EvolutionerTUI(config).run()
        return 0

    if args.cmd == "agent":
        from .agents import AgentOrchestrator
        from .ollama_backend import ollama_ok
        from .usage import UsageTracker

        if not ollama_ok(config.extra.get("ollama_host") or None):
            print("error: Ollama is not reachable. Start it with:  ollama serve", file=sys.stderr)
            return 2
        tracker = UsageTracker(config.usage_db)
        orch = AgentOrchestrator(
            config, tracker=tracker,
            workspace=args.workspace, shell=args.shell,
            confirm_shell=not args.yes,
        )
        agent = orch.run_task(" ".join(args.task), tier=args.tier)
        if args.as_json:
            print(json.dumps({
                "name": agent.name, "tier": agent.tier, "depth": agent.depth,
                "report": agent.report, "tool_calls": agent.tool_calls,
                "transcript": agent.transcript,
            }, indent=2))
        else:
            print(f"agent {agent.name} (tier {agent.tier}, tools={agent.tool_calls})")
            print("-" * 60)
            print(agent.report or "(no report)")
        tracker.close()
        return 0

    if args.cmd == "status":
        from .ollama_backend import PORTFOLIO_TAGS, list_installed, ollama_ok

        host = config.extra.get("ollama_host") or None
        if not ollama_ok(host):
            print("Ollama: NOT RUNNING  (start with:  ollama serve)")
            return 1
        print("Ollama: running")
        installed = set(list_installed(host))
        print("Portfolio models installed:")
        for key, tag in PORTFOLIO_TAGS.items():
            marker = "*" if key == config.variant else " "
            state = "installed" if tag in installed else "MISSING (ollama pull " + tag + ")"
            print(f" {marker} {key:<5} {tag:<45} {state}")
        others = installed - set(PORTFOLIO_TAGS.values())
        if others:
            print("Other installed models:", ", ".join(sorted(others)))
        return 0

    if args.cmd == "usage":
        from .usage import UsageTracker

        tracker = UsageTracker(config.usage_db)
        print(json.dumps(tracker.summary(), indent=2))
        tracker.close()
        return 0

    mock = args.cmd == "selftest"
    backend = getattr(args, "backend", "ollama")
    try:
        harness, tracker = _build(config, mock=mock, backend=backend)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.cmd == "selftest":
        print("[selftest] verifier sandbox:", end=" ")
        ver = harness.verifier.verify("print(21 * 2)")
        print(f"{'OK' if ver.ok and '42' in ver.stdout else 'FAIL'} (sandbox={ver.sandbox})")
        print("[selftest] mock harness:   ", end=" ")
        result = harness.run("2+2?", use_mcts=True, use_rag=False)
        ok = "4" in result.answer
        print(f"{'OK' if ok else 'FAIL'} (rounds={result.rounds})")
        print("[selftest] usage tracker:  ", end=" ")
        summary = tracker.summary()
        ok2 = summary["total_calls"] >= 2
        print(f"{'OK' if ok2 else 'FAIL'} ({summary['total_calls']} calls,"
              f" {summary['total_tokens']} tokens)")
        tracker.close()
        return 0 if ok and ok2 else 1

    if args.cmd == "query":
        text = " ".join(args.text)
        result = harness.run(text, use_mcts=not args.no_mcts, use_rag=not args.no_rag)
        if args.as_json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            _print_result(result)
        tracker.close()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
