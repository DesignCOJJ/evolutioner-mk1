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


def _chat_repl(config: Config) -> int:
    """opencode-style REPL: persistent session, streaming tokens, /commands.

    Chat turns run the *model* directly (fast, conversational). `/solve <q>`
    escalates a question to the full harness (MCTS + RLVR verification).
    """
    from .ollama_backend import OllamaBackend, PORTFOLIO_TAGS, ollama_ok
    from .usage import UsageTracker

    host = config.extra.get("ollama_host") or None
    if not ollama_ok(host):
        print("error: Ollama is not reachable. Start it with:  ollama serve", file=sys.stderr)
        return 2

    tag = PORTFOLIO_TAGS[config.variant]
    llm = OllamaBackend(tag, host=host)
    tracker = UsageTracker(config.usage_db)

    CY = "\033[1;36m"; DIM = "\033[2m"; GRN = "\033[1;32m"; NC = "\033[0m"
    history: List[dict] = []

    def _stream(history_snapshot: List[dict], *, max_tokens: int) -> str:
        payload = {
            "model": llm.model,
            "messages": history_snapshot,
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": config.temperature},
        }
        import urllib.request

        req = urllib.request.Request(
            f"{llm.host}/api/chat", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        chunks: List[str] = []
        prompt_n = eval_n = 0
        total_s = 0.0
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                piece = (ev.get("message") or {}).get("content", "")
                if piece:
                    chunks.append(piece)
                    print(piece, end="", flush=True)
                if ev.get("done"):
                    prompt_n = int(ev.get("prompt_eval_count", 0))
                    eval_n = int(ev.get("eval_count", 0))
                    total_s = round(ev.get("total_duration", 0) / 1e9, 2)
                    print(f"{DIM}\n  [{eval_n} tok · {total_s}s · {eval_n / max(total_s, 0.01):.1f} tok/s]{NC}", flush=True)
        tracker.record(kind="chat", model=llm.model,
                       prompt_tokens=prompt_n, gen_tokens=eval_n,
                       duration_s=total_s)
        return "".join(chunks)

    # persistent history: sessions saved as JSON after every turn, input
    # history via readline (up-arrow across restarts)
    from pathlib import Path
    import time as _time

    sessions_dir = Path(config.session_dir) / "chat"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_id = _time.strftime("%Y%m%d-%H%M%S")
    session_file = sessions_dir / f"{session_id}.json"

    def _save_session() -> None:
        session_file.write_text(json.dumps({
            "id": session_id, "model": tag,
            "saved_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": history,
        }, indent=1), encoding="utf-8")

    hist_file = sessions_dir / "input_history"
    try:
        import readline
        if hist_file.exists():
            readline.read_history_file(str(hist_file))
    except Exception:
        pass

    print(f"{CY}evolutioner chat{NC} {DIM}· {tag} · session {session_id} · /help for commands{NC}")
    while True:
        try:
            line = input("❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            if cmd in ("/exit", "/quit"):
                break
            if cmd == "/help":
                print("  /help             this text")
                print("  /new              clear conversation history")
                print("  /model <v>        switch variant (1.5B/2B/3B/5B)")
                print("  /solve <question> run the FULL harness: MCTS + sandbox verification")
                print("  /history [n]      show recent exchanges (default 10)")
                print("  /sessions         list saved chat sessions")
                print("  /resume [id]      load a previous session into context")
                print("  /save             force-save the session now")
                print("  /exit             quit (ctrl-d also works)")
                continue
            if cmd == "/new":
                history.clear()
                session_id = _time.strftime("%Y%m%d-%H%M%S")
                session_file = sessions_dir / f"{session_id}.json"
                print(f"{DIM}context cleared · new session {session_id}{NC}")
                continue
            if cmd == "/history":
                n = int(arg) if arg.strip().isdigit() else 10
                shown = 0
                for msg in reversed(history):
                    who = f"{GRN}❯{NC}" if msg["role"] == "user" else f"{CY}◂{NC}"
                    print(f"{who} {msg['content'][:200]}")
                    shown += 1
                    if shown >= n * 2:
                        break
                if shown == 0:
                    print(f"{DIM}(empty session){NC}")
                continue
            if cmd == "/sessions":
                files = sorted(sessions_dir.glob("*.json"))
                if not files:
                    print(f"{DIM}(no saved sessions){NC}")
                    continue
                for f in files[-10:]:
                    try:
                        data = json.loads(f.read_text(encoding="utf-8"))
                        print(f"  {data.get('id', f.stem)}  {DIM}{data.get('model', '?')} · {len(data.get('messages', []))} msgs · {data.get('saved_at', '?')}{NC}")
                    except Exception:
                        continue
                continue
            if cmd == "/resume":
                files = sorted(sessions_dir.glob("*.json"))
                if not files:
                    print(f"{DIM}(no saved sessions){NC}")
                    continue
                target = arg.strip()
                if not target:
                    for f in files[-10:]:
                        print(f"  {f.stem}")
                    target = input("session id (blank = latest): ").strip() or files[-1].stem
                path = sessions_dir / (target if target.endswith(".json") else f"{target}.json")
                if not path.exists():
                    print(f"{DIM}no such session: {target}{NC}")
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                history.clear()
                history.extend(data.get("messages", []))
                session_id = path.stem
                session_file = path
                print(f"{DIM}resumed {session_id} ({len(history)} messages){NC}")
                continue
            if cmd == "/save":
                _save_session()
                print(f"{DIM}saved → {session_file}{NC}")
                continue
            if cmd == "/model":
                arg = arg.strip().upper()
                if arg in PORTFOLIO_TAGS:
                    config.variant = arg
                    tag = PORTFOLIO_TAGS[arg]
                    llm = OllamaBackend(tag, host=host)
                    print(f"{DIM}switched to {tag}{NC}")
                else:
                    print(f"{DIM}variants: {', '.join(PORTFOLIO_TAGS)}{NC}")
                continue
            if cmd == "/solve":
                if not arg.strip():
                    print(f"{DIM}usage: /solve <question>{NC}")
                    continue
                print(f"{DIM}harness: MCTS → answer → sandbox verify …{NC}")
                from .context import VirtualContextEngine
                from .harness import Harness
                from .memory import MemoryStore
                from .ollama_backend import build_ollama_backend
                from .verifier import SandboxVerifier

                harness = Harness(
                    config, build_ollama_backend(config),
                    tracker=tracker,
                    context=VirtualContextEngine(config.rag_dir, top_k=config.rag_top_k) if config.enable_rag else None,
                    memory=MemoryStore(config.memories_dir),
                    verifier=SandboxVerifier(timeout_s=config.sandbox_timeout_s, use_bwrap=config.use_bwrap),
                )
                result = harness.run(arg.strip(), use_mcts=True, use_rag=config.enable_rag)
                _print_result(result)
                history.append({"role": "user", "content": arg.strip()})
                history.append({"role": "assistant", "content": result.answer})
                continue
            print(f"{DIM}unknown command {cmd} — /help{NC}")
            continue

        history.append({"role": "user", "content": line})
        print(f"{CY}◂{NC} ", end="", flush=True)
        try:
            reply = _stream(history, max_tokens=config.gen_tokens)
        except Exception as exc:  # keep the REPL alive on network hiccups
            print(f"\n{DIM}error: {exc}{NC}")
            history.pop()
            continue
        if not reply.strip():
            print(f"{DIM}(empty response — model may still be loading; retry){NC}")
            history.pop()
            continue
        history.append({"role": "assistant", "content": reply})
        _save_session()

    _save_session()
    try:
        import readline
        readline.set_history_length(1000)
        readline.write_history_file(str(hist_file))
    except Exception:
        pass
    tracker.close()
    return 0


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

    sub.add_parser("chat", help="opencode-style REPL chat with the portfolio (streaming)")
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

    if args.cmd == "chat":
        return _chat_repl(config)

    if args.cmd == "status":
        from .ollama_backend import PORTFOLIO_TAGS, list_installed, ollama_ok, resolve_model_tag

        host = config.extra.get("ollama_host") or None
        if not ollama_ok(host):
            print("Ollama: NOT RUNNING  (start with:  ollama serve)")
            return 1
        print("Ollama: running")
        installed = set(list_installed(host))
        print("Portfolio models installed:")
        for key, tag in PORTFOLIO_TAGS.items():
            marker = "*" if key == config.variant else " "
            resolved = resolve_model_tag(tag, host)
            if resolved in installed:
                state = "installed" if resolved == tag else f"installed as {resolved}"
            else:
                state = "missing locally (import GGUF: launcher menu 8 → 2)"
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
