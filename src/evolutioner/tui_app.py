"""Evolutioner MK1 — cyberpunk TUI (Textual). Real telemetry, real runs.

Layout: banner header with status badges -> tabbed main area (OVERVIEW /
MODELS / USAGE / LOGS) -> command prompt (`evolutioner :: ❯`) -> keybar.
[q] quit  [tab] switch tab  [r] refresh  [e] evolve (live MCTS+RLVR run).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

from rich.box import DOUBLE, HEAVY, ROUNDED
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Input, RichLog, Static, TabbedContent, TabPane

from . import tui_theme as T
from .config import Config
from .ollama_backend import PORTFOLIO_TAGS
from .tui_data import Telemetry

PANEL_KW = dict(box=DOUBLE, border_style=T.CYAN, padding=(0, 1), expand=True)


def _panel(title: str, body: Any, color: str = T.CYAN, box=DOUBLE) -> Panel:
    return Panel(body, title=f"[bold {color}] ◈ {title} ", title_align="left",
                 box=box, border_style=color, padding=(0, 1), expand=True)


class EvolutionerTUI(App):
    TITLE = "EVOLUTIONER MK-1"
    CSS = """
    Screen { background: #0a0c10; }
    #top { height: auto; padding: 0 1; }
    #badges { height: auto; content-align: right middle; }
    #main { height: 1fr; }
    TabbedContent { height: 100%; }
    Tabs { background: $surface-darken-3; }
    .grid2 { height: 1fr; }
    #promptbar { height: 3; padding: 0 1; background: #10141b; }
    #promptlabel { width: auto; padding: 1 1 1 0; color: #00e5ff; }
    Input { border: tall #1f2b3a; background: #0d1117; }
    Input:focus { border: tall #00e5ff; }
    Footer { background: #10141b; }
    RichLog { background: #0a0e13; border: tall #1f2b3a; }
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("tab", "switch_tab", "Switch Tab"),
        ("r", "refresh", "Refresh"),
        ("e", "evolve", "Evolve"),
    ]

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__()
        self.config = config or Config()
        self.tele = Telemetry(self.config)
        self._busy = False
        self._tab_order: List[str] = ["overview", "models", "usage", "logs"]
        self._tab_idx = 0

    # ------------------------------ UI ------------------------------- #
    def compose(self) -> ComposeResult:
        yield Static(T.render_banner(), id="top")
        yield Static("", id="badges")
        with TabbedContent(initial="overview") as tabs:
            with TabPane("◈ OVERVIEW", id="overview"):
                with Horizontal():
                    yield Static("", id="ov_left", expand=True)
                    yield Static("", id="ov_right", expand=True)
            with TabPane("◈ MODELS", id="models"):
                with Horizontal():
                    yield Static("", id="mo_left", expand=True)
                    yield Static("", id="mo_right", expand=True)
            with TabPane("◈ USAGE", id="usage"):
                with Horizontal():
                    yield Static("", id="us_left", expand=True)
                    yield Static("", id="us_right", expand=True)
            with TabPane("◈ LOGS / AGENT", id="logs"):
                with Vertical():
                    yield RichLog(id="agentlog", markup=True, wrap=True, highlight=True)
        with Horizontal(id="promptbar"):
            yield Static("evolutioner :: ❯", id="promptlabel")
            yield Input(placeholder="type a query, or press [e] to evolve  (enter to run)",
                        id="prompt")
        yield Footer()

    # --------------------------- LIFECYCLE --------------------------- #
    def on_mount(self) -> None:
        log = self.query_one("#agentlog", RichLog)
        log.write(f"[{T.MAGENTA}]◈ EVOLUTIONER MK-1 online — MCTS+RLVR harness armed")
        log.write(f"[{T.GRAY}]backend: Ollama @ {self.tele.host} (tiered portfolio)")
        self.set_interval(2.0, self.refresh_telemetry)
        self.refresh_telemetry()
        self.query_one("#prompt", Input).focus()

    # -------------------------- TELEMETRY ---------------------------- #
    def refresh_telemetry(self) -> None:
        tele = self.tele
        badges = tele.status_badges()
        btxt = "  ".join(
            f"[{color}]◆ {name}: {val}[/{color}]" for name, (val, color) in badges.items()
        )
        self.query_one("#badges", Static).update(
            Text.from_markup(f"[{T.GRAY}]UPTIME {tele.uptime()}  │  [/]{btxt}")
        )

        # ---- overview left: telemetry gauges ---- #
        cpu = tele.cpu_sample()
        mem = tele.memory()
        la1, la2, la3 = tele.loadavg()
        cpu_color = T.GREEN if cpu < 0.6 else (T.AMBER if cpu < 0.85 else T.RED)
        mem_color = T.GREEN if mem["frac"] < 0.7 else (T.AMBER if mem["frac"] < 0.9 else T.RED)
        u = tele.usage_summary()
        left_rows = [
            f"[{T.GRAY}]── SYSTEM BUS ──────────────",
            f"[{T.GRAY}]CPU   {T.bar(cpu, 20, color=cpu_color)} {T.pct(cpu)}",
            f"[{T.GRAY}]MEM   {T.bar(mem['frac'], 20, color=mem_color)} {T.pct(mem['frac'])}  ({mem['used']:.1f}/{mem['total']:.1f} GB)",
            f"[{T.GRAY}]LOAD  [{T.CYAN}]{la1:.2f}[/{T.CYAN}] [{T.DIM}]{la2:.2f} {la3:.2f}[/{T.DIM}]",
            f"[{T.GRAY}]GPU   [{T.DIM}]cpu-only mode · llama.cpp/ollama[/{T.DIM}]",
            "",
            f"[{T.GRAY}]── TOKEN STREAM ────────────",
            f"[{T.GRAY}]CALLS [{T.FG}]{u['total_calls']}[/{T.FG}]   [{T.GRAY}]TOKENS [{T.FG}]{u['total_tokens']}[/{T.FG}]",
            f"[{T.GRAY}]PROMPT[{T.FG}]{u['prompt_tokens']}[/{T.FG}]   [{T.GRAY}]GEN    [{T.FG}]{u['gen_tokens']}[/{T.FG}]",
            f"[{T.GRAY}]RATE  [{T.CYAN}]{u['avg_tok_per_s']:.1f} tok/s[/{T.CYAN}]   [{T.GRAY}]TIME [{T.FG}]{u['duration_s']:.1f}s[/{T.FG}]",
            "",
            f"[{T.GRAY}]SESSIONS [{T.MAGENTA}]{tele.session_count()}[/{T.MAGENTA}]  [{T.GRAY}]MEMORIES [{T.MAGENTA}]{tele.memory_count()}[/{T.MAGENTA}]",
        ]
        self.query_one("#ov_left", Static).update(
            _panel("TELEMETRY", "\n".join(left_rows), T.CYAN, ROUNDED))

        # ---- overview right: core matrix ---- #
        port = tele.portfolio_status()
        loaded = {m["name"]: m for m in tele.loaded_models()}
        rows = [
            f"[{T.GRAY}]── NEURAL CORES ────────────",
        ]
        for key in ("1.5B", "2B", "3B", "5B"):
            tag = PORTFOLIO_TAGS[key]
            dot = f"[{T.GREEN}]●[/{T.GREEN}]" if port.get(key) else f"[{T.RED}]○[/{T.RED}]"
            role = {k: v for k, v in {
                "1.5B": "rollouts", "2B": "structured", "3B": "WORKHORSE", "5B": "deep-synth"}.items()}[key]
            live = f"[{T.AMBER}]▲LIVE[/{T.AMBER}]" if tag in loaded else f"[{T.DIM}]idle[/{T.DIM}]"
            rows.append(f" {dot} [{T.FG}]MK1-{key:<5}[/{T.FG}] [{T.GRAY}]{role:<11}[/{T.GRAY}] {live}")
        rows += [
            "",
            f"[{T.GRAY}]── RUNTIME ─────────────────",
            f"[{T.GRAY}]OLLAMA [{T.FG}]v{tele.ollama_version()}[/{T.FG}]",
            f"[{T.GRAY}]ENGINE [{T.FG}]tiered portfolio router[/{T.FG}]",
            f"[{T.GRAY}]RLVR   [{T.FG}]bwrap/rlimit sandbox armed[/{T.FG}]",
            f"[{T.GRAY}]MCTS   [{T.FG}]uct=1.414 · iters={self.config.mcts_iterations} · d={self.config.mcts_depth}[/{T.FG}]",
        ]
        self.query_one("#ov_right", Static).update(
            _panel("CORE MATRIX", "\n".join(rows), T.MAGENTA, ROUNDED))

        # ---- models tab ---- #
        installed = tele.installed_models()
        mo_rows = [f"[{T.GRAY}]── INSTALLED MODELS ({len(installed)}) ──"]
        for name in installed[:14]:
            live = f"  [{T.AMBER}]▲live[/{T.AMBER}]" if name in loaded else ""
            mo_rows.append(f" [{T.GREEN}]▸[/{T.GREEN}] [{T.FG}]{name}[/{T.FG}]{live}")
        self.query_one("#mo_left", Static).update(
            _panel("REGISTRY", "\n".join(mo_rows) or "[dim]ollama offline[/]", T.CYAN, ROUNDED))
        mo_r = [f"[{T.GRAY}]── RESIDENT (VRAM) ──"]
        for name, m in loaded.items():
            gb = m.get("size_vram", 0) / 1e9
            mo_r.append(f" [{T.AMBER}]▲[/{T.AMBER}] [{T.FG}]{name}[/{T.FG}]")
            mo_r.append(f"   [{T.DIM}]resident {gb:.2f} GB[/{T.DIM}]")
        if not loaded:
            mo_r.append(f" [{T.DIM}]no models resident — idle[/{T.DIM}]")
        mo_r += ["", f"[{T.GRAY}]pull: ollama pull wallpillar-lm/<variant>"]
        self.query_one("#mo_right", Static).update(
            _panel("RESIDENT SET", "\n".join(mo_r), T.MAGENTA, ROUNDED))

        # ---- usage tab ---- #
        kinds = u.get("by_kind", {})
        krows = [f"[{T.GRAY}]── CALL MIX ────────────────"] if kinds else []
        for kind, s in kinds.items():
            krows.append(
                f" [{T.CYAN}]{kind:<8}[/{T.CYAN}] [{T.FG}]{s['calls']:>4}[/{T.FG}] calls"
                f"  [{T.GRAY}]{s['prompt_tokens']:>6}p {s['gen_tokens']:>6}g[/{T.GRAY}]")
        if not kinds:
            krows.append(f" [{T.DIM}]no calls yet — press [e] to evolve[/{T.DIM}]")
        self.query_one("#us_left", Static).update(
            _panel("USAGE TOTALS", "\n".join(krows), T.CYAN, ROUNDED))
        recent = tele.usage_recent(9)
        rrows = [f"[{T.GRAY}]── RECENT EVENTS ───────────"]
        for r in reversed(recent):
            ts = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
            rrows.append(
                f" [{T.DIM}]{ts}[/{T.DIM}] [{T.MAGENTA}]{r['kind']:<7}[/{T.MAGENTA}]"
                f" [{T.FG}]{r['gen_tokens']:>5}g[/{T.FG}] [{T.GRAY}]{r['duration_s']:.2f}s[/{T.GRAY}]")
        if not recent:
            rrows.append(f" [{T.DIM}]awaiting first inference[/{T.DIM}]")
        self.query_one("#us_right", Static).update(
            _panel("EVENT STREAM", "\n".join(rrows), T.MAGENTA, ROUNDED))

    # ---------------------------- ACTIONS ---------------------------- #
    def action_switch_tab(self) -> None:
        if self.query_one("#prompt", Input).has_focus:
            self.focus_next()  # default tab behaviour inside the prompt
            return
        self._tab_idx = (self._tab_idx + 1) % len(self._tab_order)
        self.query_one(TabbedContent).active = self._tab_order[self._tab_idx]

    def action_refresh(self) -> None:
        self.refresh_telemetry()
        self.notify("telemetry re-synced", title="MK-1", severity="information")

    def action_evolve(self) -> None:
        self.run_query("Prove that the sum of the first n odd numbers equals n^2. Verify with python.")

    # --------------------------- QUERY RUN --------------------------- #
    def run_query(self, query: str) -> None:
        if self._busy:
            self.notify("engine busy — one evolution at a time", severity="warning")
            return
        self._busy = True
        log = self.query_one("#agentlog", RichLog)
        log.write(f"[{T.CYAN}]❯❯ EVOLVE: {query}")
        t = threading.Thread(target=self._worker, args=(query,), daemon=True)
        t.start()

    def _worker(self, query: str) -> None:
        def ui(msg: str) -> None:
            self.call_from_thread(self.query_one("#agentlog", RichLog).write, msg)

        try:
            from .harness import Harness
            from .ollama_backend import TieredOllamaBackend, ollama_ok
            from .usage import UsageTracker
            from .verifier import SandboxVerifier

            if not ollama_ok(self.tele.host):
                ui(f"[{T.RED}]✗ ollama unreachable at {self.tele.host} — start: ollama serve")
                return
            ui(f"[{T.GRAY}]◈ spinning up tiered portfolio (1.5B rollouts + "
               f"{self.config.variant} reasoner)…")
            tracker = UsageTracker(self.config.usage_db)
            llm = TieredOllamaBackend(self.config.variant,
                                      self.config.extra.get("ollama_rollout_variant", "1.5B"),
                                      host=self.tele.host)
            verifier = SandboxVerifier(timeout_s=self.config.sandbox_timeout_s,
                                       use_bwrap=self.config.use_bwrap)
            harness = Harness(self.config, llm, tracker=tracker, verifier=verifier)
            ui(f"[{T.GRAY}]◈ MCTS searching (iters={self.config.mcts_iterations}, "
               f"depth={self.config.mcts_depth})…")
            t0 = time.perf_counter()
            res = harness.run(query)
            dt = time.perf_counter() - t0
            v = res.verification or {}
            vtxt = (f"[{T.GREEN}]VERIFIED sandbox={v.get('sandbox', '-')}"
                    if v.get("ok") else
                    f"[{T.AMBER}]sandbox={v.get('sandbox', '-')}/{v.get('message', '')[:60]}")
            ui(f"[{T.MAGENTA}]◈ rounds={res.rounds} {vtxt}[/{T.MAGENTA}]")
            for i, step in enumerate(res.steps[:6], 1):
                ui(f"[{T.DIM}]  ├ step {i}: {step[:110]}")
            ui(f"[{T.GREEN}]◆ FINAL ({dt:.1f}s): {res.answer[:600]}")
            for kind, s in (tracker.summary().get("by_kind") or {}).items():
                ui(f"[{T.DIM}]  · {kind}: {s['calls']} calls, {s['prompt_tokens']}p+{s['gen_tokens']}g tok")
            tracker.close()
        except Exception as exc:  # keep the TUI alive no matter what
            ui(f"[{T.RED}]✗ engine fault: {exc}")
        finally:
            self.call_from_thread(setattr, self, "_busy", False)
            self.call_from_thread(self.refresh_telemetry)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if text:
            self.run_query(text)
