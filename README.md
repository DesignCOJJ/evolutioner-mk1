# Evolutioner MK1 🧠⚡

Local reasoning harness implementing the master spec: **MCTS test-time-compute
scaling**, a **deterministic RLVR verifier sandbox**, **virtual context**
(RAG), **persistent memory**, **tool-using agents with sub-agent spawning**,
and a **cyberpunk TUI dashboard** — running your `wallpillar-lm` portfolio
through **Ollama** (or native llama.cpp) on plain CPU.

Preview: `docs/tui_preview.svg` (open in a browser).

---

## 0. Setup (once)

```bash
cd ~/evolutioner-mk1
bash setup.sh        # creates env/, installs deps, runs mock selftest
```

Requires: Python 3.10+, Ollama (`sudo pacman -S ollama`, then `ollama serve`).
Optional: `bubblewrap` for the hardened RLVR jail (auto-fallback otherwise).

Pull your portfolio (already done on this machine):

```bash
for v in 1.5B 2B 3B 5B; do ollama pull wallpillar-lm/evolutionermk1-$v:latest; done
```

---

## 1. The launcher (easiest way to run everything)

```bash
bash launch.sh
```

Menu: `1` TUI dashboard · `2` query harness · `3` agent task · `4` portfolio
status · `5` usage stats · `6` selftest · `7` tests · `8` Ollama controls
(serve / models / pull / chat / ps / stop).

---

## 2. TUI dashboard

```bash
PYTHONPATH=src env/bin/python -m evolutioner.cli tui
```

- **OVERVIEW** — live CPU/RAM/load gauges, token stream counters, core matrix.
- **MODELS** — installed models + which are resident in VRAM right now.
- **USAGE** — per-kind call mix (mcts/answer/agent) + live event stream.
- **LOGS / AGENT** — the agent console; your typed queries run here.
- Keys: `e` evolve (demo run) · type a query + Enter · `r` refresh ·
  `tab` switch · `q` quit. Runs use the tiered portfolio and stream progress
  (spin-up → MCTS → verification → final) into the log pane.

---

## 3. CLI harness (headless)

```bash
source env/bin/activate
python -m evolutioner.cli status        # ollama + portfolio check
python -m evolutioner.cli query "What is 37 * 43, and the square root of it? Verify with python."
python -m evolutioner.cli query "..." --no-mcts --no-rag      # fast path
python -m evolutioner.cli query "..." --variant 5B --iters 4 --depth 3
python -m evolutioner.cli query "..." --backend llama          # native GGUF (auto-downloads)
python -m evolutioner.cli usage          # token/call tracker (SQLite)
python -m evolutioner.cli selftest       # mock pipeline, no model needed
```

Every run: memory retrieval → MCTS `<think>` trajectory (1.5B rollouts) →
3B/5B answer with ```python blocks → sandbox execution → error fed back for
self-correction → verified output folded into the final answer → session +
usage persisted (`data/`).

---

## 4. Agents & sub-agents

```bash
python -m evolutioner.cli agent "Inventory the workspace and write notes.md summarizing it" --tier 3B
python -m evolutioner.cli agent "..." --tier 2B --json          # machine-readable transcript
python -m evolutioner.cli agent "..." --shell                   # enable shell tool (per-call y/N confirm)
```

- Tools: `list_dir`, `read_file`, `write_file`, `sysinfo` (+ `shell` if
  enabled). File tools are **jailed to `./workspace`** — path escapes are
  rejected. Shell is disabled unless `--shell`, and every command needs
  interactive approval (`--yes` auto-approves: dangerous).
- Sub-agents: the model emits `{"tool": "spawn_subagent", "args": {"task":
  "...", "tier": "2B"}}`; spawns are depth-limited to 2, never inherit shell,
  run on their own tier, and report back to the parent. All calls are tracked
  in usage (`kind=agent`).

---

## 5. Python API

```python
from evolutioner import Config, Harness, TieredOllamaBackend, UsageTracker, SandboxVerifier

cfg = Config(enable_rag=True)
llm = TieredOllamaBackend("3B", "1.5B")            # tiered portfolio
h = Harness(cfg, llm, tracker=UsageTracker(cfg.usage_db),
            verifier=SandboxVerifier(timeout_s=5))
res = h.run("What is 2^10? Verify with python.")
print(res.answer, res.verification)
```

---

## 6. Configuration (`config/default.json`)

| Key | Default | Meaning |
| --- | --- | --- |
| `variant` | `3B` | reasoning tier (2B/3B/5B) |
| `ollama_tiered` | `true` | 1.5B rollouts + main reasoner |
| `ollama_rollout_variant` | `1.5B` | cheap MCTS tier |
| `ollama_host` | `http://localhost:11434` | or `OLLAMA_HOST` env |
| `mcts_iterations` / `mcts_depth` | `8` / `4` | TTC budget |
| `gen_tokens` / `max_rounds` | `384` / `3` | answer budget / self-correction rounds |
| `enable_rag` / `auto_memory` | `true` / `false` | virtual context / memory writeback |
| `router_fallback` | `null` | e.g. `"api:OPENROUTER_API_KEY"` |
| `use_bwrap` | `null` | `null`=auto-probe, `true`=require, `false`=rlimit fallback |

---

## 7. Tests & crosschecking

```bash
env/bin/python -m compileall -q src tests            # 1. syntax
env/bin/python -m unittest discover -s tests         # 2. unit + TUI (headless)
PYTHONPATH=src env/bin/python -m evolutioner.cli selftest   # 3. integration
PYTHONPATH=src env/bin/python -m evolutioner.cli query "7*6? Verify."  # 4. live
```

## 8. Security notes

- Verifier runs code in `bwrap --unshare-all` (no network, no FS outside a
  scratch dir) or an rlimit subprocess (CPU/RAM/NPROC/FSIZE caps).
- Agent file tools are workspace-jailed; shell needs `--shell` + per-call
  confirmation; sub-agent spawn depth ≤ 2.
- Nothing leaves your machine unless you configure `router_fallback`.

## 9. Spec deviations (deliberate fixes)

1. `evaluate_step` never crashes on malformed code (spec regex could kill the search loop).
2. MCTS backprops once per candidate (spec triple-counted visits, corrupting UCT).
3. Rollouts stop on `</think>` only (spec's `\n` stop empties R1-distill output instantly).
4. Cloud fallback uses an OpenAI-compatible endpoint (portfolio has no hosted API).
5. `/api/chat` (not `/api/generate`) so each model's chat template is applied.
