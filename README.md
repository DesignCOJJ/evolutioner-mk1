# Evolutioner MK1 🧠⚡

> **Open-Source Evolutionary Search Harness, Verification Engine, and 3B Parameter Reasoning Framework.**

Evolutioner MK1 is a modular reasoning harness designed to scale small-footprint language models ($\le$ 5B parameters) into high-capacity reasoning engines. By shifting the workload from pure parametric memory to **Test-Time Compute (TTC) scaling**, **deterministic rule-based verifiers (RLVR)**, and **external stateful memory**, Evolutioner MK1 enables edge-class hardware to execute complex, step-by-step reasoning workflows with zero hallucination on verifiable tasks.

---

## 🏛️ Official Model Portfolio (`wallpillar-lm`)

Evolutioner MK1 supports a tiered family of fine-tuned and quantized models hosted under the official [`wallpillar-lm`](https://huggingface.co/wallpillar-lm) Hugging Face organization:

| Model ID & Repo | Base Architecture | Size | Primary Specialized Purpose |
| :--- | :--- | :--- | :--- |
| **`wallpillar-lm/evolutionermk1-1.5B`** | DeepSeek-R1-Distill-1.5B | 1.5B | **Ultra-Fast Rollouts:** Rapid `<think>` candidate generation for high-depth MCTS tree exploration and instant verification. |
| **`wallpillar-lm/evolutionermk1-2B`** | Gemma-2-2B-Instruct | 2.0B | **Instruction & Structured Tasks:** Multi-turn conversational workflows, schema enforcement, and JSON outputs. |
| **`wallpillar-lm/evolutionermk1-3B`** | Qwen-2.5-3B-Instruct | 3.0B | **Primary Local Workhorse:** Math/SymPy execution loops, Python AST checking, and balanced logic-to-RAM efficiency. |
| **`wallpillar-lm/evolutionermk1-5B`** | DeepSeek-R1-Distill-7B | ~5–7B | **High-Capacity Engine:** Deep multi-step reasoning and complex code synthesis for offloaded or high-memory runs. |

---

## 📐 Architecture & Execution Split

To run reliably within tight hardware bounds (e.g., 8 GB system RAM with zero dedicated GPU), Evolutioner MK1 decouples **Inference & Reasoning Verification** from **Model Fine-Tuning**:

```
                       ┌──────────────────────────────────────────┐
                       │          Evolutioner MK1 Harness         │
                       └────────────────────┬─────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
      ┌───────────────────────────┐                   ┌───────────────────────────┐
      │     Local CPU Harness     │                   │     Cloud GPU Engine      │
      │   (i5-4440 / 8GB RAM)     │                   │ (Colab / Kaggle / RunPod) │
      ├───────────────────────────┤                   ├───────────────────────────┤
      │ • llama.cpp (GGUF Q4_K_M) │                   │ • Unsloth + GRPO Training │
      │ • SymPy / AST Verifier    │                   │ • vLLM Rollout Generation │
      │ • MCTS Search Control     │                   │ • FP16/BF16 Merging       │
      │ • API Fallback Router     │                   │ • Final GGUF Quantization │
      └───────────────────────────┘                   └───────────────────────────┘
```

### The 4 Core Pillars

1. **Quantized Inference Engine:** Powered by `llama.cpp` using standard $Q4\_K\_M$ GGUF weights from `wallpillar-lm`, optimized for CPU AVX2 instructions without requiring an NVIDIA GPU.
2. **Deterministic RLVR Verifier:** Intercepts intermediate reasoning blocks inside `<think>` tags and executes dynamic Python/SymPy code in a sandboxed runtime (`bubblewrap`). Standard execution outputs or error messages are injected back into the active KV-cache for real-time self-correction.
3. **MCTS & Process Reward Model (PRM):** Evaluates candidate trajectories at step/newline boundaries, pruning dead paths early to save compute cycles.
4. **Virtual Context Engine:** Manages a rolling KV-cache window backed by dense vector retrieval (RAG) to simulate infinite context without exploding physical RAM limits.

---

## 💻 Hardware & Operating Limits

Evolutioner MK1 is engineered to run locally on low-cost edge machines while allowing offloaded cloud training.

| Constraint | Local CPU Runtime (Haswell / AVX2) | Cloud GPU Fine-Tuning Target |
| :--- | :--- | :--- |
| **Target CPU/GPU** | Intel Core i5-4440 (4 cores / 4 threads) | NVIDIA T4 / A10G / A100 |
| **System Memory** | 8 GB DDR3 RAM (Uses ~1.5–3.5 GB peak) | 16 GB+ VRAM |
| **Execution Acceleration** | AVX2 instructions (No CUDA required) | CUDA 12.1+ / FlashAttention-2 |
| **Expected Token Speed** | ~8–15 tokens/sec (3B GGUF) | 100+ tokens/sec |

---

## 🛠️ Environment Setup & Quickstart

### 1. System Dependencies (Arch Linux / CachyOS)

Install basic compilation tools and container isolation packages:

```bash
sudo pacman -Syu --needed base-devel git python python-pip python-virtualenv bubblewrap
```

*(On Ubuntu/Debian: `sudo apt update && sudo apt install build-essential git python3 python3-pip python3-venv bubblewrap`)*

### 2. Repository Setup

Clone the repository and prepare the isolated environment:

```bash
git clone https://github.com/your-username/evolutioner-mk1.git
cd evolutioner-mk1

# Create and activate environment
python -m venv env
source env/bin/activate  # On fish shell: source env/bin/activate.fish

# Install lightweight CPU runtime dependencies
pip install --upgrade pip uv
uv pip install llama-cpp-python sympy pydantic requests huggingface_hub
```

---

## 🚀 Execution Guide

Evolutioner MK1 automatically fetches required model weights from the `wallpillar-lm` Hugging Face organization upon launch.

### Standard Interactive Prompt

```bash
python main.py --prompt "Solve the equation 2x + 15 = 45 and check the root."
```

### Executing Specific Model Variants

```bash
# Fast 1.5B Rollout Engine
python main.py --variant 1.5B --prompt "Factorize x^2 - 5x + 6"

# 3B Primary Workhorse Engine
python main.py --variant 3B --threads 4 --ctx-size 2048 --enable-sandbox

# 5B High-Capacity Engine
python main.py --variant 5B --threads 4 --ctx-size 4096
```

---

## 🏋️ Fine-Tuning Pipeline (Cloud / GPU Only)

Local execution is strictly built for inference and MCTS search. Training (SFT and GRPO) must be run on GPU nodes.

```bash
# Inside a GPU environment (Google Colab / RunPod)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" trl peft vllm

# Phase 1: Cold Start SFT
python training/phase1_sft.py

# Phase 2: RLVR via GRPO
python training/phase2_grpo.py
```

---

## 🗺️ Roadmap & Future Aims

Evolutioner MK1 is evolving toward a fully autonomous, local-first hybrid reasoning system. Key future milestones include:

- [ ] **Native Ternary / BitNet CPU Acceleration:** Integrate native 1.58-bit kernel execution to achieve 100+ tokens/sec on AVX2 hardware without floating-point overhead.
- [ ] **Tree-of-Thoughts (ToT) Visualizer:** Build a lightweight CLI/TUI dashboard to monitor real-time MCTS branch expansions, step scores, and verifier feedback loops.
- [ ] **Multi-Modal Verifier Sandboxing:** Extend RLVR checks beyond math/code to include local web-browsing assertions and multi-file code workspace validation.
- [ ] **Autonomous Edge Federation:** Implement peer-to-peer tree search where multiple low-power local devices evaluate sub-branches of the MCTS tree in parallel.

---

## 📁 Repository Structure

```
evolutioner-mk1/
├── assets/                  # Architecture diagrams and schema illustrations
├── config/                  # System prompts and search hyperparameter configs
│   └── default_config.json
├── models/                  # Downloaded GGUF weights (Git ignored)
├── src/
│   ├── engine.py            # llama-cpp-python interface and KV-cache manager
│   ├── router.py            # Local entropy router & API fallback engine
│   ├── search.py            # MCTS search implementation & PRM scoring
│   └── verifiers/
│       ├── ast_verifier.py  # Python AST syntax & safety inspector
│       └── sympy_verifier.py# Symbolic math execution sandbox
├── training/
│   ├── phase1_sft.py        # Unsloth SFT script for GPU training
│   └── phase2_grpo.py       # GRPO reinforcement learning pipeline
├── .gitignore
├── LICENSE
├── main.py                  # Primary CLI entrypoint
└── README.md
```

---

## 📄 License

This project is licensed under the **Apache 2.0 License** - see the [LICENSE](LICENSE) file for details.
---

## 📦 Model Inventory (local Ollama store — NOT published, reference only)

The portfolio models live in your **local Ollama store** (~20 GB total) and are
**deliberately not published** with this repo. The tag resolver (below) maps
canonical portfolio names onto whatever tags you actually have. Current local
inventory on the reference machine:

| Local Ollama tag | Size | Role |
| --- | --- | --- |
| `wallpillar-lm/evolu-general-1.5B` | 1.1 GB | Ultra-fast MCTS rollouts (1.5B tier) |
| `wallpillar-lm/evolu-general-2B` | 1.6 GB | Instruction / structured tasks (2B tier) |
| `wallpillar-lm/evolu-general-3B` | 1.9 GB | **Primary reasoner** (3B tier, default) |
| `wallpillar-lm/evolu-general-5B` | 4.7 GB | Deep reasoning (5B tier) |
| `wallpillar-lm/evolu-coder-1.5B` | 986 MB | Code-specialized rollouts (1.5B) |
| `wallpillar-lm/evolu-coder-3B` | 1.9 GB | Code-specialized reasoner (3B) |
| `wallpillar-lm/evolu-uncens-3B` | 2.2 GB | Uncensored variant (3B) |
| `wallpillar-lm/evolu-uncens-r1-1.5B` | 3.6 GB | Uncensored R1-distill (1.5B) |
| `wallpillar-lm/evolu-integrated-3B` | 1.9 GB | **Methodology-integrated** build (Modelfile) |

> These were created locally via `ollama create` from local GGUFs; none are on
> registry.ollama.ai or Hugging Face. Recreate yours with `get.sh` (auto-creates
> the integrated build) or launcher menu 8 → 2 (GGUF import).

## 🔌 Local Setup Addendum (this checkout)

- **Model tag resolver** — models installed locally under different tags
  (e.g. `wallpillar-lm/evolu-general-3B` instead of `evolutionermk1-3B`)
  are matched automatically by the backend (size token + variant keyword);
  no config edits needed when tags change.
- **Chat REPL (opencode-style)** — `PYTHONPATH=src env/bin/python -m evolutioner.cli chat`:
  streaming token output, `/model 1.5B|2B|3B|5B` switch, `/new` reset,
  `/solve <q>` escalation to the full MCTS + RLVR-verified harness.
- **Launcher** — `bash launch.sh`: menu 1–9 (TUI · query · agent · status ·
  usage · selftest · tests · ollama controls · chat REPL), with GGUF import
  for locally created models.
- **Integrated model** — `Modelfile` bakes the methodology into the 3B tier:
  `ollama create evolutionermk1-integrated-3B -f Modelfile`.
- **Phase-2 distillation pipeline** — see `docs/DISTILL.md`
  (trajectory capture → SFT dataset → QLoRA on free GPU → republish).
