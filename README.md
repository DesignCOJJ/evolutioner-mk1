# Evolutioner MK1 🧠⚡

> **Open-Source Evolutionary Search Harness, Verification Engine, and 3B Parameter Reasoning Framework.**

Evolutioner MK1 is a modular reasoning harness designed to scale small-footprint language models ($\le$ 3B parameters) into high-capacity reasoning engines. By shifting the workload from pure parametric memory to **Test-Time Compute (TTC) scaling**, **deterministic rule-based verifiers (RLVR)**, and **external stateful memory**, Evolutioner MK1 enables edge-class hardware to execute complex, step-by-step reasoning workflows with zero hallucination on verifiable tasks.

---

## 📐 System Architecture

Evolutioner MK1 decouples heavy reasoning from weight-based generation using a 4-pillar architectural pipeline:

```
                              ┌──────────────────────────────────┐
                              │          Problem Input           │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  Base Rollout Engine (GGUF CPU)  │
                              │       (Qwen2.5-3B-Instruct)      │
                              └────────────────┬─────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
         ┌───────────────────────────┐                   ┌───────────────────────────┐
         │ Deterministic RLVR Engine │                   │     MCTS & PRM Search     │
         ├───────────────────────────┤                   ├───────────────────────────┤
         │ • SymPy CAS / AST Check   │                   │ • Newline-boundary scoring│
         │ • Bubblewrap Sandbox      │                   │ • Dead-path pruning       │
         │ • Immediate STDOUT Feed   │                   │ • Multi-branch selection  │
         └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                       │                                               │
                       └───────────────────────┬───────────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │   Adaptive Fallback API Router   │
                              │    (Delegates to Groq / Gemini)  │
                              └──────────────────────────────────┘
```

### Core Components

1. **Quantized Inference Core:** Powered by `llama.cpp`, running standard $Q4\_K\_M$ GGUF models optimized for AVX2 execution without requiring an NVIDIA GPU.
2. **Deterministic RLVR Verifier:** Intercepts intermediate reasoning blocks inside `<think>` tags and executes dynamic Python/SymPy code in a sandboxed runtime. Standard execution outputs (or errors) are injected directly back into the active KV-cache for immediate self-correction.
3. **MCTS & Process Reward Model (PRM):** Evaluates candidate trajectories at step boundaries, pruning low-probability branches early to maximize inference efficiency.
4. **Virtual Context Engine:** Manages a rolling KV-cache window backed by dense vector retrieval to simulate extended context without exploding physical RAM usage.

---

## 💻 Hardware & System Requirements

Evolutioner MK1 is calibrated to execute on resource-constrained host machines while offloading optional heavy fine-tuning to cloud GPU environments.

### Host Runtime Bounds
* **CPU:** x86_64 CPU with AVX2 instruction support (e.g., Intel Haswell / AMD Zen 1 or newer).
* **RAM:** Minimum 8 GB DDR3/DDR4 (uses ~2.5 GB peak system memory during inference).
* **Storage:** ~10 GB free space (for Python virtual environment, dependencies, and model checkpoints).
* **OS:** Linux (Arch Linux / CachyOS / Ubuntu recommended), macOS, or WSL2 on Windows.

---

## 🛠️ Prerequisites & Setup

### 1. System Dependencies (Arch / CachyOS)

Install base build tools and virtual container sandboxing tools:

```bash
sudo pacman -Syu --needed base-devel git python python-pip python-virtualenv bubblewrap
```

*(For Ubuntu/Debian, use `sudo apt update && sudo apt install build-essential git python3 python3-pip python3-venv bubblewrap`)*

### 2. Repository Initialization

Clone the repository and set up a virtual environment using `uv` or standard Python `venv`:

```bash
git clone https://github.com/your-username/evolutioner-mk1.git
cd evolutioner-mk1

# Initialize virtual environment
python -m venv env
source env/bin/activate  # On fish: source env/bin/activate.fish

# Install dependencies
pip install --upgrade pip uv
uv pip install llama-cpp-python sympy pydantic requests huggingface_hub
```

---

## 🚀 Running Evolutioner MK1

Evolutioner MK1 automatically fetches the default target model (`Qwen2.5-3B-Instruct-GGUF`) from Hugging Face Hub on its initial launch.

### Basic Interactive Execution

Run the main reasoning loop via the CLI:

```bash
python main.py --prompt "Solve the equation 2x + 15 = 45 and check the root."
```

### Advanced CLI Flags

```bash
python main.py \
  --model "Qwen/Qwen2.5-3B-Instruct-GGUF" \
  --filename "qwen2.5-3b-instruct-q4_k_m.gguf" \
  --threads 4 \
  --ctx-size 2048 \
  --enable-sandbox
```

---

## 🏋️ Fine-Tuning Pipeline (Cloud / GPU)

While local execution is strictly optimized for CPU inference and MCTS search, model training (SFT and GRPO) should be executed in GPU environments (e.g., Google Colab, RunPod, or local NVIDIA GPUs).

Training scripts are located in the `training/` directory:

* `training/phase1_sft.py`: Cold-start Supervised Fine-Tuning using Unsloth and TRL.
* `training/phase2_grpo.py`: Group Relative Policy Optimization (GRPO) training targeting rule-based verifiers and format enforcement.

To run GRPO fine-tuning on a GPU node:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" trl peft vllm
python training/phase2_grpo.py
```

---

## 📁 Repository Structure

```text
evolutioner-mk1/
├── assets/                  # Architecture diagrams and documentation images
├── config/                  # Model configurations and system prompts
│   └── default_config.json
├── models/                  # Local GGUF model storage (Git ignored)
├── src/
│   ├── engine.py            # llama-cpp-python wrapper and KV-cache management
│   ├── router.py            # API Fallback and entropy evaluation logic
│   ├── search.py            # Monte Carlo Tree Search (MCTS) & PRM implementation
│   └── verifiers/
│       ├── ast_verifier.py  # Python AST syntax and logic inspector
│       └── sympy_verifier.py# Symbolic mathematics execution sandbox
├── training/
│   ├── phase1_sft.py        # Supervised Fine-Tuning pipeline
│   └── phase2_grpo.py       # Reinforcement learning via GRPO
├── .gitignore
├── LICENSE
├── main.py                  # Primary execution entrypoint
└── README.md
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/verifier-enhancement`).
3. Commit your changes (`git commit -m 'Add custom SymPy matrix verifier'`).
4. Push to the branch (`git push origin feature/verifier-enhancement`).
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.