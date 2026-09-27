Evolutioner MK1 🧠⚡

Open-Source Evolutionary Search Harness, Verification Engine, and 3B Parameter Reasoning Framework.

Evolutioner MK1 is a modular reasoning harness designed to scale small-footprint language models ($\le$ 5B parameters) into high-capacity reasoning engines. By shifting the workload from pure parametric memory to Test-Time Compute (TTC) scaling, deterministic rule-based verifiers (RLVR), and external stateful memory, Evolutioner MK1 enables edge-class hardware to execute complex, step-by-step reasoning workflows with zero hallucination on verifiable tasks.

🏛️ Official Model Portfolio (wallpillar-lm)

Evolutioner MK1 supports a tiered family of fine-tuned and quantized models hosted under the official wallpillar-lm Hugging Face organization:

Model ID & Hugging Face Repo

Base Architecture

Parameter Footprint

Specialized Operational Target

wallpillar-lm/evolutionermk1-1.5B

DeepSeek-R1-Distill-1.5B

1.5 Billion

Ultra-Fast Rollouts: Rapid `` candidate generation for high-depth MCTS tree exploration and instant verification.

wallpillar-lm/evolutionermk1-2B

Gemma-2-2B-Instruct

2.0 Billion

Instruction & Structured Tasks: Multi-turn conversational workflows, schema enforcement, and JSON outputs.

wallpillar-lm/evolutionermk1-3B

Qwen-2.5-3B-Instruct

3.0 Billion

Primary Local Workhorse: Math/SymPy execution loops, Python AST checking, and balanced logic-to-RAM efficiency.

wallpillar-lm/evolutionermk1-5B

DeepSeek-R1-Distill-7B

~5–7 Billion

High-Capacity Engine: Deep multi-step reasoning and complex code synthesis for offloaded or high-memory runs.

📐 System Architecture

Evolutioner MK1 decouples heavy reasoning from weight-based generation using a 4-pillar architectural pipeline:

                              ┌──────────────────────────────────┐
                              │          Problem Input           │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  Base Rollout Engine (GGUF CPU)  │
                              │    (wallpillar-lm Model Family)  │
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


Core Components

Quantized Inference Core: Powered by llama.cpp, running standard $Q4\_K\_M$ GGUF models from wallpillar-lm optimized for AVX2 execution without requiring an NVIDIA GPU.

Deterministic RLVR Verifier: Intercepts intermediate reasoning blocks inside `` tags and executes dynamic Python/SymPy code in a sandboxed runtime. Standard execution outputs (or errors) are injected directly back into the active KV-cache for immediate self-correction.

MCTS & Process Reward Model (PRM): Evaluates candidate trajectories at step boundaries, pruning low-probability branches early to maximize inference efficiency.

Virtual Context Engine: Manages a rolling KV-cache window backed by dense vector retrieval to simulate extended context without exploding physical RAM usage.

💻 Hardware & System Requirements

Evolutioner MK1 is calibrated to execute on resource-constrained host machines while offloading optional heavy fine-tuning to cloud GPU environments.

Host Runtime Bounds

CPU: x86_64 CPU with AVX2 instruction support (e.g., Intel Haswell / AMD Zen 1 or newer).

RAM: Minimum 8 GB DDR3/DDR4 (uses ~1.5 GB to 3.5 GB peak system memory during inference depending on variant).

Storage: ~10 GB free space (for Python virtual environment, dependencies, and cached model checkpoints).

OS: Linux (Arch Linux / CachyOS / Ubuntu recommended), macOS, or WSL2 on Windows.

🛠️ Prerequisites & Setup

1. System Dependencies (Arch / CachyOS)

Install base build tools and virtual container sandboxing tools:

sudo pacman -Syu --needed base-devel git python python-pip python-virtualenv bubblewrap


(For Ubuntu/Debian, use sudo apt update && sudo apt install build-essential git python3 python3-pip python3-venv bubblewrap)

2. Repository Initialization

Clone the repository and set up a virtual environment using uv or standard Python venv:

git clone https://github.com/your-username/evolutioner-mk1.git
cd evolutioner-mk1

# Initialize virtual environment
python -m venv env
source env/bin/activate  # On fish: source env/bin/activate.fish

# Install dependencies
pip install --upgrade pip uv
uv pip install llama-cpp-python sympy pydantic requests huggingface_hub


🚀 Running Evolutioner MK1

Evolutioner MK1 automatically fetches the default target model (wallpillar-lm/evolutionermk1-3B) from Hugging Face Hub on its initial launch.

Basic Interactive Execution

Run the main reasoning loop via the CLI:

python main.py --prompt "Solve the equation 2x + 15 = 45 and check the root."


Specifying Model Variants

Select any model variant from the wallpillar-lm portfolio using the --variant flag:

# Run with 1.5B Ultra-Fast Rollout Engine
python main.py --variant 1.5B --prompt "Factorize x^2 - 5x + 6"

# Run with 3B Workhorse Engine
python main.py --variant 3B --threads 4 --ctx-size 2048 --enable-sandbox

# Run with 5B High-Capacity Engine
python main.py --variant 5B --threads 4 --ctx-size 4096


🏋️ Fine-Tuning Pipeline (Cloud / GPU)

While local execution is strictly optimized for CPU inference and MCTS search, model training (SFT and GRPO) should be executed in GPU environments (e.g., Google Colab, RunPod, or local NVIDIA GPUs).

Training scripts are located in the training/ directory:

training/phase1_sft.py: Cold-start Supervised Fine-Tuning using Unsloth and TRL.

training/phase2_grpo.py: Group Relative Policy Optimization (GRPO) training targeting rule-based verifiers and format enforcement.

To run GRPO fine-tuning on a GPU node:

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" trl peft vllm
python training/phase2_grpo.py


📁 Repository Structure

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


🤝 Contributing

Contributions are welcome! Please follow these steps:

Fork the repository.

Create a feature branch (git checkout -b feature/verifier-enhancement).

Commit your changes (git commit -m 'Add custom SymPy matrix verifier').

Push to the branch (git push origin feature/verifier-enhancement).

Open a Pull Request.

📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
