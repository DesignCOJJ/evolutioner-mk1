# Phase 2 — Distilling the Evolutioner methodology (LoRA)

Phase 1 (shipped) bakes the methodology into a model with a Modelfile system
prompt (`Modelfile` → `evolutionermk1-integrated-3B`). Phase 2 goes deeper:
train the weights themselves on trajectories produced by the full harness,
so the model behaves like Evolutioner *without* the prompt.

## Why not on this machine

| Requirement | This machine (i5-4440, 4 threads, 7 GB RAM) | Needed for 3B LoRA |
| --- | --- | --- |
| RAM | 7 GB shared | 12–16 GB (Q8 + optimizer states) |
| Throughput | ~1 tok/s on 3B | thousands of steps × 512 tok |

Verdict: **not feasible locally.** Use a free/cheap GPU (Colab T4 works for a
3B QLoRA) or rent by the hour. Everything below is scripted so the GPU step is
one cell.

## Step 1 — Capture trajectories (any machine, slow is fine)

The harness already emits machine-readable transcripts:

```bash
# a few dozen problems of graded difficulty; --json gives the full trace
for q in $(cat data/distill_prompts.txt); do
  PYTHONPATH=src env/bin/python -m evolutioner.cli query "$q" --json \
    >> data/trajectories.jsonl
done
```

Each line contains the `<think>` MCTS trace, the emitted ```python block,
the sandbox verdict, and the self-correction rounds — exactly the behavior we
want the model to internalize.

Build `data/distill_prompts.txt` from: arithmetic/algebra identities, string
manipulation, unit conversions, and anything a 5-second sandbox can verify.
Aim for 200–500 prompts; quality of the verifier loop matters more than volume.

## Step 2 — Format as supervised fine-tune data

Convert each trajectory into one training example:

- **prompt:** the raw user question
- **completion:** `<think>…</think>` + verified ```python block + `Final Answer: …`
- **drop** failed runs (never train on unverified output — that inverts RLVR)

A converter script belongs here (`tools/build_sft_dataset.py`, phase 2 TODO):
read `trajectories.jsonl`, filter on `verification == OK`, emit
`{"messages": [...]}` chat-format JSONL.

## Step 3 — QLoRA on a free GPU (Colab T4, ~2–4 h for 500 examples)

```python
# Colab cell — pip install -U unsloth
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    "wallpillar-lm/evolutionermk1-3B", load_in_4bit=True, max_seq_length=2048)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"])
model.train_on_dataset(...)          # your SFT JSONL from step 2
model.save_pretrained_gguf("out", tok, quantization_method="q4_k_m")
```

(Any trainer works — axolotl, peft+trl — as long as it reads chat-format JSONL
and trains on the completion only.)

## Step 4 — Reintegrate and publish

```bash
# local: wrap the adapter/base GGUF with the same Modelfile methodology
ollama create wallpillar-lm/evolutionermk1-distilled-3B -f Modelfile.distilled
ollama push wallpillar-lm/evolutionermk1-distilled-3B
```

`Modelfile.distilled` = same SYSTEM block (kept — it acts as the policy even
after training), but `FROM ./out/model-q4_k_m.gguf`.

## Acceptance checks

1. `ollama run wallpillar-lm/evolutionermk1-distilled-3B` on held-out problems:
   produces `<think>` + verification block with **no** prompt engineering.
2. Distilled model beats `evolutionermk1-integrated-3B` on first-try sandbox
   pass rate (compare via `python -m evolutioner.cli usage` call stats).
3. `</think>` stop behavior preserved (see README §9, deviation 3).
