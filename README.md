# Nemotron Reasoning Challenge

Our entry for the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).

The task ("Alice's Wonderland" rule induction): each prompt gives a few
`input -> output` examples of a hidden rule across six problem types
(bit manipulation, physics gravity, unit conversion, numeral system,
symbol transform, encryption cipher); the model must infer the rule and answer
the final query. We submit a **rank-32 LoRA adapter** that the host loads onto
**NVIDIA-Nemotron-3-Nano-30B-A3B** and scores with greedy vLLM inference
(`temperature=0`, pass@1, answer read from `\boxed{}`).

## Our approach: a solver-distilled curriculum

Instead of forking a public adapter, we **distill deterministic solvers into
training data**. `solvers.py` reproduces five of the six rule types exactly; for
every real train row the solver can reproduce, we emit a *verified* training
record. We add templated synthetic data for coverage, then independently
re-verify every record. This is our original contribution — the data, not the
weights.

The result is fine-tuned into the 30B with LoRA on Kaggle.

## Pipeline

```
                 +-- build_corpus.py --> real_curriculum.jsonl  --+
 train.csv ------|                                                 |--> curate.py --> curated_curriculum.jsonl
                 +-- synth.py --------> synth_curriculum.jsonl  --+        (round-trip re-verify, dedupe, balance)
                                                                                   |
                                                          build_thinking_curriculum.py
                                                                                   |
                                                                                   v
                                                                   curriculum_thinking.jsonl
                                                                   (<think>...</think> + \boxed{} — matches the host)
```

Validate the data build locally with one command (optional — the notebook
rebuilds it on Kaggle anyway):

```bash
python run_pipeline.py        # train.csv -> curriculum_thinking.jsonl
```

Then push the self-contained notebook and run it on Kaggle:

```bash
# 1. Push the notebook (uploads kaggle_remote/ to Kaggle and queues a run)
kaggle kernels push -p kaggle_remote

# 2. Poll until status changes from "running" to "complete"
kaggle kernels status nphuong302/nemotron-challenge-solver-distilled-lora

# 3. Download the output (submission.zip lands in the current directory)
kaggle kernels output nphuong302/nemotron-challenge-solver-distilled-lora -p .

# 4. Submit
kaggle competitions submit -c nvidia-nemotron-model-reasoning-challenge \
  -f submission.zip -m "solver-distilled LoRA r32 all-linear 450 steps"
```

The notebook embeds the pipeline `.py` files and rebuilds the curriculum from the
competition `train.csv` on Kaggle, so there is **no separate dataset to upload**.

## Layout

| Path | What it is |
|---|---|
| `solvers.py` | Deterministic solvers for the rule types + `detect_type` / `solve`. The *teachers*. |
| `makers.py` | Turn a solved real row into verified chat records (answer-only + reasoning trace). |
| `build_corpus.py` | Solve every `train.csv` row, keep gold-matching ones → `real_curriculum.jsonl`. |
| `synth.py` | Templated synthetic examples for extra coverage → `synth_curriculum.jsonl`. |
| `curate.py` | Re-verify, dedupe, length-cap, balance buckets → `curated_curriculum.jsonl`. |
| `build_thinking_curriculum.py` | Wrap each target as `<think>...</think>\n\n\boxed{X}` to match the host's chat template → `curriculum_thinking.jsonl`. |
| `coverage.py` | Report solver coverage per type on `train.csv`. |
| `run_pipeline.py` | Run all four data stages in order (local validation). |
| `kaggle_remote/` | The self-contained Kaggle notebook + its `kernel-metadata.json` (RTX 6000, offline, BF16 LoRA r32, packages `submission.zip`). |
| `TRAINING_OBSERVABILITY.md` | Design + status of the in-notebook training metrics (held-out eval-loss + per-bucket task-accuracy probe). |
| `BUILD.md` | Running handoff notes — read first when resuming work (current run state + open decisions). |
| `tests/` | Data-free deterministic tests for the solver/maker logic (run in CI). |

## Results

| Metric | Value |
|---|---|
| Final leaderboard score | **0.588** |
| Rank | 3571 / 4182 |
| Approach | Solver-distilled SFT, LoRA rank-32, 450 steps on RTX 6000 Pro |


## Setup

```bash
uv sync                       # create .venv and install deps from pyproject + uv.lock
cp .env.example .env          # then fill in KAGGLE_USERNAME (and tokens if needed)
```

## Key facts

- **Scored model:** NVIDIA-Nemotron-3-Nano-30B-A3B (hybrid Mamba-Transformer MoE), not the 4B.
- **Inference (host):** greedy, pass@1, reasoning ("thinking") mode **on** by default; answer from `\boxed{}`.
