# NVIDIA Nemotron Model Reasoning Challenge

## What is this competition?

Improve **Nemotron 3 Nano**'s math/reasoning accuracy on a novel benchmark.
Hosted by NVIDIA + Kaggle, using Google Cloud G4 (NVIDIA L4) infrastructure.
Prize pool: **$100K+**

## Task

- Input: math/reasoning problems (olympiad-level, similar to AMC/AIME/AIMO)
- Output: final numeric or symbolic answer (ideally inside `\boxed{}`)
- Model you MUST use: `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (can be fine-tuned)

## Evaluation Metric

- **pass@1**: accuracy on first generation sample
- **cons@k (majority vote)**: generate k answers, take the most common one
  - Higher k = higher accuracy but slower
  - Typical values: cons@8, cons@64

## Allowed Techniques

| Technique | Description |
|---|---|
| Prompting | System prompt engineering, chain-of-thought |
| Synthetic data | Generate (problem, solution) pairs with a stronger teacher model |
| Data curation | Filter synthetic data to keep only correct solutions |
| Fine-tuning (SFT) | Supervised fine-tuning with LoRA adapters |

## Workflow

```
problems.jsonl
    --> 01_inference.py      (baseline: see what the raw model scores)
    --> 02_augmentation.py   (generate synthetic correct solutions)
    --> corpus.jsonl
    --> 03_train_sft.py      (fine-tune with LoRA)
    --> adapter/
    --> 04_evaluate.py       (measure pass@1 and cons@k)
    --> 05_kaggle_submission.py  (write submission.csv on Kaggle)
```

## Key Tips from Progress Prize Winner

1. **Filter corpus strictly** — only train on solutions where `\boxed{}` matches ground truth.
2. **LoRA rank 16** is a good starting point (r=16, alpha=32).
3. **Multiple samples + majority vote** is the single highest-leverage trick.
4. Use `temperature=0.6–0.8` for sampling diversity.
5. The reasoning system prompt `"detailed thinking on"` enables chain-of-thought.

## Hardware (Kaggle G4)

- GPU: NVIDIA L4 (24 GB VRAM)
- Nemotron 3 Nano 4B fits in BF16 without quantization
- For training, gradient checkpointing + LoRA keeps VRAM under budget

## Files

| File | Purpose |
|---|---|
| `01_inference.py` | Baseline inference, no fine-tuning |
| `02_augmentation.py` | Synthetic data generation |
| `03_train_sft.py` | LoRA SFT training |
| `04_evaluate.py` | pass@1 and cons@k evaluation |
| `05_kaggle_submission.py` | Kaggle notebook entry point |
| `requirements.txt` | Python dependencies |

## Getting Started

```bash
pip install -r requirements.txt

# 1. Run baseline
python 01_inference.py

# 2. Generate training data
python 02_augmentation.py

# 3. Fine-tune
python 03_train_sft.py

# 4. Evaluate
python 04_evaluate.py --adapter adapter/ --problems val_problems.jsonl

# 5. Submit on Kaggle (run 05_kaggle_submission.py as a notebook)
```

## Resources

- Competition: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
- Model card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
- Progress prize winner repo: https://github.com/tonghuikang/nemotron
- NVIDIA Nemotron developer hub: https://developer.nvidia.com/nemotron
