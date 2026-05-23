"""
Build val_problems.jsonl from train.csv rows that were NOT used in SFT training.

Training used the first MAX_TRAIN=500 rows, so we sample from row 500 onward.
We take N_PER_TYPE examples from each active problem type.
"""

import csv
import json
import random
from pathlib import Path

DATA_CSV     = Path("data/nvidia-nemotron-model-reasoning-challenge/train.csv")
OUT_PATH     = Path("val_problems.jsonl")
TRAIN_CUTOFF = 500       # rows 0..499 were used for SFT corpus — skip them
N_PER_TYPE   = 30        # validation examples per active type
ACTIVE_TYPES = {"gravity", "unit", "numeral"}
SEED         = 42


def detect_type(prompt: str) -> str:
    p = prompt.lower()
    if "bit manipulation" in p:            return "bit"
    if "gravitational" in p:               return "gravity"
    if "unit conversion" in p:             return "unit"
    if "numeral system" in p:              return "numeral"
    if "transformation rule" in p:         return "symbol"
    if "encryption" in p or "cipher" in p: return "cipher"
    return "unknown"


def main():
    random.seed(SEED)

    with open(DATA_CSV) as f:
        rows = list(csv.DictReader(f))

    # only rows the model never trained on
    holdout = rows[TRAIN_CUTOFF:]

    by_type: dict[str, list] = {t: [] for t in ACTIVE_TYPES}
    for row in holdout:
        ptype = detect_type(row["prompt"])
        if ptype in ACTIVE_TYPES:
            by_type[ptype].append(row)

    val_problems = []
    for ptype, pool in by_type.items():
        sample = random.sample(pool, min(N_PER_TYPE, len(pool)))
        for row in sample:
            val_problems.append({
                "id":     row["id"],
                "type":   ptype,
                "prompt": row["prompt"],
                "answer": row["answer"],
            })
        print(f"  {ptype}: {len(sample)} problems sampled (pool={len(pool)})")

    random.shuffle(val_problems)

    with open(OUT_PATH, "w") as f:
        for p in val_problems:
            f.write(json.dumps(p) + "\n")

    print(f"\nWrote {len(val_problems)} problems to {OUT_PATH}")


if __name__ == "__main__":
    main()
