"""
curriculum assembler.
Combines the verified REAL set with the SYNTHETIC set into one shuffled SFT file.

Mixing logic (mirrors the failure-mining idea, bucket-level since we have no
base-model predictions yet):
  - real examples from hard buckets (bit, cipher) are oversampled
  - synthetic examples are included once each
Output: final_curriculum.jsonl  (+ prints composition and a token estimate)
"""
import json, random
from pathlib import Path
from collections import Counter

REAL = Path("real_curriculum.jsonl")
SYNTH = Path("synth_curriculum.jsonl")
OUT = Path("final_curriculum.jsonl")

HARD_OVERSAMPLE = {"bit": 2, "cipher": 2}   # real examples from these buckets repeated Nx
SEED = 3407


def load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    rng = random.Random(SEED)
    real = load(REAL) if REAL.exists() else []
    synth = load(SYNTH) if SYNTH.exists() else []

    final = []
    for r in real:
        reps = HARD_OVERSAMPLE.get(r["bucket"], 1)
        final.extend([r] * reps)
    final.extend(synth)
    rng.shuffle(final)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # report
    by_source = Counter(r["source"] for r in final)
    by_bucket = Counter(r["bucket"] for r in final)
    chars = sum(len(r["messages"][0]["content"]) + len(r["messages"][1]["content"]) for r in final)
    print(f"REAL records:  {len(real)}")
    print(f"SYNTH records: {len(synth)}")
    print(f"FINAL records: {len(final)}  -> {OUT}")
    print(f"\nby source:  {dict(by_source)}")
    print(f"by bucket:  {dict(by_bucket)}")
    print(f"\n~chars: {chars:,}  (~{chars//4:,} tokens, ~{chars//4//1000}k)")


if __name__ == "__main__":
    main()
