"""
REAL curriculum builder.
Solves each train row with our verified CoT makers and emits training-ready
chat records. Every emitted example is VERIFIED (computed answer == gold), so
the model only ever sees correct targets.

For each verified row we emit two reasoning records (never answer-only, so the
model always learns to think before boxing):
  - real_trace   : the full verified CoT + box    (the complete reasoning path)
  - real_concise : rule line + result line + box  (the same reasoning, short form)

Output: real_curriculum.jsonl  (messages format, ready for SFT)
"""
import csv, json
from pathlib import Path
from collections import defaultdict

import makers

DATA = next(p for p in [Path("data/train.csv"),
            Path("data/nvidia-nemotron-model-reasoning-challenge/train.csv")] if p.exists())
OUT = Path("real_curriculum.jsonl")
SUFFIX = "Please put your final answer inside \\boxed{}."


def make_record(rid, bucket, source, prompt, response):
    return {
        "id": rid,
        "bucket": bucket,
        "source": source,
        "messages": [
            {"role": "user", "content": prompt.strip() + "\n" + SUFFIX},
            {"role": "assistant", "content": response},
        ],
    }


def concise_trace(trace):
    """Reduce a full CoT to rule + result: the first (rule-identifying) line and
    the last (application) line. Keeps real reasoning in short form so the
    record is never answer-only."""
    lines = [ln.strip() for ln in trace.splitlines() if ln.strip()]
    if len(lines) <= 2:
        return trace.strip()
    return lines[0] + "\n" + lines[-1]


def main():
    rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
    records = []
    verified = 0
    weak_bit = 0
    by_bucket = defaultdict(int)

    for row in rows:
        t = makers.detect_type(row["prompt"])
        if t not in makers.MAKERS:
            continue
        res = makers.MAKERS[t](row["prompt"])
        if res is None:
            continue
        trace, ans = res
        if not makers.answers_match(ans, row["answer"], t):
            continue
        # Drop bit rows whose only trace is the weak per-bit-boolean table-dump
        # (no clean op recovered): it teaches no transferable method. Better to
        # learn bit from the strong single/two-op reals + synthetic coverage.
        if t == "bit" and "Per-bit boolean function:" in trace:
            weak_bit += 1
            continue

        verified += 1
        by_bucket[t] += 1
        gold = row["answer"].strip()
        box = f"\\boxed{{{gold}}}"
        full = trace.strip() + f"\n\nFinal answer: {box}"
        records.append(make_record(f"{row['id']}:trace", t, "real_trace",
                                    row["prompt"], full))
        concise = concise_trace(trace).strip() + f"\n\nFinal answer: {box}"
        records.append(make_record(f"{row['id']}:concise", t, "real_concise",
                                    row["prompt"], concise))

    with open(OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"verified rows: {verified}  ->  {len(records)} records  -> {OUT}")
    print(f"dropped weak-bit (per-bit boolean only): {weak_bit}")
    for t in ["bit", "gravity", "unit", "numeral", "cipher", "symbol"]:
        print(f"  {t:<9} {by_bucket[t]}")


if __name__ == "__main__":
    main()
