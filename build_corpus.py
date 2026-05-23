"""
REAL curriculum builder.
Solves each train row with our verified CoT makers and emits training-ready
chat records. Every emitted example is VERIFIED (computed answer == gold), so
the model only ever sees correct targets.

For each verified row we emit two records:
  - real_answer : target is just \boxed{gold}      (teaches answer formatting)
  - real_trace  : target is the verified CoT + box  (teaches the reasoning path)

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


def main():
    rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
    records = []
    verified = 0
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

        verified += 1
        by_bucket[t] += 1
        gold = row["answer"].strip()
        records.append(make_record(f"{row['id']}:ans", t, "real_answer",
                                    row["prompt"], f"\\boxed{{{gold}}}"))
        full = trace.strip() + f"\n\nFinal answer: \\boxed{{{gold}}}"
        records.append(make_record(f"{row['id']}:trace", t, "real_trace",
                                    row["prompt"], full))

    with open(OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"verified rows: {verified}  ->  {len(records)} records  -> {OUT}")
    for t in ["bit", "gravity", "unit", "numeral", "cipher", "symbol"]:
        print(f"  {t:<9} {by_bucket[t]}")


if __name__ == "__main__":
    main()
