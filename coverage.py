"""
True teacher-coverage using the CoT makers (which have the correct parsers
and already emit reasoning traces), uncapped over the full train set.
This is the real ceiling of verified examples we can distill.
"""
import csv
from pathlib import Path
from collections import defaultdict

import makers

DATA = next(
    (p for p in [Path("data/train.csv"),
                 Path("data/nvidia-nemotron-model-reasoning-challenge/train.csv")] if p.exists()),
    None,
)
if DATA is None:
    raise FileNotFoundError(
        "train.csv not found; place it at data/train.csv or "
        "data/nvidia-nemotron-model-reasoning-challenge/train.csv"
    )

total = defaultdict(int)
made = defaultdict(int)       # maker returned a result
correct = defaultdict(int)    # result verified against gold

rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
for row in rows:
    t = makers.detect_type(row["prompt"])
    total[t] += 1
    if t not in makers.MAKERS:
        continue
    res = makers.MAKERS[t](row["prompt"])
    if res is None:
        continue
    made[t] += 1
    _, ans = res
    if makers.answers_match(ans, row["answer"], t):
        correct[t] += 1

g_tot = sum(total.values()); g_cor = sum(correct.values())
print(f"DATA: {DATA}  ({g_tot} rows)\n")
print(f"{'type':<10}{'count':>7}{'made':>7}{'correct':>9}{'coverage':>10}")
print("-" * 43)
for t in ["bit", "gravity", "unit", "numeral", "cipher", "symbol"]:
    c, m, s = total[t], made[t], correct[t]
    cov = f"{100*s/c:.1f}%" if c else "-"
    print(f"{t:<10}{c:>7}{m:>7}{s:>9}{cov:>10}")
print("-" * 43)
print(f"{'TOTAL':<10}{g_tot:>7}{sum(made.values()):>7}{g_cor:>9}{100*g_cor/g_tot:>9.1f}%")
