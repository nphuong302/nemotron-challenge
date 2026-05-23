"""
DATA CURATION + assembly.
Takes the raw real + synthetic sets and produces a clean, accurate, training-ready
curriculum. Every surviving record passes the SAME accuracy bar.

Stages (each reports how many it drops):
  1. format validation     - non-empty turns, parseable \boxed{}
  2. trace-quality filter   - drop bit "per-bit boolean" table-dump traces
  3. round-trip re-verify   - re-solve the prompt independently; boxed answer must match
  4. dedupe                 - exact (prompt,response); cap identical synthetic prompts
  5. length cap             - drop records longer than the training context allows
  6. assemble               - oversample hard buckets (bit, cipher), shuffle

Input : real_curriculum.jsonl, synth_curriculum.jsonl
Output: curated_curriculum.jsonl
"""
import json, re, random
from pathlib import Path
from collections import Counter, defaultdict

import makers
from solvers import solve_bit  # stronger bit coverage than the maker

REAL = Path("real_curriculum.jsonl")
SYNTH = Path("synth_curriculum.jsonl")
OUT = Path("curated_curriculum.jsonl")

CHAR_CAP = 20000             # ~5k tokens, safely under the 6144 train context
MAX_SYNTH_PROMPT_DUPES = 1   # keep at most N synthetic records per identical prompt
HARD_OVERSAMPLE = {"bit": 2, "cipher": 2}
SEED = 3407

BOX = re.compile(r"\\boxed\{([^}]*)\}")


def boxed(text):
    m = BOX.findall(text or "")
    return m[-1].strip() if m else None


def user_prompt(rec):
    return rec["messages"][0]["content"]


def base_prompt(rec):
    # strip the trailing boxed-answer suffix to recover the raw problem text
    return user_prompt(rec).rsplit("\nPlease put your final answer", 1)[0]


# ---- stage 3: independent re-solver (cached per raw prompt) ----
_cache = {}

def resolve_answer(rec):
    bp = base_prompt(rec)
    if bp in _cache:
        return _cache[bp]
    t = makers.detect_type(bp)
    ans = None
    try:
        if t == "bit":
            ans = solve_bit(bp)
        elif t in makers.MAKERS:
            res = makers.MAKERS[t](bp)
            ans = res[1] if res else None
    except Exception:
        ans = None
    _cache[bp] = (t, ans)
    return t, ans


def main():
    real = [json.loads(l) for l in REAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    synth = [json.loads(l) for l in SYNTH.read_text(encoding="utf-8").splitlines() if l.strip()]
    drops = Counter()
    kept = []

    for rec, is_synth in [(r, False) for r in real] + [(r, True) for r in synth]:
        u = user_prompt(rec); a = rec["messages"][1]["content"]
        # 1. format
        if not u.strip() or not a.strip():
            drops["empty"] += 1; continue
        ans = boxed(a)
        if ans is None:
            drops["no_box"] += 1; continue
        # 2. trace quality
        if rec["bucket"] == "bit" and "Per-bit boolean function:" in a:
            drops["weak_bit_trace"] += 1; continue
        # 3. round-trip verify
        t, recovered = resolve_answer(rec)
        if recovered is None or not makers.answers_match(recovered, ans, t):
            drops["reverify_fail"] += 1; continue
        # 5. length (do cheap char check here)
        if len(u) + len(a) > CHAR_CAP:
            drops["too_long"] += 1; continue
        kept.append(rec)

    # 4. dedupe exact (prompt,response) + cap synthetic prompt repeats
    seen_pair, synth_prompt_count, deduped = set(), defaultdict(int), []
    for rec in kept:
        key = (user_prompt(rec), rec["messages"][1]["content"])
        if key in seen_pair:
            drops["dup_exact"] += 1; continue
        seen_pair.add(key)
        if rec["source"] == "synthetic_trace":
            p = user_prompt(rec)
            if synth_prompt_count[p] >= MAX_SYNTH_PROMPT_DUPES:
                drops["dup_synth_prompt"] += 1; continue
            synth_prompt_count[p] += 1
        deduped.append(rec)

    # 6. assemble: oversample hard real buckets, then shuffle
    rng = random.Random(SEED)
    final = []
    for rec in deduped:
        reps = HARD_OVERSAMPLE.get(rec["bucket"], 1) if rec["source"].startswith("real") else 1
        final.extend([rec] * reps)
    rng.shuffle(final)

    with open(OUT, "w", encoding="utf-8") as f:
        for rec in final:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- report ----
    print(f"loaded:   real={len(real)}  synth={len(synth)}  total={len(real)+len(synth)}")
    print(f"dropped:  {dict(drops)}")
    print(f"after curation+dedupe: {len(deduped)}")
    print(f"after oversample (final): {len(final)}  -> {OUT}\n")
    print("final by source:", dict(Counter(r["source"] for r in final)))
    print("final by bucket:", dict(Counter(r["bucket"] for r in final)))
    chars = sum(len(user_prompt(r)) + len(r["messages"][1]["content"]) for r in final)
    print(f"~chars: {chars:,}  (~{chars//4:,} tokens)")


if __name__ == "__main__":
    main()
