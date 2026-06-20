"""
SYNTHETIC generator.
We control the rule, so every generated example is verifiable and ships with a
correct reasoning trace. Prompts mirror the real train templates exactly so the
model sees consistent formatting at train/test time.

Buckets generated: bit, gravity, unit, numeral, cipher.
(symbol is intentionally omitted: real symbol_transform is not a clean per-symbol
substitution, so synthetic symbol would not match the hidden rule family.)

Weighted toward the failure-prone buckets (cipher especially) by default.
Output: synth_curriculum.jsonl
"""
import json, random, string
from pathlib import Path

from solvers import _apply_bit_op  # reuse our verified bit primitive

SUFFIX = "Please put your final answer inside \\boxed{}."
ALPHABET = string.ascii_lowercase
WORD_POOL = [
    "queen", "dragon", "castle", "wizard", "forest", "silver", "mirror", "palace",
    "secret", "garden", "bridge", "market", "planet", "rocket", "harbor", "magnet",
    "window", "stone", "cable", "river", "knight", "shadow", "flame", "jewel",
    "puzzle", "voyage", "quartz", "frozen", "bishop", "candle", "dwarf", "lucky",
    "myth", "vex", "jazz", "box", "fjord", "glyph",
]


def _box(ans):
    return f"\\boxed{{{ans}}}"


def make_record(bucket, prompt, trace, answer, idx):
    full = trace.strip() + f"\n\nFinal answer: {_box(answer)}"
    return {
        "id": f"synth:{bucket}:{idx}",
        "bucket": bucket,
        "source": "synthetic_trace",
        "messages": [
            {"role": "user", "content": prompt.strip() + "\n" + SUFFIX},
            {"role": "assistant", "content": full},
        ],
    }


# ── bit ─────────────────────────────────────────────────────────────────────
BIT_HEADER = ("In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit "
              "binary numbers. The transformation involves operations like bit shifts, "
              "rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.")

def _bit_prompt(ex, q):
    lines = [BIT_HEADER, "", "Here are some examples of input -> output:"]
    lines += [f"{i:08b} -> {o:08b}" for i, o in ex]
    lines += ["", f"Now, determine the output for: {q:08b}"]
    return "\n".join(lines)


def gen_bit(rng):
    # ~30% two-step rules so the model also learns composed transforms (these
    # are recoverable by solve_bit's _try_two_ops, so curate keeps them).
    if rng.random() < 0.3:
        return _gen_bit_two_step(rng)
    rules = (
        [("NOT", 0, "invert every bit (NOT)")]
        + [("REV", 0, "reverse the bit order")]
        + [("ROL", k, f"rotate left by {k}") for k in (1, 2, 3)]
        + [("ROR", k, f"rotate right by {k}") for k in (1, 2, 3)]
        + [("XOR", m, f"XOR with mask {m:08b}") for m in (rng.randint(1, 255),)]
        + [("AND", m, f"AND with mask {m:08b}") for m in (rng.randint(1, 255),)]
        + [("OR", m, f"OR with mask {m:08b}") for m in (rng.randint(1, 255),)]
        + [("SHL", k, f"shift left by {k}") for k in (1, 2)]
        + [("SHR", k, f"shift right by {k}") for k in (1, 2)]
    )
    op, param, desc = rng.choice(rules)
    vals = rng.sample(range(256), 9)
    ex = [(v, _apply_bit_op(v, op, param)) for v in vals[:8]]
    q = vals[8]
    ans = f"{_apply_bit_op(q, op, param):08b}"
    trace = (f"Comparing inputs to outputs, the rule is to {desc}. "
             f"Applying it to {q:08b} gives {ans}.")
    return _bit_prompt(ex, q), trace, ans


# op1 must lie in solve_bit's _try_two_ops candidate set (NOT/REV/ROL/ROR) so
# the composed rule stays recoverable on re-verify.
_TWO_OP1 = (
    [("NOT", 0, "invert every bit (NOT)"), ("REV", 0, "reverse the bit order")]
    + [("ROL", k, f"rotate left by {k}") for k in (1, 2, 3)]
    + [("ROR", k, f"rotate right by {k}") for k in (1, 2, 3)]
)


def _gen_bit_two_step(rng):
    op1, p1, desc1 = rng.choice(_TWO_OP1)
    op2, p2, desc2 = rng.choice(
        [("NOT", 0, "invert every bit (NOT)"), ("REV", 0, "reverse the bit order")]
        + [("ROL", k, f"rotate left by {k}") for k in (1, 2)]
        + [("ROR", k, f"rotate right by {k}") for k in (1, 2)]
        + [("XOR", m, f"XOR with mask {m:08b}") for m in (rng.randint(1, 255),)]
    )
    apply2 = lambda v: _apply_bit_op(_apply_bit_op(v, op1, p1), op2, p2)
    vals = rng.sample(range(256), 9)
    ex = [(v, apply2(v)) for v in vals[:8]]
    if all(i == o for i, o in ex):       # composed to identity -> retry
        return _gen_bit_two_step(rng)
    q = vals[8]
    mid = _apply_bit_op(q, op1, p1)
    ans = f"{apply2(q):08b}"
    trace = (f"Comparing inputs to outputs, the rule is two steps: first {desc1}, "
             f"then {desc2}. Applying to {q:08b}: step 1 gives {mid:08b}, "
             f"step 2 gives {ans}.")
    return _bit_prompt(ex, q), trace, ans


# ── gravity ───────────────────────────────────────────────────────────────────
def gen_gravity(rng):
    g = round(rng.uniform(4.0, 25.0), 2)
    ts = [round(rng.uniform(1.0, 5.0), 2) for _ in range(6)]
    obs = [(t, round(0.5 * g * t * t, 2)) for t in ts[:5]]
    qt = ts[5]
    ans = f"{0.5 * g * qt * qt:.2f}"
    lines = ["In Alice's Wonderland, the gravitational constant has been secretly changed. "
             "Here are some example observations:"]
    lines += [f"For t = {t}s, distance = {d} m" for t, d in obs]
    lines += [f"Now, determine the falling distance for t = {qt}s given d = 0.5*g*t^2."]
    trace = (f"From d = 0.5*g*t^2 we get g = 2d/t^2. Each observation gives g ≈ {g}. "
             f"So for t = {qt}s, d = 0.5*{g}*{qt}^2 = {ans}.")
    return "\n".join(lines), trace, ans


# ── unit ──────────────────────────────────────────────────────────────────────
def gen_unit(rng):
    k = round(rng.uniform(0.3, 3.0), 4)
    xs = [round(rng.uniform(5.0, 40.0), 2) for _ in range(rng.choice([4, 5, 6]))]
    obs = [(x, round(k * x, 2)) for x in xs[:-1]]
    qx = xs[-1]
    ans = f"{k * qx:.2f}"
    lines = ["In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:"]
    lines += [f"{x} m becomes {y}" for x, y in obs]
    lines += [f"Now, convert the following measurement: {qx} m"]
    trace = (f"The scale factor is k = output/input ≈ {k}. "
             f"So {qx} × {k} = {ans}.")
    return "\n".join(lines), trace, ans


# ── numeral (decimal -> roman) ─────────────────────────────────────────────────
_RV = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
_RS = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]

def _to_roman(n):
    r = ""
    for v, s in zip(_RV, _RS):
        while n >= v:
            r += s; n -= v
    return r

def gen_numeral(rng):
    nums = rng.sample(range(1, 200), 5)
    q = rng.randint(1, 200)
    ans = _to_roman(q)
    lines = ["In Alice's Wonderland, numbers are secretly converted into a different numeral system. "
             "Some examples are given below:"]
    lines += [f"{n} -> {_to_roman(n)}" for n in nums]
    lines += [f"Now, write the number {q} in the Wonderland numeral system."]
    trace = (f"The examples are standard Roman numerals. Converting {q}: {ans}.")
    return "\n".join(lines), trace, ans


# ── cipher (random monoalphabetic, full query coverage) ────────────────────────
def gen_cipher(rng):
    perm = list(ALPHABET)
    rng.shuffle(perm)
    enc = {p: c for p, c in zip(ALPHABET, perm)}     # plain -> cipher
    encode = lambda s: "".join(enc.get(ch, ch) for ch in s)

    ex_plain = [" ".join(rng.sample(WORD_POOL, rng.choice([3, 4]))) for _ in range(5)]
    seen = set("".join(ex_plain).replace(" ", ""))
    candidates = [w for w in WORD_POOL if set(w) <= seen]
    if len(candidates) < 2:
        return gen_cipher(rng)  # retry; rare
    q_plain = " ".join(rng.sample(candidates, rng.choice([2, 3])))

    lines = ["In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:"]
    lines += [f"{encode(p)} -> {p}" for p in ex_plain]
    lines += [f"Now, decrypt the following text: {encode(q_plain)}"]
    # trace: reconstruct cipher->plain map from a few pairs
    dec_pairs = sorted({(enc[p], p) for p in seen})[:8]
    mapping = ", ".join(f"{c}→{p}" for c, p in dec_pairs)
    trace = (f"Each cipher letter maps to a fixed plaintext letter. "
             f"From the examples: {mapping}, .... "
             f"Decoding '{encode(q_plain)}' gives '{q_plain}'.")
    return "\n".join(lines), trace, q_plain


GENERATORS = {
    "bit": gen_bit,
    "gravity": gen_gravity,
    "unit": gen_unit,
    "numeral": gen_numeral,
    "cipher": gen_cipher,
}

# default allocation — weighted toward failure-prone buckets. bit is raised to
# backfill the real bit rows now dropped for weak (per-bit-boolean) traces.
DEFAULT_ALLOC = {"cipher": 3000, "bit": 3000, "numeral": 800, "gravity": 800, "unit": 800}
OUT = Path("synth_curriculum.jsonl")


def main(alloc=None, seed=3407):
    alloc = alloc or DEFAULT_ALLOC
    rng = random.Random(seed)
    records = []
    for bucket, n in alloc.items():
        for i in range(n):
            prompt, trace, ans = GENERATORS[bucket](rng)
            records.append(make_record(bucket, prompt, trace, ans, i))
    rng.shuffle(records)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"synthetic records: {len(records)} -> {OUT}")
    for b, n in alloc.items():
        print(f"  {b:<9} {n}")


if __name__ == "__main__":
    main()
