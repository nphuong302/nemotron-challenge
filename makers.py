"""
Verified chain-of-thought makers for the v5 pipeline.

For each problem type, a maker parses the prompt, derives the hidden rule, and
returns (cot_trace, computed_answer) — or None if the rule can't be recovered.
Callers verify computed_answer against the gold label before keeping an example,
so only correct (problem, trace, answer) triples enter the curriculum.

Self-contained: depends only on solvers.py (bit primitives + example parsing).
Ported from the former 04_corpus_v4.py.
"""
import re
import statistics

from solvers import (
    solve_bit, _apply_bit_op, _try_single_ops, _try_two_ops, _parse_examples,
)


# ── Type detection ──────────────────────────────────────────────────────────
def detect_type(prompt: str) -> str:
    p = prompt.lower()
    if "gravitational" in p or ("distance" in p and "t =" in p and "d = 0.5" in p):
        return "gravity"
    if "unit conversion" in p or " becomes " in p:
        return "unit"
    if "numeral system" in p:
        return "numeral"
    if re.search(r"\b[01]{8}\b", prompt):
        return "bit"
    if re.search(r"[^a-zA-Z\s\d\->.,:;!?'\"()\[\]{}]", prompt):
        return "symbol"
    return "cipher"


# ── Gravity ───────────────────────────────────────────────────────────────────
def make_gravity_cot(prompt: str):
    pairs = re.findall(r"t\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)", prompt, re.IGNORECASE)
    q_match = re.search(r"t\s*=\s*([\d.]+)\s*s\s*given", prompt, re.IGNORECASE)
    if not pairs or not q_match:
        return None
    try:
        tds = [(float(t), float(d)) for t, d in pairs]
        q_t = float(q_match.group(1))
        g_vals = [2 * d / t**2 for t, d in tds]
        g = statistics.median(g_vals)
        d_ans = round(0.5 * g * q_t**2, 2)

        lines = ["Formula: d = 0.5·g·t²  →  g = 2d/t²"]
        for t, d in tds:
            g_i = round(2 * d / t**2, 4)
            lines.append(f"  t={t}s, d={d}m  →  g = 2·{d}/{t}² = {round(2*d,4)}/{round(t**2,4)} = {g_i}")
        lines.append(f"Median g = {round(g,4)} m/s²")
        lines.append(f"Query t={q_t}s  →  d = 0.5·{round(g,4)}·{q_t}² = {d_ans}")
        return "\n".join(lines), str(d_ans)
    except Exception:
        return None


# ── Unit conversion ───────────────────────────────────────────────────────────
def make_unit_cot(prompt: str):
    pairs = re.findall(r"([\d.]+)\s*(?:[a-zA-Z]+)?\s+becomes\s+([\d.]+)", prompt, re.IGNORECASE)
    q_match = re.search(r"convert.*?:\s*([\d.]+)", prompt, re.IGNORECASE)
    if not pairs or not q_match:
        return None
    try:
        q_val = float(q_match.group(1))
        factors = [float(b) / float(a) for a, b in pairs if float(a) != 0]
        if not factors:
            return None
        k = statistics.median(factors)
        result = round(k * q_val, 2)

        lines = ["Find scale factor k = output/input:"]
        for a, b in pairs:
            k_i = round(float(b) / float(a), 4)
            lines.append(f"  {a} → {b}  k = {b}/{a} = {k_i}")
        lines.append(f"Median k = {round(k,4)}")
        lines.append(f"Query: {q_val} × {round(k,4)} = {result}")
        return "\n".join(lines), str(result)
    except Exception:
        return None


# ── Numeral system ────────────────────────────────────────────────────────────
_ROMAN_VAL = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
_ROMAN_SYM = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]


def _to_roman(n: int) -> str:
    r = ""
    for v, s in zip(_ROMAN_VAL, _ROMAN_SYM):
        while n >= v:
            r += s; n -= v
    return r


def _parse_numeral_examples(prompt: str):
    pairs = []
    for line in prompt.splitlines():
        m = re.match(r"\s*(\S+)\s*->\s*(\S+)", line)
        if m:
            pairs.append((m.group(1).strip(), m.group(2).strip()))
    query = None
    for line in reversed(prompt.splitlines()):
        m = re.search(r"(?:number|write)\s+(\d+)", line, re.IGNORECASE)
        if m:
            query = m.group(1).strip()
            break
    if not pairs or not query:
        return None
    return pairs, query


def make_numeral_cot(prompt: str):
    parsed = _parse_numeral_examples(prompt)
    if not parsed:
        return None
    pairs, query = parsed

    is_roman = all(re.match(r"^[IVXLCDMivxlcdm]+$", o) and re.match(r"^\d+$", i)
                   for i, o in pairs)
    if is_roman:
        try:
            n = int(query)
            answer = _to_roman(n)
            lines = ["Examples confirm standard Roman numerals:"]
            for i, o in pairs[:3]:
                lines.append(f"  {i} → {o}  ✓")
            lines.append(f"Query {n} in Roman = {answer}")
            return "\n".join(lines), answer
        except Exception:
            return None

    lookup = dict(pairs)
    if query in lookup:
        answer = lookup[query]
        return f"Direct mapping from examples: {query} → {answer}", answer

    return None


# ── Cipher ────────────────────────────────────────────────────────────────────
def _build_char_map_from_pairs(inputs, outputs):
    char_map = {}
    for cipher_sent, plain_sent in zip(inputs, outputs):
        cwords = cipher_sent.strip().split()
        pwords = plain_sent.strip().split()
        if len(cwords) != len(pwords):
            continue
        for cw, pw in zip(cwords, pwords):
            if len(cw) != len(pw):
                continue
            for cc, pc in zip(cw, pw):
                char_map[cc] = pc
    return char_map


def make_cipher_cot(prompt: str):
    inputs, outputs, query = _parse_examples(prompt)
    if not inputs or not query:
        return None

    query = re.sub(r'^(?:text|following|for|number)\s*:?\s*', '', query, flags=re.IGNORECASE).strip()
    if not query:
        return None

    char_map = _build_char_map_from_pairs(inputs, outputs)
    if not char_map:
        return None

    result = ""
    for ch in query:
        if ch == " ":
            result += " "
        elif ch in char_map:
            result += char_map[ch]
        else:
            return None
    result = result.strip()
    if not result:
        return None

    lines = ["Letter substitution cipher — build mapping from examples:"]
    for k, v in sorted(char_map.items())[:10]:
        lines.append(f"  {k} → {v}")
    if len(char_map) > 10:
        lines.append(f"  ... ({len(char_map)} total mappings)")
    lines.append(f"Decode '{query}':")
    steps = "  " + " ".join(
        f"{ch}→{char_map[ch]}" if ch != " " else "[sp]" for ch in query
    )
    lines.append(steps)
    lines.append(f"Result: {result}")
    return "\n".join(lines), result


# ── Bit manipulation ──────────────────────────────────────────────────────────
def make_bit_cot(prompt: str):
    bit_pairs = re.findall(r"([01]{8})\s*->\s*([01]{8})", prompt)
    query_match = re.search(
        r"(?:determine|convert|write|now).*?([01]{8})", prompt, re.IGNORECASE | re.DOTALL
    )
    if not bit_pairs or not query_match:
        return None
    try:
        ex = [(int(i, 2), int(o, 2)) for i, o in bit_pairs]
        query_str = query_match.group(1)
        q = int(query_str, 2)
    except ValueError:
        return None

    answer = solve_bit(prompt)
    if answer is None:
        return None

    single = _try_single_ops(ex)
    if single:
        op_name, param = single
        label = op_name if not param else f"{op_name} {param}"
        lines = [f"8-bit operation detected: {label}"]
        for i_int, o_int in ex[:3]:
            lines.append(f"  {i_int:08b} → {o_int:08b}  ✓")
        lines.append(f"Apply to {query_str}: {label}({query_str}) = {answer}")
        return "\n".join(lines), answer

    two = _try_two_ops(ex)
    if two:
        op1, p1, op2, p2 = two
        label1 = op1 if not p1 else f"{op1} {p1}"
        label2 = op2 if not p2 else f"{op2} {p2}"
        mid_int = _apply_bit_op(q, op1, p1)
        mid_str = f"{mid_int:08b}"
        lines = [f"Two-step 8-bit operation: {label1} then {label2}"]
        for i_int, o_int in ex[:3]:
            m = _apply_bit_op(i_int, op1, p1)
            lines.append(f"  {i_int:08b} → {m:08b} → {o_int:08b}  ✓")
        lines.append(f"Apply to {query_str}:")
        lines.append(f"  Step 1 {label1}: {query_str} → {mid_str}")
        lines.append(f"  Step 2 {label2}: {mid_str} → {answer}")
        return "\n".join(lines), answer

    lines = ["Per-bit boolean function:"]
    for i_int, o_int in ex[:4]:
        lines.append(f"  {i_int:08b} → {o_int:08b}")
    lines.append(f"Apply pattern to {query_str}: {answer}")
    return "\n".join(lines), answer


# ── Validation ────────────────────────────────────────────────────────────────
def answers_match(computed: str, labeled: str, ptype: str) -> bool:
    if computed is None:
        return False
    if ptype in ("gravity", "unit"):
        try:
            return abs(float(computed) - float(labeled)) <= 0.02
        except Exception:
            return False
    elif ptype == "cipher":
        return str(computed).strip().lower() == str(labeled).strip().lower()
    else:  # numeral, bit
        return str(computed).strip().upper() == str(labeled).strip().upper()


MAKERS = {
    "gravity": make_gravity_cot,
    "unit":    make_unit_cot,
    "numeral": make_numeral_cot,
    "cipher":  make_cipher_cot,
    "bit":     make_bit_cot,
}
