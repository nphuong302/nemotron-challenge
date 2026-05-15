"""
Deterministic code-based solvers for each problem type.
These bypass the LLM entirely for problems that have exact algebraic/combinatorial solutions.
The LLM is only used as fallback when code can't find a consistent rule.
"""

import re
import itertools
import statistics
from typing import Optional


def _parse_examples(prompt: str) -> tuple[list[str], list[str], str]:
    """Extract (inputs, outputs, query) from any problem prompt."""
    lines = prompt.strip().splitlines()
    inputs, outputs = [], []
    query = ""
    for line in lines:
        if "->" in line:
            parts = line.split("->", 1)
            lhs = parts[0].strip()
            rhs = parts[1].strip()
            # skip header/description lines like "Here are some examples of input -> output:"
            # valid data lines: both sides are short tokens (numbers, binary, words)
            # reject if lhs contains more than ~6 words (it's a description)
            if len(lhs.split()) <= 5 and len(rhs.split()) <= 10:
                inputs.append(lhs)
                outputs.append(rhs)
        elif line.strip().lower().startswith("now") or "determine" in line.lower() or "convert" in line.lower() or "write" in line.lower() or "decrypt" in line.lower():
            m = re.search(r"(?:for:|following:?|number\s+|text:\s*)(.+)$", line, re.IGNORECASE)
            if m:
                query = m.group(1).strip().rstrip(".")
    # fallback: last non-arrow, non-header line
    if not query:
        for line in reversed(lines):
            line = line.strip()
            if line and "->" not in line and not line.lower().startswith("in alice") and not line.lower().startswith("here") and not line.lower().startswith("now"):
                query = line
                break
    return inputs, outputs, query


# ── Bit manipulation ────────────────────────────────────────────────────────

def _apply_bit_op(val: int, op: str, param: int = 0) -> int:
    if op == "NOT":  return (~val) & 0xFF
    if op == "REV":  return int(f"{val:08b}"[::-1], 2)
    if op == "ROL":  return ((val << param) | (val >> (8 - param))) & 0xFF
    if op == "ROR":  return ((val >> param) | (val << (8 - param))) & 0xFF
    if op == "XOR":  return val ^ param
    if op == "AND":  return val & param
    if op == "OR":   return val | param
    if op == "SHL":  return (val << param) & 0xFF
    if op == "SHR":  return (val >> param) & 0xFF
    return val

def _try_single_ops(examples: list[tuple[int, int]]) -> Optional[tuple]:
    for op in ("NOT", "REV"):
        if all(_apply_bit_op(i, op) == o for i, o in examples):
            return (op, 0)
    for op in ("ROL", "ROR", "SHL", "SHR"):
        for p in range(1, 8):
            if all(_apply_bit_op(i, op, p) == o for i, o in examples):
                return (op, p)
    for p in range(256):
        for op in ("XOR", "AND", "OR"):
            if all(_apply_bit_op(i, op, p) == o for i, o in examples):
                return (op, p)
    return None

def _try_two_ops(examples: list[tuple[int, int]]) -> Optional[tuple]:
    candidates = [
        ("NOT", 0), ("REV", 0),
        *[("ROL", p) for p in range(1, 8)],
        *[("ROR", p) for p in range(1, 8)],
        *[("XOR", p) for p in range(256)],
    ]
    for op1, p1 in candidates:
        mid = [(_apply_bit_op(i, op1, p1), o) for i, o in examples]
        r = _try_single_ops(mid)
        if r:
            return (op1, p1, r[0], r[1])
    return None

def _learn_per_bit(examples: list[tuple[int, int]], query: int) -> Optional[str]:
    """
    Learn each output bit as an independent boolean function of input bits.
    Tries (in order): constant 0/1, single bit, NOT single bit,
    and XOR/AND/OR of every pair of input bits.
    """
    n = len(examples)
    result = 0
    for out_pos in range(8):
        target = [(e[1] >> out_pos) & 1 for e in examples]

        # constant
        if all(b == 0 for b in target):
            result |= (0 << out_pos)
            continue
        if all(b == 1 for b in target):
            result |= (1 << out_pos)
            continue

        found = False
        # single input bit or its NOT
        for j in range(8):
            bits     = [(e[0] >> j) & 1 for e in examples]
            not_bits = [1 - b for b in bits]
            if bits == target:
                result |= (((query >> j) & 1) << out_pos)
                found = True; break
            if not_bits == target:
                result |= ((1 - ((query >> j) & 1)) << out_pos)
                found = True; break

        if found:
            continue

        # two input bits combined with XOR / AND / OR / XNOR
        for j1 in range(8):
            for j2 in range(j1, 8):
                b1 = [(e[0] >> j1) & 1 for e in examples]
                b2 = [(e[0] >> j2) & 1 for e in examples]
                combos = {
                    "XOR":  [a ^ b      for a, b in zip(b1, b2)],
                    "AND":  [a & b      for a, b in zip(b1, b2)],
                    "OR":   [a | b      for a, b in zip(b1, b2)],
                    "XNOR": [1-(a ^ b)  for a, b in zip(b1, b2)],
                    "NAND": [1-(a & b)  for a, b in zip(b1, b2)],
                    "NOR":  [1-(a | b)  for a, b in zip(b1, b2)],
                }
                for op, bits in combos.items():
                    if bits == target:
                        q1 = (query >> j1) & 1
                        q2 = (query >> j2) & 1
                        if op == "XOR":  qb = q1 ^ q2
                        elif op == "AND": qb = q1 & q2
                        elif op == "OR":  qb = q1 | q2
                        elif op == "XNOR": qb = 1 - (q1 ^ q2)
                        elif op == "NAND": qb = 1 - (q1 & q2)
                        elif op == "NOR":  qb = 1 - (q1 | q2)
                        result |= (qb << out_pos)
                        found = True; break
                if found: break

        if found:
            continue

        # three input bits ANDed together (with optional NOT on each)
        # covers patterns like: bit5 AND NOT(bit6) AND NOT(bit7)
        for j1 in range(8):
            for j2 in range(j1+1, 8):
                for j3 in range(j2+1, 8):
                    for n1, n2, n3 in itertools.product((0, 1), repeat=3):
                        b1 = [((e[0] >> j1) & 1) ^ n1 for e in examples]
                        b2 = [((e[0] >> j2) & 1) ^ n2 for e in examples]
                        b3 = [((e[0] >> j3) & 1) ^ n3 for e in examples]
                        bits = [a & b & c for a, b, c in zip(b1, b2, b3)]
                        if bits == target:
                            q1 = ((query >> j1) & 1) ^ n1
                            q2 = ((query >> j2) & 1) ^ n2
                            q3 = ((query >> j3) & 1) ^ n3
                            result |= ((q1 & q2 & q3) << out_pos)
                            found = True; break
                    if found: break
                if found: break

        if not found:
            return None  # can't express this bit with available ops

    return f"{result:08b}"

def solve_bit(prompt: str) -> Optional[str]:
    inputs, outputs, query = _parse_examples(prompt)
    if not inputs or not query:
        return None
    try:
        ex = [(int(i, 2), int(o, 2)) for i, o in zip(inputs, outputs)]
        q  = int(query, 2)
    except ValueError:
        return None

    # Try holistic ops first (fast)
    for fn in (_try_single_ops, _try_two_ops):
        op = fn(ex)
        if op and len(op) == 2:
            return f"{_apply_bit_op(q, op[0], op[1]):08b}"
        if op and len(op) == 4:
            mid = _apply_bit_op(q, op[0], op[1])
            return f"{_apply_bit_op(mid, op[2], op[3]):08b}"

    # Fall back to per-bit boolean function learning
    return _learn_per_bit(ex, q)


# ── Physics (gravity) ───────────────────────────────────────────────────────

def solve_gravity(prompt: str) -> Optional[str]:
    # extract (t, d) pairs
    pairs = re.findall(r"t\s*=\s*([\d.]+)\s*s.*?distance\s*=\s*([\d.]+)", prompt, re.IGNORECASE)
    query = re.search(r"t\s*=\s*([\d.]+)\s*s\s*given", prompt, re.IGNORECASE)
    if not pairs or not query:
        return None
    try:
        tds = [(float(t), float(d)) for t, d in pairs]
        q_t = float(query.group(1))
        g_vals = [2 * d / (t ** 2) for t, d in tds]
        g = statistics.median(g_vals)
        return str(round(0.5 * g * q_t ** 2, 2))
    except Exception:
        return None


# ── Unit conversion ─────────────────────────────────────────────────────────

def solve_unit(prompt: str) -> Optional[str]:
    pairs = re.findall(r"([\d.]+)\s*(?:m|km|kg|s)?\s+becomes\s+([\d.]+)", prompt, re.IGNORECASE)
    query = re.search(r"convert.*?:\s*([\d.]+)", prompt, re.IGNORECASE)
    if not pairs or not query:
        return None
    try:
        factors = [float(b) / float(a) for a, b in pairs if float(a) != 0]
        k = statistics.median(factors)
        return str(round(k * float(query.group(1)), 2))
    except Exception:
        return None


# ── Numeral system ──────────────────────────────────────────────────────────

def solve_numeral(prompt: str) -> Optional[str]:
    """Build a direct lookup from examples and return if query matches a seen input."""
    inputs, outputs, query = _parse_examples(prompt)
    if not inputs or not query:
        return None
    lookup = dict(zip(inputs, outputs))
    # direct match
    if query in lookup:
        return lookup[query]
    # try to detect Roman numeral pattern
    try:
        num = int(query)
        # check if outputs look like Roman numerals
        if all(re.match(r'^[IVXLCDM]+$', o) for o in outputs):
            return _to_roman(num)
    except ValueError:
        pass
    return None

def _to_roman(n: int) -> str:
    val = [1000,900,500,400,100,90,50,40,10,9,5,4,1]
    sym = ["M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"]
    result = ""
    for i, v in enumerate(val):
        while n >= v:
            result += sym[i]
            n -= v
    return result


# ── Cipher (letter substitution) ────────────────────────────────────────────

def build_cipher_map(prompt: str) -> tuple[dict[str, str], str]:
    """Return (char_map, query). char_map may be incomplete."""
    example_lines = re.findall(r"([a-z ]+)\s*->\s*([a-z ]+)", prompt.lower())
    char_map: dict[str, str] = {}
    for cipher_sent, plain_sent in example_lines:
        cw = cipher_sent.strip().split()
        pw = plain_sent.strip().split()
        if len(cw) != len(pw):
            continue
        for cword, pword in zip(cw, pw):
            if len(cword) != len(pword):
                continue
            for cc, pc in zip(cword, pword):
                char_map[cc] = pc  # last write wins on conflict

    query = ""
    for line in reversed(prompt.strip().splitlines()):
        line = line.strip().lower()
        if re.match(r'^[a-z ]+$', line) and line:
            query = line
            break
    return char_map, query


def solve_cipher(prompt: str) -> Optional[str]:
    """Build letter-level substitution map and decode query.
    Returns None if any query character is unknown (caller should use LLM)."""
    char_map, query = build_cipher_map(prompt)
    if not query or not char_map:
        return None
    result = ""
    for ch in query:
        if ch == " ":
            result += " "
        elif ch in char_map:
            result += char_map[ch]
        else:
            return None  # unknown char — fall back to LLM
    return result.strip()


def cipher_hint(prompt: str) -> str:
    """Return a context string with the known mappings for LLM-assisted decoding."""
    char_map, query = build_cipher_map(prompt)
    if not char_map:
        return ""
    known = ", ".join(f"{k}→{v}" for k, v in sorted(char_map.items()))
    partial = "".join(char_map.get(c, f"[{c}?]") if c != " " else " " for c in query)
    return (
        f"Known letter mappings: {known}\n"
        f"Partial decode of query '{query}': {partial}\n"
        f"Fill in the [?] characters to complete the decoded sentence."
    )


# ── Symbol transform ─────────────────────────────────────────────────────────

def solve_symbol(prompt: str) -> Optional[str]:
    """Build symbol-level substitution map (character by character or word by word)."""
    inputs, outputs, query = _parse_examples(prompt)
    if not inputs or not query:
        return None

    # try character-level map
    char_map: dict[str, str] = {}
    for inp, out in zip(inputs, outputs):
        if len(inp) != len(out):
            char_map = {}
            break
        for ci, co in zip(inp, out):
            if ci in char_map and char_map[ci] != co:
                char_map = {}
                break
            char_map[ci] = co

    if char_map:
        result = ""
        for ch in query:
            if ch in char_map:
                result += char_map[ch]
            else:
                result = ""
                break
        if result:
            return result

    # try word-level map
    word_map: dict[str, str] = {}
    for inp, out in zip(inputs, outputs):
        iw = inp.split()
        ow = out.split()
        if len(iw) != len(ow):
            return None
        for wi, wo in zip(iw, ow):
            if wi in word_map and word_map[wi] != wo:
                return None
            word_map[wi] = wo

    query_words = query.split()
    if all(w in word_map for w in query_words):
        return " ".join(word_map[w] for w in query_words)

    return None


# ── Dispatcher ───────────────────────────────────────────────────────────────

SOLVERS = {
    "bit":    solve_bit,
    "gravity": solve_gravity,
    "unit":   solve_unit,
    "numeral": solve_numeral,
    "cipher": solve_cipher,
    "symbol": solve_symbol,
}

def code_solve(problem_type: str, prompt: str) -> Optional[str]:
    solver = SOLVERS.get(problem_type)
    if solver is None:
        return None
    return solver(prompt)
