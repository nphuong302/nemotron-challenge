"""
Inference for the NVIDIA Nemotron Reasoning Challenge.

The dataset has 6 problem types — all "Alice's Wonderland" few-shot rule induction:
  1. bit_manipulation   — find hidden bitwise transform from examples
  2. physics_gravity    — deduce modified gravity constant, apply d=0.5*g*t^2
  3. unit_conversion    — find hidden linear scale factor
  4. numeral_system     — find number-system mapping
  5. symbol_transform   — find symbol substitution/permutation rule
  6. encryption_cipher  — find text cipher/encoding rule

Strategy:
  - Chain-of-thought prompting tuned per problem type
  - Code execution for deterministic types (bit, physics, unit conversion)
  - Majority vote across N samples
"""

import csv
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID     = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DATA_DIR     = Path("data")
OUTPUT_CSV   = Path("submission.csv")
N_VOTES      = 8
MAX_NEW_TOK  = 1024

# ── Problem type detection ──────────────────────────────────────────────────

def detect_type(prompt: str) -> str:
    p = prompt.lower()
    if "bit manipulation" in p:            return "bit"
    if "gravitational" in p:               return "gravity"
    if "unit conversion" in p:             return "unit"
    if "numeral system" in p:              return "numeral"
    if "transformation rule" in p:         return "symbol"
    if "encryption" in p or "cipher" in p: return "cipher"
    return "unknown"

# ── Per-type system prompts ─────────────────────────────────────────────────

SYSTEM = {
    "bit": """\
detailed thinking on
You are solving a bit-manipulation pattern discovery puzzle.
Steps:
1. Write out each example in binary, analyzing how each bit position changes.
2. Hypothesize the transformation (shifts, rotations, XOR, AND, OR, NOT).
3. Verify your hypothesis against ALL given examples.
4. Apply the verified rule to the query input.
5. Output your final 8-bit binary answer on the last line as \\boxed{answer}, e.g. \\boxed{10110101}.""",

    "gravity": """\
detailed thinking on
You are solving a modified-gravity physics puzzle.
Steps:
1. Use d = 0.5 * g * t^2. Solve for g using the given (t, d) pairs.
2. Verify g is consistent across all examples.
3. Calculate the answer for the query t.
4. Round to 2 decimal places.
5. Output your final numeric answer on the last line as \\boxed{answer}, e.g. \\boxed{12.34}.""",

    "unit": """\
detailed thinking on
You are solving a hidden unit-conversion puzzle.
Steps:
1. Compute the scale factor: output / input for each example.
2. Verify the factor is consistent across all examples.
3. Apply the factor to the query value.
4. Round to 2 decimal places.
5. Output your final numeric answer on the last line as \\boxed{answer}, e.g. \\boxed{3.14}.""",

    "numeral": """\
detailed thinking on
You are solving a numeral-system conversion puzzle.
Steps:
1. Identify the pattern by which numbers map to the output system.
2. Verify with all given examples.
3. Apply to the query.
4. Output your final converted value on the last line as \\boxed{answer}.""",

    "symbol": """\
detailed thinking on
You are solving a symbol-transformation puzzle.
Steps:
1. Map each input symbol to its output symbol using the examples.
2. Check for positional rules or ordering changes.
3. Apply the mapping to the query.
4. Output your final transformed result on the last line as \\boxed{answer}.""",

    "cipher": """\
detailed thinking on
You are solving a text cipher/encryption puzzle.
Steps:
1. Align input words to output words across multiple examples.
2. Build a word-level substitution map.
3. Check for structural consistency (same word count, ordering).
4. Decode the query using your map.
5. Output your final decoded text on the last line as \\boxed{answer}.""",

    "unknown": "detailed thinking on\nAnalyze the pattern from the examples and apply it to the query. Output your final answer on the last line as \\boxed{answer}.",
}

# ── Code-execution verifier (for deterministic types) ───────────────────────

def try_code_solve(problem_type: str, prompt: str) -> str | None:
    """For physics and unit conversion, solve exactly with code."""
    if problem_type not in ("gravity", "unit"):
        return None

    # extract example pairs  (number, number)
    pairs = re.findall(r"[\w\s]*?(\d+\.?\d*)\s*(?:s|m)?\s*(?:,\s*distance\s*=\s*|becomes\s+)(\d+\.?\d*)", prompt)
    # fallback: any two floats per line
    if not pairs:
        pairs = re.findall(r"(\d+\.?\d+)[^\d]+(\d+\.?\d+)", prompt.split("Now,")[0])

    query_match = re.search(r"(?:t\s*=\s*|convert\s+)([\d.]+)", prompt.split("Now,")[-1])
    if not pairs or not query_match:
        return None

    query_val = float(query_match.group(1))
    code_lines = [
        "import statistics",
        f"pairs = {[(float(a), float(b)) for a, b in pairs]}",
    ]
    if problem_type == "gravity":
        code_lines += [
            "factors = [2*d/(t**2) for t,d in pairs]",
            f"g = statistics.median(factors)",
            f"ans = round(0.5 * g * {query_val}**2, 2)",
            "print(ans)",
        ]
    else:  # unit
        code_lines += [
            "factors = [b/a for a,b in pairs]",
            f"k = statistics.median(factors)",
            f"ans = round(k * {query_val}, 2)",
            "print(ans)",
        ]
    script = "\n".join(code_lines)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        fname = f.name
    try:
        result = subprocess.run([sys.executable, fname], capture_output=True, text=True, timeout=5)
        out = result.stdout.strip()
        return out if out else None
    except Exception:
        return None

# ── Model inference ─────────────────────────────────────────────────────────

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return tokenizer, model


def extract_answer(text: str) -> str:
    """Extract from \\boxed{} first, fall back to last non-empty line."""
    m = re.search(r"\\boxed\{(.+?)\}", text)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()


@torch.inference_mode()
def sample_answers(tokenizer, model, system: str, prompt: str, n: int) -> list[str]:
    messages = [
        {"role": "system",  "content": system},
        {"role": "user",    "content": prompt},
    ]
    chat = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)
    results = []
    for _ in range(n):
        torch.cuda.empty_cache()
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOK,
            do_sample=True,
            temperature=0.6,
        )
        new = output[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(new, skip_special_tokens=True)
        results.append(extract_answer(text))
    return results


def predict(tokenizer, model, row: dict) -> str:
    ptype  = detect_type(row["prompt"])
    system = SYSTEM.get(ptype, SYSTEM["unknown"])

    # 1. Try exact code solution for deterministic types
    code_ans = try_code_solve(ptype, row["prompt"])
    if code_ans:
        print(f"  [code-solve] {code_ans}")
        return code_ans

    # 2. LLM sampling + majority vote
    answers = sample_answers(tokenizer, model, system, row["prompt"], N_VOTES)
    print(f"  [samples] {answers}")
    winner = Counter(answers).most_common(1)[0][0]
    return winner

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    tokenizer, model = load_model()

    with open(DATA_DIR / "test.csv") as f:
        test_rows = list(csv.DictReader(f))

    results = []
    for i, row in enumerate(test_rows):
        ptype = detect_type(row["prompt"])
        print(f"[{i+1}/{len(test_rows)}] id={row['id']} type={ptype}")
        answer = predict(tokenizer, model, row)
        print(f"  → {answer}")
        results.append({"id": row["id"], "answer": answer})

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "answer"])
        w.writeheader()
        w.writerows(results)

    print(f"\nSaved {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
