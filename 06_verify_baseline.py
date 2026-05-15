"""
Quick baseline verification on the 3 public test rows (whose answers we know from train).
Run this BEFORE loading the big model to verify your pipeline works.

Known answers:
  00066667 (bit)    → 10010111
  000b53cf (bit)    → 01000011
  00189f6a (cipher) → cat imagines book
"""

import csv
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from solvers import code_solve, cipher_hint

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DATA_DIR = Path("data")

KNOWN_ANSWERS = {
    "00066667": "10010111",
    "000b53cf": "01000011",
    "00189f6a": "cat imagines book",
}

# Same detection + system prompts as inference script
def detect_type(prompt):
    p = prompt.lower()
    if "bit manipulation" in p:            return "bit"
    if "gravitational" in p:               return "gravity"
    if "unit conversion" in p:             return "unit"
    if "numeral system" in p:              return "numeral"
    if "transformation rule" in p:         return "symbol"
    if "encryption" in p or "cipher" in p: return "cipher"
    return "unknown"

SYSTEM = {
    "bit": (
        "detailed thinking on\n"
        "Analyze the bit-transformation pattern from the examples step by step. "
        "Test your hypothesis against every example. "
        "At the very end, write exactly: Answer: XXXXXXXX  (8 bits, no spaces)"
    ),
    "cipher": (
        "detailed thinking on\n"
        "Build a word-level substitution map from the examples. "
        "Decode the query sentence word by word. "
        "At the very end, write exactly: Answer: <decoded text>"
    ),
}

def extract_final_answer(text: str, ptype: str = "") -> str:
    # Prefer explicit "Answer: ..." marker
    m = re.search(r"Answer:\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: last 8-bit binary string for bit problems
    if ptype == "bit":
        bins = re.findall(r"\b[01]{8}\b", text)
        if bins:
            return bins[-1]
    # General fallback: last non-empty line
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()

@torch.inference_mode()
def run(tokenizer, model, prompt, system, ptype="", n=4):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)

    # Nemotron-H (Mamba SSM) OOMs with num_return_sequences>1 because the SSM
    # state expansion (b,c,l,s,h,n) scales quadratically with batch size.
    # Generate one sample at a time and vote instead.
    answers = []
    for _ in range(n):
        out = model.generate(**inputs, max_new_tokens=3000, do_sample=True,
                             temperature=0.6)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        answers.append(extract_final_answer(text, ptype))
        torch.cuda.empty_cache()

    return Counter(answers).most_common(1)[0][0]


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    with open(DATA_DIR / "test.csv") as f:
        rows = list(csv.DictReader(f))

    correct = 0
    for row in rows:
        ptype   = detect_type(row["prompt"])
        system  = SYSTEM.get(ptype, "detailed thinking on\nOutput ONLY the answer on the last line.")

        # 1. Try deterministic code solver
        pred = code_solve(ptype, row["prompt"])
        if pred:
            print(f"  [code-solver] {pred}")
        else:
            # 2. For cipher: inject pre-computed char map as extra context
            prompt = row["prompt"]
            if ptype == "cipher":
                hint = cipher_hint(prompt)
                if hint:
                    prompt = prompt + "\n\n[Hint]\n" + hint
            pred = run(tokenizer, model, prompt, system, ptype=ptype)

        gt      = KNOWN_ANSWERS.get(row["id"], "?")
        ok      = pred.strip().lower() == gt.strip().lower()
        correct += int(ok)
        print(f"{'✓' if ok else '✗'} [{row['id']}] type={ptype}")
        print(f"  pred : {pred}")
        print(f"  truth: {gt}")
        print()

    print(f"Score: {correct}/{len(rows)} = {correct/len(rows)*100:.1f}%")

if __name__ == "__main__":
    main()
