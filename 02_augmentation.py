"""
Generate chain-of-thought solutions for train.csv problems.
Uses Nemotron itself (or a teacher model) to produce step-by-step reasoning,
then keeps only the rows where the extracted final answer matches ground truth.

Output: corpus.jsonl — (prompt, cot_solution, answer) triples for SFT.
"""

import csv
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID    = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DATA_DIR    = Path("data")
CORPUS_PATH = Path("corpus.jsonl")
N_SAMPLES   = 4      # solutions to generate per problem
MAX_TRAIN   = 500    # problems to process (increase for full run)
MAX_NEW_TOK = 512    # Mamba SSM intermediate tensor scales with seq len; 512 keeps it under 2 GB

# bit/cipher/symbol score 0% with this model — skip to avoid wasting compute
SKIP_TYPES  = {"bit", "cipher", "symbol"}

# reuse type detection and system prompts from 01_inference.py
def detect_type(prompt: str) -> str:
    p = prompt.lower()
    if "bit manipulation" in p:            return "bit"
    if "gravitational" in p:               return "gravity"
    if "unit conversion" in p:             return "unit"
    if "numeral system" in p:              return "numeral"
    if "transformation rule" in p:         return "symbol"
    if "encryption" in p or "cipher" in p: return "cipher"
    return "unknown"

SYSTEM = {
    "gravity": "detailed thinking on\nUse d = 0.5 * g * t^2. Solve for g from examples, then compute the answer. Output your final numeric answer (2 decimal places) as \\boxed{answer} on the last line.",
    "unit": "detailed thinking on\nFind the hidden scale factor from examples. Apply to the query. Output your final numeric answer (2 decimal places) as \\boxed{answer} on the last line.",
    "numeral": "detailed thinking on\nFind the numeral system mapping from examples. Apply to the query. Output your final result as \\boxed{answer} on the last line.",
    "unknown": "detailed thinking on\nFind the pattern from examples, apply it to the query. Output your final answer as \\boxed{answer} on the last line.",
}

def extract_final_answer(text: str) -> str:
    m = re.search(r"\\boxed\{(.+?)\}", text)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()

def normalize(answer: str, ptype: str) -> str:
    ans = answer.strip().lower()
    if ptype in ("gravity", "unit"):
        try:
            return str(round(float(ans), 2))
        except ValueError:
            return ans
    return ans

def answers_match(predicted: str, ground_truth: str, ptype: str) -> bool:
    return normalize(predicted, ptype) == normalize(ground_truth, ptype)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def sample_solutions(tokenizer, model, system: str, prompt: str, n: int) -> list[str]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)
    results = []
    for _ in range(n):
        torch.cuda.empty_cache()
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOK,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
        )
        new = output[0][inputs["input_ids"].shape[-1]:]
        results.append(tokenizer.decode(new, skip_special_tokens=True))
    return results


def get_done_ids(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


def main():
    tokenizer, model = load_model()

    with open(DATA_DIR / "train.csv") as f:
        train_rows = list(csv.DictReader(f))[:MAX_TRAIN]

    done_ids = get_done_ids(CORPUS_PATH)
    print(f"Resuming: {len(done_ids)} problems already done, {CORPUS_PATH} exists")

    corpus_count = sum(1 for _ in open(CORPUS_PATH)) if CORPUS_PATH.exists() else 0
    correct_count = 0

    with open(CORPUS_PATH, "a") as out_f:
        for i, row in enumerate(train_rows):
            if row["id"] in done_ids:
                print(f"[{i+1}/{len(train_rows)}] SKIP id={row['id']}")
                continue

            ptype   = detect_type(row["prompt"])
            if ptype in SKIP_TYPES:
                print(f"[{i+1}/{len(train_rows)}] SKIP type={ptype} (0% accuracy)")
                continue
            system  = SYSTEM.get(ptype, SYSTEM["unknown"])
            gt      = row["answer"]

            solutions = sample_solutions(tokenizer, model, system, row["prompt"], N_SAMPLES)

            n_correct = 0
            for sol in solutions:
                extracted = extract_final_answer(sol)
                if answers_match(extracted, gt, ptype):
                    item = {"id": row["id"], "type": ptype, "prompt": row["prompt"],
                            "solution": sol, "answer": gt}
                    out_f.write(json.dumps(item) + "\n")
                    out_f.flush()
                    corpus_count += 1
                    n_correct += 1

            correct_count += n_correct
            print(f"[{i+1}/{len(train_rows)}] type={ptype} correct={n_correct}/{N_SAMPLES} total_corpus={corpus_count}")

    print(f"\nDone. Corpus: {corpus_count} correct solutions saved to {CORPUS_PATH}")

if __name__ == "__main__":
    main()
