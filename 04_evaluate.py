"""
Evaluate the fine-tuned model (base + LoRA adapter) on validation problems.
Reports pass@1 and majority-vote (cons@k) accuracy.

Usage:
  python 04_evaluate.py --adapter adapter/ --problems val_problems.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
SYSTEM   = "detailed thinking on"
N_VOTES  = 8   # samples for majority voting


def load_model(adapter_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # merge LoRA weights for faster inference
    model.eval()
    return tokenizer, model


def extract_boxed(text: str) -> str | None:
    match = re.search(r"\\boxed\{(.+?)\}", text)
    return match.group(1).strip() if match else None


@torch.inference_mode()
def sample_n(tokenizer, model, problem: str, n: int) -> list[str]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": problem},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        num_return_sequences=n,
    )
    results = []
    for seq in outputs:
        new_tokens = seq[inputs["input_ids"].shape[-1]:]
        results.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return results


def majority_vote(answers: list[str | None]) -> str | None:
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def evaluate(tokenizer, model, problems: list[dict]) -> dict:
    pass1_correct = 0
    consN_correct = 0

    for prob in problems:
        gt = prob["answer"]
        samples = sample_n(tokenizer, model, prob["problem"], N_VOTES)
        extracted = [extract_boxed(s) for s in samples]

        # pass@1: first sample
        if extracted[0] == gt:
            pass1_correct += 1

        # cons@N: majority vote
        voted = majority_vote(extracted)
        if voted == gt:
            consN_correct += 1

        print(f"[{prob['id']}] gt={gt} | pass1={'✓' if extracted[0]==gt else '✗'} "
              f"| cons{N_VOTES}={'✓' if voted==gt else '✗'}")

    n = len(problems)
    metrics = {
        "pass@1":      round(pass1_correct / n * 100, 2),
        f"cons@{N_VOTES}": round(consN_correct / n * 100, 2),
        "n_problems":  n,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",  default=None,               help="Path to LoRA adapter (optional)")
    parser.add_argument("--problems", default="val_problems.jsonl", help="JSONL file with validation problems")
    args = parser.parse_args()

    tokenizer, model = load_model(args.adapter)

    with open(args.problems) as f:
        problems = [json.loads(l) for l in f if l.strip()]

    metrics = evaluate(tokenizer, model, problems)
    print("\n=== Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
