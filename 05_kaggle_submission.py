"""
Kaggle submission notebook entry point.

On Kaggle:
- GPU: G4 (NVIDIA L4, 24GB VRAM)
- Time limit: ~9 hours (check competition rules)
- Internet: enabled during notebook run (to pull model from HuggingFace)

This script runs inference on the test set and writes submission.csv.
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---- paths (Kaggle input structure) ----
INPUT_DIR   = Path("/kaggle/input/nvidia-nemotron-model-reasoning-challenge")
ADAPTER_DIR = Path("/kaggle/input/nemotron-adapter")  # upload your adapter as dataset
OUTPUT_CSV  = Path("/kaggle/working/submission.csv")

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
SYSTEM   = "detailed thinking on"
N_VOTES  = 4   # trade off accuracy vs. time budget


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if ADAPTER_DIR.exists():
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
        model = model.merge_and_unload()
    model.eval()
    return tokenizer, model


def extract_boxed(text: str) -> str | None:
    match = re.search(r"\\boxed\{(.+?)\}", text)
    return match.group(1).strip() if match else None


@torch.inference_mode()
def predict(tokenizer, model, problem: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": problem},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    answers = []
    for _ in range(N_VOTES):
        torch.cuda.empty_cache()
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
        )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        ans = extract_boxed(text)
        answers.append(ans)

    valid = [a for a in answers if a]
    if not valid:
        return "0"  # fallback
    from collections import Counter
    return Counter(valid).most_common(1)[0][0]


def main():
    tokenizer, model = load_model()

    test_file = INPUT_DIR / "test.jsonl"
    if not test_file.exists():
        # fallback: try CSV
        test_file = INPUT_DIR / "test.csv"
        df = pd.read_csv(test_file)
        problems = df.to_dict("records")
    else:
        with open(test_file) as f:
            problems = [json.loads(l) for l in f if l.strip()]

    rows = []
    t0 = time.time()
    for i, prob in enumerate(problems):
        problem_text = prob.get("problem") or prob.get("prompt", "")
        answer = predict(tokenizer, model, problem_text)
        rows.append({"id": prob["id"], "answer": answer})
        elapsed = time.time() - t0
        print(f"[{i+1}/{len(problems)}] id={prob['id']} answer={answer} "
              f"({elapsed:.0f}s elapsed)")

    submission = pd.DataFrame(rows)
    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSubmission saved: {OUTPUT_CSV}")
    print(submission.head())


if __name__ == "__main__":
    main()
