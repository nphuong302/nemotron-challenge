"""
Supervised Fine-Tuning (SFT) with LoRA adapters on the augmented corpus.
Uses standard HuggingFace Trainer (no TRL dependency) for version robustness.

Pipeline:
  corpus.jsonl --> train_sft.py --> adapter/ (LoRA weights)
"""

import json
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ---------- config ----------
MODEL_ID     = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
CORPUS_PATH  = "corpus.jsonl"
OUTPUT_DIR   = "adapter"

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

EPOCHS       = 3
BATCH_SIZE   = 1
GRAD_ACCUM   = 8
LR           = 2e-4
MAX_SEQ_LEN  = 512   # Mamba SSM G_intermediate OOMs at 768 during backward on 24GB L4
# ----------------------------

SYSTEM = "detailed thinking on"

# Full prompt template (system + user + assistant response)
PROMPT_TEMPLATE = """\
<|system|>
{system}<|end|>
<|user|>
{problem}<|end|>
<|assistant|>
{solution}
\\boxed{{{answer}}}<|end|>"""

# Prompt prefix only (everything the model should NOT learn to generate)
PREFIX_TEMPLATE = """\
<|system|>
{system}<|end|>
<|user|>
{problem}<|end|>
<|assistant|>
"""


def load_corpus(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


@dataclass
class MaskingCollator:
    """Pad a batch and keep labels=-100 for prompt/padding tokens."""
    pad_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }


def make_tokenize_fn(tokenizer):
    """Returns a map function that tokenizes text and masks the prompt prefix."""
    def _fn(example):
        full_ids = tokenizer.encode(
            example["full_text"], add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LEN
        )
        prefix_ids = tokenizer.encode(example["prefix_text"], add_special_tokens=False)
        # prefix_len is the number of tokens to mask (prompt tokens)
        prefix_len = min(len(prefix_ids), len(full_ids))

        lbl = list(full_ids)
        for k in range(prefix_len):
            lbl[k] = -100

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": lbl,
        }
    return _fn


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw_corpus = load_corpus(CORPUS_PATH)
    rows = []
    for ex in raw_corpus:
        rows.append({
            "full_text": PROMPT_TEMPLATE.format(
                system=SYSTEM,
                problem=ex["prompt"],
                solution=ex["solution"],
                answer=ex["answer"],
            ),
            "prefix_text": PREFIX_TEMPLATE.format(
                system=SYSTEM,
                problem=ex["prompt"],
            ),
        })
    dataset = Dataset.from_list(rows)
    print(f"Training on {len(dataset)} examples")

    dataset = dataset.map(
        make_tokenize_fn(tokenizer),
        remove_columns=["full_text", "prefix_text"],
    )

    # Sanity check: verify masking works
    sample = dataset[0]
    n_total = len(sample["labels"])
    n_unmasked = sum(1 for l in sample["labels"] if l != -100)
    print(f"Sample 0: {n_total} total tokens, {n_unmasked} unmasked (assistant response tokens)")
    if n_unmasked == 0:
        print("WARNING: no unmasked tokens in sample 0 — check prefix tokenization!")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    collator = MaskingCollator(pad_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Adapter saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
