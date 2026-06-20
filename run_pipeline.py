"""
One-command local build of the final training curriculum.

Runs the four data stages in order, each as its own process (so a failure stops
the run and the output mirrors running the steps by hand):

    1. build_corpus.py             train.csv             -> real_curriculum.jsonl
    2. synth.py                    (templated synthesis)  -> synth_curriculum.jsonl
    3. curate.py                   real + synth           -> curated_curriculum.jsonl
    4. build_thinking_curriculum.py curated               -> curriculum_thinking.jsonl

This is for LOCAL validation that the data builds cleanly. The Kaggle notebook
(`kaggle_remote/`) is self-contained: it embeds these same .py files and rebuilds
the curriculum from the competition train.csv on Kaggle, so there is no separate
dataset to upload. See SUBMIT_TO_KAGGLE.md.

Usage:
    python run_pipeline.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
STAGES = [
    "build_corpus.py",
    "synth.py",
    "curate.py",
    "build_thinking_curriculum.py",
]
FINAL = ROOT / "curriculum_thinking.jsonl"


def run(script):
    print(f"\n=== {script} ===", flush=True)
    result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"stage failed: {script} (exit {result.returncode})")


def main():
    for script in STAGES:
        run(script)

    n = sum(1 for line in FINAL.open(encoding="utf-8") if line.strip())
    print(f"\nbuilt {FINAL.name}: {n} records")
    print("local build OK. To submit, push the self-contained notebook "
          "(see SUBMIT_TO_KAGGLE.md).")


if __name__ == "__main__":
    main()
