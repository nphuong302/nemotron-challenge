"""
Rewrite the curated curriculum into the host's reasoning ("thinking") format.

The scoring backend renders test prompts with the Nemotron chat template in its
default reasoning mode, so the generation prompt ends with:

    <SPECIAL_14>Assistant\n<think>\n

i.e. the model is primed *inside* an open <think> block. Our original targets put
the reasoning as plain text and the answer inline, which would not line up with
that primed prefix (the prompt render would not be a token prefix of the full
render, breaking completion masking).

This script moves each record's reasoning inside an explicit <think>...</think>
block and places the final \boxed{} answer *after* the block:

    <think>
    {reasoning}
    </think>

    \boxed{X}

so the assistant turn begins with "<think>\n", matching the primed prefix, and
the answer is unambiguous for \boxed{} extraction. Every upstream record now
carries a real reasoning trace (build_corpus emits real_trace + real_concise,
synth emits a trace), so an empty think block is unexpected and flagged below.

Input : curated_curriculum.jsonl
Output: curriculum_thinking.jsonl
"""
import json
import re

SRC = "curated_curriculum.jsonl"
DST = "curriculum_thinking.jsonl"

# trailing lead-in phrases to drop from the reasoning once the boxed answer is
# pulled out (avoids a dangling "The answer is" before the closing </think>)
TRAILING = re.compile(
    r"\s*(the\s+answer\s+is|final\s+answer|therefore|thus|so|hence|answer)\s*[:=]?\s*$",
    re.IGNORECASE,
)


def last_boxed_span(s):
    """Span (start, end) of the last balanced \\boxed{...}; end is past the brace."""
    idx = s.rfind(r"\boxed{")
    if idx == -1:
        return None
    i = idx + len(r"\boxed{")
    depth = 1
    while i < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[i], 0)
        i += 1
    return (idx, i) if depth == 0 else None


def to_thinking(content):
    sp = last_boxed_span(content)
    if sp is None:
        raise ValueError("no balanced \\boxed{} in: " + content[:80])
    answer = content[sp[0]:sp[1]]
    reasoning = TRAILING.sub("", content[:sp[0]].rstrip()).rstrip()
    inner = f"\n{reasoning}\n" if reasoning else "\n"
    return f"<think>{inner}</think>\n\n{answer}"


def main():
    n = empty = 0
    with open(SRC, encoding="utf-8") as fin, open(DST, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            msgs = rec["messages"]
            assert msgs[-1]["role"] == "assistant", "last turn must be assistant"
            new = to_thinking(msgs[-1]["content"])
            if new.startswith("<think>\n</think>"):
                empty += 1
            msgs[-1]["content"] = new
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} records to {DST} ({empty} empty-think, {n - empty} with reasoning)")
    if empty:
        print(f"WARNING: {empty} empty-think records remain — every record should "
              f"carry reasoning now; check build_corpus/synth output.")


if __name__ == "__main__":
    main()
