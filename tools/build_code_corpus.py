"""Build the code-guardrail corpus for the CPU prober.

Interleaves MBPP-sanitized (test split) and HumanEval — the same ungated
benchmarks and task formatting as scripts/evaluate_pmra_code_likelihood.py —
into one plain-text file for llama-perplexity. Interleaved (proportional
round-robin), not concatenated: guardrail evals read only the first
chunks×ctx tokens of the file, and a concatenated corpus would never reach
the second benchmark. Deterministic: tasks sorted by task_id, no sampling,
so every run and every kernel sees the same bytes.

Only needs `datasets` (no torch), so it runs in the prober's CPU-kernel env.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from datasets import load_dataset


def mbpp_texts() -> list[str]:
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    rows = sorted((dict(row) for row in ds), key=lambda row: int(row["task_id"]))
    texts = []
    for row in rows:
        tests = "\n".join(row.get("test_list") or [])
        texts.append(
            "# Task\n"
            f"{row['prompt'].strip()}\n\n"
            "# Required behavior\n"
            f"{tests.strip()}\n\n"
            "# Solution\n"
            f"{row['code'].strip()}\n"
        )
    return texts


def humaneval_texts() -> list[str]:
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = sorted((dict(row) for row in ds), key=lambda row: row["task_id"])
    return [row["prompt"] + row["canonical_solution"].strip("\n") + "\n" for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mbpp, he = mbpp_texts(), humaneval_texts()
    # proportional round-robin: tag each task with its fractional position and
    # merge-sort, so any prefix of the file samples both benchmarks
    tagged = ([(i / len(mbpp), 0, t) for i, t in enumerate(mbpp)]
              + [((i + 0.5) / len(he), 1, t) for i, t in enumerate(he)])
    tagged.sort(key=lambda item: (item[0], item[1]))
    texts = [t for _, _, t in tagged]
    corpus = "\n\n".join(texts)
    args.output.write_text(corpus, encoding="utf-8")
    digest = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
    print(f"code corpus: {len(texts)} tasks, {len(corpus)/1024:.0f} KB, sha256={digest[:16]}… -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
