"""Download the GGUF quant spread + wikitext corpus for a cpu_prober.py run.

Reads GGUF_REPO, GGUF_PREFIX, SOURCES (comma-separated lowercase labels) from
the environment. Writes ggufs/ and work/{calib,eval}.txt plus
work/source_args.txt holding the --source label=path arguments.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


def quant_suffix(label: str) -> str:
    return label if label == "f16" else label.upper()


def main() -> None:
    repo = os.environ["GGUF_REPO"]
    prefix = os.environ["GGUF_PREFIX"]
    labels = sorted({part.strip() for part in os.environ["SOURCES"].split(",") if part.strip()})

    work = Path("work")
    work.mkdir(exist_ok=True)
    args = []
    for label in labels:
        path = hf_hub_download(repo_id=repo, filename=f"{prefix}-{quant_suffix(label)}.gguf", local_dir="ggufs")
        args.append(f"--source {label}={path}")
        print(f"[fetch] {label} -> {path}")
    (work / "source_args.txt").write_text(" ".join(args))

    from datasets import load_dataset

    for split, name, max_kb in [("train", "calib.txt", 256), ("test", "eval.txt", 256)]:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        out, size = [], 0
        for row in ds:
            text = row["text"].strip()
            if not text:
                continue
            out.append(text)
            size += len(text) + 1
            if size > max_kb * 1024:
                break
        (work / name).write_text("\n".join(out))
        print(f"[fetch] {name}: {size/1024:.0f} KB")


if __name__ == "__main__":
    main()
