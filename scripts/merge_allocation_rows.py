"""Merge allocation_rows.jsonl shards from distributed cpu_prober.py workers.

Dedup key is (group, source) — identical to the Kaggle layer-split resume
cell used by the pmra-qwen35-b1/b2 runs. Later inputs win on conflict, so
pass shards in increasing-preference order if it matters (it normally
doesn't: each (group, source) is probed by exactly one shard).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Shard allocation_rows.jsonl files (or dirs to search).")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files: list[Path] = []
    for item in args.inputs:
        if item.is_dir():
            files.extend(sorted(item.rglob("allocation_rows.jsonl")))
        else:
            files.append(item)

    merged: dict[tuple[str, str], str] = {}
    for path in files:
        count = 0
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            merged[(row.get("group"), row.get("source"))] = line
            count += 1
        print(f"[merge] {path}: {count} rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(merged.values()) + "\n")
    print(f"[merge] wrote {len(merged)} unique rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
