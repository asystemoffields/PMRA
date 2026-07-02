"""Pre-launch check: does cpu_prober's tensor->group mapping cover a GGUF?

Parses only the GGUF header — from a local file or a remote URL via HTTP
Range requests (a few MB, not the whole multi-GB file) — and reports, for a
given --tensor-profile/--group-mode, which tensors map to a promotion group,
which stay unmapped, and the byte coverage. Run this against the real GGUF
BEFORE spending a Kaggle session on a new architecture: a profile gap shows
up here as unmapped ssm_*/attn_* tails, not as a silent shrunken mix.

    python tools/check_gguf_grouping.py \
        --url https://huggingface.co/<repo>/resolve/main/<file>.gguf \
        --tensor-profile nemotron_h --group-mode tensor
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

GGUF_MAGIC = b"GGUF"
# value-type ids from the GGUF spec
_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_T_STRING, _T_ARRAY = 8, 9


class _Buf:
    def __init__(self, fetch, size: int):
        self.fetch = fetch          # (start, end) -> bytes
        self.size = size            # total remote/local size
        self.data = b""
        self.pos = 0

    def need(self, n: int) -> None:
        while len(self.data) < self.pos + n:
            grow = max(4 << 20, self.pos + n - len(self.data))
            chunk = self.fetch(len(self.data), min(self.size, len(self.data) + grow) - 1)
            if not chunk:
                raise EOFError("ran out of file while parsing GGUF header")
            self.data += chunk

    def read(self, n: int) -> bytes:
        self.need(n)
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        return self.read(self.u64()).decode("utf-8", errors="replace")

    def skip_value(self, vtype: int) -> None:
        if vtype in _SCALAR_SIZES:
            self.read(_SCALAR_SIZES[vtype])
        elif vtype == _T_STRING:
            self.string()
        elif vtype == _T_ARRAY:
            elem_type, count = self.u32(), self.u64()
            if elem_type in _SCALAR_SIZES:
                self.read(_SCALAR_SIZES[elem_type] * count)
            else:
                for _ in range(count):
                    self.skip_value(elem_type)
        else:
            raise ValueError(f"unknown GGUF value type {vtype}")


def parse_header(buf: _Buf) -> list[dict]:
    if buf.read(4) != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    version = buf.u32()
    if version < 2:
        raise ValueError(f"GGUF v{version} not supported")
    tensor_count, kv_count = buf.u64(), buf.u64()
    for _ in range(kv_count):
        buf.string()
        buf.skip_value(buf.u32())
    tensors = []
    for _ in range(tensor_count):
        name = buf.string()
        n_dims = buf.u32()
        dims = [buf.u64() for _ in range(n_dims)]
        dtype = buf.u32()
        offset = buf.u64()
        tensors.append({"name": name, "dims": dims, "dtype": dtype, "offset": offset})
    # per-tensor sizes from consecutive data offsets (header order == offset
    # order in practice; sort defensively). Last tensor: bounded by file end.
    by_offset = sorted(tensors, key=lambda t: t["offset"])
    data_span = buf.size - (by_offset[0]["offset"] if by_offset else 0)
    for cur, nxt in zip(by_offset, by_offset[1:]):
        cur["bytes"] = nxt["offset"] - cur["offset"]
    if by_offset:
        by_offset[-1]["bytes"] = data_span - (by_offset[-1]["offset"] - by_offset[0]["offset"])
    return tensors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--gguf", type=Path, help="Local GGUF path.")
    src.add_argument("--url", help="Remote GGUF URL (fetched via Range requests).")
    parser.add_argument("--tensor-profile", default="qwen")
    parser.add_argument("--group-mode", default="layer_family", choices=["layer_family", "tensor"])
    parser.add_argument("--show-groups", action="store_true", help="Print every group with its tensors.")
    args = parser.parse_args()

    if args.gguf:
        size = args.gguf.stat().st_size
        fh = open(args.gguf, "rb")

        def fetch(start: int, end: int) -> bytes:
            fh.seek(start)
            return fh.read(end - start + 1)
    else:
        head = urllib.request.Request(args.url, method="HEAD")
        with urllib.request.urlopen(head) as resp:
            size = int(resp.headers["Content-Length"])

        def fetch(start: int, end: int) -> bytes:
            req = urllib.request.Request(args.url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req) as resp:
                return resp.read()

    tensors = parse_header(_Buf(fetch, size))

    from cpu_prober import group_for_tensor

    groups: dict[str, list[dict]] = defaultdict(list)
    unmapped: list[dict] = []
    for t in tensors:
        group = group_for_tensor(t["name"], args.group_mode, args.tensor_profile)
        (groups[group].append(t) if group else unmapped.append(t))

    total = sum(t["bytes"] for t in tensors)
    mapped = sum(t["bytes"] for ts in groups.values() for t in ts)
    print(f"{len(tensors)} tensors, {total/1e9:.2f} GB data; profile={args.tensor_profile} mode={args.group_mode}")
    print(f"mapped: {sum(len(ts) for ts in groups.values())} tensors in {len(groups)} groups, "
          f"{mapped/1e9:.2f} GB ({100*mapped/total:.1f}% of bytes)")
    print(f"unmapped: {len(unmapped)} tensors, {sum(t['bytes'] for t in unmapped)/1e9:.3f} GB")
    tails = defaultdict(int)
    for t in unmapped:
        parts = t["name"].split(".")
        tails[".".join(parts[2:]) if t["name"].startswith("blk.") else t["name"]] += 1
    for tail, count in sorted(tails.items()):
        print(f"  unmapped tail: {tail} x{count}")
    if args.show_groups:
        for group in sorted(groups):
            names = ", ".join(t["name"] for t in groups[group])
            print(f"  {group}: {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
