"""PMRA-LoRA: build a llama.cpp LoRA adapter that corrects quantization error.

For each eligible 2D tensor, the residual R = dequant(ref) - dequant(base) is
factored by SVD in the imatrix-weighted norm (whiten input channels by
s_j = sqrt(E[x_j^2]); Eckart-Young then minimizes the *functional* error
E||(C - R) x||^2 over rank-r corrections C, assuming diagonal input
covariance). Ranks are allocated across tensors by greedy water-filling on
marginal weighted-SSE per byte under a total sidecar byte budget.

The output is a standard llama.cpp adapter GGUF (general.type=adapter,
adapter.type=lora, tensors <name>.lora_a/.lora_b). alpha is written as 0.0,
which llama.cpp interprets as scale = adapter_scale (1.0 by default) for any
per-tensor rank — the exact correction is baked into the factors, so no
rank-dependent rescaling applies. Use:

    llama-perplexity -m base.gguf --lora sidecar.gguf -f eval.txt

Composes with PMRA format promotions: fit the sidecar against a promoted mix
as --base to correct only the residual the promotions leave behind.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from gguf import GGUFReader, GGUFValueType, GGUFWriter

from cpu_prober import dequant, load_imatrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="Quantized GGUF the adapter will be applied to.")
    parser.add_argument("--ref", type=Path, required=True, help="Near-lossless reference GGUF (f16/Q8_0).")
    parser.add_argument("--imatrix", type=Path, required=True, help="GGUF-format imatrix for input-channel weighting.")
    parser.add_argument("--budget-mb", type=float, default=2.0, help="Total sidecar payload budget (lora_a+lora_b, f16).")
    parser.add_argument("--rank-cap", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    base_reader = GGUFReader(str(args.base))
    ref_reader = GGUFReader(str(args.ref))
    arch = str(base_reader.fields["general.architecture"].contents())
    base = {t.name: t for t in base_reader.tensors}
    ref = {t.name: t for t in ref_reader.tensors}
    imatrix = load_imatrix(args.imatrix)

    # Weighted SVD of every eligible residual
    factors: dict[str, dict] = {}
    total_err = 0.0
    for name, bt in base.items():
        if not name.startswith("blk.") or name not in ref or name not in imatrix:
            continue  # token_embd is special-cased in llama.cpp (flipped A/B); out of scope here
        if bt.tensor_type == ref[name].tensor_type and int(bt.n_bytes) == int(ref[name].n_bytes):
            continue  # base already at reference precision
        residual = dequant(ref[name]).astype(np.float64) - dequant(bt).astype(np.float64)
        if residual.ndim != 2 or residual.shape[1] != imatrix[name].shape[0]:
            continue
        weights = imatrix[name]
        scale = np.sqrt(np.maximum(weights, weights.max() * 1e-12 if weights.max() > 0 else 1.0))
        u, s, vt = np.linalg.svd(residual * scale[None, :], full_matrices=False)
        factors[name] = {"u": u, "s": s, "vt": vt, "inv_scale": 1.0 / scale,
                         "d_out": residual.shape[0], "d_in": residual.shape[1],
                         "bytes_per_rank": 2 * (residual.shape[0] + residual.shape[1])}
        total_err += float((s ** 2).sum())

    print(f"[sidecar] {len(factors)} eligible tensors, total weighted SSE: {total_err:.1f}")

    # Greedy water-filling on marginal sigma^2 per byte
    items = []
    for name, f in factors.items():
        for r in range(min(args.rank_cap, len(f["s"]))):
            items.append((float(f["s"][r] ** 2) / f["bytes_per_rank"], name, r))
    items.sort(key=lambda it: it[0], reverse=True)
    budget = int(args.budget_mb * 1e6)
    ranks: dict[str, int] = {}
    used, removed = 0, 0.0
    for gain_per_byte, name, r in items:
        cost = factors[name]["bytes_per_rank"]
        if used + cost > budget:
            continue
        if ranks.get(name, 0) != r:
            continue  # ranks must grow contiguously per tensor
        ranks[name] = r + 1
        used += cost
        removed += float(factors[name]["s"][r] ** 2)
    print(f"[sidecar] allocated {sum(ranks.values())} total ranks over {len(ranks)} tensors; "
          f"{used/1e6:.2f}/{budget/1e6:.2f} MB; predicted weighted-SSE removal "
          f"{removed/total_err*100:.1f}% of base residual")

    # Emit adapter GGUF
    writer = GGUFWriter(args.output, arch)
    writer.add_key_value("general.type", "adapter", GGUFValueType.STRING)
    writer.add_key_value("adapter.type", "lora", GGUFValueType.STRING)
    # alpha=0 => llama.cpp uses scale = adapter_scale (1.0) for any rank; the
    # exact correction is baked into the factors.
    writer.add_key_value("adapter.lora.alpha", 0.0, GGUFValueType.FLOAT32)
    writer.add_key_value("general.name", f"pmra-sidecar-{args.budget_mb:g}mb", GGUFValueType.STRING)

    report_rows = []
    for name, rank in sorted(ranks.items(), key=lambda kv: -kv[1]):
        f = factors[name]
        sqrt_s = np.sqrt(f["s"][:rank])
        # C = (u sqrt(s)) @ (sqrt(s) vt / scale) reproduces the unweighted correction
        lora_b = (f["u"][:, :rank] * sqrt_s[None, :]).astype(np.float16)          # (d_out, r)
        lora_a = (sqrt_s[:, None] * f["vt"][:rank, :] * f["inv_scale"][None, :]).astype(np.float16)  # (r, d_in)
        writer.add_tensor(f"{name}.lora_a", lora_a)
        writer.add_tensor(f"{name}.lora_b", lora_b)
        captured = float((f["s"][:rank] ** 2).sum())
        report_rows.append({"tensor": name, "rank": rank,
                            "bytes": rank * f["bytes_per_rank"],
                            "weighted_sse_removed": captured,
                            "tensor_weighted_sse": float((f["s"] ** 2).sum())})
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    print(f"[sidecar] wrote {args.output} ({args.output.stat().st_size/1e6:.2f} MB on disk)")

    if args.report:
        args.report.write_text(json.dumps({
            "base": str(args.base), "ref": str(args.ref), "imatrix": str(args.imatrix),
            "budget_mb": args.budget_mb, "rank_cap": args.rank_cap,
            "payload_bytes": used,
            "total_weighted_sse": total_err,
            "predicted_removal_fraction": removed / total_err if total_err else 0.0,
            "tensors": report_rows,
        }, indent=2))
        print(f"[sidecar] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
