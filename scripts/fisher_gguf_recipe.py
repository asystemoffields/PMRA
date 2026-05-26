#!/usr/bin/env python3
"""Fisher-guided GGUF quantization recipe generator.

Applies the PMRA principle (variable allocation based on per-tensor importance)
to GGUF quantization: allocate more bits to tensors that matter most and fewer
bits to tensors that are highly compressible.

== What This Does ==

Given per-tensor profiling data (SVD metrics, Fisher sensitivity, weight
distribution stats) and a target bits-per-weight budget, this script:

1. Computes a composite importance score per tensor using:
   - Stable rank (effective dimensionality — more dimensions = more bits needed)
   - Condition number (numerical sensitivity — higher = more fragile)
   - Weight distribution (outliers, kurtosis — harder distributions need more bits)
   - Entropy utilization (how efficiently each tensor uses N-bit quantization)
   - Layer position (bathtub curve: edge layers are critical, middles are compressible)
   - Matrix role (o_proj needs precision; gate/up_proj are robust)

2. Solves a Multiple-Choice Knapsack Problem (MCKP) via Lagrangian relaxation:
   minimize  Σ importance(i) × quant_error(i, type_i)
   subject to  Σ bytes(i, type_i) ≤ budget

3. Outputs a tensor-type file for `llama-quantize --tensor-type-file`

== What Profiling Data Is Needed ==

A JSON file (per_tensor.json) with one entry per weight matrix, keyed by the
HuggingFace parameter name. Each entry must have at minimum:

  {
    "model.layers.0.mlp.down_proj.weight": {
      "shape": [out_dim, in_dim],
      "params": 45088768,
      "layer": 0,                          # null for embed/lm_head
      "role": "mlp_down",                  # one of: embed, lm_head, mlp_down,
                                           #   mlp_gate, mlp_up, attn_q, attn_k,
                                           #   attn_v, attn_o
      "svd": {
        "stable_rank": 41.65,              # effective rank (||σ||₁² / ||σ||₂²)
        "condition_number": 582.7,          # max_sv / min_sv
        "spectral_entropy": 11.85           # -Σ p_i log p_i on normalized sv
      },
      "distribution": {
        "outlier_fraction_3s": 0.0057,     # fraction of weights > 3σ
        "kurtosis": 0.57                    # excess kurtosis (0 = Gaussian)
      },
      "entropy": {                          # optional, improves allocation
        "utilization_4bit": 0.28,          # fraction of 4-bit capacity used
        "utilization_8bit": 0.62
      }
    }
  }

== How To Obtain This Data ==

Run `structured_search.py` or a similar profiler on your model:

  1. Load the model in FP16/BF16
  2. For each weight matrix, compute:
     - SVD: torch.linalg.svdvals(W) → stable_rank, condition_number, spectral_entropy
     - Distribution: W.mean(), W.std(), kurtosis, outlier counts
     - Entropy: histogram W into 2^N bins, compute utilization = H(W) / N
  3. (Optional) For Fisher sensitivity, run a few calibration batches:
     - loss.backward() on calibration data
     - fisher_diag += param.grad ** 2   (accumulate per batch)
     - fisher_weighted_error for each compression candidate
  4. Save as JSON keyed by parameter name

The profiler in this project (structured_search.py) does all of this.

== Usage ==

  python fisher_gguf_recipe.py \\
    --profiling results/think/per_tensor.json \\
    --target-bpw 3.0 \\
    --output recipe_3.0bpw.txt \\
    --summary recipe_3.0bpw.json

  # Then feed to llama-quantize:
  llama-quantize --imatrix imatrix.gguf \\
    --tensor-type-file recipe_3.0bpw.txt \\
    model-f16.gguf model-fisher-3.0.gguf Q4_K_M
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# GGUF quantization types and their bits-per-weight
# ---------------------------------------------------------------------------

QUANT_TYPES: list[tuple[str, float]] = [
    ("iq2_xxs", 2.0625),
    ("iq2_xs",  2.3125),
    ("q2_k",    2.625),
    ("iq2_m",   2.7),
    ("iq3_xxs", 3.0625),
    ("q3_k",    3.4375),
    ("iq3_m",   3.75),
    ("iq4_xs",  4.25),
    ("q4_k",    4.5),
    ("q5_k",    5.5),
    ("q6_k",    6.5625),
    ("q8_0",    8.5),
]


# ---------------------------------------------------------------------------
# HuggingFace → GGUF tensor name mapping
# ---------------------------------------------------------------------------

# Standard LLaMA-family mapping. Works for LLaMA, Mistral, OLMo, Qwen, Gemma, etc.
# Extend this dict for non-standard architectures.
HF_MODULE_TO_GGUF = {
    "self_attn.q_proj":   "attn_q",
    "self_attn.k_proj":   "attn_k",
    "self_attn.v_proj":   "attn_v",
    "self_attn.o_proj":   "attn_output",
    "mlp.gate_proj":      "ffn_gate",
    "mlp.up_proj":        "ffn_up",
    "mlp.down_proj":      "ffn_down",
    # GQA / MQA variants
    "self_attn.qkv_proj": "attn_qkv",
    # Phi-style
    "self_attn.dense":    "attn_output",
    "mlp.fc1":            "ffn_up",
    "mlp.fc2":            "ffn_down",
}

# Role names from profiling → GGUF module names
ROLE_TO_GGUF_MODULE = {
    "embed":    "token_embd",
    "lm_head":  "output",
    "mlp_down": "ffn_down",
    "mlp_gate": "ffn_gate",
    "mlp_up":   "ffn_up",
    "attn_q":   "attn_q",
    "attn_k":   "attn_k",
    "attn_v":   "attn_v",
    "attn_o":   "attn_output",
}

# Sensitivity multipliers by GGUF module name.
# Derived from:
#   - Fisher profiling data (OLMo, SmolLM experiments)
#   - Unsloth Dynamic 2.0 findings (o_proj bf16 due to no preceding norm)
#   - SwiGLU amplification analysis (down_proj more sensitive than gate/up)
#   - Compound topology theory (attention routing errors amplify through MLP)
ROLE_SENSITIVITY = {
    "attn_output": 3.0,   # o_proj: no preceding norm layer, AWQ can't correct
    "ffn_down":    2.0,    # SwiGLU: gate*up multiply, down scales result back
    "attn_q":      1.5,    # attention routing
    "attn_k":      1.5,    # attention routing
    "attn_v":      1.5,    # attention values
    "attn_qkv":    2.0,    # fused QKV — treat as sensitive
    "ffn_gate":    1.0,    # most compressible
    "ffn_up":      1.0,    # most compressible
    "token_embd":  2.0,    # tiny fraction of params, critical for quality
    "output":      2.5,    # lm_head: tiny, directly affects output distribution
}

# Minimum quant level per role (floor — never go below this BPW).
# Embeddings and lm_head have low SVD complexity but errors cascade through
# the entire model. Unsloth keeps embed at 4-5 bits, lm_head at 6 bits.
# o_proj has no preceding norm layer so quantization error can't be corrected.
ROLE_MIN_BPW = {
    "token_embd":  4.5,    # at least Q4_K
    "output":      5.5,    # at least Q5_K
    "attn_output": 3.4375, # at least Q3_K
}


# ---------------------------------------------------------------------------
# Tensor name conversion
# ---------------------------------------------------------------------------

def hf_name_to_gguf(hf_name: str) -> str | None:
    """Convert HuggingFace parameter name to GGUF tensor name.

    Returns None for non-quantizable tensors (norms, biases).
    """
    name = hf_name.removesuffix(".weight").removesuffix(".bias")

    if "embed_tokens" in name:
        return "token_embd.weight"
    if "lm_head" in name:
        return "output.weight"

    # model.layers.{L}.{module}
    parts = name.split(".")
    try:
        layer_idx = int(parts[2])
    except (IndexError, ValueError):
        return None

    module_path = ".".join(parts[3:])
    gguf_module = HF_MODULE_TO_GGUF.get(module_path)
    if gguf_module is None:
        return None  # norm layer or unrecognized → skip
    return f"blk.{layer_idx}.{gguf_module}.weight"


def gguf_module_from_role(role: str) -> str | None:
    """Map profiling role string to GGUF module name."""
    return ROLE_TO_GGUF_MODULE.get(role)


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------

def bathtub_factor(layer: int | None, n_layers: int) -> float:
    """Edge layers need more bits than middle layers.

    The 'bathtub curve' from Fisher profiling: layers 0-3 and the last 4
    layers are functionally critical (low compressibility), while middle
    layers are 100-700x more compressible.
    """
    if layer is None:
        return 2.0  # embed / lm_head
    frac = layer / max(n_layers - 1, 1)
    if frac <= 0.06:     # first ~2 layers
        return 3.0
    if frac <= 0.12:     # layers 3-4
        return 2.0
    if frac <= 0.18:     # layers 5-6
        return 1.3
    if frac >= 0.94:     # last ~2 layers
        return 3.0
    if frac >= 0.88:     # second-to-last 2
        return 2.0
    if frac >= 0.82:
        return 1.3
    return 1.0            # compressible middle


def compute_importance(
    tensor_data: dict,
    role: str,
    layer: int | None,
    n_layers: int,
) -> float:
    """Compute composite importance score for a tensor.

    Higher importance → more bits allocated.
    """
    gguf_module = gguf_module_from_role(role)
    if gguf_module is None:
        return 0.0

    # --- SVD metrics ---
    svd = tensor_data.get("svd", {})
    stable_rank = svd.get("stable_rank", 100.0)
    cond = svd.get("condition_number", 100.0)

    # --- Distribution metrics ---
    dist = tensor_data.get("distribution", {})
    outliers = dist.get("outlier_fraction_3s", 0.005)
    kurtosis = max(dist.get("kurtosis", 0.0), 0.0)

    # --- Entropy utilization (how efficiently it uses N-bit quantization) ---
    entropy = tensor_data.get("entropy", {})
    util_4bit = entropy.get("utilization_4bit", 0.3)

    # --- Composite score ---
    # Complexity: how many effective dimensions of information
    complexity = math.log2(max(stable_rank, 1.0) + 1)

    # Sensitivity: condition number + outlier penalty
    sensitivity = (
        math.sqrt(math.log2(max(cond, 1.0) + 1))
        * (1.0 + 10.0 * outliers)
        * (1.0 + 0.1 * kurtosis)
    )

    # Entropy factor: tensors that use more of the quantization grid need more bits
    entropy_factor = 1.0 + util_4bit

    # Structural factors
    role_mult = ROLE_SENSITIVITY.get(gguf_module, 1.0)
    edge_mult = bathtub_factor(layer, n_layers)

    return complexity * sensitivity * entropy_factor * role_mult * edge_mult


# ---------------------------------------------------------------------------
# MCKP solver via Lagrangian relaxation
# ---------------------------------------------------------------------------

@dataclass
class TensorAllocation:
    gguf_name: str
    hf_name: str
    role: str
    layer: int | None
    params: int
    importance: float
    quant_type: str
    bpw: float
    nbytes: int
    error: float


def solve_mckp(
    tensors: list[dict],
    candidates: list[tuple[str, float]],
    budget_bytes: int,
) -> list[TensorAllocation]:
    """Solve Multiple-Choice Knapsack via Lagrangian binary search.

    For each tensor, pick one quant type from candidates to minimize
    total weighted quantization error subject to total bytes ≤ budget.

    Error model: importance × 2^(-2 × bpw)
    This follows rate-distortion theory: MSE ∝ σ² × 2^(-2R).
    """
    # Build options for each tensor, respecting per-role minimum BPW floors
    tensor_options = []
    for t in tensors:
        gguf_module = gguf_module_from_role(t["role"])
        min_bpw = ROLE_MIN_BPW.get(gguf_module, 0.0)
        options = []
        for qtype, bpw in candidates:
            if bpw < min_bpw:
                continue  # skip candidates below the floor for this role
            nbytes = int(math.ceil(t["params"] * bpw / 8))
            error = t["importance"] * (2.0 ** (-2.0 * bpw))
            options.append((qtype, bpw, nbytes, error))
        if not options:
            # fallback: use the highest available
            qtype, bpw = candidates[-1]
            nbytes = int(math.ceil(t["params"] * bpw / 8))
            error = t["importance"] * (2.0 ** (-2.0 * bpw))
            options.append((qtype, bpw, nbytes, error))
        tensor_options.append((t, options))

    def allocate_at_lambda(lam: float):
        allocs = []
        total_bytes = 0
        total_error = 0.0
        for t, options in tensor_options:
            best_cost = float("inf")
            best = options[-1]  # fallback: highest quality
            for opt in options:
                cost = opt[3] + lam * opt[2]  # error + lambda * bytes
                if cost < best_cost:
                    best_cost = cost
                    best = opt
            qtype, bpw, nbytes, error = best
            allocs.append(TensorAllocation(
                gguf_name=t["gguf_name"],
                hf_name=t["hf_name"],
                role=t["role"],
                layer=t["layer"],
                params=t["params"],
                importance=t["importance"],
                quant_type=qtype,
                bpw=bpw,
                nbytes=nbytes,
                error=error,
            ))
            total_bytes += nbytes
            total_error += error
        return allocs, total_bytes, total_error

    # Check if minimum possible fits
    _, min_bytes, _ = allocate_at_lambda(1e20)
    if min_bytes > budget_bytes:
        print(f"WARNING: minimum possible size ({min_bytes / 1e9:.2f} GB) exceeds "
              f"budget ({budget_bytes / 1e9:.2f} GB). Using minimum.", file=sys.stderr)
        allocs, _, _ = allocate_at_lambda(1e20)
        return allocs

    # Check if maximum fits within budget (no need to constrain)
    _, max_bytes, _ = allocate_at_lambda(0.0)
    if max_bytes <= budget_bytes:
        allocs, _, _ = allocate_at_lambda(0.0)
        return allocs

    # Binary search on lambda
    lo, hi = 0.0, 1e-6
    _, bytes_hi, _ = allocate_at_lambda(hi)
    while bytes_hi > budget_bytes:
        hi *= 10
        _, bytes_hi, _ = allocate_at_lambda(hi)

    for _ in range(200):
        mid = (lo + hi) / 2.0
        _, bytes_mid, _ = allocate_at_lambda(mid)
        if bytes_mid > budget_bytes:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-20:
            break

    allocs, _, _ = allocate_at_lambda(hi)
    return allocs


# ---------------------------------------------------------------------------
# Recipe generation
# ---------------------------------------------------------------------------

def load_profiling(path: Path) -> tuple[dict, int]:
    """Load profiling JSON. Returns (tensor_dict, n_layers)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Detect n_layers from the data
    max_layer = 0
    for name, tdata in data.items():
        layer = tdata.get("layer")
        if layer is not None:
            max_layer = max(max_layer, layer)
    n_layers = max_layer + 1

    return data, n_layers


def generate_recipe(
    profiling_path: Path,
    target_bpw: float = 3.5,
    target_gb: float | None = None,
    candidates: list[tuple[str, float]] | None = None,
) -> tuple[list[str], dict]:
    """Generate a Fisher-guided per-tensor quantization recipe.

    Returns:
        lines: list of "gguf_tensor_name=quant_type" for tensor-type file
        summary: dict with allocation details and statistics
    """
    if candidates is None:
        candidates = QUANT_TYPES

    profiling, n_layers = load_profiling(profiling_path)

    # Build tensor list with importance scores
    tensors = []
    total_params = 0
    skipped = []

    for hf_name, tdata in profiling.items():
        gguf_name = hf_name_to_gguf(hf_name)
        if gguf_name is None:
            skipped.append(hf_name)
            continue

        role = tdata.get("role", "")
        layer = tdata.get("layer")
        params = tdata["params"]
        importance = compute_importance(tdata, role, layer, n_layers)

        if importance <= 0:
            skipped.append(hf_name)
            continue

        tensors.append({
            "gguf_name": gguf_name,
            "hf_name": hf_name,
            "role": role,
            "layer": layer,
            "params": params,
            "importance": importance,
        })
        total_params += params

    # Compute budget
    if target_gb is not None:
        budget_bytes = int(target_gb * 1e9)
    else:
        budget_bytes = int(total_params * target_bpw / 8)

    print(f"Tensors: {len(tensors)}, Total params: {total_params:,}, "
          f"Budget: {budget_bytes / 1e9:.2f} GB ({target_bpw:.1f} BPW)")

    # Solve knapsack
    allocs = solve_mckp(tensors, candidates, budget_bytes)

    # Build output
    lines = []
    by_type: dict[str, int] = {}
    by_role: dict[str, list[str]] = {}
    total_bytes = 0
    total_error = 0.0
    importance_range = (
        min(a.importance for a in allocs),
        max(a.importance for a in allocs),
    )

    for a in sorted(allocs, key=lambda x: x.gguf_name):
        lines.append(f"{a.gguf_name}={a.quant_type}")
        by_type[a.quant_type] = by_type.get(a.quant_type, 0) + a.params
        role_key = a.role or "unknown"
        by_role.setdefault(role_key, []).append(a.quant_type)
        total_bytes += a.nbytes
        total_error += a.error

    avg_bpw = total_bytes * 8 / total_params if total_params else 0

    # Summary statistics
    type_summary = {}
    for qtype, params in sorted(by_type.items(),
                                 key=lambda x: dict(candidates).get(x[0], 0)):
        bpw = dict(candidates).get(qtype, 0)
        type_summary[qtype] = {
            "params": params,
            "fraction": params / total_params,
            "bpw": bpw,
        }

    role_summary = {}
    for role, qtypes in sorted(by_role.items()):
        from collections import Counter
        counts = Counter(qtypes)
        role_summary[role] = {
            "count": len(qtypes),
            "types": dict(counts),
        }

    summary = {
        "target_bpw": target_bpw,
        "actual_bpw": round(avg_bpw, 4),
        "total_params": total_params,
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / 1e9, 3),
        "total_error": total_error,
        "n_tensors": len(allocs),
        "n_layers": n_layers,
        "importance_range": importance_range,
        "skipped_tensors": len(skipped),
        "type_distribution": type_summary,
        "role_distribution": role_summary,
        "allocations": [
            {
                "tensor": a.gguf_name,
                "hf_name": a.hf_name,
                "role": a.role,
                "layer": a.layer,
                "params": a.params,
                "importance": round(a.importance, 4),
                "quant_type": a.quant_type,
                "bpw": a.bpw,
            }
            for a in sorted(allocs, key=lambda x: -x.importance)
        ],
    }

    return lines, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Fisher-guided GGUF quantization recipe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--profiling", required=True, type=Path,
        help="Path to per_tensor.json profiling data",
    )
    parser.add_argument(
        "--target-bpw", type=float, default=3.5,
        help="Target average bits-per-weight (default: 3.5)",
    )
    parser.add_argument(
        "--target-gb", type=float, default=None,
        help="Target file size in GB (overrides --target-bpw)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output tensor-type file (default: recipe_{bpw}bpw.txt)",
    )
    parser.add_argument(
        "--summary", type=Path, default=None,
        help="Output JSON summary (default: recipe_{bpw}bpw.json)",
    )
    parser.add_argument(
        "--multi", nargs="+", type=float, default=None,
        help="Generate recipes at multiple BPW targets (e.g., --multi 2.5 3.0 3.5 4.0)",
    )

    args = parser.parse_args()

    targets = args.multi if args.multi else [args.target_bpw]

    for target in targets:
        print(f"\n{'='*60}")
        print(f"Generating recipe for {target:.1f} BPW")
        print(f"{'='*60}")

        lines, summary = generate_recipe(
            args.profiling,
            target_bpw=target,
            target_gb=args.target_gb if len(targets) == 1 else None,
        )

        # Output paths
        if args.output and len(targets) == 1:
            out_path = args.output
        else:
            out_path = Path(f"recipe_{target:.1f}bpw.txt")

        if args.summary and len(targets) == 1:
            sum_path = args.summary
        else:
            sum_path = Path(f"recipe_{target:.1f}bpw.json")

        # Write tensor-type file
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        # Write summary
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        # Print highlights
        print(f"\nActual BPW: {summary['actual_bpw']:.3f}")
        print(f"File size:  {summary['total_gb']:.2f} GB")
        print(f"Tensors:    {summary['n_tensors']}")
        print(f"\nType distribution:")
        for qtype, info in summary["type_distribution"].items():
            print(f"  {qtype:10s}: {info['fraction']*100:5.1f}% of params "
                  f"({info['params']:>12,} params)")
        print(f"\nMost important tensors (top 10):")
        for a in summary["allocations"][:10]:
            print(f"  {a['tensor']:40s} -> {a['quant_type']:10s} "
                  f"(importance={a['importance']:8.2f})")
        print(f"\nLeast important tensors (bottom 5):")
        for a in summary["allocations"][-5:]:
            print(f"  {a['tensor']:40s} -> {a['quant_type']:10s} "
                  f"(importance={a['importance']:8.2f})")

        print(f"\nRecipe written to: {out_path}")
        print(f"Summary written to: {sum_path}")


if __name__ == "__main__":
    main()
