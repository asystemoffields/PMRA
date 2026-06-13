"""
DSmolLM — Apply a Per-Everything allocation to actually compress a model.

Takes the allocator's optimal allocation and applies each compression
to the real weights, then evaluates the compound PPL.
"""

import argparse
import gc
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from structured_search import (
    get_data, eval_perplexity, compute_fisher_for_matrix,
    fit_low_rank, fit_low_rank_fisher, fit_low_rank_plus_sparse,
    fit_kronecker, fit_kronecker_sum, fit_block_diagonal,
    fit_tensor_train, fit_codebook, fit_codebook_fisher,
)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(write_through=True)
    except Exception:
        pass


def parse_family(family_name, original_params, compressed_params, m, n):
    """Parse a family string back into (fit_function, kwargs) for reconstruction."""
    if family_name == "original":
        return None, {}

    if family_name == "low_rank":
        rank = compressed_params // (m + n)
        return fit_low_rank, {"rank": rank}

    if family_name == "fisher_lr":
        rank = compressed_params // (m + n)
        return "fisher_lr", {"rank": rank}

    if family_name.startswith("tt_rank"):
        max_rank = int(family_name.replace("tt_rank", ""))
        return fit_tensor_train, {"tt_ranks": [max_rank] * 6}

    if family_name.startswith("kron_") and "sum" not in family_name:
        parts = family_name.replace("kron_", "").split("x")
        return fit_kronecker, {"block_m": int(parts[0]), "block_n": int(parts[1])}

    if family_name.startswith("kronsum_"):
        parts = family_name.replace("kronsum_", "").split("x")
        return fit_kronecker_sum, {
            "block_m": int(parts[0]), "block_n": int(parts[1]), "n_terms": int(parts[2])
        }

    if family_name.startswith("block_diag_"):
        bs = int(family_name.replace("block_diag_", ""))
        return fit_block_diagonal, {"block_size": bs}

    if family_name.startswith("codebook_"):
        parts = family_name.replace("codebook_", "").split("x")
        return fit_codebook, {"n_codebooks": int(parts[0]), "codebook_size": int(parts[1])}

    if family_name.startswith("fisher_cb_"):
        parts = family_name.replace("fisher_cb_", "").split("x")
        return "fisher_cb", {"n_codebooks": int(parts[0]), "codebook_size": int(parts[1])}

    lr_sparse = re.match(r"lr(\d+)\+sparse(\d+)%", family_name)
    if lr_sparse:
        rank = int(lr_sparse.group(1))
        pct = int(lr_sparse.group(2)) / 100.0
        top_k = int(original_params * pct)
        return fit_low_rank_plus_sparse, {"rank": rank, "sparsity_top_k": top_k}

    print(f"    WARNING: unknown family '{family_name}', skipping")
    return None, {}


def apply_allocation(model, allocation, calib_data, device):
    """Apply compression allocation to the model in-place."""
    param_dict = dict(model.named_parameters())
    total_original = 0
    total_compressed = 0
    total_applied = 0
    total_skipped = 0

    for entry in allocation:
        family = entry["family"]
        if family == "original":
            total_original += entry["original_params"]
            total_compressed += entry["original_params"]
            total_skipped += 1
            continue

        group_id = entry["group_id"]
        layer = entry["layer"]
        mtype = entry["matrix_type"]

        # Find the actual parameter name
        param_name = None
        for name in param_dict:
            if f"layers.{layer}." in name and mtype in name and "weight" in name:
                param_name = name
                break

        if param_name is None:
            print(f"    {group_id}: param not found, skipping")
            total_skipped += 1
            total_original += entry["original_params"]
            total_compressed += entry["original_params"]
            continue

        param = param_dict[param_name]
        W = param.data.float().cpu()
        m, n = W.shape

        fit_fn, kwargs = parse_family(family, entry["original_params"], entry["compressed_params"], m, n)
        if fit_fn is None:
            total_skipped += 1
            total_original += entry["original_params"]
            total_compressed += entry["original_params"]
            continue

        try:
            if fit_fn == "fisher_lr":
                fisher = compute_fisher_for_matrix(model, calib_data, device, param_name)
                approx, stored, _ = fit_low_rank_fisher(W, kwargs["rank"], fisher)
                del fisher
            elif fit_fn == "fisher_cb":
                fisher = compute_fisher_for_matrix(model, calib_data, device, param_name)
                approx, stored, _ = fit_codebook_fisher(
                    W, kwargs["n_codebooks"], kwargs["codebook_size"], fisher)
                del fisher
            else:
                approx, stored, _ = fit_fn(W, **kwargs)

            if approx is None:
                print(f"    {group_id}: fit returned None, skipping")
                total_skipped += 1
                total_original += entry["original_params"]
                total_compressed += entry["original_params"]
                continue

            param.data.copy_(approx.to(param.dtype).to(param.device))
            total_applied += 1
            total_original += entry["original_params"]
            total_compressed += entry["compressed_params"]

            ratio = entry["original_params"] / entry["compressed_params"]
            print(f"    {group_id:<20} {family:<22} {ratio:>6.1f}x  applied")

        except Exception as e:
            print(f"    {group_id}: FAILED ({e}), keeping original")
            total_skipped += 1
            total_original += entry["original_params"]
            total_compressed += entry["original_params"]

        del W
        if total_applied % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n    Applied: {total_applied}, Skipped: {total_skipped}")
    print(f"    Original params: {total_original:,}")
    print(f"    Compressed params: {total_compressed:,}")
    print(f"    Effective ratio: {total_original / total_compressed:.2f}x")
    return total_original, total_compressed


def main(commit_fn=None):
    parser = argparse.ArgumentParser(description="DSmolLM — Compress a model per allocation")
    parser.add_argument("--allocation", type=str, required=True,
                        help="Allocator JSON output (contains optimal.allocations)")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--output", type=str, default="results/dsmollm")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-eval", type=int, default=128)
    parser.add_argument("--n-calib", type=int, default=64)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  DSmolLM — Whole-Model Compression Test")
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    print(f"{'=' * 70}")

    with open(args.allocation) as f:
        alloc_data = json.load(f)

    allocation = alloc_data["optimal"]["allocations"]
    est_dloss = alloc_data["optimal"]["total_dloss"]
    est_ppl = alloc_data["optimal"]["est_ppl"]
    target_ratio = alloc_data["target_layer_ratio"]

    n_to_compress = sum(1 for a in allocation if a["family"] != "original")
    print(f"\n  Allocation: {target_ratio}x target, {len(allocation)} matrices, "
          f"{n_to_compress} to compress")
    print(f"  Estimated dloss: {est_dloss:+.4f}, Est. PPL: {est_ppl:.2f}")

    print(f"\n  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    model.eval()

    print(f"\n  Loading data...")
    calib_data = get_data(tokenizer, "train", n_samples=args.n_calib, skip=0)
    eval_data = get_data(tokenizer, "test", n_samples=args.n_eval, skip=args.n_calib)

    print(f"\n  Measuring baseline...")
    base_ppl, base_loss = eval_perplexity(model, eval_data, device)
    print(f"    Baseline: PPL={base_ppl:.2f}, loss={base_loss:.4f}")

    print(f"\n  Applying allocation ({n_to_compress} matrices)...")
    t0 = time.time()
    orig, compressed = apply_allocation(model, allocation, calib_data, device)
    elapsed = time.time() - t0
    print(f"    Compression took {elapsed:.1f}s")

    print(f"\n  Evaluating compressed model...")
    final_ppl, final_loss = eval_perplexity(model, eval_data, device)
    actual_dloss = final_loss - base_loss

    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Baseline PPL:      {base_ppl:>10.2f}")
    print(f"  Compressed PPL:    {final_ppl:>10.2f}")
    print(f"  Actual dloss:      {actual_dloss:>+10.4f}")
    print(f"  Estimated dloss:   {est_dloss:>+10.4f}")
    print(f"  Estimate error:    {(est_dloss - actual_dloss):>+10.4f} "
          f"({'over' if est_dloss > actual_dloss else 'under'}estimated)")
    print(f"  Target ratio:      {target_ratio:>10.1f}x")
    print(f"  Achieved ratio:    {orig / compressed:>10.2f}x")

    results = {
        "model": args.model,
        "target_ratio": target_ratio,
        "baseline_ppl": base_ppl,
        "baseline_loss": base_loss,
        "compressed_ppl": final_ppl,
        "compressed_loss": final_loss,
        "actual_dloss": actual_dloss,
        "estimated_dloss": est_dloss,
        "estimated_ppl": est_ppl,
        "original_params": orig,
        "compressed_params": compressed,
        "achieved_ratio": orig / compressed,
        "n_compressed": n_to_compress,
        "n_total": len(allocation),
    }

    results_path = output_path / f"dsmollm_{target_ratio}x.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {results_path}")

    if commit_fn:
        try:
            commit_fn()
        except Exception:
            pass


if __name__ == "__main__":
    main()
