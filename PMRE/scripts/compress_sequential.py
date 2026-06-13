"""
Sequential SAES-SVD Compression — Cumulative Error-Aware Low-Rank Compression.

Implements the core idea from SAES-SVD (arxiv 2602.03051): compress layers
front-to-back, using BOTH the compressed activations AND the full-precision
reference to correct for upstream error accumulation.

Key formula:  G = W(H + βΔ)H^{-1/2}
  H = XX^T  (compressed activation covariance)
  Δ = (X^f - X)X^T  (error cross-covariance)
  β ∈ [0,1]  (correction strength, auto-tuned via ACES)

Truncated SVD of G gives the optimal rank-r factors.
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
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from structured_search import get_data, eval_perplexity

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(write_through=True)
    except Exception:
        pass


def collect_layer_activations(model, data, device, batch_size=8):
    """Run forward pass, collect input activations at every transformer layer."""
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            if name not in activations:
                activations[name] = []
            activations[name].append(x.detach().cpu())
        return hook

    hooks = []
    for name, module in model.named_modules():
        if re.match(r"model\.layers\.\d+$", name):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].to(device)
            model(input_ids=batch)

    for h in hooks:
        h.remove()

    result = {}
    for name, chunks in activations.items():
        result[name] = torch.cat(chunks, dim=0)  # (N, seq_len, hidden_dim)
    return result


def compute_covariance_stats(X_compressed, X_full):
    """Compute H = XX^T and Δ = (X^f - X)X^T from activation tensors.
    X shapes: (N, seq_len, hidden_dim) → flatten to (N*seq_len, hidden_dim)."""
    Xc = X_compressed.reshape(-1, X_compressed.shape[-1]).float()  # (T, d)
    Xf = X_full.reshape(-1, X_full.shape[-1]).float()              # (T, d)

    H = (Xc.T @ Xc) / Xc.shape[0]       # (d, d)
    diff = Xf - Xc                        # (T, d)
    delta = (diff.T @ Xc) / Xc.shape[0]  # (d, d)

    return H, delta


def stable_inv_sqrt(H):
    """Compute H^{-1/2} via SVD with relative eigenvalue clipping."""
    U, S, Vt = torch.linalg.svd(H, full_matrices=False)
    threshold = S.max() * 1e-4
    S_clipped = S.clamp(min=threshold)
    return U @ torch.diag(1.0 / S_clipped.sqrt()) @ Vt


def compute_aces_beta(W, H, delta, rank, n_candidates=20):
    """ACES: find β that maximizes retained energy ratio after rank-r truncation."""
    scale = max(H.trace().item() / H.shape[0], 1e-8)
    reg = 1e-3 * scale * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
    H_reg = H + reg
    H_inv_sqrt = stable_inv_sqrt(H_reg)

    best_beta = 0.0
    best_ratio = -1.0

    for beta in np.linspace(0, 0.95, n_candidates):
        G = W @ (H + beta * delta) @ H_inv_sqrt
        sv = torch.linalg.svdvals(G)
        total_energy = (sv ** 2).sum().item()
        retained_energy = (sv[:rank] ** 2).sum().item()
        ratio = retained_energy / (total_energy + 1e-10)
        if ratio > best_ratio:
            best_ratio = ratio
            best_beta = beta

    return best_beta


def saes_svd_compress(W, H, delta, rank, beta=None):
    """SAES-SVD: compute optimal rank-r factors using cumulative error compensation.

    G = W(H + βΔ)H^{-1/2}
    T* = [G]_r (truncated SVD)
    A = U_r Σ_r^{1/2},  B = Σ_r^{1/2} V_r^T H^{-1/2}
    """
    d = H.shape[0]
    scale = max(H.trace().item() / d, 1e-8)
    reg = 1e-3 * scale * torch.eye(d, device=H.device, dtype=H.dtype)
    H_reg = H + reg
    H_inv_sqrt = stable_inv_sqrt(H_reg)

    if beta is None:
        beta = compute_aces_beta(W, H, delta, rank)

    G = W @ (H + beta * delta) @ H_inv_sqrt

    U, S, Vt = torch.linalg.svd(G, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]

    S_sqrt = S_r.sqrt()
    A = U_r * S_sqrt.unsqueeze(0)
    B = (Vt_r * S_sqrt.unsqueeze(1)) @ H_inv_sqrt

    approx = A @ B
    stored = rank * (W.shape[0] + W.shape[1])
    return approx, stored, beta


def get_layer_weight_names(model, layer_idx):
    """Get all 2D weight parameter names for a given layer."""
    names = []
    prefix = f"model.layers.{layer_idx}."
    for name, param in model.named_parameters():
        if name.startswith(prefix) and param.ndim == 2 and param.numel() >= 10000:
            names.append(name)
    return names


def parse_rank_from_allocation(entry, m, n):
    """Extract the target rank from an allocation entry."""
    family = entry["family"]
    compressed = entry["compressed_params"]

    if family in ("low_rank", "fisher_lr"):
        return compressed // (m + n)
    if family.startswith("tt_rank"):
        return int(family.replace("tt_rank", ""))
    # Default: estimate rank from compression ratio
    rank = compressed // (m + n)
    return max(1, rank)


def main(commit_fn=None):
    parser = argparse.ArgumentParser(description="Sequential SAES-SVD Compression")
    parser.add_argument("--allocation", type=str, required=True)
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--output", type=str, default="results/dsmollm_seq")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-eval", type=int, default=128)
    parser.add_argument("--n-calib", type=int, default=64)
    parser.add_argument("--beta", type=float, default=None,
                        help="Fixed β (default: auto-tune via ACES per layer)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  DSmolLM-seq — Sequential SAES-SVD Compression")
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    print(f"{'=' * 70}")

    with open(args.allocation) as f:
        alloc_data = json.load(f)

    allocation = alloc_data["optimal"]["allocations"]
    est_dloss = alloc_data["optimal"]["total_dloss"]
    target_ratio = alloc_data["target_layer_ratio"]

    # Index allocation by (layer, matrix_type)
    alloc_by_key = {}
    for entry in allocation:
        key = (entry["layer"], entry["matrix_type"])
        alloc_by_key[key] = entry

    n_layers = max(e["layer"] for e in allocation) + 1

    print(f"  Target: {target_ratio}x, {len(allocation)} matrices, {n_layers} layers")
    print(f"  Additive estimate: dloss={est_dloss:+.4f}")

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

    # Phase 1: Collect full-precision activations at every layer
    print(f"\n  Phase 1: Collecting full-precision activations...")
    t0 = time.time()
    fp_activations = collect_layer_activations(model, calib_data, device)
    print(f"    Collected {len(fp_activations)} layers in {time.time()-t0:.1f}s")

    # Phase 2: Sequential compression with SAES-SVD
    print(f"\n  Phase 2: Sequential SAES-SVD compression...")
    total_original = 0
    total_compressed = 0
    betas_used = []

    for layer_idx in range(n_layers):
        layer_key = f"model.layers.{layer_idx}"
        weight_names = get_layer_weight_names(model, layer_idx)

        if not weight_names:
            continue

        # Get compressed activations at this layer (through already-compressed prefix)
        compressed_activations = collect_layer_activations(model, calib_data, device)
        X_compressed = compressed_activations.get(layer_key)
        X_full = fp_activations.get(layer_key)

        if X_compressed is None or X_full is None:
            print(f"  L{layer_idx:02d}: no activations, skipping")
            continue

        # Compute covariance stats
        H, delta = compute_covariance_stats(X_compressed, X_full)
        H = H.to(device)
        delta = delta.to(device)

        del compressed_activations
        gc.collect()

        layer_applied = 0
        for param_name in weight_names:
            parts = param_name.split(".")
            mtype = None
            for p in parts:
                if p.endswith("_proj"):
                    mtype = p
                    break
            if mtype is None:
                continue

            entry = alloc_by_key.get((layer_idx, mtype))
            if entry is None or entry["family"] == "original":
                total_original += dict(model.named_parameters())[param_name].numel()
                total_compressed += dict(model.named_parameters())[param_name].numel()
                continue

            param = dict(model.named_parameters())[param_name]
            W = param.data.float()
            m, n = W.shape

            rank = parse_rank_from_allocation(entry, m, n)
            rank = min(rank, min(m, n) - 1)
            rank = max(rank, 1)

            try:
                # Use SAES-SVD: G = W(H + βΔ)H^{-1/2}, truncated SVD of G
                # H is (hidden, hidden) but W might be (out, in) where in=hidden
                # We need the INPUT covariance for this weight
                if n == H.shape[0]:
                    approx, stored, beta = saes_svd_compress(W, H, delta, rank, args.beta)
                elif m == H.shape[0]:
                    approx, stored, beta = saes_svd_compress(
                        W.T, H, delta, rank, args.beta)
                    approx = approx.T
                    stored = rank * (m + n)
                else:
                    # Dimension mismatch — fall back to standard SVD
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                    approx = (U[:, :rank] * S[:rank]) @ Vt[:rank, :]
                    stored = rank * (m + n)
                    beta = -1

                param.data.copy_(approx.to(param.dtype))
                betas_used.append(beta)

                ratio = entry["original_params"] / stored
                total_original += entry["original_params"]
                total_compressed += stored
                layer_applied += 1

            except Exception as e:
                print(f"    L{layer_idx:02d}.{mtype}: FAILED ({e}), keeping original")
                total_original += entry["original_params"]
                total_compressed += entry["original_params"]

        if layer_applied > 0:
            print(f"  L{layer_idx:02d}: {layer_applied} matrices compressed "
                  f"(avg beta={np.mean([b for b in betas_used[-layer_applied:] if b >= 0]):.3f})")

        del H, delta
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    effective_ratio = total_original / total_compressed if total_compressed > 0 else 1.0

    # Phase 3: Evaluate
    print(f"\n  Phase 3: Evaluating compressed model...")
    final_ppl, final_loss = eval_perplexity(model, eval_data, device)
    actual_dloss = final_loss - base_loss

    print(f"\n{'=' * 70}")
    print(f"  RESULTS — Sequential SAES-SVD")
    print(f"{'=' * 70}")
    print(f"  Baseline PPL:      {base_ppl:>12.2f}")
    print(f"  Compressed PPL:    {final_ppl:>12.2f}")
    print(f"  Actual dloss:      {actual_dloss:>+12.4f}")
    print(f"  Additive estimate: {est_dloss:>+12.4f}")
    print(f"  Compound factor:   {actual_dloss / est_dloss:>12.2f}x")
    print(f"  Target ratio:      {target_ratio:>12.1f}x")
    print(f"  Achieved ratio:    {effective_ratio:>12.2f}x")

    # Compare with naive (if available)
    naive_path = Path(args.output).parent / "dsmollm" / f"dsmollm_{target_ratio}x.json"
    if naive_path.exists():
        with open(naive_path) as f:
            naive = json.load(f)
        naive_dloss = naive["actual_dloss"]
        recovery = naive_dloss - actual_dloss
        print(f"\n  vs Naive all-at-once:")
        print(f"    Naive dloss:     {naive_dloss:>+12.4f}  (PPL {naive['compressed_ppl']:.0f})")
        print(f"    SAES-SVD dloss:  {actual_dloss:>+12.4f}  (PPL {final_ppl:.0f})")
        print(f"    Recovery:        {recovery:>+12.4f}  "
              f"({recovery / (naive_dloss - est_dloss) * 100:.0f}% of compound gap closed)")

    results = {
        "method": "saes_svd_sequential",
        "model": args.model,
        "target_ratio": target_ratio,
        "baseline_ppl": base_ppl,
        "baseline_loss": base_loss,
        "compressed_ppl": final_ppl,
        "compressed_loss": final_loss,
        "actual_dloss": actual_dloss,
        "estimated_dloss": est_dloss,
        "compound_factor": actual_dloss / est_dloss if est_dloss > 0 else 0,
        "original_params": total_original,
        "compressed_params": total_compressed,
        "achieved_ratio": effective_ratio,
        "betas": [float(b) for b in betas_used],
    }

    results_path = output_path / f"dsmollm_seq_{target_ratio}x.json"
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
