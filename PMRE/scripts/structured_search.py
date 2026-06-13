"""
Structured Family Search — Finding executable representations inside the Fisher ellipsoid.

For each weight matrix, we try to fit structured decompositions that:
1. Are directly executable (no decode-to-dense)
2. Preserve the function (stay within KL tolerance)
3. Are compact (fewer stored parameters than the original)

Candidate families:
- Tensor-Train (TT): W ≈ G1 × G2 × ... × Gd (product of small cores)
- Kronecker: W ≈ A ⊗ B (Kronecker product of smaller matrices)
- Butterfly: W ≈ B1 @ B2 @ ... @ Bd (product of sparse butterfly factors)
- Low-rank + sparse: W ≈ UV^T + S (low-rank approximation + sparse correction)
- Codebook: W ≈ lookup(indices, codebook) (vector quantization)

For each family, we measure:
- Compression ratio (stored params / original params)
- Reconstruction error (Frobenius norm)
- Functional error (KL divergence from teacher on calibration data)
- Executability (can we compute y = W_approx @ x without materializing W?)

The Fisher enters as a TOLERANCE METRIC: we measure δ^T F δ for the approximation
error δ = W_approx - W, which tells us the functional cost of the approximation.
"""

import argparse
import gc
import json
import math
import os
import signal
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


@dataclass
class FitResult:
    """Result of fitting a structured family to a weight matrix."""
    family: str
    matrix_name: str
    original_params: int
    compressed_params: int
    compression_ratio: float
    frobenius_error: float       # ||W - W_approx||_F / ||W||_F
    fisher_weighted_error: float  # δ^T F δ (functional cost)
    ppl_after: float = 0.0
    delta_loss: float = 0.0


def get_data(tokenizer, split="train", n_samples=128, seq_len=512, seed=42, skip=0):
    """Load evaluation data from C4 (web text, in-distribution for web-trained models).
    Takes document-start chunks so the model has proper context.
    skip: number of valid samples to skip (for separating calib from eval)."""
    try:
        dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    except Exception:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                               split="validation" if split == "test" else split)
        dataset = iter([{"text": t} for t in dataset["text"] if len(t.strip()) > 200])

    samples = []
    skipped = 0
    for doc in dataset:
        if len(samples) >= n_samples:
            break
        text = doc["text"].strip()
        if len(text) < 100:
            continue
        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=seq_len)["input_ids"][0]
        if len(tokens) >= seq_len:
            if skipped < skip:
                skipped += 1
                continue
            samples.append(tokens[:seq_len])

    if len(samples) < n_samples:
        print(f"    WARNING: only got {len(samples)}/{n_samples} full-length samples")

    return torch.stack(samples[:n_samples])


@torch.no_grad()
def eval_perplexity(model, data, device, batch_size=8):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size].to(device)
        outputs = model(input_ids=batch, labels=batch)
        # HF causal LM models shift labels internally; loss is over (seq_len - 1) tokens
        n_tokens = (batch.shape[1] - 1) * batch.shape[0]
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
    return math.exp(total_loss / total_tokens), total_loss / total_tokens


def compute_fisher_for_matrix(model, data, device, target_name, n_samples=64, batch_size=4):
    """Compute diagonal Fisher for a specific parameter matrix."""
    param = dict(model.named_parameters())[target_name]
    fisher = torch.zeros_like(param, device="cpu")
    model.train()
    n = 0
    for i in range(0, min(n_samples, len(data)), batch_size):
        batch = data[i:i + batch_size].to(device)
        model.zero_grad()
        outputs = model(input_ids=batch, labels=batch)
        outputs.loss.backward()
        if param.grad is not None:
            fisher += (param.grad.detach().cpu() ** 2)
        n += 1
    model.eval()
    return fisher / n


# ═══════════════════════════════════════════════════════════════════════
# STRUCTURED FAMILIES
# ═══════════════════════════════════════════════════════════════════════

def fit_low_rank(W, rank):
    """Fit W ≈ U @ V^T via truncated SVD. Directly executable as two matmuls."""
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    U_k = U[:, :rank] * S[:rank].unsqueeze(0)
    V_k = Vt[:rank, :]
    approx = U_k @ V_k
    stored = rank * (W.shape[0] + W.shape[1])
    return approx, stored, (U_k, V_k)


def fit_low_rank_fisher(W, rank, fisher_diag, n_iter=10):
    """Fisher-weighted low-rank via alternating least squares (vectorized).
    Minimizes Σ F_ij × (W_ij - (UV^T)_ij)² instead of Frobenius error.
    This finds the rank-k subspace that best preserves the FUNCTION, not the norm."""
    m, n = W.shape

    # Fisher is extremely concentrated (top 0.1% = 97% of mass).
    # Raw sqrt(Fisher) has ~10^6 dynamic range which makes ALS collapse to zero
    # on near-zero-weight rows. Normalize to unit mean and floor at 1% so every
    # row gets some reconstruction while high-Fisher rows dominate the objective.
    F_sqrt = fisher_diag.sqrt()
    F_mean = F_sqrt.mean()
    if F_mean > 0:
        F_sqrt = F_sqrt / F_mean
    F_sqrt = F_sqrt.clamp(min=0.01)

    # Initialize from standard SVD
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    U_k = (U[:, :rank] * S[:rank].unsqueeze(0)).clone()
    V_k = Vt[:rank, :].clone()

    eye_r = torch.eye(rank, device=W.device, dtype=W.dtype)

    for it in range(n_iter):
        # Decay regularization so early iterations are stable, late ones are precise
        reg = 1e-4 / (1 + it)

        # Fix V, solve for all rows of U simultaneously
        Vw = V_k.unsqueeze(0) * F_sqrt.unsqueeze(1)  # (m, rank, n)
        A = torch.bmm(Vw, Vw.transpose(1, 2)) + reg * eye_r.unsqueeze(0)
        Ww = W * F_sqrt  # (m, n)
        b = torch.bmm(Ww.unsqueeze(1), Vw.transpose(1, 2)).squeeze(1)  # (m, rank)
        U_k = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)

        # Fix U, solve for all columns of V simultaneously
        F_sqrt_T = F_sqrt.T  # (n, m)
        Uw = U_k.unsqueeze(0) * F_sqrt_T.unsqueeze(2)  # (n, m, rank)
        A = torch.bmm(Uw.transpose(1, 2), Uw) + reg * eye_r.unsqueeze(0)
        Ww_T = (W * F_sqrt).T  # (n, m)
        b = torch.bmm(Uw.transpose(1, 2), Ww_T.unsqueeze(-1)).squeeze(-1)  # (n, rank)
        V_k = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1).T  # (rank, n)

    approx = U_k @ V_k
    stored = rank * (m + n)
    return approx, stored, (U_k, V_k)


def fit_codebook_fisher(W, n_codebooks, codebook_size, fisher_diag, n_iter=20):
    """Fisher-weighted product quantization.
    Weights k-means clustering by Fisher importance so codebook entries
    prioritize representing functionally important regions."""
    m, n = W.shape
    subvec_dim = n // n_codebooks
    if n % n_codebooks != 0:
        pad_n = (n_codebooks - n % n_codebooks)
        W_padded = F.pad(W, (0, pad_n))
        F_padded = F.pad(fisher_diag, (0, pad_n))
        n_padded = n + pad_n
        subvec_dim = n_padded // n_codebooks
    else:
        W_padded = W
        F_padded = fisher_diag
        n_padded = n

    codebooks = []
    indices = []
    approx_parts = []

    for cb_idx in range(n_codebooks):
        start = cb_idx * subvec_dim
        end = start + subvec_dim
        subvecs = W_padded[:, start:end]  # (m, subvec_dim)
        f_weights = F_padded[:, start:end]  # (m, subvec_dim)
        f_row_importance = f_weights.sum(dim=1).clamp(min=1e-10)  # (m,) per-row weight

        # Fisher-weighted k-means
        rng = torch.Generator().manual_seed(42 + cb_idx)
        perm = torch.randperm(m, generator=rng)[:codebook_size]
        centroids = subvecs[perm].clone()

        for _ in range(n_iter):
            diffs = subvecs.unsqueeze(1) - centroids.unsqueeze(0)  # (m, K, subvec_dim)
            weighted_dists = (diffs ** 2 * f_weights.unsqueeze(1)).sum(dim=2)  # (m, K)
            assigns = weighted_dists.argmin(dim=1)

            for k in range(codebook_size):
                mask = assigns == k
                if mask.any():
                    weights = f_row_importance[mask]
                    w_sum = weights.sum()
                    if w_sum > 0:
                        centroids[k] = (subvecs[mask] * weights.unsqueeze(1)).sum(0) / w_sum

        # Final assignment
        diffs = subvecs.unsqueeze(1) - centroids.unsqueeze(0)
        weighted_dists = (diffs ** 2 * f_weights.unsqueeze(1)).sum(dim=2)
        assigns = weighted_dists.argmin(dim=1)

        codebooks.append(centroids)
        indices.append(assigns)
        approx_parts.append(centroids[assigns])

    approx = torch.cat(approx_parts, dim=1)[:, :n]
    codebook_storage = n_codebooks * codebook_size * subvec_dim
    index_storage = m * n_codebooks
    stored = codebook_storage + index_storage // 2

    return approx, stored, (codebooks, indices)


def fit_low_rank_plus_sparse(W, rank, sparsity_top_k):
    """W ≈ UV^T + S where S is sparse (top-k elements of the residual)."""
    approx_lr, stored_lr, (U_k, V_k) = fit_low_rank(W, rank)
    residual = W - approx_lr

    flat_res = residual.flatten()
    _, top_indices = torch.topk(flat_res.abs(), sparsity_top_k)
    sparse_values = flat_res[top_indices]

    sparse_correction = torch.zeros_like(flat_res)
    sparse_correction[top_indices] = sparse_values
    approx = approx_lr + sparse_correction.reshape(W.shape)

    stored = stored_lr + sparsity_top_k * 2
    return approx, stored, (U_k, V_k, top_indices, sparse_values)


def fit_kronecker(W, block_m, block_n):
    """
    Fit W ≈ A ⊗ B where A is (m/block_m, n/block_n) and B is (block_m, block_n).

    Uses the nearest Kronecker product algorithm (Van Loan & Pitsianis).
    Directly executable: (A ⊗ B)x can be computed as vec(B @ X @ A^T) with reshaping.
    """
    m, n = W.shape
    if m % block_m != 0 or n % block_n != 0:
        return None, float('inf'), None

    am, an = m // block_m, n // block_n

    blocks = W.reshape(am, block_m, an, block_n).permute(0, 2, 1, 3).reshape(am * an, block_m * block_n)
    U, S, Vt = torch.linalg.svd(blocks, full_matrices=False)

    A = (U[:, 0] * math.sqrt(S[0].item())).reshape(am, an)
    B = (Vt[0, :] * math.sqrt(S[0].item())).reshape(block_m, block_n)

    approx = torch.kron(A, B)
    stored = am * an + block_m * block_n
    return approx, stored, (A, B)


def fit_kronecker_sum(W, block_m, block_n, n_terms=4):
    """
    Fit W ≈ Σ_i A_i ⊗ B_i (sum of Kronecker products).
    More expressive than single Kronecker, still structured.
    """
    m, n = W.shape
    if m % block_m != 0 or n % block_n != 0:
        return None, float('inf'), None

    am, an = m // block_m, n // block_n
    blocks = W.reshape(am, block_m, an, block_n).permute(0, 2, 1, 3).reshape(am * an, block_m * block_n)

    U, S, Vt = torch.linalg.svd(blocks, full_matrices=False)
    k = min(n_terms, len(S))

    approx = torch.zeros_like(W)
    factors = []
    for i in range(k):
        A_i = (U[:, i] * math.sqrt(S[i].item())).reshape(am, an)
        B_i = (Vt[i, :] * math.sqrt(S[i].item())).reshape(block_m, block_n)
        approx += torch.kron(A_i, B_i)
        factors.append((A_i, B_i))

    stored = k * (am * an + block_m * block_n)
    return approx, stored, factors


def fit_block_diagonal(W, block_size):
    """
    Approximate W as block-diagonal.
    Directly executable: separate matmuls per block.
    """
    m, n = W.shape
    if m != n or m % block_size != 0:
        return None, float('inf'), None

    n_blocks = m // block_size
    approx = torch.zeros_like(W)
    blocks = []

    for i in range(n_blocks):
        block = W[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size]
        approx[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size] = block
        blocks.append(block)

    stored = n_blocks * block_size * block_size
    return approx, stored, blocks


def fit_tensor_train(W, tt_ranks):
    """
    Reshape W into a higher-order tensor, then fit a tensor-train decomposition.
    TT format: W[i1,i2,...,id] = G1[i1] @ G2[i2] @ ... @ Gd[id]

    Directly executable via sequential contractions (no materialization needed
    for structured matmul, though naive implementation does materialize).
    """
    m, n = W.shape

    m_factors = factorize_balanced(m)
    n_factors = factorize_balanced(n)

    if m_factors is None or n_factors is None:
        return None, float('inf'), None

    shape = m_factors + n_factors
    d = len(shape)

    try:
        tensor = W.reshape(shape)
    except RuntimeError:
        return None, float('inf'), None

    cores = []
    remaining = tensor.reshape(shape[0], -1)
    ranks = [1] + list(tt_ranks[:d-1]) + [1]

    while len(ranks) < d + 1:
        ranks.insert(-1, min(ranks[-2] * shape[len(ranks)-2], 64))

    stored = 0
    for k in range(d - 1):
        r_prev = remaining.shape[0] // shape[k] if k > 0 else 1
        curr_shape = ranks[k] * shape[k]

        if remaining.shape[0] < curr_shape:
            remaining = remaining.reshape(-1, remaining.shape[-1] if remaining.ndim > 1 else 1)

        try:
            mat = remaining.reshape(ranks[k] * shape[k], -1)
        except RuntimeError:
            return None, float('inf'), None

        rank = min(ranks[k+1], mat.shape[0], mat.shape[1])
        U, S, Vt = torch.linalg.svd(mat, full_matrices=False)
        U_k = U[:, :rank]
        S_k = S[:rank]
        Vt_k = Vt[:rank, :]

        core = U_k.reshape(ranks[k], shape[k], rank)
        cores.append(core)
        stored += core.numel()

        remaining = torch.diag(S_k) @ Vt_k
        ranks[k+1] = rank

    last_core = remaining.reshape(ranks[-2], shape[-1], 1)
    cores.append(last_core)
    stored += last_core.numel()

    result = cores[0].squeeze(0)
    for core in cores[1:]:
        result = torch.einsum('...i,ijk->...jk', result, core)
    approx = result.squeeze(-1).reshape(m, n)

    return approx, stored, cores


def factorize_balanced(n, max_factors=4):
    """Find a balanced factorization of n into 2-4 factors."""
    if n <= 1:
        return None

    for i in range(int(math.sqrt(n)), 1, -1):
        if n % i == 0:
            j = n // i
            if max(i, j) / min(i, j) < 8:
                return (i, j)

    cbrt = int(n ** (1/3))
    for i in range(cbrt + 2, 1, -1):
        if n % i == 0:
            remainder = n // i
            sub = factorize_balanced(remainder, 2)
            if sub is not None:
                return (i,) + sub

    return None


def fit_codebook(W, n_codebooks=4, codebook_size=256):
    """
    Product quantization: split weight rows into subvectors, quantize each
    to nearest codebook entry.

    Storage: codebooks + indices (very compact).
    Execution: gather from codebooks (fast on GPU with fused kernels).
    """
    m, n = W.shape
    subvec_dim = n // n_codebooks
    if n % n_codebooks != 0:
        pad_n = (n_codebooks - n % n_codebooks)
        W_padded = F.pad(W, (0, pad_n))
        n_padded = n + pad_n
        subvec_dim = n_padded // n_codebooks
    else:
        W_padded = W
        n_padded = n

    codebooks = []
    indices = []
    approx_parts = []

    for cb_idx in range(n_codebooks):
        start = cb_idx * subvec_dim
        end = start + subvec_dim
        subvecs = W_padded[:, start:end]

        rng = torch.Generator().manual_seed(42 + cb_idx)
        perm = torch.randperm(m, generator=rng)[:codebook_size]
        centroids = subvecs[perm].clone()

        for _ in range(20):
            dists = torch.cdist(subvecs, centroids)
            assigns = dists.argmin(dim=1)
            for k in range(codebook_size):
                mask = assigns == k
                if mask.any():
                    centroids[k] = subvecs[mask].mean(dim=0)

        dists = torch.cdist(subvecs, centroids)
        assigns = dists.argmin(dim=1)

        codebooks.append(centroids)
        indices.append(assigns)
        approx_parts.append(centroids[assigns])

    approx = torch.cat(approx_parts, dim=1)[:, :n]

    codebook_storage = n_codebooks * codebook_size * subvec_dim
    index_storage = m * n_codebooks
    stored = codebook_storage + index_storage // 2

    return approx, stored, (codebooks, indices)


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION & ROBUST CONFIG RUNNER
# ═══════════════════════════════════════════════════════════════════════

def evaluate_fit(model, target_name, approx_W, eval_data, device, base_loss):
    """Substitute approximation into model and evaluate."""
    param = dict(model.named_parameters())[target_name]
    original = param.data.clone()

    param.data.copy_(approx_W.to(param.dtype).to(param.device))
    ppl, loss = eval_perplexity(model, eval_data, device)

    param.data.copy_(original)
    return ppl, loss - base_loss


def safe_run_config(label, family_name, fit_fn, W, W_norm, fisher_diag,
                    model, target_name, eval_data, device, base_loss, original_params):
    """Run one compression config with full error handling. Returns FitResult or None."""
    try:
        result = fit_fn()
        if result is None or result[0] is None:
            return None
        approx, stored, factors = result
        if stored >= original_params:
            return None
        ratio = original_params / stored
        frob_err = (W - approx).norm().item() / W_norm
        fisher_err = ((W - approx).flatten() ** 2 * fisher_diag.flatten()).sum().item()
        ppl, dloss = evaluate_fit(model, target_name, approx, eval_data, device, base_loss)
        r = FitResult(family_name, target_name, original_params, stored,
                      ratio, frob_err, fisher_err, ppl, dloss)
        print(f"      {label}: {ratio:>5.1f}x | frob={frob_err:.4f} | "
              f"dloss={dloss:>+.4f} | PPL={ppl:.1f}")
        return r
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"      {label}: OOM (skipped)")
        return None
    except Exception as e:
        print(f"      {label}: FAILED ({type(e).__name__}: {e})")
        return None


# ═══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════

def run_family_search(model, target_name, W, fisher_diag, eval_data, device, base_loss):
    """Try all structured families on one weight matrix.
    Each family and each config within it is wrapped in error handling —
    one failure never kills the whole search."""
    m, n = W.shape
    original_params = m * n
    W_norm = W.norm().item()
    results = []

    common = dict(W=W, W_norm=W_norm, fisher_diag=fisher_diag, model=model,
                  target_name=target_name, eval_data=eval_data, device=device,
                  base_loss=base_loss, original_params=original_params)

    print(f"\n    Original: {m}x{n} = {original_params:,} params")

    # ── Low-rank at various ranks ──
    try:
        print(f"    Testing low-rank...")
        for rank in [4, 8, 16, 32, 64, 128, 256]:
            if rank >= min(m, n):
                continue
            r = safe_run_config(f"rank={rank:>4d}", "low_rank",
                                lambda rank=rank: fit_low_rank(W, rank), **common)
            if r:
                results.append(r)
    except Exception as e:
        print(f"    LOW-RANK FAMILY FAILED: {e}")

    # ── Low-rank + sparse ──
    try:
        print(f"    Testing low-rank + sparse correction...")
        for rank, top_k_frac in [(16, 0.01), (32, 0.01), (32, 0.05), (64, 0.02)]:
            if rank >= min(m, n):
                continue
            top_k = int(original_params * top_k_frac)
            r = safe_run_config(
                f"rank={rank}+top{top_k_frac:.0%}", f"lr{rank}+sparse{top_k_frac:.0%}",
                lambda rank=rank, top_k=top_k: fit_low_rank_plus_sparse(W, rank, top_k),
                **common)
            if r:
                results.append(r)
    except Exception as e:
        print(f"    LR+SPARSE FAMILY FAILED: {e}")

    # ── Kronecker (single term) ──
    try:
        print(f"    Testing Kronecker product...")
        for block_m, block_n in [(16, 16), (24, 24), (32, 32), (8, 8)]:
            if m % block_m != 0 or n % block_n != 0:
                continue
            r = safe_run_config(
                f"block={block_m}x{block_n}", f"kron_{block_m}x{block_n}",
                lambda bm=block_m, bn=block_n: fit_kronecker(W, bm, bn), **common)
            if r:
                results.append(r)
    except Exception as e:
        print(f"    KRONECKER FAMILY FAILED: {e}")

    # ── Kronecker sum (multiple terms) ──
    try:
        print(f"    Testing Kronecker sum...")
        for block_m, block_n, n_terms in [(24, 24, 4), (24, 24, 8), (24, 24, 16), (32, 32, 8)]:
            if m % block_m != 0 or n % block_n != 0:
                continue
            r = safe_run_config(
                f"block={block_m}x{block_n}x{n_terms}", f"kronsum_{block_m}x{block_n}x{n_terms}",
                lambda bm=block_m, bn=block_n, nt=n_terms: fit_kronecker_sum(W, bm, bn, nt),
                **common)
            if r:
                results.append(r)
    except Exception as e:
        print(f"    KRONECKER SUM FAMILY FAILED: {e}")

    # ── Block diagonal (for square matrices) ──
    if m == n:
        try:
            print(f"    Testing block diagonal...")
            for block_size in [32, 64, 128]:
                if m % block_size != 0:
                    continue
                r = safe_run_config(
                    f"block={block_size}", f"block_diag_{block_size}",
                    lambda bs=block_size: fit_block_diagonal(W, bs), **common)
                if r:
                    results.append(r)
        except Exception as e:
            print(f"    BLOCK DIAGONAL FAMILY FAILED: {e}")

    # ── Product quantization / codebook ──
    try:
        print(f"    Testing codebook (product quantization)...")
        for n_cb, cb_size in [(4, 64), (4, 128), (8, 128), (4, 256), (8, 256), (16, 256)]:
            subvec_dim = n // n_cb + (1 if n % n_cb != 0 else 0)
            est_storage = n_cb * cb_size * subvec_dim + m * n_cb // 2
            if est_storage >= original_params:
                continue
            r = safe_run_config(
                f"{n_cb}x{cb_size}", f"codebook_{n_cb}x{cb_size}",
                lambda nc=n_cb, cs=cb_size: fit_codebook(W, n_codebooks=nc, codebook_size=cs),
                **common)
            if r:
                results.append(r)
    except Exception as e:
        print(f"    CODEBOOK FAMILY FAILED: {e}")

    # ── Tensor-train ──
    try:
        print(f"    Testing tensor-train...")
        for max_rank in [4, 8, 16, 32]:
            tt_ranks = [max_rank] * 6
            r = safe_run_config(
                f"tt_rank={max_rank}", f"tt_rank{max_rank}",
                lambda tr=list(tt_ranks): fit_tensor_train(W, tr), **common)
            if r:
                results.append(r)
    except Exception as e:
        print(f"    TENSOR-TRAIN FAMILY FAILED: {e}")

    # ── Fisher-weighted low-rank ──
    try:
        print(f"    Testing Fisher-weighted low-rank...")
        for rank in [4, 8, 16, 32, 64]:
            if rank >= min(m, n):
                continue
            r = safe_run_config(
                f"rank={rank:>4d}", "fisher_lr",
                lambda rank=rank: fit_low_rank_fisher(W, rank, fisher_diag), **common)
            if r:
                results.append(r)
    except Exception as e:
        print(f"    FISHER LR FAMILY FAILED: {e}")

    # ── Fisher-weighted codebook ──
    try:
        print(f"    Testing Fisher-weighted codebook...")
        for n_cb, cb_size in [(4, 64), (4, 128), (8, 128)]:
            subvec_dim = n // n_cb + (1 if n % n_cb != 0 else 0)
            est_storage = n_cb * cb_size * subvec_dim + m * n_cb // 2
            if est_storage >= original_params:
                continue
            r = safe_run_config(
                f"{n_cb}x{cb_size}", f"fisher_cb_{n_cb}x{cb_size}",
                lambda nc=n_cb, cs=cb_size: fit_codebook_fisher(W, nc, cs, fisher_diag),
                **common)
            if r:
                results.append(r)
    except Exception as e:
        print(f"    FISHER CODEBOOK FAMILY FAILED: {e}")

    return results


def _result_to_dict(r):
    """Convert FitResult to serializable dict."""
    return {
        "family": r.family,
        "matrix": r.matrix_name,
        "original_params": r.original_params,
        "compressed_params": r.compressed_params,
        "compression_ratio": r.compression_ratio,
        "frobenius_error": r.frobenius_error,
        "fisher_weighted_error": r.fisher_weighted_error,
        "ppl": r.ppl_after,
        "delta_loss": r.delta_loss,
    }


def _print_summary(all_results_dicts):
    """Print summary tables from result dicts."""
    if not all_results_dicts:
        print("  No results to summarize.")
        return

    print(f"\n  Best results per family (sorted by delta_loss):")
    print(f"  {'Family':>30s} {'Matrix':>25s} {'Ratio':>6s} {'Frob':>6s} {'dloss':>8s}")
    print(f"  {'-' * 80}")

    by_family = defaultdict(list)
    for r in all_results_dicts:
        by_family[r["family"]].append(r)

    summary = []
    for family, family_results in by_family.items():
        best = min(family_results,
                   key=lambda r: r["delta_loss"] if r["compression_ratio"] > 1.5 else float('inf'))
        if best["compression_ratio"] > 1.5:
            summary.append(best)

    for r in sorted(summary, key=lambda x: x["delta_loss"]):
        mat_short = r["matrix"].split(".")[-2] + "." + r["matrix"].split(".")[-1]
        print(f"  {r['family']:>30s} {mat_short:>25s} {r['compression_ratio']:>5.1f}x "
              f"{r['frobenius_error']:>6.4f} {r['delta_loss']:>+8.4f}")

    print(f"\n  PARETO FRONTIER (best delta_loss at each compression tier):")
    print(f"  {'Compression':>12s} {'Family':>30s} {'dloss':>8s} {'PPL':>10s}")
    print(f"  {'-' * 65}")

    for tier_min, tier_max in [(2, 4), (4, 8), (8, 16), (16, 50), (50, 200)]:
        tier_results = [r for r in all_results_dicts
                        if tier_min <= r["compression_ratio"] < tier_max]
        if tier_results:
            best = min(tier_results, key=lambda r: r["delta_loss"])
            label = f"{tier_min}-{tier_max}x"
            print(f"  {label:>12s} {best['family']:>30s} "
                  f"{best['delta_loss']:>+8.4f} {best['ppl']:>10.1f}")


def main(commit_fn=None):
    """Main entry point.

    commit_fn: optional callback invoked after each matrix's results are saved.
               Modal wrapper passes results_vol.commit here so incremental
               results survive container kills / client disconnects.
    """
    # Unbuffered stdout so Modal log streaming works
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(write_through=True)
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Structured Family Search")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--output", default="results/structured_search")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--n-eval", type=int, default=64)
    parser.add_argument("--layers", type=int, nargs="*", default=[0, 14, 29])
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    per_matrix_dir = output_path / "per_matrix"
    per_matrix_dir.mkdir(parents=True, exist_ok=True)

    ts = lambda: datetime.now(timezone.utc).strftime('%H:%M:%S UTC')

    print(f"\n{'=' * 70}")
    print(f"  STRUCTURED FAMILY SEARCH")
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    print(f"  Target layers: {args.layers}")
    print(f"  Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    # ── Checkpoint/resume: load existing per-matrix results ──
    completed = {}
    for f in sorted(per_matrix_dir.glob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
                completed[f.stem] = data
                print(f"  RESUMED: {f.stem} ({len(data)} configs)")
        except Exception as e:
            print(f"  WARNING: corrupt checkpoint {f.name}, deleting: {e}")
            try:
                f.unlink()
            except Exception:
                pass

    # ── Load model and data ──
    print(f"\n  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}")

    print(f"\n  Loading data...")
    calib_data = get_data(tokenizer, "train", n_samples=args.n_calib, skip=0)
    eval_data = get_data(tokenizer, "test", n_samples=args.n_eval, skip=args.n_calib)

    print(f"\n  Measuring baseline...")
    base_ppl, base_loss = eval_perplexity(model, eval_data, device)
    print(f"    Baseline: PPL={base_ppl:.2f}, loss={base_loss:.4f}")
    if base_ppl > 500:
        print(f"    WARNING: Baseline PPL={base_ppl:.0f} is unusually high!")
        print(f"    This may indicate OOD evaluation data or a loading issue.")

    # ── Build target list ──
    targets = []
    for name, param in model.named_parameters():
        if param.ndim < 2 or param.numel() < 50000:
            continue
        for layer_idx in args.layers:
            if f"layers.{layer_idx}." in name:
                targets.append(name)
                break

    n_completed = sum(1 for t in targets if t.replace(".", "__") in completed)
    print(f"\n  Target matrices: {len(targets)} ({n_completed} already done, "
          f"{len(targets) - n_completed} remaining)")

    # ── Signal handler for graceful shutdown ──
    interrupted = [False]

    def handle_signal(signum, frame):
        print(f"\n  SIGNAL {signum} received at {ts()}")
        print(f"  Will finish current matrix, save, then exit.")
        interrupted[0] = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # ── Main loop ──
    all_results_dicts = []
    for data in completed.values():
        all_results_dicts.extend(data)

    for i, target_name in enumerate(targets):
        if interrupted[0]:
            print(f"\n  Interrupted -- skipping remaining matrices")
            break

        safe_name = target_name.replace(".", "__")
        if safe_name in completed:
            continue

        param = dict(model.named_parameters())[target_name]
        W = param.data.float().cpu()

        print(f"\n{'=' * 70}")
        print(f"  [{i+1}/{len(targets)}] {target_name} ({list(param.shape)})  [{ts()}]")
        print(f"{'=' * 70}")

        # Fisher computation with fallback
        print(f"    Computing Fisher...")
        try:
            fisher_diag = compute_fisher_for_matrix(model, calib_data, device, target_name)
        except Exception as e:
            print(f"    Fisher FAILED ({e}), using uniform weights")
            fisher_diag = torch.ones_like(W)

        # Run all families
        matrix_results = run_family_search(
            model, target_name, W, fisher_diag, eval_data, device, base_loss)

        # Incremental save
        matrix_dicts = [_result_to_dict(r) for r in matrix_results]
        try:
            with open(per_matrix_dir / f"{safe_name}.json", "w") as f:
                json.dump(matrix_dicts, f, indent=2)
            print(f"\n    Saved {len(matrix_dicts)} results -> {safe_name}.json")
        except Exception as e:
            print(f"\n    WARNING: save failed for {safe_name}: {e}")

        all_results_dicts.extend(matrix_dicts)

        # Commit volume (Modal)
        if commit_fn:
            try:
                commit_fn()
                print(f"    Volume committed at {ts()}")
            except Exception as e:
                print(f"    Volume commit failed (non-fatal): {e}")

        # Memory cleanup
        del W, fisher_diag, matrix_results, matrix_dicts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Save combined results ──
    try:
        with open(output_path / "structured_search.json", "w") as f:
            json.dump(all_results_dicts, f, indent=2)
        print(f"\n  Combined results: {output_path / 'structured_search.json'} "
              f"({len(all_results_dicts)} total configs)")
    except Exception as e:
        print(f"\n  WARNING: combined save failed: {e}")

    if commit_fn:
        try:
            commit_fn()
        except Exception:
            pass

    # ── Summary ──
    print(f"\n\n{'=' * 70}")
    print(f"  SUMMARY  [{ts()}]")
    print(f"  Total configs: {len(all_results_dicts)}")
    print(f"{'=' * 70}")

    _print_summary(all_results_dicts)

    if interrupted[0]:
        print(f"\n  NOTE: Run was interrupted. Re-run to complete remaining matrices "
              f"(completed work is checkpointed).")


if __name__ == "__main__":
    main()
