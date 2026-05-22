from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader, dequantize
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


DEFAULT_MODEL_DIR = os.environ.get("PMRA_MODEL_DIR", "models/qwen3-1.7b-base")
DEFAULT_HF = os.environ.get("PMRA_HF", str(Path(DEFAULT_MODEL_DIR) / "model.safetensors"))
DEFAULT_IQ4 = os.environ.get("PMRA_IQ4", "models/gguf/Qwen_Qwen3-1.7B-IQ4_XS.gguf")

BLOCK = 256
BITS = 3
ALPHAS = (0.70, 0.85, 1.0, 1.20)
NBINS = 4


PROMPTS = [
    "Explain why a transformer model uses attention heads, in plain language.",
    "Write a short Python function that computes the median of a list.",
    "The old observatory stood above the harbor, its brass telescope aimed at the winter sky.",
    "Compare photosynthesis and cellular respiration in two concise paragraphs.",
    "def merge_intervals(intervals): intervals.sort(); result = []",
    "A packet switched network routes data through independent hops rather than a fixed circuit.",
    "Summarize the economic causes of the 1973 oil crisis.",
    "In a small workshop, the watchmaker sorted gears by diameter and tooth count.",
    "SELECT customer_id, SUM(total) FROM invoices GROUP BY customer_id HAVING SUM(total) > 1000;",
    "Why does a refrigerator require work to move heat from cold air to warm air?",
    "The river widened after the spring thaw and carried branches into the lower fields.",
    "Describe how gradient descent updates parameters in a neural network.",
    "A cryptographic hash should be deterministic, collision resistant, and hard to invert.",
    "Explain the difference between mass and weight using an everyday example.",
    "The pianist practiced the same four measures until the rhythm became automatic.",
    "for token in sequence: cache.append(model.step(token)); logits = project(cache[-1])",
    "How does a barometer measure changes in atmospheric pressure?",
    "The library catalog used careful cross references for authors who wrote under several names.",
    "Describe the role of mitochondria in a cell.",
    "A finite state machine has states, inputs, transitions, and accepting conditions.",
    "The tea kettle clicked softly as steam gathered under the lid.",
    "Explain Bayes' theorem without using equations.",
    "Rust ownership rules prevent data races by enforcing borrowing constraints at compile time.",
    "The mountain pass remained closed until crews cleared the avalanche debris.",
    "What makes a proof by contradiction valid?",
    "A binary search tree stores smaller keys on the left and larger keys on the right.",
    "The ceramic glaze changed color where the kiln had run hottest.",
    "Why does salt lower the freezing point of water?",
    "A database index trades additional storage for faster lookup.",
    "The cartographer marked the shoals in red ink on the harbor map.",
    "Explain what a Fourier transform reveals about a signal.",
    "The server retried the request with exponential backoff after receiving a timeout.",
]


FAMILY_TO_MODULE = {
    "attn_q": ("self_attn.q_proj", "self_attn.q_proj.weight", "attn_q.weight"),
    "attn_k": ("self_attn.k_proj", "self_attn.k_proj.weight", "attn_k.weight"),
    "attn_v": ("self_attn.v_proj", "self_attn.v_proj.weight", "attn_v.weight"),
    "attn_o": ("self_attn.o_proj", "self_attn.o_proj.weight", "attn_output.weight"),
    "ffn_gate": ("mlp.gate_proj", "mlp.gate_proj.weight", "ffn_gate.weight"),
    "ffn_up": ("mlp.up_proj", "mlp.up_proj.weight", "ffn_up.weight"),
    "ffn_down": ("mlp.down_proj", "mlp.down_proj.weight", "ffn_down.weight"),
}


@dataclass(frozen=True)
class TensorJob:
    layer: int
    family: str
    hf_name: str
    gguf_name: str

    @property
    def key(self) -> str:
        return f"L{self.layer}:{self.family}"


def parse_layers(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_families(text: str) -> list[str]:
    families = [part.strip() for part in text.split(",") if part.strip()]
    bad = sorted(set(families) - set(FAMILY_TO_MODULE))
    if bad:
        raise ValueError(f"unknown families: {bad}")
    return families


def make_jobs(layers: Iterable[int], families: Iterable[str]) -> list[TensorJob]:
    jobs = []
    for layer in layers:
        for family in families:
            _, hf_suffix, gguf_suffix = FAMILY_TO_MODULE[family]
            jobs.append(
                TensorJob(
                    layer=layer,
                    family=family,
                    hf_name=f"model.layers.{layer}.{hf_suffix}",
                    gguf_name=f"blk.{layer}.{gguf_suffix}",
                )
            )
    return jobs


def get_module(root: torch.nn.Module, dotted: str) -> torch.nn.Module:
    obj: torch.nn.Module = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def build_prompts(tokenizer, n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    prompts = list(PROMPTS)
    rng.shuffle(prompts)
    while len(prompts) < n:
        ids = rng.integers(0, tokenizer.vocab_size, size=32).tolist()
        prompts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return prompts[:n]


def collect_activations(
    model,
    tokenizer,
    jobs: list[TensorJob],
    prompts: list[str],
    max_length: int,
    max_tokens_per_job: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    storage: dict[str, list[torch.Tensor]] = {job.key: [] for job in jobs}
    counts: dict[str, int] = {job.key: 0 for job in jobs}
    captured: dict[str, torch.Tensor] = {}

    handles = []
    for job in jobs:
        module_path = FAMILY_TO_MODULE[job.family][0]
        module = get_module(model.model.layers[job.layer], module_path)

        def make_hook(key: str):
            def hook(_mod, inputs):
                if counts[key] >= max_tokens_per_job:
                    return
                x = inputs[0].detach().to(torch.float32).cpu()
                captured[key] = x

            return hook

        handles.append(module.register_forward_pre_hook(make_hook(job.key)))

    try:
        with torch.inference_mode():
            for i, prompt in enumerate(prompts):
                captured.clear()
                encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
                ids = encoded.input_ids.to(device)
                if ids.shape[-1] < 2:
                    continue
                try:
                    model(input_ids=ids, use_cache=False)
                except Exception as exc:
                    print(f"  prompt {i} failed during activation capture: {exc}", flush=True)
                    continue

                for key, x in captured.items():
                    if counts[key] >= max_tokens_per_job:
                        continue
                    flat = x.reshape(-1, x.shape[-1])
                    remaining = max_tokens_per_job - counts[key]
                    take = min(remaining, flat.shape[0])
                    if take > 0:
                        storage[key].append(flat[:take].contiguous())
                        counts[key] += take
                if all(count >= max_tokens_per_job for count in counts.values()):
                    break
    finally:
        for handle in handles:
            handle.remove()

    out: dict[str, np.ndarray] = {}
    for job in jobs:
        key = job.key
        if storage[key]:
            out[key] = torch.cat(storage[key], dim=0).numpy().astype(np.float32, copy=False)
        else:
            out[key] = np.empty((0, 0), dtype=np.float32)
    return out


def load_hf_tensor(handle, name: str) -> np.ndarray:
    return handle.get_tensor(name).detach().cpu().to(dtype=torch.float32).numpy()


def load_iq4_tensor(gguf_tensors: dict, name: str) -> np.ndarray:
    tensor = gguf_tensors[name]
    if tensor.tensor_type != GGMLQuantizationType.IQ4_XS:
        raise ValueError(f"{name} is {tensor.tensor_type}, expected IQ4_XS")
    return dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False)


def align_shape(fp: np.ndarray, other: np.ndarray) -> np.ndarray:
    if other.shape == fp.shape:
        return other
    if other.T.shape == fp.shape:
        return other.T.copy()
    if other.size == fp.size and tuple(dim for dim in fp.shape if dim != 1) == other.shape:
        return other.reshape(fp.shape).copy()
    raise ValueError(f"shape mismatch: fp {fp.shape}, other {other.shape}")


def row_sample(
    fp: np.ndarray,
    iq4: np.ndarray,
    max_rows: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = fp.shape[0]
    take = min(rows, max_rows)
    if take < rows:
        idx = np.sort(rng.choice(rows, size=take, replace=False))
        return fp[idx], iq4[idx], idx.astype(np.int64)
    return fp, iq4, np.arange(rows, dtype=np.int64)


def quantize_variant_blocks(w: np.ndarray, alpha: float) -> np.ndarray:
    if w.ndim != 2:
        raise ValueError("expected 2D weight matrix")
    rows, cols = w.shape
    if cols % BLOCK != 0:
        raise ValueError(f"input columns {cols} are not divisible by block {BLOCK}")
    levels = (1 << BITS) - 1
    wb = w.reshape(rows, cols // BLOCK, BLOCK)
    lo = wb.min(axis=2, keepdims=True)
    hi = wb.max(axis=2, keepdims=True)
    center = (hi + lo) * 0.5
    half = np.maximum((hi - lo) * 0.5 * alpha, 1e-8)
    qlo = center - half
    scale = (2.0 * half) / levels
    q = np.rint((wb - qlo) / scale)
    q = np.clip(q, 0, levels)
    return (q * scale + qlo).astype(np.float32, copy=False)


def make_variants(w: np.ndarray) -> list[np.ndarray]:
    return [quantize_variant_blocks(w, alpha) for alpha in ALPHAS]


def feature_bins(x: np.ndarray, thresholds: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    tokens, cols = x.shape
    if cols % BLOCK != 0:
        raise ValueError(f"activation width {cols} is not divisible by block {BLOCK}")
    xb = x.reshape(tokens, cols // BLOCK, BLOCK)
    feature = np.sqrt(np.mean(xb * xb, axis=2))
    if thresholds is None:
        thresholds = np.quantile(feature, [0.25, 0.50, 0.75], axis=0).astype(np.float32)
    bins = np.zeros_like(feature, dtype=np.int8)
    for b in range(feature.shape[1]):
        bins[:, b] = np.searchsorted(thresholds[:, b], feature[:, b], side="right")
    return bins, thresholds


def train_static_indices(x: np.ndarray, w_fp: np.ndarray, variants: list[np.ndarray]) -> np.ndarray:
    rows, nblocks, _ = variants[0].shape
    idx = np.zeros((rows, nblocks), dtype=np.int8)
    for block in range(nblocks):
        xs = x[:, block * BLOCK : (block + 1) * BLOCK]
        target = xs @ w_fp[:, block * BLOCK : (block + 1) * BLOCK].T
        errors = []
        for variant in variants:
            pred = xs @ variant[:, block, :].T
            errors.append(np.mean((target - pred) ** 2, axis=0))
        idx[:, block] = np.argmin(np.stack(errors, axis=0), axis=0).astype(np.int8)
    return idx


def train_dynamic_indices(
    x: np.ndarray,
    w_fp: np.ndarray,
    variants: list[np.ndarray],
    bins: np.ndarray,
    rng: np.random.Generator,
    mode: str,
) -> np.ndarray:
    rows, nblocks, _ = variants[0].shape
    idx = np.zeros((rows, nblocks, NBINS), dtype=np.int8)
    train_bins = bins.copy()
    if mode == "shuffled":
        for block in range(nblocks):
            train_bins[:, block] = rng.permutation(train_bins[:, block])
    elif mode != "real":
        raise ValueError(mode)

    for block in range(nblocks):
        xs_all = x[:, block * BLOCK : (block + 1) * BLOCK]
        w_block = w_fp[:, block * BLOCK : (block + 1) * BLOCK]
        target_all = xs_all @ w_block.T
        all_errors = []
        pred_all_by_alpha = []
        for variant in variants:
            pred_all = xs_all @ variant[:, block, :].T
            pred_all_by_alpha.append(pred_all)
            all_errors.append(np.mean((target_all - pred_all) ** 2, axis=0))
        fallback = np.argmin(np.stack(all_errors, axis=0), axis=0).astype(np.int8)
        for bin_id in range(NBINS):
            token_idx = np.flatnonzero(train_bins[:, block] == bin_id)
            if token_idx.size < 2:
                idx[:, block, bin_id] = fallback
                continue
            target = target_all[token_idx]
            errors = []
            for pred_all in pred_all_by_alpha:
                pred = pred_all[token_idx]
                errors.append(np.mean((target - pred) ** 2, axis=0))
            idx[:, block, bin_id] = np.argmin(np.stack(errors, axis=0), axis=0).astype(np.int8)
    return idx


def random_dynamic_indices(shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, len(ALPHAS), size=shape, dtype=np.int8)


def eval_static(x: np.ndarray, variants: list[np.ndarray], idx: np.ndarray) -> np.ndarray:
    tokens = x.shape[0]
    rows, nblocks, _ = variants[0].shape
    out = np.zeros((tokens, rows), dtype=np.float32)
    for block in range(nblocks):
        xb = x[:, block * BLOCK : (block + 1) * BLOCK]
        for alpha_idx in range(len(ALPHAS)):
            row_idx = np.flatnonzero(idx[:, block] == alpha_idx)
            if row_idx.size:
                out[:, row_idx] += xb @ variants[alpha_idx][row_idx, block, :].T
    return out


def eval_dynamic(x: np.ndarray, bins: np.ndarray, variants: list[np.ndarray], idx: np.ndarray) -> np.ndarray:
    tokens = x.shape[0]
    rows, nblocks, _ = variants[0].shape
    out = np.zeros((tokens, rows), dtype=np.float32)
    for block in range(nblocks):
        xb_all = x[:, block * BLOCK : (block + 1) * BLOCK]
        for bin_id in range(NBINS):
            token_idx = np.flatnonzero(bins[:, block] == bin_id)
            if token_idx.size == 0:
                continue
            xb = xb_all[token_idx]
            for alpha_idx in range(len(ALPHAS)):
                row_idx = np.flatnonzero(idx[:, block, bin_id] == alpha_idx)
                if row_idx.size:
                    out[np.ix_(token_idx, row_idx)] += xb @ variants[alpha_idx][row_idx, block, :].T
    return out


def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.mean(diff * diff))


def gap_closed(base_mse: float, candidate_mse: float, iq4_mse: float) -> float | None:
    gap = base_mse - iq4_mse
    if gap <= 0:
        return None
    return (base_mse - candidate_mse) / gap


def analyze_job(
    job: TensorJob,
    fp: np.ndarray,
    iq4: np.ndarray,
    x_calib: np.ndarray,
    x_eval: np.ndarray,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict:
    if x_calib.shape[0] < 8 or x_eval.shape[0] < 8:
        raise ValueError(f"{job.key} has too few captured activation tokens")
    if x_calib.shape[1] != fp.shape[1] or x_eval.shape[1] != fp.shape[1]:
        raise ValueError(
            f"{job.key} activation width mismatch: calib {x_calib.shape}, eval {x_eval.shape}, weight {fp.shape}"
        )

    fp, iq4, row_idx = row_sample(fp, iq4, args.max_output_rows_per_tensor, rng)
    variants = make_variants(fp)
    base_alpha_idx = int(ALPHAS.index(1.0))
    base_idx = np.full((fp.shape[0], fp.shape[1] // BLOCK), base_alpha_idx, dtype=np.int8)

    calib_bins, thresholds = feature_bins(x_calib)
    eval_bins, _ = feature_bins(x_eval, thresholds)

    static_idx = train_static_indices(x_calib, fp, variants)
    dynamic_idx = train_dynamic_indices(x_calib, fp, variants, calib_bins, rng, mode="real")
    shuffled_idx = train_dynamic_indices(x_calib, fp, variants, calib_bins, rng, mode="shuffled")
    random_idx = random_dynamic_indices(dynamic_idx.shape, rng)

    fp_out = x_eval @ fp.T
    iq4_out = x_eval @ iq4.T
    base_out = eval_static(x_eval, variants, base_idx)
    static_out = eval_static(x_eval, variants, static_idx)
    dynamic_out = eval_dynamic(x_eval, eval_bins, variants, dynamic_idx)
    shuffled_out = eval_dynamic(x_eval, eval_bins, variants, shuffled_idx)
    random_out = eval_dynamic(x_eval, eval_bins, variants, random_idx)

    iq4_mse = mse(fp_out, iq4_out)
    base_mse = mse(fp_out, base_out)
    static_mse = mse(fp_out, static_out)
    dynamic_mse = mse(fp_out, dynamic_out)
    shuffled_mse = mse(fp_out, shuffled_out)
    random_mse = mse(fp_out, random_out)

    dyn_vs_static = 0.0 if static_mse == 0.0 else (static_mse - dynamic_mse) / static_mse
    dyn_vs_base = 0.0 if base_mse == 0.0 else (base_mse - dynamic_mse) / base_mse

    element_count = int(fp_out.size)
    return {
        "key": job.key,
        "layer": job.layer,
        "family": job.family,
        "hf_name": job.hf_name,
        "gguf_name": job.gguf_name,
        "weight_shape": list(fp.shape),
        "sampled_output_rows": int(fp.shape[0]),
        "sampled_output_row_indices": row_idx.tolist(),
        "calib_tokens": int(x_calib.shape[0]),
        "heldout_tokens": int(x_eval.shape[0]),
        "element_count": element_count,
        "iq4_mse": iq4_mse,
        "base3_mse": base_mse,
        "static_mse": static_mse,
        "dynamic_mse": dynamic_mse,
        "shuffled_mse": shuffled_mse,
        "random_mse": random_mse,
        "static_gap_closed_to_iq4": gap_closed(base_mse, static_mse, iq4_mse),
        "dynamic_gap_closed_to_iq4": gap_closed(base_mse, dynamic_mse, iq4_mse),
        "shuffled_gap_closed_to_iq4": gap_closed(base_mse, shuffled_mse, iq4_mse),
        "random_gap_closed_to_iq4": gap_closed(base_mse, random_mse, iq4_mse),
        "dynamic_vs_static_rel_mse": dyn_vs_static,
        "dynamic_vs_base_rel_mse": dyn_vs_base,
    }


def weighted(rows: list[dict], key: str) -> float:
    num = sum(row[key] * row["element_count"] for row in rows)
    den = sum(row["element_count"] for row in rows)
    return float(num / den) if den else float("nan")


def weighted_optional(rows: list[dict], key: str) -> float | None:
    valid = [row for row in rows if row[key] is not None and np.isfinite(row[key])]
    if not valid:
        return None
    return weighted(valid, key)


def aggregate_by(rows: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    out = []
    for value, items in sorted(groups.items()):
        static_mse = weighted(items, "static_mse")
        dynamic_mse = weighted(items, "dynamic_mse")
        out.append(
            {
                key: value,
                "tensors": len(items),
                "element_count": int(sum(row["element_count"] for row in items)),
                "iq4_mse": weighted(items, "iq4_mse"),
                "base3_mse": weighted(items, "base3_mse"),
                "static_mse": static_mse,
                "dynamic_mse": dynamic_mse,
                "shuffled_mse": weighted(items, "shuffled_mse"),
                "random_mse": weighted(items, "random_mse"),
                "static_gap_closed_to_iq4": weighted_optional(items, "static_gap_closed_to_iq4"),
                "dynamic_gap_closed_to_iq4": weighted_optional(items, "dynamic_gap_closed_to_iq4"),
                "dynamic_vs_static_rel_mse": 0.0 if static_mse == 0.0 else (static_mse - dynamic_mse) / static_mse,
            }
        )
    return out


def byte_accounting() -> dict:
    alpha_bits = math.ceil(math.log2(len(ALPHAS)))
    return {
        "block_size": BLOCK,
        "base_code_bpw": float(BITS),
        "base_scale_center_bpw": 32.0 / BLOCK,
        "static_alpha_index_bpw": alpha_bits / BLOCK,
        "dynamic_alpha_index_bpw": NBINS * alpha_bits / BLOCK,
        "static_total_bpw": BITS + 32.0 / BLOCK + alpha_bits / BLOCK,
        "dynamic_total_bpw": BITS + 32.0 / BLOCK + NBINS * alpha_bits / BLOCK,
        "alpha_values": list(ALPHAS),
        "feature_bins": NBINS,
        "feature": "per-token per-input-block RMS with calibration quartile thresholds",
    }


def decide(result: dict) -> tuple[str, str, str]:
    accounting = result["byte_accounting"]
    family_aggs = result["family_aggregates"]
    dynamic_bpw_ok = accounting["dynamic_total_bpw"] <= 3.25
    strong_families = [
        row
        for row in family_aggs
        if (row["dynamic_gap_closed_to_iq4"] or 0.0) >= 0.40
        and row["dynamic_vs_static_rel_mse"] >= 0.15
    ]
    weak_families = [
        row
        for row in family_aggs
        if (row["dynamic_gap_closed_to_iq4"] or 0.0) < 0.20
    ]
    static_beats = [
        row for row in family_aggs if row["dynamic_vs_static_rel_mse"] <= 0.0
    ]
    control_bad = [
        row
        for row in family_aggs
        if row["dynamic_mse"] >= min(row["shuffled_mse"], row["random_mse"])
    ]

    if dynamic_bpw_ok and len(strong_families) >= 2 and not control_bad:
        return (
            "GO",
            "GO: dynamic activation-conditioned scale selection beats equal-byte static scale controls on held-out layer outputs under the category-jump byte budget.",
            "Promote X-03A to a tiny NLL/logit-slice test and design a GGUF-compatible selector/table layout.",
        )
    if len(static_beats) >= max(1, len(family_aggs) // 2) or len(weak_families) >= max(1, len(family_aggs) // 2):
        return (
            "NO-GO",
            "NO-GO: dynamic activation-conditioned scale selection does not beat static calibration strongly enough on held-out layer outputs.",
            "Kill this small-table dynamic scale selector class; keep broader token-dynamic routing and mixed-precision escalation alive.",
        )
    return (
        "GRAY",
        "GRAY: dynamic activation-conditioned scaling shows some signal, but not enough to clear the equal-byte static-table and held-out controls.",
        "If revisited, change the selector object or table parameterization rather than rerunning this exact scale-multiplier variant.",
    )


def make_markdown(result: dict) -> str:
    lines = [
        "# Result Card - run_002-X-03A",
        "",
        "## Status",
        "",
        result["verdict"],
        "",
        "## Decisive Measurement",
        "",
        "Real held-out activations from Qwen3-1.7B were used to compare FP16 layer outputs against actual IQ4_XS dequantized weights, a 3-bit affine base, an equal-byte static scale-index table, a runtime activation-conditioned selector, and shuffled/random selector controls.",
        "",
        "## Byte Accounting",
        "",
        f"- Base 3-bit affine with fp16 center+scale per `{BLOCK}` weights: `{result['byte_accounting']['base_code_bpw'] + result['byte_accounting']['base_scale_center_bpw']:.6f}` bpw",
        f"- Static alpha index total: `{result['byte_accounting']['static_total_bpw']:.6f}` bpw",
        f"- Dynamic 4-bin alpha index total: `{result['byte_accounting']['dynamic_total_bpw']:.6f}` bpw",
        f"- Alpha values: `{result['byte_accounting']['alpha_values']}`",
        "",
        "## Family Aggregates",
        "",
        "| Family | Tensors | Elements | IQ4 MSE | 3-bit base MSE | Static MSE | Dynamic MSE | Dynamic gap closed | Dynamic vs static | Shuffled MSE | Random MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["family_aggregates"]:
        gap = row["dynamic_gap_closed_to_iq4"]
        lines.append(
            f"| {row['family']} | {row['tensors']} | {row['element_count']} | "
            f"{row['iq4_mse']:.6g} | {row['base3_mse']:.6g} | {row['static_mse']:.6g} | "
            f"{row['dynamic_mse']:.6g} | {'n/a' if gap is None else f'{gap:.3%}'} | "
            f"{row['dynamic_vs_static_rel_mse']:.3%} | {row['shuffled_mse']:.6g} | {row['random_mse']:.6g} |"
        )

    overall = result["overall"]
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- IQ4 MSE: `{overall['iq4_mse']:.6g}`",
            f"- 3-bit base MSE: `{overall['base3_mse']:.6g}`",
            f"- static table MSE: `{overall['static_mse']:.6g}`",
            f"- dynamic selector MSE: `{overall['dynamic_mse']:.6g}`",
            f"- dynamic gap closed to IQ4: `{'n/a' if overall['dynamic_gap_closed_to_iq4'] is None else f'{overall['dynamic_gap_closed_to_iq4']:.3%}'}`",
            f"- dynamic vs static relative MSE: `{overall['dynamic_vs_static_rel_mse']:.3%}`",
            "",
            "## GO / NO-GO",
            "",
            result["decision_text"],
            "",
            "## Category-Jump Relevance",
            "",
            "This test directly asks whether a 3-bit resident representation plus a tiny runtime selector payload can move held-out layer outputs toward IQ4_XS. Static calibration wins do not count as a discovery here, because prior art already covers static activation-aware scaling.",
            "",
            "## Next",
            "",
            result["next_step"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hf", default=DEFAULT_HF)
    parser.add_argument("--iq4", default=DEFAULT_IQ4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="0,7,14,21,27")
    parser.add_argument("--families", default="attn_q,attn_o,ffn_gate,ffn_up,ffn_down")
    parser.add_argument("--calib-prompts", type=int, default=128)
    parser.add_argument("--heldout-prompts", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-activation-tokens", type=int, default=256)
    parser.add_argument("--max-output-rows-per-tensor", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    layers = parse_layers(args.layers)
    families = parse_families(args.families)
    jobs = make_jobs(layers, families)

    print("[x03a] loading tokenizer/model for activation capture", flush=True)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    prompts = build_prompts(tokenizer, args.calib_prompts + args.heldout_prompts, args.seed)
    calib_prompts = prompts[: args.calib_prompts]
    heldout_prompts = prompts[args.calib_prompts :]

    print("[x03a] capturing calibration activations", flush=True)
    x_calib = collect_activations(
        model,
        tokenizer,
        jobs,
        calib_prompts,
        args.max_length,
        args.max_activation_tokens,
        device,
    )
    print("[x03a] capturing held-out activations", flush=True)
    x_eval = collect_activations(
        model,
        tokenizer,
        jobs,
        heldout_prompts,
        args.max_length,
        args.max_activation_tokens,
        device,
    )
    del model

    print("[x03a] loading FP16 and IQ4 weights", flush=True)
    reader = GGUFReader(args.iq4)
    gguf_tensors = {tensor.name: tensor for tensor in reader.tensors}

    rows = []
    skipped = []
    with safe_open(args.hf, framework="pt", device="cpu") as hf:
        for job in jobs:
            try:
                print(f"[x03a] analyzing {job.key}", flush=True)
                fp = load_hf_tensor(hf, job.hf_name)
                iq4 = align_shape(fp, load_iq4_tensor(gguf_tensors, job.gguf_name))
                row = analyze_job(job, fp, iq4, x_calib[job.key], x_eval[job.key], args, rng)
                rows.append(row)
            except Exception as exc:
                skipped.append({"key": job.key, "error": str(exc)})
                print(f"[x03a] skipped {job.key}: {exc}", flush=True)

    if not rows:
        raise RuntimeError("no tensor jobs completed")

    family_aggs = aggregate_by(rows, "family")
    layer_aggs = aggregate_by(rows, "layer")
    overall_rows = rows
    overall_static_mse = weighted(overall_rows, "static_mse")
    overall_dynamic_mse = weighted(overall_rows, "dynamic_mse")
    result = {
        "id": "run_002-X-03A",
        "created_utc": datetime.now(UTC).isoformat(),
        "args": vars(args) | {"output_dir": str(args.output_dir)},
        "byte_accounting": byte_accounting(),
        "completed_tensors": len(rows),
        "skipped": skipped,
        "rows": rows,
        "family_aggregates": family_aggs,
        "layer_aggregates": layer_aggs,
        "overall": {
            "element_count": int(sum(row["element_count"] for row in overall_rows)),
            "iq4_mse": weighted(overall_rows, "iq4_mse"),
            "base3_mse": weighted(overall_rows, "base3_mse"),
            "static_mse": overall_static_mse,
            "dynamic_mse": overall_dynamic_mse,
            "shuffled_mse": weighted(overall_rows, "shuffled_mse"),
            "random_mse": weighted(overall_rows, "random_mse"),
            "static_gap_closed_to_iq4": weighted_optional(overall_rows, "static_gap_closed_to_iq4"),
            "dynamic_gap_closed_to_iq4": weighted_optional(overall_rows, "dynamic_gap_closed_to_iq4"),
            "dynamic_vs_static_rel_mse": (
                0.0 if overall_static_mse == 0.0 else (overall_static_mse - overall_dynamic_mse) / overall_static_mse
            ),
        },
    }
    verdict, decision_text, next_step = decide(result)
    result["verdict"] = verdict
    result["decision_text"] = decision_text
    result["next_step"] = next_step

    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "result.md").write_text(make_markdown(result), encoding="utf-8")
    print(f"[x03a] verdict: {verdict}", flush=True)
    print(f"[x03a] wrote {args.output_dir / 'result.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
