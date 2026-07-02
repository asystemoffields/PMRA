"""PMRA CPU prober: two-tier llama.cpp-based probing. No GPU, no torch.

Tier 1 scores every (group, source) promotion candidate with an
imatrix-weighted SSE proxy (activation-aware, AWQ-style): for weight delta
dW between a candidate source and a near-lossless reference,
E[||dW x||^2] ~= sum_j E[x_j^2] * sum_i dW_ij^2, where E[x_j^2] comes from a
llama-imatrix run. This needs only numpy dequantization -- no forward passes.

Tier 2 assembles a real single-promotion GGUF for the candidates that the
proxy knapsack puts at or near the byte-budget boundary, and measures true
NLL with llama-perplexity on the calibration text. The final knapsack runs
on empirical improvements only (unprobed tail is proxy-zeroed, mirroring
--triage in production_mixed_rate_transcoder_gate.py).

Outputs are drop-in compatible with the existing toolchain:
  - checkpoints/allocation_rows.jsonl  (dedup key: (group, source) -- same as
    the Kaggle layer-split merge cell)
  - result.json                        (consumed unchanged by
    build_mixed_gguf_artifact.py)

Probing is resumable and shardable: --shard k/N partitions tier-2 probes by
stable hash so independent workers (Kaggle CPU kernels, GH Actions matrix
jobs) can each emit a partial allocation_rows.jsonl for merging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter
from gguf import quants as gguf_quants

# ---------------------------------------------------------------------------
# Group mapping. Group names MUST match build_tensor_specs() in
# production_mixed_rate_transcoder_gate.py for the given profile, because
# build_mixed_gguf_artifact.py rebuilds the tensor->group mapping from gate
# specs when materializing the artifact. Hybrid profiles (qwen35, nemotron_h)
# carry non-llama tails (ssm_*, attn_qkv, ...) that the default mapping would
# silently exclude from the promotion space.
# ---------------------------------------------------------------------------

GLOBAL_GROUPS = {
    "token_embd.weight": "global:embed",
    "output.weight": "global:output",
    "output_norm.weight": "global:norm",
}
ATTN_TAILS = {"attn_q", "attn_k", "attn_v", "attn_output", "attn_norm", "attn_q_norm", "attn_k_norm"}
MLP_TAILS = {"ffn_gate", "ffn_up", "ffn_down", "ffn_norm"}
_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

# Qwen3.5 DeltaNet hybrid: full-tail -> (layer_family, tensor-mode short).
QWEN35_TAILS = {
    "attn_q.weight": ("attn", "attn_q"),
    "attn_k.weight": ("attn", "attn_k"),
    "attn_v.weight": ("attn", "attn_v"),
    "attn_output.weight": ("attn", "attn_output"),
    "attn_qkv.weight": ("attn", "attn_qkv"),
    "attn_gate.weight": ("attn", "attn_gate"),
    "ssm_a": ("attn", "ssm_a"),
    "ssm_conv1d.weight": ("attn", "ssm_conv1d"),
    "ssm_dt.bias": ("attn", "ssm_dt"),
    "ssm_norm.weight": ("attn", "ssm_norm"),
    "ssm_out.weight": ("attn", "ssm_out"),
    "ssm_alpha.weight": ("attn", "ssm_alpha"),
    "ssm_beta.weight": ("attn", "ssm_beta"),
    "ffn_gate.weight": ("mlp", "ffn_gate"),
    "ffn_up.weight": ("mlp", "ffn_up"),
    "ffn_down.weight": ("mlp", "ffn_down"),
    "attn_norm.weight": ("attn", "attn_norm"),
    "post_attention_norm.weight": ("mlp", "post_attention_norm"),
    "attn_q_norm.weight": ("attn", "attn_q_norm"),
    "attn_k_norm.weight": ("attn", "attn_k_norm"),
}

# NemotronH hybrid: M=Mamba2, -=MLP-only, *=Attention-only, per layer index.
NEMOTRON_H_PATTERN = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"
NEMOTRON_H_FAMILY = {"M": "ssm", "-": "mlp", "*": "attn"}
NEMOTRON_H_TAILS = {
    "M": {
        "attn_norm.weight": "attn_norm",
        "ssm_in.weight": "ssm_in",
        "ssm_conv1d.weight": "ssm_conv1d",
        "ssm_conv1d.bias": "ssm_conv1d_bias",
        "ssm_dt.bias": "ssm_dt_bias",
        "ssm_a": "ssm_a",
        "ssm_d": "ssm_d",
        "ssm_norm.weight": "ssm_norm",
        "ssm_out.weight": "ssm_out",
    },
    "-": {
        "attn_norm.weight": "attn_norm",
        "ffn_up.weight": "ffn_up",
        "ffn_down.weight": "ffn_down",
    },
    "*": {
        "attn_norm.weight": "attn_norm",
        "attn_q.weight": "attn_q",
        "attn_k.weight": "attn_k",
        "attn_v.weight": "attn_v",
        "attn_output.weight": "attn_output",
    },
}


def group_for_tensor(name: str, group_mode: str, tensor_profile: str = "qwen") -> str | None:
    if name in GLOBAL_GROUPS:
        return GLOBAL_GROUPS[name]
    match = _BLK_RE.match(name)
    if not match:
        return None  # rope_freqs etc: stays at base source, never promoted
    layer, tail = int(match.group(1)), match.group(2)
    if group_mode not in ("tensor", "layer_family"):
        raise ValueError(f"unknown group mode {group_mode!r}")

    if tensor_profile == "nemotron_h":
        if layer >= len(NEMOTRON_H_PATTERN):
            raise ValueError(f"layer {layer} outside NEMOTRON_H_PATTERN ({len(NEMOTRON_H_PATTERN)} layers); "
                             "wrong model for this profile?")
        layer_type = NEMOTRON_H_PATTERN[layer]
        short = NEMOTRON_H_TAILS[layer_type].get(tail)
        if short is None:
            return None
        if group_mode == "tensor":
            return f"L{layer}:{short}"
        return f"L{layer}:{NEMOTRON_H_FAMILY[layer_type]}"

    if tensor_profile == "qwen35":
        entry = QWEN35_TAILS.get(tail)
        if entry is None:
            return None
        family, short = entry
        return f"L{layer}:{short}" if group_mode == "tensor" else f"L{layer}:{family}"

    # Default llama-family profiles (qwen/granite/olmo/...): weights and biases
    # of a tensor share one group, norms ride with their family.
    stem, _, kind = tail.partition(".")
    if kind not in ("weight", "bias") or stem not in ATTN_TAILS | MLP_TAILS:
        return None
    if group_mode == "tensor":
        return f"L{layer}:{stem}"
    family = "attn" if stem in ATTN_TAILS else "mlp"
    return f"L{layer}:{family}"


def build_groups(reader: GGUFReader, group_mode: str, tensor_profile: str = "qwen") -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for tensor in reader.tensors:
        group = group_for_tensor(tensor.name, group_mode, tensor_profile)
        if group is not None:
            groups[group].append(tensor.name)
    return dict(groups)


# ---------------------------------------------------------------------------
# GGUF helpers
# ---------------------------------------------------------------------------

def open_sources(paths: dict[str, Path]) -> dict[str, GGUFReader]:
    readers = {}
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing GGUF source {label}: {path}")
        readers[label] = GGUFReader(str(path))
    return readers


def tensor_map(reader: GGUFReader) -> dict[str, object]:
    return {tensor.name: tensor for tensor in reader.tensors}


def dequant(tensor) -> np.ndarray:
    """Dequantize a ReaderTensor to float32 (rows, in_features)."""
    if tensor.data.dtype != np.uint8:
        data = tensor.data.astype(np.float32)
    else:
        data = gguf_quants.dequantize(tensor.data, tensor.tensor_type)
    in_features = int(tensor.shape[0])
    if data.size % in_features == 0 and data.ndim <= 2:
        return data.reshape(-1, in_features)
    return data.reshape(-1, data.shape[-1])


def parse_block_count(reader: GGUFReader) -> int:
    arch = str(reader.fields["general.architecture"].contents())
    return int(reader.fields[f"{arch}.block_count"].contents())


def field_value(field):
    value = field.contents()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [item.item() if isinstance(item, np.generic) else item for item in value]
    return value


def assemble_mixed_gguf(
    output_path: Path,
    base_reader: GGUFReader,
    source_tensors: dict[str, dict[str, object]],
    tensor_sources: dict[str, str],
    base_label: str,
) -> int:
    """Write a GGUF copying each tensor's raw payload from its assigned source."""
    writer = GGUFWriter(output_path, str(base_reader.fields["general.architecture"].contents()))
    writer.data_alignment = int(base_reader.alignment)
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for key, field in base_reader.fields.items():
        if key in skip:
            continue
        value_type = field.types[0]
        sub_type = field.types[1] if value_type == GGUFValueType.ARRAY and len(field.types) > 1 else None
        writer.add_key_value(key, field_value(field), value_type, sub_type=sub_type)
    payload = 0
    for tensor in base_reader.tensors:
        source = tensor_sources.get(tensor.name, base_label)
        chosen = source_tensors[source][tensor.name]
        data = chosen.data
        raw_dtype = chosen.tensor_type if data.dtype == np.uint8 else None
        writer.add_tensor(
            chosen.name,
            data,
            raw_shape=[int(dim) for dim in data.shape],
            raw_dtype=raw_dtype,
        )
        payload += int(chosen.n_bytes)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    return payload


# ---------------------------------------------------------------------------
# llama.cpp subprocess wrappers
# ---------------------------------------------------------------------------

_PPL_RE = re.compile(r"Final estimate: PPL = ([0-9.]+(?:e[+-]?\d+)?)")


def run_perplexity(bin_dir: Path, gguf: Path, text: Path, ctx: int, chunks: int, threads: int) -> dict:
    cmd = [
        str(bin_dir / "llama-perplexity"),
        "-m", str(gguf),
        "-f", str(text),
        "--ctx-size", str(ctx),
        "--chunks", str(chunks),
        "-t", str(threads),
        "--no-warmup",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    match = _PPL_RE.search(output)
    if not match:
        raise RuntimeError(f"llama-perplexity gave no PPL for {gguf.name} (exit {proc.returncode}):\n{output[-2000:]}")
    ppl = float(match.group(1))
    result = {"ppl": ppl, "nll": math.log(ppl), "tokens": ctx * chunks}
    # The "calculating perplexity" marker logs to stderr while the [k]ppl
    # progress prints to stdout, so try stderr-first ordering too.
    cum = (_parse_cumulative_ppls(proc.stderr + proc.stdout)
           or _parse_cumulative_ppls(output))
    if cum:
        result["chunk_nlls"] = chunk_nlls_from_cumulative(cum)
    return result


# Running per-chunk progress, e.g. "[1]4.6043,[2]4.9871,...". Only valid after
# the "calculating perplexity over N chunks" marker (model-load logs also
# contain bracketed numbers).
_CHUNK_PROGRESS_RE = re.compile(r"\[(\d+)\](\d+\.\d+)")
_CHUNK_START_MARKER = "calculating perplexity over"


def _parse_cumulative_ppls(output: str) -> list[float]:
    start = output.find(_CHUNK_START_MARKER)
    if start < 0:
        return []
    cum: list[float] = []
    for match in _CHUNK_PROGRESS_RE.finditer(output, start):
        if int(match.group(1)) == len(cum) + 1:  # accept only the monotone sequence
            cum.append(float(match.group(2)))
    return cum


def chunk_nlls_from_cumulative(cum: list[float]) -> list[float]:
    """Recover per-chunk mean NLLs from llama-perplexity's running PPL values."""
    out, prev_total = [], 0.0
    for k, ppl in enumerate(cum, start=1):
        total = k * math.log(ppl)
        out.append(total - prev_total)
        prev_total = total
    return out


def paired_chunk_stats(base_chunk_nlls: list[float], probe_chunk_nlls: list[float]) -> tuple[float, float]:
    """(mean, se) of the paired per-chunk NLL delta base - probe (positive = probe better)."""
    k = min(len(base_chunk_nlls), len(probe_chunk_nlls))
    deltas = [b - p for b, p in zip(base_chunk_nlls[:k], probe_chunk_nlls[:k])]
    mean = sum(deltas) / k
    if k < 2:
        return mean, float("inf")
    var = sum((d - mean) ** 2 for d in deltas) / (k - 1)
    return mean, (var / k) ** 0.5


def paired_probe_stats(base_chunk_nlls: list[float], cum_ppls: list[float]) -> tuple[float, float]:
    """(mean_improvement, standard_error) of the paired per-chunk NLL delta
    base - probe over the chunks seen so far. The mean telescopes to
    mean(base[:k]) - log(cum_ppl_k), so printed-precision rounding does not
    accumulate; per-chunk rounding noise only enters the variance estimate."""
    k = len(cum_ppls)
    mean = sum(base_chunk_nlls[:k]) / k - math.log(cum_ppls[-1])
    if k < 2:
        return mean, float("inf")
    deltas = [b - p for b, p in zip(base_chunk_nlls[:k], chunk_nlls_from_cumulative(cum_ppls))]
    var = sum((d - mean) ** 2 for d in deltas) / (k - 1)
    return mean, (var / k) ** 0.5


def run_perplexity_early_stop(
    bin_dir: Path,
    gguf: Path,
    text: Path,
    ctx: int,
    chunks: int,
    threads: int,
    base_chunk_nlls: list[float],
    min_chunks: int,
    se_stop: float,
    rel_stop: float = 0.10,
) -> dict:
    """Tier-2 probe evaluation with sequential stopping on the paired
    per-chunk delta vs the low base. Stops once the 95% CI half-width of the
    mean improvement drops below max(se_stop, rel_stop*|mean|) — an absolute
    floor for sub-noise candidates, relative precision for large deltas — or
    once the improvement is clearly negative; either way the estimate stays
    usable by the knapsack."""
    cmd = [
        str(bin_dir / "llama-perplexity"),
        "-m", str(gguf),
        "-f", str(text),
        "--ctx-size", str(ctx),
        "--chunks", str(chunks),
        "-t", str(threads),
        "--no-warmup",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    fd = proc.stdout.fileno()
    buf = b""
    cum: list[float] = []
    early_stopped = False
    try:
        while True:
            data = os.read(fd, 4096)
            if not data:
                break
            buf += data
            parsed = _parse_cumulative_ppls(buf.decode("utf-8", errors="replace"))
            if len(parsed) <= len(cum):
                continue
            cum = parsed
            k = len(cum)
            if k < min_chunks or k > len(base_chunk_nlls):
                continue
            mean, se = paired_probe_stats(base_chunk_nlls, cum)
            half_width = 1.96 * se
            if half_width < max(se_stop, rel_stop * abs(mean)) or mean + half_width < 0:
                early_stopped = True
                proc.kill()
                break
    finally:
        proc.wait()
    if not cum:
        output = buf.decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-perplexity gave no chunk progress for {gguf.name} "
                           f"(exit {proc.returncode}):\n{output[-2000:]}")
    mean, se = paired_probe_stats(base_chunk_nlls, cum)
    return {
        "improvement": mean,
        "improvement_se": se,
        "chunks_used": len(cum),
        "early_stopped": early_stopped,
        "tokens": ctx * len(cum),
    }


def run_imatrix(bin_dir: Path, gguf: Path, text: Path, out: Path, ctx: int, threads: int) -> Path:
    cmd = [
        str(bin_dir / "llama-imatrix"),
        "-m", str(gguf),
        "-f", str(text),
        "-o", str(out),
        "--output-format", "gguf",
        "--process-output",
        "--ctx-size", str(ctx),
        "-t", str(threads),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if not out.exists():
        raise RuntimeError(f"llama-imatrix produced no output (exit {proc.returncode}):\n{(proc.stdout + proc.stderr)[-2000:]}")
    return out


def load_imatrix(path: Path) -> dict[str, np.ndarray]:
    """Return per-tensor mean squared input activation E[x_j^2] vectors."""
    reader = GGUFReader(str(path))
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for tensor in reader.tensors:
        if tensor.name.endswith(".in_sum2"):
            sums[tensor.name[: -len(".in_sum2")]] = np.asarray(tensor.data, dtype=np.float64).reshape(-1)
        elif tensor.name.endswith(".counts"):
            counts[tensor.name[: -len(".counts")]] = np.asarray(tensor.data, dtype=np.float64).reshape(-1)
    out = {}
    for name, total in sums.items():
        n = counts.get(name)
        out[name] = total / float(n.max()) if n is not None and n.max() > 0 else total
    return out


# ---------------------------------------------------------------------------
# Tier 1: imatrix-weighted SSE proxy
# ---------------------------------------------------------------------------

def weighted_sse(candidate, reference, imatrix_vec: np.ndarray | None) -> float:
    """sum_j E[x_j^2] * sum_i dW_ij^2 over the tensor (float64 accumulate)."""
    delta = dequant(candidate).astype(np.float64) - dequant(reference).astype(np.float64)
    if imatrix_vec is not None and delta.ndim == 2 and delta.shape[1] == imatrix_vec.shape[0]:
        return float((np.square(delta).sum(axis=0) * imatrix_vec).sum())
    return float(np.square(delta).sum())


def tier1_scores(
    groups: dict[str, list[str]],
    readers: dict[str, GGUFReader],
    tensors_by_source: dict[str, dict[str, object]],
    low: str,
    high_sources: list[str],
    ref: str,
    imatrix: dict[str, np.ndarray],
    cache_path: Path,
) -> dict:
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo_groups = [g for g in groups if g not in cache]
    for idx, group in enumerate(sorted(todo_groups), start=1):
        errs: dict[str, float] = {}
        for source in [low, *high_sources]:
            total = 0.0
            for name in groups[group]:
                cand = tensors_by_source[source].get(name)
                reft = tensors_by_source[ref].get(name)
                if cand is None or reft is None:
                    continue
                if int(cand.n_bytes) == int(reft.n_bytes) and cand.tensor_type == reft.tensor_type:
                    continue  # identical format to reference: zero delta
                total += weighted_sse(cand, reft, imatrix.get(name))
            errs[source] = total
        cache[group] = errs
        if idx % 10 == 0 or idx == len(todo_groups):
            cache_path.write_text(json.dumps(cache))
            print(f"[tier1] scored group {idx}/{len(todo_groups)}", flush=True)
    cache_path.write_text(json.dumps(cache))
    return cache


def build_candidates(
    groups: dict[str, list[str]],
    tensors_by_source: dict[str, dict[str, object]],
    scores: dict,
    low: str,
    high_sources: list[str],
) -> list[dict]:
    def group_bytes(group: str, source: str) -> int:
        return sum(
            int(tensors_by_source[source][name].n_bytes)
            for name in groups[group]
            if name in tensors_by_source[source]
        )

    candidates = []
    for group in sorted(groups):
        low_bytes = group_bytes(group, low)
        err_low = scores[group][low]
        for source in high_sources:
            high_bytes = group_bytes(group, source)
            extra = high_bytes - low_bytes
            if extra <= 0:
                continue
            proxy = max(0.0, err_low - scores[group][source])
            candidates.append(
                {
                    "group": group,
                    "source": source,
                    "low_bytes": int(low_bytes),
                    "high_bytes": int(high_bytes),
                    "extra_bytes": int(extra),
                    "proxy_improvement": proxy,
                    "proxy_score_per_mbyte": proxy / (extra / 1_000_000),
                }
            )
    return candidates


# ---------------------------------------------------------------------------
# Knapsack (mirrors select_knapsack_by_value in the gate script; stdlib-only
# copy so the prober stays torch-free)
# ---------------------------------------------------------------------------

def select_knapsack(rows: list[dict], budget_extra: int, value_key: str, max_units: int = 200_000) -> list[dict]:
    options_by_group: dict[str, list[dict]] = {}
    for row in rows:
        extra = int(row.get("extra_bytes", 0))
        value = float(row.get(value_key, 0.0))
        if extra <= 0 or value <= 0.0 or extra > budget_extra:
            continue
        options_by_group.setdefault(row["group"], []).append(row)
    if not options_by_group:
        return []
    byte_unit = int(budget_extra)
    for options in options_by_group.values():
        for row in options:
            byte_unit = math.gcd(byte_unit, int(row["extra_bytes"]))
    if byte_unit <= 0 or budget_extra // byte_unit > max_units:
        byte_unit = max(1, budget_extra // max_units)
    budget_units = budget_extra // byte_unit
    neg_inf = float("-inf")
    dp = [neg_inf] * (budget_units + 1)
    dp[0] = 0.0
    stage_choices: list[tuple[list[int], list[int], list[dict]]] = []
    for group in sorted(options_by_group):
        options = sorted(
            options_by_group[group],
            key=lambda row: (float(row.get(value_key, 0.0)), -int(row["extra_bytes"]), row["source"]),
            reverse=True,
        )
        next_dp = dp[:]
        choices = [-1] * (budget_units + 1)
        previous = [-1] * (budget_units + 1)
        for option_idx, row in enumerate(options):
            weight = int(row["extra_bytes"]) // byte_unit
            if weight > budget_units:
                continue
            value = float(row[value_key])
            for used, current in enumerate(dp[: budget_units - weight + 1]):
                if current == neg_inf:
                    continue
                if current + value > next_dp[used + weight] + 1e-12:
                    next_dp[used + weight] = current + value
                    choices[used + weight] = option_idx
                    previous[used + weight] = used
        dp = next_dp
        stage_choices.append((choices, previous, options))
    best_used = max(range(budget_units + 1), key=lambda used: (dp[used], -used))
    selected: list[dict] = []
    used = best_used
    for choices, previous, options in reversed(stage_choices):
        option_idx = choices[used]
        if option_idx >= 0:
            selected.append(options[option_idx])
            used = previous[used]
    selected.reverse()
    return selected


def select_random(rows: list[dict], budget_extra: int, rng: random.Random) -> list[dict]:
    pool = [row for row in rows if row["extra_bytes"] > 0]
    rng.shuffle(pool)
    selected, seen, used = [], set(), 0
    for row in pool:
        if row["group"] in seen or used + row["extra_bytes"] > budget_extra:
            continue
        selected.append(row)
        seen.add(row["group"])
        used += row["extra_bytes"]
    return selected


# ---------------------------------------------------------------------------
# Tier 2: empirical llama-perplexity probes
# ---------------------------------------------------------------------------

def probe_key(row: dict) -> str:
    return f"{row['group']}|{row['source']}"


def shard_of(key: str, shards: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % shards


def choose_probe_set(
    candidates: list[dict],
    proxy_selection: list[dict],
    budget_extra: int,
    probe_fraction: float,
    boundary_band: float,
    max_probes: int,
) -> list[dict]:
    """Proxy-knapsack picks plus the triage-style top-fraction + boundary band."""
    ranked = sorted(candidates, key=lambda c: c["proxy_score_per_mbyte"], reverse=True)
    seen_groups: set[str] = set()
    cum = 0
    cutoff = len(ranked)
    for idx, cand in enumerate(ranked):
        if cand["group"] in seen_groups:
            continue
        seen_groups.add(cand["group"])
        cum += cand["extra_bytes"]
        if cum > budget_extra:
            cutoff = idx
            break
    upto = min(len(ranked), max(math.ceil(len(ranked) * probe_fraction), cutoff + math.ceil(len(ranked) * boundary_band)))
    chosen: dict[str, dict] = {probe_key(row): row for row in proxy_selection}
    for cand in ranked[:upto]:
        chosen.setdefault(probe_key(cand), cand)
    ordered = list(chosen.values())
    if len(ordered) > max_probes:
        in_selection = {probe_key(row) for row in proxy_selection}
        ordered.sort(key=lambda c: (probe_key(c) not in in_selection, -c["proxy_score_per_mbyte"]))
        ordered = ordered[:max_probes]
    return ordered


def load_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[probe_key(row)] = row
    return rows


def append_row(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def apply_code_guardrail(
    verdict: str,
    code_nlls: dict[str, float | None],
    candidate: str,
    target: str,
    epsilon: float,
    code_chunk_nlls: dict[str, list[float] | None] | None = None,
) -> tuple[str, dict]:
    """Release guardrail: the candidate mix's code-corpus NLL must not exceed
    the stock target quant's by more than epsilon nats. epsilon<=0 records the
    code NLLs without enforcing. A failing guardrail downgrades GO to GRAY —
    the mix still beats target/random on the general corpus, but is not
    ship-clean per-domain."""
    block: dict = {
        "enforced": epsilon > 0,
        "epsilon": epsilon,
        "candidate": candidate,
        "target": target,
        "code_nll": {k: v for k, v in code_nlls.items() if v is not None},
    }
    cand_nll = code_nlls.get(candidate)
    stock_nll = code_nlls.get(target)
    if cand_nll is None or stock_nll is None:
        block["pass"] = None
        return verdict, block
    regression = cand_nll - stock_nll
    block["regression_vs_target"] = regression
    if code_chunk_nlls:
        cand_chunks = code_chunk_nlls.get(candidate)
        stock_chunks = code_chunk_nlls.get(target)
        if cand_chunks and stock_chunks:
            k = min(len(cand_chunks), len(stock_chunks))
            deltas = [c - s for c, s in zip(cand_chunks[:k], stock_chunks[:k])]
            mean = sum(deltas) / k
            if k >= 2:
                var = sum((d - mean) ** 2 for d in deltas) / (k - 1)
                block["regression_se"] = (var / k) ** 0.5
                block["regression_chunks"] = k
    block["pass"] = (regression <= epsilon) if epsilon > 0 else None
    if block["pass"] is False and verdict == "GO":
        block["verdict_before_guardrail"] = verdict
        verdict = "GRAY"
    return verdict, block


def parse_source_specs(items: list[str]) -> dict[str, Path]:
    sources = {}
    for item in items:
        label, _, path = item.partition("=")
        if not label or not path:
            raise ValueError(f"source must look like label=path, got {item!r}")
        sources[label.strip()] = Path(path.strip())
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", default=[], help="GGUF source as label=path (repeat).")
    parser.add_argument("--low-source", required=True)
    parser.add_argument("--target-source", required=True, help="Defines the byte budget.")
    parser.add_argument("--high-sources", required=True, help="Comma-separated promotion sources.")
    parser.add_argument("--ref-source", default=None, help="Near-lossless reference for tier-1 proxy (default: f16 if present, else target).")
    parser.add_argument("--calib-text", type=Path, required=True)
    parser.add_argument("--eval-text", type=Path, default=None, help="Held-out text for final variant NLLs (default: calib text).")
    parser.add_argument("--code-text", type=Path, default=None,
                        help="Code corpus for the release guardrail: measures code-domain NLL of low/target/mix variants at finalize.")
    parser.add_argument("--code-no-regress", type=float, default=0.0,
                        help="Release guardrail epsilon (nats): the mix's code NLL must not exceed the stock target's "
                             "by more than this. Requires --code-text. 0=record code NLLs without enforcing.")
    parser.add_argument("--code-chunks", type=int, default=0,
                        help="Chunks for code-corpus evals (0=same as --chunks). More chunks tighten the guardrail SE.")
    parser.add_argument("--tier2-time-budget-min", type=float, default=0.0,
                        help="Stop launching tier-2 probes once the projected total exceeds this many minutes "
                             "(completed rows are kept; unprobed candidates fall to proxy backfill). 0=off.")
    parser.add_argument("--imatrix", type=Path, default=None, help="Existing imatrix GGUF (skips llama-imatrix run).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llama-bin", type=Path, required=True, help="Directory with llama-perplexity / llama-imatrix.")
    parser.add_argument("--group-mode", default="layer_family", choices=["layer_family", "tensor"])
    parser.add_argument("--tensor-profile", default="qwen",
                        help="Selects the tensor->group mapping (must match the gate profile; hybrid profiles "
                             "qwen35/nemotron_h map ssm_*/attn_qkv tails) and is recorded for "
                             "build_mixed_gguf_artifact.py group reconstruction.")
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=24)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--probe-fraction", type=float, default=0.15)
    parser.add_argument("--boundary-band", type=float, default=0.10)
    parser.add_argument("--max-probes", type=int, default=48)
    parser.add_argument("--probe-min-chunks", type=int, default=8,
                        help="Minimum chunks before a tier-2 probe may stop early.")
    parser.add_argument("--probe-se-stop", type=float, default=0.001,
                        help="Stop a probe once the 95%% CI half-width of the paired "
                             "improvement estimate (nats) drops below this.")
    parser.add_argument("--probe-rel-stop", type=float, default=0.10,
                        help="...or below this fraction of |improvement| (relative "
                             "precision for large deltas).")
    parser.add_argument("--no-probe-early-stop", action="store_true",
                        help="Always run tier-2 probes for the full --chunks.")
    parser.add_argument("--shard", default=None, help="k/N: only run tier-2 probes whose hash lands in shard k (0-based).")
    parser.add_argument("--stages", default="all", choices=["all", "tier1", "tier2", "finalize"],
                        help="tier1: stop after proxy scoring (prepare job). tier2: stop after probes "
                             "(shard worker). finalize: skip probing, select from merged rows.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--variant-name", default="c2_calib_knapsack_mixed")
    parser.add_argument("--no-backfill", action="store_true",
                        help="Leave budget unspent when empirical improvements run out, instead of "
                             "filling the remainder by proxy rank (probed-negative candidates stay excluded).")
    args = parser.parse_args()
    if args.code_no_regress > 0 and args.code_text is None:
        parser.error("--code-no-regress requires --code-text")

    high_sources = [s.strip() for s in args.high_sources.split(",") if s.strip()]
    source_paths = parse_source_specs(args.source)
    for required in [args.low_source, args.target_source, *high_sources]:
        if required not in source_paths:
            raise ValueError(f"--source missing required label {required!r}")
    ref = args.ref_source or ("f16" if "f16" in source_paths else args.target_source)
    eval_text = args.eval_text or args.calib_text

    out_dir = args.output_dir
    ckpt_dir = out_dir / "checkpoints"
    work_dir = out_dir / "work"
    for d in [out_dir, ckpt_dir, work_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[cpu-prober] sources: {sorted(source_paths)} low={args.low_source} target={args.target_source} ref={ref}", flush=True)
    readers = open_sources(source_paths)
    tensors_by_source = {label: tensor_map(reader) for label, reader in readers.items()}
    base_reader = readers[args.low_source]
    groups = build_groups(base_reader, args.group_mode, args.tensor_profile)
    grouped_names = {name for names in groups.values() for name in names}
    total_weights = sum(int(t.n_elements) for t in base_reader.tensors if t.name in grouped_names)
    n_layers = parse_block_count(base_reader)
    layers = list(range(n_layers))
    print(f"[cpu-prober] {len(groups)} groups over {len(grouped_names)} tensors, {n_layers} layers, {total_weights/1e6:.1f}M weights", flush=True)

    def payload_bytes(label: str) -> int:
        return sum(int(t.n_bytes) for t in readers[label].tensors if t.name in grouped_names)

    budget_extra = payload_bytes(args.target_source) - payload_bytes(args.low_source)
    if budget_extra <= 0:
        raise ValueError(f"target {args.target_source} is not larger than low {args.low_source}; no promotion budget")
    print(f"[cpu-prober] budget_extra={budget_extra/1e6:.2f} MB", flush=True)

    # ---- Cache signature ---------------------------------------------------
    # Checkpoint rows are only trusted when they were measured under the same
    # conditions: model, ctx, llama.cpp build, and the exact corpus bytes.
    # Anything else (a stale kernel attachment, a different llama.cpp release,
    # a changed corpus) silently mixing into the verdict is the condition-
    # mismatch failure mode this guards against.
    ver_proc = subprocess.run([str(args.llama_bin / "llama-perplexity"), "--version"],
                              capture_output=True, text=True)
    llama_build = ((ver_proc.stderr + ver_proc.stdout).strip().splitlines() or ["unknown"])[0]
    base_sig = f"{source_paths[args.target_source].name}|ctx{args.ctx}|{llama_build}"
    print(f"[cpu-prober] cache signature: {base_sig}", flush=True)

    _text_sha_cache: dict[str, str] = {}

    def text_sha(path: Path) -> str:
        key = str(path)
        if key not in _text_sha_cache:
            _text_sha_cache[key] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        return _text_sha_cache[key]

    # ---- Tier 0: baselines ------------------------------------------------
    scalar_path = ckpt_dir / "scalar_evals.jsonl"
    scalar_cache = {row["label"]: row for row in (json.loads(l) for l in scalar_path.read_text().splitlines())} if scalar_path.exists() else {}

    def eval_gguf(label: str, gguf: Path, text: Path, tag: str, require_chunks: bool = False,
                  chunks: int = 0) -> dict:
        chunks = chunks or args.chunks
        key = f"{label}@{tag}"
        cached = scalar_cache.get(key)
        if cached is not None:
            stale = (cached.get("sig") != base_sig or cached.get("text_sha") != text_sha(text)
                     or cached.get("chunks") != chunks)
            if not stale and not (require_chunks and "chunk_nlls" not in cached):
                return cached
            reason = "stale conditions" if stale else "lacks per-chunk NLLs"
            print(f"[tier0] cached {key} {reason}; re-measuring", flush=True)
        print(f"[tier0] llama-perplexity {label} on {text.name}", flush=True)
        result = run_perplexity(args.llama_bin, gguf, text, args.ctx, chunks, args.threads)
        row = {"label": key, "sig": base_sig, "text_sha": text_sha(text), "chunks": chunks, **result}
        scalar_cache[key] = row
        append_row(scalar_path, row)
        return row

    # Paired early stopping needs the low base's per-chunk NLLs; finalize never
    # probes, so a chunk-less cached row only forces a re-measure when this
    # invocation will actually run tier-2 probes.
    need_chunks = args.stages in ("all", "tier2") and not args.no_probe_early_stop
    low_calib = eval_gguf(args.low_source, source_paths[args.low_source], args.calib_text, "calib",
                          require_chunks=need_chunks)
    nll_low = low_calib["nll"]
    base_chunk_nlls = low_calib.get("chunk_nlls")
    if not args.no_probe_early_stop and not base_chunk_nlls:
        print("[tier0] cached low-source eval has no per-chunk NLLs; "
              "tier-2 probes will run full-length", flush=True)

    # ---- Tier 1: proxy ----------------------------------------------------
    imatrix_path = args.imatrix
    if imatrix_path is None:
        imatrix_path = work_dir / "imatrix.gguf"
        if not imatrix_path.exists():
            print(f"[tier1] computing imatrix on {ref}", flush=True)
            run_imatrix(args.llama_bin, source_paths[ref], args.calib_text, imatrix_path, args.ctx, args.threads)
    imatrix = load_imatrix(imatrix_path)
    print(f"[tier1] imatrix entries: {len(imatrix)}", flush=True)

    scores = tier1_scores(groups, readers, tensors_by_source, args.low_source, high_sources, ref,
                          imatrix, ckpt_dir / "tier1_scores.json")
    candidates = build_candidates(groups, tensors_by_source, scores, args.low_source, high_sources)
    print(f"[tier1] {len(candidates)} promotion candidates", flush=True)

    proxy_selection = select_knapsack(candidates, budget_extra, "proxy_improvement")
    print(f"[tier1] proxy knapsack selected {len(proxy_selection)} groups", flush=True)
    if args.stages == "tier1":
        print("[tier1] stage complete; checkpoints ready for shard workers", flush=True)
        return 0

    # ---- Tier 2: empirical probes ------------------------------------------
    rows_path = ckpt_dir / "allocation_rows.jsonl"
    done = load_rows(rows_path)
    stale_keys = {k for k, r in done.items()
                  if r.get("sig") != base_sig or r.get("text_sha") != text_sha(args.calib_text)}
    if stale_keys:
        print(f"[tier2] discarding {len(stale_keys)} checkpoint rows measured under different conditions", flush=True)
        done = {k: r for k, r in done.items() if k not in stale_keys}
    probe_set = choose_probe_set(candidates, proxy_selection, budget_extra,
                                 args.probe_fraction, args.boundary_band, args.max_probes)
    shard_idx = shard_count = None
    if args.shard:
        shard_idx, shard_count = (int(part) for part in args.shard.split("/"))

    if args.stages != "finalize":
        todo = [c for c in probe_set if probe_key(c) not in done]
        if shard_count:
            todo = [c for c in todo if shard_of(probe_key(c), shard_count) == shard_idx]
        print(f"[tier2] probing {len(todo)} of {len(probe_set)} candidates"
              + (f" (shard {shard_idx}/{shard_count})" if shard_count else ""), flush=True)
        probe_gguf = work_dir / "probe.gguf"
        tier2_start = time.monotonic()
        for idx, cand in enumerate(todo, start=1):
            if args.tier2_time_budget_min > 0 and idx > 1:
                elapsed_min = (time.monotonic() - tier2_start) / 60
                per_probe = elapsed_min / (idx - 1)
                if elapsed_min + per_probe > args.tier2_time_budget_min:
                    print(f"[tier2] time budget: {elapsed_min:.0f} min after {idx - 1} probes "
                          f"(~{per_probe:.1f} min/probe); skipping remaining {len(todo) - idx + 1} "
                          f"(they fall to proxy backfill)", flush=True)
                    break
            assemble_mixed_gguf(
                probe_gguf, base_reader, tensors_by_source,
                {name: cand["source"] for name in groups[cand["group"]]},
                args.low_source,
            )
            if base_chunk_nlls and not args.no_probe_early_stop:
                probe = run_perplexity_early_stop(
                    args.llama_bin, probe_gguf, args.calib_text, args.ctx, args.chunks,
                    args.threads, base_chunk_nlls, args.probe_min_chunks,
                    args.probe_se_stop, args.probe_rel_stop)
                improvement = probe["improvement"]
                probe_meta = {
                    "probe_chunks": probe["chunks_used"],
                    "probe_early_stopped": probe["early_stopped"],
                    "probe_improvement_se": probe["improvement_se"],
                }
            else:
                result = run_perplexity(args.llama_bin, probe_gguf, args.calib_text, args.ctx, args.chunks, args.threads)
                improvement = nll_low - result["nll"]
                probe_meta = {"probe_chunks": args.chunks, "probe_early_stopped": False}
            row = {
                **{k: cand[k] for k in ["group", "source", "low_bytes", "high_bytes", "extra_bytes"]},
                "sig": base_sig,
                "text_sha": text_sha(args.calib_text),
                "saved_bytes": 0,
                "base_source": None,
                "calib_nll": nll_low - improvement,
                "calib_nll_improvement": improvement,
                "calib_nll_loss": max(0.0, -improvement),
                "calib_score_per_mbyte": improvement / (cand["extra_bytes"] / 1_000_000),
                "calib_knapsack_value": improvement,
                "weight_sse_delta": cand["proxy_improvement"],
                "weight_score_per_mbyte": cand["proxy_score_per_mbyte"],
                "probe_backend": "llama_cpp_cpu",
                **probe_meta,
            }
            done[probe_key(row)] = row
            append_row(rows_path, row)
            print(f"[tier2] {idx}/{len(todo)} {row['group']} <- {row['source']}: dNLL={improvement:+.5f}"
                  f" ({row['probe_chunks']} chunks{' early-stop' if row['probe_early_stopped'] else ''})", flush=True)
        if probe_gguf.exists():
            probe_gguf.unlink()
        if args.stages == "tier2":
            print(f"[tier2] shard complete; rows in {rows_path}", flush=True)
            return 0

    # ---- Final selection ----------------------------------------------------
    empirical = list(done.values())
    proxy_zeroed = [
        {**cand, "calib_nll_improvement": 0.0, "calib_score_per_mbyte": 0.0,
         "weight_sse_delta": cand["proxy_improvement"], "weight_score_per_mbyte": cand["proxy_score_per_mbyte"],
         "triage_proxy": True}
        for cand in candidates if probe_key(cand) not in done
    ]
    final_selection = select_knapsack(empirical, budget_extra, "calib_nll_improvement")
    used = sum(row["extra_bytes"] for row in final_selection)
    print(f"[final] knapsack: {len(final_selection)} promotions, {used/1e6:.2f}/{budget_extra/1e6:.2f} MB", flush=True)

    # Backfill: sub-noise candidates shouldn't be selected on, but the byte
    # budget shouldn't be forfeited either — the bpw->NLL ladder is smooth, so
    # unspent bytes are pure loss. Fill the remainder by proxy rank with
    # unprobed candidates (probed-negative stay excluded: we have evidence).
    if not args.no_backfill:
        selected_groups = {row["group"] for row in final_selection}
        pool = sorted(
            (c for c in candidates
             if c["group"] not in selected_groups
             and probe_key(c) not in done
             and c["proxy_improvement"] > 0),
            key=lambda c: c["proxy_score_per_mbyte"], reverse=True,
        )
        backfilled = 0
        for cand in pool:
            if used + cand["extra_bytes"] > budget_extra:
                continue
            final_selection.append({
                **{k: cand[k] for k in ["group", "source", "low_bytes", "high_bytes", "extra_bytes"]},
                "saved_bytes": 0, "base_source": None,
                "calib_nll": None, "calib_nll_improvement": 0.0,
                "calib_nll_loss": 0.0, "calib_score_per_mbyte": 0.0,
                "calib_knapsack_value": 0.0,
                "weight_sse_delta": cand["proxy_improvement"],
                "weight_score_per_mbyte": cand["proxy_score_per_mbyte"],
                "backfill_proxy": True,
            })
            selected_groups.add(cand["group"])
            used += cand["extra_bytes"]
            backfilled += 1
        print(f"[final] backfill: +{backfilled} proxy-ranked promotions -> {used/1e6:.2f}/{budget_extra/1e6:.2f} MB", flush=True)

    # Allocation control: 3 seeded random draws; the verdict leg reads the
    # median so one lucky/unlucky draw can't flip GO/GRAY on its own.
    random_selections = {
        "c2_random_same_budget": select_random(candidates, budget_extra, random.Random(args.seed)),
        "c2_random_same_budget_b": select_random(candidates, budget_extra, random.Random(args.seed + 1)),
        "c2_random_same_budget_c": select_random(candidates, budget_extra, random.Random(args.seed + 2)),
    }

    def selection_sources(selection: list[dict]) -> dict[str, str]:
        mapping = {}
        for row in selection:
            for name in groups[row["group"]]:
                mapping[name] = row["source"]
        return mapping

    variants: dict[str, dict] = {}
    eval_chunks_by_variant: dict[str, list[float] | None] = {}
    code_chunks_by_variant: dict[str, list[float] | None] = {}
    for label in sorted({args.low_source, args.target_source, ref, *high_sources}):
        entry = eval_gguf(label, source_paths[label], eval_text, "eval")
        eval_chunks_by_variant[label] = entry.get("chunk_nlls")
        variants[label] = {
            "nll": entry["nll"], "ppl": entry["ppl"], "tokens": entry["tokens"],
            "payload_bytes": payload_bytes(label),
            "payload_bpw": payload_bytes(label) * 8 / total_weights,
        }
    if args.code_text:
        for label in sorted({args.low_source, args.target_source}):
            code_row = eval_gguf(label, source_paths[label], args.code_text, "code",
                                 chunks=args.code_chunks)
            code_chunks_by_variant[label] = code_row.get("chunk_nlls")
            variants[label]["code_nll"] = code_row["nll"]
            variants[label]["code_ppl"] = code_row["ppl"]

    mixes = {args.variant_name: final_selection, **random_selections}
    for name, selection in mixes.items():
        mix_path = work_dir / f"{name}.gguf"
        mix_payload = assemble_mixed_gguf(mix_path, base_reader, tensors_by_source,
                                          selection_sources(selection), args.low_source)
        # mix payload accounting must match payload_bytes(): grouped tensors only
        mix_grouped = sum(
            int(tensors_by_source[selection_sources(selection).get(t.name, args.low_source)][t.name].n_bytes)
            for t in base_reader.tensors if t.name in grouped_names
        )
        result = run_perplexity(args.llama_bin, mix_path, eval_text, args.ctx, args.chunks, args.threads)
        eval_chunks_by_variant[name] = result.get("chunk_nlls")
        variants[name] = {
            "nll": result["nll"], "ppl": result["ppl"], "tokens": result["tokens"],
            "payload_bytes": mix_grouped,
            "payload_bpw": mix_grouped * 8 / total_weights,
            "file_bytes": mix_path.stat().st_size,
        }
        if args.code_text and name == args.variant_name:
            code_result = run_perplexity(args.llama_bin, mix_path, args.code_text, args.ctx,
                                         args.code_chunks or args.chunks, args.threads)
            code_chunks_by_variant[name] = code_result.get("chunk_nlls")
            variants[name]["code_nll"] = code_result["nll"]
            variants[name]["code_ppl"] = code_result["ppl"]
        mix_path.unlink()
        print(f"[final] {name}: NLL={result['nll']:.5f}"
              + (f" codeNLL={variants[name]['code_nll']:.5f}" if "code_nll" in variants[name] else "")
              + f" payload={mix_grouped/1e6:.1f} MB", flush=True)

    candidate_nll = variants[args.variant_name]["nll"]
    target_nll = variants[args.target_source]["nll"]
    random_by_nll = sorted(((variants[name]["nll"], name) for name in random_selections))
    random_nll, random_median_name = random_by_nll[len(random_by_nll) // 2]
    beats_target = candidate_nll < target_nll
    beats_random = candidate_nll <= random_nll
    if beats_target and beats_random:
        verdict = "GO"
    elif beats_target or beats_random:
        verdict = "GRAY"
    else:
        verdict = "NO-GO"

    # Paired per-chunk deltas for the two verdict legs (positive = mix better);
    # point estimates alone can't distinguish a real edge from 12k-token noise.
    verdict_stats = {}
    cand_chunks = eval_chunks_by_variant.get(args.variant_name)
    for leg, other in [("vs_target", args.target_source), ("vs_random_median", random_median_name)]:
        other_chunks = eval_chunks_by_variant.get(other)
        if cand_chunks and other_chunks:
            mean, se = paired_chunk_stats(other_chunks, cand_chunks)
            verdict_stats[leg] = {"delta": mean, "se": se}

    decision_bits = [f"cpu prober mix NLL {candidate_nll:.5f} vs target {target_nll:.5f} "
                     f"(random control median-of-3 {random_nll:.5f})"]
    if "vs_target" in verdict_stats:
        vt = verdict_stats["vs_target"]
        decision_bits.append(f"target leg paired dNLL {vt['delta']:+.5f}±{vt['se']:.5f}")
    code_guardrail = None
    if args.code_text:
        code_nlls = {label: entry.get("code_nll") for label, entry in variants.items()}
        verdict, code_guardrail = apply_code_guardrail(
            verdict, code_nlls, args.variant_name, args.target_source, args.code_no_regress,
            code_chunk_nlls=code_chunks_by_variant)
        if code_guardrail.get("regression_vs_target") is not None:
            state = {True: "pass", False: "FAIL", None: "recorded"}[code_guardrail["pass"]]
            se_bit = (f"±{code_guardrail['regression_se']:.5f}"
                      if "regression_se" in code_guardrail else "")
            decision_bits.append(
                f"code guardrail {state}: regression {code_guardrail['regression_vs_target']:+.5f}{se_bit} "
                f"(eps {args.code_no_regress:g})")

    all_rows = empirical + proxy_zeroed
    result_doc = {
        "created_utc": datetime.now(UTC).isoformat(),
        "prober": "cpu_llama_cpp",
        "verdict": verdict,
        "status": verdict,
        "decision_text": f"{verdict}: " + "; ".join(decision_bits),
        "code_guardrail": code_guardrail,
        "args": {
            "low_source": args.low_source,
            "target_source": args.target_source,
            "high_sources": high_sources,
            "ref_source": ref,
            "layers": layers,
            "group_mode": args.group_mode,
            "tensor_profile": args.tensor_profile,
            "candidate_variant": args.variant_name,
            "calib_text": str(args.calib_text),
            "eval_text": str(eval_text),
            "code_text": str(args.code_text) if args.code_text else None,
            "code_no_regress": args.code_no_regress,
            "code_chunks": args.code_chunks or args.chunks,
            "tier2_time_budget_min": args.tier2_time_budget_min,
            "cache_sig": base_sig,
            "ctx": args.ctx, "chunks": args.chunks,
            "probe_fraction": args.probe_fraction,
            "boundary_band": args.boundary_band,
            "max_probes": args.max_probes,
            "seed": args.seed,
            "source": [f"{label}={path}" for label, path in source_paths.items()],
        },
        "total_weight_count": int(total_weights),
        "budget_extra_bytes": int(budget_extra),
        "verdict_stats": verdict_stats,
        "random_control_median": random_median_name,
        "backfill_count": sum(1 for row in final_selection if row.get("backfill_proxy")),
        "backfill_extra_bytes": sum(row["extra_bytes"] for row in final_selection if row.get("backfill_proxy")),
        "selection_base_sources": {name: args.low_source for name in mixes},
        "selections": {name: selection for name, selection in mixes.items()},
        "allocation_rows": all_rows,
        "tier2_probe_count": len(done),
        "tier1_candidate_count": len(candidates),
        "variants": variants,
        "source_payload_bytes": {label: payload_bytes(label) for label in source_paths},
    }
    (out_dir / "result.json").write_text(json.dumps(result_doc, indent=2))
    print(f"[final] verdict={verdict}; wrote {out_dir / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
