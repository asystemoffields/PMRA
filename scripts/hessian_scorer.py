"""Hessian-sketch scorer: analytic PMRA candidate scoring, zero per-candidate forwards.

One forward+backward pass over the calibration tokens captures, per linear
layer: the full input covariance A = sum_t x_t x_t^T (the imatrix is exactly
diag(A)), the output-gradient second moment diag G = sum_t delta_t^2, and the
weight gradient sum_t delta_t x_t^T (free in p.grad). Every (group, source)
candidate is then scored analytically with a K-FAC-style local expansion of
NLL around the fp16 weights:

    dNLL(dW) ~= <grad_W NLL, dW>  +  1/2 * sum_i G_ii (dW A dW^T)_ii

where dW = dequant(source) - W_fp16. The first term is signed and exactly
additive across groups; the second is the curvature-weighted local error
(diagonal in the output dimension only — rows of W are independent given x).

Outputs:
  - hessian_scores.json in the cpu_prober tier1_scores.json format
    ({group: {source: predicted_dNLL_vs_fp16}}), so the prober consumes it as
    a drop-in Tier-1 cache: improvement = score(low) - score(high).
  - optional validation against an empirical allocation_rows.jsonl: Spearman
    and knapsack-regret ladder (bytes-only < imatrix-SSE < quad < first+quad).

Runs on CPU. 135M: ~10 min capture; scoring all candidates: seconds.

Memory at 4B+: the down_proj input covariance is (d_ffn, d_ffn) fp32 —
~380 MB/layer at d_ffn≈10k — so holding every layer's A at once plus fp32
grads for the whole model blows past a 30 GB Kaggle CPU kernel. Use
--capture-passes N to split layers into N ranges: each pass freezes all
weights except the in-range targets (p.grad allocated only for those), runs
the same calibration chunks, scores the in-range groups, then frees the
covariances and grads before the next pass. --families mlp skips the
attention sites entirely (on hybrid DeltaNet models the linear-attention
layers don't match self_attn.* names anyway).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader

from cpu_prober import dequant, select_knapsack

# llama-family HF module -> GGUF tensor name tails
HF_TO_GGUF = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}
# modules sharing an input site share one covariance accumulator
INPUT_SITE = {
    "self_attn.q_proj": "attn_in", "self_attn.k_proj": "attn_in", "self_attn.v_proj": "attn_in",
    "self_attn.o_proj": "o_in",
    "mlp.gate_proj": "ffn_in", "mlp.up_proj": "ffn_in",
    "mlp.down_proj": "down_in",
}
FAMILY = {"attn_q": "attn", "attn_k": "attn", "attn_v": "attn", "attn_output": "attn",
          "ffn_gate": "mlp", "ffn_up": "mlp", "ffn_down": "mlp"}


def find_targets(model, families: set[str]) -> dict[str, torch.nn.Module]:
    """Map "layer_idx|hf_tail" -> linear module for the requested families."""
    targets: dict[str, torch.nn.Module] = {}
    for name, module in model.named_modules():
        parts = name.split(".")
        if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
            tail = ".".join(parts[3:])
            if tail in HF_TO_GGUF and isinstance(module, torch.nn.Linear):
                if FAMILY[HF_TO_GGUF[tail]] in families:
                    targets[f"{parts[2]}|{tail}"] = module
    return targets


def capture_pass(model, targets: dict[str, torch.nn.Module], ids: torch.Tensor, ctx: int):
    """Forward+backward over the calibration chunks for one subset of targets.

    Only the target weights have requires_grad; G is captured with hooks on
    the module OUTPUT tensors (a full_backward_hook would never fire on the
    lowest unfrozen layer, whose input needs no grad).
    """
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    for module in targets.values():
        module.weight.requires_grad_(True)

    cov: dict[str, torch.Tensor] = {}      # site key -> A (d_in, d_in) fp32
    gdiag: dict[str, torch.Tensor] = {}    # target key -> G (d_out,) fp32
    hooks = []
    site_owner = {}                        # site key -> target key that accumulates it
    for key in targets:
        layer, tail = key.split("|")
        site_owner.setdefault(f"{layer}|{INPUT_SITE[tail]}", key)

    def fwd_hook(key, site_key):
        owns_site = site_owner[site_key] == key

        def hook(module, inputs, output):
            if owns_site:
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
                if site_key not in cov:
                    cov[site_key] = torch.zeros(x.shape[1], x.shape[1])
                cov[site_key] += x.T @ x

            def grad_hook(g):
                gf = g.detach().reshape(-1, g.shape[-1]).float()
                if key not in gdiag:
                    gdiag[key] = torch.zeros(gf.shape[1])
                gdiag[key] += (gf * gf).sum(dim=0)

            output.register_hook(grad_hook)
        return hook

    for key, module in targets.items():
        layer, tail = key.split("|")
        hooks.append(module.register_forward_hook(fwd_hook(key, f"{layer}|{INPUT_SITE[tail]}")))

    n_tokens = 0
    n_chunks = len(ids) // ctx
    for c in range(n_chunks):
        window = ids[c * ctx : (c + 1) * ctx].unsqueeze(0)
        out = model(window)
        logits = out.logits[0, :-1]
        labels = window[0, 1:]
        loss = torch.nn.functional.cross_entropy(logits.float(), labels, reduction="sum")
        loss.backward()  # p.grad accumulates across chunks (never zeroed)
        n_tokens += labels.numel()
        print(f"[capture] chunk {c + 1}/{n_chunks}  sum-NLL={loss.item():.1f}", flush=True)
    for h in hooks:
        h.remove()
    return cov, gdiag, n_tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="HF model dir (fp16/fp32 reference).")
    parser.add_argument("--source", action="append", default=[], help="GGUF source as label=path (repeat).")
    parser.add_argument("--low-source", required=True)
    parser.add_argument("--high-sources", required=True)
    parser.add_argument("--calib-text", type=Path, required=True)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=24)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--group-mode", default="layer_family", choices=["layer_family", "tensor"])
    parser.add_argument("--families", default="attn,mlp",
                        help="Comma list of tensor families to score (attn,mlp).")
    parser.add_argument("--capture-passes", type=int, default=1,
                        help="Split layers into N capture passes to bound covariance+grad RAM (4B+: use 4).")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                        help="Model dtype; accumulators stay fp32 either way.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the tier1-compatible scores JSON.")
    parser.add_argument("--validate-rows", type=Path, default=None, help="Empirical allocation_rows.jsonl for the regret ladder.")
    parser.add_argument("--budget-bytes", type=int, default=None, help="Knapsack budget for regret validation.")
    args = parser.parse_args()

    sources = {}
    for item in args.source:
        label, _, path = item.partition("=")
        sources[label.strip()] = {t.name: t for t in GGUFReader(path.strip()).tensors}
    high_sources = [s.strip() for s in args.high_sources.split(",") if s.strip()]
    families = {f.strip() for f in args.families.split(",") if f.strip()}

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=getattr(torch, args.dtype)).eval()
    all_targets = find_targets(model, families)
    if not all_targets:
        raise RuntimeError(f"no target linears matched families={sorted(families)}; "
                           "module naming may differ for this architecture")
    layers = sorted({int(k.split("|")[0]) for k in all_targets})
    ranges = np.array_split(np.array(layers), max(1, args.capture_passes))
    ids = tokenizer(args.calib_text.read_text(encoding="utf-8"),
                    return_tensors="pt").input_ids[0][: args.ctx * args.chunks]

    scores: dict[str, dict[str, float]] = defaultdict(dict)
    detail: dict[str, dict] = {}
    n_tokens = 0
    for i, layer_range in enumerate(r for r in ranges if len(r)):
        in_range = set(int(x) for x in layer_range)
        targets = {k: m for k, m in all_targets.items() if int(k.split("|")[0]) in in_range}
        print(f"[capture] pass {i + 1}/{len(ranges)}: layers {min(in_range)}-{max(in_range)} "
              f"({len(targets)} linears)", flush=True)
        cov, gdiag, n_tokens = capture_pass(model, targets, ids, args.ctx)

        def tensor_score(key: str, gguf_name: str, source_label: str):
            tensor = sources[source_label].get(gguf_name)
            if tensor is None:
                return None
            layer, tail = key.split("|")
            module = targets[key]
            dw = (torch.from_numpy(np.ascontiguousarray(dequant(tensor))).float()
                  - module.weight.detach().float())
            a = cov[f"{layer}|{INPUT_SITE[tail]}"]
            first = float((module.weight.grad.float() * dw).sum())
            quad = float((gdiag[key] * ((dw @ a) * dw).sum(dim=1)).sum())
            return first + 0.5 * quad, first, quad

        for key in targets:
            layer, tail = key.split("|")
            gguf_tail = HF_TO_GGUF[tail]
            gguf_name = f"blk.{layer}.{gguf_tail}.weight"
            group = (f"L{layer}:{FAMILY[gguf_tail]}" if args.group_mode == "layer_family"
                     else f"L{layer}:{gguf_tail}")
            for label in [args.low_source, *high_sources]:
                result = tensor_score(key, gguf_name, label)
                if result is None:
                    continue
                total, first, quad = result
                scores[group][label] = scores[group].get(label, 0.0) + total
                d = detail.setdefault(f"{group}|{label}", {"first": 0.0, "quad": 0.0})
                d["first"] += first
                d["quad"] += quad

        # free this pass's accumulators before the next range
        cov.clear()
        gdiag.clear()
        for module in targets.values():
            module.weight.grad = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({g: dict(s) for g, s in scores.items()}))
    print(f"[score] wrote {args.output} ({len(scores)} groups; predicted dNLL vs fp16, sum over {n_tokens} tokens)", flush=True)

    if args.validate_rows:
        rows = [json.loads(l) for l in args.validate_rows.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r["group"] in scores and r["source"] in scores[r["group"]]
                and args.__dict__["low_source"] in scores[r["group"]]]
        measured = np.array([r["calib_nll_improvement"] for r in rows])
        preds = {
            "bytes_only": np.array([r["extra_bytes"] for r in rows], dtype=float),
            "imatrix_sse": np.array([r.get("weight_sse_delta", 0.0) for r in rows]),
            "hess_quad": np.array([
                (detail[f"{r['group']}|{args.low_source}"]["quad"] - detail[f"{r['group']}|{r['source']}"]["quad"]) / 2
                for r in rows]),
            "hess_first_plus_quad": np.array([
                scores[r["group"]][args.low_source] - scores[r["group"]][r["source"]] for r in rows]),
        }

        def spearman(a, b):
            ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
            return float(np.corrcoef(ra, rb)[0, 1])

        budget = args.budget_bytes or int(sum(sorted((r["extra_bytes"] for r in rows), reverse=True)[: len(rows) // 3]))
        oracle = select_knapsack([dict(r) for r in rows], budget, "calib_nll_improvement")
        oracle_val = sum(r["calib_nll_improvement"] for r in oracle)
        print(f"\n[validate] n={len(rows)} empirical rows; knapsack budget {budget/1e6:.2f} MB; "
              f"oracle value {oracle_val:.5f}")
        for name, p in preds.items():
            rho = spearman(measured, p)
            test_rows = [{**r, "_pred": float(v)} for r, v in zip(rows, p)]
            sel = select_knapsack(test_rows, budget, "_pred")
            val = sum(r["calib_nll_improvement"] for r in sel)
            print(f"[validate] {name:22s} Spearman={rho:+.3f}  knapsack-value={val:.5f} "
                  f"({val/oracle_val*100 if oracle_val else 0:5.1f}% of oracle, {len(sel)} groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
