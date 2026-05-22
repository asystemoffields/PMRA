from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFReader, dequantize
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_conditioned_scale_mirage import (
    DEFAULT_HF,
    DEFAULT_IQ4,
    DEFAULT_MODEL_DIR,
    align_shape,
    build_prompts,
    collect_activations,
    load_hf_tensor,
    make_jobs,
    mse,
    parse_layers,
)
try:
    from additive_residual_codebook_gate import blocks_from_matrix, fit_multistage_codebooks, random_multistage_codebooks
    from mlp_up_codebook_composition_gate import activation_weights_by_position
    from mlp_joint_codebook_composition_gate import (
        apply_payloads,
        down_inputs,
        fit_projection_payload,
        mlp_output,
        parse_targets,
        payload_bits,
        static_alpha_mlp_base,
    )
except ModuleNotFoundError:
    blocks_from_matrix = None
    fit_multistage_codebooks = None
    random_multistage_codebooks = None
    activation_weights_by_position = None
    apply_payloads = None
    down_inputs = None
    fit_projection_payload = None
    mlp_output = None
    parse_targets = None
    payload_bits = None
    static_alpha_mlp_base = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


def parse_variants(text: str) -> list[str]:
    allowed = {
        "fp16",
        "static_int3_mlp",
        "joint_codebook_mlp",
        "iq4_mlp",
        "iq4_all",
        "static_int3_mlp_iq4_rest",
        "joint_codebook_mlp_iq4_rest",
        "prod_residual_mlp_iq4_rest",
        "prod_residual_random_mlp_iq4_rest",
    }
    variants = [part.strip() for part in text.split(",") if part.strip()]
    bad = sorted(set(variants) - allowed)
    if bad:
        raise ValueError(f"unknown variants: {bad}")
    if "fp16" not in variants:
        variants.insert(0, "fp16")
    return variants


def set_weight(module: torch.nn.Module, name: str, array: np.ndarray) -> None:
    weight = getattr(module, name).weight
    tensor = torch.from_numpy(np.array(array, copy=True, order="C")).to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        weight.copy_(tensor)


def copy_array_to_parameter(param: torch.nn.Parameter, array: np.ndarray) -> None:
    tensor = torch.from_numpy(np.array(array, copy=True, order="C")).to(device=param.device, dtype=param.dtype)
    with torch.no_grad():
        param.copy_(tensor)


def set_lm_head_weight(model, array: np.ndarray) -> None:
    tensor = torch.from_numpy(np.array(array, copy=True, order="C")).to(device=model.lm_head.weight.device, dtype=model.lm_head.weight.dtype)
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(tensor.clone())
    else:
        with torch.no_grad():
            model.lm_head.weight.copy_(tensor)


def load_gguf_tensor_any(gguf_tensors: dict, name: str) -> np.ndarray:
    tensor = gguf_tensors[name]
    if tensor.tensor_type == GGMLQuantizationType.F32:
        return tensor.data.astype(np.float32, copy=False)
    return dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False)


def patch_layer(model, layer: int, weights: dict[str, np.ndarray]) -> None:
    mlp = model.model.layers[layer].mlp
    set_weight(mlp, "gate_proj", weights["gate"])
    set_weight(mlp, "up_proj", weights["up"])
    set_weight(mlp, "down_proj", weights["down"])


def patch_non_mlp(model, hf, gguf_tensors: dict, layers: list[int], mode: str) -> None:
    if mode not in {"fp16", "iq4"}:
        raise ValueError(f"unknown non-MLP patch mode {mode}")

    def hf_weight(name: str) -> np.ndarray:
        return load_hf_tensor(hf, name)

    def gguf_weight(hf_ref: np.ndarray, name: str) -> np.ndarray:
        return align_shape(hf_ref, load_gguf_tensor_any(gguf_tensors, name))

    if mode == "fp16":
        embed = hf_weight("model.embed_tokens.weight")
        copy_array_to_parameter(model.model.embed_tokens.weight, embed)
        if hasattr(model, "lm_head"):
            if "lm_head.weight" in hf.keys():
                set_lm_head_weight(model, hf_weight("lm_head.weight"))
            else:
                set_lm_head_weight(model, embed)
        copy_array_to_parameter(model.model.norm.weight, hf_weight("model.norm.weight"))
    else:
        embed_ref = hf_weight("model.embed_tokens.weight")
        copy_array_to_parameter(model.model.embed_tokens.weight, gguf_weight(embed_ref, "token_embd.weight"))
        if hasattr(model, "lm_head"):
            out_ref = model.lm_head.weight.detach().to(torch.float32).cpu().numpy()
            set_lm_head_weight(model, gguf_weight(out_ref, "output.weight"))
        norm_ref = hf_weight("model.norm.weight")
        copy_array_to_parameter(model.model.norm.weight, gguf_weight(norm_ref, "output_norm.weight"))

    for layer in layers:
        hf_prefix = f"model.layers.{layer}"
        gguf_prefix = f"blk.{layer}"
        layer_mod = model.model.layers[layer]
        attn = layer_mod.self_attn

        tensor_pairs = [
            (attn, "q_proj", f"{hf_prefix}.self_attn.q_proj.weight", f"{gguf_prefix}.attn_q.weight"),
            (attn, "k_proj", f"{hf_prefix}.self_attn.k_proj.weight", f"{gguf_prefix}.attn_k.weight"),
            (attn, "v_proj", f"{hf_prefix}.self_attn.v_proj.weight", f"{gguf_prefix}.attn_v.weight"),
            (attn, "o_proj", f"{hf_prefix}.self_attn.o_proj.weight", f"{gguf_prefix}.attn_output.weight"),
        ]
        for module, attr, hf_name, gguf_name in tensor_pairs:
            fp = hf_weight(hf_name)
            set_weight(module, attr, fp if mode == "fp16" else gguf_weight(fp, gguf_name))

        norm_pairs = [
            (layer_mod.input_layernorm.weight, f"{hf_prefix}.input_layernorm.weight", f"{gguf_prefix}.attn_norm.weight"),
            (
                layer_mod.post_attention_layernorm.weight,
                f"{hf_prefix}.post_attention_layernorm.weight",
                f"{gguf_prefix}.ffn_norm.weight",
            ),
            (attn.q_norm.weight, f"{hf_prefix}.self_attn.q_norm.weight", f"{gguf_prefix}.attn_q_norm.weight"),
            (attn.k_norm.weight, f"{hf_prefix}.self_attn.k_norm.weight", f"{gguf_prefix}.attn_k_norm.weight"),
        ]
        for param, hf_name, gguf_name in norm_pairs:
            fp = hf_weight(hf_name)
            copy_array_to_parameter(param, fp if mode == "fp16" else gguf_weight(fp, gguf_name))


def fit_projection_payload_quality_only(
    target: str,
    fp: np.ndarray,
    qbase: np.ndarray,
    fit_x: np.ndarray,
    args,
    rng,
) -> dict:
    rows, cols = fp.shape
    if cols % args.block_size != 0:
        raise ValueError(f"{target} cols {cols} not divisible by block size {args.block_size}")
    delta = (fp - qbase).astype(np.float32, copy=False)
    blocks = blocks_from_matrix(delta, args.block_size)

    print(f"[c1m] fitting {target} weight residual codebook", flush=True)
    weight_residual, weight_code_bits, weight_id_bits = fit_multistage_codebooks(
        blocks,
        args.block_size,
        rows,
        cols,
        args.codebook_size,
        args.stages,
        args.train_blocks,
        args.kmeans_iters,
        rng,
        weights_by_position=None,
    )

    print(f"[c1m] fitting {target} activation-weighted residual codebook", flush=True)
    act_residual, act_code_bits, act_id_bits = fit_multistage_codebooks(
        blocks,
        args.block_size,
        rows,
        cols,
        args.codebook_size,
        args.stages,
        args.train_blocks,
        args.kmeans_iters,
        rng,
        weights_by_position=activation_weights_by_position(fit_x, args.block_size),
    )

    return {
        "target": target,
        "shape": [int(rows), int(cols)],
        "weight_count": int(fp.size),
        "weight_residual": weight_residual,
        "weight_side_bits": int(weight_code_bits + weight_id_bits),
        "activation_residual": act_residual,
        "activation_side_bits": int(act_code_bits + act_id_bits),
    }


def layer_static_base(layer: int, hf, gguf_tensors: dict, x_calib: np.ndarray, x_eval: np.ndarray, args):
    prefix = f"model.layers.{layer}.mlp"
    blk = f"blk.{layer}"
    gate = load_hf_tensor(hf, f"{prefix}.gate_proj.weight")
    up = load_hf_tensor(hf, f"{prefix}.up_proj.weight")
    down = load_hf_tensor(hf, f"{prefix}.down_proj.weight")
    iq4_gate = align_shape(gate, load_gguf_tensor_any(gguf_tensors, f"{blk}.ffn_gate.weight"))
    iq4_up = align_shape(up, load_gguf_tensor_any(gguf_tensors, f"{blk}.ffn_up.weight"))
    iq4_down = align_shape(down, load_gguf_tensor_any(gguf_tensors, f"{blk}.ffn_down.weight"))

    x_calib = x_calib[: args.max_tokens]
    x_eval = x_eval[: args.max_tokens]
    fp_calib = mlp_output(x_calib, gate, up, down)
    fp_eval = mlp_output(x_eval, gate, up, down)
    base_rec, q_gate, q_up, q_down = static_alpha_mlp_base(x_calib, x_eval, fp_calib, fp_eval, gate, up, down)
    return {
        "fp": {"gate": gate, "up": up, "down": down},
        "iq4": {"gate": iq4_gate, "up": iq4_up, "down": iq4_down},
        "qbase": {"gate": q_gate, "up": q_up, "down": q_down},
        "base_rec": base_rec,
        "fp_calib": fp_calib,
        "fp_eval": fp_eval,
        "x_calib": x_calib,
        "x_eval": x_eval,
    }


def build_joint_codebook_weights(layer: int, base: dict, args, rng) -> tuple[dict[str, np.ndarray], dict]:
    fp = base["fp"]
    qbase = base["qbase"]
    x_calib = base["x_calib"]
    fp_calib = base["fp_calib"]
    fit_inputs = {
        "gate": x_calib,
        "up": x_calib,
        "down": down_inputs(x_calib, fp["gate"], fp["up"]),
    }

    payloads = {}
    for target in args.codebook_targets:
        if args.fit_payload_controls:
            payloads[target] = fit_projection_payload(target, fp[target], qbase[target], fit_inputs[target], args, rng)
        else:
            payloads[target] = fit_projection_payload_quality_only(
                target,
                fp[target],
                qbase[target],
                fit_inputs[target],
                args,
                rng,
            )

    weight_weights = apply_payloads(qbase, payloads, "weight")
    activation_weights = apply_payloads(qbase, payloads, "activation")
    weight_calib_mse = mse(fp_calib, mlp_output(x_calib, weight_weights["gate"], weight_weights["up"], weight_weights["down"]))
    activation_calib_mse = mse(
        fp_calib,
        mlp_output(x_calib, activation_weights["gate"], activation_weights["up"], activation_weights["down"]),
    )
    if activation_calib_mse <= weight_calib_mse:
        chosen = activation_weights
        chosen_name = "activation_weighted"
        side_bits = payload_bits(payloads, "activation")
        chosen_calib_mse = activation_calib_mse
    else:
        chosen = weight_weights
        chosen_name = "weight_mse"
        side_bits = payload_bits(payloads, "weight")
        chosen_calib_mse = weight_calib_mse

    total_weights = sum(arr.size for arr in fp.values())
    return chosen, {
        "layer": int(layer),
        "chosen": chosen_name,
        "base_alpha": base["base_rec"]["alpha"],
        "base_bits": int(base["base_rec"]["bits"]),
        "side_bits": int(side_bits),
        "candidate_mlp_bpw": float((base["base_rec"]["bits"] + side_bits) / total_weights),
        "calib_mlp_mse": float(chosen_calib_mse),
    }


def fit_projection_payload_random(
    target: str,
    fp: np.ndarray,
    qbase: np.ndarray,
    args,
    rng,
) -> dict:
    rows, cols = fp.shape
    if cols % args.block_size != 0:
        raise ValueError(f"{target} cols {cols} not divisible by block size {args.block_size}")
    delta = (fp - qbase).astype(np.float32, copy=False)
    blocks = blocks_from_matrix(delta, args.block_size)
    print(f"[c1m] fitting {target} random residual codebook control", flush=True)
    residual, code_bits, id_bits = random_multistage_codebooks(
        blocks,
        args.block_size,
        rows,
        cols,
        args.codebook_size,
        args.stages,
        rng,
    )
    return {
        "target": target,
        "shape": [int(rows), int(cols)],
        "weight_count": int(fp.size),
        "weight_residual": residual,
        "weight_side_bits": int(code_bits + id_bits),
        "activation_residual": residual,
        "activation_side_bits": int(code_bits + id_bits),
    }


def build_residual_codebook_weights(
    layer: int,
    base: dict,
    qbase_key: str,
    args,
    rng,
    random_control: bool = False,
) -> tuple[dict[str, np.ndarray], dict]:
    fp = base["fp"]
    qbase = base[qbase_key]
    x_calib = base["x_calib"]
    fp_calib = base["fp_calib"]
    fit_inputs = {
        "gate": x_calib,
        "up": x_calib,
        "down": down_inputs(x_calib, fp["gate"], fp["up"]),
    }

    payloads = {}
    for target in args.codebook_targets:
        if random_control:
            payloads[target] = fit_projection_payload_random(target, fp[target], qbase[target], args, rng)
        elif args.fit_payload_controls:
            payloads[target] = fit_projection_payload(target, fp[target], qbase[target], fit_inputs[target], args, rng)
        else:
            payloads[target] = fit_projection_payload_quality_only(
                target,
                fp[target],
                qbase[target],
                fit_inputs[target],
                args,
                rng,
            )

    weight_weights = apply_payloads(qbase, payloads, "weight")
    activation_weights = apply_payloads(qbase, payloads, "activation")
    weight_calib_mse = mse(fp_calib, mlp_output(x_calib, weight_weights["gate"], weight_weights["up"], weight_weights["down"]))
    activation_calib_mse = mse(
        fp_calib,
        mlp_output(x_calib, activation_weights["gate"], activation_weights["up"], activation_weights["down"]),
    )
    if activation_calib_mse <= weight_calib_mse:
        chosen = activation_weights
        chosen_name = "activation_weighted"
        side_bits = payload_bits(payloads, "activation")
        chosen_calib_mse = activation_calib_mse
    else:
        chosen = weight_weights
        chosen_name = "weight_mse"
        side_bits = payload_bits(payloads, "weight")
        chosen_calib_mse = weight_calib_mse

    total_weights = sum(arr.size for arr in fp.values())
    return chosen, {
        "layer": int(layer),
        "chosen": "random_control" if random_control else chosen_name,
        "base": qbase_key,
        "side_bits": int(side_bits),
        "side_mlp_bpw": float(side_bits / total_weights),
        "candidate_mlp_bpw": float(side_bits / total_weights),
        "calib_mlp_mse": float(chosen_calib_mse),
    }


def apply_variant(model, variant: str, layers: list[int], hf, gguf_tensors: dict, activations: dict, args, rng) -> list[dict]:
    summaries = []
    rest_mode = (
        "iq4"
        if variant
        in {
            "iq4_all",
            "static_int3_mlp_iq4_rest",
            "joint_codebook_mlp_iq4_rest",
            "prod_residual_mlp_iq4_rest",
            "prod_residual_random_mlp_iq4_rest",
        }
        else "fp16"
    )
    patch_non_mlp(model, hf, gguf_tensors, layers, rest_mode)
    for layer in layers:
        key = f"L{layer}:ffn_gate"
        base = layer_static_base(layer, hf, gguf_tensors, activations["calib"][key], activations["eval"][key], args)
        if variant == "fp16":
            weights = base["fp"]
            summary = {"layer": int(layer), "variant": variant, "candidate_mlp_bpw": 16.0}
        elif variant in {"static_int3_mlp", "static_int3_mlp_iq4_rest"}:
            weights = base["qbase"]
            total_weights = sum(arr.size for arr in base["fp"].values())
            summary = {
                "layer": int(layer),
                "variant": variant,
                "base_alpha": base["base_rec"]["alpha"],
                "candidate_mlp_bpw": float(base["base_rec"]["bits"] / total_weights),
            }
        elif variant in {"iq4_mlp", "iq4_all"}:
            weights = base["iq4"]
            summary = {"layer": int(layer), "variant": variant, "candidate_mlp_bpw": None}
        elif variant in {"joint_codebook_mlp", "joint_codebook_mlp_iq4_rest"}:
            weights, summary = build_joint_codebook_weights(layer, base, args, rng)
            summary["variant"] = variant
        elif variant == "prod_residual_mlp_iq4_rest":
            weights, summary = build_residual_codebook_weights(layer, base, "iq4", args, rng, random_control=False)
            summary["variant"] = variant
        elif variant == "prod_residual_random_mlp_iq4_rest":
            weights, summary = build_residual_codebook_weights(layer, base, "iq4", args, rng, random_control=True)
            summary["variant"] = variant
        else:
            raise ValueError(f"unhandled variant {variant}")
        patch_layer(model, layer, weights)
        summaries.append(summary)
    return summaries


def evaluate_model(model, tokenizer, prompts: list[str], args, fp_last_logits: list[np.ndarray] | None = None) -> dict:
    total_nll = 0.0
    total_tokens = 0
    logit_mses = []
    top10_overlaps = []
    captured_logits = []
    device = next(model.parameters()).device

    with torch.inference_mode():
        for idx, prompt in enumerate(prompts):
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.eval_max_length)
            ids = encoded.input_ids.to(device)
            if ids.shape[-1] < 2:
                continue
            out = model(input_ids=ids, labels=ids, use_cache=False)
            count = int(ids.shape[-1] - 1)
            total_nll += float(out.loss.detach().cpu()) * count
            total_tokens += count
            last = out.logits[:, -1, :].detach().to(torch.float32).cpu().numpy().reshape(-1)
            if fp_last_logits is None:
                captured_logits.append(last)
            else:
                ref = fp_last_logits[idx]
                logit_mses.append(float(np.mean((last - ref) ** 2)))
                top_ref = set(np.argpartition(ref, -10)[-10:].tolist())
                top_cur = set(np.argpartition(last, -10)[-10:].tolist())
                top10_overlaps.append(len(top_ref & top_cur) / 10.0)

    result = {
        "tokens": int(total_tokens),
        "nll": float(total_nll / max(1, total_tokens)),
        "ppl": float(np.exp(total_nll / max(1, total_tokens))),
    }
    if fp_last_logits is None:
        result["captured_last_logits"] = captured_logits
    else:
        result["last_logit_mse_to_fp16"] = float(np.mean(logit_mses)) if logit_mses else None
        result["top10_overlap_to_fp16"] = float(np.mean(top10_overlaps)) if top10_overlaps else None
    return result


def strip_logits(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "captured_last_logits"}


def make_markdown(result: dict) -> str:
    lines = [
        "# Result Card - Model-Forward MLP Replacement Gate",
        "",
        "## Status",
        "",
        result["verdict"],
        "",
        "## Decisive Measurement",
        "",
        "The actual Qwen3-1.7B model was forward-evaluated after replacing every MLP block with each tested representation. Depending on the variant, attention, embeddings, norms, and lm_head are either FP16/BF16 or patched from the selected GGUF baseline.",
        "",
        "## Variants",
        "",
        "| Variant | NLL | PPL | Delta NLL vs FP16 | Last-Logit MSE vs FP16 | Top-10 Overlap | Mean MLP bpw |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    fp_nll = result["variants"]["fp16"]["nll"]
    for name, row in result["variants"].items():
        delta = row["nll"] - fp_nll
        bpw = row.get("mean_mlp_bpw")
        logit_mse = row.get("last_logit_mse_to_fp16")
        overlap = row.get("top10_overlap_to_fp16")
        lines.append(
            f"| {name} | {row['nll']:.6f} | {row['ppl']:.6f} | {delta:.6f} | "
            f"{logit_mse:.6g}" if logit_mse is not None else f"| {name} | {row['nll']:.6f} | {row['ppl']:.6f} | {delta:.6f} | n/a"
        )
        suffix = f" | {overlap:.3f}" if overlap is not None else " | n/a"
        suffix += f" | {bpw:.6f} |" if bpw is not None else " | n/a |"
        lines[-1] += suffix
    lines.extend(
        [
            "",
            "## GO / NO-GO",
            "",
            result["decision_text"],
            "",
            "## Next Step",
            "",
            result["next_step"],
            "",
        ]
    )
    return "\n".join(lines)


def decide(result: dict) -> tuple[str, str, str]:
    variants = result["variants"]
    if "prod_residual_mlp_iq4_rest" in variants and "iq4_all" in variants:
        base = variants["iq4_all"]
        cand = variants["prod_residual_mlp_iq4_rest"]
        random_control = variants.get("prod_residual_random_mlp_iq4_rest")
        improvement = base["nll"] - cand["nll"]
        cand["nll_delta_vs_production_base"] = float(cand["nll"] - base["nll"])
        cand["nll_improvement_vs_production_base"] = float(improvement)
        if random_control is not None:
            cand["nll_improvement_vs_random_control"] = float(random_control["nll"] - cand["nll"])
        if cand.get("last_logit_mse_to_fp16") is not None and base.get("last_logit_mse_to_fp16"):
            cand["logit_mse_ratio_vs_production_base"] = float(cand["last_logit_mse_to_fp16"] / base["last_logit_mse_to_fp16"])
        if improvement >= 0.03 and cand.get("top10_overlap_to_fp16", 0.0) >= base.get("top10_overlap_to_fp16", 0.0):
            if random_control is not None and random_control["nll"] - cand["nll"] < 0.01:
                return (
                    "GRAY",
                    "GRAY: production-base residual overlay improves the source baseline, but the random side-payload control is too close.",
                    "Run stronger same-byte controls before promotion.",
                )
            return (
                "GO",
                "GO: production-base residual overlay improves recursive model-forward quality over the selected production GGUF baseline.",
                "Escalate to three seeds, exact side-payload artifact accounting, and same-byte low-rank/RVQ controls.",
            )
        if improvement <= 0.0:
            return (
                "NO-GO",
                "NO-GO: production-base residual overlay does not improve over the selected production GGUF baseline.",
                "Kill this side-payload point or redesign the objective/format before spending more Modal time.",
            )
        return (
            "GRAY",
            "GRAY: production-base residual overlay improves the source baseline, but below the predeclared NLL threshold.",
            "Run only if a larger eval or lower-byte operating point is strategically useful.",
        )

    cand_key = "joint_codebook_mlp" if "joint_codebook_mlp" in variants else "joint_codebook_mlp_iq4_rest"
    static_key = "static_int3_mlp" if "static_int3_mlp" in variants else "static_int3_mlp_iq4_rest"
    if cand_key not in variants or static_key not in variants:
        return "GRAY", "GRAY: missing direct static-int3 or candidate comparison.", "Run with static and candidate variants."
    fp = variants["fp16"]
    static = variants[static_key]
    cand = variants[cand_key]
    iq4 = variants.get("iq4_all") or variants.get("iq4_mlp")
    static_delta = static["nll"] - fp["nll"]
    cand_delta = cand["nll"] - fp["nll"]
    if static_delta <= 0:
        return (
            "GRAY",
            "GRAY: static-int3 MLP replacement did not degrade NLL on this tiny eval, so NLL is not discriminating enough.",
            "Increase eval set and inspect logit drift.",
        )
    nll_gap_closed = (static_delta - cand_delta) / static_delta
    cand["nll_gap_closed_vs_static"] = float(nll_gap_closed)
    if iq4 is not None and static["nll"] > iq4["nll"]:
        cand["nll_gap_closed_to_iq4"] = float((static["nll"] - cand["nll"]) / (static["nll"] - iq4["nll"]))
    if "iq4_all" in variants and cand_key == "joint_codebook_mlp_iq4_rest":
        iq4_all = variants["iq4_all"]
        cand["nll_delta_vs_iq4_all"] = float(cand["nll"] - iq4_all["nll"])
        cand["logit_mse_ratio_vs_iq4_all"] = (
            float(cand["last_logit_mse_to_fp16"] / iq4_all["last_logit_mse_to_fp16"])
            if cand.get("last_logit_mse_to_fp16") is not None and iq4_all.get("last_logit_mse_to_fp16")
            else None
        )
    if nll_gap_closed >= 0.25 and cand_delta < static_delta and cand.get("top10_overlap_to_fp16", 0.0) >= static.get(
        "top10_overlap_to_fp16", 0.0
    ):
        if cand_key == "joint_codebook_mlp_iq4_rest" and "iq4_all" in variants:
            return (
                "GO",
                "GO: mixed-rate joint MLP codebooks improve over static-int3 MLPs with IQ4 non-MLP weights; compare delta to full IQ4 for frontier placement.",
                "Escalate to larger eval, encoded artifact byte accounting, and production baseline bakeoff.",
            )
        return (
            "GO",
            "GO: joint MLP codebooks improve recursive model-forward behavior over static-int3 MLP replacement.",
            "Escalate to a larger eval set and save a reusable encoded artifact for loader-level testing.",
        )
    if nll_gap_closed < 0.05:
        return (
            "NO-GO",
            "NO-GO: local MLP improvements do not transfer to this recursive model-forward metric.",
            "Park this static format or redesign with propagation-aware calibration.",
        )
    return (
        "GRAY",
        "GRAY: candidate improves recursive model-forward behavior, but not enough for a GO on this eval.",
        "Run a larger eval or propagation-aware calibration before promotion.",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hf", default=DEFAULT_HF)
    parser.add_argument("--iq4", default=DEFAULT_IQ4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27")
    parser.add_argument("--variants", default="fp16,static_int3_mlp,joint_codebook_mlp,iq4_mlp")
    parser.add_argument("--codebook-targets", default="gate,up,down")
    parser.add_argument("--calib-prompts", type=int, default=96)
    parser.add_argument("--heldout-prompts", type=int, default=96)
    parser.add_argument("--eval-prompts", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--eval-max-length", type=int, default=96)
    parser.add_argument("--max-activation-tokens", type=int, default=192)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=128)
    parser.add_argument("--stages", type=int, default=1)
    parser.add_argument("--train-blocks", type=int, default=50000)
    parser.add_argument("--kmeans-iters", type=int, default=12)
    parser.add_argument("--factor-bits", type=int, default=8)
    parser.add_argument(
        "--fit-payload-controls",
        action="store_true",
        help="Fit low-rank/random payload controls while building the candidate. Off by default for model-forward sprint speed because C1L already ran those controls.",
    )
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.layers = parse_layers(args.layers)
    args.variants = parse_variants(args.variants)
    args.codebook_targets = parse_targets(args.codebook_targets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("[c1m] loading tokenizer/model", flush=True)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    jobs = make_jobs(args.layers, ["ffn_gate"])
    prompts = build_prompts(tokenizer, args.calib_prompts + args.heldout_prompts, args.seed)
    print("[c1m] capturing calibration MLP inputs", flush=True)
    x_calib = collect_activations(
        model,
        tokenizer,
        jobs,
        prompts[: args.calib_prompts],
        args.max_length,
        args.max_activation_tokens,
        device,
    )
    print("[c1m] capturing held-out MLP inputs for base selection", flush=True)
    x_eval = collect_activations(
        model,
        tokenizer,
        jobs,
        prompts[args.calib_prompts :],
        args.max_length,
        args.max_activation_tokens,
        device,
    )
    activations = {"calib": x_calib, "eval": x_eval}

    eval_prompts = build_prompts(tokenizer, args.eval_prompts, args.seed + 1000)
    reader = GGUFReader(args.iq4)
    gguf_tensors = {tensor.name: tensor for tensor in reader.tensors}

    results = {}
    variant_summaries = {}
    fp_last_logits = None
    with safe_open(args.hf, framework="pt", device="cpu") as hf:
        for variant in args.variants:
            print(f"[c1m] applying variant {variant}", flush=True)
            summaries = apply_variant(model, variant, args.layers, hf, gguf_tensors, activations, args, rng)
            print(f"[c1m] evaluating variant {variant}", flush=True)
            eval_result = evaluate_model(model, tokenizer, eval_prompts, args, fp_last_logits)
            if variant == "fp16":
                fp_last_logits = eval_result["captured_last_logits"]
            stripped = strip_logits(eval_result)
            mean_bpw_values = [row.get("candidate_mlp_bpw") for row in summaries if row.get("candidate_mlp_bpw") is not None]
            stripped["mean_mlp_bpw"] = float(np.mean(mean_bpw_values)) if mean_bpw_values else None
            results[variant] = stripped
            variant_summaries[variant] = summaries

    result = {
        "created_utc": datetime.now(UTC).isoformat(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "variants": results,
        "variant_summaries": variant_summaries,
    }
    status, decision_text, next_step = decide(result)
    result["status"] = status
    result["verdict"] = status
    result["decision_text"] = decision_text
    result["next_step"] = next_step

    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "result.md").write_text(make_markdown(result), encoding="utf-8")
    print(f"[c1m] wrote {args.output_dir / 'result.md'}", flush=True)
    print(decision_text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
