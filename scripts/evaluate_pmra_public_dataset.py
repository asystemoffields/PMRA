from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from datasets import load_dataset
import torch
from transformers import AutoTokenizer

from activation_conditioned_scale_mirage import parse_layers
from mlp_codebook_model_forward_gate import evaluate_model, strip_logits
from production_mixed_rate_transcoder_gate import (
    apply_selection,
    build_tensor_specs,
    filter_specs_for_model,
    group_specs,
    load_model_for_profile,
    open_hf_tensor_source,
    parse_source_specs,
    patch_all_from_source,
    selected_extra_bytes,
    source_readers,
    total_weight_count,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


def normalize_source_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def aggregate_prompt_hash(prompts: list[str]) -> str:
    joined = "\n---PMRA-PROMPT---\n".join(prompts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def load_public_prompts(
    tokenizer,
    dataset: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    prompt_count: int,
    seed: int,
    eval_max_length: int,
    min_tokens: int,
) -> tuple[list[str], dict]:
    ds = load_dataset(dataset, dataset_config, split=split) if dataset_config else load_dataset(dataset, split=split)
    chunks: list[str] = []
    buffer: list[str] = []
    char_floor = max(400, eval_max_length * 4)

    for row in ds:
        text = str(row.get(text_column, "")).strip()
        if not text:
            continue
        buffer.append(text)
        candidate = "\n\n".join(buffer)
        if len(candidate) < char_floor:
            continue
        token_count = len(tokenizer(candidate, add_special_tokens=True).input_ids)
        if token_count >= min_tokens:
            chunks.append(candidate)
            buffer = []

    if buffer:
        candidate = "\n\n".join(buffer)
        if len(tokenizer(candidate, add_special_tokens=True).input_ids) >= min_tokens:
            chunks.append(candidate)

    if len(chunks) < prompt_count:
        raise RuntimeError(
            f"public dataset produced only {len(chunks)} usable chunks; requested {prompt_count}"
        )

    rng = random.Random(seed)
    rng.shuffle(chunks)
    prompts = chunks[:prompt_count]
    token_lengths = [
        min(len(tokenizer(prompt, add_special_tokens=True).input_ids), eval_max_length)
        for prompt in prompts
    ]
    audit = {
        "dataset": dataset,
        "dataset_config": dataset_config,
        "split": split,
        "text_column": text_column,
        "seed": int(seed),
        "prompt_count": int(len(prompts)),
        "eval_max_length": int(eval_max_length),
        "min_tokens": int(min_tokens),
        "prompt_hash_sha256": aggregate_prompt_hash(prompts),
        "prompt_hashes_first16": [prompt_hash(prompt) for prompt in prompts[:16]],
        "min_truncated_tokens": int(min(token_lengths)),
        "mean_truncated_tokens": float(sum(token_lengths) / len(token_lengths)),
        "max_truncated_tokens": int(max(token_lengths)),
    }
    return prompts, audit


def result_json_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_byte_fields(row: dict, payload_bytes: int, total_weights: int, fp_nll: float) -> dict:
    row["payload_bytes"] = int(payload_bytes)
    row["payload_bpw"] = float(payload_bytes * 8 / total_weights)
    row["delta_nll_vs_fp16"] = float(row["nll"] - fp_nll)
    return row


def source_payload_from_result(result: dict, source: str) -> int:
    if source in result.get("source_payload_bytes", {}):
        return int(result["source_payload_bytes"][source])
    if source in result.get("variants", {}):
        return int(result["variants"][source]["payload_bytes"])
    raise KeyError(f"no payload byte count for source {source!r}")


def variant_payload_bytes(result: dict, variant: str, low_payload: int, total_weights: int) -> int:
    if variant == "fp16":
        return int(total_weights * 2)
    if variant in result.get("variants", {}):
        return int(result["variants"][variant]["payload_bytes"])
    if variant in result.get("selections", {}):
        base_source = result.get("selection_base_sources", {}).get(variant, result["args"]["low_source"])
        base_payload = source_payload_from_result(result, base_source)
        selected = result["selections"][variant]
        saved_bytes = sum(int(row.get("saved_bytes", 0)) for row in selected)
        return int(base_payload + selected_extra_bytes(selected) - saved_bytes)
    return source_payload_from_result(result, variant)


def needed_source_labels(result: dict, variants: list[str]) -> set[str]:
    args = result["args"]
    labels = {
        args["low_source"],
        args["target_source"],
        *normalize_source_list(args.get("high_sources")),
    }
    for variant in variants:
        if variant == "fp16" or variant in result.get("selections", {}):
            continue
        labels.add(variant)
    for selection_name in variants:
        base_source = result.get("selection_base_sources", {}).get(selection_name)
        if base_source:
            labels.add(base_source)
        for row in result.get("selections", {}).get(selection_name, []):
            labels.add(row["source"])
    return labels


def evaluate_variants(
    model,
    tokenizer,
    hf,
    readers: dict[str, dict],
    groups: dict,
    result: dict,
    variants: list[str],
    prompts: list[str],
    eval_max_length: int,
    total_weights: int,
    tensor_profile: str,
) -> dict[str, dict]:
    low_source = result["args"]["low_source"]
    low_payload = source_payload_from_result(result, low_source)
    eval_args = SimpleNamespace(eval_max_length=eval_max_length)
    out: dict[str, dict] = {}

    print("[pmra-public] evaluating fp16 reference", flush=True)
    fp_eval = evaluate_model(model, tokenizer, prompts, eval_args)
    fp_logits = fp_eval["captured_last_logits"]
    out["fp16"] = add_byte_fields(strip_logits(fp_eval), total_weights * 2, total_weights, fp_eval["nll"])

    for variant in variants:
        if variant == "fp16":
            continue
        if variant in result.get("selections", {}):
            base_source = result.get("selection_base_sources", {}).get(variant, low_source)
            print(f"[pmra-public] evaluating mixed selection {variant}", flush=True)
            patch_all_from_source(
                model,
                hf,
                readers,
                result["args"]["layers"],
                base_source,
                result["args"]["group_mode"],
                tensor_profile,
            )
            apply_selection(model, hf, readers, groups, result["selections"][variant])
        else:
            print(f"[pmra-public] evaluating source {variant}", flush=True)
            patch_all_from_source(
                model,
                hf,
                readers,
                result["args"]["layers"],
                variant,
                result["args"]["group_mode"],
                tensor_profile,
            )

        eval_result = evaluate_model(model, tokenizer, prompts, eval_args, fp_logits)
        out[variant] = add_byte_fields(
            strip_logits(eval_result),
            variant_payload_bytes(result, variant, low_payload, total_weights),
            total_weights,
            out["fp16"]["nll"],
        )
    return out


def decide(result: dict, candidate_variant: str, random_variant: str | None) -> tuple[str, str, str]:
    variants = result["variants"]
    candidate = variants[candidate_variant]
    target_source = result["args"]["target_source"]
    target = variants[target_source]
    candidate["nll_improvement_vs_target"] = float(target["nll"] - candidate["nll"])
    candidate["payload_bytes_vs_target"] = int(candidate["payload_bytes"] - target["payload_bytes"])

    q3ks_margin = None
    if "q3_k_s" in variants:
        q3ks_margin = float(variants["q3_k_s"]["nll"] - candidate["nll"])
        candidate["nll_improvement_vs_q3_k_s"] = q3ks_margin
        candidate["payload_bytes_vs_q3_k_s"] = int(candidate["payload_bytes"] - variants["q3_k_s"]["payload_bytes"])

    random_margin = None
    if random_variant and random_variant in variants:
        random_margin = float(variants[random_variant]["nll"] - candidate["nll"])
        candidate["nll_improvement_vs_random_control"] = random_margin

    if candidate["payload_bytes"] > target["payload_bytes"]:
        return (
            "NO-GO",
            "NO-GO: public eval candidate exceeds the target payload byte budget.",
            "Do not publish this operating point without a smaller allocation.",
        )
    if candidate["nll_improvement_vs_target"] < 0.03:
        return (
            "NO-GO",
            "NO-GO: public eval does not clear the target-source quality margin.",
            "Treat the project-local result as overfit until another selector survives public data.",
        )
    if q3ks_margin is not None and q3ks_margin < 0.01:
        return (
            "GRAY",
            "GRAY: public eval beats the target source but does not clearly beat Q3_K_S.",
            "Run a larger public eval and inspect whether Q3_K_S is the true frontier control.",
        )
    if random_margin is not None and random_margin < 0.01:
        return (
            "GRAY",
            "GRAY: public eval beats the target source but not the same-budget random allocation by a clear margin.",
            "Harden the selector before making a public method claim.",
        )
    return (
        "GO",
        "GO: public eval preserves the PMRA quality/size advantage against the target and controls.",
        "Package the method and rerun on a larger public benchmark mix.",
    )


def make_markdown(result: dict) -> str:
    lines = [
        "# Result Card - PMRA Public Dataset Evaluation",
        "",
        "## Status",
        "",
        result["verdict"],
        "",
        "## Dataset",
        "",
        f"- dataset: `{result['dataset_audit']['dataset']}`",
        f"- config: `{result['dataset_audit']['dataset_config']}`",
        f"- split: `{result['dataset_audit']['split']}`",
        f"- prompts: `{result['dataset_audit']['prompt_count']}`",
        f"- prompt hash: `{result['dataset_audit']['prompt_hash_sha256']}`",
        "",
        "## Variants",
        "",
        "| Variant | NLL | Delta vs FP16 | Payload bpw | Payload bytes | Last-logit MSE | Top-10 overlap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["variants"].items():
        logit_mse = row.get("last_logit_mse_to_fp16")
        overlap = row.get("top10_overlap_to_fp16")
        lines.append(
            f"| {name} | {row['nll']:.6f} | {row['delta_nll_vs_fp16']:.6f} | "
            f"{row['payload_bpw']:.6f} | {row['payload_bytes']} | "
            f"{logit_mse:.6g}" if logit_mse is not None else f"| {name} | {row['nll']:.6f} | {row['delta_nll_vs_fp16']:.6f} | {row['payload_bpw']:.6f} | {row['payload_bytes']} | n/a"
        )
        lines[-1] += f" | {overlap:.3f} |" if overlap is not None else " | n/a |"
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--source", action="append", default=[], help="Production GGUF source as label=path.")
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", default="fp16,iq2_m,iq3_xs,q3_k_s,c2_calib_greedy_mixed,c2_random_same_budget")
    parser.add_argument("--candidate-variant", default="c2_calib_greedy_mixed")
    parser.add_argument("--random-variant", default="c2_random_same_budget")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--prompt-count", type=int, default=512)
    parser.add_argument("--prompt-seed", type=int, default=1701)
    parser.add_argument("--eval-max-length", type=int, default=256)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    result["args"]["layers"] = parse_layers(",".join(map(str, result["args"]["layers"])))
    tensor_profile = result["args"].get("tensor_profile", "qwen")
    variants = [part.strip() for part in args.variants.split(",") if part.strip()]
    source_paths = parse_source_specs(args.source)
    missing = sorted(needed_source_labels(result, variants) - set(source_paths))
    if missing:
        raise ValueError(f"missing source paths for {missing}")

    print("[pmra-public] loading tokenizer/model", flush=True)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = load_model_for_profile(args.model_dir, tensor_profile, device)

    prompts, dataset_audit = load_public_prompts(
        tokenizer,
        args.dataset,
        args.dataset_config,
        args.split,
        args.text_column,
        args.prompt_count,
        args.prompt_seed,
        args.eval_max_length,
        args.min_tokens,
    )
    readers = source_readers(source_paths)
    specs, skipped_specs = filter_specs_for_model(
        model,
        build_tensor_specs(result["args"]["layers"], result["args"]["group_mode"], tensor_profile),
        log_prefix="[pmra-public]",
    )
    groups = group_specs(specs)

    with open_hf_tensor_source(args.hf) as hf:
        total_weights = total_weight_count(hf, model, specs)
        variants_result = evaluate_variants(
            model,
            tokenizer,
            hf,
            readers,
            groups,
            result,
            variants,
            prompts,
            args.eval_max_length,
            total_weights,
            tensor_profile,
        )

    out = {
        "created_utc": datetime.now(UTC).isoformat(),
        "args": {
            "model_dir": args.model_dir,
            "hf": args.hf,
            "result_json": str(args.result_json),
            "result_json_sha256": result_json_sha256(args.result_json),
            "low_source": result["args"]["low_source"],
            "target_source": result["args"]["target_source"],
            "variants": variants,
            "candidate_variant": args.candidate_variant,
            "random_variant": args.random_variant,
            "tensor_profile": tensor_profile,
            "device": args.device,
        },
        "source_result_prompt_audit": result.get("prompt_audit"),
        "dataset_audit": dataset_audit,
        "skipped_model_tensors": [spec.logical_name for spec in skipped_specs],
        "variants": variants_result,
    }
    (args.output_dir / "partial_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    status, decision_text, next_step = decide(out, args.candidate_variant, args.random_variant)
    out["status"] = status
    out["verdict"] = status
    out["decision_text"] = decision_text
    out["next_step"] = next_step

    (args.output_dir / "result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (args.output_dir / "result.md").write_text(make_markdown(out), encoding="utf-8")
    print(f"[pmra-public] wrote {args.output_dir / 'result.md'}", flush=True)
    print(decision_text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
