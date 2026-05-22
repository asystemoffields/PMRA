from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from activation_conditioned_scale_mirage import parse_layers
from production_mixed_rate_transcoder_gate import (
    apply_selection,
    build_tensor_specs,
    filter_specs_for_model,
    group_specs,
    load_model_for_profile,
    open_hf_tensor_source,
    parse_source_specs,
    patch_all_from_source,
    source_readers,
    total_weight_count,
)
from evaluate_pmra_public_dataset import (
    needed_source_labels,
    source_payload_from_result,
    variant_payload_bytes,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


def result_json_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate_task_hash(rows: list[dict]) -> str:
    joined = "\n---PMRA-CODE-LIKELIHOOD-TASK---\n".join(
        row["task_id"] + "\n" + row["context"] + "\n" + row["solution"]
        for row in rows
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def load_mbpp_tasks(split: str, limit: int, seed: int) -> tuple[list[dict], dict]:
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
    rows = [dict(row) for row in ds]
    rows.sort(key=lambda row: int(row["task_id"]))
    if seed:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if limit > 0:
        rows = rows[:limit]

    tasks = []
    for row in rows:
        tests = "\n".join(row.get("test_list") or [])
        context = (
            "# Task\n"
            f"{row['prompt'].strip()}\n\n"
            "# Required behavior\n"
            f"{tests.strip()}\n\n"
            "# Solution\n"
        )
        solution = row["code"].strip() + "\n"
        tasks.append(
            {
                "task_id": str(row["task_id"]),
                "context": context,
                "solution": solution,
            }
        )
    audit = {
        "benchmark": "mbpp_sanitized",
        "dataset": "google-research-datasets/mbpp",
        "config": "sanitized",
        "split": split,
        "task_count": len(tasks),
        "seed": int(seed),
        "task_hash_sha256": aggregate_task_hash(tasks),
        "task_ids": [row["task_id"] for row in tasks],
    }
    return tasks, audit


def load_humaneval_tasks(limit: int, seed: int) -> tuple[list[dict], dict]:
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = [dict(row) for row in ds]
    rows.sort(key=lambda row: row["task_id"])
    if seed:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if limit > 0:
        rows = rows[:limit]

    tasks = []
    for row in rows:
        tasks.append(
            {
                "task_id": row["task_id"],
                "context": row["prompt"],
                "solution": row["canonical_solution"].strip("\n") + "\n",
            }
        )
    audit = {
        "benchmark": "humaneval",
        "dataset": "openai/openai_humaneval",
        "config": None,
        "split": "test",
        "task_count": len(tasks),
        "seed": int(seed),
        "task_hash_sha256": aggregate_task_hash(tasks),
        "task_ids": [row["task_id"] for row in tasks],
    }
    return tasks, audit


def load_code_tasks(benchmark: str, split: str, limit: int, seed: int) -> tuple[list[dict], dict]:
    if benchmark == "mbpp_sanitized":
        return load_mbpp_tasks(split, limit, seed)
    if benchmark == "humaneval":
        return load_humaneval_tasks(limit, seed)
    raise ValueError(f"unknown benchmark {benchmark!r}")


def encode_scored_example(tokenizer, context: str, solution: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    context_ids = tokenizer(context, add_special_tokens=True).input_ids
    solution_ids = tokenizer(solution, add_special_tokens=False).input_ids
    if len(solution_ids) == 0:
        return None
    ids = context_ids + solution_ids
    if len(ids) > max_length:
        return None
    labels = [-100] * len(context_ids) + solution_ids
    return (
        torch.tensor([ids], dtype=torch.long),
        torch.tensor([labels], dtype=torch.long),
    )


def score_task(model, input_ids: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
    token_count = int((labels != -100).sum().item())
    return float(out.loss.detach().cpu()) * token_count, token_count


def score_variant(model, tokenizer, tasks: list[dict], max_length: int) -> dict:
    total_nll = 0.0
    total_tokens = 0
    skipped = []
    task_rows = []
    for idx, task in enumerate(tasks, start=1):
        encoded = encode_scored_example(tokenizer, task["context"], task["solution"], max_length)
        if encoded is None:
            skipped.append(task["task_id"])
            continue
        nll_sum, token_count = score_task(model, encoded[0], encoded[1])
        total_nll += nll_sum
        total_tokens += token_count
        task_nll = nll_sum / max(1, token_count)
        task_rows.append(
            {
                "task_id": task["task_id"],
                "solution_tokens": int(token_count),
                "nll": float(task_nll),
                "ppl": float(math.exp(min(50.0, task_nll))),
                "solution_sha256": hashlib.sha256(task["solution"].encode("utf-8")).hexdigest(),
            }
        )
        if idx % 25 == 0 or idx == len(tasks):
            print(f"[pmra-code-likelihood] scored {idx}/{len(tasks)} tasks", flush=True)

    mean_nll = total_nll / max(1, total_tokens)
    return {
        "tasks_scored": int(len(task_rows)),
        "tasks_skipped": int(len(skipped)),
        "skipped_task_ids": skipped,
        "solution_tokens": int(total_tokens),
        "nll": float(mean_nll),
        "ppl": float(math.exp(min(50.0, mean_nll))),
        "tasks": task_rows,
    }


def add_byte_fields(row: dict, payload_bytes: int, total_weights: int, fp_nll: float | None) -> dict:
    row["payload_bytes"] = int(payload_bytes)
    row["payload_bpw"] = float(payload_bytes * 8 / total_weights)
    if fp_nll is not None:
        row["delta_nll_vs_fp16"] = float(row["nll"] - fp_nll)
    return row


def evaluate_variants(
    model,
    tokenizer,
    hf,
    readers: dict[str, dict],
    groups: dict,
    result: dict,
    variants: list[str],
    tasks: list[dict],
    max_length: int,
    total_weights: int,
    tensor_profile: str,
) -> dict[str, dict]:
    low_source = result["args"]["low_source"]
    low_payload = source_payload_from_result(result, low_source)
    out: dict[str, dict] = {}
    fp_nll = None

    if "fp16" in variants:
        print("[pmra-code-likelihood] evaluating fp16 reference", flush=True)
        fp_row = score_variant(model, tokenizer, tasks, max_length)
        fp_nll = fp_row["nll"]
        out["fp16"] = add_byte_fields(fp_row, total_weights * 2, total_weights, fp_nll)

    for variant in variants:
        if variant == "fp16":
            continue
        if variant in result.get("selections", {}):
            base_source = result.get("selection_base_sources", {}).get(variant, low_source)
            print(f"[pmra-code-likelihood] evaluating mixed selection {variant}", flush=True)
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
            print(f"[pmra-code-likelihood] evaluating source {variant}", flush=True)
            patch_all_from_source(
                model,
                hf,
                readers,
                result["args"]["layers"],
                variant,
                result["args"]["group_mode"],
                tensor_profile,
            )
        row = score_variant(model, tokenizer, tasks, max_length)
        out[variant] = add_byte_fields(
            row,
            variant_payload_bytes(result, variant, low_payload, total_weights),
            total_weights,
            fp_nll,
        )
    return out


def decide(out: dict, candidate_variant: str, random_variant: str | None, tolerance: float) -> tuple[str, str, str]:
    variants = out["variants"]
    candidate = variants[candidate_variant]
    target = variants[out["args"]["target_source"]]
    candidate["nll_delta_vs_target"] = float(candidate["nll"] - target["nll"])
    candidate["payload_bytes_vs_target"] = int(candidate["payload_bytes"] - target["payload_bytes"])
    if random_variant and random_variant in variants:
        candidate["nll_delta_vs_random_control"] = float(candidate["nll"] - variants[random_variant]["nll"])

    if candidate["payload_bytes"] > target["payload_bytes"]:
        return (
            "FAIL",
            "Candidate is over the target tensor-payload budget.",
            "Reduce the allocation budget or choose a smaller source mix.",
        )
    if candidate["nll_delta_vs_target"] > tolerance:
        return (
            "MISS",
            "Candidate trails the target control on canonical-solution code likelihood.",
            "Use this result to steer the selector toward code-sensitive tensor groups.",
        )
    if random_variant and random_variant in variants and candidate["nll_delta_vs_random_control"] > tolerance:
        return (
            "GRAY",
            "Candidate is within tolerance of the target but trails the same-budget random control.",
            "Run another seed and inspect whether the selector is choosing code-relevant tensors.",
        )
    return (
        "GO",
        "Candidate preserves or improves canonical-solution code likelihood at the target tensor-payload budget.",
        "Replicate on another seed or code corpus before method-level generalization.",
    )


def render_markdown(out: dict) -> str:
    lines = [
        "# Result Card - PMRA Code Likelihood Benchmark",
        "",
        "## Status",
        "",
        out["verdict"],
        "",
        "## Benchmark",
        "",
        f"- benchmark: `{out['dataset_audit']['benchmark']}`",
        f"- dataset: `{out['dataset_audit']['dataset']}`",
        f"- config: `{out['dataset_audit']['config']}`",
        f"- split: `{out['dataset_audit']['split']}`",
        f"- tasks: `{out['dataset_audit']['task_count']}`",
        f"- task hash: `{out['dataset_audit']['task_hash_sha256']}`",
        "",
        "## Variants",
        "",
        "| Variant | Code NLL | Code PPL | Solution tokens | Payload bpw | Payload bytes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in out["variants"].items():
        lines.append(
            f"| {name} | {row['nll']:.6f} | {row['ppl']:.6f} | {row['solution_tokens']} | "
            f"{row['payload_bpw']:.6f} | {row['payload_bytes']} |"
        )
    candidate = out["variants"][out["args"]["candidate_variant"]]
    lines.extend(
        [
            "",
            "## Key Comparisons",
            "",
            f"- candidate NLL delta vs target: `{candidate['nll_delta_vs_target']:.6f}`",
            f"- candidate payload bytes vs target: `{candidate['payload_bytes_vs_target']}`",
        ]
    )
    if "nll_delta_vs_random_control" in candidate:
        lines.append(f"- candidate NLL delta vs same-budget random: `{candidate['nll_delta_vs_random_control']:.6f}`")
    lines.extend(["", "## Decision", "", out["decision_text"], "", "## Next Step", "", out["next_step"], ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--source", action="append", default=[], help="Production GGUF source as label=path.")
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", default="fp16,q2_k,q3_k_s,q4_k_m,c2_calib_greedy_mixed,c2_random_same_budget")
    parser.add_argument("--candidate-variant", default="c2_calib_greedy_mixed")
    parser.add_argument("--random-variant", default="c2_random_same_budget")
    parser.add_argument("--benchmark", choices=["mbpp_sanitized", "humaneval"], default="mbpp_sanitized")
    parser.add_argument("--split", default="test")
    parser.add_argument("--tasks", type=int, default=0, help="0 means all tasks in the split.")
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--tolerance", type=float, default=0.02)
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

    print("[pmra-code-likelihood] loading tokenizer/model", flush=True)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = load_model_for_profile(args.model_dir, tensor_profile, device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tasks, dataset_audit = load_code_tasks(args.benchmark, args.split, args.tasks, args.task_seed)
    readers = source_readers(source_paths)
    specs, _skipped_specs = filter_specs_for_model(
        model,
        build_tensor_specs(result["args"]["layers"], result["args"]["group_mode"], tensor_profile),
        log_prefix="[pmra-code-likelihood]",
    )
    groups = group_specs(specs)

    with open_hf_tensor_source(args.hf) as hf:
        total_weights = total_weight_count(hf, model, specs)
        variant_results = evaluate_variants(
            model,
            tokenizer,
            hf,
            readers,
            groups,
            result,
            variants,
            tasks,
            args.max_length,
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
            "variants": variants,
            "candidate_variant": args.candidate_variant,
            "random_variant": args.random_variant,
            "low_source": result["args"]["low_source"],
            "target_source": result["args"]["target_source"],
            "tensor_profile": tensor_profile,
            "benchmark": args.benchmark,
            "split": args.split,
            "tasks": args.tasks,
            "task_seed": args.task_seed,
            "max_length": args.max_length,
            "tolerance": args.tolerance,
            "device": args.device,
        },
        "source_result_prompt_audit": result.get("prompt_audit"),
        "dataset_audit": dataset_audit,
        "variants": variant_results,
    }
    (args.output_dir / "partial_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    status, decision_text, next_step = decide(out, args.candidate_variant, args.random_variant, args.tolerance)
    out["status"] = status
    out["verdict"] = status
    out["decision_text"] = decision_text
    out["next_step"] = next_step
    (args.output_dir / "result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (args.output_dir / "result.md").write_text(render_markdown(out), encoding="utf-8")
    print(f"[pmra-code-likelihood] wrote {args.output_dir / 'result.md'}", flush=True)
    print(decision_text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
