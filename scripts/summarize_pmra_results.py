from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def fmt_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def fmt_int(value: int | None) -> str:
    return "" if value is None else str(value)


def fmt_mean_int(value: float | None) -> str:
    return "" if value is None else str(round(value))


def read_result(path: Path, variant: str) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    variants = result["variants"]
    if variant not in variants:
        raise KeyError(f"{path}: variant {variant!r} not found")
    candidate = variants[variant]
    target_source = result["args"]["target_source"]
    target = variants[target_source]
    q3ks = variants.get("q3_k_s")
    random_control = variants.get("c2_random_same_budget")
    total_weights = int(result["total_weight_count"])
    seed = int(result["args"]["seed"])
    payload = int(candidate["payload_bytes"])
    row = {
        "path": str(path),
        "seed": seed,
        "verdict": result.get("verdict", ""),
        "prompt_overlap": int(result.get("prompt_audit", {}).get("overlap_count", -1)),
        "variant": variant,
        "target_source": target_source,
        "candidate_nll": float(candidate["nll"]),
        "target_nll": float(target["nll"]),
        "q3_k_s_nll": float(q3ks["nll"]) if q3ks else None,
        "random_nll": float(random_control["nll"]) if random_control else None,
        "improvement_vs_target": float(target["nll"] - candidate["nll"]),
        "improvement_vs_q3_k_s": float(q3ks["nll"] - candidate["nll"]) if q3ks else None,
        "improvement_vs_random": float(random_control["nll"] - candidate["nll"]) if random_control else None,
        "payload_bytes": payload,
        "payload_vs_target": payload - int(target["payload_bytes"]),
        "payload_vs_q3_k_s": payload - int(q3ks["payload_bytes"]) if q3ks else None,
        "payload_bpw": float(payload * 8 / total_weights),
    }
    return row


def numeric_mean(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def markdown_table(rows: list[dict]) -> str:
    headers = [
        "Seed",
        "Verdict",
        "Overlap",
        "Cand NLL",
        "Target NLL",
        "Q3_K_S NLL",
        "Random NLL",
        "Vs target",
        "Vs Q3_K_S",
        "Vs random",
        "Payload bytes",
        "Payload vs target",
        "Payload vs Q3_K_S",
        "Payload bpw",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item["seed"]):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    row["verdict"],
                    str(row["prompt_overlap"]),
                    fmt_float(row["candidate_nll"]),
                    fmt_float(row["target_nll"]),
                    fmt_float(row["q3_k_s_nll"]),
                    fmt_float(row["random_nll"]),
                    fmt_float(row["improvement_vs_target"]),
                    fmt_float(row["improvement_vs_q3_k_s"]),
                    fmt_float(row["improvement_vs_random"]),
                    fmt_int(row["payload_bytes"]),
                    fmt_int(row["payload_vs_target"]),
                    fmt_int(row["payload_vs_q3_k_s"]),
                    fmt_float(row["payload_bpw"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Means:",
            "",
            f"- NLL improvement vs target: `{fmt_float(numeric_mean(rows, 'improvement_vs_target'))}`",
            f"- NLL improvement vs Q3_K_S: `{fmt_float(numeric_mean(rows, 'improvement_vs_q3_k_s'))}`",
            f"- NLL improvement vs random same-budget: `{fmt_float(numeric_mean(rows, 'improvement_vs_random'))}`",
            f"- payload bytes vs target: `{fmt_mean_int(numeric_mean(rows, 'payload_vs_target'))}`",
            f"- payload bytes vs Q3_K_S: `{fmt_mean_int(numeric_mean(rows, 'payload_vs_q3_k_s'))}`",
            f"- payload bpw: `{fmt_float(numeric_mean(rows, 'payload_bpw'))}`",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--variant", default="c2_calib_greedy_mixed")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    rows = [read_result(path, args.variant) for path in args.results]
    print(markdown_table(rows))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
