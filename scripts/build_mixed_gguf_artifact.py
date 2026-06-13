from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter

from activation_conditioned_scale_mirage import parse_layers
from production_mixed_rate_transcoder_gate import SUPPORTED_TENSOR_PROFILES, build_tensor_specs, group_specs

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


@dataclass(frozen=True)
class SourceBundle:
    path: Path
    reader: GGUFReader
    tensors: dict[str, object]


def parse_source_specs(items: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"source must look like label=path, got {item!r}")
        label, path = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"empty source label in {item!r}")
        sources[label] = Path(path.strip())
    if not sources:
        raise ValueError("at least one --source label=path is required")
    return sources


def normalize_source_list(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def load_sources(paths: dict[str, Path]) -> dict[str, SourceBundle]:
    bundles = {}
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing GGUF source {label}: {path}")
        print(f"[artifact] loading source {label}: {path}", flush=True)
        reader = GGUFReader(str(path))
        bundles[label] = SourceBundle(path=path, reader=reader, tensors={tensor.name: tensor for tensor in reader.tensors})
    return bundles


def field_value(field):
    value = field.contents()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [item.item() if isinstance(item, np.generic) else item for item in value]
    return value


def copy_metadata(writer: GGUFWriter, reader: GGUFReader) -> int:
    copied = 0
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    for key, field in reader.fields.items():
        if key in skip:
            continue
        value_type = field.types[0]
        sub_type = field.types[1] if value_type == GGUFValueType.ARRAY and len(field.types) > 1 else None
        writer.add_key_value(key, field_value(field), value_type, sub_type=sub_type)
        copied += 1
    return copied


def add_pmra_metadata(
    writer: GGUFWriter,
    *,
    result_json: Path,
    variant: str,
    low_source: str,
    target_source: str,
    high_sources: list[str],
    metadata_source: str,
    payload_bytes: int,
    payload_bpw: float,
    source_counts: Counter,
    source_payloads: dict[str, int],
) -> int:
    source_mix = {
        source: {
            "tensor_count": int(source_counts[source]),
            "payload_bytes": int(source_payloads[source]),
        }
        for source in sorted(source_counts)
    }
    result_digest = hashlib.sha256(result_json.read_bytes()).hexdigest()
    fields = {
        "pmra.method": "production_mixed_rate_allocation",
        "pmra.format_version": "1",
        "pmra.variant": variant,
        "pmra.low_source": low_source,
        "pmra.target_source": target_source,
        "pmra.high_sources": json.dumps(high_sources, separators=(",", ":")),
        "pmra.metadata_source": metadata_source,
        "pmra.payload_bytes": str(int(payload_bytes)),
        "pmra.payload_bpw": f"{payload_bpw:.9f}",
        "pmra.source_mix_json": json.dumps(source_mix, sort_keys=True, separators=(",", ":")),
        "pmra.result_json_sha256": result_digest,
    }
    for key, value in fields.items():
        writer.add_key_value(key, value, GGUFValueType.STRING)
    return len(fields)


def raw_shape_for_writer(tensor) -> list[int]:
    return [int(dim) for dim in tensor.data.shape]


def tensor_digest(tensor, max_bytes: int = 1 << 20) -> str:
    arr = tensor.data
    view = memoryview(np.ascontiguousarray(arr.reshape(-1)[: min(arr.size, max_bytes)]))
    return hashlib.sha256(view).hexdigest()


def build_tensor_source_map(
    result: dict,
    variant: str,
    low_source: str,
    layers: list[int],
    group_mode: str,
    tensor_profile: str = "qwen",
) -> tuple[dict[str, str], dict]:
    specs = build_tensor_specs(layers, group_mode, tensor_profile)
    groups = group_specs(specs)
    default_source = result.get("selection_base_sources", {}).get(variant, low_source)
    source_by_group = {row["group"]: row["source"] for row in result["selections"][variant]}
    tensor_sources: dict[str, str] = {}
    for group, group_specs_list in groups.items():
        source = source_by_group.get(group, default_source)
        for spec in group_specs_list:
            tensor_sources[spec.gguf_name] = source
    return tensor_sources, {
        "groups": groups,
        "source_by_group": source_by_group,
        "spec_count": len(specs),
        "tensor_profile": tensor_profile,
        "default_source": default_source,
    }


def add_tensor(writer: GGUFWriter, name: str, tensor) -> None:
    data = tensor.data
    raw_dtype = tensor.tensor_type if data.dtype == np.uint8 else None
    writer.add_tensor(name, data, raw_shape=raw_shape_for_writer(tensor), raw_dtype=raw_dtype)


def render_markdown(report: dict) -> str:
    c2 = report.get("c2_metrics", {})
    lines = [
        "# Result Card - C2 Mixed GGUF Artifact",
        "",
        f"## Status",
        "",
        report["status"],
        "",
        "## Artifact",
        "",
        f"- file: `{report['output_gguf']}`",
        f"- file size: `{report['file_size_bytes']}` bytes",
        f"- payload bytes: `{report['payload_bytes']}`",
        f"- metadata + alignment overhead: `{report['metadata_alignment_overhead_bytes']}` bytes",
        f"- payload bpw: `{report['payload_bpw']:.6f}`",
        f"- file bpw: `{report['file_bpw']:.6f}`",
        f"- PMRA metadata fields: `{report['pmra_kv_fields']}`",
        "",
        "## C2 Quality Marker",
        "",
    ]
    if c2:
        lines.extend(
            [
                f"- candidate NLL: `{c2['candidate_nll']:.6f}`",
                f"- target `{c2['target_source']}` NLL: `{c2['target_nll']:.6f}`",
                f"- NLL improvement vs target: `{c2['nll_improvement_vs_target']:.6f}`",
                f"- candidate payload bytes vs target: `{c2['payload_bytes_vs_target']}`",
            ]
        )
    else:
        lines.append("- no C2 metric block found")
    lines.extend(
        [
            "",
            "## Source Mix",
            "",
            "| Source | Tensors | Payload bytes |",
            "|---|---:|---:|",
        ]
    )
    for source, count in sorted(report["source_tensor_counts"].items()):
        lines.append(f"| {source} | {count} | {report['source_payload_bytes'][source]} |")
    lines.extend(
        [
            "",
            "## Load Check",
            "",
            f"- tensor count: `{report['load_check']['tensor_count']}`",
            f"- kv count: `{report['load_check']['kv_count']}`",
            f"- mismatched tensors: `{len(report['load_check']['mismatches'])}`",
            "",
            "## Decision",
            "",
            report["decision_text"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], help="Production GGUF source as label=path.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-gguf", default="mixed.gguf")
    parser.add_argument("--variant", default="c2_calib_greedy_mixed")
    parser.add_argument("--metadata-source", default=None)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--group-mode", default=None)
    parser.add_argument("--tensor-profile", default=None, choices=sorted(SUPPORTED_TENSOR_PROFILES))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    result_args = result["args"]
    low_source = result_args["low_source"]
    target_source = result_args["target_source"]
    selection_base_source = result.get("selection_base_sources", {}).get(args.variant, low_source)
    layers = parse_layers(args.layers) if args.layers else [int(layer) for layer in result_args["layers"]]
    group_mode = args.group_mode or result_args["group_mode"]
    tensor_profile = args.tensor_profile or result_args.get("tensor_profile", "qwen")
    source_paths = parse_source_specs(args.source)
    required_sources = {
        low_source,
        selection_base_source,
        *[row["source"] for row in result.get("selections", {}).get(args.variant, [])],
    }
    missing_sources = sorted(required_sources - set(source_paths))
    if missing_sources:
        raise ValueError(f"selection sources missing from --source inputs: {missing_sources}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_gguf = args.output_dir / args.output_gguf
    bundles = load_sources(source_paths)
    metadata_source = args.metadata_source or selection_base_source
    if metadata_source not in bundles:
        raise ValueError(f"metadata source {metadata_source!r} missing from --source inputs")

    tensor_sources, mapping_info = build_tensor_source_map(
        result,
        args.variant,
        low_source,
        layers,
        group_mode,
        tensor_profile,
    )
    base_reader = bundles[metadata_source].reader
    base_tensor_names = [tensor.name for tensor in base_reader.tensors]
    tensor_sources = {name: source for name, source in tensor_sources.items() if name in set(base_tensor_names)}
    missing_from_mapping = sorted(set(base_tensor_names) - set(tensor_sources))
    missing_from_sources = []
    for name, source in tensor_sources.items():
        if source not in bundles:
            missing_from_sources.append({"tensor": name, "source": source, "reason": "unknown_source"})
        elif name not in bundles[source].tensors:
            missing_from_sources.append({"tensor": name, "source": source, "reason": "tensor_not_in_source"})
    if missing_from_sources:
        raise ValueError(f"selection references unavailable tensors: {missing_from_sources[:5]}")

    source_counts = Counter()
    source_payloads: dict[str, int] = defaultdict(int)
    tensor_records = []
    payload_bytes = 0
    for name in base_tensor_names:
        source = tensor_sources.get(name, selection_base_source)
        tensor = bundles[source].tensors[name]
        source_counts[source] += 1
        source_payloads[source] += int(tensor.n_bytes)
        payload_bytes += int(tensor.n_bytes)
        tensor_records.append(
            {
                "name": name,
                "source": source,
                "type": tensor.tensor_type.name if isinstance(tensor.tensor_type, GGMLQuantizationType) else str(tensor.tensor_type),
                "n_elements": int(tensor.n_elements),
                "n_bytes": int(tensor.n_bytes),
                "digest_1m": tensor_digest(tensor),
            }
        )

    total_weights = int(result["total_weight_count"])
    payload_bpw = float(payload_bytes * 8 / total_weights)
    high_sources = normalize_source_list(result_args["high_sources"])
    candidate = result["variants"][args.variant]
    target = result["variants"][target_source]
    copied_kv = 0
    pmra_kv = 0
    file_size = 0
    load_check = {"tensor_count": 0, "kv_count": 0, "mismatches": []}
    if not args.dry_run:
        print(f"[artifact] writing {output_gguf}", flush=True)
        writer = GGUFWriter(output_gguf, str(base_reader.fields["general.architecture"].contents()))
        writer.data_alignment = int(base_reader.alignment)
        copied_kv = copy_metadata(writer, base_reader)
        pmra_kv = add_pmra_metadata(
            writer,
            result_json=args.result_json,
            variant=args.variant,
            low_source=low_source,
            target_source=target_source,
            high_sources=high_sources,
            metadata_source=metadata_source,
            payload_bytes=payload_bytes,
            payload_bpw=payload_bpw,
            source_counts=source_counts,
            source_payloads=source_payloads,
        )
        for idx, name in enumerate(base_tensor_names, start=1):
            source = tensor_sources.get(name, selection_base_source)
            tensor = bundles[source].tensors[name]
            if idx == 1 or idx % 25 == 0 or idx == len(base_tensor_names):
                print(f"[artifact] adding tensor {idx}/{len(base_tensor_names)}: {name} <- {source}", flush=True)
            add_tensor(writer, name, tensor)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        file_size = output_gguf.stat().st_size

        print("[artifact] verifying written GGUF", flush=True)
        check_reader = GGUFReader(str(output_gguf))
        check_tensors = {tensor.name: tensor for tensor in check_reader.tensors}
        mismatches = []
        for record in tensor_records:
            tensor = check_tensors.get(record["name"])
            if tensor is None:
                mismatches.append({"name": record["name"], "reason": "missing"})
                continue
            if int(tensor.n_bytes) != record["n_bytes"]:
                mismatches.append(
                    {
                        "name": record["name"],
                        "reason": "n_bytes",
                        "expected": record["n_bytes"],
                        "actual": int(tensor.n_bytes),
                    }
                )
                continue
            if tensor_digest(tensor) != record["digest_1m"]:
                mismatches.append({"name": record["name"], "reason": "digest_1m"})
        load_check = {
            "tensor_count": len(check_reader.tensors),
            "kv_count": len([key for key in check_reader.fields if not key.startswith("GGUF.")]),
            "mismatches": mismatches,
        }

    overhead = max(0, file_size - payload_bytes) if file_size else 0
    status = "GO" if not load_check["mismatches"] and (args.dry_run or file_size > 0) else "NO-GO"
    decision_text = (
        "GO: selected production-format tensor payloads were materialized into one loadable mixed GGUF artifact."
        if status == "GO"
        else "NO-GO: artifact materialization or reload verification failed."
    )
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "decision_text": decision_text,
        "result_json": str(args.result_json),
        "variant": args.variant,
        "low_source": low_source,
        "target_source": target_source,
        "selection_base_source": selection_base_source,
        "tensor_profile": tensor_profile,
        "metadata_source": metadata_source,
        "output_gguf": str(output_gguf),
        "dry_run": bool(args.dry_run),
        "file_size_bytes": int(file_size),
        "payload_bytes": int(payload_bytes),
        "metadata_alignment_overhead_bytes": int(overhead),
        "payload_bpw": payload_bpw,
        "file_bpw": float(file_size * 8 / total_weights) if file_size else 0.0,
        "total_weight_count": total_weights,
        "copied_kv_fields": int(copied_kv),
        "pmra_kv_fields": int(pmra_kv),
        "missing_from_mapping": missing_from_mapping,
        "source_tensor_counts": dict(source_counts),
        "source_payload_bytes": {key: int(val) for key, val in source_payloads.items()},
        "selection_summary": {
            "group_mode": group_mode,
            "tensor_profile": tensor_profile,
            "spec_count": mapping_info["spec_count"],
            "selected_groups": len(mapping_info["source_by_group"]),
        },
        "c2_metrics": {
            "candidate_nll": float(candidate["nll"]),
            "target_source": target_source,
            "target_nll": float(target["nll"]),
            "nll_improvement_vs_target": float(target["nll"] - candidate["nll"]),
            "payload_bytes_vs_target": int(payload_bytes - int(target["payload_bytes"])),
            "candidate_reported_payload_bytes": int(candidate["payload_bytes"]),
        },
        "load_check": load_check,
        "tensor_records": tensor_records,
    }
    (args.output_dir / "artifact_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "artifact_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"[artifact] wrote {args.output_dir / 'artifact_report.md'}", flush=True)
    print(decision_text, flush=True)
    return 0 if status == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
