from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

import numpy as np
import torch
from datasets import load_dataset
from gguf import GGUFReader
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_conditioned_scale_mirage import (
    DEFAULT_HF,
    DEFAULT_MODEL_DIR,
    align_shape,
    build_prompts,
    load_hf_tensor,
    parse_layers,
)
from mlp_codebook_model_forward_gate import (
    copy_array_to_parameter,
    evaluate_model,
    load_gguf_tensor_any,
    patch_layer,
    patch_non_mlp,
    set_lm_head_weight,
    set_weight,
    strip_logits,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


@dataclass(frozen=True)
class TensorSpec:
    logical_name: str
    gguf_name: str
    group: str
    logical_slice: str = ""


class HFTensorSource:
    def keys(self) -> set[str]:
        raise NotImplementedError

    def get_tensor(self, name: str) -> torch.Tensor:
        raise NotImplementedError


class SingleSafetensorsSource(HFTensorSource):
    def __init__(self, path: Path):
        self.path = path
        self._ctx = None
        self._handle = None
        self._keys: set[str] = set()

    def __enter__(self) -> "SingleSafetensorsSource":
        self._ctx = safe_open(str(self.path), framework="pt", device="cpu")
        self._handle = self._ctx.__enter__()
        self._keys = set(self._handle.keys())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(exc_type, exc, tb)

    def keys(self) -> set[str]:
        return self._keys

    def get_tensor(self, name: str) -> torch.Tensor:
        if self._handle is None:
            raise RuntimeError("HF tensor source is not open")
        return self._handle.get_tensor(name)


class ShardedSafetensorsSource(HFTensorSource):
    def __init__(self, index_path: Path):
        self.index_path = index_path
        index = json.loads(index_path.read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = index["weight_map"]
        self._contexts = {}
        self._handles = {}

    def __enter__(self) -> "ShardedSafetensorsSource":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        for ctx in self._contexts.values():
            ctx.__exit__(exc_type, exc, tb)
        self._contexts.clear()
        self._handles.clear()

    def keys(self) -> set[str]:
        return set(self.weight_map)

    def get_tensor(self, name: str) -> torch.Tensor:
        if name not in self.weight_map:
            raise KeyError(f"missing HF tensor {name}")
        shard = self.weight_map[name]
        if shard not in self._handles:
            shard_path = self.index_path.parent / shard
            ctx = safe_open(str(shard_path), framework="pt", device="cpu")
            self._contexts[shard] = ctx
            self._handles[shard] = ctx.__enter__()
        return self._handles[shard].get_tensor(name)


def open_hf_tensor_source(path: str | Path) -> HFTensorSource:
    hf_path = Path(path)
    if hf_path.name.endswith(".safetensors.index.json"):
        return ShardedSafetensorsSource(hf_path)
    return SingleSafetensorsSource(hf_path)


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


def parse_csv(text: str | None) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_float_csv(text: str | None) -> list[float]:
    return [float(part) for part in parse_csv(text)]


def load_jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"[c2] warning: ignoring corrupt checkpoint row {path}:{line_number}: {exc}",
                    flush=True,
                )
    return rows


def append_jsonl_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()


def write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def prompt_digest(prompts: list[str]) -> str:
    digest = hashlib.sha1()
    for prompt in prompts:
        digest.update(prompt.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def selection_rows_digest(rows: list[dict]) -> str:
    keys = (
        "group",
        "source",
        "extra_bytes",
        "saved_bytes",
        "low_bytes",
        "high_bytes",
        "base_bytes",
        "demoted_bytes",
        "calib_nll_improvement",
        "calib_nll_loss",
    )
    payload = [
        {key: row[key] for key in keys if key in row}
        for row in sorted(rows, key=lambda item: (str(item.get("group", "")), str(item.get("source", ""))))
    ]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def selection_search_context(base_context: dict, args, rows: list[dict]) -> dict:
    return {
        **base_context,
        "group_mode": str(getattr(args, "group_mode", "")),
        "layers": [int(layer) for layer in (getattr(args, "layers", []) or [])],
        "source_labels": sorted({str(row.get("source", "")) for row in rows}),
        "eligible_row_count": int(len(rows)),
        "eligible_rows_digest": selection_rows_digest(rows),
    }


def clean_checkpoint_row(row: dict) -> dict:
    clean = dict(row)
    clean.pop("_checkpoint_context", None)
    return clean


def load_eval_checkpoint(path: Path, context: dict) -> dict | None:
    for row in reversed(load_jsonl_rows(path)):
        if row.get("_checkpoint_context") == context and row.get("result") is not None:
            print(f"[c2] checkpoint hit {context.get('kind', 'eval')}", flush=True)
            return row["result"]
    return None


def append_eval_checkpoint(path: Path, context: dict, result: dict) -> None:
    append_jsonl_row(path, {"_checkpoint_context": context, "result": result})


def load_fp16_checkpoint(checkpoint_dir: Path, context: dict) -> dict | None:
    meta_path = checkpoint_dir / "fp16_eval.json"
    logits_path = checkpoint_dir / "fp16_last_logits.npz"
    if not meta_path.exists() or not logits_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("_checkpoint_context") != context:
            return None
        loaded = np.load(logits_path)
        result = dict(meta["result"])
        result["captured_last_logits"] = [row for row in loaded["logits"]]
        print("[c2] checkpoint hit fp16 reference eval", flush=True)
        return result
    except Exception as exc:
        print(f"[c2] warning: ignoring fp16 checkpoint: {exc}", flush=True)
        return None


def write_fp16_checkpoint(checkpoint_dir: Path, context: dict, result: dict) -> None:
    logits = result.get("captured_last_logits") or []
    if not logits:
        return
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tmp_logits_path = checkpoint_dir / "fp16_last_logits.tmp.npz"
    logits_path = checkpoint_dir / "fp16_last_logits.npz"
    np.savez_compressed(tmp_logits_path, logits=np.stack(logits))
    tmp_logits_path.replace(logits_path)
    write_json_atomic(
        checkpoint_dir / "fp16_eval.json",
        {"_checkpoint_context": context, "result": strip_logits(result)},
    )


def load_selection_search_cache(path: Path, index_key: str, context: dict | list[dict]) -> dict[tuple[tuple[str, str], ...], dict]:
    contexts = context if isinstance(context, list) else [context]
    cache: dict[tuple[tuple[str, str], ...], dict] = {}
    for row in load_jsonl_rows(path):
        if row.get("context") not in contexts:
            continue
        selected = row.get("selected") or []
        if row.get("signature") is not None:
            sig = tuple(tuple(pair) for pair in row.get("signature", []))
        else:
            sig = selection_signature(selected)
        nll = row.get("nll", row.get("search_nll"))
        if nll is None:
            continue
        item = {
            index_key: int(row.get(index_key, len(cache) + 1)),
            "nll": float(nll),
            "search_nll": float(row.get("search_nll", nll)),
            "extra_bytes": int(row.get("extra_bytes", selected_extra_bytes(selected))),
            "selected": selected,
        }
        if row.get("validation_nll") is not None:
            item["validation_nll"] = float(row["validation_nll"])
        cache[sig] = item
    if cache:
        print(f"[c2] loaded {len(cache)} cached search fitness rows from {path}", flush=True)
    return cache


def append_selection_search_cache(path: Path, item: dict, index_key: str, context: dict) -> None:
    selected = item["selected"]
    row = {
        "context": context,
        index_key: int(item[index_key]),
        "signature": [list(pair) for pair in selection_signature(selected)],
        "nll": float(item["nll"]),
        "search_nll": float(item["search_nll"]),
        "extra_bytes": int(item["extra_bytes"]),
        "selected": selected,
    }
    if item.get("validation_nll") is not None:
        row["validation_nll"] = float(item["validation_nll"])
    append_jsonl_row(path, row)


def source_readers(paths: dict[str, Path]) -> dict[str, dict]:
    readers = {}
    for label, path in paths.items():
        print(f"[c2] loading GGUF source {label}: {path}", flush=True)
        reader = GGUFReader(str(path))
        readers[label] = {tensor.name: tensor for tensor in reader.tensors}
    return readers


SUPPORTED_TENSOR_PROFILES = {
    "qwen",
    "qwen35",
    "gemma4",
    "mistral3",
    "granite",
    "nemotron_h",
    "olmo2",
    "olmo3",
    "gpt_oss",
}


def load_model_for_profile(model_dir: str, tensor_profile: str, device: torch.device):
    if tensor_profile not in SUPPORTED_TENSOR_PROFILES:
        raise ValueError(f"unknown tensor profile {tensor_profile!r}")
    model_cls = AutoModelForCausalLM
    if tensor_profile == "mistral3":
        from transformers import Mistral3ForConditionalGeneration

        model_cls = Mistral3ForConditionalGeneration
    elif tensor_profile == "qwen35":
        try:
            from transformers import Qwen3_5ForConditionalGeneration

            model_cls = Qwen3_5ForConditionalGeneration
        except ImportError:
            model_cls = AutoModelForCausalLM
    extra_kwargs: dict = {}
    if tensor_profile in {"nemotron_h", "qwen35"}:
        extra_kwargs["trust_remote_code"] = True
    return model_cls.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        **extra_kwargs,
    ).to(device).eval()


def build_tensor_specs(layers: list[int], group_mode: str, tensor_profile: str = "qwen") -> list[TensorSpec]:
    if tensor_profile in {"qwen", "granite", "olmo2", "olmo3", "gpt_oss"}:
        specs: list[TensorSpec] = [
            TensorSpec("model.embed_tokens.weight", "token_embd.weight", "global:embed"),
            TensorSpec("lm_head.weight", "output.weight", "global:output"),
            TensorSpec("model.norm.weight", "output_norm.weight", "global:norm"),
        ]
    elif tensor_profile in {"mistral3", "qwen35"}:
        hf_root = "model.language_model"
        specs = [
            TensorSpec(f"{hf_root}.embed_tokens.weight", "token_embd.weight", "global:embed"),
            TensorSpec("lm_head.weight", "output.weight", "global:output"),
            TensorSpec(f"{hf_root}.norm.weight", "output_norm.weight", "global:norm"),
        ]
    elif tensor_profile == "gemma4":
        specs = [
            TensorSpec("model.language_model.norm.weight", "output_norm.weight", "global:norm"),
            TensorSpec(
                "model.language_model.per_layer_model_projection.weight",
                "per_layer_model_proj.weight",
                "global:per_layer_model_proj",
            ),
            TensorSpec(
                "model.language_model.per_layer_projection_norm.weight",
                "per_layer_proj_norm.weight",
                "global:per_layer_proj_norm",
            ),
            TensorSpec(
                "model.language_model.embed_tokens_per_layer.weight",
                "per_layer_token_embd.weight",
                "global:per_layer_token_embd",
            ),
            TensorSpec("model.language_model.embed_tokens.weight", "token_embd.weight", "global:embed"),
        ]
    elif tensor_profile == "nemotron_h":
        # NemotronH hybrid: M=Mamba2, -=MLP-only, *=Attention-only (42 layers)
        _NEMOTRON_H_PATTERN = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"
        hf_root = "backbone"
        specs = [
            TensorSpec(f"{hf_root}.embeddings.weight", "token_embd.weight", "global:embed"),
            TensorSpec("lm_head.weight", "output.weight", "global:output"),
            TensorSpec(f"{hf_root}.norm_f.weight", "output_norm.weight", "global:norm"),
        ]
    else:
        raise ValueError(f"unknown tensor profile {tensor_profile!r}")

    for layer in layers:
        if tensor_profile == "gemma4":
            hf_prefix = f"model.language_model.layers.{layer}"
        elif tensor_profile in {"mistral3", "qwen35"}:
            hf_prefix = f"model.language_model.layers.{layer}"
        elif tensor_profile == "nemotron_h":
            hf_prefix = f"backbone.layers.{layer}"
        else:
            hf_prefix = f"model.layers.{layer}"
        gguf_prefix = f"blk.{layer}"
        if group_mode == "layer_family":
            attn_group = f"L{layer}:attn"
            mlp_group = f"L{layer}:mlp"
            per_layer_group = f"L{layer}:per_layer"
        elif group_mode == "tensor":
            attn_group = ""
            mlp_group = ""
            per_layer_group = ""
        else:
            raise ValueError(f"unknown group mode {group_mode}")

        if tensor_profile == "gpt_oss":
            attn_pairs = [
                ("self_attn.sinks", "attn_sinks.weight", "attn_sinks"),
                ("self_attn.q_proj.weight", "attn_q.weight", "attn_q"),
                ("self_attn.q_proj.bias", "attn_q.bias", "attn_q_bias"),
                ("self_attn.k_proj.weight", "attn_k.weight", "attn_k"),
                ("self_attn.k_proj.bias", "attn_k.bias", "attn_k_bias"),
                ("self_attn.v_proj.weight", "attn_v.weight", "attn_v"),
                ("self_attn.v_proj.bias", "attn_v.bias", "attn_v_bias"),
                ("self_attn.o_proj.weight", "attn_output.weight", "attn_output"),
                ("self_attn.o_proj.bias", "attn_output.bias", "attn_output_bias"),
            ]
            for hf_tail, gguf_tail, short in attn_pairs:
                group = attn_group or f"L{layer}:{short}"
                specs.append(TensorSpec(f"{hf_prefix}.{hf_tail}", f"{gguf_prefix}.{gguf_tail}", group))

            mlp_pairs = [
                ("mlp.router.weight", "ffn_gate_inp.weight", "ffn_router", ""),
                ("mlp.router.bias", "ffn_gate_inp.bias", "ffn_router_bias", ""),
                ("mlp.experts.gate_up_proj", "ffn_gate_exps.weight", "ffn_gate_exps", "gate"),
                ("mlp.experts.gate_up_proj_bias", "ffn_gate_exps.bias", "ffn_gate_exps_bias", "gate"),
                ("mlp.experts.gate_up_proj", "ffn_up_exps.weight", "ffn_up_exps", "up"),
                ("mlp.experts.gate_up_proj_bias", "ffn_up_exps.bias", "ffn_up_exps_bias", "up"),
                ("mlp.experts.down_proj", "ffn_down_exps.weight", "ffn_down_exps", ""),
                ("mlp.experts.down_proj_bias", "ffn_down_exps.bias", "ffn_down_exps_bias", ""),
            ]
            for hf_tail, gguf_tail, short, logical_slice in mlp_pairs:
                group = mlp_group or f"L{layer}:{short}"
                specs.append(
                    TensorSpec(
                        f"{hf_prefix}.{hf_tail}",
                        f"{gguf_prefix}.{gguf_tail}",
                        group,
                        logical_slice=logical_slice,
                    )
                )

            norm_pairs = [
                ("input_layernorm.weight", "attn_norm.weight", "attn_norm"),
                ("post_attention_layernorm.weight", "post_attention_norm.weight", "post_attention_norm"),
            ]
            for hf_tail, gguf_tail, short in norm_pairs:
                group = attn_group if short.startswith("attn") else mlp_group
                group = group or f"L{layer}:{short}"
                specs.append(TensorSpec(f"{hf_prefix}.{hf_tail}", f"{gguf_prefix}.{gguf_tail}", group))
            continue

        if tensor_profile == "nemotron_h":
            # Heterogeneous layers: M=Mamba2, -=MLP-only, *=Attention-only
            layer_type = _NEMOTRON_H_PATTERN[layer] if layer < len(_NEMOTRON_H_PATTERN) else "M"
            if group_mode == "layer_family":
                if layer_type == "M":
                    block_group = f"L{layer}:ssm"
                elif layer_type == "-":
                    block_group = f"L{layer}:mlp"
                else:
                    block_group = f"L{layer}:attn"
            else:
                block_group = ""
            # Input norm (all layer types)
            norm_group = block_group or f"L{layer}:attn_norm"
            specs.append(TensorSpec(f"{hf_prefix}.norm.weight", f"{gguf_prefix}.attn_norm.weight", norm_group))
            if layer_type == "M":  # Mamba2 SSM
                ssm_pairs = [
                    ("mixer.in_proj.weight", "ssm_in.weight", "ssm_in"),
                    ("mixer.conv1d.weight", "ssm_conv1d.weight", "ssm_conv1d"),
                    ("mixer.conv1d.bias", "ssm_conv1d.bias", "ssm_conv1d_bias"),
                    ("mixer.dt_bias", "ssm_dt.bias", "ssm_dt_bias"),
                    ("mixer.A_log", "ssm_a", "ssm_a"),
                    ("mixer.D", "ssm_d", "ssm_d"),
                    ("mixer.norm.weight", "ssm_norm.weight", "ssm_norm"),
                    ("mixer.out_proj.weight", "ssm_out.weight", "ssm_out"),
                ]
                for hf_tail, gguf_tail, short in ssm_pairs:
                    group = block_group or f"L{layer}:{short}"
                    specs.append(TensorSpec(f"{hf_prefix}.{hf_tail}", f"{gguf_prefix}.{gguf_tail}", group))
            elif layer_type == "-":  # MLP-only (relu² — no gate_proj)
                mlp_pairs = [
                    ("mixer.up_proj.weight", "ffn_up.weight", "ffn_up"),
                    ("mixer.down_proj.weight", "ffn_down.weight", "ffn_down"),
                ]
                for hf_tail, gguf_tail, short in mlp_pairs:
                    group = block_group or f"L{layer}:{short}"
                    specs.append(TensorSpec(f"{hf_prefix}.{hf_tail}", f"{gguf_prefix}.{gguf_tail}", group))
            elif layer_type == "*":  # Attention-only (no MLP sublayer)
                attn_pairs = [
                    ("mixer.q_proj.weight", "attn_q.weight", "attn_q"),
                    ("mixer.k_proj.weight", "attn_k.weight", "attn_k"),
                    ("mixer.v_proj.weight", "attn_v.weight", "attn_v"),
                    ("mixer.o_proj.weight", "attn_output.weight", "attn_output"),
                ]
                for hf_tail, gguf_tail, short in attn_pairs:
                    group = block_group or f"L{layer}:{short}"
                    specs.append(TensorSpec(f"{hf_prefix}.{hf_tail}", f"{gguf_prefix}.{gguf_tail}", group))
            continue

        attn_pairs = [
            ("self_attn.q_proj.weight", "attn_q.weight", "attn_q"),
            ("self_attn.k_proj.weight", "attn_k.weight", "attn_k"),
            ("self_attn.v_proj.weight", "attn_v.weight", "attn_v"),
            ("self_attn.o_proj.weight", "attn_output.weight", "attn_output"),
        ]
        for hf_tail, gguf_tail, short in attn_pairs:
            group = attn_group or f"L{layer}:{short}"
            specs.append(TensorSpec(f"{hf_prefix}.{hf_tail}", f"{gguf_prefix}.{gguf_tail}", group))

        if tensor_profile == "qwen35":
            linear_attn_pairs = [
                ("linear_attn.in_proj_qkv.weight", "attn_qkv.weight", "attn_qkv"),
                ("linear_attn.in_proj_z.weight", "attn_gate.weight", "attn_gate"),
                ("linear_attn.A_log", "ssm_a", "ssm_a"),
                ("linear_attn.conv1d.weight", "ssm_conv1d.weight", "ssm_conv1d"),
                ("linear_attn.dt_bias", "ssm_dt.bias", "ssm_dt"),
                ("linear_attn.norm.weight", "ssm_norm.weight", "ssm_norm"),
                ("linear_attn.out_proj.weight", "ssm_out.weight", "ssm_out"),
                ("linear_attn.in_proj_a.weight", "ssm_alpha.weight", "ssm_alpha"),
                ("linear_attn.in_proj_b.weight", "ssm_beta.weight", "ssm_beta"),
            ]
            for hf_tail, gguf_tail, short in linear_attn_pairs:
                group = attn_group or f"L{layer}:{short}"
                specs.append(TensorSpec(f"{hf_prefix}.{hf_tail}", f"{gguf_prefix}.{gguf_tail}", group))

        mlp_pairs = [
            ("mlp.gate_proj.weight", "ffn_gate.weight", "ffn_gate"),
            ("mlp.up_proj.weight", "ffn_up.weight", "ffn_up"),
            ("mlp.down_proj.weight", "ffn_down.weight", "ffn_down"),
        ]
        for hf_tail, gguf_tail, short in mlp_pairs:
            group = mlp_group or f"L{layer}:{short}"
            specs.append(TensorSpec(f"{hf_prefix}.{hf_tail}", f"{gguf_prefix}.{gguf_tail}", group))

        if tensor_profile in {"olmo2", "olmo3"}:
            norm_pairs = [
                ("post_attention_layernorm.weight", "post_attention_norm.weight", "post_attention_norm"),
                ("post_feedforward_layernorm.weight", "post_ffw_norm.weight", "post_ffw_norm"),
            ]
        else:
            norm_pairs = [
                ("input_layernorm.weight", "attn_norm.weight", "attn_norm"),
                (
                    "post_attention_layernorm.weight",
                    "post_attention_norm.weight" if tensor_profile in {"gemma4", "qwen35"} else "ffn_norm.weight",
                    "post_attention_norm" if tensor_profile in {"gemma4", "qwen35"} else "ffn_norm",
                ),
            ]
        if tensor_profile != "granite":
            norm_pairs.extend(
                [
                    ("self_attn.q_norm.weight", "attn_q_norm.weight", "attn_q_norm"),
                    ("self_attn.k_norm.weight", "attn_k_norm.weight", "attn_k_norm"),
                ]
            )
        for hf_tail, gguf_tail, short in norm_pairs:
            group = attn_group if short.startswith("attn") else mlp_group
            group = group or f"L{layer}:{short}"
            specs.append(TensorSpec(f"{hf_prefix}.{hf_tail}", f"{gguf_prefix}.{gguf_tail}", group))
        if tensor_profile == "gemma4":
            gemma_pairs = [
                ("pre_feedforward_layernorm.weight", "ffn_norm.weight", "ffn_norm"),
                ("post_feedforward_layernorm.weight", "post_ffw_norm.weight", "post_ffw_norm"),
                ("post_per_layer_input_norm.weight", "post_norm.weight", "post_norm"),
                ("per_layer_input_gate.weight", "inp_gate.weight", "per_layer_inp_gate"),
                ("per_layer_projection.weight", "proj.weight", "per_layer_proj"),
                ("layer_scalar", "layer_output_scale.weight", "layer_output_scale"),
            ]
            for hf_tail, gguf_tail, short in gemma_pairs:
                if short.startswith("per_layer"):
                    group = per_layer_group or f"L{layer}:{short}"
                else:
                    group = mlp_group or f"L{layer}:{short}"
                specs.append(TensorSpec(f"{hf_prefix}.{hf_tail}", f"{gguf_prefix}.{gguf_tail}", group))
    return specs


def group_specs(specs: list[TensorSpec]) -> dict[str, list[TensorSpec]]:
    groups: dict[str, list[TensorSpec]] = {}
    for spec in specs:
        groups.setdefault(spec.group, []).append(spec)
    return groups


def filter_specs_for_model(model, specs: list[TensorSpec], log_prefix: str = "[c2]") -> tuple[list[TensorSpec], list[TensorSpec]]:
    model_tensor_names = set(dict(model.named_parameters()).keys()) | set(dict(model.named_buffers()).keys())
    kept = []
    skipped = []
    for spec in specs:
        if spec.logical_name in model_tensor_names or spec.logical_name == "lm_head.weight":
            kept.append(spec)
        else:
            skipped.append(spec)
    if skipped:
        print(
            f"{log_prefix} skipped {len(skipped)} specs absent from the loaded model: "
            f"{', '.join(spec.logical_name for spec in skipped[:8])}",
            flush=True,
        )
    return kept, skipped


def filter_specs_for_sources(
    readers: dict[str, dict],
    specs: list[TensorSpec],
    log_prefix: str = "[c2]",
) -> tuple[list[TensorSpec], list[TensorSpec]]:
    kept = []
    skipped = []
    for spec in specs:
        if all(spec.gguf_name in tensors for tensors in readers.values()):
            kept.append(spec)
        else:
            skipped.append(spec)
    if skipped:
        print(
            f"{log_prefix} skipped {len(skipped)} specs absent from at least one GGUF source: "
            f"{', '.join(spec.gguf_name for spec in skipped[:8])}",
            flush=True,
        )
    return kept, skipped


def apply_logical_slice_array(array: np.ndarray, spec: TensorSpec) -> np.ndarray:
    if not spec.logical_slice:
        return array
    if spec.logical_slice not in {"gate", "up"}:
        raise ValueError(f"unknown logical slice {spec.logical_slice!r}")
    if array.shape[-1] % 2:
        raise ValueError(f"cannot split odd last dimension for {spec.logical_name}: {array.shape}")
    midpoint = array.shape[-1] // 2
    if spec.logical_slice == "gate":
        return array[..., :midpoint]
    return array[..., midpoint:]


def copy_array_to_parameter_logical_slice(param: torch.nn.Parameter | torch.Tensor, array: np.ndarray, spec: TensorSpec) -> None:
    if not spec.logical_slice:
        copy_array_to_parameter(param, array)
        return
    if spec.logical_slice not in {"gate", "up"}:
        raise ValueError(f"unknown logical slice {spec.logical_slice!r}")
    if param.shape[-1] % 2:
        raise ValueError(f"cannot split odd last dimension for {spec.logical_name}: {tuple(param.shape)}")
    midpoint = param.shape[-1] // 2
    target = param[..., :midpoint] if spec.logical_slice == "gate" else param[..., midpoint:]
    tensor = torch.from_numpy(np.array(array, copy=True, order="C")).to(device=target.device, dtype=target.dtype)
    with torch.no_grad():
        target.copy_(tensor)


def hf_ref_array(hf, model, spec: TensorSpec) -> np.ndarray:
    if spec.logical_name in hf.keys():
        return apply_logical_slice_array(load_hf_tensor(hf, spec.logical_name), spec)
    if spec.logical_name == "lm_head.weight":
        if "lm_head.weight" in hf.keys():
            return apply_logical_slice_array(load_hf_tensor(hf, "lm_head.weight"), spec)
        return apply_logical_slice_array(model.lm_head.weight.detach().to(torch.float32).cpu().numpy(), spec)
    params = dict(model.named_parameters())
    if spec.logical_name in params:
        return apply_logical_slice_array(params[spec.logical_name].detach().to(torch.float32).cpu().numpy(), spec)
    buffers = dict(model.named_buffers())
    if spec.logical_name in buffers:
        return apply_logical_slice_array(buffers[spec.logical_name].detach().to(torch.float32).cpu().numpy(), spec)
    raise KeyError(f"missing HF tensor {spec.logical_name}")


def patch_tensor(model, hf, source_tensors: dict, spec: TensorSpec) -> None:
    ref = hf_ref_array(hf, model, spec)
    arr = align_shape(ref, load_gguf_tensor_any(source_tensors, spec.gguf_name))
    name = spec.logical_name
    params = dict(model.named_parameters())
    if name in params:
        copy_array_to_parameter_logical_slice(params[name], arr, spec)
        return
    buffers = dict(model.named_buffers())
    if name in buffers:
        copy_array_to_parameter_logical_slice(buffers[name], arr, spec)
        return
    if name == "model.embed_tokens.weight":
        copy_array_to_parameter(model.model.embed_tokens.weight, arr)
        return
    if name == "lm_head.weight":
        set_lm_head_weight(model, arr)
        return
    if name == "model.norm.weight":
        copy_array_to_parameter(model.model.norm.weight, arr)
        return

    parts = name.split(".")
    layer = int(parts[2])
    layer_mod = model.model.layers[layer]
    if ".self_attn." in name:
        attn = layer_mod.self_attn
        if name.endswith("q_proj.weight"):
            set_weight(attn, "q_proj", arr)
        elif name.endswith("k_proj.weight"):
            set_weight(attn, "k_proj", arr)
        elif name.endswith("v_proj.weight"):
            set_weight(attn, "v_proj", arr)
        elif name.endswith("o_proj.weight"):
            set_weight(attn, "o_proj", arr)
        elif name.endswith("q_norm.weight"):
            copy_array_to_parameter(attn.q_norm.weight, arr)
        elif name.endswith("k_norm.weight"):
            copy_array_to_parameter(attn.k_norm.weight, arr)
        else:
            raise ValueError(f"unhandled attention tensor {name}")
        return
    if ".mlp." in name:
        mlp = layer_mod.mlp
        if name.endswith("gate_proj.weight"):
            set_weight(mlp, "gate_proj", arr)
        elif name.endswith("up_proj.weight"):
            set_weight(mlp, "up_proj", arr)
        elif name.endswith("down_proj.weight"):
            set_weight(mlp, "down_proj", arr)
        else:
            raise ValueError(f"unhandled MLP tensor {name}")
        return
    if name.endswith("input_layernorm.weight"):
        copy_array_to_parameter(layer_mod.input_layernorm.weight, arr)
        return
    if name.endswith("post_attention_layernorm.weight"):
        copy_array_to_parameter(layer_mod.post_attention_layernorm.weight, arr)
        return
    raise ValueError(f"unhandled tensor {name}")


def patch_group(model, hf, readers: dict[str, dict], groups: dict[str, list[TensorSpec]], group: str, source: str) -> None:
    for spec in groups[group]:
        patch_tensor(model, hf, readers[source], spec)


def patch_all_from_source(
    model,
    hf,
    readers: dict[str, dict],
    layers: list[int],
    source: str,
    group_mode: str = "tensor",
    tensor_profile: str = "qwen",
) -> None:
    tensors = readers[source]
    specs, _skipped = filter_specs_for_model(
        model,
        build_tensor_specs(layers, group_mode, tensor_profile),
    )
    for spec in specs:
        if spec.gguf_name in tensors:
            patch_tensor(model, hf, tensors, spec)


def tensor_payload_bytes(tensors: dict, spec: TensorSpec) -> int:
    return int(tensors[spec.gguf_name].data.nbytes)


def payload_bytes_by_source(readers: dict[str, dict], specs: list[TensorSpec]) -> dict[str, int]:
    out = {}
    for label, tensors in readers.items():
        out[label] = int(sum(tensor_payload_bytes(tensors, spec) for spec in specs if spec.gguf_name in tensors))
    return out


def group_payload_bytes(readers: dict[str, dict], groups: dict[str, list[TensorSpec]], group: str, source: str) -> int:
    return int(sum(tensor_payload_bytes(readers[source], spec) for spec in groups[group]))


def total_weight_count(hf, model, specs: list[TensorSpec]) -> int:
    total = 0
    for spec in specs:
        total += int(hf_ref_array(hf, model, spec).size)
    return total


def evaluate_nll(model, tokenizer, prompts: list[str], max_length: int) -> dict:
    total_nll = 0.0
    total_tokens = 0
    device = next(model.parameters()).device
    with torch.inference_mode():
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            ids = encoded.input_ids.to(device)
            if ids.shape[-1] < 2:
                continue
            out = model(input_ids=ids, labels=ids, use_cache=False)
            count = int(ids.shape[-1] - 1)
            total_nll += float(out.loss.detach().cpu()) * count
            total_tokens += count
    nll = total_nll / max(1, total_tokens)
    return {"tokens": int(total_tokens), "nll": float(nll), "ppl": float(math.exp(nll))}


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def aggregate_prompt_hash(prompts: list[str]) -> str:
    joined = "\n---PMRA-PROMPT---\n".join(prompts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_disjoint_prompt_split(tokenizer, calib_count: int, eval_count: int, seed: int) -> tuple[list[str], list[str], dict]:
    prompts = build_prompts(tokenizer, calib_count + eval_count, seed)
    calib_prompts = prompts[:calib_count]
    eval_prompts = prompts[calib_count:]
    overlap = set(calib_prompts) & set(eval_prompts)
    if overlap:
        raise AssertionError(f"calibration/eval prompt overlap detected: {len(overlap)}")
    return calib_prompts, eval_prompts, {
        "split": "single_stream_disjoint",
        "seed": int(seed),
        "calib_prompt_hashes": [prompt_hash(prompt) for prompt in calib_prompts],
        "eval_prompt_hashes": [prompt_hash(prompt) for prompt in eval_prompts],
        "overlap_count": 0,
    }


def load_public_prompt_chunks(
    tokenizer,
    dataset: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    prompt_count: int,
    seed: int,
    max_length: int,
    min_tokens: int,
) -> list[str]:
    ds = load_dataset(dataset, dataset_config, split=split) if dataset_config else load_dataset(dataset, split=split)
    chunks: list[str] = []
    buffer: list[str] = []
    char_floor = max(400, max_length * 4)

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
        raise RuntimeError(f"public split {split!r} produced only {len(chunks)} usable chunks; requested {prompt_count}")

    rng = random.Random(seed)
    rng.shuffle(chunks)
    return chunks[:prompt_count]


def public_prompt_audit(
    tokenizer,
    dataset: str,
    dataset_config: str | None,
    text_column: str,
    calib_split: str,
    eval_split: str,
    seed: int,
    calib_prompts: list[str],
    eval_prompts: list[str],
    calib_max_length: int,
    eval_max_length: int,
    min_tokens: int,
) -> dict:
    calib_lengths = [
        min(len(tokenizer(prompt, add_special_tokens=True).input_ids), calib_max_length)
        for prompt in calib_prompts
    ]
    eval_lengths = [
        min(len(tokenizer(prompt, add_special_tokens=True).input_ids), eval_max_length)
        for prompt in eval_prompts
    ]
    overlap = set(calib_prompts) & set(eval_prompts)
    return {
        "split": "public_disjoint",
        "dataset": dataset,
        "dataset_config": dataset_config,
        "text_column": text_column,
        "calib_split": calib_split,
        "eval_split": eval_split,
        "seed": int(seed),
        "calib_prompt_count": int(len(calib_prompts)),
        "eval_prompt_count": int(len(eval_prompts)),
        "calib_max_length": int(calib_max_length),
        "eval_max_length": int(eval_max_length),
        "min_tokens": int(min_tokens),
        "calib_prompt_hash_sha256": aggregate_prompt_hash(calib_prompts),
        "eval_prompt_hash_sha256": aggregate_prompt_hash(eval_prompts),
        "calib_prompt_hashes_first16": [prompt_hash(prompt) for prompt in calib_prompts[:16]],
        "eval_prompt_hashes_first16": [prompt_hash(prompt) for prompt in eval_prompts[:16]],
        "calib_mean_truncated_tokens": float(sum(calib_lengths) / max(1, len(calib_lengths))),
        "eval_mean_truncated_tokens": float(sum(eval_lengths) / max(1, len(eval_lengths))),
        "overlap_count": int(len(overlap)),
    }


def build_public_prompt_split(tokenizer, args) -> tuple[list[str], list[str], dict]:
    seed = args.prompt_seed if args.prompt_seed is not None else args.seed + 2000
    if args.calib_split == args.eval_split:
        prompts = load_public_prompt_chunks(
            tokenizer,
            args.dataset,
            args.dataset_config,
            args.calib_split,
            args.text_column,
            args.calib_prompts + args.eval_prompts,
            seed,
            max(args.calib_max_length, args.eval_max_length),
            args.min_tokens,
        )
        calib_prompts = prompts[: args.calib_prompts]
        eval_prompts = prompts[args.calib_prompts :]
    else:
        calib_prompts = load_public_prompt_chunks(
            tokenizer,
            args.dataset,
            args.dataset_config,
            args.calib_split,
            args.text_column,
            args.calib_prompts,
            seed,
            args.calib_max_length,
            args.min_tokens,
        )
        eval_prompts = load_public_prompt_chunks(
            tokenizer,
            args.dataset,
            args.dataset_config,
            args.eval_split,
            args.text_column,
            args.eval_prompts,
            seed + 17,
            args.eval_max_length,
            args.min_tokens,
        )
    audit = public_prompt_audit(
        tokenizer,
        args.dataset,
        args.dataset_config,
        args.text_column,
        args.calib_split,
        args.eval_split,
        seed,
        calib_prompts,
        eval_prompts,
        args.calib_max_length,
        args.eval_max_length,
        args.min_tokens,
    )
    if audit["overlap_count"]:
        raise AssertionError(f"calibration/eval prompt overlap detected: {audit['overlap_count']}")
    return calib_prompts, eval_prompts, audit


def group_weight_sse_delta(hf, model, readers: dict[str, dict], groups: dict[str, list[TensorSpec]], group: str, low: str, source: str) -> float:
    low_sse = 0.0
    high_sse = 0.0
    for spec in groups[group]:
        ref = hf_ref_array(hf, model, spec)
        low_arr = align_shape(ref, load_gguf_tensor_any(readers[low], spec.gguf_name))
        high_arr = align_shape(ref, load_gguf_tensor_any(readers[source], spec.gguf_name))
        low_sse += float(np.sum((ref - low_arr) ** 2))
        high_sse += float(np.sum((ref - high_arr) ** 2))
    return low_sse - high_sse


def build_direct_promotion_rows(
    readers: dict[str, dict],
    groups: dict[str, list[TensorSpec]],
    low_source: str,
    high_sources: list[str],
) -> list[dict]:
    rows: list[dict] = []
    for group in groups:
        low_bytes = group_payload_bytes(readers, groups, group, low_source)
        for source in high_sources:
            high_bytes = group_payload_bytes(readers, groups, group, source)
            extra = high_bytes - low_bytes
            if extra <= 0:
                continue
            rows.append(
                {
                    "group": group,
                    "source": source,
                    "low_bytes": int(low_bytes),
                    "high_bytes": int(high_bytes),
                    "extra_bytes": int(extra),
                    "calib_nll": None,
                    "calib_nll_improvement": 1.0,
                    "calib_score_per_mbyte": float(1.0 / (extra / 1_000_000)),
                    "weight_sse_delta": 0.0,
                    "weight_score_per_mbyte": 0.0,
                    "direct_search_only": True,
                }
            )
    return rows


def select_by_score(rows: list[dict], budget_extra: int, score_key: str) -> list[dict]:
    selected: list[dict] = []
    seen_groups: set[str] = set()
    used = 0
    ranked = sorted(rows, key=lambda row: row.get(score_key, float("-inf")), reverse=True)
    for row in ranked:
        if row["group"] in seen_groups:
            continue
        if row["extra_bytes"] <= 0:
            continue
        if row.get(score_key, 0.0) <= 0.0:
            continue
        if used + row["extra_bytes"] > budget_extra:
            continue
        selected.append(row)
        seen_groups.add(row["group"])
        used += row["extra_bytes"]
    return selected


def prune_knapsack_states(
    states: list[tuple[int, float, tuple[dict, ...]]],
    max_states: int,
) -> list[tuple[int, float, tuple[dict, ...]]]:
    max_states = max(1, int(max_states))
    best_by_used: dict[int, tuple[float, tuple[dict, ...]]] = {}
    for used, value, selected in states:
        previous = best_by_used.get(used)
        if previous is None or value > previous[0]:
            best_by_used[used] = (value, selected)

    frontier: list[tuple[int, float, tuple[dict, ...]]] = []
    best_value = float("-inf")
    for used in sorted(best_by_used):
        value, selected = best_by_used[used]
        if value <= best_value + 1e-12:
            continue
        frontier.append((used, value, selected))
        best_value = value

    if len(frontier) <= max_states:
        return frontier

    keep: dict[int, tuple[int, float, tuple[dict, ...]]] = {}
    value_keep_count = max(1, max_states // 2)
    for state in sorted(frontier, key=lambda item: (item[1], -item[0]), reverse=True)[:value_keep_count]:
        keep[state[0]] = state

    spread_count = max_states - len(keep)
    if spread_count > 0:
        denom = max(1, spread_count - 1)
        last_idx = len(frontier) - 1
        for idx in range(spread_count):
            state = frontier[round(idx * last_idx / denom)]
            previous = keep.get(state[0])
            if previous is None or state[1] > previous[1]:
                keep[state[0]] = state

    limited = sorted(keep.values(), key=lambda item: item[0])
    if len(limited) > max_states:
        limited = sorted(limited, key=lambda item: (item[1], -item[0]), reverse=True)[:max_states]
        limited.sort(key=lambda item: item[0])
    return limited


def select_exact_scaled_knapsack(
    options_by_group: dict[str, list[dict]],
    budget_extra: int,
    value_key: str,
    byte_unit: int,
) -> list[dict]:
    budget_units = budget_extra // byte_unit
    groups = sorted(options_by_group)
    neg_inf = float("-inf")
    dp = [neg_inf] * (budget_units + 1)
    dp[0] = 0.0
    parent_choices: list[list[int]] = []
    parent_used: list[list[int]] = []
    stage_options: list[list[dict]] = []

    for group in groups:
        options = sorted(
            options_by_group[group],
            key=lambda row: (
                float(row.get(value_key, 0.0)),
                -int(row.get("extra_bytes", 0)),
                row["source"],
            ),
            reverse=True,
        )
        next_dp = dp[:]
        choices = [-1] * (budget_units + 1)
        previous = [-1] * (budget_units + 1)
        for option_idx, row in enumerate(options):
            weight = int(row["extra_bytes"]) // byte_unit
            value = float(row[value_key])
            if weight > budget_units:
                continue
            for used, current in enumerate(dp[: budget_units - weight + 1]):
                if current == neg_inf:
                    continue
                next_used = used + weight
                candidate = current + value
                if candidate > next_dp[next_used] + 1e-12:
                    next_dp[next_used] = candidate
                    choices[next_used] = option_idx
                    previous[next_used] = used
        dp = next_dp
        parent_choices.append(choices)
        parent_used.append(previous)
        stage_options.append(options)

    best_used = max(range(budget_units + 1), key=lambda used: (dp[used], -used))
    selected: list[dict] = []
    used = best_used
    for choices, previous, options in zip(reversed(parent_choices), reversed(parent_used), reversed(stage_options)):
        option_idx = choices[used]
        if option_idx >= 0:
            selected.append(options[option_idx])
            used = previous[used]
    selected.reverse()
    return selected


def select_knapsack_by_value(
    rows: list[dict],
    budget_extra: int,
    value_key: str,
    max_states: int = 50_000,
) -> list[dict]:
    max_states = max(1, int(max_states))
    options_by_group: dict[str, list[dict]] = {}
    for row in rows:
        extra = int(row.get("extra_bytes", 0))
        value = float(row.get(value_key, 0.0))
        if extra <= 0 or value <= 0.0 or extra > budget_extra:
            continue
        options_by_group.setdefault(row["group"], []).append(row)

    byte_unit = int(budget_extra)
    for options in options_by_group.values():
        for row in options:
            byte_unit = math.gcd(byte_unit, int(row["extra_bytes"]))
    if byte_unit > 0 and budget_extra // byte_unit <= max_states:
        return select_exact_scaled_knapsack(options_by_group, budget_extra, value_key, byte_unit)

    states: list[tuple[int, float, tuple[dict, ...]]] = [(0, 0.0, ())]
    for group in sorted(options_by_group):
        options = sorted(
            options_by_group[group],
            key=lambda row: (
                float(row.get(value_key, 0.0)),
                -int(row.get("extra_bytes", 0)),
                row["source"],
            ),
            reverse=True,
        )
        next_states = states[:]
        for used, value, selected in states:
            for row in options:
                next_used = used + int(row["extra_bytes"])
                if next_used > budget_extra:
                    continue
                next_states.append((next_used, value + float(row[value_key]), selected + (row,)))
        states = prune_knapsack_states(next_states, max_states)

    best = max(states, key=lambda item: (item[1], -item[0]))
    return list(best[2])


def build_promotion_selection(
    selector: str,
    rows: list[dict],
    budget_extra: int,
    max_states: int,
) -> list[dict]:
    selector = selector.strip()
    if selector in {"greedy", "calib_greedy"}:
        return select_by_score(rows, budget_extra, "calib_score_per_mbyte")
    if selector in {"knapsack", "calib_knapsack"}:
        return select_knapsack_by_value(rows, budget_extra, "calib_nll_improvement", max_states)
    if selector in {"weight", "weight_mse"}:
        return select_by_score(rows, budget_extra, "weight_score_per_mbyte")
    if selector in {"blend", "calib_weight_blend"}:
        return select_by_score(rows, budget_extra, "calib_weight_rank_blend")
    raise ValueError(f"unknown promotion sweep selector {selector!r}")


def promotion_selector_variant_prefix(selector: str) -> str:
    selector = selector.strip()
    if selector in {"greedy", "calib_greedy"}:
        return "c2_calib_greedy"
    if selector in {"knapsack", "calib_knapsack"}:
        return "c2_calib_knapsack"
    if selector in {"weight", "weight_mse"}:
        return "c2_weight_mse"
    if selector in {"blend", "calib_weight_blend"}:
        return "c2_calib_weight_blend"
    raise ValueError(f"unknown promotion sweep selector {selector!r}")


def bpw_tag(value: float) -> str:
    return f"{value:.3f}".replace("-", "m").replace(".", "p")


def sweep_variant_name(selector: str, payload_bpw: float) -> str:
    return f"{promotion_selector_variant_prefix(selector)}_bpw_{bpw_tag(payload_bpw)}_mixed"


def local_search_variant_name(base_variant: str) -> str:
    if base_variant.endswith("_mixed"):
        return f"{base_variant[:-len('_mixed')]}_local_mixed"
    return f"{base_variant}_local"


def genetic_variant_name(base_variant: str) -> str:
    if base_variant.endswith("_mixed"):
        return f"{base_variant[:-len('_mixed')]}_genetic_mixed"
    return f"{base_variant}_genetic"


def anneal_variant_name(base_variant: str) -> str:
    if base_variant.endswith("_mixed"):
        return f"{base_variant[:-len('_mixed')]}_anneal_mixed"
    return f"{base_variant}_anneal"


def infer_local_search_base_variant(candidate_variant: str) -> str:
    if candidate_variant.endswith("_local_mixed"):
        return f"{candidate_variant[:-len('_local_mixed')]}_mixed"
    return "c2_calib_knapsack_mixed"


def infer_genetic_search_base_variant(candidate_variant: str) -> str:
    if candidate_variant.endswith("_genetic_mixed"):
        return f"{candidate_variant[:-len('_genetic_mixed')]}_mixed"
    return "c2_calib_knapsack_mixed"


def infer_anneal_search_base_variant(candidate_variant: str) -> str:
    if candidate_variant.endswith("_anneal_mixed"):
        return f"{candidate_variant[:-len('_anneal_mixed')]}_mixed"
    return "c2_calib_knapsack_mixed"


def selection_signature(selected: list[dict]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(row["group"]), str(row["source"])) for row in selected))


def selection_by_group(selected: list[dict]) -> dict[str, dict]:
    return {str(row["group"]): row for row in selected}


def selection_from_group_map(group_map: dict[str, dict]) -> list[dict]:
    return [group_map[group] for group in sorted(group_map)]


def local_search_neighbor_candidates(
    rows: list[dict],
    current: list[dict],
    budget_extra: int,
    max_candidates: int,
) -> list[dict]:
    max_candidates = max(1, int(max_candidates))
    current_by_group = selection_by_group(current)
    current_extra = selected_extra_bytes(current)
    current_sig = selection_signature(current)
    selected_groups = set(current_by_group)
    selected_rows = list(current_by_group.values())
    eligible = [
        row
        for row in rows
        if int(row.get("extra_bytes", 0)) > 0 and float(row.get("calib_nll_improvement", 0.0)) > 0.0
    ]
    value = lambda row: float(row.get("calib_nll_improvement", 0.0))
    seen: set[tuple[tuple[str, str], ...]] = {current_sig}
    candidates: list[dict] = []

    def add_candidate(kind: str, desc: str, group_map: dict[str, dict], proxy_delta: float) -> None:
        selected = selection_from_group_map(group_map)
        sig = selection_signature(selected)
        if sig in seen:
            return
        seen.add(sig)
        extra = selected_extra_bytes(selected)
        if extra > budget_extra:
            return
        candidates.append(
            {
                "kind": kind,
                "move": desc,
                "proxy_delta": float(proxy_delta),
                "selected": selected,
                "extra_bytes": int(extra),
            }
        )

    add_pool = sorted(
        [row for row in eligible if str(row["group"]) not in selected_groups],
        key=lambda row: (value(row), -int(row["extra_bytes"]), str(row["group"]), str(row["source"])),
        reverse=True,
    )[: max_candidates * 3]
    replace_pool = sorted(
        [row for row in eligible if str(row["group"]) in selected_groups],
        key=lambda row: (value(row), -int(row["extra_bytes"]), str(row["group"]), str(row["source"])),
        reverse=True,
    )[: max_candidates * 3]
    remove_pool = sorted(
        selected_rows,
        key=lambda row: (value(row), -int(row["extra_bytes"]), str(row["group"]), str(row["source"])),
    )[: max(4, max_candidates)]

    for row in add_pool:
        group = str(row["group"])
        if current_extra + int(row["extra_bytes"]) > budget_extra:
            continue
        group_map = dict(current_by_group)
        group_map[group] = row
        add_candidate("add", f"add {group}->{row['source']}", group_map, value(row))

    for row in replace_pool:
        group = str(row["group"])
        old = current_by_group[group]
        if str(old["source"]) == str(row["source"]):
            continue
        new_extra = current_extra - int(old["extra_bytes"]) + int(row["extra_bytes"])
        if new_extra > budget_extra:
            continue
        group_map = dict(current_by_group)
        group_map[group] = row
        add_candidate(
            "replace",
            f"replace {group}: {old['source']}->{row['source']}",
            group_map,
            value(row) - value(old),
        )

    for old in remove_pool:
        group = str(old["group"])
        group_map = dict(current_by_group)
        del group_map[group]
        add_candidate("drop", f"drop {group}->{old['source']}", group_map, -value(old))

    for old in remove_pool:
        old_group = str(old["group"])
        for row in add_pool:
            new_group = str(row["group"])
            if new_group == old_group:
                continue
            new_extra = current_extra - int(old["extra_bytes"]) + int(row["extra_bytes"])
            if new_extra > budget_extra:
                continue
            group_map = dict(current_by_group)
            del group_map[old_group]
            group_map[new_group] = row
            add_candidate(
                "swap",
                f"swap {old_group}->{old['source']} for {new_group}->{row['source']}",
                group_map,
                value(row) - value(old),
            )

    candidates.sort(key=lambda item: (item["proxy_delta"], item["extra_bytes"]), reverse=True)
    return candidates[:max_candidates]


def evaluate_selection_nll(
    model,
    tokenizer,
    hf,
    readers: dict[str, dict],
    groups: dict[str, list[TensorSpec]],
    args,
    base_source: str,
    selected: list[dict],
    prompts: list[str],
    max_length: int,
) -> dict:
    patch_all_from_source(model, hf, readers, args.layers, base_source, args.group_mode, args.tensor_profile)
    apply_selection(model, hf, readers, groups, selected)
    return evaluate_nll(model, tokenizer, prompts, max_length)


def local_search_repair_selection(
    model,
    tokenizer,
    hf,
    readers: dict[str, dict],
    groups: dict[str, list[TensorSpec]],
    args,
    rows: list[dict],
    start_selected: list[dict],
    base_source: str,
    budget_extra: int,
    local_search_prompts: list[str],
) -> tuple[list[dict], dict]:
    if selected_saved_bytes(start_selected) > 0:
        raise ValueError("local search currently supports promotion selections only")

    base_cache_context = {
        "kind": "local_search",
        "base_source": base_source,
        "budget_extra": int(budget_extra),
        "fitness_prompt_digest": prompt_digest(local_search_prompts),
        "max_length": int(args.calib_max_length),
        "tensor_profile": str(args.tensor_profile),
        "group_mode": str(args.group_mode),
    }
    cache_context = selection_search_context(base_cache_context, args, rows)
    cache_path = args.output_dir / "checkpoints" / "local_search_fitness.jsonl"
    cache = load_selection_search_cache(cache_path, "candidate_index", [cache_context, base_cache_context])

    def evaluate_cached(selected: list[dict], label: str) -> dict:
        sig = selection_signature(selected)
        if sig not in cache:
            candidate_index = max((int(item["candidate_index"]) for item in cache.values()), default=0) + 1
            print(
                f"[c2] local search evaluating {label}: candidate={candidate_index} "
                f"groups={len(selected)} extra={selected_extra_bytes(selected)}",
                flush=True,
            )
            eval_result = evaluate_selection_nll(
                model,
                tokenizer,
                hf,
                readers,
                groups,
                args,
                base_source,
                selected,
                prompts=local_search_prompts,
                max_length=args.calib_max_length,
            )
            cache[sig] = {
                "candidate_index": int(candidate_index),
                "nll": float(eval_result["nll"]),
                "search_nll": float(eval_result["nll"]),
                "extra_bytes": int(selected_extra_bytes(selected)),
                "selected": list(selected),
            }
            append_selection_search_cache(cache_path, cache[sig], "candidate_index", cache_context)
        else:
            print(
                f"[c2] local search checkpoint hit {label}: "
                f"candidate={cache[sig]['candidate_index']} nll={cache[sig]['nll']:.6f}",
                flush=True,
            )
        return cache[sig]

    current = list(start_selected)
    current_eval = evaluate_cached(current, "start")
    current_nll = float(current_eval["nll"])
    start_nll = current_nll
    accepted: list[dict] = []
    rejected_best: list[dict] = []
    evaluated_moves = 0
    min_improvement = float(args.local_search_min_improvement)

    print(
        f"[c2] local search start nll={current_nll:.6f} groups={len(current)} extra={selected_extra_bytes(current)}",
        flush=True,
    )

    for step in range(1, int(args.local_search_steps) + 1):
        candidates = local_search_neighbor_candidates(
            rows,
            current,
            budget_extra,
            int(args.local_search_candidates),
        )
        if not candidates:
            break

        best: dict | None = None
        best_improvement = float("-inf")
        for candidate_idx, candidate in enumerate(candidates, start=1):
            evaluated_moves += 1
            print(
                f"[c2] local search step {step} evaluating "
                f"{candidate_idx}/{len(candidates)}: {candidate['move']}",
                flush=True,
            )
            cand_eval = evaluate_cached(candidate["selected"], f"step {step} {candidate_idx}/{len(candidates)}")
            cand_nll = float(cand_eval["nll"])
            improvement = current_nll - cand_nll
            if improvement > best_improvement:
                best_improvement = improvement
                best = {
                    "step": step,
                    "move": candidate["move"],
                    "kind": candidate["kind"],
                    "proxy_delta": float(candidate["proxy_delta"]),
                    "nll": cand_nll,
                    "nll_improvement": float(improvement),
                    "extra_bytes": int(candidate["extra_bytes"]),
                    "selected": candidate["selected"],
                }

        if best is None:
            break
        print(
            f"[c2] local search step {step} best {best['move']} "
            f"nll={best['nll']:.6f} improvement={best['nll_improvement']:.6f}",
            flush=True,
        )
        if best["nll_improvement"] < min_improvement:
            rejected_best.append({k: v for k, v in best.items() if k != "selected"})
            break
        current = list(best["selected"])
        current_nll = float(best["nll"])
        accepted.append({k: v for k, v in best.items() if k != "selected"})

    report = {
        "start_nll": float(start_nll),
        "final_nll": float(current_nll),
        "total_nll_improvement": float(start_nll - current_nll),
        "steps_requested": int(args.local_search_steps),
        "candidate_limit": int(args.local_search_candidates),
        "min_improvement": float(min_improvement),
        "evaluated_moves": int(evaluated_moves),
        "accepted_moves": accepted,
        "rejected_best_moves": rejected_best,
    }
    return current, report


def repair_selection_to_budget(
    selected: list[dict],
    budget_extra: int,
    value_key: str = "calib_nll_improvement",
) -> list[dict]:
    group_map = selection_by_group(selected)
    while selected_extra_bytes(list(group_map.values())) > budget_extra and group_map:
        worst_group = min(
            group_map,
            key=lambda group: (
                float(group_map[group].get(value_key, 0.0)) / max(1, int(group_map[group].get("extra_bytes", 0))),
                float(group_map[group].get(value_key, 0.0)),
            ),
        )
        del group_map[worst_group]
    return selection_from_group_map(group_map)


def random_promotion_selection(rows: list[dict], budget_extra: int, rng: random.Random) -> list[dict]:
    eligible_by_group: dict[str, list[dict]] = {}
    direct_mode = any(bool(row.get("direct_search_only", False)) for row in rows)
    for row in rows:
        if int(row.get("extra_bytes", 0)) <= 0 or float(row.get("calib_nll_improvement", 0.0)) <= 0.0:
            continue
        if int(row["extra_bytes"]) > budget_extra:
            continue
        eligible_by_group.setdefault(str(row["group"]), []).append(row)
    for group_rows in eligible_by_group.values():
        group_rows.sort(
            key=lambda row: (
                float(row.get("calib_nll_improvement", 0.0)),
                -int(row.get("extra_bytes", 0)),
                str(row.get("source", "")),
            ),
            reverse=True,
        )

    selected: list[dict] = []
    used = 0
    groups = list(eligible_by_group)
    rng.shuffle(groups)
    for group in groups:
        choices = eligible_by_group[group]
        if direct_mode:
            row = rng.choice(choices)
        else:
            row = rng.choice(choices[: min(4, len(choices))])
        extra = int(row["extra_bytes"])
        if used + extra > budget_extra:
            continue
        selected.append(row)
        used += extra
    return selected


def crossover_selections(
    parent_a: list[dict],
    parent_b: list[dict],
    budget_extra: int,
    rng: random.Random,
) -> list[dict]:
    a_by_group = selection_by_group(parent_a)
    b_by_group = selection_by_group(parent_b)
    child: dict[str, dict] = {}
    for group in sorted(set(a_by_group) | set(b_by_group)):
        options = [row for row in (a_by_group.get(group), b_by_group.get(group)) if row is not None]
        if not options:
            continue
        if len(options) == 1:
            chosen = options[0]
        else:
            chosen = rng.choice(options)
        child[group] = chosen
    return repair_selection_to_budget(selection_from_group_map(child), budget_extra)


def mutate_selection(
    rows: list[dict],
    selected: list[dict],
    budget_extra: int,
    rng: random.Random,
    mutation_rate: float,
) -> list[dict]:
    eligible = [
        row
        for row in rows
        if int(row.get("extra_bytes", 0)) > 0
        and float(row.get("calib_nll_improvement", 0.0)) > 0.0
        and int(row["extra_bytes"]) <= budget_extra
    ]
    if not eligible:
        return selected
    by_group: dict[str, list[dict]] = {}
    for row in eligible:
        by_group.setdefault(str(row["group"]), []).append(row)
    group_map = selection_by_group(selected)
    mutation_count = max(1, int(round(max(1, len(group_map)) * max(0.0, mutation_rate))))

    for _ in range(mutation_count):
        possible_ops = ["add", "replace"]
        if group_map:
            possible_ops.extend(["drop", "swap"])
        op = rng.choice(possible_ops)

        if op == "add":
            groups = [group for group in by_group if group not in group_map]
            if not groups:
                continue
            group = rng.choice(groups)
            group_map[group] = rng.choice(by_group[group])
        elif op == "replace":
            groups = list(group_map) or list(by_group)
            group = rng.choice(groups)
            options = [
                row
                for row in by_group.get(group, [])
                if str(row.get("source")) != str(group_map.get(group, {}).get("source"))
            ]
            if options:
                group_map[group] = rng.choice(options)
        elif op == "drop" and group_map:
            del group_map[rng.choice(list(group_map))]
        elif op == "swap" and group_map:
            old_group = rng.choice(list(group_map))
            del group_map[old_group]
            groups = [group for group in by_group if group not in group_map]
            if groups:
                new_group = rng.choice(groups)
                group_map[new_group] = rng.choice(by_group[new_group])

        group_map = selection_by_group(repair_selection_to_budget(selection_from_group_map(group_map), budget_extra))

    return selection_from_group_map(group_map)


def tournament_select(evaluated: list[dict], rng: random.Random, k: int = 3) -> list[dict]:
    contenders = [rng.choice(evaluated) for _ in range(min(k, len(evaluated)))]
    return min(contenders, key=lambda item: (item["nll"], -item["extra_bytes"]))["selected"]


def generation_fitness_summary(evaluated: list[dict]) -> dict:
    nlls = sorted(float(item["nll"]) for item in evaluated)
    if not nlls:
        return {"best_nll": None, "median_nll": None, "worst_nll": None}
    midpoint = len(nlls) // 2
    if len(nlls) % 2:
        median = nlls[midpoint]
    else:
        median = 0.5 * (nlls[midpoint - 1] + nlls[midpoint])
    return {
        "best_nll": nlls[0],
        "median_nll": float(median),
        "worst_nll": nlls[-1],
    }


def split_genetic_search_prompts(prompts: list[str], validation_count: int) -> tuple[list[str], list[str]]:
    validation_count = max(0, int(validation_count))
    if validation_count <= 0 or len(prompts) <= 1:
        return list(prompts), []
    validation_count = min(validation_count, len(prompts) - 1)
    return list(prompts[:-validation_count]), list(prompts[-validation_count:])


def genetic_search_selection(
    model,
    tokenizer,
    hf,
    readers: dict[str, dict],
    groups: dict[str, list[TensorSpec]],
    args,
    rows: list[dict],
    seed_selections: dict[str, list[dict]],
    base_source: str,
    budget_extra: int,
    prompts: list[str],
    rng: random.Random,
) -> tuple[list[dict], dict]:
    population_size = max(2, int(args.genetic_search_population))
    generations = max(0, int(args.genetic_search_generations))
    elite_count = max(1, min(population_size, int(args.genetic_search_elite)))
    mutation_rate = max(0.0, float(args.genetic_search_mutation_rate))
    search_prompts, validation_prompts = split_genetic_search_prompts(
        prompts,
        int(getattr(args, "genetic_search_validation_prompts", 0) or 0),
    )
    rerank_top_k = max(0, int(getattr(args, "genetic_search_rerank_top_k", 0) or 0))
    if validation_prompts and rerank_top_k <= 0:
        rerank_top_k = min(population_size, 8)
    base_cache_context = {
        "kind": "genetic",
        "base_source": base_source,
        "budget_extra": int(budget_extra),
        "direct_search": bool(getattr(args, "genetic_search_direct", False)),
        "fitness_prompt_digest": prompt_digest(search_prompts),
        "validation_prompt_digest": prompt_digest(validation_prompts),
        "max_length": int(args.calib_max_length),
        "tensor_profile": str(args.tensor_profile),
    }
    cache_context = selection_search_context(base_cache_context, args, rows)
    cache_path = args.output_dir / "checkpoints" / "genetic_search_fitness.jsonl"
    cache = load_selection_search_cache(cache_path, "genome_index", [cache_context, base_cache_context])

    print(
        f"[c2] genetic search fitness prompts={len(search_prompts)} "
        f"validation_prompts={len(validation_prompts)} rerank_top_k={rerank_top_k}",
        flush=True,
    )

    def normalize(selected: list[dict]) -> list[dict]:
        return repair_selection_to_budget(selected, budget_extra)

    def add_unique(population: list[list[dict]], selected: list[dict]) -> None:
        normalized = normalize(selected)
        sig = selection_signature(normalized)
        if sig in {selection_signature(item) for item in population}:
            return
        population.append(normalized)

    def evaluate(selected: list[dict]) -> dict:
        sig = selection_signature(selected)
        if sig not in cache:
            genome_index = max((int(item["genome_index"]) for item in cache.values()), default=0) + 1
            print(
                f"[c2] genetic search evaluating genome {genome_index}: "
                f"groups={len(selected)} extra={selected_extra_bytes(selected)}",
                flush=True,
            )
            eval_result = evaluate_selection_nll(
                model,
                tokenizer,
                hf,
                readers,
                groups,
                args,
                base_source,
                selected,
                prompts=search_prompts,
                max_length=args.calib_max_length,
            )
            cache[sig] = {
                "genome_index": int(genome_index),
                "nll": float(eval_result["nll"]),
                "search_nll": float(eval_result["nll"]),
                "extra_bytes": int(selected_extra_bytes(selected)),
                "selected": selected,
            }
            append_selection_search_cache(cache_path, cache[sig], "genome_index", cache_context)
            print(
                f"[c2] genetic search genome {genome_index} nll={cache[sig]['nll']:.6f}",
                flush=True,
            )
        return cache[sig]

    def evaluate_validation(finalists: list[dict]) -> list[dict]:
        if not validation_prompts or not finalists:
            return finalists
        for idx, item in enumerate(finalists, start=1):
            if "validation_nll" in item:
                continue
            print(
                f"[c2] genetic validation evaluating finalist {idx}/{len(finalists)}: "
                f"genome={item['genome_index']} search_nll={item['search_nll']:.6f} "
                f"groups={len(item['selected'])} extra={item['extra_bytes']}",
                flush=True,
            )
            eval_result = evaluate_selection_nll(
                model,
                tokenizer,
                hf,
                readers,
                groups,
                args,
                base_source,
                item["selected"],
                prompts=validation_prompts,
                max_length=args.calib_max_length,
            )
            item["validation_nll"] = float(eval_result["nll"])
            append_selection_search_cache(cache_path, item, "genome_index", cache_context)
            print(
                f"[c2] genetic validation finalist genome={item['genome_index']} "
                f"validation_nll={item['validation_nll']:.6f}",
                flush=True,
            )
        return finalists

    population: list[list[dict]] = []
    for selected in seed_selections.values():
        add_unique(population, selected)
    attempts = 0
    while len(population) < population_size and attempts < population_size * 20:
        attempts += 1
        add_unique(population, random_promotion_selection(rows, budget_extra, rng))
        if len(population) < population_size:
            seed = rng.choice(population) if population else []
            add_unique(population, mutate_selection(rows, seed, budget_extra, rng, mutation_rate or 0.25))
    if not population:
        population.append([])
    while len(population) < population_size:
        population.append(list(rng.choice(population)))

    history: list[dict] = []
    best: dict | None = None
    evaluated = [evaluate(selected) for selected in population]
    evaluated.sort(key=lambda item: (item["nll"], -item["extra_bytes"]))
    best = evaluated[0]
    fitness = generation_fitness_summary(evaluated)
    history.append(
        {
            "generation": 0,
            "best_nll": float(best["nll"]),
            "generation_best_nll": float(fitness["best_nll"]),
            "generation_median_nll": float(fitness["median_nll"]),
            "generation_worst_nll": float(fitness["worst_nll"]),
            "best_extra_bytes": int(best["extra_bytes"]),
            "unique_evaluated": len(cache),
        }
    )
    print(
        f"[c2] genetic search generation 0 fitness "
        f"best={fitness['best_nll']:.6f} median={fitness['median_nll']:.6f} "
        f"worst={fitness['worst_nll']:.6f} global_best={best['nll']:.6f} "
        f"extra={best['extra_bytes']}",
        flush=True,
    )

    for generation in range(1, generations + 1):
        next_population = [item["selected"] for item in evaluated[:elite_count]]
        attempts = 0
        while len(next_population) < population_size and attempts < population_size * 20:
            attempts += 1
            parent_a = tournament_select(evaluated, rng)
            parent_b = tournament_select(evaluated, rng)
            child = crossover_selections(parent_a, parent_b, budget_extra, rng)
            child = mutate_selection(rows, child, budget_extra, rng, mutation_rate)
            add_unique(next_population, child)
            if len(next_population) < population_size and len({selection_signature(item) for item in next_population}) == len(next_population):
                add_unique(next_population, random_promotion_selection(rows, budget_extra, rng))
        while len(next_population) < population_size:
            next_population.append(list(rng.choice(next_population)))

        evaluated = [evaluate(selected) for selected in next_population]
        evaluated.sort(key=lambda item: (item["nll"], -item["extra_bytes"]))
        if best is None or evaluated[0]["nll"] < best["nll"]:
            best = evaluated[0]
        fitness = generation_fitness_summary(evaluated)
        history.append(
            {
                "generation": generation,
                "best_nll": float(best["nll"]),
                "generation_best_nll": float(fitness["best_nll"]),
                "generation_median_nll": float(fitness["median_nll"]),
                "generation_worst_nll": float(fitness["worst_nll"]),
                "best_extra_bytes": int(best["extra_bytes"]),
                "unique_evaluated": len(cache),
            }
        )
        print(
            f"[c2] genetic search generation {generation} fitness "
            f"best={fitness['best_nll']:.6f} median={fitness['median_nll']:.6f} "
            f"worst={fitness['worst_nll']:.6f} global_best={best['nll']:.6f} "
            f"unique={len(cache)}",
            flush=True,
        )

    assert best is not None
    search_best = best
    validation_finalists: list[dict] = []
    if validation_prompts:
        finalist_count = min(max(1, rerank_top_k), len(cache))
        validation_finalists = sorted(
            cache.values(),
            key=lambda item: (item["search_nll"], -item["extra_bytes"]),
        )[:finalist_count]
        evaluate_validation(validation_finalists)
        best = min(
            validation_finalists,
            key=lambda item: (
                float(item.get("validation_nll", float("inf"))),
                float(item["search_nll"]),
                -int(item["extra_bytes"]),
            ),
        )
        print(
            f"[c2] genetic validation selected genome={best['genome_index']} "
            f"validation_nll={best['validation_nll']:.6f} "
            f"search_nll={best['search_nll']:.6f} extra={best['extra_bytes']}",
            flush=True,
        )

    report = {
        "final_nll": float(best["search_nll"]),
        "final_search_nll": float(best["search_nll"]),
        "final_validation_nll": (
            float(best["validation_nll"]) if "validation_nll" in best else None
        ),
        "search_best_nll": float(search_best["search_nll"]),
        "search_best_extra_bytes": int(search_best["extra_bytes"]),
        "final_extra_bytes": int(best["extra_bytes"]),
        "generations": generations,
        "population_size": population_size,
        "elite_count": elite_count,
        "mutation_rate": mutation_rate,
        "direct_search": bool(getattr(args, "genetic_search_direct", False)),
        "fitness_prompt_count": int(len(search_prompts)),
        "validation_prompt_count": int(len(validation_prompts)),
        "validation_rerank_top_k": int(rerank_top_k),
        "selection_metric": "validation_nll" if validation_prompts else "search_nll",
        "evaluated_genomes": len(cache),
        "validation_finalists": [
            {
                "genome_index": int(item["genome_index"]),
                "search_nll": float(item["search_nll"]),
                "validation_nll": (
                    float(item["validation_nll"]) if "validation_nll" in item else None
                ),
                "extra_bytes": int(item["extra_bytes"]),
                "group_count": int(len(item["selected"])),
            }
            for item in validation_finalists
        ],
        "history": history,
    }
    return list(best["selected"]), report


def anneal_search_selection(
    model,
    tokenizer,
    hf,
    readers: dict[str, dict],
    groups: dict[str, list[TensorSpec]],
    args,
    rows: list[dict],
    seed_selections: dict[str, list[dict]],
    base_source: str,
    budget_extra: int,
    prompts: list[str],
    rng: random.Random,
) -> tuple[list[dict], dict]:
    steps = max(0, int(args.anneal_search_steps))
    mutation_rate = max(0.0, float(args.anneal_search_mutation_rate))
    initial_temp = max(1e-9, float(args.anneal_search_initial_temp))
    final_temp = max(1e-9, float(args.anneal_search_final_temp))
    search_prompts, validation_prompts = split_genetic_search_prompts(
        prompts,
        int(getattr(args, "anneal_search_validation_prompts", 0) or 0),
    )
    rerank_top_k = max(0, int(getattr(args, "anneal_search_rerank_top_k", 0) or 0))
    if validation_prompts and rerank_top_k <= 0:
        rerank_top_k = min(8, max(1, steps))
    base_cache_context = {
        "kind": "anneal",
        "base_source": base_source,
        "budget_extra": int(budget_extra),
        "direct_search": bool(getattr(args, "anneal_search_direct", False)),
        "fitness_prompt_digest": prompt_digest(search_prompts),
        "validation_prompt_digest": prompt_digest(validation_prompts),
        "max_length": int(args.calib_max_length),
        "tensor_profile": str(args.tensor_profile),
    }
    cache_context = selection_search_context(base_cache_context, args, rows)
    cache_path = args.output_dir / "checkpoints" / "anneal_search_fitness.jsonl"
    cache = load_selection_search_cache(cache_path, "state_index", [cache_context, base_cache_context])

    print(
        f"[c2] anneal search fitness prompts={len(search_prompts)} "
        f"validation_prompts={len(validation_prompts)} rerank_top_k={rerank_top_k} "
        f"steps={steps} temp={initial_temp:.6g}->{final_temp:.6g}",
        flush=True,
    )

    def normalize(selected: list[dict]) -> list[dict]:
        return repair_selection_to_budget(selected, budget_extra)

    def evaluate(selected: list[dict]) -> dict:
        selected = normalize(selected)
        sig = selection_signature(selected)
        if sig not in cache:
            state_index = max((int(item["state_index"]) for item in cache.values()), default=0) + 1
            print(
                f"[c2] anneal search evaluating state {state_index}: "
                f"groups={len(selected)} extra={selected_extra_bytes(selected)}",
                flush=True,
            )
            eval_result = evaluate_selection_nll(
                model,
                tokenizer,
                hf,
                readers,
                groups,
                args,
                base_source,
                selected,
                prompts=search_prompts,
                max_length=args.calib_max_length,
            )
            cache[sig] = {
                "state_index": int(state_index),
                "nll": float(eval_result["nll"]),
                "search_nll": float(eval_result["nll"]),
                "extra_bytes": int(selected_extra_bytes(selected)),
                "selected": selected,
            }
            append_selection_search_cache(cache_path, cache[sig], "state_index", cache_context)
            print(
                f"[c2] anneal search state {state_index} nll={cache[sig]['nll']:.6f}",
                flush=True,
            )
        return cache[sig]

    seed_pool: list[list[dict]] = [normalize(selected) for selected in seed_selections.values()]
    attempts = 0
    while len(seed_pool) < 4 and attempts < 40:
        attempts += 1
        seed_pool.append(normalize(random_promotion_selection(rows, budget_extra, rng)))
    if not seed_pool:
        seed_pool = [[]]

    evaluated_seeds = [evaluate(selected) for selected in seed_pool]
    evaluated_seeds.sort(key=lambda item: (item["nll"], -item["extra_bytes"]))
    current = evaluated_seeds[0]
    best = evaluated_seeds[0]
    history: list[dict] = [
        {
            "step": 0,
            "temperature": float(initial_temp),
            "current_nll": float(current["nll"]),
            "best_nll": float(best["nll"]),
            "current_extra_bytes": int(current["extra_bytes"]),
            "best_extra_bytes": int(best["extra_bytes"]),
            "accepted": True,
            "accepted_worse": False,
            "unique_evaluated": len(cache),
        }
    ]
    accepted = 0
    accepted_worse = 0

    for step in range(1, steps + 1):
        if steps <= 1:
            temperature = final_temp
        else:
            progress = (step - 1) / max(1, steps - 1)
            temperature = initial_temp * ((final_temp / initial_temp) ** progress)
        candidate_selected = mutate_selection(
            rows,
            current["selected"],
            budget_extra,
            rng,
            mutation_rate,
        )
        candidate = evaluate(candidate_selected)
        delta = float(candidate["nll"] - current["nll"])
        accept = delta <= 0.0
        worse = False
        if not accept:
            threshold = math.exp(-delta / max(temperature, 1e-12))
            accept = rng.random() < threshold
            worse = accept
        if accept:
            current = candidate
            accepted += 1
            if worse:
                accepted_worse += 1
        if candidate["nll"] < best["nll"]:
            best = candidate
        if step == 1 or step == steps or step % max(1, steps // 10) == 0:
            print(
                f"[c2] anneal search step {step}/{steps} temp={temperature:.6g} "
                f"current={current['nll']:.6f} candidate={candidate['nll']:.6f} "
                f"best={best['nll']:.6f} accepted={accept} unique={len(cache)}",
                flush=True,
            )
            history.append(
                {
                    "step": int(step),
                    "temperature": float(temperature),
                    "current_nll": float(current["nll"]),
                    "candidate_nll": float(candidate["nll"]),
                    "best_nll": float(best["nll"]),
                    "current_extra_bytes": int(current["extra_bytes"]),
                    "candidate_extra_bytes": int(candidate["extra_bytes"]),
                    "best_extra_bytes": int(best["extra_bytes"]),
                    "accepted": bool(accept),
                    "accepted_worse": bool(worse),
                    "unique_evaluated": len(cache),
                }
            )

    search_best = best
    validation_finalists: list[dict] = []
    if validation_prompts:
        finalist_count = min(max(1, rerank_top_k), len(cache))
        validation_finalists = sorted(
            cache.values(),
            key=lambda item: (item["search_nll"], -item["extra_bytes"]),
        )[:finalist_count]
        for idx, item in enumerate(validation_finalists, start=1):
            print(
                f"[c2] anneal validation evaluating finalist {idx}/{len(validation_finalists)}: "
                f"state={item['state_index']} search_nll={item['search_nll']:.6f} "
                f"groups={len(item['selected'])} extra={item['extra_bytes']}",
                flush=True,
            )
            eval_result = evaluate_selection_nll(
                model,
                tokenizer,
                hf,
                readers,
                groups,
                args,
                base_source,
                item["selected"],
                prompts=validation_prompts,
                max_length=args.calib_max_length,
            )
            item["validation_nll"] = float(eval_result["nll"])
            append_selection_search_cache(cache_path, item, "state_index", cache_context)
            print(
                f"[c2] anneal validation finalist state={item['state_index']} "
                f"validation_nll={item['validation_nll']:.6f}",
                flush=True,
            )
        best = min(
            validation_finalists,
            key=lambda item: (
                float(item.get("validation_nll", float("inf"))),
                float(item["search_nll"]),
                -int(item["extra_bytes"]),
            ),
        )
        print(
            f"[c2] anneal validation selected state={best['state_index']} "
            f"validation_nll={best['validation_nll']:.6f} "
            f"search_nll={best['search_nll']:.6f} extra={best['extra_bytes']}",
            flush=True,
        )

    report = {
        "final_nll": float(best["search_nll"]),
        "final_search_nll": float(best["search_nll"]),
        "final_validation_nll": (
            float(best["validation_nll"]) if "validation_nll" in best else None
        ),
        "search_best_nll": float(search_best["search_nll"]),
        "search_best_extra_bytes": int(search_best["extra_bytes"]),
        "final_extra_bytes": int(best["extra_bytes"]),
        "steps": int(steps),
        "mutation_rate": float(mutation_rate),
        "initial_temperature": float(initial_temp),
        "final_temperature": float(final_temp),
        "direct_search": bool(getattr(args, "anneal_search_direct", False)),
        "accepted_moves": int(accepted),
        "accepted_worse_moves": int(accepted_worse),
        "fitness_prompt_count": int(len(search_prompts)),
        "validation_prompt_count": int(len(validation_prompts)),
        "validation_rerank_top_k": int(rerank_top_k),
        "selection_metric": "validation_nll" if validation_prompts else "search_nll",
        "evaluated_states": len(cache),
        "validation_finalists": [
            {
                "state_index": int(item["state_index"]),
                "search_nll": float(item["search_nll"]),
                "validation_nll": (
                    float(item["validation_nll"]) if "validation_nll" in item else None
                ),
                "extra_bytes": int(item["extra_bytes"]),
                "group_count": int(len(item["selected"])),
            }
            for item in validation_finalists
        ],
        "history": history,
    }
    return list(best["selected"]), report


def add_rank_blend_scores(rows: list[dict], output_key: str, weights: dict[str, float]) -> None:
    eligible = [row for row in rows if row["extra_bytes"] > 0]
    if not eligible:
        return
    for key, weight in weights.items():
        ranked = sorted(eligible, key=lambda row: row.get(key, float("-inf")))
        denom = max(1, len(ranked) - 1)
        for rank, row in enumerate(ranked):
            row[output_key] = float(row.get(output_key, 0.0) + weight * (rank / denom))


def select_random(rows: list[dict], budget_extra: int, rng: random.Random, trials: int = 64) -> list[dict]:
    eligible = [row for row in rows if row["extra_bytes"] > 0]
    best: list[dict] = []
    best_used = -1
    for _ in range(trials):
        trial = eligible[:]
        rng.shuffle(trial)
        selected = []
        seen_groups: set[str] = set()
        used = 0
        for row in trial:
            if row["group"] in seen_groups:
                continue
            if used + row["extra_bytes"] > budget_extra:
                continue
            selected.append(row)
            seen_groups.add(row["group"])
            used += row["extra_bytes"]
        if used > best_used:
            best = selected
            best_used = used
    return best


def prune_demotion_states(
    states: list[tuple[int, float, tuple[dict, ...]]],
    max_states: int,
) -> list[tuple[int, float, tuple[dict, ...]]]:
    max_states = max(1, int(max_states))
    best_by_saved: dict[int, tuple[float, tuple[dict, ...]]] = {}
    for saved, loss, selected in states:
        previous = best_by_saved.get(saved)
        if previous is None or loss < previous[0]:
            best_by_saved[saved] = (loss, selected)

    frontier: list[tuple[int, float, tuple[dict, ...]]] = []
    best_loss = float("inf")
    for saved in sorted(best_by_saved, reverse=True):
        loss, selected = best_by_saved[saved]
        if loss >= best_loss - 1e-12:
            continue
        frontier.append((saved, loss, selected))
        best_loss = loss
    frontier.sort(key=lambda item: item[0])

    if len(frontier) <= max_states:
        return frontier

    keep: dict[int, tuple[int, float, tuple[dict, ...]]] = {}
    loss_keep_count = max(1, max_states // 2)
    for state in sorted(frontier, key=lambda item: (item[1], -item[0]))[:loss_keep_count]:
        keep[state[0]] = state

    spread_count = max_states - len(keep)
    if spread_count > 0:
        denom = max(1, spread_count - 1)
        last_idx = len(frontier) - 1
        for idx in range(spread_count):
            state = frontier[round(idx * last_idx / denom)]
            previous = keep.get(state[0])
            if previous is None or state[1] < previous[1]:
                keep[state[0]] = state

    limited = sorted(keep.values(), key=lambda item: item[0])
    if len(limited) > max_states:
        limited = sorted(limited, key=lambda item: (item[1], -item[0]))[:max_states]
        limited.sort(key=lambda item: item[0])
    return limited


def select_demotions_min_loss(
    rows: list[dict],
    required_saving: int,
    max_states: int = 50_000,
) -> list[dict]:
    required_saving = max(0, int(required_saving))
    if required_saving <= 0:
        return []

    options_by_group: dict[str, list[dict]] = {}
    for row in rows:
        saved = int(row.get("saved_bytes", 0))
        if saved <= 0:
            continue
        options_by_group.setdefault(row["group"], []).append(row)

    states: list[tuple[int, float, tuple[dict, ...]]] = [(0, 0.0, ())]
    for group in sorted(options_by_group):
        options = sorted(
            options_by_group[group],
            key=lambda row: (
                float(row.get("calib_nll_loss", 0.0)),
                -int(row.get("saved_bytes", 0)),
                row["source"],
            ),
        )
        next_states = states[:]
        for saved, loss, selected in states:
            for row in options:
                next_states.append(
                    (
                        saved + int(row["saved_bytes"]),
                        loss + float(row["calib_nll_loss"]),
                        selected + (row,),
                    )
                )
        states = prune_demotion_states(next_states, max_states)

    feasible = [state for state in states if state[0] >= required_saving]
    if feasible:
        best = min(feasible, key=lambda item: (item[1], item[0]))
        return list(best[2])
    best = max(states, key=lambda item: (item[0], -item[1]))
    return list(best[2])


def select_demotions_greedy(rows: list[dict], required_saving: int) -> list[dict]:
    required_saving = max(0, int(required_saving))
    if required_saving <= 0:
        return []
    selected: list[dict] = []
    seen_groups: set[str] = set()
    saved = 0
    ranked = sorted(
        [row for row in rows if int(row.get("saved_bytes", 0)) > 0],
        key=lambda row: (
            float(row.get("demotion_loss_per_mbyte", 0.0)),
            -int(row.get("saved_bytes", 0)),
            row["source"],
        ),
    )
    for row in ranked:
        if row["group"] in seen_groups:
            continue
        selected.append(row)
        seen_groups.add(row["group"])
        saved += int(row["saved_bytes"])
        if saved >= required_saving:
            break
    return selected


def build_demotion_selection(
    selector: str,
    rows: list[dict],
    required_saving: int,
    max_states: int,
) -> list[dict]:
    selector = selector.strip()
    if selector in {"reverse_knapsack", "demotion_knapsack", "knapsack"}:
        return select_demotions_min_loss(rows, required_saving, max_states)
    if selector in {"reverse_greedy", "demotion_greedy", "greedy"}:
        return select_demotions_greedy(rows, required_saving)
    raise ValueError(f"unknown demotion selector {selector!r}")


def demotion_variant_name(selector: str, payload_bpw: float) -> str:
    selector = selector.strip()
    if selector in {"reverse_knapsack", "demotion_knapsack", "knapsack"}:
        label = "reverse_knapsack"
    elif selector in {"reverse_greedy", "demotion_greedy", "greedy"}:
        label = "reverse_greedy"
    else:
        raise ValueError(f"unknown demotion selector {selector!r}")
    return f"c2_{label}_bpw_{bpw_tag(payload_bpw)}_mixed"


def select_random_demotions(
    rows: list[dict],
    required_saving: int,
    rng: random.Random,
    trials: int = 64,
) -> list[dict]:
    eligible = [row for row in rows if int(row.get("saved_bytes", 0)) > 0]
    best: list[dict] = []
    best_key = (-1, float("-inf"))
    for _ in range(trials):
        trial = eligible[:]
        rng.shuffle(trial)
        selected = []
        seen_groups: set[str] = set()
        saved = 0
        loss = 0.0
        for row in trial:
            if row["group"] in seen_groups:
                continue
            selected.append(row)
            seen_groups.add(row["group"])
            saved += int(row["saved_bytes"])
            loss += float(row.get("calib_nll_loss", 0.0))
            if saved >= required_saving:
                break
        if saved >= required_saving:
            key = (1, -loss)
        else:
            key = (0, saved)
        if key > best_key:
            best = selected
            best_key = key
    return best


def selected_extra_bytes(selected: list[dict]) -> int:
    return int(sum(row.get("extra_bytes", 0) for row in selected))


def selected_saved_bytes(selected: list[dict]) -> int:
    return int(sum(row.get("saved_bytes", 0) for row in selected))


def selected_payload_bytes(base_payload: int, selected: list[dict]) -> int:
    return int(base_payload + selected_extra_bytes(selected) - selected_saved_bytes(selected))


def apply_selection(model, hf, readers: dict[str, dict], groups: dict[str, list[TensorSpec]], selected: list[dict]) -> None:
    for row in selected:
        patch_group(model, hf, readers, groups, row["group"], row["source"])


def compact_selection(selected: list[dict]) -> list[dict]:
    return [
        {
            "group": row["group"],
            "source": row["source"],
            "extra_bytes": int(row.get("extra_bytes", 0)),
            "saved_bytes": int(row.get("saved_bytes", 0)),
            "base_source": row.get("base_source"),
            "calib_nll_improvement": float(row.get("calib_nll_improvement", 0.0)),
            "calib_nll_loss": float(row.get("calib_nll_loss", 0.0)),
            "calib_score_per_mbyte": float(row.get("calib_score_per_mbyte", 0.0)),
            "weight_sse_delta": float(row.get("weight_sse_delta", 0.0)),
            "weight_score_per_mbyte": float(row.get("weight_score_per_mbyte", 0.0)),
            "calib_weight_rank_blend": float(row.get("calib_weight_rank_blend", 0.0)),
            "calib_knapsack_value": float(row.get("calib_nll_improvement", 0.0)),
            "demotion_loss_per_mbyte": float(row.get("demotion_loss_per_mbyte", 0.0)),
        }
        for row in selected
    ]


def materialize_selection_rows(seed_selection: list[dict], rows: list[dict], label: str) -> list[dict]:
    by_key = {
        (str(row["group"]), str(row["source"])): row
        for row in rows
    }
    selected: list[dict] = []
    missing: list[str] = []
    for item in seed_selection:
        key = (str(item["group"]), str(item["source"]))
        row = by_key.get(key)
        if row is None:
            missing.append(f"{key[0]}->{key[1]}")
        else:
            selected.append(row)
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise ValueError(f"seed selection {label!r} has groups not available in this run: {preview}{suffix}")
    return selected


def load_seed_selection_result(path: Path, variant: str, rows: list[dict]) -> tuple[str, list[dict]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    selections = result.get("selections") or {}
    selected_variant = variant or result.get("args", {}).get("candidate_variant") or ""
    if not selected_variant:
        selected_variant = next(iter(selections), "")
    if selected_variant not in selections:
        raise ValueError(
            f"seed selection variant {selected_variant!r} not found in {path}; "
            f"available variants: {sorted(selections)}"
        )
    selected = materialize_selection_rows(selections[selected_variant], rows, selected_variant)
    return selected_variant, selected


def add_byte_fields(row: dict, payload_bytes: int, total_weights: int, fp_nll: float) -> dict:
    row["payload_bytes"] = int(payload_bytes)
    row["payload_bpw"] = float(payload_bytes * 8 / total_weights)
    row["delta_nll_vs_fp16"] = float(row["nll"] - fp_nll)
    return row


def decide(result: dict) -> tuple[str, str, str]:
    variants = result["variants"]
    candidate_name = result["args"].get("candidate_variant", "c2_calib_greedy_mixed")
    candidate = variants[candidate_name]
    target = variants[result["args"]["target_source"]]
    high_sources = result["args"].get("high_sources", [])
    high = variants[high_sources[-1]] if high_sources else None
    target_margin = target["nll"] - candidate["nll"]
    candidate["nll_improvement_vs_target"] = float(target_margin)
    candidate["payload_bytes_vs_target"] = int(candidate["payload_bytes"] - target["payload_bytes"])
    high_saving_bpw = None
    if high is not None:
        high_saving_bpw = high["payload_bpw"] - candidate["payload_bpw"]
        candidate["payload_bpw_saving_vs_high"] = float(high_saving_bpw)
        candidate["nll_delta_vs_high"] = float(candidate["nll"] - high["nll"])

    random_control = variants.get("c2_random_same_budget")
    weight_control = variants.get("c2_weight_mse_mixed")
    q3ks_control = variants.get("q3_k_s")
    if random_control is not None:
        candidate["nll_improvement_vs_random_control"] = float(random_control["nll"] - candidate["nll"])
    if weight_control is not None:
        candidate["nll_improvement_vs_weight_mse_control"] = float(weight_control["nll"] - candidate["nll"])
    if q3ks_control is not None:
        candidate["nll_improvement_vs_q3_k_s"] = float(q3ks_control["nll"] - candidate["nll"])
        candidate["payload_bytes_vs_q3_k_s"] = int(candidate["payload_bytes"] - q3ks_control["payload_bytes"])

    selection_saved = result.get("byte_accounting", {}).get("selection_saved_bytes", {}).get(candidate_name, 0)
    if selection_saved > 0:
        loss_vs_target = float(candidate["nll"] - target["nll"])
        saved_vs_target = int(target["payload_bytes"] - candidate["payload_bytes"])
        candidate["nll_loss_vs_target"] = loss_vs_target
        candidate["payload_bytes_saved_vs_target"] = saved_vs_target
        max_loss = float(result["args"].get("max_shrink_nll_loss", 0.05))
        if saved_vs_target > 0 and loss_vs_target <= max_loss:
            return (
                "GO",
                "GO: reverse-demotion allocation saves tensor payload bytes while staying within the shrink-loss budget.",
                "Evaluate the selected frontier point on public held-out datasets and materialize the smallest passing artifact.",
            )
        if saved_vs_target > 0:
            return (
                "GRAY",
                "GRAY: reverse-demotion allocation saves bytes, but the calibration loss exceeds the shrink-loss budget.",
                "Inspect neighboring budget points and cross-corpus public eval before promoting this operating point.",
            )
        return (
            "NO-GO",
            "NO-GO: reverse-demotion allocation did not reduce tensor payload bytes.",
            "Use a lower payload-bpw budget or add cheaper demotion sources.",
        )

    if candidate["payload_bytes"] <= target["payload_bytes"] and target_margin >= 0.05:
        if q3ks_control is not None and q3ks_control["nll"] - candidate["nll"] < 0.01:
            return (
                "GRAY",
                "GRAY: mixed production-format allocation beats the target baseline, but does not clearly beat Q3_K_S.",
                "Treat Q3_K_S as the blocking production control before any release claim.",
            )
        if random_control is not None and random_control["nll"] - candidate["nll"] < 0.01:
            return (
                "GRAY",
                "GRAY: calibration-selected mixed production tensors beat the target baseline, but random same-budget allocation is too close.",
                "Run more seeds and a page-level control before promotion.",
            )
        return (
            "GO",
            "GO: mixed production-format allocation beats the target uniform production baseline at or below its tensor payload bytes.",
            "Escalate to three seeds and page/block-level artifact accounting.",
        )
    if target_margin < 0.03:
        return (
            "NO-GO",
            "NO-GO: the tensor allocator does not beat the target production baseline by the predeclared margin.",
            "Do not promote this operating point unless another selector or source set changes the premise.",
        )
    if high_saving_bpw is not None and high_saving_bpw >= 0.25 and candidate["nll"] <= high["nll"] + 0.03:
        return (
            "GO",
            "GO: mixed production-format allocation approaches the high baseline while saving at least 0.25 bpw.",
            "Escalate to three seeds and page/block-level artifact accounting.",
        )
    return (
        "GRAY",
        "GRAY: mixed production-format allocation improves the target baseline, but below the promotion threshold.",
        "Run more seeds if the result is stable or points to a sharper selector.",
    )


def make_markdown(result: dict) -> str:
    candidate_variant = result["args"].get("candidate_variant", "c2_calib_greedy_mixed")
    candidate_extra_key = f"{candidate_variant}_extra_bytes"
    selection_extra = result["byte_accounting"].get("selection_extra_bytes", {})
    selection_saved = result["byte_accounting"].get("selection_saved_bytes", {})
    candidate_extra = selection_extra.get(
        candidate_variant,
        result["byte_accounting"].get(candidate_extra_key, 0),
    )
    candidate_saved = selection_saved.get(candidate_variant, 0)
    selection_base_source = result.get("selection_base_sources", {}).get(
        candidate_variant,
        result["args"]["low_source"],
    )
    lines = [
        "# Result Card - C2 Production Mixed-Rate Transcoder Gate",
        "",
        "## Status",
        "",
        result["verdict"],
        "",
        "## Decisive Measurement",
        "",
        f"The `{result['args'].get('tensor_profile', 'qwen')}` model profile was forward-evaluated after patching real production GGUF tensor payloads. The candidate starts from the low source and promotes selected tensor groups to stronger source formats under the target tensor-payload byte budget.",
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
            "## Selector Summary",
            "",
            f"- low source: `{result['args']['low_source']}`",
            f"- target source: `{result['args']['target_source']}`",
            f"- high sources: `{', '.join(result['args']['high_sources'])}`",
            f"- group mode: `{result['args']['group_mode']}`",
            f"- tensor profile: `{result['args'].get('tensor_profile', 'qwen')}`",
            f"- candidate variant: `{candidate_variant}`",
            f"- selection base source: `{selection_base_source}`",
            f"- calibration groups tested: `{len(result['allocation_rows'])}`",
            f"- selected groups: `{len(result['selections'][candidate_variant])}`",
            f"- selected extra bytes: `{candidate_extra}`",
            f"- selected saved bytes: `{candidate_saved}`",
        ]
    )
    local_report = result.get("local_search_reports", {}).get(candidate_variant)
    genetic_report = result.get("genetic_search_reports", {}).get(candidate_variant)
    anneal_report = result.get("anneal_search_reports", {}).get(candidate_variant)
    if genetic_report:
        lines.extend(
            [
                "",
                "## Genetic Search",
                "",
                f"- base variant: `{genetic_report['base_variant']}`",
                f"- generations: `{genetic_report['generations']}`",
                f"- population size: `{genetic_report['population_size']}`",
                f"- evaluated genomes: `{genetic_report['evaluated_genomes']}`",
                f"- fitness prompts: `{genetic_report.get('fitness_prompt_count', 0)}`",
                f"- validation prompts: `{genetic_report.get('validation_prompt_count', 0)}`",
                f"- selection metric: `{genetic_report.get('selection_metric', 'search_nll')}`",
                f"- search final NLL: `{genetic_report.get('final_search_nll', genetic_report['final_nll']):.6f}`",
                f"- final extra bytes: `{genetic_report['final_extra_bytes']}`",
            ]
        )
        if genetic_report.get("final_validation_nll") is not None:
            lines.extend(
                [
                    f"- validation final NLL: `{genetic_report['final_validation_nll']:.6f}`",
                    f"- validation rerank top-k: `{genetic_report.get('validation_rerank_top_k', 0)}`",
                ]
            )
    if anneal_report:
        lines.extend(
            [
                "",
                "## Simulated Annealing",
                "",
                f"- base variant: `{anneal_report['base_variant']}`",
                f"- steps: `{anneal_report['steps']}`",
                f"- evaluated states: `{anneal_report['evaluated_states']}`",
                f"- accepted moves: `{anneal_report['accepted_moves']}`",
                f"- accepted worse moves: `{anneal_report['accepted_worse_moves']}`",
                f"- temperature: `{anneal_report['initial_temperature']:.6g}` -> `{anneal_report['final_temperature']:.6g}`",
                f"- fitness prompts: `{anneal_report.get('fitness_prompt_count', 0)}`",
                f"- validation prompts: `{anneal_report.get('validation_prompt_count', 0)}`",
                f"- selection metric: `{anneal_report.get('selection_metric', 'search_nll')}`",
                f"- search final NLL: `{anneal_report.get('final_search_nll', anneal_report['final_nll']):.6f}`",
                f"- final extra bytes: `{anneal_report['final_extra_bytes']}`",
            ]
        )
        if anneal_report.get("final_validation_nll") is not None:
            lines.extend(
                [
                    f"- validation final NLL: `{anneal_report['final_validation_nll']:.6f}`",
                    f"- validation rerank top-k: `{anneal_report.get('validation_rerank_top_k', 0)}`",
                ]
            )
    if local_report:
        lines.extend(
            [
                "",
                "## Local Search Repair",
                "",
                f"- base variant: `{local_report['base_variant']}`",
                f"- calibration start NLL: `{local_report['start_nll']:.6f}`",
                f"- calibration final NLL: `{local_report['final_nll']:.6f}`",
                f"- calibration NLL improvement: `{local_report['total_nll_improvement']:.6f}`",
                f"- evaluated moves: `{local_report['evaluated_moves']}`",
                f"- accepted moves: `{len(local_report['accepted_moves'])}`",
            ]
        )
        for move in local_report["accepted_moves"]:
            lines.append(
                f"- step `{move['step']}`: {move['move']} "
                f"(calib NLL improvement `{move['nll_improvement']:.6f}`)"
            )
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
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hf", default=DEFAULT_HF)
    parser.add_argument("--source", action="append", default=[], help="Production GGUF source as label=path.")
    parser.add_argument("--low-source", default="iq3_xs")
    parser.add_argument("--target-source", default="q3_k_m")
    parser.add_argument("--high-sources", default="q3_k_m,iq4_xs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27")
    parser.add_argument("--group-mode", choices=["layer_family", "tensor"], default="tensor")
    parser.add_argument("--tensor-profile", choices=sorted(SUPPORTED_TENSOR_PROFILES), default="qwen")
    parser.add_argument("--calib-prompts", type=int, default=12)
    parser.add_argument("--eval-prompts", type=int, default=64)
    parser.add_argument("--calib-max-length", type=int, default=96)
    parser.add_argument("--eval-max-length", type=int, default=128)
    parser.add_argument("--prompt-source", choices=["synthetic", "public"], default="synthetic")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--prompt-seed", type=int, default=None)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate-variant", default="c2_calib_greedy_mixed")
    parser.add_argument("--knapsack-max-states", type=int, default=50_000)
    parser.add_argument(
        "--seed-selection-result-json",
        type=Path,
        default=None,
        help="Previous result.json whose selected variant should seed direct GA/anneal/local refinement.",
    )
    parser.add_argument(
        "--seed-selection-variant",
        default="",
        help="Variant name to load from --seed-selection-result-json. Defaults to that result's candidate variant.",
    )
    parser.add_argument(
        "--seed-selection-name",
        default="external_seed_mixed",
        help="Internal selection variant name for the loaded seed selection.",
    )
    parser.add_argument(
        "--local-search-from",
        default="",
        help="Base selection variant to repair with whole-model local search.",
    )
    parser.add_argument(
        "--local-search-steps",
        type=int,
        default=0,
        help="Number of accepted local-search repair moves to try. Disabled at 0.",
    )
    parser.add_argument(
        "--local-search-candidates",
        type=int,
        default=24,
        help="Maximum neighbor candidates to evaluate per local-search step.",
    )
    parser.add_argument(
        "--local-search-min-improvement",
        type=float,
        default=1e-4,
        help="Minimum calibration NLL improvement needed to accept a local-search move.",
    )
    parser.add_argument(
        "--genetic-search-from",
        default="",
        help="Base selection variant to seed a calibration-NLL genetic search.",
    )
    parser.add_argument(
        "--genetic-search-generations",
        type=int,
        default=0,
        help="Number of genetic-search generations. Disabled at 0.",
    )
    parser.add_argument(
        "--genetic-search-population",
        type=int,
        default=8,
        help="Population size for genetic candidate search.",
    )
    parser.add_argument(
        "--genetic-search-elite",
        type=int,
        default=2,
        help="Elite genomes preserved per generation.",
    )
    parser.add_argument(
        "--genetic-search-mutation-rate",
        type=float,
        default=0.25,
        help="Approximate fraction of selected groups mutated in each child genome.",
    )
    parser.add_argument(
        "--genetic-search-direct",
        action="store_true",
        help="Skip per-group promotion NLL scoring and evolve whole promoted subsets directly from byte-accounted options.",
    )
    parser.add_argument(
        "--genetic-search-validation-prompts",
        type=int,
        default=0,
        help="Reserve this many calibration prompts as a GA validation fold for top-genome reranking.",
    )
    parser.add_argument(
        "--genetic-search-rerank-top-k",
        type=int,
        default=0,
        help="Number of search-best GA genomes to rerank on the reserved validation fold. Defaults to min(population, 8) when validation prompts are reserved.",
    )
    parser.add_argument(
        "--anneal-search-from",
        default="",
        help="Base selection variant to seed simulated annealing.",
    )
    parser.add_argument(
        "--anneal-search-steps",
        type=int,
        default=0,
        help="Number of simulated-annealing mutation/evaluation steps. Disabled at 0.",
    )
    parser.add_argument(
        "--anneal-search-mutation-rate",
        type=float,
        default=0.20,
        help="Approximate fraction of selected groups mutated per annealing proposal.",
    )
    parser.add_argument(
        "--anneal-search-initial-temp",
        type=float,
        default=0.02,
        help="Initial simulated-annealing temperature in NLL units.",
    )
    parser.add_argument(
        "--anneal-search-final-temp",
        type=float,
        default=0.001,
        help="Final simulated-annealing temperature in NLL units.",
    )
    parser.add_argument(
        "--anneal-search-direct",
        action="store_true",
        help="Skip per-group promotion NLL scoring and anneal whole promoted subsets directly from byte-accounted options.",
    )
    parser.add_argument(
        "--anneal-search-validation-prompts",
        type=int,
        default=0,
        help="Reserve this many calibration prompts as an annealing validation fold for top-state reranking.",
    )
    parser.add_argument(
        "--anneal-search-rerank-top-k",
        type=int,
        default=0,
        help="Number of search-best annealing states to rerank on the reserved validation fold. Defaults to min(steps, 8) when validation prompts are reserved.",
    )
    parser.add_argument(
        "--sweep-payload-bpws",
        default="",
        help="Comma-separated payload-bpw budgets for extra promotion sweep variants.",
    )
    parser.add_argument(
        "--sweep-selectors",
        default="calib_knapsack",
        help="Comma-separated promotion selectors for sweeps: calib_knapsack, calib_greedy, weight_mse, blend.",
    )
    parser.add_argument(
        "--demotion-sources",
        default="",
        help="Comma-separated lower-byte sources to test by demoting from the demotion base source.",
    )
    parser.add_argument("--demotion-base-source", default=None)
    parser.add_argument(
        "--demotion-selectors",
        default="reverse_knapsack",
        help="Comma-separated demotion selectors for sweeps: reverse_knapsack, reverse_greedy.",
    )
    parser.add_argument("--max-shrink-nll-loss", type=float, default=0.05)
    args = parser.parse_args()
    args.layers = parse_layers(args.layers)
    high_sources = parse_csv(args.high_sources)
    sweep_payload_bpws = parse_float_csv(args.sweep_payload_bpws)
    sweep_selectors = parse_csv(args.sweep_selectors)
    demotion_sources = parse_csv(args.demotion_sources)
    demotion_base_source = args.demotion_base_source or args.target_source
    demotion_selectors = parse_csv(args.demotion_selectors)
    source_paths = parse_source_specs(args.source)
    required = {args.low_source, args.target_source, *high_sources, demotion_base_source, *demotion_sources}
    missing = sorted(required - set(source_paths))
    if missing:
        raise ValueError(f"missing source paths for {missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    allocation_checkpoint_path = checkpoint_dir / "allocation_rows.jsonl"
    demotion_checkpoint_path = checkpoint_dir / "demotion_rows.jsonl"
    scalar_eval_checkpoint_path = checkpoint_dir / "scalar_evals.jsonl"

    py_rng = random.Random(args.seed)
    print("[c2] loading tokenizer/model", flush=True)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = load_model_for_profile(args.model_dir, args.tensor_profile, device)

    readers = source_readers(source_paths)
    specs, skipped_specs = filter_specs_for_model(
        model,
        build_tensor_specs(args.layers, args.group_mode, args.tensor_profile),
    )
    specs, skipped_source_specs = filter_specs_for_sources(readers, specs)
    skipped_specs = [*skipped_specs, *skipped_source_specs]
    groups = group_specs(specs)
    source_payload_bytes = payload_bytes_by_source(readers, specs)
    if args.prompt_source == "public":
        calib_prompts, eval_prompts, prompt_audit = build_public_prompt_split(tokenizer, args)
    else:
        calib_prompts, eval_prompts, prompt_audit = build_disjoint_prompt_split(
            tokenizer,
            args.calib_prompts,
            args.eval_prompts,
            args.seed + 2000,
        )
    print(
        f"[c2] prompt audit split={prompt_audit['split']} overlap_count={prompt_audit['overlap_count']}",
        flush=True,
    )

    results: dict[str, dict] = {}
    selections: dict[str, list[dict]] = {}
    allocation_rows: list[dict] = []
    fp_last_logits = None

    with open_hf_tensor_source(args.hf) as hf:
        total_weights = total_weight_count(hf, model, specs)
        fp_context = {
            "kind": "fp16_reference",
            "model_dir": str(args.model_dir),
            "eval_prompt_digest": prompt_digest(eval_prompts),
            "eval_max_length": int(args.eval_max_length),
            "tensor_profile": args.tensor_profile,
        }
        fp_eval = load_fp16_checkpoint(checkpoint_dir, fp_context)
        if fp_eval is None:
            print("[c2] evaluating fp16 reference", flush=True)
            fp_eval = evaluate_model(model, tokenizer, eval_prompts, args)
            write_fp16_checkpoint(checkpoint_dir, fp_context, fp_eval)
        fp_last_logits = fp_eval["captured_last_logits"]
        results["fp16"] = add_byte_fields(strip_logits(fp_eval), total_weights * 2, total_weights, fp_eval["nll"])

        print(f"[c2] patching low source {args.low_source}", flush=True)
        patch_all_from_source(model, hf, readers, args.layers, args.low_source, args.group_mode, args.tensor_profile)
        low_calib_context = {
            "kind": "low_calibration",
            "source": args.low_source,
            "calib_prompt_digest": prompt_digest(calib_prompts),
            "calib_max_length": int(args.calib_max_length),
            "tensor_profile": args.tensor_profile,
            "group_mode": args.group_mode,
        }
        low_calib = load_eval_checkpoint(scalar_eval_checkpoint_path, low_calib_context)
        if low_calib is None:
            low_calib = evaluate_nll(model, tokenizer, calib_prompts, args.calib_max_length)
            append_eval_checkpoint(scalar_eval_checkpoint_path, low_calib_context, low_calib)
        print(f"[c2] low-source calibration NLL: {low_calib['nll']:.6f}", flush=True)

        direct_promotion_search = args.genetic_search_direct or args.anneal_search_direct
        if direct_promotion_search:
            print("[c2] building direct promotion options without per-group NLL scoring", flush=True)
            allocation_rows = build_direct_promotion_rows(readers, groups, args.low_source, high_sources)
        else:
            allocation_context = {
                "kind": "promotion",
                "low_source": args.low_source,
                "high_sources": high_sources,
                "calib_prompt_digest": prompt_digest(calib_prompts),
                "calib_max_length": int(args.calib_max_length),
                "tensor_profile": args.tensor_profile,
                "group_mode": args.group_mode,
            }
            allocation_rows = [
                clean_checkpoint_row(row)
                for row in load_jsonl_rows(allocation_checkpoint_path)
                if row.get("_checkpoint_context") == allocation_context
            ]
            seen_promotion_keys = {
                (str(row["group"]), str(row["source"]))
                for row in allocation_rows
            }
            if allocation_rows:
                print(
                    f"[c2] loaded {len(allocation_rows)} promotion score checkpoints",
                    flush=True,
                )
            print("[c2] scoring production-format group promotions", flush=True)
            score_idx = 0
            score_total = len(groups) * len(high_sources)
            for group in groups:
                low_bytes = group_payload_bytes(readers, groups, group, args.low_source)
                for source in high_sources:
                    score_idx += 1
                    key = (str(group), str(source))
                    if key in seen_promotion_keys:
                        if score_idx == 1 or score_idx % 20 == 0 or score_idx == score_total:
                            print(
                                f"[c2] checkpoint hit group {score_idx}/{score_total}: {group} -> {source}",
                                flush=True,
                            )
                        continue
                    if score_idx == 1 or score_idx % 20 == 0 or score_idx == score_total:
                        print(f"[c2] scoring group {score_idx}/{score_total}: {group} -> {source}", flush=True)
                    high_bytes = group_payload_bytes(readers, groups, group, source)
                    extra = high_bytes - low_bytes
                    if extra <= 0:
                        continue
                    patch_group(model, hf, readers, groups, group, source)
                    promoted_calib = evaluate_nll(model, tokenizer, calib_prompts, args.calib_max_length)
                    patch_group(model, hf, readers, groups, group, args.low_source)
                    improvement = low_calib["nll"] - promoted_calib["nll"]
                    weight_delta = group_weight_sse_delta(hf, model, readers, groups, group, args.low_source, source)
                    row = {
                        "group": group,
                        "source": source,
                        "low_bytes": int(low_bytes),
                        "high_bytes": int(high_bytes),
                        "extra_bytes": int(extra),
                        "calib_nll": float(promoted_calib["nll"]),
                        "calib_nll_improvement": float(improvement),
                        "calib_score_per_mbyte": float(improvement / (extra / 1_000_000)),
                        "weight_sse_delta": float(weight_delta),
                        "weight_score_per_mbyte": float(weight_delta / (extra / 1_000_000)),
                    }
                    allocation_rows.append(row)
                    append_jsonl_row(
                        allocation_checkpoint_path,
                        {**row, "_checkpoint_context": allocation_context},
                    )
                    seen_promotion_keys.add(key)

        demotion_rows: list[dict] = []
        if demotion_sources:
            print(f"[c2] patching demotion base source {demotion_base_source}", flush=True)
            patch_all_from_source(
                model,
                hf,
                readers,
                args.layers,
                demotion_base_source,
                args.group_mode,
                args.tensor_profile,
            )
            base_calib_context = {
                "kind": "demotion_base_calibration",
                "source": demotion_base_source,
                "calib_prompt_digest": prompt_digest(calib_prompts),
                "calib_max_length": int(args.calib_max_length),
                "tensor_profile": args.tensor_profile,
                "group_mode": args.group_mode,
            }
            base_calib = load_eval_checkpoint(scalar_eval_checkpoint_path, base_calib_context)
            if base_calib is None:
                base_calib = evaluate_nll(model, tokenizer, calib_prompts, args.calib_max_length)
                append_eval_checkpoint(scalar_eval_checkpoint_path, base_calib_context, base_calib)
            print(f"[c2] demotion-base calibration NLL: {base_calib['nll']:.6f}", flush=True)
            demotion_context = {
                "kind": "demotion",
                "base_source": demotion_base_source,
                "demotion_sources": demotion_sources,
                "calib_prompt_digest": prompt_digest(calib_prompts),
                "calib_max_length": int(args.calib_max_length),
                "tensor_profile": args.tensor_profile,
                "group_mode": args.group_mode,
            }
            demotion_rows = [
                clean_checkpoint_row(row)
                for row in load_jsonl_rows(demotion_checkpoint_path)
                if row.get("_checkpoint_context") == demotion_context
            ]
            seen_demotion_keys = {
                (str(row["group"]), str(row["source"]))
                for row in demotion_rows
            }
            if demotion_rows:
                print(f"[c2] loaded {len(demotion_rows)} demotion score checkpoints", flush=True)
            print("[c2] scoring production-format group demotions", flush=True)
            demote_idx = 0
            demote_total = len(groups) * len(demotion_sources)
            for group in groups:
                base_bytes = group_payload_bytes(readers, groups, group, demotion_base_source)
                for source in demotion_sources:
                    demote_idx += 1
                    key = (str(group), str(source))
                    if key in seen_demotion_keys:
                        if demote_idx == 1 or demote_idx % 20 == 0 or demote_idx == demote_total:
                            print(
                                f"[c2] checkpoint hit demotion {demote_idx}/{demote_total}: {group} -> {source}",
                                flush=True,
                            )
                        continue
                    if demote_idx == 1 or demote_idx % 20 == 0 or demote_idx == demote_total:
                        print(
                            f"[c2] scoring demotion {demote_idx}/{demote_total}: {group} -> {source}",
                            flush=True,
                        )
                    demoted_bytes = group_payload_bytes(readers, groups, group, source)
                    saved = base_bytes - demoted_bytes
                    if saved <= 0:
                        continue
                    patch_group(model, hf, readers, groups, group, source)
                    demoted_calib = evaluate_nll(model, tokenizer, calib_prompts, args.calib_max_length)
                    patch_group(model, hf, readers, groups, group, demotion_base_source)
                    loss = demoted_calib["nll"] - base_calib["nll"]
                    weight_delta = group_weight_sse_delta(
                        hf,
                        model,
                        readers,
                        groups,
                        group,
                        source,
                        demotion_base_source,
                    )
                    row = {
                        "group": group,
                        "source": source,
                        "base_source": demotion_base_source,
                        "base_bytes": int(base_bytes),
                        "demoted_bytes": int(demoted_bytes),
                        "extra_bytes": 0,
                        "saved_bytes": int(saved),
                        "calib_nll": float(demoted_calib["nll"]),
                        "calib_nll_loss": float(loss),
                        "demotion_loss_per_mbyte": float(loss / (saved / 1_000_000)),
                        "weight_sse_delta": float(weight_delta),
                        "weight_score_per_mbyte": float(weight_delta / (saved / 1_000_000)),
                    }
                    demotion_rows.append(row)
                    append_jsonl_row(
                        demotion_checkpoint_path,
                        {**row, "_checkpoint_context": demotion_context},
                    )
                    seen_demotion_keys.add(key)

        low_payload = source_payload_bytes[args.low_source]
        target_payload = source_payload_bytes[args.target_source]
        budget_extra = max(0, target_payload - low_payload)
        if direct_promotion_search and budget_extra <= 0:
            raise ValueError(
                "direct promotion search requires target payload to be larger than low payload; "
                f"low_source={args.low_source} low_payload={low_payload} "
                f"target_source={args.target_source} target_payload={target_payload}"
            )
        add_rank_blend_scores(
            allocation_rows,
            "calib_weight_rank_blend",
            {"calib_score_per_mbyte": 0.7, "weight_score_per_mbyte": 0.3},
        )
        calib_selected = select_by_score(allocation_rows, budget_extra, "calib_score_per_mbyte")
        weight_selected = select_by_score(allocation_rows, budget_extra, "weight_score_per_mbyte")
        blend_selected = select_by_score(allocation_rows, budget_extra, "calib_weight_rank_blend")
        knapsack_selected = select_knapsack_by_value(
            allocation_rows,
            budget_extra,
            "calib_nll_improvement",
            args.knapsack_max_states,
        )
        raw_selections: dict[str, list[dict]] = {}
        selection_base_sources: dict[str, str] = {}
        selection_extra_budgets: dict[str, int] = {}
        genetic_search_reports: dict[str, dict] = {}
        anneal_search_reports: dict[str, dict] = {}
        local_search_reports: dict[str, dict] = {}

        def record_selection(
            name: str,
            selected: list[dict],
            base_source: str,
            extra_budget: int | None = None,
        ) -> None:
            raw_selections[name] = selected
            selections[name] = compact_selection(selected)
            selection_base_sources[name] = base_source
            if extra_budget is not None:
                selection_extra_budgets[name] = int(extra_budget)

        if args.seed_selection_result_json is not None:
            seed_variant, seed_selected = load_seed_selection_result(
                args.seed_selection_result_json,
                args.seed_selection_variant,
                allocation_rows,
            )
            seed_name = args.seed_selection_name or "external_seed_mixed"
            seed_selected = repair_selection_to_budget(seed_selected, budget_extra)
            record_selection(seed_name, seed_selected, args.low_source, budget_extra)
            print(
                f"[c2] loaded seed selection {seed_variant} as {seed_name}: "
                f"groups={len(seed_selected)} extra={selected_extra_bytes(seed_selected)}",
                flush=True,
            )

        if not direct_promotion_search:
            record_selection("c2_calib_greedy_mixed", calib_selected, args.low_source, budget_extra)
            record_selection("c2_calib_knapsack_mixed", knapsack_selected, args.low_source, budget_extra)
            record_selection("c2_weight_mse_mixed", weight_selected, args.low_source, budget_extra)
            record_selection("c2_calib_weight_blend_mixed", blend_selected, args.low_source, budget_extra)

            for payload_bpw in sweep_payload_bpws:
                payload_budget = int(math.floor(payload_bpw * total_weights / 8))
                sweep_budget_extra = max(0, payload_budget - low_payload)
                for selector in sweep_selectors:
                    selected = build_promotion_selection(
                        selector,
                        allocation_rows,
                        sweep_budget_extra,
                        args.knapsack_max_states,
                    )
                    record_selection(sweep_variant_name(selector, payload_bpw), selected, args.low_source, sweep_budget_extra)

        demotion_base_payload = source_payload_bytes[demotion_base_source]
        demotion_budget_bpws = sweep_payload_bpws or [low_payload * 8 / total_weights]
        if demotion_rows:
            for payload_bpw in demotion_budget_bpws:
                payload_budget = int(math.floor(payload_bpw * total_weights / 8))
                required_saving = max(0, demotion_base_payload - payload_budget)
                for selector in demotion_selectors:
                    selected = build_demotion_selection(
                        selector,
                        demotion_rows,
                        required_saving,
                        args.knapsack_max_states,
                    )
                    record_selection(demotion_variant_name(selector, payload_bpw), selected, demotion_base_source)

        if int(args.genetic_search_generations) > 0:
            if args.genetic_search_direct:
                genetic_base_variant = args.genetic_search_from or "direct_random_population"
                genetic_base_source = args.low_source
                genetic_budget_extra = budget_extra
                genetic_variant = args.candidate_variant if args.candidate_variant.endswith("_genetic_mixed") else "c2_direct_genetic_mixed"
                seed_variants = {
                    name: selected
                    for name, selected in raw_selections.items()
                    if selection_base_sources.get(name) == genetic_base_source
                    and selected_saved_bytes(selected) == 0
                    and selected_extra_bytes(selected) <= genetic_budget_extra
                }
            else:
                genetic_base_variant = args.genetic_search_from or infer_genetic_search_base_variant(args.candidate_variant)
                if genetic_base_variant not in raw_selections:
                    raise ValueError(f"genetic-search base variant {genetic_base_variant!r} is not a selection variant")
                genetic_base_source = selection_base_sources[genetic_base_variant]
                if genetic_base_source != args.low_source:
                    raise ValueError("genetic search currently supports promotion selections from the low source only")
                genetic_variant = genetic_variant_name(genetic_base_variant)
                genetic_budget_extra = selection_extra_budgets.get(
                    genetic_base_variant,
                    selected_extra_bytes(raw_selections[genetic_base_variant]),
                )
                seed_variants = {
                    name: selected
                    for name, selected in raw_selections.items()
                    if selection_base_sources.get(name) == genetic_base_source
                    and selected_saved_bytes(selected) == 0
                    and selected_extra_bytes(selected) <= genetic_budget_extra
                }
            print(
                f"[c2] genetic search start base={genetic_base_variant} "
                f"seeds={len(seed_variants)} budget_extra={genetic_budget_extra}",
                flush=True,
            )
            genetic_selected, report = genetic_search_selection(
                model,
                tokenizer,
                hf,
                readers,
                groups,
                args,
                allocation_rows,
                seed_variants,
                genetic_base_source,
                genetic_budget_extra,
                calib_prompts,
                py_rng,
            )
            report["base_variant"] = genetic_base_variant
            report["variant"] = genetic_variant
            report["seed_variants"] = sorted(seed_variants)
            report["start_extra_bytes"] = int(
                selected_extra_bytes(raw_selections[genetic_base_variant])
                if genetic_base_variant in raw_selections
                else 0
            )
            record_selection(genetic_variant, genetic_selected, genetic_base_source, genetic_budget_extra)
            genetic_search_reports[genetic_variant] = report

        if int(args.anneal_search_steps) > 0:
            if args.anneal_search_direct:
                anneal_base_variant = args.anneal_search_from or "direct_random_population"
                anneal_base_source = args.low_source
                anneal_budget_extra = budget_extra
                anneal_variant = args.candidate_variant if args.candidate_variant.endswith("_anneal_mixed") else "c2_direct_anneal_mixed"
                seed_variants = {
                    name: selected
                    for name, selected in raw_selections.items()
                    if selection_base_sources.get(name) == anneal_base_source
                    and selected_saved_bytes(selected) == 0
                    and selected_extra_bytes(selected) <= anneal_budget_extra
                }
            else:
                anneal_base_variant = args.anneal_search_from or infer_anneal_search_base_variant(args.candidate_variant)
                if anneal_base_variant not in raw_selections:
                    raise ValueError(f"anneal-search base variant {anneal_base_variant!r} is not a selection variant")
                anneal_base_source = selection_base_sources[anneal_base_variant]
                if anneal_base_source != args.low_source:
                    raise ValueError("anneal search currently supports promotion selections from the low source only")
                anneal_variant = anneal_variant_name(anneal_base_variant)
                anneal_budget_extra = selection_extra_budgets.get(
                    anneal_base_variant,
                    selected_extra_bytes(raw_selections[anneal_base_variant]),
                )
                seed_variants = {
                    name: selected
                    for name, selected in raw_selections.items()
                    if selection_base_sources.get(name) == anneal_base_source
                    and selected_saved_bytes(selected) == 0
                    and selected_extra_bytes(selected) <= anneal_budget_extra
                }
            print(
                f"[c2] anneal search start base={anneal_base_variant} "
                f"seeds={len(seed_variants)} budget_extra={anneal_budget_extra}",
                flush=True,
            )
            anneal_selected, report = anneal_search_selection(
                model,
                tokenizer,
                hf,
                readers,
                groups,
                args,
                allocation_rows,
                seed_variants,
                anneal_base_source,
                anneal_budget_extra,
                calib_prompts,
                py_rng,
            )
            report["base_variant"] = anneal_base_variant
            report["variant"] = anneal_variant
            report["seed_variants"] = sorted(seed_variants)
            report["start_extra_bytes"] = int(
                selected_extra_bytes(raw_selections[anneal_base_variant])
                if anneal_base_variant in raw_selections
                else 0
            )
            record_selection(anneal_variant, anneal_selected, anneal_base_source, anneal_budget_extra)
            anneal_search_reports[anneal_variant] = report

        if int(args.local_search_steps) > 0:
            local_base_variant = args.local_search_from or infer_local_search_base_variant(args.candidate_variant)
            if local_base_variant not in raw_selections:
                raise ValueError(f"local-search base variant {local_base_variant!r} is not a selection variant")
            local_base_source = selection_base_sources[local_base_variant]
            if local_base_source != args.low_source:
                raise ValueError("local search currently supports promotion selections from the low source only")
            local_variant = local_search_variant_name(local_base_variant)
            local_budget_extra = selection_extra_budgets.get(
                local_base_variant,
                selected_extra_bytes(raw_selections[local_base_variant]),
            )
            repaired, report = local_search_repair_selection(
                model,
                tokenizer,
                hf,
                readers,
                groups,
                args,
                allocation_rows,
                raw_selections[local_base_variant],
                local_base_source,
                local_budget_extra,
                calib_prompts,
            )
            report["base_variant"] = local_base_variant
            report["variant"] = local_variant
            report["start_extra_bytes"] = int(selected_extra_bytes(raw_selections[local_base_variant]))
            report["final_extra_bytes"] = int(selected_extra_bytes(repaired))
            record_selection(local_variant, repaired, local_base_source, local_budget_extra)
            local_search_reports[local_variant] = report

        if args.candidate_variant not in raw_selections:
            raise ValueError(f"candidate variant {args.candidate_variant!r} is not a selection variant")
        candidate_selected = raw_selections[args.candidate_variant]
        candidate_base_source = selection_base_sources[args.candidate_variant]
        if selected_saved_bytes(candidate_selected) > 0:
            random_selected = select_random_demotions(
                demotion_rows,
                selected_saved_bytes(candidate_selected),
                py_rng,
            )
            random_base_source = candidate_base_source
        else:
            random_selected = select_random(
                allocation_rows,
                selected_extra_bytes(candidate_selected),
                py_rng,
            )
            random_base_source = args.low_source
        record_selection("c2_random_same_budget", random_selected, random_base_source)

        variants_to_eval: list[tuple[str, str | None, list[dict] | None, int]] = [
            (args.low_source, args.low_source, None, source_payload_bytes[args.low_source]),
            (args.target_source, args.target_source, None, source_payload_bytes[args.target_source]),
        ]
        for source in sorted(source_paths):
            if source not in {args.low_source, args.target_source, *high_sources}:
                variants_to_eval.append((source, source, None, source_payload_bytes[source]))
        for source in high_sources:
            if source not in {args.low_source, args.target_source}:
                variants_to_eval.append((source, source, None, source_payload_bytes[source]))
        for name, selected in raw_selections.items():
            base_source = selection_base_sources[name]
            base_payload = source_payload_bytes[base_source]
            variants_to_eval.append((name, None, selected, selected_payload_bytes(base_payload, selected)))

        variant_checkpoint_path = checkpoint_dir / "variant_results.jsonl"
        variant_context = {
            "kind": "variant_eval",
            "model_dir": str(args.model_dir),
            "low_source": args.low_source,
            "target_source": args.target_source,
            "high_sources": high_sources,
            "eval_prompt_digest": prompt_digest(eval_prompts),
            "eval_max_length": int(args.eval_max_length),
            "tensor_profile": args.tensor_profile,
            "group_mode": args.group_mode,
            "candidate_variant": args.candidate_variant,
        }
        variant_cache = {
            str(row["name"]): row["result"]
            for row in load_jsonl_rows(variant_checkpoint_path)
            if row.get("_checkpoint_context") == variant_context and row.get("result") is not None
        }
        if variant_cache:
            print(f"[c2] loaded {len(variant_cache)} variant eval checkpoints", flush=True)
        for name, source, selected, payload_bytes in variants_to_eval:
            if name in variant_cache:
                print(f"[c2] checkpoint hit variant {name}", flush=True)
                results[name] = variant_cache[name]
                continue
            if source is not None:
                print(f"[c2] evaluating uniform source {name}", flush=True)
                patch_all_from_source(model, hf, readers, args.layers, source, args.group_mode, args.tensor_profile)
            else:
                print(f"[c2] evaluating mixed variant {name}", flush=True)
                patch_all_from_source(
                    model,
                    hf,
                    readers,
                    args.layers,
                    selection_base_sources[name],
                    args.group_mode,
                    args.tensor_profile,
                )
                apply_selection(model, hf, readers, groups, selected or [])
            eval_result = evaluate_model(model, tokenizer, eval_prompts, args, fp_last_logits)
            row_result = add_byte_fields(strip_logits(eval_result), payload_bytes, total_weights, results["fp16"]["nll"])
            results[name] = row_result
            append_jsonl_row(
                variant_checkpoint_path,
                {
                    "_checkpoint_context": variant_context,
                    "name": name,
                    "source": source,
                    "payload_bytes": int(payload_bytes),
                    "result": row_result,
                },
            )

    serial_args = {
        "model_dir": args.model_dir,
        "hf": args.hf,
        "source": {label: str(path) for label, path in source_paths.items()},
        "low_source": args.low_source,
        "target_source": args.target_source,
        "high_sources": high_sources,
        "layers": args.layers,
        "group_mode": args.group_mode,
        "tensor_profile": args.tensor_profile,
        "calib_prompts": args.calib_prompts,
        "eval_prompts": args.eval_prompts,
        "calib_max_length": args.calib_max_length,
        "eval_max_length": args.eval_max_length,
        "prompt_source": args.prompt_source,
        "dataset": args.dataset if args.prompt_source == "public" else None,
        "dataset_config": args.dataset_config if args.prompt_source == "public" else None,
        "text_column": args.text_column if args.prompt_source == "public" else None,
        "calib_split": args.calib_split if args.prompt_source == "public" else None,
        "eval_split": args.eval_split if args.prompt_source == "public" else None,
        "prompt_seed": (args.prompt_seed if args.prompt_seed is not None else args.seed + 2000),
        "min_tokens": args.min_tokens if args.prompt_source == "public" else None,
        "seed": args.seed,
        "device": args.device,
        "candidate_variant": args.candidate_variant,
        "knapsack_max_states": args.knapsack_max_states,
        "seed_selection_result_json": (
            str(args.seed_selection_result_json) if args.seed_selection_result_json is not None else ""
        ),
        "seed_selection_variant": args.seed_selection_variant,
        "seed_selection_name": args.seed_selection_name,
        "local_search_from": args.local_search_from,
        "local_search_steps": args.local_search_steps,
        "local_search_candidates": args.local_search_candidates,
        "local_search_min_improvement": args.local_search_min_improvement,
        "genetic_search_from": args.genetic_search_from,
        "genetic_search_generations": args.genetic_search_generations,
        "genetic_search_population": args.genetic_search_population,
        "genetic_search_elite": args.genetic_search_elite,
        "genetic_search_mutation_rate": args.genetic_search_mutation_rate,
        "genetic_search_direct": args.genetic_search_direct,
        "genetic_search_validation_prompts": args.genetic_search_validation_prompts,
        "genetic_search_rerank_top_k": args.genetic_search_rerank_top_k,
        "anneal_search_from": args.anneal_search_from,
        "anneal_search_steps": args.anneal_search_steps,
        "anneal_search_mutation_rate": args.anneal_search_mutation_rate,
        "anneal_search_initial_temp": args.anneal_search_initial_temp,
        "anneal_search_final_temp": args.anneal_search_final_temp,
        "anneal_search_direct": args.anneal_search_direct,
        "anneal_search_validation_prompts": args.anneal_search_validation_prompts,
        "anneal_search_rerank_top_k": args.anneal_search_rerank_top_k,
        "sweep_payload_bpws": sweep_payload_bpws,
        "sweep_selectors": sweep_selectors,
        "demotion_sources": demotion_sources,
        "demotion_base_source": demotion_base_source,
        "demotion_selectors": demotion_selectors,
        "max_shrink_nll_loss": args.max_shrink_nll_loss,
    }
    selection_extra_bytes = {name: int(selected_extra_bytes(rows)) for name, rows in raw_selections.items()}
    selection_saved_bytes = {name: int(selected_saved_bytes(rows)) for name, rows in raw_selections.items()}
    selection_payloads = {
        name: int(selected_payload_bytes(source_payload_bytes[selection_base_sources[name]], rows))
        for name, rows in raw_selections.items()
    }
    result = {
        "created_utc": datetime.now(UTC).isoformat(),
        "args": serial_args,
        "prompt_audit": prompt_audit,
        "source_payload_bytes": source_payload_bytes,
        "total_weight_count": int(total_weights),
        "skipped_model_tensors": [spec.logical_name for spec in skipped_specs],
        "byte_accounting": {
            "low_payload_bytes": int(low_payload),
            "target_payload_bytes": int(target_payload),
            "budget_extra_bytes": int(budget_extra),
            "c2_calib_greedy_mixed_extra_bytes": int(selected_extra_bytes(calib_selected)),
            "c2_calib_knapsack_mixed_extra_bytes": int(selected_extra_bytes(knapsack_selected)),
            "c2_weight_mse_mixed_extra_bytes": int(selected_extra_bytes(weight_selected)),
            "c2_calib_weight_blend_mixed_extra_bytes": int(selected_extra_bytes(blend_selected)),
            "c2_random_same_budget_extra_bytes": int(selected_extra_bytes(random_selected)),
            "selection_extra_bytes": selection_extra_bytes,
            "selection_saved_bytes": selection_saved_bytes,
            "selection_payload_bytes": selection_payloads,
            "selection_extra_budgets": selection_extra_budgets,
            "tag_overhead_bits_estimate": int(math.ceil(math.log2(max(2, len(source_paths)))) * len(groups)),
        },
        "allocation_rows": allocation_rows,
        "demotion_rows": demotion_rows,
        "genetic_search_reports": genetic_search_reports,
        "anneal_search_reports": anneal_search_reports,
        "local_search_reports": local_search_reports,
        "selection_base_sources": selection_base_sources,
        "selections": selections,
        "variants": results,
    }
    status, decision_text, next_step = decide(result)
    result["status"] = status
    result["verdict"] = status
    result["decision_text"] = decision_text
    result["next_step"] = next_step

    write_json_atomic(args.output_dir / "result.json", result)
    write_text_atomic(args.output_dir / "result.md", make_markdown(result))
    print(f"[c2] wrote {args.output_dir / 'result.md'}", flush=True)
    print(decision_text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
