"""Unit tests for profile-aware group_for_tensor in cpu_prober.py.

Needs the gguf package importable (same as test_early_stop.py):
    /data/pmra-dev-venv/bin/python3 tests/test_grouping.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from cpu_prober import NEMOTRON_H_PATTERN, group_for_tensor  # noqa: E402


def test_default_profile_unchanged():
    # llama-family behavior must be byte-identical to the pre-profile code
    assert group_for_tensor("blk.5.attn_q.weight", "layer_family") == "L5:attn"
    assert group_for_tensor("blk.5.attn_q.weight", "tensor") == "L5:attn_q"
    assert group_for_tensor("blk.5.ffn_up.weight", "layer_family") == "L5:mlp"
    assert group_for_tensor("blk.5.ffn_norm.weight", "layer_family") == "L5:mlp"
    assert group_for_tensor("token_embd.weight", "layer_family") == "global:embed"
    assert group_for_tensor("rope_freqs.weight", "layer_family") is None
    assert group_for_tensor("blk.5.ssm_in.weight", "layer_family") is None


def test_nemotron_pattern_covers_42_layers():
    assert len(NEMOTRON_H_PATTERN) == 42


def test_nemotron_mamba_layer():
    # layer 0 is 'M' (Mamba2)
    assert NEMOTRON_H_PATTERN[0] == "M"
    assert group_for_tensor("blk.0.ssm_in.weight", "layer_family", "nemotron_h") == "L0:ssm"
    assert group_for_tensor("blk.0.ssm_in.weight", "tensor", "nemotron_h") == "L0:ssm_in"
    assert group_for_tensor("blk.0.ssm_conv1d.bias", "tensor", "nemotron_h") == "L0:ssm_conv1d_bias"
    assert group_for_tensor("blk.0.ssm_dt.bias", "tensor", "nemotron_h") == "L0:ssm_dt_bias"
    assert group_for_tensor("blk.0.ssm_a", "tensor", "nemotron_h") == "L0:ssm_a"
    assert group_for_tensor("blk.0.ssm_d", "tensor", "nemotron_h") == "L0:ssm_d"
    assert group_for_tensor("blk.0.attn_norm.weight", "layer_family", "nemotron_h") == "L0:ssm"
    assert group_for_tensor("blk.0.attn_norm.weight", "tensor", "nemotron_h") == "L0:attn_norm"
    # ffn/attn tails don't exist in a Mamba layer
    assert group_for_tensor("blk.0.ffn_up.weight", "tensor", "nemotron_h") is None
    assert group_for_tensor("blk.0.attn_q.weight", "tensor", "nemotron_h") is None


def test_nemotron_mlp_layer():
    # layer 1 is '-' (MLP-only)
    assert NEMOTRON_H_PATTERN[1] == "-"
    assert group_for_tensor("blk.1.ffn_up.weight", "layer_family", "nemotron_h") == "L1:mlp"
    assert group_for_tensor("blk.1.ffn_down.weight", "tensor", "nemotron_h") == "L1:ffn_down"
    assert group_for_tensor("blk.1.attn_norm.weight", "layer_family", "nemotron_h") == "L1:mlp"
    assert group_for_tensor("blk.1.ssm_in.weight", "tensor", "nemotron_h") is None


def test_nemotron_attention_layer():
    # layer 12 is '*' (attention-only)
    assert NEMOTRON_H_PATTERN[12] == "*"
    assert group_for_tensor("blk.12.attn_q.weight", "layer_family", "nemotron_h") == "L12:attn"
    assert group_for_tensor("blk.12.attn_output.weight", "tensor", "nemotron_h") == "L12:attn_output"
    assert group_for_tensor("blk.12.attn_norm.weight", "layer_family", "nemotron_h") == "L12:attn"
    assert group_for_tensor("blk.12.ffn_up.weight", "tensor", "nemotron_h") is None


def test_nemotron_pattern_matches_gate_source():
    # the pattern is duplicated (prober + gate); the artifact builder uses the
    # gate copy, the verdict uses the prober copy — drift ships a wrong artifact
    import re
    gate = os.path.join(os.path.dirname(__file__), "..", "scripts",
                        "production_mixed_rate_transcoder_gate.py")
    match = re.search(r'_NEMOTRON_H_PATTERN = "([^"]+)"', open(gate, encoding="utf-8").read())
    assert match and match.group(1) == NEMOTRON_H_PATTERN


def test_nemotron_layer_out_of_pattern_is_hard_error():
    try:
        group_for_tensor("blk.42.ssm_in.weight", "tensor", "nemotron_h")
    except ValueError:
        pass
    else:
        raise AssertionError("layer 42 should raise, not silently default")


def test_qwen35_deltanet_coverage():
    assert group_for_tensor("blk.3.attn_qkv.weight", "layer_family", "qwen35") == "L3:attn"
    assert group_for_tensor("blk.3.attn_qkv.weight", "tensor", "qwen35") == "L3:attn_qkv"
    assert group_for_tensor("blk.3.attn_gate.weight", "layer_family", "qwen35") == "L3:attn"
    assert group_for_tensor("blk.3.ssm_beta.weight", "layer_family", "qwen35") == "L3:attn"
    # qwen35 gate spec names the dt bias short "ssm_dt" (unlike nemotron's "ssm_dt_bias")
    assert group_for_tensor("blk.3.ssm_dt.bias", "tensor", "qwen35") == "L3:ssm_dt"
    assert group_for_tensor("blk.3.ssm_a", "tensor", "qwen35") == "L3:ssm_a"
    assert group_for_tensor("blk.3.post_attention_norm.weight", "layer_family", "qwen35") == "L3:mlp"
    assert group_for_tensor("blk.3.ffn_gate.weight", "layer_family", "qwen35") == "L3:mlp"
    assert group_for_tensor("blk.3.attn_q.weight", "tensor", "qwen35") == "L3:attn_q"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"{len(fns)} passed")
