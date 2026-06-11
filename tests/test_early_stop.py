"""Unit tests for the tier-2 early-stopping probe math in cpu_prober.py.

Run on CPU (needs the prober's deps: numpy, gguf):
    python tests/test_early_stop.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from cpu_prober import (  # noqa: E402
    _parse_cumulative_ppls,
    chunk_nlls_from_cumulative,
    paired_probe_stats,
)


def cumulative_ppls(chunk_nlls):
    cum, total = [], 0.0
    for k, nll in enumerate(chunk_nlls, start=1):
        total += nll
        cum.append(math.exp(total / k))
    return cum


def test_chunk_recovery_roundtrip():
    nlls = [3.1, 2.8, 3.4, 2.95, 3.2, 3.05]
    recovered = chunk_nlls_from_cumulative(cumulative_ppls(nlls))
    assert all(abs(a - b) < 1e-9 for a, b in zip(nlls, recovered)), recovered
    print("ok: per-chunk NLL recovery round-trips")


def test_paired_mean_telescopes():
    base = [3.10, 2.85, 3.40, 2.90, 3.25, 3.00, 3.15, 2.95]
    probe = [b - 0.02 for b in base]  # uniform 0.02-nat improvement
    mean, se = paired_probe_stats(base, cumulative_ppls(probe))
    assert abs(mean - 0.02) < 1e-9, mean
    assert se < 1e-9, se  # constant delta -> zero variance
    print("ok: paired mean telescopes; constant delta gives ~zero SE")


def test_se_reflects_noise():
    base = [3.0, 3.2, 2.9, 3.1, 3.05, 2.95, 3.15, 2.85]
    probe = [b - d for b, d in zip(base, [0.05, -0.01, 0.04, 0.0, 0.03, -0.02, 0.05, 0.01])]
    mean, se = paired_probe_stats(base, cumulative_ppls(probe))
    assert abs(mean - 0.01875) < 1e-9, mean
    assert 0.005 < se < 0.015, se
    print(f"ok: noisy deltas -> mean={mean:.5f}, se={se:.5f}")


def test_partial_prefix_uses_matching_base_chunks():
    base = [3.0, 3.2, 2.9, 3.1, 3.05, 2.95, 3.15, 2.85]
    probe = [b - 0.03 for b in base]
    mean, _ = paired_probe_stats(base, cumulative_ppls(probe)[:5])  # only 5 chunks seen
    assert abs(mean - 0.03) < 1e-9, mean
    print("ok: partial prefix pairs against matching base chunks")


def test_progress_parser_gates_on_marker_and_order():
    out = ("llama_model_loader: - kv [1]2.5 noise\n"
           "perplexity: calculating perplexity over 4 chunks\n"
           "[1]4.6043,[2]4.9871,[3]5.0102,"
           "stray [7]9.9999 ignored, [4]4.8001,")
    cum = _parse_cumulative_ppls(out)
    assert cum == [4.6043, 4.9871, 5.0102, 4.8001], cum
    assert _parse_cumulative_ppls("no marker [1]2.0") == []
    print("ok: parser requires marker and monotone chunk ids")


if __name__ == "__main__":
    test_chunk_recovery_roundtrip()
    test_paired_mean_telescopes()
    test_se_reflects_noise()
    test_partial_prefix_uses_matching_base_chunks()
    test_progress_parser_gates_on_marker_and_order()
    print("all early-stop tests passed")
