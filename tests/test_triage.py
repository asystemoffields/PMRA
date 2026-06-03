"""Pure-logic unit tests for triage_select_probe_keys (PMRA v2 #8).

Extracts the function from the gate via AST so we exercise the real source
without importing the heavy gate module (torch/gguf/etc.). Run on CPU:
    python tests/test_triage.py
"""
import ast
import math
import os

GATE = os.path.join(os.path.dirname(__file__), "..", "scripts",
                    "production_mixed_rate_transcoder_gate.py")


def _load_fn(name):
    src = open(GATE, encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"math": math}
            exec(ast.get_source_segment(src, node), ns)
            return ns[name]
    raise AssertionError(f"{name} not found in gate")


triage_select_probe_keys = _load_fn("triage_select_probe_keys")


def _cands():
    # 3 groups x 2 sources; ranked by ws desc: g0hi,g1hi,g2hi,g0lo,g1lo,g2lo
    return [
        {"key": ("g0", "hi"), "group": "g0", "extra": 100, "ws": 10.0},
        {"key": ("g0", "lo"), "group": "g0", "extra": 50, "ws": 4.0},
        {"key": ("g1", "hi"), "group": "g1", "extra": 100, "ws": 8.0},
        {"key": ("g1", "lo"), "group": "g1", "extra": 50, "ws": 3.0},
        {"key": ("g2", "hi"), "group": "g2", "extra": 100, "ws": 6.0},
        {"key": ("g2", "lo"), "group": "g2", "extra": 50, "ws": 2.0},
    ]


def test_fraction_one_probes_all():
    c = _cands()
    keys = triage_select_probe_keys(c, budget_extra=250, probe_fraction=1.0, boundary_band=0.0)
    assert keys == {x["key"] for x in c}, "fraction=1.0 must reduce to full probing (v1 path)"
    # any fraction >= 1.0 probes all, regardless of budget/band
    for f in (1.0, 1.5, 2.0):
        assert triage_select_probe_keys(c, 0, f, 0.0) == {x["key"] for x in c}


def test_fraction_half_probes_top_half():
    c = _cands()
    keys = triage_select_probe_keys(c, budget_extra=250, probe_fraction=0.5, boundary_band=0.0)
    assert len(keys) == 3, f"expected ceil(6*0.5)=3 probed, got {len(keys)}"
    assert keys == {("g0", "hi"), ("g1", "hi"), ("g2", "hi")}, "must probe the top-ranked candidates"


def test_boundary_band_covers_budget_cutoff():
    c = _cands()
    # budget=250 fits g0hi+g1hi (200), g2hi (300) exceeds -> greedy cutoff at rank 2.
    # fraction=0 + band=0 => probe strictly within budget (ranks [0,2) = top 2).
    keys = triage_select_probe_keys(c, budget_extra=250, probe_fraction=0.0, boundary_band=0.0)
    assert keys == {("g0", "hi"), ("g1", "hi")}, f"band=0 should probe within-budget only, got {keys}"
    # a band extends probing past the cutoff so the boundary decision is empirical.
    wide = triage_select_probe_keys(c, budget_extra=250, probe_fraction=0.0, boundary_band=0.5)
    assert len(wide) >= 5 and keys.issubset(wide), "boundary band must extend the probe set past the cutoff"


def test_empty():
    assert triage_select_probe_keys([], 100, 0.5, 0.1) == set()


def test_reduction_factor_realistic():
    # 60 candidates, modest fraction -> meaningful probe reduction but cutoff still covered
    c = [{"key": (f"g{i}", "s"), "group": f"g{i}", "extra": 10, "ws": float(60 - i)} for i in range(60)]
    keys = triage_select_probe_keys(c, budget_extra=200, probe_fraction=0.4, boundary_band=0.1)
    # budget 200 / 10 bytes = 20 groups fit -> cutoff ~20; top=24, band=6 -> upto=max(24,26)=26
    assert len(keys) == 26, f"expected 26 probed of 60 (~2.3x reduction), got {len(keys)}"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} triage unit tests passed")
    raise SystemExit(0 if passed == len(tests) else 1)
