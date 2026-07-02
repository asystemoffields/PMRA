"""Pure-logic unit tests for apply_code_guardrail (PMRA v2 release guardrail).

Extracts the function from cpu_prober.py via AST so we exercise the real
source without importing the module (gguf/numpy deps). Run on CPU:
    python tests/test_code_guardrail.py
"""
import ast
import os

PROBER = os.path.join(os.path.dirname(__file__), "..", "scripts", "cpu_prober.py")


def _load_fn(name):
    src = open(PROBER, encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {}
            exec(ast.get_source_segment(src, node), ns)
            return ns[name]
    raise AssertionError(f"{name} not found in {PROBER}")


apply_code_guardrail = _load_fn("apply_code_guardrail")

MIX = "c2_calib_knapsack_mixed"
TGT = "iq3_xs"


def test_pass_within_epsilon():
    verdict, block = apply_code_guardrail(
        "GO", {MIX: 1.510, TGT: 1.500}, MIX, TGT, epsilon=0.02)
    assert verdict == "GO"
    assert block["pass"] is True
    assert abs(block["regression_vs_target"] - 0.010) < 1e-12


def test_fail_downgrades_go_to_gray():
    verdict, block = apply_code_guardrail(
        "GO", {MIX: 1.530, TGT: 1.500}, MIX, TGT, epsilon=0.02)
    assert verdict == "GRAY"
    assert block["pass"] is False
    assert block["verdict_before_guardrail"] == "GO"


def test_fail_leaves_gray_and_nogo_untouched():
    for base in ["GRAY", "NO-GO"]:
        verdict, block = apply_code_guardrail(
            base, {MIX: 1.600, TGT: 1.500}, MIX, TGT, epsilon=0.02)
        assert verdict == base
        assert block["pass"] is False
        assert "verdict_before_guardrail" not in block


def test_improvement_passes():
    verdict, block = apply_code_guardrail(
        "GO", {MIX: 1.480, TGT: 1.500}, MIX, TGT, epsilon=0.02)
    assert verdict == "GO"
    assert block["pass"] is True
    assert block["regression_vs_target"] < 0


def test_record_only_when_epsilon_zero():
    verdict, block = apply_code_guardrail(
        "GO", {MIX: 1.600, TGT: 1.500}, MIX, TGT, epsilon=0.0)
    assert verdict == "GO"
    assert block["enforced"] is False
    assert block["pass"] is None
    assert abs(block["regression_vs_target"] - 0.100) < 1e-12


def test_missing_nll_is_inconclusive_not_crash():
    verdict, block = apply_code_guardrail(
        "GO", {MIX: None, TGT: 1.500}, MIX, TGT, epsilon=0.02)
    assert verdict == "GO"
    assert block["pass"] is None
    assert "regression_vs_target" not in block
    verdict, block = apply_code_guardrail("GO", {}, MIX, TGT, epsilon=0.02)
    assert verdict == "GO"
    assert block["pass"] is None


def test_boundary_regression_equal_to_epsilon_passes():
    # exactly representable floats so regression == epsilon without FP fuzz
    verdict, block = apply_code_guardrail(
        "GO", {MIX: 1.75, TGT: 1.5}, MIX, TGT, epsilon=0.25)
    assert verdict == "GO"
    assert block["pass"] is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"{len(fns)} passed")
