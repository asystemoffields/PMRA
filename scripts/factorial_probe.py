"""Factorial (multi-promotion) probing: estimate per-group effects from
whole-mix evals instead of one-at-a-time probes.

One-factor-at-a-time tier-2 spends a full llama-perplexity run per candidate.
This runs a Plackett-Burman design over per-group binary factors (promote
group g to its design level vs leave at the low base): each eval splices ~half
the groups at once, and least squares recovers every main effect with
variance sigma^2/N — better than N single probes at 1/Nth the runs. Because
the knapsack's additivity assumption is exactly the design's main-effects
model, the same evals double as a direct additivity test: every row's
measured paired dNLL is compared against the sum of its singles.

Modes (composable):
  --make-design   build design.json from an empirical allocation_rows.jsonl
                  (design level per group = best measured score/MB level)
  (default)       run the design's evals, sharded, checkpointed by row id
  --fit           recover main effects + additivity report from finished evals

The design is deterministic given the same rows file, so shard workers
regenerate it independently — no design distribution step.

Validation experiment for Qwen3.5-4B: 19 MLP groups -> PB20 (20 evals)
+ all_cheap / level_high / level_knap rows = 23 evals vs 64 single probes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from cpu_prober import (
    assemble_mixed_gguf,
    build_groups,
    open_sources,
    parse_source_specs,
    run_perplexity,
    select_knapsack,
    shard_of,
)

# Plackett-Burman N=20 cyclic generator (Plackett & Burman 1946).
PB20_GENERATOR = "++--++++-+-+----++-"


def pb20_matrix(n_factors: int) -> np.ndarray:
    """20 x n_factors ±1 design; columns are orthogonal and balanced."""
    if n_factors > 19:
        raise ValueError(f"PB20 supports at most 19 factors, got {n_factors}")
    gen = np.array([1 if c == "+" else -1 for c in PB20_GENERATOR])
    rows = [np.roll(gen, i) for i in range(19)] + [np.full(19, -1)]
    X = np.array(rows)[:, :n_factors]
    XtX = np.column_stack([np.ones(20), X]).T @ np.column_stack([np.ones(20), X])
    assert np.allclose(XtX, 20 * np.eye(n_factors + 1)), "PB20 generator not orthogonal"
    return X


def layer_of(group: str) -> int:
    return int(group.split(":")[0][1:])


def load_singles(rows_path: Path) -> dict[str, dict[str, dict]]:
    singles: dict[str, dict[str, dict]] = {}
    for line in rows_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        singles.setdefault(r["group"], {})[r["source"]] = r
    return singles


def make_design(rows_path: Path, budget_bytes: int | None) -> dict:
    singles = load_singles(rows_path)
    groups = sorted(singles, key=layer_of)

    def cheapest(g: str) -> str:
        return min(singles[g], key=lambda s: singles[g][s]["extra_bytes"])

    def best_level(g: str) -> str:
        return max(singles[g], key=lambda s: singles[g][s]["calib_nll_improvement"])

    design_level = {g: cheapest(g) for g in groups}
    X = pb20_matrix(len(groups))
    rows = []
    for i in range(X.shape[0]):
        assignment = {g: design_level[g] for j, g in enumerate(groups) if X[i, j] > 0}
        rows.append({"id": f"pb{i:02d}", "kind": "pb", "assignment": assignment})

    positive = [g for g in groups
                if singles[g][design_level[g]]["calib_nll_improvement"] > 0]
    rows.append({"id": "all_cheap", "kind": "extra",
                 "assignment": {g: design_level[g] for g in positive}})
    rows.append({"id": "level_high", "kind": "extra",
                 "assignment": {g: best_level(g) for g in positive
                                if singles[g][best_level(g)]["calib_nll_improvement"] > 0}})
    flat = [r for g in singles for r in singles[g].values()]
    budget = budget_bytes or int(sum(sorted((r["extra_bytes"] for r in flat), reverse=True)[: len(flat) // 3]))
    knap = select_knapsack([dict(r) for r in flat], budget, "calib_nll_improvement")
    rows.append({"id": "level_knap", "kind": "extra",
                 "assignment": {r["group"]: r["source"] for r in knap}})

    for row in rows:
        row["predicted_dnll"] = sum(
            singles[g][s]["calib_nll_improvement"] for g, s in row["assignment"].items())
        row["extra_bytes"] = sum(
            singles[g][s]["extra_bytes"] for g, s in row["assignment"].items())
    return {"factors": groups, "design_level": design_level,
            "knap_budget": budget, "rows": rows}


def paired_delta(base_chunk_nlls: list[float], probe_chunk_nlls: list[float]) -> tuple[float, float]:
    k = len(probe_chunk_nlls)
    deltas = [b - p for b, p in zip(base_chunk_nlls[:k], probe_chunk_nlls)]
    mean = sum(deltas) / k
    if k < 2:
        return mean, math.inf
    var = sum((d - mean) ** 2 for d in deltas) / (k - 1)
    return mean, math.sqrt(var / k)


def fit_report(design: dict, evals_path: Path, rows_path: Path) -> None:
    singles = load_singles(rows_path)
    evals = {json.loads(l)["id"]: json.loads(l)
             for l in evals_path.read_text().splitlines() if l.strip()}
    groups = design["factors"]
    done = [r for r in design["rows"] if r["id"] in evals]
    print(f"[fit] {len(done)}/{len(design['rows'])} design rows evaluated")

    print("\n[additivity] predicted (sum of singles) vs measured paired dNLL:")
    print(f"  {'id':10s} {'k':>3s} {'predicted':>10s} {'measured':>10s} {'resid':>9s} {'resid/se':>8s}")
    preds, meas = [], []
    for row in done:
        e = evals[row["id"]]
        resid = e["measured_dnll"] - row["predicted_dnll"]
        z = resid / e["measured_se"] if e["measured_se"] else float("nan")
        preds.append(row["predicted_dnll"])
        meas.append(e["measured_dnll"])
        print(f"  {row['id']:10s} {len(row['assignment']):3d} {row['predicted_dnll']:+10.5f} "
              f"{e['measured_dnll']:+10.5f} {resid:+9.5f} {z:8.2f}")
    preds, meas = np.array(preds), np.array(meas)
    slope = float(np.sum(preds * meas) / np.sum(preds * preds))
    r2 = 1.0 - float(np.sum((meas - preds) ** 2) / np.sum((meas - np.mean(meas)) ** 2))
    print(f"[additivity] slope(through origin)={slope:.3f}  R^2(vs predicted)={r2:.4f}  "
          f"mean|resid|={np.mean(np.abs(meas - preds)):.5f}")

    pb = [r for r in done if r["kind"] == "pb"]
    if len(pb) >= len(groups) + 1:
        X = np.column_stack([np.ones(len(pb))] +
                            [[1.0 if g in r["assignment"] else 0.0 for r in pb] for g in groups])
        y = np.array([evals[r["id"]]["measured_dnll"] for r in pb])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        recovered = dict(zip(groups, beta[1:]))
        truth = {g: singles[g][design["design_level"][g]]["calib_nll_improvement"] for g in groups}
        a = np.array([recovered[g] for g in groups])
        b = np.array([truth[g] for g in groups])
        rho = float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
        rmse = float(np.sqrt(np.mean((a - b) ** 2)))
        se = float(np.sqrt(np.mean([evals[r["id"]]["measured_se"] ** 2 for r in pb]) / len(pb)) * 2)
        print(f"\n[recovery] {len(pb)}-run PB main effects vs single probes "
              f"(design level): Spearman={rho:+.3f}  RMSE={rmse:.5f}  ~SE(effect)={se:.5f}")
        print(f"  {'group':10s} {'single':>9s} {'factorial':>10s} {'diff':>9s}")
        for g in groups:
            print(f"  {g:10s} {truth[g]:+9.5f} {recovered[g]:+10.5f} {recovered[g]-truth[g]:+9.5f}")
    else:
        print(f"\n[recovery] skipped: only {len(pb)} PB rows done, need {len(groups) + 1}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--low-source", required=True)
    parser.add_argument("--rows", type=Path, required=True, help="Empirical singles allocation_rows.jsonl.")
    parser.add_argument("--calib-text", type=Path)
    parser.add_argument("--llama-bin", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scalar-evals", type=Path, default=None,
                        help="scalar_evals.jsonl holding the low base eval with chunk_nlls.")
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=24)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--group-mode", default="layer_family")
    parser.add_argument("--budget-bytes", type=int, default=None)
    parser.add_argument("--shard", default=None, help="k/N stable-hash shard of design rows.")
    parser.add_argument("--make-design", action="store_true", help="Write design.json and exit.")
    parser.add_argument("--fit", action="store_true", help="Fit + report from finished evals and exit.")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    design_path = out / "design.json"
    evals_path = out / "factorial_evals.jsonl"

    design = make_design(args.rows, args.budget_bytes)
    design_path.write_text(json.dumps(design, indent=2))
    print(f"[design] {len(design['factors'])} factors, {len(design['rows'])} rows "
          f"-> {design_path}", flush=True)
    if args.make_design:
        return 0
    if args.fit:
        fit_report(design, evals_path, args.rows)
        return 0

    source_paths = parse_source_specs(args.source)
    readers = open_sources(source_paths)
    tensors_by_source = {label: {t.name: t for t in reader.tensors}
                         for label, reader in readers.items()}
    base_reader = readers[args.low_source]
    groups = build_groups(base_reader, args.group_mode)

    base_chunk_nlls = None
    if args.scalar_evals and args.scalar_evals.exists():
        for line in args.scalar_evals.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["label"] == f"{args.low_source}@calib" and "chunk_nlls" in row:
                base_chunk_nlls = row["chunk_nlls"]
    if base_chunk_nlls is None:
        print("[base] measuring low base (no cached chunk NLLs)", flush=True)
        result = run_perplexity(args.llama_bin, source_paths[args.low_source],
                                args.calib_text, args.ctx, args.chunks, args.threads)
        base_chunk_nlls = result["chunk_nlls"]
        with (out / "base_eval.json").open("w") as fh:
            json.dump(result, fh)

    done_ids = {json.loads(l)["id"] for l in evals_path.read_text().splitlines() if l.strip()} \
        if evals_path.exists() else set()
    todo = [r for r in design["rows"] if r["id"] not in done_ids]
    if args.shard:
        idx, count = (int(p) for p in args.shard.split("/"))
        todo = [r for r in todo if shard_of(r["id"], count) == idx]
    print(f"[run] {len(todo)} evals to run"
          + (f" (shard {args.shard})" if args.shard else ""), flush=True)

    mix_gguf = out / "factorial_mix.gguf"
    for i, row in enumerate(todo, start=1):
        tensor_sources = {}
        for g, s in row["assignment"].items():
            for name in groups[g]:
                tensor_sources[name] = s
        assemble_mixed_gguf(mix_gguf, base_reader, tensors_by_source,
                            tensor_sources, args.low_source)
        result = run_perplexity(args.llama_bin, mix_gguf, args.calib_text,
                                args.ctx, args.chunks, args.threads)
        mean, se = paired_delta(base_chunk_nlls, result["chunk_nlls"])
        record = {"id": row["id"], "kind": row["kind"],
                  "n_promoted": len(row["assignment"]),
                  "predicted_dnll": row["predicted_dnll"],
                  "measured_dnll": mean, "measured_se": se,
                  "chunks": len(result["chunk_nlls"]), "tokens": result.get("tokens"),
                  "assignment": row["assignment"]}
        with evals_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"[run] {i}/{len(todo)} {row['id']}: measured={mean:+.5f}±{se:.5f} "
              f"predicted={row['predicted_dnll']:+.5f} ({len(row['assignment'])} promotions)", flush=True)
    if mix_gguf.exists():
        mix_gguf.unlink()
    print(f"[run] shard complete; evals in {evals_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
