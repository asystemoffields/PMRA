"""
Per-Everything Compression Budget Allocator

PMRA-style multiple-choice knapsack over (layer, matrix, family, rank).
Model-agnostic: auto-detects architecture from profiling results.
Works with any transformer — standard, GQA, MoE, custom attention, etc.
"""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DP_UNIT = 1024


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Option:
    family: str
    compressed_params: int
    compression_ratio: float
    delta_loss: float
    weight_units: int = 0

@dataclass
class Group:
    group_id: str
    layer: int
    matrix_type: str
    original_params: int
    source_layer: int
    options: list[Option] = field(default_factory=list)

@dataclass
class Allocation:
    group_id: str
    layer: int
    matrix_type: str
    original_params: int
    chosen: Option

@dataclass
class ModelConfig:
    """Auto-detected model configuration from profiling results."""
    n_layers: int
    embed_params: int
    base_ppl: float
    base_loss: float
    profiled_layers: list[int]
    matrix_types: list[str]
    matrix_original_params: dict[str, int]
    per_layer_params: int
    layer_params: int
    total_params: int
    component_map: dict[str, str]  # matrix_type -> component (attn/mlp/expert/other)


# ═══════════════════════════════════════════════════════════════════════
# AUTO-DETECTION
# ═══════════════════════════════════════════════════════════════════════

_LAYER_RE = re.compile(r"\.layers?\.(\d+)\.")

def parse_matrix_name(name):
    """Extract (layer_index, matrix_type) from any HF-style parameter name."""
    m = _LAYER_RE.search(name)
    if not m:
        return -1, name
    layer_idx = int(m.group(1))
    suffix = name[m.end():]
    suffix = suffix.replace(".weight", "").replace(".bias", "")
    parts = suffix.split(".")
    if any(p.endswith("_proj") for p in parts):
        matrix_type = next(p for p in parts if p.endswith("_proj"))
    elif len(parts) >= 2:
        matrix_type = ".".join(parts[-2:])
    else:
        matrix_type = parts[-1]
    return layer_idx, matrix_type


def classify_component(matrix_type):
    """Heuristic: classify matrix type into component for budget breakdown."""
    t = matrix_type.lower()
    if any(k in t for k in ("q_proj", "k_proj", "v_proj", "o_proj", "qkv", "self_attn", "attention")):
        return "attn"
    if any(k in t for k in ("gate", "up_proj", "down_proj", "mlp", "fc1", "fc2", "dense_h_to_4h", "dense_4h_to_h")):
        return "mlp"
    if "expert" in t:
        return "expert"
    return "other"


def detect_model_config(results, n_layers, embed_params, base_ppl):
    """Auto-detect model configuration from profiling results."""
    profiled = {}
    matrix_types = {}

    for config in results:
        layer_idx, mtype = parse_matrix_name(config["matrix"])
        if layer_idx < 0:
            continue
        profiled.setdefault(layer_idx, set()).add(mtype)
        if mtype not in matrix_types:
            matrix_types[mtype] = config["original_params"]

    profiled_layers = sorted(profiled.keys())
    mtypes = sorted(matrix_types.keys())
    per_layer = sum(matrix_types.values())

    base_loss = math.log(base_ppl)
    layer_params = per_layer * n_layers
    total_params = layer_params + embed_params

    component_map = {mt: classify_component(mt) for mt in mtypes}

    cfg = ModelConfig(
        n_layers=n_layers,
        embed_params=embed_params,
        base_ppl=base_ppl,
        base_loss=base_loss,
        profiled_layers=profiled_layers,
        matrix_types=mtypes,
        matrix_original_params=matrix_types,
        per_layer_params=per_layer,
        layer_params=layer_params,
        total_params=total_params,
        component_map=component_map,
    )

    return cfg


def source_layer_for(target_layer, profiled_layers, n_layers):
    """Map an unprofiled layer to the best profiled layer.
    Edge layers (first/last) are special — only map to edge profiled layers.
    All middle layers map to the nearest non-edge profiled layer."""
    if target_layer == 0:
        return profiled_layers[0]
    if target_layer == n_layers - 1:
        return profiled_layers[-1]
    middle_profiled = [p for p in profiled_layers if p != 0 and p != n_layers - 1]
    if not middle_profiled:
        return min(profiled_layers, key=lambda p: abs(target_layer - p))
    return min(middle_profiled, key=lambda p: abs(target_layer - p))


# ═══════════════════════════════════════════════════════════════════════
# LOADING & GROUPS
# ═══════════════════════════════════════════════════════════════════════

def load_results(path):
    with open(path) as f:
        return json.load(f)


def pareto_frontier(options):
    """Tradeoff curve: fewer params vs lower dloss.
    Sweep from most-compressed to least, keeping options that improve dloss."""
    sorted_opts = sorted(options, key=lambda o: o.compressed_params)
    frontier = []
    best_dloss = float("inf")
    for opt in sorted_opts:
        if opt.delta_loss < best_dloss:
            frontier.append(opt)
            best_dloss = opt.delta_loss
    return frontier


def build_groups(results, cfg, compress_all=False, preserve_layers=None):
    """Build matrix groups from profiling results.

    compress_all: if True, remove "no compression" option from most groups,
        forcing the knapsack to spread compression across all functional units.
        This prevents routing/execution mismatch where compressed attention
        sends wrong signals into intact MLP that faithfully amplifies them.
    preserve_layers: set of layer indices exempt from compress_all (e.g., {0}
        for layer 0 MLP which is genuinely incompressible).
    """
    if preserve_layers is None:
        preserve_layers = set()

    by_source = {}
    for config in results:
        layer_idx, matrix_type = parse_matrix_name(config["matrix"])
        key = (layer_idx, matrix_type)
        by_source.setdefault(key, []).append(config)

    groups = []
    for layer in range(cfg.n_layers):
        src = source_layer_for(layer, cfg.profiled_layers, cfg.n_layers)
        for mtype in cfg.matrix_types:
            key = (src, mtype)
            orig = cfg.matrix_original_params[mtype]

            raw_options = []
            for config in by_source.get(key, []):
                raw_options.append(Option(
                    family=config["family"],
                    compressed_params=config["compressed_params"],
                    compression_ratio=config["compression_ratio"],
                    delta_loss=config["delta_loss"],
                ))

            allow_original = (not compress_all) or (layer in preserve_layers)
            if allow_original:
                raw_options.append(Option("original", orig, 1.0, 0.0))

            frontier = pareto_frontier(raw_options)
            for opt in frontier:
                opt.weight_units = opt.compressed_params // DP_UNIT

            groups.append(Group(
                group_id=f"L{layer:02d}.{mtype}",
                layer=layer,
                matrix_type=mtype,
                original_params=orig,
                source_layer=src,
                options=frontier,
            ))

    return groups


# ═══════════════════════════════════════════════════════════════════════
# KNAPSACK SOLVER (numpy-vectorized MCKP)
# ═══════════════════════════════════════════════════════════════════════

def solve_knapsack(groups, budget_params):
    budget_units = budget_params // DP_UNIT
    INF = np.float64(1e30)

    dp = np.full(budget_units + 1, INF, dtype=np.float64)
    dp[0] = 0.0

    all_choices = []
    all_prev = []

    for group in groups:
        new_dp = np.full(budget_units + 1, INF, dtype=np.float64)
        choice_arr = np.full(budget_units + 1, -1, dtype=np.int32)
        prev_arr = np.full(budget_units + 1, -1, dtype=np.int32)

        for opt_idx, opt in enumerate(group.options):
            w = opt.weight_units
            if w > budget_units:
                continue
            c = opt.delta_loss
            end = budget_units - w + 1
            source = dp[:end]
            candidates = source + c
            target = slice(w, w + end)
            improved = candidates < new_dp[target]
            new_dp[target] = np.where(improved, candidates, new_dp[target])
            choice_arr[target] = np.where(improved, opt_idx, choice_arr[target])
            prev_arr[target] = np.where(improved, np.arange(end, dtype=np.int32), prev_arr[target])

        dp = new_dp
        all_choices.append(choice_arr)
        all_prev.append(prev_arr)

    feasible = dp.copy()
    feasible[feasible >= INF / 2] = INF
    if np.all(feasible >= INF / 2):
        return None, float("inf")

    best_u = int(np.argmin(feasible))
    total_dloss = float(dp[best_u])

    allocations = []
    u = best_u
    for g in range(len(groups) - 1, -1, -1):
        opt_idx = int(all_choices[g][u])
        u = int(all_prev[g][u])
        allocations.append(Allocation(
            group_id=groups[g].group_id,
            layer=groups[g].layer,
            matrix_type=groups[g].matrix_type,
            original_params=groups[g].original_params,
            chosen=groups[g].options[opt_idx],
        ))
    allocations.reverse()
    return allocations, total_dloss


# ═══════════════════════════════════════════════════════════════════════
# GREEDY BASELINE
# ═══════════════════════════════════════════════════════════════════════

def solve_greedy(groups, budget_params, cfg):
    savings_needed = cfg.layer_params - budget_params

    candidates = []
    for g, group in enumerate(groups):
        for opt in group.options:
            saved = group.original_params - opt.compressed_params
            if saved <= 0:
                continue
            efficiency = opt.delta_loss / saved
            candidates.append((efficiency, g, opt))

    candidates.sort(key=lambda x: x[0])

    assigned = {}
    total_saved = 0
    for eff, g, opt in candidates:
        if g in assigned:
            continue
        assigned[g] = opt
        total_saved += groups[g].original_params - opt.compressed_params
        if total_saved >= savings_needed:
            break

    allocations = []
    total_dloss = 0.0
    for g, group in enumerate(groups):
        chosen = assigned.get(g, group.options[-1])
        if chosen.family != "original" and chosen.compressed_params >= group.original_params:
            chosen = Option("original", group.original_params, 1.0, 0.0, group.original_params // DP_UNIT)
        total_dloss += chosen.delta_loss
        allocations.append(Allocation(
            group_id=group.group_id, layer=group.layer,
            matrix_type=group.matrix_type, original_params=group.original_params,
            chosen=chosen,
        ))
    return allocations, total_dloss


# ═══════════════════════════════════════════════════════════════════════
# UNIFORM BASELINE
# ═══════════════════════════════════════════════════════════════════════

def solve_uniform(groups, budget_params):
    ratios_to_try = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0,
                     16.0, 20.0, 26.0, 36.0, 52.0, 72.0, 100.0]

    best_allocs = None
    best_dloss = float("inf")

    for target_r in ratios_to_try:
        allocs = []
        total_params = 0
        total_dloss = 0.0
        for group in groups:
            best_opt = group.options[-1]
            for opt in group.options:
                if opt.compression_ratio >= target_r * 0.8:
                    if opt.delta_loss < best_opt.delta_loss or best_opt.family == "original":
                        best_opt = opt
            allocs.append(Allocation(
                group_id=group.group_id, layer=group.layer,
                matrix_type=group.matrix_type, original_params=group.original_params,
                chosen=best_opt,
            ))
            total_params += best_opt.compressed_params
            total_dloss += best_opt.delta_loss
        if total_params <= budget_params and total_dloss < best_dloss:
            best_allocs = allocs
            best_dloss = total_dloss

    if best_allocs is None:
        allocs = []
        total_dloss = 0.0
        for group in groups:
            best = min(group.options, key=lambda o: o.compressed_params)
            allocs.append(Allocation(
                group_id=group.group_id, layer=group.layer,
                matrix_type=group.matrix_type, original_params=group.original_params,
                chosen=best,
            ))
            total_dloss += best.delta_loss
        return allocs, total_dloss
    return best_allocs, best_dloss


# ═══════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════

def alloc_stats(allocations, cfg):
    total_compressed = sum(a.chosen.compressed_params for a in allocations)
    total_original = sum(a.original_params for a in allocations)
    total_dloss = sum(a.chosen.delta_loss for a in allocations)

    by_component = {}
    for a in allocations:
        comp = cfg.component_map.get(a.matrix_type, "other")
        entry = by_component.setdefault(comp, {"params": 0, "dloss": 0.0})
        entry["params"] += a.chosen.compressed_params
        entry["dloss"] += a.chosen.delta_loss

    edge_layers = {0, cfg.n_layers - 1}
    edge_params = sum(a.chosen.compressed_params for a in allocations if a.layer in edge_layers)
    edge_dloss = sum(a.chosen.delta_loss for a in allocations if a.layer in edge_layers)
    mid_params = sum(a.chosen.compressed_params for a in allocations if a.layer not in edge_layers)
    mid_dloss = sum(a.chosen.delta_loss for a in allocations if a.layer not in edge_layers)

    return {
        "total_compressed": total_compressed,
        "total_original": total_original,
        "total_dloss": total_dloss,
        "layer_ratio": total_original / total_compressed if total_compressed > 0 else float("inf"),
        "overall_ratio": (total_original + cfg.embed_params) / (total_compressed + cfg.embed_params),
        "est_ppl": math.exp(cfg.base_loss + total_dloss),
        "by_component": by_component,
        "edge_params": edge_params, "edge_dloss": edge_dloss,
        "mid_params": mid_params, "mid_dloss": mid_dloss,
    }


def print_report(allocations, greedy_allocs, uniform_allocs, budget_params, target_ratio, cfg):
    s = alloc_stats(allocations, cfg)
    gs = alloc_stats(greedy_allocs, cfg)
    us = alloc_stats(uniform_allocs, cfg)

    profiled_str = ", ".join(str(l) for l in cfg.profiled_layers)
    n_unprofiled = cfg.n_layers - len(cfg.profiled_layers)

    print(f"\n{'=' * 80}")
    print(f"  COMPRESSION BUDGET ALLOCATOR")
    print(f"  {cfg.n_layers} layers, {len(cfg.matrix_types)} matrix types, "
          f"profiled layers: [{profiled_str}]")
    print(f"  Target layer ratio: {target_ratio}x")
    print(f"  Layer budget: {budget_params:,} params (of {cfg.layer_params:,})")
    print(f"{'=' * 80}")

    print(f"\n  ALLOCATION (compressed matrices only)")
    print(f"  {'Layer':<6} {'Matrix':<16} {'Family':<22} {'Compressed':>12} {'Ratio':>7} {'dloss':>10}")
    print(f"  {'-' * 78}")

    last_layer = -1
    for a in allocations:
        if a.chosen.family == "original":
            continue
        if a.layer != last_layer:
            if last_layer >= 0:
                print()
            last_layer = a.layer
        print(f"  L{a.layer:02d}   {a.matrix_type:<16} {a.chosen.family:<22} "
              f"{a.chosen.compressed_params:>12,} {a.chosen.compression_ratio:>6.1f}x "
              f"{a.chosen.delta_loss:>+10.6f}")

    n_original = sum(1 for a in allocations if a.chosen.family == "original")
    n_compressed = len(allocations) - n_original
    print(f"\n  {n_compressed} compressed, {n_original} kept original")

    print(f"\n  SUMMARY")
    print(f"  {'-' * 78}")
    print(f"  Compressed layer params:   {s['total_compressed']:>14,}")
    print(f"  Achieved layer ratio:      {s['layer_ratio']:>14.2f}x")
    print(f"  Overall model ratio:       {s['overall_ratio']:>14.2f}x  "
          f"(incl. {cfg.embed_params:,} embed)")
    print(f"  Total delta_loss:          {s['total_dloss']:>+14.6f}")
    print(f"  Estimated PPL:             {s['est_ppl']:>14.2f}  (baseline: {cfg.base_ppl:.2f})")

    print(f"\n  BUDGET BREAKDOWN")
    print(f"  {'-' * 78}")
    pct = lambda x: x / s['total_compressed'] * 100 if s['total_compressed'] > 0 else 0
    for comp in sorted(s["by_component"]):
        c = s["by_component"][comp]
        print(f"  {comp:>12s}:  {c['params']:>12,} params ({pct(c['params']):>5.1f}%)  "
              f"dloss={c['dloss']:>+.6f}")
    print()
    print(f"  Edge (L0,L{cfg.n_layers-1}): {s['edge_params']:>11,} params ({pct(s['edge_params']):>5.1f}%)  "
          f"dloss={s['edge_dloss']:>+.6f}")
    print(f"  Middle:        {s['mid_params']:>11,} params ({pct(s['mid_params']):>5.1f}%)  "
          f"dloss={s['mid_dloss']:>+.6f}")

    print(f"\n  COMPARISON")
    print(f"  {'-' * 78}")
    print(f"  {'Method':<12} {'Params':>14} {'LayerRatio':>12} {'dloss':>12} {'Est.PPL':>10}")
    for label, st in [("Optimal", s), ("Greedy", gs), ("Uniform", us)]:
        print(f"  {label:<12} {st['total_compressed']:>14,} {st['layer_ratio']:>11.2f}x "
              f"{st['total_dloss']:>+12.6f} {st['est_ppl']:>10.2f}")

    gap = gs['total_dloss'] - s['total_dloss']
    if s['total_dloss'] > 0:
        print(f"\n  Knapsack advantage over greedy: {gap:+.6f} dloss "
              f"({gap / s['total_dloss'] * 100:+.1f}%)")

    print(f"\n  CAVEATS")
    print(f"  {'-' * 78}")
    print(f"  * delta_loss is additive (first-order). True compound loss may differ.")
    if n_unprofiled > 0:
        print(f"  * {n_unprofiled} layers extrapolated from {len(cfg.profiled_layers)} profiled "
              f"[{profiled_str}] via nearest-neighbor.")
    if cfg.embed_params > 0:
        print(f"  * Embeddings ({cfg.embed_params:,} params) treated as incompressible.")
    print()


def save_json(allocations, greedy_allocs, uniform_allocs, budget_params, target_ratio, cfg, path):
    def allocs_to_list(allocs):
        return [{
            "group_id": a.group_id, "layer": a.layer, "matrix_type": a.matrix_type,
            "original_params": a.original_params, "family": a.chosen.family,
            "compressed_params": a.chosen.compressed_params,
            "compression_ratio": a.chosen.compression_ratio,
            "delta_loss": a.chosen.delta_loss,
        } for a in allocs]

    data = {
        "n_layers": cfg.n_layers,
        "total_params": cfg.total_params,
        "embed_params": cfg.embed_params,
        "layer_params": cfg.layer_params,
        "base_ppl": cfg.base_ppl,
        "profiled_layers": cfg.profiled_layers,
        "matrix_types": cfg.matrix_types,
        "target_layer_ratio": target_ratio,
        "budget_layer_params": budget_params,
        "optimal": {**alloc_stats(allocations, cfg), "allocations": allocs_to_list(allocations)},
        "greedy": {**alloc_stats(greedy_allocs, cfg), "allocations": allocs_to_list(greedy_allocs)},
        "uniform": {**alloc_stats(uniform_allocs, cfg), "allocations": allocs_to_list(uniform_allocs)},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved to {path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Per-Everything Compression Budget Allocator (model-agnostic)")
    parser.add_argument("--results", type=str, default="results/structured_search.json")
    parser.add_argument("--target-ratio", type=float, nargs="+", required=True)
    parser.add_argument("--n-layers", type=int, required=True,
                        help="Total transformer layers in the full model")
    parser.add_argument("--embed-params", type=int, default=0,
                        help="Non-layer params (embeddings, lm_head) treated as incompressible")
    parser.add_argument("--base-ppl", type=float, default=None,
                        help="Baseline perplexity (auto-detected from results if available)")
    parser.add_argument("--dp-unit", type=int, default=1024)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--compress-all", action="store_true",
                        help="Force all matrices to be compressed (no 'original' option). "
                             "Prevents routing/execution mismatch where compressed attention "
                             "sends wrong signals into intact MLP that amplifies them.")
    parser.add_argument("--preserve-layers", type=int, nargs="*", default=[0],
                        help="Layers exempt from --compress-all (default: [0] for layer 0 "
                             "MLP which is genuinely incompressible)")
    args = parser.parse_args()

    global DP_UNIT
    DP_UNIT = args.dp_unit

    print(f"Loading results from {args.results}...")
    results = load_results(Path(args.results))
    print(f"  {len(results)} configs loaded")

    base_ppl = args.base_ppl
    if base_ppl is None:
        ppls = [c["ppl"] for c in results if abs(c["delta_loss"]) < 0.001]
        base_ppl = min(ppls) if ppls else results[0]["ppl"] - results[0]["delta_loss"]
        print(f"  Auto-detected base PPL: {base_ppl:.3f}")

    cfg = detect_model_config(results, args.n_layers, args.embed_params, base_ppl)

    print(f"\n  Model: {cfg.n_layers} layers, {len(cfg.matrix_types)} matrix types")
    print(f"  Profiled layers: {cfg.profiled_layers}")
    print(f"  Matrix types: {cfg.matrix_types}")
    print(f"  Per-layer params: {cfg.per_layer_params:,}")
    print(f"  Layer params: {cfg.layer_params:,}  Embed: {cfg.embed_params:,}  "
          f"Total: {cfg.total_params:,}")
    print(f"  Components: {dict(cfg.component_map)}")

    preserve = set(args.preserve_layers) if args.preserve_layers else set()
    groups = build_groups(results, cfg,
                          compress_all=args.compress_all,
                          preserve_layers=preserve)
    total_options = sum(len(g.options) for g in groups)
    mode = "compress-all" if args.compress_all else "standard"
    print(f"\n  {len(groups)} groups, {total_options} Pareto-optimal options "
          f"(avg {total_options / len(groups):.1f}/group) [{mode}]")

    for target_ratio in args.target_ratio:
        budget = int(cfg.layer_params / target_ratio)

        min_possible = sum(min(o.compressed_params for o in g.options) for g in groups)
        if min_possible > budget:
            max_ratio = cfg.layer_params / min_possible
            print(f"\n  INFEASIBLE: {target_ratio}x requires {budget:,} params but minimum "
                  f"is {min_possible:,} (max ratio: {max_ratio:.1f}x)")
            continue

        print(f"\nSolving for {target_ratio}x (budget: {budget:,} layer params)...")

        opt_allocs, opt_dloss = solve_knapsack(groups, budget)
        if opt_allocs is None:
            print(f"  Knapsack found no feasible solution")
            continue

        greedy_allocs, greedy_dloss = solve_greedy(groups, budget, cfg)
        uniform_allocs, uniform_dloss = solve_uniform(groups, budget)

        print_report(opt_allocs, greedy_allocs, uniform_allocs, budget, target_ratio, cfg)

        if args.output_json:
            out = Path(args.output_json)
            if len(args.target_ratio) > 1:
                out = out.with_stem(f"{out.stem}_{target_ratio}x")
            save_json(opt_allocs, greedy_allocs, uniform_allocs, budget, target_ratio, cfg, out)


if __name__ == "__main__":
    main()
