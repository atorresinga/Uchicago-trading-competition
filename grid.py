#!/usr/bin/env python3
"""Grid search over portfolio strategies — CSV output, no extra dependencies.

Produces one .csv per strategy with every parameter combo, its per-holdout-month
Sharpe values, mean Sharpe, variance, std, min, and max. Sorted by mean Sharpe
descending.

Usage:
    python grid_search.py                        # run both strategies
    python grid_search.py momentum_tilt          # run just one
    python grid_search.py --list                 # show combo counts
    python grid_search.py --output-dir results   # custom output dir

Requires: prices.csv, meta.csv, submission.py, months_validate.py in cwd.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
import traceback

import numpy as np

from submission import (
    PublicMeta, StrategyBase, load_meta, load_prices,
    RiskParity, VolTargetRiskParity, CostAwareTilt, MomentumTilt,
    IntradayRiskParity, SectorAwareIntradayRiskParity, N_ASSETS,
)
from months_validate import (
    evaluate_sliding_month_sharpes, sliding_month_groups_shape,
)

# ── Parameter grids ──────────────────────────────────────────────────────────

GRIDS: dict[str, dict] = {
    "momentum_tilt": {
        "class": MomentumTilt,
        "params": {
            "momentum_lookback": [10, 20, 30, 40, 60, 90, 120, 180],
            "rebalance_freq":    [5, 10, 20, 40, 60],
            "spread_penalty":    [0.0, 0.001, 0.005, 0.01, 0.02, 0.05],
            "borrow_penalty":    [0.0, 0.001, 0.005, 0.01, 0.02, 0.05],
            "zero_negative":     [True, False],
            "blend_rate":        [0.5, 0.7, 1.0],
            "equal_weight_blend": [0.0, 0.1, 0.2, 0.3],
        },
    },
    "intraday_risk_parity": {
        "class": IntradayRiskParity,
        "params": {
            "cov_lookback_days": [20, 40, 60, 90, 120],
            "vol_lookback_days": [10, 20, 40],
            "rebalance_freq":    [10, 20, 40],
            "blend_rate":        [0.3, 0.5, 1.0],
            "vol_target":        [0.08, 0.10, 0.15, 0.20],
            "vol_cap":           [1.0],
            "vol_floor":         [0.1, 0.2, 0.3],
            "use_intraday_cov":  [True],
            "spread_penalty":    [0.0, 0.5, 1.0],
            "borrow_penalty":    [0.0, 0.5, 1.0],
        },
    },
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _combo_count(params: dict) -> int:
    n = 1
    for v in params.values():
        n *= len(v)
    return n


def _param_combos(params: dict):
    keys = list(params.keys())
    for vals in itertools.product(*params.values()):
        yield dict(zip(keys, vals))


def _format_val(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


# ── CSV writer ───────────────────────────────────────────────────────────────

def write_results_csv(
    param_keys: list[str],
    rows: list[dict],
    n_holdout_months: int,
    output_path: str,
):
    headers = (
        list(param_keys)
        + [f"Month_{i}" for i in range(n_holdout_months)]
        + ["Mean_Sharpe", "Var_Sharpe", "Std_Sharpe", "Min_Sharpe", "Max_Sharpe", "Time_s"]
    )

    rows_sorted = sorted(rows, key=lambda r: r.get("mean", -999), reverse=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row_data in rows_sorted:
            row = []
            for k in param_keys:
                row.append(row_data["params"].get(k))
            for mi in range(n_holdout_months):
                val = row_data["sharpes"][mi] if mi < len(row_data["sharpes"]) else ""
                row.append(f"{val:.6f}" if isinstance(val, float) and np.isfinite(val) else val)
            for stat_key in ["mean", "var", "std", "min_s", "max_s"]:
                val = row_data.get(stat_key)
                row.append(f"{val:.6f}" if isinstance(val, float) and np.isfinite(val) else val)
            row.append(f"{row_data.get('elapsed', 0):.1f}")
            writer.writerow(row)


# ── Main grid search logic ──────────────────────────────────────────────────

def run_grid_search(
    strategy_name: str,
    grid_spec: dict,
    prices: np.ndarray,
    meta: PublicMeta,
    output_dir: str = ".",
):
    cls = grid_spec["class"]
    params = grid_spec["params"]
    param_keys = list(params.keys())
    total_combos = _combo_count(params)

    _, n_groups, _ = sliding_month_groups_shape(prices)

    print(f"\n{'='*60}")
    print(f"GRID SEARCH: {strategy_name}")
    print(f"  Strategy class: {cls.__name__}")
    print(f"  Parameters: {param_keys}")
    print(f"  Total combos: {total_combos:,}")
    print(f"  Holdout months: {n_groups}")
    print(f"{'='*60}\n")

    rows = []
    t_start_all = time.time()

    for i, combo in enumerate(_param_combos(params)):
        combo_str = ", ".join(f"{k}={_format_val(v)}" for k, v in combo.items())
        print(f"  [{i+1}/{total_combos}] {combo_str}", end=" ... ", flush=True)

        def factory(c=combo):
            return cls(**c)

        t0 = time.time()
        try:
            sharpes = evaluate_sliding_month_sharpes(prices, meta, factory, verbose=False)
            elapsed = time.time() - t0
            mean_s = float(np.mean(sharpes))
            var_s = float(np.var(sharpes, ddof=1)) if len(sharpes) >= 2 else 0.0
            std_s = float(np.std(sharpes, ddof=1)) if len(sharpes) >= 2 else 0.0
            min_s = float(np.min(sharpes))
            max_s = float(np.max(sharpes))

            print(f"mean={mean_s:+.4f}  std={std_s:.4f}  ({elapsed:.1f}s)")

            rows.append({
                "params": combo,
                "sharpes": list(sharpes),
                "mean": mean_s,
                "var": var_s,
                "std": std_s,
                "min_s": min_s,
                "max_s": max_s,
                "elapsed": elapsed,
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR: {e} ({elapsed:.1f}s)")
            traceback.print_exc()
            rows.append({
                "params": combo,
                "sharpes": [float("nan")] * n_groups,
                "mean": float("nan"),
                "var": float("nan"),
                "std": float("nan"),
                "min_s": float("nan"),
                "max_s": float("nan"),
                "elapsed": elapsed,
            })

    total_elapsed = time.time() - t_start_all
    print(f"\n  Finished {strategy_name}: {total_combos} combos in {total_elapsed:.0f}s")

    out_path = os.path.join(output_dir, f"grid_{strategy_name}.csv")
    write_results_csv(param_keys, rows, n_groups, out_path)
    print(f"  Saved: {out_path}")

    # Print top 5 summary
    rows_sorted = sorted(rows, key=lambda r: r.get("mean", -999), reverse=True)
    print(f"\n  Top 5 by Mean Sharpe:")
    for j, rd in enumerate(rows_sorted[:5]):
        p = ", ".join(f"{k}={_format_val(v)}" for k, v in rd["params"].items())
        print(f"    {j+1}. mean={rd['mean']:+.4f}  std={rd['std']:.4f}  | {p}")

    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grid search over portfolio strategies")
    parser.add_argument(
        "strategies", nargs="*", default=[],
        help="Strategy names to search (default: all). Use --list to see names.",
    )
    parser.add_argument("--list", action="store_true", help="List available strategies and exit")
    parser.add_argument("--output-dir", default=".", help="Directory for csv output")
    args = parser.parse_args()

    if args.list:
        print("Available strategies:")
        for name, spec in GRIDS.items():
            n = _combo_count(spec["params"])
            print(f"  {name:30s}  {n:>6,} combos  ({spec['class'].__name__})")
        return

    targets = args.strategies if args.strategies else list(GRIDS.keys())
    for t in targets:
        if t not in GRIDS:
            print(f"Unknown strategy: {t}. Use --list to see options.")
            sys.exit(1)

    print("Loading data...")
    prices = load_prices()
    meta = load_meta()
    print(f"  {prices.shape[0]:,} ticks, {prices.shape[1]} assets")

    os.makedirs(args.output_dir, exist_ok=True)

    output_files = []
    for name in targets:
        path = run_grid_search(name, GRIDS[name], prices, meta, args.output_dir)
        output_files.append(path)

    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")
    for f in output_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()