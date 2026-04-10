"""Compare all registered strategies using the same sliding-month evaluation as months_validate.

Runs train months j..j+23 and holdout month j+24 for each disjoint holdout month, then reports
mean / variance / std of holdout Sharpes per strategy.

Usage:
    python compare_strategies_months.py
    python compare_strategies_months.py baseline intraday_risk_parity
    python compare_strategies_months.py --csv sharpes_long.csv

Plots (matplotlib): python plot_strategy_comparison.py -o figures/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from months_validate import (
    TRAIN_MONTHS_PER_BLOCK,
    evaluate_sliding_month_sharpes,
    sliding_month_groups_shape,
)
from submission import N_ASSETS, PublicMeta, STRATEGY_REGISTRY, load_meta, load_prices


def evaluate_strategies_sharpes(
    prices: np.ndarray,
    meta: PublicMeta,
    names: list[str] | None,
) -> tuple[dict[str, np.ndarray], dict[str, str], int]:
    """Run sliding-month evaluation for each strategy.

    Returns:
        sharpes: name -> 1d array of annualized Sharpes per holdout month
        errors: name -> error message for strategies that failed
        n_groups: number of holdout months (length of each successful array)
    """
    _, n_groups, _ = sliding_month_groups_shape(prices)
    name_list = sorted(STRATEGY_REGISTRY.keys()) if not names else list(names)
    sharpes: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}

    for name in name_list:
        if name not in STRATEGY_REGISTRY:
            errors[name] = "unknown strategy"
            continue
        factory = STRATEGY_REGISTRY[name]
        try:
            arr = evaluate_sliding_month_sharpes(prices, meta, factory, verbose=False)
            sharpes[name] = arr
        except Exception as e:
            errors[name] = str(e)

    return sharpes, errors, n_groups


def write_sharpes_csv(
    path: Path,
    sharpes: dict[str, np.ndarray],
    *,
    train_months: int = TRAIN_MONTHS_PER_BLOCK,
) -> None:
    """Long-format CSV: strategy, group_idx, holdout_month_index, sharpe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "group_idx", "holdout_month_index", "sharpe"])
        for name in sorted(sharpes.keys()):
            arr = sharpes[name]
            for j, sr in enumerate(arr.tolist()):
                w.writerow([name, j, train_months + j, sr])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare strategies on sliding 24m train / 1m holdout (see months_validate.py)."
    )
    parser.add_argument(
        "strategies",
        nargs="*",
        metavar="NAME",
        help="Strategy keys from STRATEGY_REGISTRY (default: all)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        metavar="FILE",
        help="Write long-format Sharpes to CSV (for plotting or Excel)",
    )
    args = parser.parse_args()

    print("Loading data...")
    prices = load_prices()
    meta = load_meta()
    assert prices.shape[1] == N_ASSETS, f"Expected {N_ASSETS} assets, got {prices.shape[1]}"

    total_months, n_groups, _ = sliding_month_groups_shape(prices)
    print(
        f"  {prices.shape[0]:,} ticks; ~{total_months} trading months; "
        f"{n_groups} holdout month(s) per strategy\n"
    )

    names = list(args.strategies) if args.strategies else sorted(STRATEGY_REGISTRY.keys())
    unknown = [n for n in names if n not in STRATEGY_REGISTRY]
    if unknown:
        print("Unknown strategy name(s):", ", ".join(unknown), file=sys.stderr)
        print("Available:", ", ".join(sorted(STRATEGY_REGISTRY.keys())), file=sys.stderr)
        raise SystemExit(1)

    sharpes, eval_errors, _ = evaluate_strategies_sharpes(prices, meta, names)

    rows: list[tuple[str, float, float, float, int, str]] = []
    for name in names:
        if name in eval_errors and name not in sharpes:
            rows.append((name, float("nan"), float("nan"), float("nan"), n_groups, f"ERROR: {eval_errors[name]}"))
            print(f"Running {name!r}...")
            print(f"  failed: {eval_errors[name]}\n")
            continue

        s = sharpes[name]
        mean_sr = float(np.mean(s))
        if n_groups >= 2:
            var_sr = float(np.var(s, ddof=1))
            std_sr = float(np.std(s, ddof=1))
        else:
            var_sr = float("nan")
            std_sr = float("nan")
        rows.append((name, mean_sr, var_sr, std_sr, len(s), ""))
        print(f"Running {name!r}...")
        print(f"  mean={mean_sr:+.4f}  var={var_sr:.6f}  std={std_sr:.4f}\n")

    if args.csv is not None and sharpes:
        write_sharpes_csv(args.csv, sharpes)
        print(f"Wrote {args.csv}\n")

    w_name = max(len(r[0]) for r in rows)
    print("=" * (w_name + 52))
    print(f"{'strategy':<{w_name}}  {'mean_Sharpe':>12}  {'var_Sharpe':>14}  {'std_Sharpe':>10}  n")
    print("=" * (w_name + 52))
    for name, mean_sr, var_sr, std_sr, n, err in rows:
        if err:
            print(f"{name:<{w_name}}  {err}")
        else:
            vs = f"{var_sr:.6f}" if n_groups >= 2 else "n/a"
            ss = f"{std_sr:.4f}" if n_groups >= 2 else "n/a"
            print(f"{name:<{w_name}}  {mean_sr:+12.4f}  {vs:>14}  {ss:>10}  {n}")
    print("=" * (w_name + 52))


if __name__ == "__main__":
    main()
