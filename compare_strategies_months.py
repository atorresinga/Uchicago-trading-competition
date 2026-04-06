"""Compare all registered strategies using the same sliding-month evaluation as months_validate.

Runs train months j..j+23 and holdout month j+24 for each disjoint holdout month, then reports
mean / variance / std of holdout Sharpes per strategy.

Usage:
    python compare_strategies_months.py
    python compare_strategies_months.py baseline intraday_risk_parity
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from months_validate import evaluate_sliding_month_sharpes, sliding_month_groups_shape
from submission import N_ASSETS, STRATEGY_REGISTRY, load_meta, load_prices


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

    rows: list[tuple[str, float, float, float, int, str]] = []

    for name in names:
        factory = STRATEGY_REGISTRY[name]
        print(f"Running {name!r}...")
        try:
            sharpes = evaluate_sliding_month_sharpes(prices, meta, factory, verbose=False)
        except Exception as e:
            rows.append((name, float("nan"), float("nan"), float("nan"), n_groups, f"ERROR: {e}"))
            print(f"  failed: {e}\n")
            continue

        mean_sr = float(np.mean(sharpes))
        if n_groups >= 2:
            var_sr = float(np.var(sharpes, ddof=1))
            std_sr = float(np.std(sharpes, ddof=1))
        else:
            var_sr = float("nan")
            std_sr = float("nan")
        rows.append((name, mean_sr, var_sr, std_sr, len(sharpes), ""))
        print(f"  mean={mean_sr:+.4f}  var={var_sr:.6f}  std={std_sr:.4f}\n")

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
