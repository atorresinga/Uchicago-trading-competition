"""Grid search hyperparameters for IntradayRiskParity on sliding-month evaluation.

Uses the same mechanics as months_validate (24m train, 1m holdout, disjoint months).
Scores each configuration by mean holdout Sharpe, optionally minus a stability penalty
on the std of monthly Sharpes.

Usage:
    python grid_search_intraday.py
    python grid_search_intraday.py --out grid_results.csv
    python grid_search_intraday.py --grid my_grid.json --out results.csv
    python grid_search_intraday.py --limit 3              # smoke test (first 3 combos)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from months_validate import evaluate_sliding_month_summary, sliding_month_groups_shape
from submission import IntradayRiskParity, N_ASSETS, load_meta, load_prices

# Fixed hyperparameters merged into every grid point (competition-style defaults).
_FIXED: dict[str, Any] = {
    "vol_cap": 1.0,
    "vol_floor": 0.2,
    "use_intraday_cov": True,
}

# Default search grid (~36 combos). Keys must match IntradayRiskParity.__init__.
DEFAULT_GRID: dict[str, list[Any]] = {
    "cov_lookback_days": [45, 60, 75],
    "vol_lookback_days": [15, 20],
    "rebalance_freq": [20],
    "blend_rate": [0.4, 0.55, 0.7],
    "vol_target": [0.12, 0.15],
    "spread_penalty": [0.5],
    "borrow_penalty": [1.0],
}


def _grid_dict_to_rows(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    rows = []
    for combo in itertools.product(*values):
        row = dict(_FIXED)
        row.update(dict(zip(keys, combo)))
        rows.append(row)
    return rows


def _make_factory(params: dict[str, Any]):
    def factory() -> IntradayRiskParity:
        return IntradayRiskParity(**params)

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search IntradayRiskParity (sliding months).")
    parser.add_argument(
        "--grid",
        type=Path,
        metavar="FILE.json",
        help="JSON object: param name -> list of values (merged with fixed vol_cap/vol_floor/use_intraday_cov)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("grid_search_intraday_results.csv"),
        help="Output CSV path (default: grid_search_intraday_results.csv)",
    )
    parser.add_argument(
        "--stability-weight",
        type=float,
        default=0.0,
        metavar="LAMBDA",
        help="Score = mean_sharpe - LAMBDA * std_sharpe (default: 0 = mean only)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="If >0, only run the first N combinations (debug / smoke test)",
    )
    args = parser.parse_args()

    if args.grid is not None:
        grid = json.loads(args.grid.read_text(encoding="utf-8"))
        if not isinstance(grid, dict):
            raise SystemExit("--grid must be a JSON object of lists")
    else:
        grid = DEFAULT_GRID

    rows_params = _grid_dict_to_rows(grid)
    n_total = len(rows_params)
    if args.limit > 0:
        rows_params = rows_params[: args.limit]
        print(f"Limit: running {len(rows_params)} of {n_total} combinations")
    else:
        print(f"Grid: {n_total} combinations")

    print("Loading data...")
    prices = load_prices()
    meta = load_meta()
    assert prices.shape[1] == N_ASSETS, f"Expected {N_ASSETS} assets, got {prices.shape[1]}"
    _, n_groups, _ = sliding_month_groups_shape(prices)
    print(f"  {n_groups} holdout months per configuration\n")

    param_keys = sorted({k for r in rows_params for k in r.keys()})
    fieldnames = param_keys + [
        "mean_sharpe",
        "std_sharpe",
        "score",
        "n_blowups",
        "n_holdout_months",
    ]

    results: list[dict[str, Any]] = []
    lam = float(args.stability_weight)

    for i, params in enumerate(rows_params):
        factory = _make_factory(params)
        try:
            sharpes, blowups = evaluate_sliding_month_summary(
                prices, meta, factory, verbose=False
            )
        except Exception as e:
            print(f"[{i + 1}/{len(rows_params)}] FAIL {params}: {e}")
            row = {k: params.get(k) for k in param_keys}
            row.update(
                {
                    "mean_sharpe": float("nan"),
                    "std_sharpe": float("nan"),
                    "score": float("nan"),
                    "n_blowups": -1,
                    "n_holdout_months": n_groups,
                    "error": str(e),
                }
            )
            results.append(row)
            continue

        mean_sr = float(np.mean(sharpes))
        std_sr = float(np.std(sharpes, ddof=1)) if n_groups >= 2 else 0.0
        score = mean_sr - lam * std_sr
        row = {k: params.get(k) for k in param_keys}
        row.update(
            {
                "mean_sharpe": mean_sr,
                "std_sharpe": std_sr,
                "score": score,
                "n_blowups": blowups,
                "n_holdout_months": n_groups,
            }
        )
        results.append(row)
        print(
            f"[{i + 1}/{len(rows_params)}] mean={mean_sr:+.4f} std={std_sr:.4f} "
            f"score={score:+.4f} blowups={blowups}  {params}"
        )

    # Write CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nWrote {args.out}")

    valid = [r for r in results if np.isfinite(r.get("score", float("nan")))]
    if valid:
        best = max(valid, key=lambda r: r["score"])
        print("\nBest by score (mean_sharpe - lambda*std):")
        print(f"  score={best['score']:+.4f}  mean={best['mean_sharpe']:+.4f}  std={best['std_sharpe']:.4f}  blowups={best['n_blowups']}")
        pk = [k for k in param_keys if k in best and k not in ("mean_sharpe", "std_sharpe", "score", "n_blowups", "n_holdout_months")]
        print(f"  params: {{{', '.join(f'{k!r}: {best[k]!r}' for k in pk)}}}")


if __name__ == "__main__":
    main()
