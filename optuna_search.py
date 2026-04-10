#!/usr/bin/env python3
"""Bayesian hyperparameter search using Optuna (TPE sampler).

Produces one .csv per strategy sorted by mean Sharpe descending.
Much more efficient than grid search for high-dimensional spaces —
typically finds top regions in 200-500 trials instead of 30k+.

Usage:
    python optuna_search.py                              # both strategies, 500 trials each
    python optuna_search.py momentum_tilt                # just one
    python optuna_search.py --n-trials 1000              # more trials
    python optuna_search.py --list                       # show strategies
    python optuna_search.py --objective mean_sharpe      # maximize mean Sharpe (default)
    python optuna_search.py --objective sharpe_per_risk  # maximize mean/std ratio

Requires: prices.csv, meta.csv, submission.py, months_validate.py in cwd.
         pip install optuna
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import optuna
from optuna.samplers import TPESampler

from submission import (
    PublicMeta, StrategyBase, load_meta, load_prices,
    MomentumTilt, IntradayRiskParity, N_ASSETS, VolatilityFlatOverlay
)
from months_validate import (
    evaluate_sliding_month_sharpes, sliding_month_groups_shape,
)

# ── Strategy search spaces ───────────────────────────────────────────────────

def suggest_momentum_tilt(trial: optuna.Trial) -> dict:
    return {
        "momentum_lookback": trial.suggest_int("momentum_lookback", 5, 200),
        "rebalance_freq":    trial.suggest_int("rebalance_freq", 3, 80),
        "spread_penalty":    trial.suggest_float("spread_penalty", 0.0, 0.1),
        "borrow_penalty":    trial.suggest_float("borrow_penalty", 0.0, 0.1),
        "zero_negative":     trial.suggest_categorical("zero_negative", [True, False]),
        "blend_rate":        trial.suggest_float("blend_rate", 0.2, 1.0),
        "equal_weight_blend": trial.suggest_float("equal_weight_blend", 0.0, 0.6),
    }

def suggest_momentum_vol_flat(trial: optuna.Trial) -> dict:
    # Inner MomentumTilt Params
    momentum_params = {
        "momentum_lookback": trial.suggest_int("momentum_lookback", 10, 150),
        "rebalance_freq":    trial.suggest_int("rebalance_freq", 5, 60),
        "spread_penalty":    trial.suggest_float("spread_penalty", 0.0, 2.0),
        "borrow_penalty":    trial.suggest_float("borrow_penalty", 0.0, 2.0),
        "zero_negative":     trial.suggest_categorical("zero_negative", [True, False]),
        "blend_rate":        trial.suggest_float("blend_rate", 0.1, 1.0),
        "equal_weight_blend": trial.suggest_float("equal_weight_blend", 0.0, 0.5),
    }
    
    # Outer VolatilityFlatOverlay Params
    overlay_params = {
        "vol_lookback_days": trial.suggest_int("vol_lookback_days", 2, 20),
        "spike_multiplier":  trial.suggest_float("spike_multiplier", 1.0, 2.0),
        "floor_ann_vol":     trial.suggest_float("floor_ann_vol", 0.0, 0.2),
    }
    
    return {"inner_params": momentum_params, "overlay_params": overlay_params}


def suggest_intraday_risk_parity(trial: optuna.Trial) -> dict:
    return {
        "cov_lookback_days": trial.suggest_int("cov_lookback_days", 10, 150),
        "vol_lookback_days": trial.suggest_int("vol_lookback_days", 5, 60),
        "rebalance_freq":    trial.suggest_int("rebalance_freq", 3, 60),
        "blend_rate":        trial.suggest_float("blend_rate", 0.1, 1.0),
        "vol_target":        trial.suggest_float("vol_target", 0.03, 0.30),
        "vol_cap":           trial.suggest_float("vol_cap", 0.5, 1.0),
        "vol_floor":         trial.suggest_float("vol_floor", 0.05, 0.5),
        "use_intraday_cov":  True,
        "spread_penalty":    trial.suggest_float("spread_penalty", 0.0, 2.0),
        "borrow_penalty":    trial.suggest_float("borrow_penalty", 0.0, 2.0),
    }

STRATEGIES: dict[str, dict] = {
    "momentum_tilt": {
        "class": MomentumTilt,
        "suggest": suggest_momentum_tilt,
    },
    "intraday_risk_parity": {
        "class": IntradayRiskParity,
        "suggest": suggest_intraday_risk_parity,
    },
    "momentum_vol_flat": {
        "class": VolatilityFlatOverlay, # Outer wrapper
        "suggest": suggest_momentum_vol_flat,
    }
}

def make_objective(
    cls,
    suggest_fn,
    prices: np.ndarray,
    meta: PublicMeta,
    objective_mode: str,
):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial)

        # Ensure vol_floor < vol_cap for intraday_risk_parity
        if "vol_floor" in params and "vol_cap" in params:
            if params["vol_floor"] >= params["vol_cap"]:
                raise optuna.TrialPruned("vol_floor >= vol_cap")

        def factory():
            # Check if this is our nested momentum_vol_flat strategy
            if "inner_params" in params:
                inner = MomentumTilt(**params["inner_params"])
                return VolatilityFlatOverlay(inner, **params["overlay_params"])
            
            # Standard non-nested strategies
            return cls(**params)

        try:
            sharpes = evaluate_sliding_month_sharpes(prices, meta, factory, verbose=False)
        except Exception as e:
            raise optuna.TrialPruned(str(e))

        mean_s = float(np.mean(sharpes))
        std_s = float(np.std(sharpes, ddof=1)) if len(sharpes) >= 2 else 1e-6

        # Store extras for CSV output
        trial.set_user_attr("sharpes", list(sharpes))
        trial.set_user_attr("mean_sharpe", mean_s)
        trial.set_user_attr("var_sharpe", float(np.var(sharpes, ddof=1)) if len(sharpes) >= 2 else 0.0)
        trial.set_user_attr("std_sharpe", std_s)
        trial.set_user_attr("min_sharpe", float(np.min(sharpes)))
        trial.set_user_attr("max_sharpe", float(np.max(sharpes)))

        if objective_mode == "sharpe_per_risk":
            return mean_s / max(std_s, 1e-6)
        else:
            return mean_s

    return objective


# ── CSV output ───────────────────────────────────────────────────────────────

def write_csv(
    strategy_name: str,
    study: optuna.Study,
    n_holdout_months: int,
    output_path: str,
):
    # Collect all completed trials
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        print("  No completed trials to write.")
        return

    param_keys = sorted(trials[0].params.keys())
    headers = (
        ["trial"]
        + param_keys
        + [f"Month_{i}" for i in range(n_holdout_months)]
        + ["Mean_Sharpe", "Var_Sharpe", "Std_Sharpe", "Min_Sharpe", "Max_Sharpe",
           "Objective_Value", "Duration_s"]
    )

    # Sort by mean Sharpe descending
    trials_sorted = sorted(trials, key=lambda t: t.user_attrs.get("mean_sharpe", -999), reverse=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for t in trials_sorted:
            row = [t.number]
            for k in param_keys:
                row.append(t.params.get(k, ""))
            sharpes = t.user_attrs.get("sharpes", [])
            for mi in range(n_holdout_months):
                val = sharpes[mi] if mi < len(sharpes) else ""
                row.append(f"{val:.6f}" if isinstance(val, float) and np.isfinite(val) else val)
            for attr in ["mean_sharpe", "var_sharpe", "std_sharpe", "min_sharpe", "max_sharpe"]:
                val = t.user_attrs.get(attr)
                row.append(f"{val:.6f}" if isinstance(val, (int, float)) and np.isfinite(val) else val)
            row.append(f"{t.value:.6f}" if t.value is not None and np.isfinite(t.value) else t.value)
            dur = t.duration.total_seconds() if t.duration else 0
            row.append(f"{dur:.1f}")
            writer.writerow(row)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_search(
    strategy_name: str,
    spec: dict,
    prices: np.ndarray,
    meta: PublicMeta,
    n_trials: int,
    objective_mode: str,
    output_dir: str,
    seed: int,
) -> str:
    _, n_groups, _ = sliding_month_groups_shape(prices)

    print(f"\n{'='*60}")
    print(f"OPTUNA SEARCH: {strategy_name}")
    print(f"  Trials: {n_trials}")
    print(f"  Objective: {objective_mode}")
    print(f"  Holdout months: {n_groups}")
    print(f"{'='*60}\n")

    sampler = TPESampler(seed=seed, n_startup_trials=max(20, n_trials // 10))
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=strategy_name,
    )

    objective = make_objective(spec["class"], spec["suggest"], prices, meta, objective_mode)

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    elapsed = time.time() - t0

    # Results
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print(f"\n  Completed: {len(completed)}, Pruned: {len(pruned)}, Time: {elapsed:.0f}s")

    if completed:
        best = study.best_trial
        print(f"\n  Best trial #{best.number}:")
        print(f"    Objective value: {best.value:.6f}")
        print(f"    Mean Sharpe:     {best.user_attrs.get('mean_sharpe', 'n/a')}")
        print(f"    Std Sharpe:      {best.user_attrs.get('std_sharpe', 'n/a')}")
        print(f"    Params:")
        for k, v in sorted(best.params.items()):
            print(f"      {k}: {v}")

        # Top 5
        by_mean = sorted(completed, key=lambda t: t.user_attrs.get("mean_sharpe", -999), reverse=True)
        print(f"\n  Top 5 by Mean Sharpe:")
        for j, t in enumerate(by_mean[:5]):
            ms = t.user_attrs.get("mean_sharpe", float("nan"))
            ss = t.user_attrs.get("std_sharpe", float("nan"))
            print(f"    {j+1}. trial={t.number}  mean={ms:+.4f}  std={ss:.4f}")

    out_path = os.path.join(output_dir, f"optuna_{strategy_name}.csv")
    write_csv(strategy_name, study, n_groups, out_path)
    print(f"\n  Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search")
    parser.add_argument("strategies", nargs="*", default=[])
    parser.add_argument("--list", action="store_true", help="List strategies and exit")
    parser.add_argument("--n-trials", type=int, default=500, help="Trials per strategy (default 500)")
    # parser.add_argument("--objective", default="mean_sharpe",
    #                     choices=["mean_sharpe", "sharpe_per_risk"],
    #                     help="Objective to maximize")
    parser.add_argument("--objective", default="sharpe_per_risk",
                    choices=["mean_sharpe", "sharpe_per_risk"],
                    help="Objective to maximize")
    parser.add_argument("--output-dir", default=".", help="Directory for csv output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampler")
    parser.add_argument("--quiet", action="store_true", help="Suppress Optuna trial logs")
    args = parser.parse_args()

    if args.list:
        print("Available strategies:")
        for name in STRATEGIES:
            print(f"  {name}")
        return

    if args.quiet:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    targets = args.strategies if args.strategies else list(STRATEGIES.keys())
    for t in targets:
        if t not in STRATEGIES:
            print(f"Unknown strategy: {t}. Use --list to see options.")
            sys.exit(1)

    print("Loading data...")
    prices = load_prices()
    meta = load_meta()
    print(f"  {prices.shape[0]:,} ticks, {prices.shape[1]} assets")

    os.makedirs(args.output_dir, exist_ok=True)

    output_files = []
    for name in targets:
        path = run_search(
            name, STRATEGIES[name], prices, meta,
            n_trials=args.n_trials,
            objective_mode=args.objective,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        output_files.append(path)

    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")
    for f in output_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()