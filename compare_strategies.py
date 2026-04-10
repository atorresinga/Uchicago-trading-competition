"""Compare all strategies: summary table + performance plots.

Reads the best hyperparameters from each grid search CV file,
runs full backtests across all 3 CV folds, and generates:
  1. A summary table (printed + CSV)
  2. Cumulative wealth plots per fold
  3. A combined bar chart of mean Sharpe across strategies
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from submission import (
    Baseline, RiskParity, VolTargetRiskParity, CostAwareTilt,
    MomentumTilt, IntradayRiskParity, load_prices, load_meta,
)
from validate import (
    run_backtest, annualized_sharpe,
    TRADING_DAYS_PER_YEAR, TICKS_PER_DAY,
)

# ── Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
prices = load_prices()
meta = load_meta()
ticks_per_year = TRADING_DAYS_PER_YEAR * TICKS_PER_DAY

# ── Define folds (same as grid searches) ──────────────────────────────────
folds = []
total_years = prices.shape[0] // ticks_per_year
for k in range(2, total_years):
    train_end = k * ticks_per_year
    test_end = (k + 1) * ticks_per_year
    if test_end > prices.shape[0]:
        break
    folds.append((f"Year {k}", train_end, test_end))

print(f"  {len(folds)} CV folds\n")

# ── Read best hyperparams from each grid search ──────────────────────────
# RiskParity
df_rp = pd.read_csv("grid_search_cv_results.csv").sort_values("mean_sharpe", ascending=False)
best_rp = df_rp.iloc[0]

# VolTargetRiskParity
df_vt = pd.read_csv("grid_search_voltarget_cv_results.csv").sort_values("mean_sharpe", ascending=False)
best_vt = df_vt.iloc[0]

# CostAwareTilt
df_ct = pd.read_csv("grid_search_costtilt_cv_results.csv").sort_values("mean_sharpe", ascending=False)
best_ct = df_ct.iloc[0]

# MomentumTilt
df_mt = pd.read_csv("grid_search_momentum_cv_results.csv").sort_values("mean_sharpe", ascending=False)
best_mt = df_mt.iloc[0]

# IntradayRiskParity
df_ir = pd.read_csv("grid_search_intraday_rp_cv_results.csv").sort_values("mean_sharpe", ascending=False)
best_ir = df_ir.iloc[0]

# ── Strategy factory with best params ────────────────────────────────────
strategies = {
    "Equal Weight": lambda: Baseline(),
    "Risk Parity": lambda: RiskParity(
        lookback=int(best_rp["lookback"]),
        rebalance_freq=int(best_rp["rebalance_freq"]),
        blend_rate=float(best_rp["blend_rate"]),
    ),
    "VolTarget RP": lambda: VolTargetRiskParity(
        lookback=int(best_vt["lookback"]),
        rebalance_freq=int(best_vt["rebalance_freq"]),
        blend_rate=float(best_vt["blend_rate"]),
        vol_lookback=int(best_vt["vol_lookback"]),
        vol_target=float(best_vt["vol_target"]),
        vol_cap=float(best_vt["vol_cap"]),
        vol_floor=float(best_vt["vol_floor"]),
    ),
    "Cost-Aware Tilt": lambda: CostAwareTilt(
        sharpe_weight=float(best_ct["sharpe_weight"]),
        spread_penalty=float(best_ct["spread_penalty"]),
        borrow_penalty=float(best_ct["borrow_penalty"]),
        min_weight=float(best_ct["min_weight"]),
        zero_negative=bool(best_ct["zero_negative"]),
    ),
    "Momentum Tilt": lambda: MomentumTilt(
        momentum_lookback=int(best_mt["momentum_lookback"]),
        rebalance_freq=int(best_mt["rebalance_freq"]),
        spread_penalty=float(best_mt["spread_penalty"]),
        borrow_penalty=float(best_mt["borrow_penalty"]),
        zero_negative=bool(best_mt["zero_negative"]),
        blend_rate=float(best_mt["blend_rate"]),
        equal_weight_blend=float(best_mt["equal_weight_blend"]),
    ),
    "Intraday RP": lambda: IntradayRiskParity(
        cov_lookback_days=int(best_ir["cov_lookback"]),
        vol_lookback_days=int(best_ir["vol_lookback"]),
        rebalance_freq=int(best_ir["rebalance_freq"]),
        blend_rate=float(best_ir["blend_rate"]),
        vol_target=float(best_ir["vol_target"]),
        vol_cap=1.0,
        vol_floor=float(best_ir["vol_floor"]),
        use_intraday_cov=bool(best_ir["intraday_cov"]),
        spread_penalty=float(best_ir["spread_penalty"]),
        borrow_penalty=float(best_ir["borrow_penalty"]),
    ),
}

# ── Run backtests ────────────────────────────────────────────────────────
print("Running backtests...")
all_results = {}  # {strategy_name: {fold_label: result_dict}}

for name, make_strat in strategies.items():
    all_results[name] = {}
    for fold_label, train_end, test_end in folds:
        strat = make_strat()
        result = run_backtest(
            prices[:train_end], prices[train_end:test_end], strat, meta
        )
        all_results[name][fold_label] = result
    print(f"  Done: {name}")

# ── Build summary table ──────────────────────────────────────────────────
print("\nBuilding summary table...")
rows = []
for name in strategies:
    fold_sharpes = []
    fold_returns = []
    fold_costs = []
    fold_dds = []

    for fold_label, _, _ in folds:
        r = all_results[name][fold_label]
        dr = r["daily_returns"]
        sharpe = annualized_sharpe(dr)
        total_ret = float(np.prod(1.0 + dr) - 1.0)
        total_cost = float(np.sum(r["daily_costs"]))
        cum = np.cumprod(1.0 + dr)
        max_dd = float(np.min(
            np.minimum.accumulate(cum) / np.maximum.accumulate(cum) - 1.0
        ))
        fold_sharpes.append(sharpe)
        fold_returns.append(total_ret)
        fold_costs.append(total_cost)
        fold_dds.append(max_dd)

    rows.append({
        "Strategy": name,
        "Mean Sharpe": np.mean(fold_sharpes),
        "Std Sharpe": np.std(fold_sharpes, ddof=1),
        "Min Sharpe": np.min(fold_sharpes),
        "Max Sharpe": np.max(fold_sharpes),
        "Fold 0 Sharpe": fold_sharpes[0],
        "Fold 1 Sharpe": fold_sharpes[1],
        "Fold 2 Sharpe": fold_sharpes[2],
        "Mean Return": np.mean(fold_returns),
        "Mean Cost": np.mean(fold_costs),
        "Worst Drawdown": np.min(fold_dds),
    })

df_summary = pd.DataFrame(rows).sort_values("Mean Sharpe", ascending=False)
df_summary.to_csv("strategy_comparison.csv", index=False)

# ── Print table ──────────────────────────────────────────────────────────
print("\n" + "=" * 130)
print("STRATEGY COMPARISON (3-fold CV, best hyperparameters per strategy)")
print("=" * 130)
print(f"{'Strategy':<18} {'Mean Sharpe':>12} {'Std Sharpe':>12} {'Min Sharpe':>12} "
      f"{'Fold 0':>10} {'Fold 1':>10} {'Fold 2':>10} "
      f"{'Mean Ret':>10} {'Mean Cost':>10} {'Worst DD':>10}")
print("-" * 130)
for _, row in df_summary.iterrows():
    print(f"{row['Strategy']:<18} {row['Mean Sharpe']:>+12.4f} {row['Std Sharpe']:>12.4f} "
          f"{row['Min Sharpe']:>+12.4f} {row['Fold 0 Sharpe']:>+10.4f} "
          f"{row['Fold 1 Sharpe']:>+10.4f} {row['Fold 2 Sharpe']:>+10.4f} "
          f"{row['Mean Return']:>+10.2%} {row['Mean Cost']:>10.4%} "
          f"{row['Worst Drawdown']:>10.2%}")
print("=" * 130)

# ── Plot 1: Mean Sharpe bar chart ────────────────────────────────────────
print("\nGenerating plots...")

colors = ['#636363', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3', '#e41a1c']
fig, ax = plt.subplots(figsize=(12, 6))

names_sorted = df_summary["Strategy"].tolist()
mean_sharpes = df_summary["Mean Sharpe"].tolist()
std_sharpes = df_summary["Std Sharpe"].tolist()

bars = ax.bar(range(len(names_sorted)), mean_sharpes, yerr=std_sharpes,
              capsize=5, color=colors[:len(names_sorted)], alpha=0.85,
              edgecolor="black", linewidth=0.5)

ax.set_xticks(range(len(names_sorted)))
ax.set_xticklabels(names_sorted, fontsize=10, rotation=15, ha="right")
ax.set_ylabel("Annualized Sharpe Ratio", fontsize=12)
ax.set_title("Strategy Comparison: Mean Sharpe (3-Fold CV) ± 1 Std Dev", fontsize=14)
ax.axhline(0, color="black", linewidth=0.5)
ax.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bar, val in zip(bars, mean_sharpes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("comparison_sharpe_bars.png", dpi=150)
print("  Saved comparison_sharpe_bars.png")

# ── Plot 2: Cumulative wealth per fold ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

for fi, (fold_label, _, _) in enumerate(folds):
    ax = axes[fi]
    for si, name in enumerate(strategies):
        dr = all_results[name][fold_label]["daily_returns"]
        cum = np.cumprod(1.0 + dr)
        ax.plot(cum, label=name, color=colors[si], linewidth=1.5, alpha=0.85)

    ax.set_title(f"{fold_label} (Test)", fontsize=12)
    ax.set_xlabel("Trading Day")
    if fi == 0:
        ax.set_ylabel("Cumulative Wealth ($1 start)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")

plt.suptitle("Cumulative Wealth Across CV Folds (net of all costs)", fontsize=14)
plt.tight_layout()
plt.savefig("comparison_wealth_curves.png", dpi=150)
print("  Saved comparison_wealth_curves.png")

# ── Plot 3: Per-fold Sharpe grouped bar chart ────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

n_strats = len(strategies)
n_folds = len(folds)
x = np.arange(n_strats)
width = 0.25

strat_names = list(strategies.keys())
for fi, (fold_label, _, _) in enumerate(folds):
    fold_sharpes = []
    for name in strat_names:
        dr = all_results[name][fold_label]["daily_returns"]
        fold_sharpes.append(annualized_sharpe(dr))
    offset = (fi - 1) * width
    ax.bar(x + offset, fold_sharpes, width, label=fold_label,
           alpha=0.8, edgecolor="black", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(strat_names, fontsize=10, rotation=15, ha="right")
ax.set_ylabel("Annualized Sharpe", fontsize=12)
ax.set_title("Per-Fold Sharpe Ratio by Strategy", fontsize=14)
ax.legend(fontsize=10)
ax.axhline(0, color="black", linewidth=0.5)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("comparison_per_fold_sharpe.png", dpi=150)
print("  Saved comparison_per_fold_sharpe.png")

print("\nDone!")
