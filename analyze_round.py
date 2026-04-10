"""
analyze_round.py — Post-round analysis for Case 1 bot.

Reads the most recent logs/round_*.jsonl file and produces a single 
HTML report with charts, including MtM PnL, Cash, and Exposure.

Usage:
    python analyze_round.py                       # latest round
    python analyze_round.py logs/round_xxx.jsonl  # specific file
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import base64
from io import BytesIO


# ---------------------------------------------------------------------------
# Load & Process
# ---------------------------------------------------------------------------

def load_log(path: Path) -> list[dict]:
    events = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def latest_log() -> Path:
    logs = sorted(Path("logs").glob("round_*.jsonl"))
    if not logs:
        sys.exit("No logs found in ./logs/. Run the bot first.")
    return logs[-1]


def enrich_with_cash_and_pnl(events: list[dict]) -> list[dict]:
    """Reconstruct cash flow from fills and swaps to calculate MtM PnL."""
    cash = 0.0
    prev_pos = {}
    
    for e in events:
        if e["kind"] == "fill":
            pos = e.get("positions") or {}
            sym = "?"
            diff_qty = 0
            for k, v in pos.items():
                if prev_pos.get(k, 0) != v:
                    sym = k
                    diff_qty = v - prev_pos.get(k, 0)
                    break
            
            # diff_qty > 0 means we bought (cash decreases)
            # diff_qty < 0 means we sold (cash increases)
            if sym != "?":
                cash -= diff_qty * e.get("price", 0)
            prev_pos = pos

        elif e["kind"] == "swap":
            # ETF creation/redemption costs 5 cash per swap
            if e.get("success"):
                cash -= 5.0
                
        elif e["kind"] == "snapshot":
            e["calc_cash"] = cash
            mids = e.get("mids") or {}
            pos = e.get("positions") or {}
            
            # MtM = Cash + Value of all holdings at current mid price
            mtm = cash
            gross_exposure = 0.0
            for k, v in pos.items():
                if k in mids and mids[k] is not None:
                    position_value = v * mids[k]
                    mtm += position_value
                    gross_exposure += abs(position_value)
                    
            e["calc_mtm"] = mtm
            e["calc_gross_exposure"] = gross_exposure
            
    return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def to_relative_seconds(ts_series: pd.Series) -> pd.Series:
    return ts_series - ts_series.min()


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def chart_pnl(snapshots: pd.DataFrame) -> str:
    if snapshots.empty or "calc_mtm" not in snapshots.columns:
        return ""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(snapshots["t_rel"], snapshots["calc_cash"], label="Cash Balance", color="#27ae60", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.plot(snapshots["t_rel"], snapshots["calc_mtm"], label="Mark-to-Market PnL", color="#2980b9", linewidth=2.5)
    
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Seconds since round start")
    ax.set_ylabel("Value ($)")
    ax.set_title("Cash vs. True PnL (Mark-to-Market)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)


def chart_exposure(snapshots: pd.DataFrame) -> str:
    if snapshots.empty or "calc_gross_exposure" not in snapshots.columns:
        return ""
    fig, ax = plt.subplots(figsize=(10, 3))
    
    ax.fill_between(snapshots["t_rel"], 0, snapshots["calc_gross_exposure"], color="#e74c3c", alpha=0.3)
    ax.plot(snapshots["t_rel"], snapshots["calc_gross_exposure"], color="#c0392b", linewidth=1.5, label="Gross Capital Exposure")
    
    ax.set_xlabel("Seconds since round start")
    ax.set_ylabel("Absolute Market Value ($)")
    ax.set_title("Gross Inventory Exposure (Risk)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)


def chart_fv_and_rates(snapshots: pd.DataFrame) -> str:
    if snapshots.empty:
        return ""
    
    # Extract FV and A mid
    rows = []
    for _, r in snapshots.iterrows():
        mids = r.get("mids") or {}
        rows.append({
            "t": r["t_rel"], 
            "mid_A": mids.get("A", None),
            "fv_A": r.get("fv_A", None),
            "rate_exp": r.get("rate_exp_bps", 0)
        })
        
    df = pd.DataFrame(rows).dropna(subset=["mid_A", "fv_A"])
    if df.empty:
        return ""

    fig, ax1 = plt.subplots(figsize=(10, 4))
    
    # Plot Asset A vs FV
    ax1.plot(df["t"], df["mid_A"], label="Asset A Mid", color="#8e44ad", linewidth=1.5)
    ax1.plot(df["t"], df["fv_A"], label="FV Model A", color="#f39c12", linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Seconds since round start")
    ax1.set_ylabel("Price of A", color="#8e44ad")
    ax1.tick_params(axis="y", labelcolor="#8e44ad")
    ax1.legend(loc="upper left")
    
    # Plot Rates on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(df["t"], df["rate_exp"], label="Rate Expectation (bps)", color="#34495e", linewidth=1.0, alpha=0.5)
    ax2.set_ylabel("Rate Exp. (bps)", color="#34495e")
    ax2.tick_params(axis="y", labelcolor="#34495e")
    ax2.legend(loc="upper right")
    
    plt.title("Asset A Tracking vs. Macro Rate Signals")
    ax1.grid(alpha=0.3)
    return fig_to_base64(fig)


def chart_positions(snapshots: pd.DataFrame) -> str:
    if snapshots.empty:
        return ""
    fig, ax = plt.subplots(figsize=(10, 4))
    rows = []
    for _, r in snapshots.iterrows():
        pos = r.get("positions") or {}
        for sym, qty in pos.items():
            rows.append({"t": r["t_rel"], "sym": sym, "qty": qty})
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    for sym, sub in df.groupby("sym"):
        ax.plot(sub["t"], sub["qty"], label=sym, linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.5)
    
    # Add gentle visual bounds for risk limits (assuming +/- 200)
    ax.axhline(200, color="red", linestyle=":", alpha=0.5)
    ax.axhline(-200, color="red", linestyle=":", alpha=0.5)
    
    ax.set_xlabel("Seconds since round start")
    ax.set_ylabel("Position (signed units)")
    ax.set_title("Positions over time (Dotted red = Typical Risk Limit)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)


def chart_mids(snapshots: pd.DataFrame, fills: pd.DataFrame) -> str:
    if snapshots.empty:
        return ""
    rows = []
    for _, r in snapshots.iterrows():
        mids = r.get("mids") or {}
        for sym, m in mids.items():
            rows.append({"t": r["t_rel"], "sym": sym, "mid": m})
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    main_syms = [s for s in ["A", "B", "C", "ETF"] if s in df["sym"].unique()]
    if not main_syms:
        return ""
    fig, ax = plt.subplots(figsize=(10, 4))
    for sym in main_syms:
        sub = df[df["sym"] == sym]
        ax.plot(sub["t"], sub["mid"], label=sym, linewidth=1.2)
    # Overlay fills as dots
    if not fills.empty:
        for sym in main_syms:
            f = fills[fills["symbol_guess"] == sym]
            if not f.empty:
                ax.scatter(f["t_rel"], f["price"], s=18, alpha=0.7, label=f"{sym} fills")
    ax.set_xlabel("Seconds since round start")
    ax.set_ylabel("Mid price")
    ax.set_title("Main Assets: Mid prices with fills overlaid")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    return fig_to_base64(fig)


def chart_rejects(rejects: pd.DataFrame) -> str:
    if rejects.empty:
        return ""
    fig, ax = plt.subplots(figsize=(10, 3))
    counts = rejects.groupby("reason").size().sort_values(ascending=True)
    ax.barh(counts.index, counts.values, color="#c0392b")
    ax.set_xlabel("Count")
    ax.set_title(f"Rejected orders by reason ({len(rejects)} total)")
    ax.grid(alpha=0.3, axis="x")
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Stats summary
# ---------------------------------------------------------------------------

def summarize(events: list[dict], fills: pd.DataFrame, rejects: pd.DataFrame,
              snapshots: pd.DataFrame) -> dict:
    duration = 0.0
    if events:
        duration = events[-1]["ts"] - events[0]["ts"]
        
    final_positions = {}
    final_pnl = 0.0
    if not snapshots.empty:
        final_positions = snapshots.iloc[-1].get("positions") or {}
        final_pnl = snapshots.iloc[-1].get("calc_mtm", 0.0)
        
    return {
        "events_total": len(events),
        "fills_total": len(fills),
        "rejects_total": len(rejects),
        "snapshots_total": len(snapshots),
        "duration_sec": round(duration, 1),
        "final_positions": final_positions,
        "final_pnl": round(final_pnl, 2)
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Case 1 Round Report — {filename}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1100px;
          margin: 30px auto; padding: 0 20px; color: #222; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
  h2 {{ margin-top: 36px; color: #2c3e50; font-size: 20px; border-left: 4px solid #3498db; padding-left: 10px; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
            background: #f4f6f8; padding: 16px; border-radius: 8px; margin-bottom: 20px; }}
  .stat {{ font-size: 14px; }}
  .stat .label {{ color: #666; text-transform: uppercase; font-size: 11px; font-weight: bold; }}
  .stat .value {{ font-size: 24px; font-weight: 600; color: #2c3e50; }}
  .pnl-positive {{ color: #27ae60 !important; }}
  .pnl-negative {{ color: #c0392b !important; }}
  .chart {{ margin: 18px 0; background: #fff; padding: 10px; border: 1px solid #eaeaea; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }}
  img {{ max-width: 100%; border-radius: 4px; display: block; }}
  pre {{ background: #f4f6f8; padding: 12px; border-radius: 4px; font-size: 13px; color: #333; }}
</style>
</head>
<body>
<h1>Case 1 Round Analysis</h1>
<p style="color: #666;"><strong>Log file:</strong> {filename} | <strong>Generated:</strong> {generated}</p>

<div class="stats">
  <div class="stat"><div class="label">Duration</div><div class="value">{duration_sec}s</div></div>
  <div class="stat"><div class="label">Est. MtM PnL</div><div class="value {pnl_class}">${final_pnl}</div></div>
  <div class="stat"><div class="label">Fills</div><div class="value">{fills_total}</div></div>
  <div class="stat"><div class="label">Rejects</div><div class="value">{rejects_total}</div></div>
</div>

<h2>1. Performance & Exposure</h2>
<div class="chart"><img src="data:image/png;base64,{chart_pnl}"></div>
<div class="chart"><img src="data:image/png;base64,{chart_exposure}"></div>

<h2>2. Strategy Signals</h2>
<div class="chart"><img src="data:image/png;base64,{chart_fv_and_rates}"></div>

<h2>3. Position Management</h2>
<div class="chart"><img src="data:image/png;base64,{chart_positions}"></div>
<pre><strong>Final Positions:</strong> {final_positions}</pre>

<h2>4. Market Context</h2>
<div class="chart"><img src="data:image/png;base64,{chart_mids}"></div>

<h2>5. Errors & Rejections</h2>
<div class="chart"><img src="data:image/png;base64,{chart_rejects}"></div>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_log()
    print(f"Reading {path}")
    
    events = load_log(path)
    if not events:
        sys.exit("Log file is empty.")

    # Apply cash/PnL reconstruction logic
    events = enrich_with_cash_and_pnl(events)

    df = pd.DataFrame(events)
    df["t_rel"] = to_relative_seconds(df["ts"])

    fills     = df[df["kind"] == "fill"].copy()
    rejects   = df[df["kind"] == "reject"].copy()
    snapshots = df[df["kind"] == "snapshot"].copy()

    # Best-effort symbol guessing for fills
    fills["symbol_guess"] = "?"
    prev_pos = {}
    guessed = []
    for _, r in fills.iterrows():
        pos = r.get("positions") or {}
        sym = "?"
        for k, v in pos.items():
            if prev_pos.get(k, 0) != v:
                sym = k
                break
        guessed.append(sym)
        prev_pos = pos
    fills["symbol_guess"] = guessed

    stats = summarize(events, fills, rejects, snapshots)

    pnl_class = "pnl-positive" if float(stats["final_pnl"]) >= 0 else "pnl-negative"

    html = HTML_TEMPLATE.format(
        filename=path.name,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        duration_sec=stats["duration_sec"],
        final_pnl=stats["final_pnl"],
        pnl_class=pnl_class,
        fills_total=stats["fills_total"],
        rejects_total=stats["rejects_total"],
        final_positions=json.dumps(stats["final_positions"], indent=2),
        chart_pnl=chart_pnl(snapshots),
        chart_exposure=chart_exposure(snapshots),
        chart_fv_and_rates=chart_fv_and_rates(snapshots),
        chart_positions=chart_positions(snapshots),
        chart_mids=chart_mids(snapshots, fills),
        chart_rejects=chart_rejects(rejects),
    )

    out = path.with_suffix(".html")
    out.write_text(html)
    print(f"\nReport written to: {out}")
    print(f"  Duration:  {stats['duration_sec']}s")
    print(f"  Est. PnL:  ${stats['final_pnl']}")
    print(f"  Fills:     {stats['fills_total']}")
    print(f"  Rejects:   {stats['rejects_total']}")
    print(f"\nOpen with: open {out}")


if __name__ == "__main__":
    main()
