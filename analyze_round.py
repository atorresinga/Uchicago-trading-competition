"""
analyze_round.py — Post-round analysis for Case 1 bot.

Reads the most recent logs/round_*.jsonl file (or one you pass on the
command line) and produces a single HTML report with charts and stats.

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
# Load
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
    ax.set_xlabel("Seconds since round start")
    ax.set_ylabel("Position (signed units)")
    ax.set_title("Positions over time")
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
    ax.set_title("Mid prices with fills overlaid")
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


def chart_fill_rate(snapshots: pd.DataFrame, fills: pd.DataFrame) -> str:
    if snapshots.empty:
        return ""
    fig, ax = plt.subplots(figsize=(10, 3))
    if not fills.empty:
        fills_per_sec = fills.groupby(fills["t_rel"].astype(int)).size()
        ax.bar(fills_per_sec.index, fills_per_sec.values, color="#2980b9", width=0.9)
    ax.set_xlabel("Seconds since round start")
    ax.set_ylabel("Fills / second")
    ax.set_title("Fill activity over time")
    ax.grid(alpha=0.3)
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
    if not snapshots.empty:
        final_positions = snapshots.iloc[-1].get("positions") or {}
    return {
        "events_total": len(events),
        "fills_total": len(fills),
        "rejects_total": len(rejects),
        "snapshots_total": len(snapshots),
        "duration_sec": round(duration, 1),
        "final_positions": final_positions,
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
  h2 {{ margin-top: 36px; color: #2c3e50; }}
  .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
            background: #f4f6f8; padding: 16px; border-radius: 8px; }}
  .stat {{ font-size: 14px; }}
  .stat .label {{ color: #666; text-transform: uppercase; font-size: 11px; }}
  .stat .value {{ font-size: 22px; font-weight: 600; color: #2c3e50; }}
  .chart {{ margin: 18px 0; }}
  img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  pre {{ background: #f4f6f8; padding: 12px; border-radius: 4px; font-size: 12px; }}
</style>
</head>
<body>
<h1>Case 1 Round Report</h1>
<p><strong>Log file:</strong> {filename}<br>
<strong>Generated:</strong> {generated}</p>

<div class="stats">
  <div class="stat"><div class="label">Duration</div><div class="value">{duration_sec}s</div></div>
  <div class="stat"><div class="label">Fills</div><div class="value">{fills_total}</div></div>
  <div class="stat"><div class="label">Rejects</div><div class="value">{rejects_total}</div></div>
</div>

<h2>Final positions</h2>
<pre>{final_positions}</pre>

<h2>Positions over time</h2>
<div class="chart"><img src="data:image/png;base64,{chart_positions}"></div>

<h2>Mid prices with fills</h2>
<div class="chart"><img src="data:image/png;base64,{chart_mids}"></div>

<h2>Fill activity</h2>
<div class="chart"><img src="data:image/png;base64,{chart_fill_rate}"></div>

<h2>Rejected orders</h2>
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

    df = pd.DataFrame(events)
    df["t_rel"] = to_relative_seconds(df["ts"])

    fills     = df[df["kind"] == "fill"].copy()
    rejects   = df[df["kind"] == "reject"].copy()
    snapshots = df[df["kind"] == "snapshot"].copy()

    # Best-effort: guess which symbol a fill belongs to from the position delta.
    # (If you ever start logging the symbol explicitly in fills, drop this.)
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

    html = HTML_TEMPLATE.format(
        filename=path.name,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        duration_sec=stats["duration_sec"],
        fills_total=stats["fills_total"],
        rejects_total=stats["rejects_total"],
        final_positions=json.dumps(stats["final_positions"], indent=2),
        chart_positions=chart_positions(snapshots),
        chart_mids=chart_mids(snapshots, fills),
        chart_fill_rate=chart_fill_rate(snapshots, fills),
        chart_rejects=chart_rejects(rejects),
    )

    out = path.with_suffix(".html")
    out.write_text(html)
    print(f"\nReport written to: {out}")
    print(f"  Duration:  {stats['duration_sec']}s")
    print(f"  Fills:     {stats['fills_total']}")
    print(f"  Rejects:   {stats['rejects_total']}")
    print(f"  Snapshots: {stats['snapshots_total']}")
    print(f"\nOpen with: open {out}")


if __name__ == "__main__":
    main()