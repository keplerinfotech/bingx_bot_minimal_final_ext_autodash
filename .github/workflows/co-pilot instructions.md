# AI Assistant Guidelines

Project: **bingx_bot_minimal_final_ext_autodash**

---

## ⚡ Purpose
This project is an **advanced modular crypto trading bot** built with Python 3.10+, combining:
- **Smart Money Concepts (SMC)** and **ICT principles**  
- **ML-based trade filtering** (Decision Tree / Logistic Regression)  
- **Automated risk management** (Kelly Criterion, max drawdown)  
- **Replay/backtesting capabilities**  
- **Script-based execution & dashboard integration readiness**  

> The AI assistant is expected to **understand the architecture, maintain modularity, and optimize Python code**, while ensuring all edits are safe, testable, and PEP8-compliant.

---

## 🔧 General Rules
- Execute all code **inside PyCharm Terminal or Python Console**. No manual copy-paste.  
- Always return **full updated files** after edits.  
- Auto-handle missing dependencies:
```bash
pip install <package>
# then update requirements.txt
Format all code PEP8 + inline comments + docstrings.
Assume “yes” for safe steps (creating venv, overwriting files, updating scripts).
Always log operations, especially changes to strategy, execution, or risk modules.
⚙️ Environment Setup
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
▶️ Bot Execution
# Normal execution
python -m core.controller.bot_controller

# Debug mode
python -m core.controller.bot_controller --debug

✅ Testing
pytest -v
Run tests after every change in core, strategies, replay, or research modules.
Unit test all new functions.
Validate ML filters with sample data before deploying.

📚 Coding Guidelines
Folder responsibilities:
core/ → engine, controller, execution, data handling, risk management
strategies/ → trading strategies (SMC/ICT)
replay/ → replay & backtesting modules
research/ → experiments, indicators, funding, sizing, HTF
scripts/ → runnable scripts (replay, simulation, dashboard)
tests/ → unit & integration tests
Maintain single responsibility per module.
Functions ≤ 50 lines preferred.
Avoid hardcoding secrets; use CONFIG_TEMPLATE.json or settings.yaml.
Include structured logging in every module.

🤖 Autopilot Directives
AI should assume full project access. No manual instructions needed.
Debug step by step, output logs for each module.
Suggest improvements in:
Strategy modularity
ML filter optimization
Risk management logic
Execution efficiency
Always verify changes by running scripts and tests before suggesting next steps.
Keep all scripts idempotent — changes should not break replay/backtest functionality.
Ensure backward compatibility with existing strategy logic.

🔄 Workflow Summary (for AI understanding)
Exchange API → core/data.py → strategies/smc_sweep.py → core/risk.py → core/execution.py → core/controller.py → scripts/*.py → tests/*.py
ML filter runs between Strategy and Risk.
Replay/backtesting scripts operate independently but must conform to core/ APIs.
All logging should include timestamps and module/function names.

🛡️ Advanced Guidelines
Optimize for modular, AI-friendly Python coding.
Auto-generate docstrings for new functions/classes.
Suggest improvements to risk/reward, ML thresholds, and position sizing.
Keep debug mode verbose for any AI action affecting execution.
Treat this repo as an AI-aware Python project: any edits must maintain clarity, performance, and safety.
This AI guideline file is the single source of truth for coding, debugging, testing, and enhancing the bot.
Always reference it before making any changes.

#Final Project Structure
bingx_bot_minimal_final_ext_autodash/
│
├── config/                     # Configuration files
│   └── settings.yaml           # User-defined settings
│
├── core/                       # Main engine & bot logic
│   ├── __init__.py
│   ├── controller.py           # Main orchestrator (entry point)
│   ├── core.py                 # Core engine logic
│   ├── data.py                 # Market data handling
│   ├── execution.py            # Order execution layer
│   └── risk.py                 # Risk manager & position sizing
│
├── strategies/                 # Trading strategy modules
│   └── smc_sweep.py            # SMC/ICT sweep strategy
│
├── replay/                     # Replay & backtesting modules
│   ├── __init__.py
│   ├── l2_replayer.py
│   ├── replayer.py
│   └── execution_l2.py
│
├── research/                   # Experiments, indicators, and sizing
│   ├── __init__.py
│   ├── detectors.py
│   ├── funding_backtest.py
│   ├── fvg.py
│   ├── htf.py
│   └── sizing.py
│
├── scripts/                    # Runnable scripts & simulations
│   ├── convert_recorder.py
│   ├── convert_recorder_multi.py
│   ├── dashboard.py
│   ├── run_backtest.py
│   ├── run_final_replay.py
│   ├── run_multi_venue_router.py
│   ├── run_replay.py
│   ├── run_smc_backtest.py
│   ├── run_week_replay.py
│   └── simulate_sweep.py
│
├── tests/                      # Unit & integration tests
│   └── test_funding.py
│
├── venv/                       # Python virtual environment (ignored in git)
│
├── README.md                    # Project overview & setup
├── AI_GUIDELINES.md             # Advanced AI assistant instructions
├── ARCHITECTURE.md              # System architecture & module flow
├── CONTRIBUTING.md              # Contribution guidelines
├── CONFIG_TEMPLATE.json          # Template for API & trade configs
├── requirements.txt             # Python dependencies
└── .gitignore                   # Ignore venv, logs, caches, etc.

# 🔑 Key Notes for AI & Human Contributors
AI Visibility:
AI can see all code in core/, strategies/, replay/, research/, scripts/.
tests/ ensures any changes can be validated automatically.
Modularity:
Each module has a single responsibility.
core/ handles engine + execution + risk; strategies only focus on signals.

Scripts:
All scripts are runnable and safe for replay/backtest.
AI can add new scripts, tests, or dashboards without touching core logic.

Documentation:

README.md → human + AI overview

ARCHITECTURE.md → workflow + module diagram

AI_GUIDELINES.md → AI operational instructions

Config & Secrets:
CONFIG_TEMPLATE.json or settings.yaml is single source of truth for API keys, risk settings, ML thresholds.

Testing:
tests/ ensures AI edits or human improvements don’t break existing logic.
```python
"""
Final demo script: multi-venue L2 replay + router + latency + scoring.
  - Synthesizes 3 venue datasets with different liquidity profiles and latency.
  - Uses a simple emitter that detects sweeps and posts limit orders routed to best venue.
  - Collects fill records across venues and exports a final CSV and summary.
"""
from __future__ import annotations

import os
import math
import json
import pandas as pd
import numpy as np

from typing import Tuple, List, Dict, Any

# Project imports
from multi_venue import Venue, Router, LatencyModel
from replay.l2_replayer import L2Replay  # noqa: F401 (kept for side-effects / completeness)
from replay.execution_l2 import ExecutionSimulatorL2  # noqa: F401 (kept for side-effects / completeness)
from research.detectors import find_sweeps


def synthesize_venue(
    name: str,
    n: int = 2000,
    seed: int = 1,
    liquidity_scale: float = 1.0,
    latency_base: float = 20.0,
) -> Tuple[Venue, LatencyModel]:
    """
    Synthesize simple L2 snapshots and trade prints for a venue.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="T")
    price = 100 + np.cumsum(rng.normal(0, 0.05, n))

    l2_rows: List[Dict[str, Any]] = []
    trades_rows: List[Dict[str, Any]] = []
    for i, t in enumerate(idx):
        center = float(price[i])
        # Five levels on each side
        for lvl in range(1, 6):
            bp = round(center - 0.01 * lvl, 2)
            ap = round(center + 0.01 * lvl, 2)
            bs = float(max(0.1, rng.exponential(10.0) * liquidity_scale))
            asz = float(max(0.1, rng.exponential(8.0) * liquidity_scale))
            l2_rows.append({"timestamp": t, "side": "bid", "price": bp, "size": bs, "update_type": "snapshot"})
            l2_rows.append({"timestamp": t, "side": "ask", "price": ap, "size": asz, "update_type": "snapshot"})

        # Aggressive trades
        ntr = int(rng.integers(0, 4)) if liquidity_scale < 2.0 else int(rng.integers(1, 6))
        for _ in range(ntr):
            side = "buy" if rng.random() < 0.5 else "sell"
            px = round(center + rng.normal(0, 0.02), 2)
            sz = float(max(0.01, rng.exponential(2.0) * liquidity_scale))
            trades_rows.append({"timestamp": t, "price": px, "size": sz, "side": side})

    l2_df = pd.DataFrame(l2_rows).set_index("timestamp").sort_index()
    trades_df = pd.DataFrame(trades_rows).set_index("timestamp").sort_index()

    return Venue(name=name, l2_diffs=l2_df, trades=trades_df), LatencyModel(base_ms=latency_base, jitter_ms=5.0)


def run_final_replay(output_csv: str | None = None, output_dir: str = ".") -> str:
    """
    Runs the final multi-venue L2 replay + routing simulation and writes a CSV report.

    Args:
        output_csv: Optional explicit path to CSV file. If None, uses output_dir/final_fill_quality_report.csv
        output_dir: Directory to place outputs (CSV, dashboard). Created if not exists.

    Returns:
        The path to the generated CSV.
    """
    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    out_csv = output_csv or os.path.join(output_dir, "final_fill_quality_report.csv")

    # Create 3 venues
    v1, lat1 = synthesize_venue("alpha", n=2000, seed=11, liquidity_scale=1.0, latency_base=20.0)
    v2, lat2 = synthesize_venue("beta", n=2000, seed=22, liquidity_scale=2.5, latency_base=50.0)
    v3, lat3 = synthesize_venue("gamma", n=2000, seed=33, liquidity_scale=0.6, latency_base=10.0)
    venues = [v1, v2, v3]
    latency_models = {"alpha": lat1, "beta": lat2, "gamma": lat3}
    router = Router(venues, latency_models=latency_models)

    # Build LTF bar df used by sweep detector (use venue alpha price for bars)
    alpha_px = v1.trades["price"].resample("1T").last().ffill().bfill()

    df = pd.DataFrame(index=alpha_px.index)
    df["close"] = alpha_px
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) + 0.05
    df["low"] = df[["open", "close"]].min(axis=1) - 0.05
    df["volume"] = 1.0

    # Detect sweep events
    events = find_sweeps(df, lookback=20, wick_ratio=0.5, vol_burst_z=1.5)
    print(f"Detected sweep events: {len(events)}")

    # Build orders from events
    orders: List[Dict[str, Any]] = []
    for ts, ev in events.iterrows():
        direction = ev.get("direction", "long")
        side = "long" if direction == "long" else "short"
        price = float(ev.get("entry_price", df.loc[ts, "close"]))
        qty = float(ev.get("qty", 1.0))
        # Make a stable ID without spaces
        oid = f"e-{int(pd.Timestamp(ts).value // 1_000_000)}"
        orders.append({"id": oid, "timestamp": pd.Timestamp(ts), "side": side, "price": price, "qty": qty})

    # Route & place orders
    fill_records: List[Dict[str, Any]] = []
    for o in orders:
        ts = pd.Timestamp(o["timestamp"])
        side = o["side"]
        price = float(o["price"])
        qty = float(o["qty"])

        venue_name, prob = router.choose(ts, side, price, qty, horizon_ms=60_000)
        venue = next(v for v in venues if v.name == venue_name)

        # Simulate latency on submission
        lat_ms = float(latency_models[venue_name].sample_ms())
        ts_submit = ts + pd.Timedelta(milliseconds=lat_ms)

        order = {"id": o["id"], "timestamp": ts_submit, "side": side, "price": price, "qty": qty}
        res = venue.exec_sim.place_limit(order)

        rec = {
            "order_id": o["id"],
            "venue": venue_name,
            "orig_ts": ts,
            "submit_ts": ts_submit,
            "side": side,
            "price": price,
            "qty": qty,
            "route_prob": prob,
            "filled": res.get("filled"),
            "fill_price": res.get("fill_price"),
            "filled_qty": res.get("filled_qty"),
            "queue_ahead": res.get("queue_ahead"),
            "agg_consumed": res.get("agg_consumed"),
        }
        fill_records.append(rec)

    rep_df = pd.DataFrame(fill_records)
    rep_df.to_csv(out_csv, index=False)
    print(f"Wrote final report: {out_csv}")

    # Summary
    try:
        summary = rep_df.groupby("venue")["filled"].agg(["sum", "count", "mean"])