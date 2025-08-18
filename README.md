# bingx_bot_minimal

Minimal, batteries-included scaffold for a crypto-algo bot focused on:
- (a) Liquidity sweep detection (SMC/ICT-flavored)
- (b) Kelly-capped position sizing
- (c) Funding accrual backtest primitive

## Layout
```
bingx_bot_minimal/
  config/
    settings.yaml
  engine/
    __init__.py
    core.py
    risk.py
    execution.py
    data.py
  research/
    __init__.py
    detectors.py
    sizing.py
    funding_backtest.py
  scripts/
    run_backtest.py
    simulate_sweep.py
  tests/
    test_funding.py
  requirements.txt
  README.md
```
## Quick start

```bash
pip install -r requirements.txt
python scripts/simulate_sweep.py
python scripts/run_backtest.py
pytest -q
```

All modules use pure `pandas`/`numpy` to keep the example lightweight. Adapt to `polars` or your in-house stack as needed.


## SMC Sweep Strategy (HTF + FVG)

Run a simple end-to-end backtest with HTF bias (Donchian + AVWAP slope agreement) and FVG-based entries:

```bash
python scripts/run_smc_backtest.py
```

Outputs basic stats and a trades table with entry/exit/rr.


## L2-aware Replay & Execution Simulator

New modules:
- replay/l2_replayer.py: maintains an L2 book from diffs and applies trades.
- replay/execution_l2.py: ExecutionSimulatorL2 that computes queue-ahead per level, applies trades, and scores fills.

Quick demo: run `python -c "from replay.l2_replayer import L2Replay; print('ok')"` or use `scripts/run_replay.py` as a template to switch to L2Replay + ExecutionSimulatorL2.


## L2 Week Replay + Fill Report

Run a one-week synthetic demo that creates L2 diffs + trade prints and outputs a per-order fill-quality CSV:

```
python scripts/run_week_replay.py
```

Or convert your recorder output (jsonlines) into parquet inputs:

```
python -c "from scripts.convert_recorder import convert_recorder_to_l2_and_trades; convert_recorder_to_l2_and_trades('my_recorder.jl','tmp_out')"
```


## Multi-Venue Router with Latency Simulation

Run a multi-venue simulation with latency-adjusted fill probability routing:

```
python scripts/run_multi_venue_router.py
```
Outputs:
- fill_report_multi.csv: All fills from all venues
- router_decisions.csv: Router decisions per order
```


## Final Multi-Venue L2 Replay + Router

Run the flagship demo that synthesizes 3 venues, routes sweep-detected entries, simulates latency, and exports `final_fill_quality_report.csv`:

```
python scripts/run_final_replay.py
```

This is the highest-fidelity local simulation in the scaffold: multi-venue, L2-aware, latency model, router + per-order scoring.

<!-- Merged unique lines from README copy.md -->

# Bingx Bot Minimal - Final Extended
## New Modules
- `scripts/convert_recorder_multi.py`: Convert multiple-venue JSONL recorders to L2 + trades parquet format.
- `scripts/dashboard.py`: Generate an interactive HTML dashboard from `final_fill_quality_report.csv`.
## Usage
1. Convert your multi-venue recorder files:
python scripts/convert_recorder_multi.py
(Edit the recorder_paths dict in the script.)
2. Run your replay to generate `final_fill_quality_report.csv`.
3. Build dashboard:
python scripts/dashboard.py
View `dashboard.html` in your browser.

## Supervised Live run (SAFE mode)

This repository includes helper scripts to run the code in a supervised,
conservative "live" mode. These are intentionally minimal and designed to be
used by an operator who confirms keys and watches the process.

1. Prepare conservative live configs:

  - `config/settings.live.yaml` — conservative defaults (no API keys inside).
  - `config/trading_config.live.yaml` — conservative trading params.

2. Provide exchange API keys via environment variables (recommended):

  - KuCoin: `KUCOIN_API_KEY`, `KUCOIN_API_SECRET`, `KUCOIN_API_PASSPHRASE`
  - Binance: `BINANCE_API_KEY`, `BINANCE_API_SECRET`

3. Start the telemetry service (optional but recommended):

```bash
python3 scripts/telemetry_service.py &
```

4. Start supervised live runner (interactive):

```bash
python3 scripts/start_live.py
# confirm by typing YES when prompted
```

What the supervisor does:
- Runs preflight checks against `config/*.live.yaml`.
- Starts the runner and writes `.live_bot.pid` for the monitor to target.
- Starts `kill_switch` in a background thread; kills the runner on drawdown or stale heartbeat and sends alerts.

Notification environment variables (optional):

- `SLACK_WEBHOOK_URL` — Slack incoming webhook used for alerts.
- `ALERT_SMTP_HOST`, `ALERT_SMTP_PORT`, `ALERT_SMTP_USER`, `ALERT_SMTP_PASS`, `ALERT_TO` — SMTP settings for email alerts.

Use this flow for supervised initial live testing. For production, use a proper service supervisor and secure secrets storage.

CI Status: ![CI](https://github.com/keplerinfotech/bingx_bot_minimal_final_ext_autodash/actions/workflows/ci.yml/badge.svg)
