# python
import os

import pandas as pd


def test_run_final_replay_outputs(tmp_path, monkeypatch):
    # Arrange: temp output and temp settings with custom SMC params
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "\n".join(
            [
                "smc:",
                "  sweep_lookback: 15",
                "  wick_ratio: 0.4",
                "  vol_burst_z: 1.2",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SMC_SETTINGS_PATH", str(settings_path))

    # Act
    from scripts.run_final_replay import run_final_replay

    csv_path = run_final_replay(output_dir=str(out_dir))

    # Assert: CSV and dashboard exist
    assert os.path.exists(csv_path), "CSV not created"
    dash_path = os.path.join(str(out_dir), "dashboard.html")
    assert os.path.exists(dash_path), "Dashboard not created"

    # Assert: CSV has expected columns, even if empty
    df = pd.read_csv(csv_path)
    expected_cols = {
        "order_id",
        "venue",
        "orig_ts",
        "submit_ts",
        "side",
        "price",
        "qty",
        "route_prob",
        "filled",
        "fill_price",
        "filled_qty",
        "queue_ahead",
        "agg_consumed",
    }
    assert expected_cols.issubset(
        set(df.columns)
    ), f"Missing columns: {expected_cols - set(df.columns)}"
