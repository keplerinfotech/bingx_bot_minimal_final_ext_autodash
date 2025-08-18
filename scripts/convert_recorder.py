import json
import os

import pandas as pd


def convert_recorder_to_l2_and_trades(recorder_path: str, out_dir: str):
    """
    Example converter: expects a simple JSON lines recorder with two event types:
      - {"type":"l2_update","timestamp":"ISO","side":"bid/ask","price":float,"size":float,"update_type":"update|snapshot|delete"}
      - {"type":"trade","timestamp":"ISO","price":float,"size":float,"side":"buy/sell"}
    This function will produce two parquet files:
      - l2_diffs.parquet (indexed by timestamp)
      - trades.parquet (indexed by timestamp)
    """
    os.makedirs(out_dir, exist_ok=True)
    l2_rows = []
    trade_rows = []
    with open(recorder_path, "r") as f:
        for line in f:
            ob = json.loads(line)
            t = pd.to_datetime(ob.get("timestamp"))
            if ob.get("type") == "l2_update":
                l2_rows.append(
                    {
                        "timestamp": t,
                        "side": ob.get("side"),
                        "price": float(ob.get("price")),
                        "size": float(ob.get("size", 0.0)),
                        "update_type": ob.get("update_type", "update"),
                    }
                )
            elif ob.get("type") == "trade":
                trade_rows.append(
                    {
                        "timestamp": t,
                        "price": float(ob.get("price")),
                        "size": float(ob.get("size", 0.0)),
                        "side": ob.get("side"),
                    }
                )
    l2_df = pd.DataFrame(l2_rows).set_index("timestamp").sort_index()
    trades_df = pd.DataFrame(trade_rows).set_index("timestamp").sort_index()
    l2_df.to_parquet(os.path.join(out_dir, "l2_diffs.parquet"))
    trades_df.to_parquet(os.path.join(out_dir, "trades.parquet"))
    return os.path.join(out_dir, "l2_diffs.parquet"), os.path.join(
        out_dir, "trades.parquet"
    )
