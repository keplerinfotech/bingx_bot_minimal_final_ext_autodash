import pandas as pd, json, os

def convert_recorder_multi(recorder_paths: dict, out_dir: str):
    """
    Convert multiple venue recorders (JSONL) to parquet L2 + trades.
    recorder_paths: {venue_name: path_to_jsonl}
    Expects two event types per line:
      - {"type":"l2_update", "timestamp":"ISO", "side":"bid/ask", "price":float, "size":float, "update_type":"update|snapshot|delete"}
      - {"type":"trade", "timestamp":"ISO", "price":float, "size":float, "side":"buy/sell"}
    Outputs parquet files under out_dir/{venue}/l2_diffs.parquet and trades.parquet
    """
    for venue, path in recorder_paths.items():
        l2_rows, trade_rows = [], []
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                ob = json.loads(line)
                t = pd.to_datetime(ob.get("timestamp"))
                if ob.get("type") == "l2_update":
                    l2_rows.append({"timestamp": t, "side": ob.get("side"), "price": float(ob.get("price")), "size": float(ob.get("size",0.0)), "update_type": ob.get("update_type","update")})
                elif ob.get("type") == "trade":
                    trade_rows.append({"timestamp": t, "price": float(ob.get("price")), "size": float(ob.get("size",0.0)), "side": ob.get("side")})
        l2_df = pd.DataFrame(l2_rows).set_index("timestamp").sort_index()
        trades_df = pd.DataFrame(trade_rows).set_index("timestamp").sort_index()
        vdir = os.path.join(out_dir, venue)
        os.makedirs(vdir, exist_ok=True)
        l2_df.to_parquet(os.path.join(vdir, "l2_diffs.parquet"))
        trades_df.to_parquet(os.path.join(vdir, "trades.parquet"))
        print(f"Converted {venue}: {len(l2_df)} L2 updates, {len(trades_df)} trades")

if __name__ == "__main__":
    # Example usage
    recorder_paths = {
        "venueA": "data/venueA_recorder.jsonl",
        "venueB": "data/venueB_recorder.jsonl"
    }
    convert_recorder_multi(recorder_paths, "parquet_out")
