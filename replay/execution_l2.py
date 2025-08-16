import pandas as pd, numpy as np
from .l2_replayer import L2Replay

class ExecutionSimulatorL2:
    """
    Execution simulator that is L2-aware. It uses L2Replay to maintain book state and resolve fills
    while providing queue-ahead and fill-quality scoring.
    """
    def __init__(self, l2_replay: L2Replay):
        self.replay = l2_replay
        self.fill_records = []

    def place_limit(self, order: dict):
        """
        order keys: timestamp, side ('long' or 'short'), price, qty, id
        Returns fill dict with metadata including queue_ahead, agg_consumed, fill_price, filled_qty
        """
        ts = order.get("timestamp")
        side = order.get("side")
        price = float(order.get("price"))
        qty = float(order.get("qty", 0.0))

        # ensure book is updated to this timestamp
        tob = self.replay.top_of_book_at(ts)
        # map side to book side for queue calc: posting 'long' means bid side
        book_side = "bid" if side == "long" else "ask"
        queue_ahead = self.replay.book.queue_ahead(book_side, price)

        # Now iterate trades from ts forward and apply them to the book until fill condition satisfied
        cum_agg = 0.0  # cumulative opposing aggressor volume at or through price
        filled = False
        fill_px = None
        filled_qty = 0.0
        for _, tr in self.replay.trades_since(ts).iterrows():
            trade_side = tr.get("side")  # 'buy' if aggressor bought at ask (consumes asks)
            trade_price = float(tr.get("price"))
            trade_size = float(tr.get("size", 0.0))
            # Apply trade to book first (to reflect book changes)
            self.replay.book.apply_trade(trade_side, trade_price, trade_size)
            # if trade is opposing aggressor relative to our resting order, and price is favorable, count it
            if side == "long" and trade_side == "sell" and trade_price <= price + 1e-12:
                cum_agg += trade_size
            if side == "short" and trade_side == "buy" and trade_price >= price - 1e-12:
                cum_agg += trade_size
            # if cumulative aggressor volume exceeds queue_ahead + our_qty, fill occurs
            if cum_agg >= queue_ahead + qty - 1e-12:
                filled = True
                fill_px = price
                filled_qty = qty
                break

        # record
        rec = {"order": order, "filled": filled, "fill_price": fill_px, "filled_qty": filled_qty,
               "queue_ahead": queue_ahead, "agg_consumed": cum_agg}
        self.fill_records.append(rec)
        return rec

    def place_market(self, order: dict):
        # Market: consume opposing aggressor trades until qty met; compute VWAP fill price
        ts = order.get("timestamp")
        side = order.get("side")
        qty = float(order.get("qty", 0.0))
        cum = 0.0
        vwap = 0.0
        for _, tr in self.replay.trades_since(ts).iterrows():
            trade_side = tr.get("side")
            trade_price = float(tr.get("price"))
            trade_size = float(tr.get("size", 0.0))
            if side == "long" and trade_side == "buy":
                take = min(trade_size, qty - cum)
                vwap = (vwap * cum + trade_price * take) / (cum + take) if cum + take > 0 else trade_price
                cum += take
            if side == "short" and trade_side == "sell":
                take = min(trade_size, qty - cum)
                vwap = (vwap * cum + trade_price * take) / (cum + take) if cum + take > 0 else trade_price
                cum += take
            self.replay.book.apply_trade(trade_side, trade_price, trade_size)
            if cum >= qty - 1e-12:
                break
        filled = cum >= qty - 1e-12
        rec = {"order": order, "filled": filled, "fill_price": vwap if filled else None, "filled_qty": cum}
        self.fill_records.append(rec)
        return rec

    def export_fill_report(self) -> pd.DataFrame:
        rows = []
        for r in self.fill_records:
            o = r.get("order", {})
            rows.append({
                "order_id": o.get("id"),
                "ts": o.get("timestamp"),
                "side": o.get("side"),
                "price": o.get("price"),
                "qty": o.get("qty"),
                "filled": r.get("filled"),
                "fill_price": r.get("fill_price"),
                "filled_qty": r.get("filled_qty"),
                "queue_ahead": r.get("queue_ahead"),
                "agg_consumed": r.get("agg_consumed")
            })
        return pd.DataFrame(rows)
