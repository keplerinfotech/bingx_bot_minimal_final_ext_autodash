import pandas as pd


class MarketReplay:
    """
    Very lightweight market replay engine that accepts:
      - trades_df: historical aggressive trades with columns ['timestamp','price','size','side'] where
        side is 'buy' if aggressor bought at ask, 'sell' if aggressor sold at bid.
      - depth_df: (optional) top-of-book snapshots with columns ['timestamp','best_bid','bid_size','best_ask','ask_size'].
    The replayer yields ticks (trade events) and can be queried for prevailing top-of-book.
    """

    def __init__(self, trades_df: pd.DataFrame, depth_df: pd.DataFrame | None = None):
        self.trades = trades_df.sort_index()
        self.depth = depth_df.sort_index() if depth_df is not None else None

    def get_trades_between(self, start_ts, end_ts):
        return self.trades.loc[start_ts:end_ts]

    def top_of_book_at(self, ts):
        if self.depth is not None:
            # get nearest snapshot at or before ts
            try:
                return self.depth.loc[:ts].iloc[-1].to_dict()
            except Exception:
                return None
        return None


class ExecutionSimulator:
    """
    Simulates limit and market orders' fills using trade flow and optional depth snapshots.
    Simplified model:
      - For limit orders resting at best bid/ask: we compute queue_ahead from depth snapshot (bid_size/ask_size).
      - A fill occurs when cumulative opposite-side trade aggressor volume at or through that price >= queue_ahead + our_qty.
      - Market orders execute immediately against next trades until qty consumed.
    Returns fill dict with 'filled' flag, 'fill_price', 'filled_qty', and metadata including estimated queue_position.
    """

    def __init__(self, replay: MarketReplay):
        self.replay = replay
        self.resting_orders = []  # track our resting limit orders as dicts
        self.fills = []

    def _estimate_queue_ahead(self, ts, side, price):
        tob = self.replay.top_of_book_at(ts)
        if tob is None:
            # fallback heuristic: assume small queue ahead
            return 0.0
        if side == "buy":
            # placing bid, queue ahead is bid_size at that price
            return float(tob.get("bid_size", 0.0))
        else:
            return float(tob.get("ask_size", 0.0))

    def place_order(self, order: dict):
        otype = order.get("type", "limit")
        ts = order.get("timestamp")
        side = order.get("side")
        qty = float(order.get("qty", 0.0))
        price = (
            float(order.get("price", 0.0)) if order.get("price") is not None else None
        )

        if otype == "market":
            # consume trades from ts forward until qty filled
            cum = 0.0
            fill_px = 0.0
            for _, tr in self.replay.trades.loc[ts:].iterrows():
                trade_side = tr.get("side")
                trade_size = float(tr.get("size", 0.0))
                trade_price = float(tr.get("price", 0.0))
                # market buy consumes asks (aggressor=buy)
                if side == "long" and trade_side == "buy":
                    take = min(trade_size, qty - cum)
                    cum += take
                    fill_px = (
                        trade_price
                        if fill_px == 0
                        else (fill_px * (cum - take) + trade_price * take) / cum
                    )
                if side == "short" and trade_side == "sell":
                    take = min(trade_size, qty - cum)
                    cum += take
                    fill_px = (
                        trade_price
                        if fill_px == 0
                        else (fill_px * (cum - take) + trade_price * take) / cum
                    )
                if cum >= qty - 1e-12:
                    break
            filled = cum >= qty - 1e-12
            return {
                "filled": filled,
                "fill_price": fill_px if filled else None,
                "filled_qty": cum,
                "order": order,
                "reason": "market",
            }

        # limit order: check if at or inside TOB; estimate queue and check subsequent aggressor trades
        queue_ahead = self._estimate_queue_ahead(
            ts, side="buy" if side == "long" else "sell", price=price
        )
        cum_agg = 0.0
        cum = 0.0
        fill_px = None
        # iterate trades and sum opposing aggressor volume at or through our limit price
        for _, tr in self.replay.trades.loc[ts:].iterrows():
            trade_side = tr.get("side")
            trade_price = float(tr.get("price", 0.0))
            trade_size = float(tr.get("size", 0.0))
            if side == "long":
                # need sells (aggressor sold at bid) at price <= our price
                if trade_side == "sell" and trade_price <= price + 1e-12:
                    cum_agg += trade_size
            else:
                if trade_side == "buy" and trade_price >= price - 1e-12:
                    cum_agg += trade_size
            # if cumulative aggressor volume exceeds queue ahead, we assume fills happen up to our qty
            if cum_agg >= queue_ahead + qty - 1e-12:
                fill_px = price
                cum = qty
                break
        filled = cum >= qty - 1e-12
        return {
            "filled": filled,
            "fill_price": fill_px,
            "filled_qty": cum,
            "order": order,
            "queue_ahead": queue_ahead,
            "agg_consumed": cum_agg,
        }
