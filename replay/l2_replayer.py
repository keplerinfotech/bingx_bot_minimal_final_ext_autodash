import pandas as pd, numpy as np
from collections import defaultdict, deque
import bisect

class L2OrderBook:
    """
    Simple in-memory L2 orderbook representation using price->size maps for bids and asks.
    Maintains sorted price lists for quick top-of-book and queue-ahead calculations.
    """
    def __init__(self):
        # price levels stored as sorted lists (desc bids, asc asks)
        self.bids = {}  # price -> size
        self.asks = {}  # price -> size
        self.bid_prices = []  # descending
        self.ask_prices = []  # ascending

    def set_level(self, side: str, price: float, size: float):
        d = self.bids if side == "bid" else self.asks
        plist = self.bid_prices if side == "bid" else self.ask_prices
        if size <= 0:
            # remove level if present
            if price in d:
                del d[price]
                # remove price from list
                try:
                    plist.remove(price)
                except ValueError:
                    pass
        else:
            if price not in d:
                # insert into sorted list
                if side == "bid":
                    # maintain descending
                    bisect.insort_left(plist, -price)
                else:
                    bisect.insort_left(plist, price)
            d[price] = float(size)
            # normalize bid_prices to actual prices for convenience
            if side == "bid":
                plist[:] = [-p for p in plist]
                plist.sort(reverse=True)

    def top_of_book(self):
        best_bid = max(self.bids.keys()) if self.bids else None
        best_ask = min(self.asks.keys()) if self.asks else None
        bid_size = self.bids.get(best_bid, 0.0) if best_bid is not None else 0.0
        ask_size = self.asks.get(best_ask, 0.0) if best_ask is not None else 0.0
        return {"best_bid": best_bid, "bid_size": bid_size, "best_ask": best_ask, "ask_size": ask_size}

    def queue_ahead(self, side: str, price: float):
        """
        For a resting limit order at 'price' on side 'buy' (meaning we are posting a bid),
        queue ahead is the sum of existing sizes at that price for same side.
        We assume price levels equal to exact price ticks.
        """
        if side == "buy":
            return float(self.bids.get(price, 0.0))
        else:
            return float(self.asks.get(price, 0.0))

    def apply_trade(self, trade_side: str, price: float, size: float):
        """
        Apply an aggressive trade to the book: aggressive buy consumes asks at price or better;
        aggressive sell consumes bids. We decrement sizes at or through the price level(s).
        """
        remaining = size
        if trade_side == "buy":
            # consume asks from lowest ask upwards where price <= trade_price
            for p in sorted([p for p in self.asks.keys() if p <= price]):
                avail = self.asks.get(p, 0.0)
                take = min(avail, remaining)
                avail -= take
                remaining -= take
                if avail <= 1e-12:
                    del self.asks[p]
                else:
                    self.asks[p] = avail
                if remaining <= 1e-12:
                    break
        else:
            for p in sorted([p for p in self.bids.keys() if p >= price], reverse=True):
                avail = self.bids.get(p, 0.0)
                take = min(avail, remaining)
                avail -= take
                remaining -= take
                if avail <= 1e-12:
                    del self.bids[p]
                else:
                    self.bids[p] = avail
                if remaining <= 1e-12:
                    break

    def snapshot(self):
        return {"bids": dict(self.bids), "asks": dict(self.asks)}


class L2Replay:
    """
    Accepts:
      - l2_diff_df: DataFrame of orderbook diffs with columns ['timestamp','side','price','size','update_type']
          where update_type in {'snapshot','update','delete'}.
      - trades_df: DataFrame of trade prints with ['timestamp','price','size','side'] (aggressor side)
    The class will apply diffs in timestamp order to maintain an L2 orderbook and can feed trades.
    """
    def __init__(self, l2_diff_df: pd.DataFrame, trades_df: pd.DataFrame):
        # sort by timestamp
        self.diffs = l2_diff_df.sort_index() if l2_diff_df is not None else pd.DataFrame()
        self.trades = trades_df.sort_index() if trades_df is not None else pd.DataFrame()
        self.book = L2OrderBook()

    def apply_diffs_up_to(self, ts):
        # apply diffs up to timestamp ts (inclusive)
        if self.diffs.empty:
            return
        for _, row in self.diffs.loc[:ts].iterrows():
            side = row.get("side")
            price = float(row.get("price"))
            size = float(row.get("size", 0.0))
            utype = row.get("update_type", "update")
            if utype in ("snapshot","update"):
                if side in ("bid","ask"):
                    self.book.set_level(side, price, size)
            elif utype == "delete":
                if side in ("bid","ask"):
                    self.book.set_level(side, price, 0.0)

    def trades_since(self, ts):
        return self.trades.loc[ts:]

    def top_of_book_at(self, ts):
        self.apply_diffs_up_to(ts)
        return self.book.top_of_book()
