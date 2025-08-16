import pandas as pd, numpy as np
from replay.l2_replayer import L2Replay, L2OrderBook
from replay.execution_l2 import ExecutionSimulatorL2
from dataclasses import dataclass, field
import math, random, statistics, itertools

@dataclass
class Venue:
    name: str
    l2_diffs: pd.DataFrame
    trades: pd.DataFrame
    replay: L2Replay = field(init=False)
    exec_sim: ExecutionSimulatorL2 = field(init=False)

    def __post_init__(self):
        self.replay = L2Replay(self.l2_diffs, self.trades)
        self.exec_sim = ExecutionSimulatorL2(self.replay)

class LatencyModel:
    """
    Simple latency model returning a simulated network+matching latency in milliseconds.
    You can supply per-venue base latency and jitter (ms).
    """
    def __init__(self, base_ms=20.0, jitter_ms=10.0, rng=None):
        self.base = base_ms
        self.jitter = jitter_ms
        self.rng = rng or random.Random(42)

    def sample_ms(self):
        return max(0.0, self.base + self.rng.normalvariate(0, self.jitter))

class Router:
    """
    Router chooses a venue to post a limit order to, based on a simple expected fill probability heuristic:
      - We compute queue_ahead at the target price on each venue (via L2Replay.top_of_book_at + queue_ahead())
      - We compute recent opposing aggressor flow rate at/through that price over a window
      - Compute expected time-to-fill and probability to fill within a horizon using exponential model
    The router returns ranked venue choices and picks the top one by default.
    """
    def __init__(self, venues: list, latency_models: dict | None = None):
        self.venues = {v.name: v for v in venues}
        self.latency_models = latency_models or {name: LatencyModel() for name in self.venues.keys()}

    def estimate_flow_rate(self, venue: Venue, ts, lookback_ms=60*60*1000):
        # crude: count opposing aggressor volume in last lookback window (ms) relative to ts
        # inputs use pandas Timestamp index; convert window to pandas Timedelta
        w = pd.Timedelta(milliseconds=lookback_ms)
        trades = venue.trades.loc[max(venue.trades.index[0], ts - w):ts]
        # return opposing aggressor volume per ms for both sides as dict
        if trades.empty:
            return {"buy_ms": 0.0, "sell_ms": 0.0}
        # sum sizes grouped by side
        s = trades.groupby("side")["size"].sum().to_dict()
        dt_ms = max(1.0, (ts - trades.index[0]).total_seconds()*1000.0)
        return {"buy_ms": s.get("buy", 0.0)/dt_ms, "sell_ms": s.get("sell", 0.0)/dt_ms}

    def expected_fill_probability(self, venue: Venue, ts, side: str, price: float, qty: float, horizon_ms=60_000):
        """
        Heuristic model:
          - queue_ahead = book.queue_ahead at that price
          - opposing_flow_rate = estimated opposing aggressor volume per ms at that venue
          - expected_agg_in_horizon = opposing_flow_rate * horizon_ms
          - probability ≈ 1 - exp(-(expected_agg_in_horizon) / (queue_ahead + qty + eps))
          - adjust for latency: if latency_ms > horizon_ms then reduce prob
        """
        venue.replay.apply_diffs_up_to(ts)
        book = venue.replay.book
        book_side = "bid" if side=="long" else "ask"
        queue_ahead = book.queue_ahead(book_side, price)
        flow = self.estimate_flow_rate(venue, ts, lookback_ms=60*60*1000)
        opp_rate = flow["sell_ms"] if side=="long" else flow["buy_ms"]
        expected_agg = opp_rate * horizon_ms
        eps = 1e-9
        prob = 1.0 - math.exp(- expected_agg / (queue_ahead + qty + eps))
        # latency impact
        lat = self.latency_models.get(venue.name, LatencyModel()).sample_ms()
        if lat > horizon_ms:
            prob *= max(0.0, horizon_ms/lat)
        return float(max(0.0, min(prob, 1.0)))

    def rank_venues(self, ts, side, price, qty, horizon_ms=60_000):
        scores = []
        for v in self.venues.values():
            p = self.expected_fill_probability(v, ts, side, price, qty, horizon_ms=horizon_ms)
            scores.append((v.name, p))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def choose(self, ts, side, price, qty, horizon_ms=60_000):
        ranked = self.rank_venues(ts, side, price, qty, horizon_ms=horizon_ms)
        return ranked[0] if ranked else (None, 0.0)
