import pandas as pd, numpy as np, os, random, math
from replay.l2_replayer import L2Replay
from replay.execution_l2 import ExecutionSimulatorL2

class VenueConfig:
    def __init__(self, name, latency_mean_ms=20, latency_jitter_ms=5):
        self.name = name
        self.latency_mean_ms = latency_mean_ms
        self.latency_jitter_ms = latency_jitter_ms

    def sample_latency(self):
        return max(0, np.random.normal(self.latency_mean_ms, self.latency_jitter_ms))

class MultiVenueRouter:
    def __init__(self, venue_data: dict):
        """
        venue_data: {venue_name: (L2Replay, VenueConfig)}
        """
        self.venue_data = venue_data
        self.simulators = {v: ExecutionSimulatorL2(l2r) for v, (l2r, cfg) in venue_data.items()}
        self.configs = {v: cfg for v, (l2r, cfg) in venue_data.items()}
        self.router_log = []

    def decide_and_place(self, order):
        # Evaluate fill probability per venue given latency
        venue_scores = []
        for venue, sim in self.simulators.items():
            cfg = self.configs[venue]
            latency_ms = cfg.sample_latency()
            # Fill probability ~ 1 / (queue_ahead + latency factor)
            replay = self.venue_data[venue][0]
            tob = replay.top_of_book_at(order["timestamp"])
            if order["side"] == "long":
                price = tob["best_bid"]
            else:
                price = tob["best_ask"]
            if price is None:
                continue
            queue_ahead = replay.queue_ahead_at(order["timestamp"], order["side"], price)
            # small model: probability = exp(- (queue_ahead/size_scale + latency_ms/latency_scale))
            prob = math.exp(-(queue_ahead/10.0 + latency_ms/50.0))
            venue_scores.append((venue, prob, latency_ms, price, queue_ahead))

        if not venue_scores:
            return None

        # Pick best venue
        venue_scores.sort(key=lambda x: x[1], reverse=True)
        best_venue, best_prob, lat, price, qa = venue_scores[0]

        # Adjust order to venue's price
        routed_order = order.copy()
        routed_order["price"] = float(price)
        res = self.simulators[best_venue].place_limit(routed_order)

        # Log decision
        self.router_log.append({
            "order_id": order["id"], "chosen_venue": best_venue, "prob": best_prob, "latency_ms": lat,
            "price": price, "queue_ahead": qa
        })
        return res

    def export_fill_report(self):
        reports = []
        for venue, sim in self.simulators.items():
            rep = sim.export_fill_report()
            rep["venue"] = venue
            reports.append(rep)
        router_df = pd.DataFrame(self.router_log)
        return pd.concat(reports, ignore_index=True), router_df

def synthesize_multi_venue(start_ts="2024-01-01", periods=24*3*4, freq="20T", venues=("Binance","Bybit","OKX"), seed=42):
    np.random.seed(seed)
    data = {}
    for venue in venues:
        # Slightly different price paths
        l2_df, trades_df = synthesize_week_data(start_ts=start_ts, periods=periods, freq=freq, seed=seed + hash(venue) % 1000)
        cfg = VenueConfig(venue, latency_mean_ms=random.choice([10, 20, 35]), latency_jitter_ms=random.choice([2,5,10]))
        data[venue] = (L2Replay(l2_df, trades_df), cfg)
    return data

def run_multi_venue_demo(out_csv="fill_report_multi.csv", out_router_csv="router_decisions.csv"):
    venues = synthesize_multi_venue()
    router = MultiVenueRouter(venues)
    # Place orders every hour
    idx_hours = sorted(list(set(list(venues.values())[0][0].l2_df.index.floor("H"))))
    for i, ts in enumerate(idx_hours):
        order = {"id":f"o{i}", "timestamp": ts, "side":"long" if i%2==0 else "short", "price": None, "qty": 1.0}
        router.decide_and_place(order)
    fills, router_log = router.export_fill_report()
    fills.to_csv(out_csv, index=False)
    router_log.to_csv(out_router_csv, index=False)
    print("Wrote:", out_csv, out_router_csv)
    return fills, router_log

# import synthesize_week_data from week replay
from scripts.run_week_replay import synthesize_week_data

if __name__ == "__main__":
    run_multi_venue_demo()
