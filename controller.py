from engine.core import BotConfig, BotEngine
from engine.data import DataSource
from engine.execution import ExecParams, ExecutionEngine
from engine.risk import RiskManager, RiskParams
from replay.replayer import ExecutionSimulator, MarketReplay
from strategies.smc_sweep import backtest_smc_sweep


class Controller:
    def __init__(self, df):
        cfg = BotConfig()
        self.risk = RiskManager(RiskParams())
        self.exec = ExecutionEngine(ExecParams())
        self.data = DataSource(df)
        self.engine = BotEngine(cfg, self.risk, self.exec, self.data)

    def run_backtest_like(self):
        # For compatibility with previous interface, call the strategy backtest on raw df
        return backtest_smc_sweep(self.data.df)

    def run_replay(self, replay: MarketReplay, strategy_func, **kwargs):
        """
        Run a live-ish replay: strategy_func is expected to emit orders when fed bars.
        Strategy_func signature: on_bar(bar, state) -> list of orders
        Orders should be dicts: {'type':'limit'/'market','price':..., 'qty':..., 'side':'long'/'short', 'id':...}
        Execution is handled by ExecutionSimulator which consumes replay ticks and resolves fills.
        """
        sim = ExecutionSimulator(replay)
        state = {"equity": kwargs.get("equity_usd", 10000)}
        trades = []
        for bar in self.data.iter_bars():
            orders = strategy_func(bar, state) or []
            for o in orders:
                res = sim.place_order(o)
                if res.get("filled"):
                    trades.append(res)
        return trades
