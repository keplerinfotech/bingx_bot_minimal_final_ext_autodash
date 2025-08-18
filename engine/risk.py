from dataclasses import dataclass


@dataclass
class RiskParams:
    max_daily_loss_bps: float = 50.0
    cool_down_minutes: int = 120


class RiskManager:
    def __init__(self, params: RiskParams):
        self.params = params
        self.day_start_equity = None
        self.cool_down_until = None
        self.tripped_flag = False

    def set_day_start_equity(self, eq: float):
        self.day_start_equity = eq

    def update_pnl(self, realized_pnl: float, now_ts=None):
        if self.day_start_equity is None:
            return
        dd_bps = 10000.0 * (max(0.0, -realized_pnl) / self.day_start_equity)
        if dd_bps > self.params.max_daily_loss_bps:
            self.tripped_flag = True
            # set cool_down_until in your scheduler/clock

    def tripped(self) -> bool:
        return self.tripped_flag
