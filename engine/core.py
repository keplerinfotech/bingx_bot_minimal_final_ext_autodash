from pydantic import BaseModel
from .risk import RiskManager
from .execution import ExecutionEngine
from .data import DataSource

class BotConfig(BaseModel):
    name: str = "bingx_bot_minimal"
    version: str = "0.1.0"

class BotEngine:
    def __init__(self, config: BotConfig, risk: RiskManager, exec_engine: ExecutionEngine, data: DataSource):
        self.config = config
        self.risk = risk
        self.exec = exec_engine
        self.data = data

    def on_bar(self, bar):
        # placeholder: connect your strategy here
        pass

    def run(self):
        for bar in self.data.iter_bars():
            if self.risk.tripped():
                break
            self.on_bar(bar)
