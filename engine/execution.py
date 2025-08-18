from dataclasses import dataclass


@dataclass
class ExecParams:
    protection_ticks: int = 2
    post_only: bool = True
    reduce_only: bool = True


class ExecutionEngine:
    def __init__(self, params: ExecParams):
        self.params = params

    def place_limit(
        self,
        symbol: str,
        price: float,
        qty: float,
        side: str,
        post_only=None,
        reduce_only=None,
    ):
        # placeholder for real exchange integration
        return {
            "cl_id": f"LIM-{symbol}-{side}",
            "price": price,
            "qty": qty,
            "side": side,
        }

    def place_market(self, symbol: str, qty: float, side: str):
        return {"cl_id": f"MRK-{symbol}-{side}", "qty": qty, "side": side}
