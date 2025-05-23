from enum import Enum


class TradingEnvironment(Enum):
    PAPER = "paper_trading"
    LIVE = "live_trading"
    BACKTEST = "backtest"
