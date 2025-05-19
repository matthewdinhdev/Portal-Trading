import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os
import json
import logging
from typing import Dict, List, Optional, Any
import time
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

from logger import setup_logger
from strategies import calculate_indicators
import llm

# Set up logger
logger = setup_logger("backtest.log")

# Load environment variables
load_dotenv()

# Initialize Alpaca clients
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file")

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Define test period (one week in 2016)
TEST_START_DATE = datetime(2016, 1, 4, tzinfo=timezone.utc)  # First trading day of 2016
TEST_END_DATE = datetime(2016, 2, 4, tzinfo=timezone.utc)


class MarketConditionCache:
    """Cache for market conditions to avoid redundant LLM calls."""

    def __init__(self, max_size=1000, max_age_hours=4):
        self.cache = {}
        self.max_size = max_size
        self.max_age = timedelta(hours=max_age_hours)
        self.hits = 0
        self.misses = 0

    def get_market_condition_hash(self, data):
        """Create a hash of current market conditions."""
        # Get the most recent data point
        current = data.iloc[-1]

        # Create a normalized market condition hash
        return {
            "trend": round(current["trend_strength"], 2) if "trend_strength" in current else 0,
            "volatility": round(current["atr"] / current["close"], 3) if "atr" in current else 0,
            "volume_ratio": round(current["volume"] / current["volume_ma"], 2) if "volume_ma" in current else 1,
            "rsi_zone": round(current["rsi"] / 10) * 10 if "rsi" in current else 50,
            "macd_signal": round(current["macd"] / current["close"], 4) if "macd" in current else 0,
        }

    def get_cache_key(self, market_hash):
        """Convert market hash to a cache key."""
        return json.dumps(market_hash, sort_keys=True)

    def is_similar_condition(self, hash1, hash2, tolerances=None):
        """Check if two market conditions are similar within tolerances."""
        if tolerances is None:
            tolerances = {"trend": 0.1, "volatility": 0.01, "volume_ratio": 0.2, "rsi_zone": 10, "macd_signal": 0.001}

        logger.info("Comparing market conditions:")
        logger.info(f"Current: {hash1}")
        logger.info(f"Cached:  {hash2}")
        logger.info("Differences:")

        for key in hash1:
            diff = abs(hash1[key] - hash2[key])
            is_similar = diff <= tolerances[key]
            logger.info(f"  {key}: {diff:.4f} (tolerance: {tolerances[key]}, similar: {is_similar})")
            if not is_similar:
                return False
        return True

    def get(self, data, current_time):
        """Get cached analysis for current market conditions."""
        current_hash = self.get_market_condition_hash(data)
        current_key = self.get_cache_key(current_hash)

        # Check for exact match first
        if current_key in self.cache:
            cache_entry = self.cache[current_key]
            if current_time - cache_entry["timestamp"] <= self.max_age:
                logger.info(f"Found exact match in cache from {cache_entry['timestamp']}")
                self.hits += 1
                return cache_entry["analysis"]

        # Check for similar conditions
        for key, entry in self.cache.items():
            if current_time - entry["timestamp"] <= self.max_age:
                cached_hash = json.loads(key)
                logger.info(f"\nChecking cache entry from {entry['timestamp']}")
                if self.is_similar_condition(current_hash, cached_hash):
                    self.hits += 1
                    return entry["analysis"]

        self.misses += 1
        return None

    def put(self, data, analysis, current_time):
        """Store analysis in cache."""
        current_hash = self.get_market_condition_hash(data)
        current_key = self.get_cache_key(current_hash)

        # Remove oldest entry if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]

        self.cache[current_key] = {"analysis": analysis, "timestamp": current_time, "hit_count": 0}

    def get_stats(self):
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": hit_rate, "cache_size": len(self.cache)}


class LLMStrategy(bt.Strategy):
    """
    Strategy that uses LLM for trading decisions
    """

    params = (("lookback_periods", 60),)

    def __init__(self):
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.current_position = None
        self.trades = []  # List to store all trades
        self.cache = MarketConditionCache()  # Initialize market condition cache

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, "
                    f"Comm: {order.executed.comm:.2f}"
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(
                    f"SELL EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Cost: {order.executed.value:.2f}, "
                    f"Comm: {order.executed.comm:.2f}"
                )

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        # Store trade information
        trade_info = {
            "entry_date": self.data.datetime.datetime(0).strftime("%Y-%m-%d %H:%M:%S"),
            "exit_date": self.data.datetime.datetime(0).strftime("%Y-%m-%d %H:%M:%S"),
            "entry_price": trade.price,
            "exit_price": (
                trade.pnl / trade.size + trade.price if trade.size != 0 else trade.price
            ),  # Calculate exit price
            "size": trade.size,
            "pnl": trade.pnl,
            "pnlcomm": trade.pnlcomm,
            "status": "WIN" if trade.pnl > 0 else "LOSS",
        }
        self.trades.append(trade_info)

        self.log(f"OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}")

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f"{dt.isoformat()} {txt}")

    def next(self):
        """Main strategy logic"""
        logger.info("Next")

        # Check if we have enough data points
        if len(self.data) < self.p.lookback_periods:
            logger.info(f" . Waiting for more data... Current: {len(self.data)}, Required: {self.p.lookback_periods}")
            return

        # Get current data
        current_data = self.get_current_data()

        # Get LLM analysis
        analysis = self.get_llm_analysis(current_data)

        # Skip trading if analysis is None (cache hit)
        if analysis is None:
            return

        # Process trade if recommendation is BUY or SELL
        if analysis["recommendation"] in ["BUY", "SELL"]:
            if self.order:
                return

            if not self.position:  # No position
                if analysis["recommendation"] == "BUY":
                    self.order = self.buy()
                    self.current_position = {
                        "type": "BUY",
                        "stop_loss": analysis["price_targets"]["stop_loss"],
                        "take_profit": analysis["price_targets"]["take_profit"],
                        "trade_type": analysis["trade_type"],
                    }
                else:  # SELL
                    self.order = self.sell()
                    self.current_position = {
                        "type": "SELL",
                        "stop_loss": analysis["price_targets"]["stop_loss"],
                        "take_profit": analysis["price_targets"]["take_profit"],
                        "trade_type": analysis["trade_type"],
                    }
            else:  # Have position
                # Check stop loss
                if self.current_position["type"] == "BUY":
                    if self.data.close[0] <= self.current_position["stop_loss"]:
                        self.order = self.sell()
                        self.current_position = None
                else:  # SELL position
                    if self.data.close[0] >= self.current_position["stop_loss"]:
                        self.order = self.buy()
                        self.current_position = None

                # Check take profit
                if self.current_position:
                    if self.current_position["type"] == "BUY":
                        if self.data.close[0] >= self.current_position["take_profit"]:
                            self.order = self.sell()
                            self.current_position = None
                    else:  # SELL position
                        if self.data.close[0] <= self.current_position["take_profit"]:
                            self.order = self.buy()
                            self.current_position = None

    def get_current_data(self):
        """Get current data for LLM analysis"""

        logger.info(" . Generating data for LLM analysis")

        # collect historical data
        data = []
        for i in range(-self.p.lookback_periods, 1):
            dt = self.data.datetime.datetime(i)
            data.append(
                {
                    "datetime": dt,
                    "open": float(self.data.open[i]),
                    "high": float(self.data.high[i]),
                    "low": float(self.data.low[i]),
                    "close": float(self.data.close[i]),
                    "volume": float(self.data.volume[i]),
                }
            )

        # create indicator dataframe
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index("datetime", inplace=True)
            df.index = pd.to_datetime(df.index)

            # Use the calculate_indicators function from strategies.py
            df = calculate_indicators(df)

            if df.empty:
                raise ValueError("No data available for LLM analysis after calculating indicators")
            elif len(df) < 5:
                raise ValueError("Insufficient data points for LLM analysis")
        return df

    def get_llm_analysis(self, data):
        """Get trading analysis from LLM"""
        current_time = self.data.datetime.datetime(0)

        # Try to get analysis from cache first
        cached_analysis = self.cache.get(data, current_time)
        if cached_analysis:
            logger.info(f" . Using cached analysis for {current_time} (skipping trade execution)")
            # Return None to skip trading for this period
            return None

        logger.info(f" . Getting LLM analysis for {current_time}")

        analysis = None
        num_retries = 0
        RETRY_LIMIT = 3
        while not analysis:
            analysis = llm.get_llm_response(
                df=data, lookback_periods=self.p.lookback_periods, analysis_date=current_time
            )
            if not analysis:
                logger.info(" . No analysis available from LLM, retrying...")
                time.sleep(3)
                num_retries += 1
                if num_retries > RETRY_LIMIT:
                    raise ValueError(f" . Failed to get analysis from LLM after {RETRY_LIMIT} retries")

        # Cache the new analysis with backtest timestamp
        self.cache.put(data, analysis, current_time)

        # Log cache statistics periodically
        if len(self.trades) % 10 == 0:  # Log every 10 trades
            stats = self.cache.get_stats()
            logger.info(
                f"Cache stats - Hits: {stats['hits']}, Misses: {stats['misses']}, Hit Rate: {stats['hit_rate']:.1f}%"
            )

        return analysis


def get_historical_data(
    symbol: str, start_date: datetime, end_date: datetime, timeframe: TimeFrame = TimeFrame.Hour
) -> pd.DataFrame:
    """Get historical data for a symbol from Alpaca."""
    logger.info(" . Fetching historical data")
    # Convert to UTC if not already
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    logger.info(f"Requesting data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    request_params = StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start_date, end=end_date)

    bars = data_client.get_stock_bars(request_params)
    df = bars.df

    # Reset multi-index to single index
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    if df.empty:
        raise ValueError("No historical data available for the specified date range")

    return df


def run_backtest(
    symbol: str = "SPY",
    initial_capital: float = 100000,
) -> Optional[Dict[str, Any]]:
    """Run a backtest using Backtrader."""
    logger.info(
        f"Running backtest for {symbol} from {TEST_START_DATE.strftime('%Y-%m-%d')} to {TEST_END_DATE.strftime('%Y-%m-%d')}"
    )

    # Get historical data with additional lookback period
    lookback_start = TEST_START_DATE - timedelta(days=5)
    df = get_historical_data(symbol, lookback_start, TEST_END_DATE, TimeFrame.Hour)

    if df.empty:
        raise ValueError("No historical data available for the specified date range")

    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Create a Data Feed
    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,  # Use index as datetime
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,
    )

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(initial_capital)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.PercentSizer, percents=10)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)

    # Add a strategy
    cerebro.addstrategy(LLMStrategy)

    # Print out the starting conditions
    logger.info(f"Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}")

    # Run over everything
    results = cerebro.run()

    # Get the strategy instance from the results
    strategy = results[0]

    # Print out the final result
    logger.info(f"Final Portfolio Value: ${cerebro.broker.getvalue():.2f}")

    return {
        "initial_value": initial_capital,
        "final_value": cerebro.broker.getvalue(),
        "return": (cerebro.broker.getvalue() - initial_capital) / initial_capital,
        "trades": strategy.trades,  # Add trades to the results
    }


def main() -> None:
    """Main function to run the backtest."""
    logger.info("Starting backtest...")

    # Run backtest
    results = run_backtest()

    if results:
        # Print results
        logger.info("\nBacktest Results:")
        logger.info("=" * 50)
        logger.info(f"Initial Value: ${results['initial_value']:,.2f}")
        logger.info(f"Final Value: ${results['final_value']:,.2f}")
        logger.info(f"Total Return: {results['return']*100:.2f}%")

        # Print trade summary
        logger.info("\nTrade Summary:")
        logger.info("=" * 50)
        total_trades = len(results["trades"])
        winning_trades = sum(1 for trade in results["trades"] if trade["status"] == "WIN")
        losing_trades = total_trades - winning_trades
        total_pnl = sum(trade["pnl"] for trade in results["trades"])
        total_commission = sum(trade["pnl"] - trade["pnlcomm"] for trade in results["trades"])

        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Winning Trades: {winning_trades}")
        logger.info(f"Losing Trades: {losing_trades}")
        logger.info(f"Win Rate: {(winning_trades/total_trades*100):.2f}%")
        logger.info(f"Total P&L: ${total_pnl:,.2f}")
        logger.info(f"Total Commission: ${total_commission:,.2f}")

        # Print individual trades
        logger.info("\nIndividual Trades:")
        logger.info("=" * 50)
        for i, trade in enumerate(results["trades"], 1):
            logger.info(f"\nTrade #{i}:")
            logger.info(f"Entry Date: {trade['entry_date']}")
            logger.info(f"Exit Date: {trade['exit_date']}")
            logger.info(f"Entry Price: ${trade['entry_price']:.2f}")
            logger.info(f"Exit Price: ${trade['exit_price']:.2f}")
            logger.info(f"Size: {trade['size']}")
            logger.info(f"P&L: ${trade['pnl']:.2f}")
            logger.info(f"Net P&L (with commission): ${trade['pnlcomm']:.2f}")
            logger.info(f"Status: {trade['status']}")

        # Save results to file
        output_dir = "backtest_results"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_results_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to {filepath}")
    else:
        logger.error("Backtest failed")


if __name__ == "__main__":
    main()
