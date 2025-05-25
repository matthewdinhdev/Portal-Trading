import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import backtrader as bt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from logger import setup_logger
from trading_enums import TradingEnvironment
from utility import (
    TRADING_STRATEGY,
    MarketDataManager,
)

# Set up logger
logger = setup_logger("backtest.log")

# Setup market data manager
market_data_manager = MarketDataManager()

# Suppress all matplotlib logging
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# Constants
TEST_START_DATE = datetime(2017, 1, 4)
# TEST_END_DATE = datetime(2017, 4, 4)
TEST_END_DATE = datetime(2021, 1, 4)


class Trade:
    """Class to store and manage trade information."""

    def __init__(
        self,
        type: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        size: int,
        entry_date: str,
        confidence: float,
    ):
        """Initialize a new trade.

        Args:
            type: Trade type (BUY/SELL)
            entry_price: Entry price of the trade
            stop_loss: Stop loss price
            take_profit: Take profit price
            size: Number of shares
            entry_date: Entry date and time
            confidence: Confidence level of the trade (0-1)
        """
        self.type = type
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.size = size
        self.entry_date = entry_date
        self.confidence = confidence
        self.exit_date = None
        self.exit_price = None
        self.pnl = 0.0
        self.pnlcomm = 0.0
        self.commission = 0.0
        self.status = None
        self.exit_reason = None
        self.return_pct = 0.0

    def close(self, exit_price: float, exit_date: str, pnl: float, pnlcomm: float, commission: float) -> None:
        """Close the trade and calculate final metrics.

        Args:
            exit_price: Price at which the trade was closed
            exit_date: Date and time when the trade was closed
            pnl: Gross profit/loss
            pnlcomm: Net profit/loss after commission
            commission: Commission paid
        """
        self.exit_price = exit_price
        self.exit_date = exit_date
        self.pnl = pnl
        self.pnlcomm = pnlcomm
        self.commission = commission
        self.status = "WIN" if pnl > 0 else "LOSS"
        self.return_pct = ((exit_price - self.entry_price) / self.entry_price) * 100

        # Determine exit reason based on price levels with a small tolerance
        tolerance = 0.01  # 1% tolerance for price levels
        if exit_price <= self.stop_loss * (1 + tolerance):
            self.exit_reason = "STOP_LOSS"
        elif exit_price >= self.take_profit * (1 - tolerance):
            self.exit_reason = "TAKE_PROFIT"
        else:
            # If price is close to stop loss or take profit, use that as the reason
            if abs(exit_price - self.stop_loss) < abs(exit_price - self.take_profit):
                self.exit_reason = "STOP_LOSS"
            else:
                self.exit_reason = "TAKE_PROFIT"

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade information to dictionary format.

        Returns:
            Dictionary containing all trade information
        """
        return {
            "type": self.type,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "pnl": self.pnl,
            "pnlcomm": self.pnlcomm,
            "status": self.status,
            "commission": self.commission,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "return_pct": self.return_pct,
            "exit_reason": self.exit_reason,
            "confidence": self.confidence,
        }


class LLMStrategy(bt.Strategy):
    """Strategy that uses LLM for trading decisions."""

    params = (
        (
            "lookback_periods",
            TRADING_STRATEGY["lookback_periods"],
        ),
    )

    def __init__(self):
        """Initialize strategy parameters."""
        self.order = None
        self.current_trade = None
        self.trades = []  # List to store all trades
        self.data_window = None  # Will be initialized in start

    def start(self):
        """Called once before the strategy starts running."""
        # Initialize data window with all historical data
        logger.info("Getting historical data for data window...")

        # Calculate lookback start date (add buffer for weekends and holidays)
        logger.info(f" . Test start date: {TEST_START_DATE}")
        lookback_start = TEST_START_DATE - timedelta(days=self.p.lookback_periods)
        logger.info(f" . Lookback start date: {lookback_start}")

        # Get historical data directly - only up to test start date
        self.data_window = market_data_manager.get_historical_data(
            symbol=self.data._name,
            start_date=lookback_start,
            end_date=TEST_START_DATE,  # Only get data up to test start
        )

        if self.data_window.empty:
            logger.error("No historical data available for data window")
            return

    def notify_order(self, order):
        """Handle order notifications.

        Args:
            order: Order object from broker
        """
        if order.status in [order.Submitted, order.Accepted]:
            return

        elif order.status == order.Margin:
            logger.info(
                f"MARGIN REJECTION - Size: {order.size}, "
                f"Current Position: {self.position.size if self.position else 0}, "
                f"Cash: ${self.broker.getcash():.2f}, "
                f"Value: ${self.broker.getvalue():.2f}, "
                f"Order Type: {'BUY' if order.isbuy() else 'SELL'}"
            )
            self.position_closing = False  # Reset the flag if order is rejected

        self.order = None

    def notify_trade(self, trade):
        """Handle trade notifications.

        Args:
            trade: Trade object from broker
        """
        if not trade.isclosed:
            return

        # Calculate entry and exit prices
        is_buy = trade.size > 0  # Positive size means buy, negative means sell
        exit_price = trade.data.low[0] if is_buy else trade.data.high[0]  # Low for buys, high for sells

        # Close the current trade
        if self.current_trade:
            self.current_trade.close(
                exit_price=exit_price,
                exit_date=self.data.datetime.datetime(0).strftime("%Y-%m-%d %H:%M:%S"),
                pnl=trade.pnl,
                pnlcomm=trade.pnlcomm,
                commission=trade.pnl - trade.pnlcomm,
            )
            self.trades.append(self.current_trade)  # Store the Trade object directly
            self.current_trade = None

    def execute_trade(self, analysis: Dict[str, Any]) -> None:
        """Execute a trade based on LLM analysis.

        Args:
            analysis: Dictionary containing trading analysis and recommendations
        """
        logger.info(" . Executing trade")

        # check if recommendation is HOLD
        if analysis["recommendation"] == "HOLD":
            logger.info("   . Reccomendation is HOLD. Skipping trade.")
            return

        # Calculate position size using utility function
        current_price = self.data.close[0]
        size = market_data_manager.calculate_position_size(
            confidence=analysis["confidence"], available_cash=self.broker.getcash(), current_price=current_price
        )

        # Set variables for the order
        entry_price = current_price
        stop_loss = analysis["price_targets"]["stop_loss"]
        take_profit = analysis["price_targets"]["take_profit"]

        if analysis["recommendation"] == "BUY":
            # Place the main order with bracket orders
            self.order = self.buy_bracket(
                size=size,
                price=entry_price,
                stopprice=stop_loss,
                limitprice=take_profit,
                exectype=bt.Order.Market,  # Main order is market
                stopexec=bt.Order.Stop,  # Stop loss is stop order
                limitexec=bt.Order.Limit,  # Take profit is limit order
            )
        else:  # SELL
            # Place the main order with bracket orders
            self.order = self.sell_bracket(
                size=size,
                price=entry_price,
                stopprice=stop_loss,
                limitprice=take_profit,
                exectype=bt.Order.Market,  # Main order is market
                stopexec=bt.Order.Stop,  # Stop loss is stop order
                limitexec=bt.Order.Limit,  # Take profit is limit order
            )

        # Create new trade object
        self.current_trade = Trade(
            type=analysis["recommendation"],
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=size,
            entry_date=self.data.datetime.datetime(0).strftime("%Y-%m-%d %H:%M:%S"),
            confidence=analysis["confidence"],
        )

    def next(self):
        """Execute strategy logic for each bar."""
        current_date = self.data.datetime.datetime(0)
        logger.info(f"Processing date: {current_date.strftime('%Y-%m-%d %H:%M:%S')}")

        # Skip if we have a pending order
        if self.order:
            logger.info(" . Waiting for pending order to complete")
            return

        # Check if we already have an open position
        if self.position:
            logger.info(f" . Position already open: Size={self.position.size}, Price=${self.data.close[0]:.2f}")
            return

        # Update data window with new data point
        new_data = pd.DataFrame(
            {
                "datetime": [current_date],
                "open": [self.data.open[0]],
                "high": [self.data.high[0]],
                "low": [self.data.low[0]],
                "close": [self.data.close[0]],
                "volume": [self.data.volume[0]],
            }
        )
        new_data.set_index("datetime", inplace=True)

        # Append new data and keep only the lookback period
        self.data_window = pd.concat([self.data_window, new_data])
        if len(self.data_window) > self.p.lookback_periods:
            self.data_window = self.data_window.iloc[-self.p.lookback_periods :]

        # verify that data_window has enough data
        num_days_six_months = 126
        if len(self.data_window) < num_days_six_months:
            logger.error(
                f"Not enough data to make a decision. Need {num_days_six_months} days, have {len(self.data_window)}"
            )
            return

        # Get LLM analysis using utility function
        result = market_data_manager.analyze_symbol(
            symbol=self.data._name,
            data=self.data_window,
            env=TradingEnvironment.BACKTEST,
            save_analysis=False,
        )

        if not result:
            raise Exception(f"No analysis found for {self.data._name}")

        self.execute_trade(result["analysis"])


def run_backtest(
    symbol: str = "SPY",
    initial_capital: float = 100000,
) -> Optional[Dict[str, Any]]:
    """Run a backtest using Backtrader.

    Args:
        symbol: Trading symbol to backtest (default: 'SPY')
        initial_capital: Initial capital for backtest (default: 100000)

    Returns:
        Dictionary containing backtest results or None if error
    """
    logger.info(
        f"Running backtest for {symbol} from {TEST_START_DATE.strftime('%Y-%m-%d')} to {TEST_END_DATE.strftime('%Y-%m-%d')}"
    )

    df = market_data_manager.get_historical_data(symbol, TEST_START_DATE, TEST_END_DATE)

    if df.empty:
        raise ValueError("No historical data available for the specified date range")

    logger.info(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")

    # Validate DataFrame format
    required_columns = ["open", "high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

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
        name=symbol,  # Explicitly set the symbol name
        timeframe=bt.TimeFrame.Days,  # Explicitly set timeframe
        compression=1,  # No compression
    )

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(initial_capital)

    # Set the commission to match Alpaca's structure
    cerebro.broker.setcommission(
        commission=0.01,  # $0.01 per share
        mult=1.0,  # Multiplier
        margin=False,  # No margin
        automargin=False,  # No auto margin
        commtype=bt.CommInfoBase.COMM_FIXED,  # Fixed commission per share
        percabs=False,  # Not percentage based
        stocklike=True,  # Stock-like instrument
    )

    # Configure slippage
    cerebro.broker.set_slippage_perc(0.001)  # 0.1% slippage
    cerebro.broker.set_slippage_fixed(0.01)  # $0.01 fixed slippage

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

    # Calculate performance metrics
    portfolio_value = cerebro.broker.getvalue()
    total_return = (portfolio_value - initial_capital) / initial_capital
    trades = [trade.to_dict() for trade in strategy.trades]

    # Calculate trade metrics
    total_trades = len(trades)
    winning_trades = sum(1 for trade in trades if trade["status"] == "WIN")
    losing_trades = total_trades - winning_trades
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Calculate P&L metrics
    total_pnl = sum(trade["pnl"] for trade in trades)
    total_commission = sum(trade["commission"] for trade in trades)
    net_pnl = total_pnl - total_commission

    # Calculate average trade metrics
    avg_win = (
        sum(trade["pnl"] for trade in trades if trade["status"] == "WIN") / winning_trades if winning_trades > 0 else 0
    )
    avg_loss = (
        sum(trade["pnl"] for trade in trades if trade["status"] == "LOSS") / losing_trades if losing_trades > 0 else 0
    )
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # Calculate trades per day
    trades_per_day = total_trades / ((TEST_END_DATE - TEST_START_DATE).days + 1)

    # Calculate drawdown
    peak = initial_capital
    max_drawdown = 0
    current_drawdown = 0
    for trade in trades:
        peak = max(peak, trade["entry_price"] * trade["size"])
        current_drawdown = (peak - trade["exit_price"] * trade["size"]) / peak
        max_drawdown = max(max_drawdown, current_drawdown)

    # Calculate risk metrics
    avg_risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    sharpe_ratio = (total_return / max_drawdown) if max_drawdown > 0 else float("inf")

    return {
        "initial_value": initial_capital,
        "final_value": portfolio_value,
        "total_return": total_return,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "total_pnl": total_pnl,
        "net_pnl": net_pnl,
        "total_commission": total_commission,
        "trades_per_day": trades_per_day,
        "avg_risk_reward": avg_risk_reward,
        "sharpe_ratio": sharpe_ratio,
        "trades": trades,
    }


def plot_trades(df: pd.DataFrame, trades: list, symbol: str) -> None:
    """Plot price action and trades.

    Args:
        df: DataFrame containing price data
        trades: List of trade dictionaries
        symbol: Trading symbol
    """
    plt.figure(figsize=(15, 8))

    # Plot price action
    plt.plot(df.index, df["close"], label="Price", color="gray", alpha=0.5)

    # Plot trades
    for trade in trades:
        entry_date = pd.to_datetime(trade["entry_date"])
        exit_date = pd.to_datetime(trade["exit_date"])

        # Plot entry point
        plt.scatter(
            entry_date,
            trade["entry_price"],
            color="green" if trade["type"] == "BUY" else "red",
            marker="^" if trade["type"] == "BUY" else "v",
            s=100,
            label=f"{trade['type']} Entry" if trade == trades[0] else "",
        )

        # Plot exit point
        plt.scatter(
            exit_date,
            trade["exit_price"],
            color="blue" if trade["status"] == "WIN" else "black",
            marker="o",
            s=100,
            label=f"{trade['status']} Exit" if trade == trades[0] else "",
        )

        # Draw line between entry and exit
        plt.plot(
            [entry_date, exit_date],
            [trade["entry_price"], trade["exit_price"]],
            color="green" if trade["status"] == "WIN" else "red",
            alpha=0.3,
        )

        # Add stop loss and take profit levels
        plt.plot([entry_date, exit_date], [trade["stop_loss"], trade["stop_loss"]], "r--", alpha=0.3)
        plt.plot([entry_date, exit_date], [trade["take_profit"], trade["take_profit"]], "g--", alpha=0.3)

    # Customize plot
    plt.title(f"{symbol} Trading History")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)

    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gcf().autofmt_xdate()

    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Save plot
    output_dir = "backtest_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_dir, f"trade_plot_{timestamp}.png"))
    plt.close()


def main() -> None:
    """Run the backtest and save results."""
    logger.info("Starting backtest...")

    # Run backtest
    results = run_backtest()

    if results:
        # Print results
        logger.info("\nBacktest Results:")
        logger.info("==================================================")
        logger.info(f"Initial Value: ${results['initial_value']:,.2f}")
        logger.info(f"Final Value: ${results['final_value']:,.2f}")
        logger.info(f"Total Return: {results['total_return']*100:.2f}%")
        logger.info(f"Net P&L: ${results['net_pnl']:,.2f}")
        logger.info(f"Total Commission: ${results['total_commission']:,.2f}")
        logger.info("")
        logger.info("Trade Summary:")
        logger.info("==================================================")
        logger.info(f"Test Period: {TEST_START_DATE.strftime('%Y-%m-%d')} to {TEST_END_DATE.strftime('%Y-%m-%d')}")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Winning Trades: {results['winning_trades']}")
        logger.info(f"Losing Trades: {results['losing_trades']}")
        logger.info(f"Win Rate: {results['win_rate']*100:.2f}%")
        logger.info(f"Average Win: ${results['avg_win']:,.2f}")
        logger.info(f"Average Loss: ${results['avg_loss']:,.2f}")
        logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
        logger.info(f"Average Risk/Reward: {results['avg_risk_reward']:.2f}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
        logger.info(f"Trades per Day: {results['trades_per_day']:.1f}")

        # Print individual trades
        logger.info("\nIndividual Trades:")
        logger.info("==================================================")
        for i, trade in enumerate(results["trades"], 1):
            logger.info(f"\nTrade #{i}:")
            logger.info(f"Trade Type: ({trade['type']})")
            logger.info(f"Entry Date: {trade['entry_date']}")
            logger.info(f"Exit Date: {trade['exit_date']}")
            logger.info(f"Confidence: {trade['confidence']*100:.1f}%")
            logger.info(f"Size: {trade['size']}")
            logger.info(f"Entry Price: ${trade['entry_price']:.2f}")
            logger.info(f"Stop Loss: ${trade['stop_loss']:.2f}")
            logger.info(f"Take Profit: ${trade['take_profit']:.2f}")
            logger.info(f"Exit Price: ${trade['exit_price']:.2f}")
            logger.info(f"P&L: ${trade['pnl']:.2f}")
            logger.info(f"Return: {trade['return_pct']:.2f}%")
            logger.info(f"Exit Reason: {trade['exit_reason']}")

        # Get historical data for plotting
        df = market_data_manager.get_historical_data("SPY", TEST_START_DATE, TEST_END_DATE)

        # Plot trades
        plot_trades(df, results["trades"], "SPY")
        logger.info("\nTrade plot has been saved to backtest_results directory")

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
