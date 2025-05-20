import backtrader as bt
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import json
from typing import Dict, Optional, Any
import time
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

from logger import setup_logger
from strategies import calculate_indicators
from llm import get_llm_response

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

# Define test period
TEST_TIMEFRAME = TimeFrame.Day
TEST_START_DATE = datetime(2016, 1, 4, tzinfo=timezone.utc)  # First trading day of 2016
TEST_END_DATE = datetime(2016, 6, 1, tzinfo=timezone.utc)


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

    params = (("lookback_periods", 60),)

    def __init__(self):
        """Initialize strategy parameters."""
        self.order = None
        self.current_trade = None
        self.trades = []  # List to store all trades

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
        # Get position size as percentage from LLM analysis
        position_size_pct = analysis.get("position_size")
        available_cash = self.broker.getcash()
        position_value = available_cash * position_size_pct
        current_price = self.data.close[0]

        # Calculate number of shares based on position value
        size = int(position_value / current_price)
        size = 1 if size < 1 else size

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

        # Check if we have enough data points
        if len(self.data) < self.p.lookback_periods:
            logger.info(f" . Waiting for more data... Current: {len(self.data)}, Required: {self.p.lookback_periods}")
            return

        # Skip if we have a pending order
        if self.order:
            logger.info(" . Waiting for pending order to complete")
            return

        # Check if we already have an open position
        if self.position:
            logger.info(f" . Position already open: Size={self.position.size}, Price=${self.data.close[0]:.2f}")
            return

        # Get current data
        current_data = self.get_current_data()

        # Get LLM analysis
        analysis = self.get_llm_analysis(current_data)
        if not analysis:
            return

        # Process trade if recommendation is BUY or SELL
        if analysis["recommendation"] in ["BUY", "SELL"]:
            self.execute_trade(analysis)

    def get_current_data(self):
        """Get current market data for LLM analysis.

        Returns:
            DataFrame containing current market data and indicators
        """
        # Create DataFrame with current market data
        df = pd.DataFrame()
        df["datetime"] = [self.data.datetime.datetime(i) for i in range(-self.p.lookback_periods, 1)]
        df["open"] = [self.data.open[i] for i in range(-self.p.lookback_periods, 1)]
        df["high"] = [self.data.high[i] for i in range(-self.p.lookback_periods, 1)]
        df["low"] = [self.data.low[i] for i in range(-self.p.lookback_periods, 1)]
        df["close"] = [self.data.close[i] for i in range(-self.p.lookback_periods, 1)]
        df["volume"] = [self.data.volume[i] for i in range(-self.p.lookback_periods, 1)]

        # Set datetime as index
        df.set_index("datetime", inplace=True)

        # Calculate indicators using the imported function
        df = calculate_indicators(df)

        return df

    def get_llm_analysis(self, data):
        """Get trading analysis from LLM.

        Args:
            data: Market data for analysis

        Returns:
            Dictionary containing trading analysis or None if error
        """
        logger.info(" . Getting LLM analysis")

        # Create account information dictionary with account metrics
        account_info = {
            "portfolio_value": self.broker.getvalue(),
            "cash": self.broker.getcash(),
            "buying_power": self.broker.getcash(),  # In backtest, buying power equals cash
            "daytrade_count": 0,  # Not tracked in backtest
            "pattern_day_trader": False,  # Not applicable in backtest
            "shorting_enabled": True,  # Always enabled in backtest
            "status": "ACTIVE",  # Always active in backtest
        }

        analysis = None
        num_retries = 0
        RETRY_LIMIT = 3
        while num_retries < RETRY_LIMIT and not analysis:
            try:
                analysis = get_llm_response(data, account_info)
                if analysis:
                    break
            except Exception as e:
                logger.error(f"Error getting LLM analysis (attempt {num_retries + 1}/{RETRY_LIMIT}): {str(e)}")
                num_retries += 1
                if num_retries < RETRY_LIMIT:
                    time.sleep(1)  # Wait before retrying

        if not analysis:
            logger.error("Failed to get LLM analysis after all retries")
            return None

        return analysis


def get_historical_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Get historical data for a symbol from Alpaca.

    Args:
        symbol: Trading symbol (e.g., 'AAPL')
        start_date: Start date for data
        end_date: End date for data

    Returns:
        DataFrame containing historical market data

    Raises:
        ValueError: If no data is available for the specified date range
    """
    logger.info(" . Fetching historical data")
    # Convert to UTC if not already
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    logger.info(f"Requesting data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    request_params = StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TEST_TIMEFRAME, start=start_date, end=end_date
    )

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

    # Calculate lookback start date
    lookback_start = TEST_START_DATE - timedelta(days=30)  # Add 30 days for indicators

    df = get_historical_data(symbol, lookback_start, TEST_END_DATE)

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


def main() -> None:
    """Run the backtest and save results."""
    logger.info("Starting backtest...")

    # Run backtest
    results = run_backtest()

    if results:
        # Print results
        logger.info("\nBacktest Results:")
        logger.info("=" * 50)
        logger.info(f"Initial Value: ${results['initial_value']:,.2f}")
        logger.info(f"Final Value: ${results['final_value']:,.2f}")
        logger.info(f"Total Return: {results['total_return']*100:.2f}%")
        logger.info(f"Net P&L: ${results['net_pnl']:,.2f}")
        logger.info(f"Total Commission: ${results['total_commission']:,.2f}")

        # Print trade summary
        logger.info("\nTrade Summary:")
        logger.info("=" * 50)
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
        logger.info("=" * 50)
        for i, trade in enumerate(results["trades"], 1):
            logger.info(f"\nTrade #{i}:")
            logger.info(f"Trade Type: ({trade['type']})")
            logger.info(f"Entry Date: {trade['entry_date']}")
            logger.info(f"Exit Date: {trade['exit_date']}")
            logger.info(f"Entry Price: ${trade['entry_price']:.2f}")
            logger.info(f"Exit Price: ${trade['exit_price']:.2f}")
            logger.info(f"Size: {trade['size']}")
            logger.info(f"P&L: ${trade['pnl']:.2f}")
            logger.info(f"Return: {trade['return_pct']:.2f}%")
            logger.info(f"Commission: ${trade['commission']:.2f}")
            logger.info(f"Stop Loss: ${trade['stop_loss']:.2f}")
            logger.info(f"Take Profit: ${trade['take_profit']:.2f}")
            logger.info(f"Exit Reason: {trade['exit_reason']}")
            logger.info(f"Confidence: {trade['confidence']*100:.1f}%")

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
