import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
import strategies
import llm
import os
import json
from logger import setup_logger
from typing import Dict, List, Optional, Any, Union, Tuple

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

# Define trading strategy parameters
TRADING_STRATEGY: Dict[str, Any] = {
    "symbols": ["SPY"],  # S&P 500 ETF
    "timeframe": TimeFrame.Hour,
    "lookback_periods": 120,  # 5 days of hourly data
}


def get_historical_data(
    symbol: str, start_date: datetime, end_date: datetime, timeframe: TimeFrame = TimeFrame.Hour
) -> pd.DataFrame:
    """Get historical data for a symbol from Alpaca.

    Retrieves historical price data for the specified symbol and time range,
    converting dates to UTC if necessary.

    Args:
        symbol: Trading symbol to get data for.
        start_date: Start date for historical data.
        end_date: End date for historical data.
        timeframe: Timeframe for the data (default: TimeFrame.Hour).

    Returns:
        pd.DataFrame: DataFrame containing historical price data with datetime index.

    Raises:
        ValueError: If no historical data is available for the specified date range.
    """
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
    symbol: str,
    timeframe: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    initial_capital: float = 100000,
) -> Optional[Dict[str, Any]]:
    """Run a backtest for a given symbol and date range using a single random day.

    Simulates trading based on LLM analysis for a randomly selected day within
    the specified date range, tracking trades, equity, and performance metrics.

    Args:
        symbol: Trading symbol to backtest.
        timeframe: Timeframe for the backtest (e.g., "1m", "4h", "1d").
        start_date: Start date for backtest (string or datetime).
        end_date: End date for backtest (string or datetime).
        initial_capital: Initial capital for backtest (default: 100000).

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing backtest results including
            trades, equity curve, and performance metrics, or None if backtest fails.

    Raises:
        ValueError: If invalid timeframe, insufficient data, or other errors occur.
    """
    try:
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # Convert to UTC if not already
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        # Convert timeframe string to TimeFrame enum
        timeframe_map = {
            "1m": TimeFrame.Minute,
            "5m": TimeFrame.Minute,
            "15m": TimeFrame.Minute,
            "30m": TimeFrame.Minute,
            "1h": TimeFrame.Hour,
            "4h": TimeFrame.Hour,
            "1d": TimeFrame.Day,
        }

        if timeframe not in timeframe_map:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        tf = timeframe_map[timeframe]

        # Add timeframe multiplier for non-standard intervals
        timeframe_multiplier = {"5m": 5, "15m": 15, "30m": 30, "4h": 4}

        multiplier = timeframe_multiplier.get(timeframe, 1)
        if multiplier > 1:
            tf = TimeFrame(tf.value * multiplier)

        logger.info(
            f"Running backtest for {symbol} from {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}"
        )

        # Get historical data
        logger.info("Fetching historical data...")
        df = get_historical_data(symbol, start_date, end_date, tf)

        if df.empty:
            raise ValueError("No historical data available for the specified date range")

        # Calculate indicators
        logger.info("Calculating technical indicators...")
        df = strategies.calculate_indicators(df)

        # Group data by day and select a random day
        df["date"] = df.index.date
        available_days = sorted(df["date"].unique())

        if len(available_days) == 0:
            raise ValueError("No trading days available in the date range")

        # Select a random day from available data
        selected_day = np.random.choice(available_days)
        day_data = df[df["date"] == selected_day]

        if day_data.empty:
            raise ValueError(f"No data available for selected day: {selected_day}")

        if len(day_data) < 6:  # Ensure we have enough data points for meaningful analysis
            raise ValueError(f"Insufficient data points ({len(day_data)}) for selected day: {selected_day}")

        logger.info(f"\nSelected day for analysis: {selected_day}")
        logger.info(f"Available data points: {len(day_data)}")
        logger.info(f"Data range: {day_data.index[0].strftime('%H:%M')} to {day_data.index[-1].strftime('%H:%M')}")

        # Initialize results
        results: Dict[str, Any] = {
            "trades": [],
            "equity_curve": [],
            "metrics": {
                "total_trades": 0,
                "day_trades": 0,
                "swing_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "final_equity": initial_capital,
                "total_return": 0.0,
                "max_drawdown": 0.0,
            },
            "selected_day": selected_day.strftime("%Y-%m-%d"),
        }

        # Run simulation
        position: int = 0
        equity: float = initial_capital
        entry_price: float = 0
        entry_time: Optional[datetime] = None

        # Get single LLM analysis for the selected day
        logger.info("Getting LLM analysis...")
        try:
            # Set the analysis date to the selected day
            analysis_date = datetime.combine(selected_day, datetime.min.time(), tzinfo=timezone.utc)
            llm_prompt = strategies.get_llm_prompt(df, analysis_date=analysis_date)

            if not llm_prompt:
                raise ValueError("Failed to generate LLM prompt")

            analysis = llm.get_trading_analysis(llm_prompt)

            if not analysis or not isinstance(analysis, dict):
                raise ValueError("Invalid LLM analysis response")

            if "recommendation" not in analysis:
                raise ValueError("LLM analysis missing recommendation")

            logger.info("\nLLM Analysis:")
            logger.info(llm.format_analysis_for_display(analysis))

        except Exception as e:
            logger.error(f"Error getting LLM analysis: {str(e)}")
            return None

        # Process each hour for the day
        for i in range(len(day_data)):
            current_data = day_data.iloc[i]
            current_date = day_data.index[i]

            # Check if we need to close position due to max hold time
            if position != 0 and entry_time is not None:
                hold_time = current_date - entry_time
                trade_type = results["trades"][-1]["trade_type"] if results["trades"] else "day"
                max_hold_time = timedelta(hours=6) if trade_type == "day" else timedelta(days=5)

                if hold_time > max_hold_time:
                    # Close position due to max hold time
                    pnl = (current_data["close"] - entry_price) * position
                    equity += pnl
                    results["trades"].append(
                        {
                            "date": current_date.strftime("%Y-%m-%d %H:%M"),
                            "type": "SELL",
                            "price": current_data["close"],
                            "quantity": position,
                            "value": position * current_data["close"],
                            "pnl": pnl,
                            "reason": "max_hold_time",
                            "trade_type": trade_type,
                        }
                    )
                    position = 0
                    entry_time = None

            # Record equity
            results["equity_curve"].append({"date": current_date.strftime("%Y-%m-%d %H:%M"), "equity": equity})

            # Execute trades based on analysis
            if analysis["recommendation"] == "BUY" and position <= 0:
                # Get position size and trade type from LLM analysis
                position_size_pct = float(analysis.get("position_size", 0.01))  # Default to 1% if not specified
                trade_type = analysis.get("trade_type", "day")  # Default to day trade if not specified

                # Ensure position size is within reasonable bounds
                position_size_pct = max(0.005, min(position_size_pct, 0.05))  # Between 0.5% and 5%

                position_size = equity * position_size_pct
                qty = int(position_size / current_data["close"])
                position = qty
                entry_price = current_data["close"]
                entry_time = current_date

                # Update trade type counters
                if trade_type == "day":
                    results["metrics"]["day_trades"] += 1
                else:
                    results["metrics"]["swing_trades"] += 1

                results["trades"].append(
                    {
                        "date": current_date.strftime("%Y-%m-%d %H:%M"),
                        "type": "BUY",
                        "price": current_data["close"],
                        "quantity": qty,
                        "value": qty * current_data["close"],
                        "position_size_pct": position_size_pct * 100,
                        "trade_type": trade_type,
                    }
                )

            elif analysis["recommendation"] == "SELL" and position >= 0:
                if position > 0:
                    # Close long position
                    pnl = (current_data["close"] - entry_price) * position
                    equity += pnl
                    trade_type = results["trades"][-1]["trade_type"] if results["trades"] else "day"
                    results["trades"].append(
                        {
                            "date": current_date.strftime("%Y-%m-%d %H:%M"),
                            "type": "SELL",
                            "price": current_data["close"],
                            "quantity": position,
                            "value": position * current_data["close"],
                            "pnl": pnl,
                            "reason": "signal",
                            "trade_type": trade_type,
                        }
                    )
                    position = 0
                    entry_time = None

        # Close any remaining position at the end of the day
        if position != 0:
            pnl = (day_data.iloc[-1]["close"] - entry_price) * position
            equity += pnl
            trade_type = results["trades"][-1]["trade_type"] if results["trades"] else "day"
            results["trades"].append(
                {
                    "date": day_data.index[-1].strftime("%Y-%m-%d %H:%M"),
                    "type": "SELL",
                    "price": day_data.iloc[-1]["close"],
                    "quantity": position,
                    "value": position * day_data.iloc[-1]["close"],
                    "pnl": pnl,
                    "reason": "end_of_day",
                    "trade_type": trade_type,
                }
            )

        # Calculate performance metrics
        results["metrics"]["total_trades"] = len(results["trades"])
        results["metrics"]["winning_trades"] = sum(1 for trade in results["trades"] if trade.get("pnl", 0) > 0)
        results["metrics"]["losing_trades"] = sum(1 for trade in results["trades"] if trade.get("pnl", 0) < 0)
        results["metrics"]["win_rate"] = float(
            results["metrics"]["winning_trades"] / results["metrics"]["total_trades"]
            if results["metrics"]["total_trades"] > 0
            else 0
        )
        results["metrics"]["total_pnl"] = float(sum(trade.get("pnl", 0) for trade in results["trades"]))
        results["metrics"]["final_equity"] = float(equity)
        results["metrics"]["total_return"] = float((equity - initial_capital) / initial_capital * 100)

        # Calculate average win and loss
        winning_trades = [float(trade["pnl"]) for trade in results["trades"] if trade.get("pnl", 0) > 0]
        losing_trades = [float(trade["pnl"]) for trade in results["trades"] if trade.get("pnl", 0) < 0]
        results["metrics"]["avg_win"] = float(sum(winning_trades) / len(winning_trades) if winning_trades else 0)
        results["metrics"]["avg_loss"] = float(sum(losing_trades) / len(losing_trades) if losing_trades else 0)

        # Calculate profit factor
        total_profit = float(sum(winning_trades) if winning_trades else 0)
        total_loss = float(abs(sum(losing_trades)) if losing_trades else 0)
        results["metrics"]["profit_factor"] = float(total_profit / total_loss if total_loss > 0 else float("inf"))

        # Calculate max drawdown
        results["metrics"]["max_drawdown"] = float(calculate_max_drawdown(results["equity_curve"]))

        return results

    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        return None


def calculate_max_drawdown(equity_curve: Union[pd.Series, List[Dict[str, Any]]]) -> float:
    """Calculate the maximum drawdown from an equity curve.

    Computes the largest peak-to-trough decline in the equity curve,
    expressed as a percentage.

    Args:
        equity_curve: Series or list of dictionaries containing equity values
            over time.

    Returns:
        float: Maximum drawdown as a percentage (e.g., 0.15 for 15% drawdown).
    """
    # Convert list of dicts to Series if needed
    if isinstance(equity_curve, list):
        if not equity_curve:
            return 0.0
        equity_values = pd.Series([point["equity"] for point in equity_curve])
    else:
        equity_values = equity_curve

    if equity_values.empty:
        return 0.0

    # Calculate running maximum
    running_max = equity_values.expanding().max()

    # Calculate drawdowns
    drawdowns = (equity_values - running_max) / running_max * 100

    # Get maximum drawdown
    max_drawdown = abs(drawdowns.min())

    return max_drawdown


def main() -> None:
    """Main function to run backtests.

    Executes backtests for configured symbols and timeframes, saving results
    and sending notifications via Discord.
    """
    # Set date range for backtest
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30)  # Last 30 days

    # Run backtest for each symbol
    for symbol in TRADING_STRATEGY["symbols"]:
        results = run_backtest(symbol, "1h", start_date, end_date)
        if results:
            # Save results to file
            filename = f"backtest_results_{symbol}_{end_date.strftime('%Y%m%d')}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Backtest results saved to {filename}")


if __name__ == "__main__":
    main()
