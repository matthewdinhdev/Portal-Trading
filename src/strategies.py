import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union, Any

# Get logger
logger = logging.getLogger(__name__)


def calculate_price_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic price-based indicators"""
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    return df


def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate various moving averages"""
    df = df.copy()
    for window in [5, 10, 20, 50, 200]:
        df[f"SMA_{window}"] = df["close"].rolling(window=window).mean()
        df[f"EMA_{window}"] = df["close"].ewm(span=window, adjust=False).mean()
    return df


def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
    """Calculate Bollinger Bands"""
    df = df.copy()
    df["BB_middle"] = df["close"].rolling(window=window).mean()
    df["BB_std"] = df["close"].rolling(window=window).std()
    df["BB_upper"] = df["BB_middle"] + (df["BB_std"] * num_std)
    df["BB_lower"] = df["BB_middle"] - (df["BB_std"] * num_std)
    return df


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate Relative Strength Index"""
    df = df.copy()
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate MACD and related indicators"""
    df = df.copy()
    exp1 = df["close"].ewm(span=fast, adjust=False).mean()
    exp2 = df["close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df


def calculate_stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """Calculate Stochastic Oscillator"""
    df = df.copy()
    low_14 = df["low"].rolling(window=k_window).min()
    high_14 = df["high"].rolling(window=k_window).max()
    df["%K"] = 100 * ((df["close"] - low_14) / (high_14 - low_14))
    df["%D"] = df["%K"].rolling(window=d_window).mean()
    return df


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate Average True Range"""
    df = df.copy()
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["ATR"] = true_range.rolling(window).mean()
    return df


def calculate_volume_indicators(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculate volume-based indicators"""
    df = df.copy()
    df["volume_ma_20"] = df["volume"].rolling(window=window).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"]
    return df


def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum indicators"""
    df = df.copy()
    df["momentum"] = df["close"] - df["close"].shift(10)
    df["rate_of_change"] = df["close"].pct_change(periods=10) * 100
    return df


def calculate_volatility_indicators(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculate volatility indicators"""
    df = df.copy()
    df["volatility"] = df["returns"].rolling(window=window).std() * np.sqrt(252)
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators"""
    df = df.copy()

    # Apply all indicator calculations
    df = calculate_price_indicators(df)
    df = calculate_moving_averages(df)
    df = calculate_bollinger_bands(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_stochastic(df)
    df = calculate_atr(df)
    df = calculate_volume_indicators(df)
    df = calculate_momentum_indicators(df)
    df = calculate_volatility_indicators(df)

    # Clean up NaN values using bfill() instead of fillna(method='bfill')
    df = df.bfill()

    return df


def rsi_strategy(df: pd.DataFrame, initial_capital: float = 100000) -> pd.DataFrame:
    """Simple RSI-based strategy"""
    df = df.copy()

    # Generate signals based on RSI
    df["signal"] = 0
    df.loc[df["RSI"] < 30, "signal"] = 1  # Oversold - Buy signal
    df.loc[df["RSI"] > 70, "signal"] = -1  # Overbought - Sell signal

    # Initialize portfolio metrics
    df["position"] = df["signal"].shift(1)
    df["position"] = df["position"].fillna(0)

    # Calculate returns
    df["strategy_returns"] = df["position"] * df["returns"]
    df["strategy_cumulative_returns"] = (1 + df["strategy_returns"]).cumprod()
    df["portfolio_value"] = initial_capital * df["strategy_cumulative_returns"]

    return df


def format_for_llm(
    df: pd.DataFrame, lookback_periods: int = 48, analysis_date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Format the data for LLM consumption.
    Returns a list of dictionaries, where each dictionary represents a time period
    with its indicators and market context.
    """
    # If analysis_date is provided, filter data to that date
    if analysis_date is not None:
        if isinstance(analysis_date, datetime):
            analysis_date = analysis_date.date()
        df = df[df.index.date == analysis_date]
        if df.empty:
            raise ValueError(f"No data available for analysis date: {analysis_date}")

    # Adjust lookback periods if we don't have enough data
    lookback_periods = min(lookback_periods, len(df) - 1)
    if lookback_periods < 5:
        raise ValueError(f"Insufficient data points. Need at least 5 periods, got {len(df)}")

    # Create a list to store our formatted data
    llm_data: List[Dict[str, Any]] = []

    # Get the last data point for analysis
    try:
        i = len(df) - 1  # Use the last data point
        current_price = df["close"].iloc[i]
        price_change_1h = float(df["returns"].iloc[i])

        # Calculate multi-period changes safely
        price_change_5h = 0.0
        price_change_20h = 0.0
        price_change_48h = 0.0
        if i >= 5:
            price_change_5h = float(df["close"].iloc[i] / df["close"].iloc[i - 5] - 1)
        if i >= 20:
            price_change_20h = float(df["close"].iloc[i] / df["close"].iloc[i - 20] - 1)
        if i >= 48:
            price_change_48h = float(df["close"].iloc[i] / df["close"].iloc[i - 48] - 1)

        current_data = {
            "timestamp": df.index[i].strftime("%Y-%m-%d %H:%M"),
            "price_data": {
                "open": float(df["open"].iloc[i]),
                "high": float(df["high"].iloc[i]),
                "low": float(df["low"].iloc[i]),
                "close": float(current_price),
                "volume": float(df["volume"].iloc[i]),
                "price_changes": {
                    "1h": price_change_1h,
                    "5h": price_change_5h,
                    "20h": price_change_20h,
                    "48h": price_change_48h,
                },
            },
            "trend_indicators": {
                "sma_20": float(df["SMA_20"].iloc[i]),
                "sma_50": float(df["SMA_50"].iloc[i]),
                "sma_200": float(df["SMA_200"].iloc[i]),
                "ema_20": float(df["EMA_20"].iloc[i]),
                "ema_50": float(df["EMA_50"].iloc[i]),
            },
            "momentum_indicators": {
                "rsi": float(df["RSI"].iloc[i]),
                "macd": float(df["MACD"].iloc[i]),
                "macd_signal": float(df["MACD_signal"].iloc[i]),
                "macd_hist": float(df["MACD_hist"].iloc[i]),
                "stochastic_k": float(df["%K"].iloc[i]),
                "stochastic_d": float(df["%D"].iloc[i]),
            },
            "volatility_indicators": {
                "bollinger_upper": float(df["BB_upper"].iloc[i]),
                "bollinger_middle": float(df["BB_middle"].iloc[i]),
                "bollinger_lower": float(df["BB_lower"].iloc[i]),
                "atr": float(df["ATR"].iloc[i]),
                "volatility": float(df["volatility"].iloc[i]),
            },
            "volume_indicators": {
                "volume_ma_20": float(df["volume_ma_20"].iloc[i]),
                "volume_ratio": float(df["volume_ratio"].iloc[i]),
            },
            "market_context": {
                "price_change_1h": price_change_1h,
                "price_change_5h": price_change_5h,
                "price_change_20h": price_change_20h,
                "price_change_48h": price_change_48h,
                "momentum": float(df["momentum"].iloc[i]),
                "rate_of_change": float(df["rate_of_change"].iloc[i]),
            },
            "historical_context": {
                "price_trend": "up" if df["close"].iloc[i] > df["SMA_20"].iloc[i] else "down",
                "volume_trend": (
                    "high"
                    if df["volume_ratio"].iloc[i] > 1.5
                    else "low" if df["volume_ratio"].iloc[i] < 0.5 else "normal"
                ),
                "volatility_state": (
                    "high" if df["volatility"].iloc[i] > df["volatility"].rolling(20).mean().iloc[i] else "low"
                ),
                "rsi_state": (
                    "overbought" if df["RSI"].iloc[i] > 70 else "oversold" if df["RSI"].iloc[i] < 30 else "neutral"
                ),
                "trend_strength": (
                    "strong" if abs(price_change_48h) > 0.1 else "moderate" if abs(price_change_48h) > 0.05 else "weak"
                ),
            },
        }

        # Add previous periods' data for context
        current_data["previous_periods"] = []
        for j in range(1, min(lookback_periods + 1, i + 1)):
            prev_data = {
                "timestamp": df.index[i - j].strftime("%Y-%m-%d %H:%M"),
                "close": float(df["close"].iloc[i - j]),
                "volume": float(df["volume"].iloc[i - j]),
                "rsi": float(df["RSI"].iloc[i - j]),
                "macd": float(df["MACD"].iloc[i - j]),
            }
            current_data["previous_periods"].append(prev_data)

        llm_data.append(current_data)

    except Exception as e:
        print(f"Error processing data point: {str(e)}")
        raise ValueError(f"Failed to process data: {str(e)}")

    if not llm_data:
        raise ValueError("No valid data points could be processed")

    return llm_data


def get_llm_prompt(
    df: pd.DataFrame,
    account_info: Optional[Dict[str, Any]] = None,
    positions: Optional[List[Dict[str, Any]]] = None,
    lookback_periods: int = 48,
    analysis_date: Optional[datetime] = None,
) -> Optional[str]:
    """
    Generate a prompt for the LLM based on the current market conditions.
    Uses 48 periods of historical data by default.
    """
    try:
        if df.empty:
            raise ValueError("No data available for analysis")

        # Ensure we have enough data points
        if len(df) < 5:
            raise ValueError(f"Insufficient data points. Need at least 5 periods, got {len(df)}")

        # Get formatted data
        llm_data = format_for_llm(df, lookback_periods, analysis_date)
        if not llm_data:
            raise ValueError("No valid data points could be processed")

        # Get the most recent data point
        current = llm_data[-1]

        # Calculate support and resistance levels safely
        recent_lows = [period["close"] for period in current["previous_periods"][:24] if "close" in period]
        recent_highs = [period["close"] for period in current["previous_periods"][:24] if "close" in period]

        if not recent_lows or not recent_highs:
            raise ValueError("Insufficient data for support/resistance calculation")

        support_level = min(recent_lows)
        resistance_level = max(recent_highs)

        # Calculate ATR-based price levels
        atr = current["volatility_indicators"]["atr"]
        current_price = current["price_data"]["close"]
        stop_loss_level = current_price - (2 * atr)  # 2 ATR below current price
        take_profit_level = current_price + (4 * atr)

        # Format the prompt with clear sections and better readability
        prompt = [
            f"Trading Analysis ({analysis_date.strftime('%Y-%m-%d') if analysis_date else datetime.now().strftime('%Y-%m-%d')})",
            "=" * 40,
            "",
            "Based on the following market data and account information, provide a trading signal (BUY, SELL, or HOLD) and explain your reasoning:",
            "",
            "Account Information:",
        ]

        # Add account information if available
        if account_info:
            prompt.extend(
                [
                    f"- Portfolio Value: ${account_info['portfolio_value']:,.2f}",
                    f"- Available Cash: ${account_info['cash']:,.2f}",
                    f"- Buying Power: ${account_info['buying_power']:,.2f}",
                    f"- Day Trade Count: {account_info['daytrade_count']}",
                    f"- Pattern Day Trader: {'Yes' if account_info['pattern_day_trader'] else 'No'}",
                    f"- Shorting Enabled: {'Yes' if account_info['shorting_enabled'] else 'No'}",
                    f"- Account Status: {account_info['status']}",
                ]
            )
        else:
            prompt.append("- Account information not available")

        # Add current positions if available
        if positions:
            prompt.extend(["", "Current Positions:"])
            for position in positions:
                prompt.extend(
                    [
                        f"- {position['symbol']}:",
                        f"  Quantity: {position['qty']}",
                        f"  Average Entry: ${position['avg_entry_price']:.2f}",
                        f"  Current Price: ${position['current_price']:.2f}",
                        f"  Market Value: ${position['market_value']:,.2f}",
                        f"  Unrealized P/L: ${position['unrealized_pl']:,.2f} ({position['unrealized_plpc']*100:+.2f}%)",
                        f"  Today's Change: {position['change_today']*100:+.2f}%",
                    ]
                )
        else:
            prompt.append("- No current positions")

        # Add market data
        prompt.extend(
            [
                "",
                f"Current Market Conditions ({current['timestamp']}):",
                f"- Price: ${current['price_data']['close']:.2f}",
                f"- Hourly Change: {current['price_data']['price_changes']['1h']*100:+.2f}%",
                f"- 5-Hour Change: {current['price_data']['price_changes']['5h']*100:+.2f}%",
                f"- 20-Hour Change: {current['price_data']['price_changes']['20h']*100:+.2f}%",
                f"- 48-Hour Change: {current['price_data']['price_changes']['48h']*100:+.2f}%",
                "",
                "Price Levels:",
                f"- Support Level: ${support_level:.2f}",
                f"- Resistance Level: ${resistance_level:.2f}",
                f"- ATR-Based Stop Loss: ${stop_loss_level:.2f}",
                f"- ATR-Based Take Profit: ${take_profit_level:.2f}",
                "",
                "Technical Indicators:",
                f"- RSI: {current['momentum_indicators']['rsi']:.2f}",
                f"- MACD: {current['momentum_indicators']['macd']:.2f}",
                f"- MACD Signal: {current['momentum_indicators']['macd_signal']:.2f}",
                f"- MACD Histogram: {current['momentum_indicators']['macd_hist']:.2f}",
                f"- Stochastic K: {current['momentum_indicators']['stochastic_k']:.2f}",
                f"- Stochastic D: {current['momentum_indicators']['stochastic_d']:.2f}",
                f"- Volume Ratio: {current['volume_indicators']['volume_ratio']:.2f}",
                f"- Volatility: {current['volatility_indicators']['volatility']:.2f}",
                "",
                "Market Context:",
                f"- Price Trend: {current['historical_context']['price_trend']}",
                f"- Volume Trend: {current['historical_context']['volume_trend']}",
                f"- Volatility State: {current['historical_context']['volatility_state']}",
                f"- RSI State: {current['historical_context']['rsi_state']}",
                f"- Trend Strength: {current['historical_context']['trend_strength']}",
                "",
                "Recent Price History (Last 48 Hours):",
            ]
        )

        # Add recent price history (last 48 hours)
        for period in current["previous_periods"][:48]:
            prompt.append(
                f"- {period['timestamp']}: ${period['close']:.2f} "
                f"(RSI: {period['rsi']:.2f}, MACD: {period['macd']:.2f})"
            )

        prompt.extend(
            [
                "",
                "Please provide your analysis and trading recommendation, considering both technical indicators and market context.",
                "Include:",
                "1. Overall market sentiment",
                "2. Key technical signals",
                "3. Risk assessment",
                "4. Clear trading recommendation (BUY/SELL/HOLD)",
                "5. Brief explanation of your reasoning",
                "6. Specific price targets based on support/resistance levels and ATR",
                "7. Position sizing recommendation based on account equity and risk tolerance",
                "",
                "For price targets, use the following format:",
                "- stop_loss: Use either the ATR-based stop loss or the nearest support level",
                "- take_profit: Use either the ATR-based take profit or the nearest resistance level",
                "Example: stop_loss: '150.25', take_profit: '165.50'",
            ]
        )

        final_prompt = "\n".join(prompt)
        logger.debug("Generated LLM prompt:\n%s", final_prompt)
        return final_prompt

    except Exception as e:
        logger.error(f"Error generating LLM prompt: {str(e)}")
        return None


def automated_strategy(df: pd.DataFrame, llm_analysis: Dict[str, Any], initial_capital: float = 100000) -> pd.DataFrame:
    """
    Automated trading strategy using LLM analysis for stop losses and take profits.
    """
    df = df.copy()

    # Initialize strategy columns
    df["signal"] = 0
    df["position"] = 0
    df["stop_loss"] = None
    df["take_profit"] = None
    df["entry_price"] = None
    df["exit_price"] = None
    df["exit_reason"] = None

    # Get price targets from LLM analysis
    stop_loss = llm_analysis.get("price_targets", {}).get("stop_loss")
    take_profit = llm_analysis.get("price_targets", {}).get("take_profit")
    position_size = llm_analysis.get("position_size", 0.01)  # Default to 1%
    trade_type = llm_analysis.get("trade_type", "day")

    # Validate price targets
    if not stop_loss or not take_profit:
        logger.warning("Missing price targets in LLM analysis")
        return df

    # Calculate position size in dollars
    position_value = initial_capital * position_size

    # Initialize tracking variables
    current_position = 0
    entry_price = None
    max_hold_time = timedelta(hours=6) if trade_type == "day" else timedelta(days=5)
    entry_time = None

    # Process each bar
    for i in range(len(df)):
        current_price = df["close"].iloc[i]
        current_time = df.index[i]

        # If we have a position, check stop loss and take profit
        if current_position != 0:
            # Check stop loss
            if current_price <= stop_loss:
                df.loc[df.index[i], "signal"] = -1
                df.loc[df.index[i], "exit_price"] = current_price
                df.loc[df.index[i], "exit_reason"] = "stop_loss"
                current_position = 0
                entry_price = None
                entry_time = None
                continue

            # Check take profit
            if current_price >= take_profit:
                df.loc[df.index[i], "signal"] = -1
                df.loc[df.index[i], "exit_price"] = current_price
                df.loc[df.index[i], "exit_reason"] = "take_profit"
                current_position = 0
                entry_price = None
                entry_time = None
                continue

            # Check max hold time
            if entry_time and (current_time - entry_time) > max_hold_time:
                df.loc[df.index[i], "signal"] = -1
                df.loc[df.index[i], "exit_price"] = current_price
                df.loc[df.index[i], "exit_reason"] = "max_hold_time"
                current_position = 0
                entry_price = None
                entry_time = None
                continue

        # If we don't have a position, check for entry
        elif llm_analysis["recommendation"] == "BUY":
            df.loc[df.index[i], "signal"] = 1
            df.loc[df.index[i], "entry_price"] = current_price
            df.loc[df.index[i], "stop_loss"] = stop_loss
            df.loc[df.index[i], "take_profit"] = take_profit
            current_position = int(position_value / current_price)
            entry_price = current_price
            entry_time = current_time

        # Update position
        df.loc[df.index[i], "position"] = current_position

    # Calculate strategy metrics
    df["strategy_returns"] = 0.0
    df["strategy_cumulative_returns"] = 1.0
    df["portfolio_value"] = initial_capital

    # Calculate returns for each trade
    entry_prices = df[df["entry_price"].notna()]["entry_price"]
    exit_prices = df[df["exit_price"].notna()]["exit_price"]

    for i in range(len(entry_prices)):
        if i < len(exit_prices):
            entry_price = entry_prices.iloc[i]
            exit_price = exit_prices.iloc[i]
            trade_return = (exit_price - entry_price) / entry_price

            # Find the exit index
            exit_idx = df[df["exit_price"] == exit_price].index[0]
            df.loc[exit_idx, "strategy_returns"] = trade_return

            # Update cumulative returns and portfolio value
            if i == 0:
                df.loc[exit_idx:, "strategy_cumulative_returns"] *= 1 + trade_return
            else:
                prev_cum_returns = df.loc[:exit_idx, "strategy_cumulative_returns"].iloc[-2]
                df.loc[exit_idx:, "strategy_cumulative_returns"] = prev_cum_returns * (1 + trade_return)

    # Calculate portfolio value
    df["portfolio_value"] = initial_capital * df["strategy_cumulative_returns"]

    return df
