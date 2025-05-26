import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import talib

# Get logger
logger = logging.getLogger(__name__)


class TechnicalAnalysis:
    """Class for technical analysis calculations."""

    @staticmethod
    def calculate_pivot_points(df: pd.DataFrame) -> Tuple[float, float, float, float, float, float, float]:
        """Calculate pivot points and support/resistance levels.

        Args:
            df: DataFrame with OHLC data

        Returns:
            Tuple of (Pivot, R1, R2, R3, S1, S2, S3)
        """
        # Get previous day's data
        prev_high = df["high"].iloc[-2]
        prev_low = df["low"].iloc[-2]
        prev_close = df["close"].iloc[-2]

        # Calculate pivot point
        pivot = (prev_high + prev_low + prev_close) / 3

        # Calculate support and resistance levels
        r1 = (2 * pivot) - prev_low
        s1 = (2 * pivot) - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)

        return pivot, r1, r2, r3, s1, s2, s3

    @staticmethod
    def find_historical_levels(df: pd.DataFrame) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        """Find historical support and resistance levels using price action.

        Args:
            df: DataFrame with OHLC data

        Returns:
            Tuple of (support_levels_dict, resistance_levels_dict) where each dict has keys:
            - short_term (30 days)
            - medium_term (90 days)
            - long_term (180 days)
        """
        support_levels = {"short_term": [], "medium_term": [], "long_term": []}
        resistance_levels = {"short_term": [], "medium_term": [], "long_term": []}
        windows = {"short_term": 30, "medium_term": 90, "long_term": 180}

        for timeframe, window in windows.items():
            # Get the window of data
            recent_data = df.tail(window)

            for i in range(1, len(recent_data) - 1):
                # Check for local minima (potential support)
                if (
                    recent_data["low"].iloc[i] < recent_data["low"].iloc[i - 1]
                    and recent_data["low"].iloc[i] < recent_data["low"].iloc[i + 1]
                ):
                    support_levels[timeframe].append(recent_data["low"].iloc[i])

                # Check for local maxima (potential resistance)
                if (
                    recent_data["high"].iloc[i] > recent_data["high"].iloc[i - 1]
                    and recent_data["high"].iloc[i] > recent_data["high"].iloc[i + 1]
                ):
                    resistance_levels[timeframe].append(recent_data["high"].iloc[i])

            # Cluster nearby levels to reduce noise
            support_levels[timeframe] = TechnicalAnalysis._cluster_levels(support_levels[timeframe], threshold=0.01)
            resistance_levels[timeframe] = TechnicalAnalysis._cluster_levels(
                resistance_levels[timeframe], threshold=0.01
            )

        return support_levels, resistance_levels

    @staticmethod
    def _cluster_levels(levels: List[float], threshold: float = 0.02) -> List[float]:
        """Cluster nearby price levels to reduce noise.

        Args:
            levels: List of price levels
            threshold: Percentage threshold for clustering (default 2%)

        Returns:
            List of clustered price levels
        """
        if not levels:
            return []

        # Sort levels
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]

        for level in levels[1:]:
            # If level is within threshold of current cluster, add to cluster
            if abs(level - current_cluster[0]) / current_cluster[0] <= threshold:
                current_cluster.append(level)
            else:
                # Add average of current cluster to result
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]

        # Add final cluster
        clustered.append(sum(current_cluster) / len(current_cluster))

        return clustered

    @staticmethod
    def calculate_price_changes(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate multi-period price changes.

        Args:
            df: DataFrame with OHLC data

        Returns:
            Dictionary of price changes for different periods
        """
        i = len(df) - 1
        price_changes = {"1w": 0.0, "2w": 0.0, "1m": 0.0, "6m": 0.0}

        if i >= 5:
            price_changes["1w"] = float(df["close"].iloc[i] / df["close"].iloc[i - 5] - 1)
        if i >= 10:
            price_changes["2w"] = float(df["close"].iloc[i] / df["close"].iloc[i - 10] - 1)
        if i >= 21:
            price_changes["1m"] = float(df["close"].iloc[i] / df["close"].iloc[i - 21] - 1)
        if i >= 126:
            price_changes["6m"] = float(df["close"].iloc[i] / df["close"].iloc[i - 126] - 1)

        return price_changes

    @staticmethod
    def calculate_fibonacci_levels(df: pd.DataFrame, window: int = 180) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels from recent price action.

        Args:
            df: DataFrame with OHLC data
            window: Number of periods to look back for high/low (default 20)

        Returns:
            Dictionary of Fibonacci levels (0.0 to 1.0)
        """
        # Get the window of data
        recent_data = df.tail(window)

        # Find the high and low in the period
        high = recent_data["high"].max()
        low = recent_data["low"].min()

        # Calculate the range
        price_range = high - low

        # Calculate Fibonacci levels
        levels = {
            "0.0": low,
            "0.236": low + 0.236 * price_range,
            "0.382": low + 0.382 * price_range,
            "0.5": low + 0.5 * price_range,
            "0.618": low + 0.618 * price_range,
            "0.786": low + 0.786 * price_range,
            "1.0": high,
        }

        return levels

    @staticmethod
    def get_current_market_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Get current market data including price, volume, and indicators.

        Args:
            df: DataFrame with OHLC and indicator data

        Returns:
            Dictionary containing current market data
        """
        i = len(df) - 1
        current_price = df["close"].iloc[i]

        # Get historical levels for all timeframes
        support_levels, resistance_levels = TechnicalAnalysis.find_historical_levels(df)

        # Calculate Fibonacci levels
        fib_levels = TechnicalAnalysis.calculate_fibonacci_levels(df)

        # Calculate all indicators
        df = calculate_all_indicators(df)

        return {
            "timestamp": df.index[i].strftime("%Y-%m-%d %H:%M"),
            "price_data": {
                "open": float(df["open"].iloc[i]),
                "high": float(df["high"].iloc[i]),
                "low": float(df["low"].iloc[i]),
                "close": float(current_price),
                "volume": float(df["volume"].iloc[i]),
                "price_changes": TechnicalAnalysis.calculate_price_changes(df),
                "support_levels": {
                    timeframe: [float(s) for s in levels] for timeframe, levels in support_levels.items()
                },
                "resistance_levels": {
                    timeframe: [float(r) for r in levels] for timeframe, levels in resistance_levels.items()
                },
                "fibonacci_levels": {level: float(price) for level, price in fib_levels.items()},
            },
            "indicators": {
                "bollinger_bands": {
                    "upper": float(df["BB_upper"].iloc[i]),
                    "middle": float(df["BB_middle"].iloc[i]),
                    "lower": float(df["BB_lower"].iloc[i]),
                },
                "moving_averages": {
                    "sma_20": float(df["SMA_20"].iloc[i]),
                    "sma_50": float(df["SMA_50"].iloc[i]),
                    "sma_200": float(df["SMA_200"].iloc[i]),
                    "ema_20": float(df["EMA_20"].iloc[i]),
                    "ema_50": float(df["EMA_50"].iloc[i]),
                    "ema_200": float(df["EMA_200"].iloc[i]),
                },
                "momentum": {
                    "rsi": float(df["RSI"].iloc[i]),
                    "macd": float(df["MACD"].iloc[i]),
                    "macd_signal": float(df["MACD_signal"].iloc[i]),
                    "macd_hist": float(df["MACD_hist"].iloc[i]),
                    "stoch_k": float(df["%K"].iloc[i]),
                    "stoch_d": float(df["%D"].iloc[i]),
                },
                "volatility": {"atr": float(df["ATR"].iloc[i]), "volatility": float(df["volatility"].iloc[i])},
                "volume": {
                    "volume_ma": float(df["volume_ma_20"].iloc[i]),
                    "volume_ratio": float(df["volume_ratio"].iloc[i]),
                },
            },
        }

    @staticmethod
    def calculate_price_targets_from_fib(
        current_price: float, fib_levels: Dict[str, float], recommendation: str, min_risk_reward: float = 1.5
    ) -> Dict[str, float]:
        """Calculate price targets using Fibonacci levels, falling back to ATR if risk/reward is insufficient.

        Args:
            current_price: Current price of the asset
            fib_levels: Dictionary containing Fibonacci levels and ATR
            recommendation: Trading recommendation ("BUY" or "SELL")
            min_risk_reward: Minimum risk/reward ratio required to use Fibonacci levels (default: 1.5)

        Returns:
            Dictionary containing stop_loss and take_profit levels
        """
        # Get ATR for fallback
        atr = fib_levels.get("atr", 0.0)
        if atr <= 0:
            raise ValueError("Invalid ATR value")

        # Convert fib levels to sorted list for easier comparison
        levels = [(float(level), price) for level, price in fib_levels.items() if level != "atr"]
        levels.sort(key=lambda x: x[1])  # Sort by price

        # Try Fibonacci levels first
        if recommendation == "BUY":
            # Find nearest fib level below current price for stop loss
            stop_loss = None
            for _, price in reversed(levels):
                if price < current_price:
                    stop_loss = price
                    break

            # Find nearest fib level above current price for take profit
            take_profit = None
            for _, price in levels:
                if price > current_price:
                    take_profit = price
                    break

        else:  # SELL
            # Find nearest fib level above current price for stop loss
            stop_loss = None
            for _, price in levels:
                if price > current_price:
                    stop_loss = price
                    break

            # Find nearest fib level below current price for take profit
            take_profit = None
            for _, price in reversed(levels):
                if price < current_price:
                    take_profit = price
                    break

        # Check if we have valid Fibonacci targets
        if stop_loss and take_profit:
            # Calculate risk/reward ratio
            if recommendation == "BUY":
                risk = current_price - stop_loss
                reward = take_profit - current_price
            else:  # SELL
                risk = stop_loss - current_price
                reward = current_price - take_profit

            risk_reward_ratio = reward / risk if risk > 0 else 0

            # If risk/reward is good enough, use Fibonacci targets
            if risk_reward_ratio >= min_risk_reward:
                return {"stop_loss": round(stop_loss, 2), "take_profit": round(take_profit, 2)}

        # Fall back to ATR-based targets if Fibonacci levels don't give good risk/reward
        if recommendation == "BUY":
            stop_loss = current_price - (2 * atr)  # 2 ATR for stop loss
            take_profit = current_price + (3 * atr)  # 3 ATR for take profit
        else:  # SELL
            stop_loss = current_price + (2 * atr)  # 2 ATR for stop loss
            take_profit = current_price - (3 * atr)  # 3 ATR for take profit

        return {"stop_loss": round(stop_loss, 2), "take_profit": round(take_profit, 2)}


class PriceIndicators:
    """Class for price-based technical indicators."""

    @staticmethod
    def calculate_price_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based indicators"""
        df = df.copy()

        # Calculate returns using TA-Lib
        df["returns"] = talib.ROCP(df["close"], timeperiod=1)
        df["log_returns"] = np.log(1 + df["returns"])

        return df

    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages"""
        df = df.copy()

        # Calculate SMAs
        for window in [5, 10, 20, 50, 200]:
            df[f"SMA_{window}"] = talib.SMA(df["close"], timeperiod=window)
            df[f"EMA_{window}"] = talib.EMA(df["close"], timeperiod=window)

        return df

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df = df.copy()

        # Calculate Bollinger Bands using TA-Lib
        upper, middle, lower = talib.BBANDS(
            df["close"], timeperiod=window, nbdevup=num_std, nbdevdn=num_std, matype=0  # 0 = SMA
        )

        df["BB_upper"] = upper
        df["BB_middle"] = middle
        df["BB_lower"] = lower
        df["BB_std"] = (upper - lower) / (2 * num_std)

        return df


class MomentumIndicators:
    """Class for momentum-based technical indicators."""

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate RSI"""
        df = df.copy()
        df["RSI"] = talib.RSI(df["close"], timeperiod=window)
        return df

    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD"""
        df = df.copy()

        # Calculate MACD using TA-Lib
        macd, signal_line, hist = talib.MACD(df["close"], fastperiod=fast, slowperiod=slow, signalperiod=signal)

        df["MACD"] = macd
        df["MACD_signal"] = signal_line
        df["MACD_hist"] = hist

        return df

    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        df = df.copy()

        # Calculate Stochastic using TA-Lib
        slowk, slowd = talib.STOCH(
            df["high"],
            df["low"],
            df["close"],
            fastk_period=k_window,
            slowk_period=d_window,
            slowk_matype=0,
            slowd_period=d_window,
            slowd_matype=0,
        )

        df["%K"] = slowk
        df["%D"] = slowd

        return df

    @staticmethod
    def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators.

        Args:
            df: DataFrame containing price data with 'close' column.

        Returns:
            DataFrame with added momentum indicator columns.
        """
        df = df.copy()
        df["momentum"] = df["close"] - df["close"].shift(10)
        df["rate_of_change"] = df["close"].pct_change(periods=10) * 100
        return df


class VolumeIndicators:
    """Class for volume-based technical indicators."""

    @staticmethod
    def calculate_volume_indicators(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        df = df.copy()

        # Calculate volume moving average
        df["volume_ma_20"] = talib.SMA(df["volume"], timeperiod=window)

        # Calculate volume ratio
        df["volume_ratio"] = df["volume"] / df["volume_ma_20"]

        # Add On Balance Volume (OBV)
        df["OBV"] = talib.OBV(df["close"], df["volume"])

        return df


class VolatilityIndicators:
    """Class for volatility-based technical indicators."""

    @staticmethod
    def calculate_volatility_indicators(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate volatility indicators"""
        df = df.copy()

        # Calculate ATR
        df["ATR"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=window)

        # Calculate Normalized ATR
        df["NATR"] = talib.NATR(df["high"], df["low"], df["close"], timeperiod=window)

        # Calculate True Range
        df["TRANGE"] = talib.TRANGE(df["high"], df["low"], df["close"])

        # Calculate annualized volatility from returns
        if "returns" in df.columns:
            df["volatility"] = df["returns"].rolling(window=window).std() * np.sqrt(252)

        return df


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators.

    Args:
        df: DataFrame containing price and volume data.

    Returns:
        DataFrame with all technical indicators added.
    """
    logger.info(" . Calculating technical indicators")
    df = df.copy()

    # Apply all indicator calculations
    df = PriceIndicators.calculate_price_indicators(df)
    df = PriceIndicators.calculate_moving_averages(df)
    df = PriceIndicators.calculate_bollinger_bands(df)
    df = MomentumIndicators.calculate_rsi(df)
    df = MomentumIndicators.calculate_macd(df)
    df = MomentumIndicators.calculate_stochastic(df)
    df = VolatilityIndicators.calculate_volatility_indicators(df)
    df = VolumeIndicators.calculate_volume_indicators(df)
    df = MomentumIndicators.calculate_momentum_indicators(df)

    # Clean up NaN values
    df = df.bfill()

    return df
