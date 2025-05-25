import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

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
    def find_nearest_level(price: float, levels: List[float]) -> float:
        """Find the nearest support/resistance level to the current price.

        Args:
            price: Current price
            levels: List of support/resistance levels

        Returns:
            Nearest level to the current price
        """
        return min(levels, key=lambda x: abs(x - price))

    @staticmethod
    def get_support_resistance_levels(df: pd.DataFrame, current_price: float) -> Tuple[float, float]:
        """Get the nearest support and resistance levels for the current price.

        Args:
            df: DataFrame with OHLC data
            current_price: Current price of the asset

        Returns:
            Tuple of (support_level, resistance_level)
        """
        # Calculate pivot points
        pivot, r1, r2, r3, s1, s2, s3 = TechnicalAnalysis.calculate_pivot_points(df)

        # Find nearest support and resistance levels
        all_levels = [s3, s2, s1, pivot, r1, r2, r3]
        nearest_level = TechnicalAnalysis.find_nearest_level(current_price, all_levels)

        # Determine if nearest level is support or resistance
        if nearest_level < current_price:
            support_level = nearest_level
            resistance_level = min([r for r in [r1, r2, r3] if r > current_price], default=r1)
        else:
            resistance_level = nearest_level
            support_level = max([s for s in [s1, s2, s3] if s < current_price], default=s1)

        return support_level, resistance_level

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
    def get_current_market_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Get current market data including price, volume, and indicators.

        Args:
            df: DataFrame with OHLC and indicator data

        Returns:
            Dictionary containing current market data
        """
        i = len(df) - 1
        current_price = df["close"].iloc[i]

        # Calculate support and resistance levels
        support_level, resistance_level = TechnicalAnalysis.get_support_resistance_levels(df, current_price)

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
                "support_level": float(support_level),
                "resistance_level": float(resistance_level),
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


class PriceIndicators:
    """Class for price-based technical indicators."""

    @staticmethod
    def calculate_price_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic price-based indicators.

        Args:
            df: DataFrame containing price data with 'close' column.

        Returns:
            DataFrame with added 'returns' and 'log_returns' columns.
        """
        df = df.copy()
        df["returns"] = df["close"].pct_change(periods=1)
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["returns"] = df["returns"].fillna(0)
        df["log_returns"] = df["log_returns"].fillna(0)
        return df

    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages.

        Args:
            df: DataFrame containing price data with 'close' column.

        Returns:
            DataFrame with added SMA and EMA columns.
        """
        df = df.copy()
        for window in [5, 10, 20, 50, 200]:
            df[f"SMA_{window}"] = df["close"].rolling(window=window).mean()
            df[f"EMA_{window}"] = df["close"].ewm(span=window, adjust=False).mean()
        return df

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands.

        Args:
            df: DataFrame containing price data with 'close' column.
            window: Number of periods for the moving average (default: 20).
            num_std: Number of standard deviations for the bands (default: 2).

        Returns:
            DataFrame with added Bollinger Bands columns.
        """
        df = df.copy()
        df["BB_middle"] = df["close"].rolling(window=window).mean()
        df["BB_std"] = df["close"].rolling(window=window).std()
        df["BB_upper"] = df["BB_middle"] + (df["BB_std"] * num_std)
        df["BB_lower"] = df["BB_middle"] - (df["BB_std"] * num_std)
        return df


class MomentumIndicators:
    """Class for momentum-based technical indicators."""

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index (RSI).

        Args:
            df: DataFrame containing price data with 'close' column.
            window: Number of periods for RSI calculation (default: 14).

        Returns:
            DataFrame with added 'RSI' column.
        """
        df = df.copy()
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD and related indicators.

        Args:
            df: DataFrame containing price data with 'close' column.
            fast: Period for fast EMA (default: 12).
            slow: Period for slow EMA (default: 26).
            signal: Period for signal line smoothing (default: 9).

        Returns:
            DataFrame with added MACD columns.
        """
        df = df.copy()
        exp1 = df["close"].ewm(span=fast, adjust=False).mean()
        exp2 = df["close"].ewm(span=slow, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
        return df

    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator.

        Args:
            df: DataFrame containing price data with 'high', 'low', and 'close' columns.
            k_window: Period for %K calculation (default: 14).
            d_window: Period for %D smoothing (default: 3).

        Returns:
            DataFrame with added '%K' and '%D' columns.
        """
        df = df.copy()
        low_14 = df["low"].rolling(window=k_window).min()
        high_14 = df["high"].rolling(window=k_window).max()
        df["%K"] = 100 * ((df["close"] - low_14) / (high_14 - low_14))
        df["%D"] = df["%K"].rolling(window=d_window).mean()
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
        """Calculate volume-based indicators.

        Args:
            df: DataFrame containing volume data with 'volume' column.
            window: Period for volume moving average (default: 20).

        Returns:
            DataFrame with added volume indicator columns.
        """
        df = df.copy()
        df["volume_ma_20"] = df["volume"].rolling(window=window).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma_20"]
        return df


class VolatilityIndicators:
    """Class for volatility-based technical indicators."""

    @staticmethod
    def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculate Average True Range (ATR).

        Args:
            df: DataFrame containing price data with 'high', 'low', and 'close' columns.
            window: Number of periods for ATR calculation (default: 14).

        Returns:
            DataFrame with added 'ATR' column.
        """
        df = df.copy()
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df["ATR"] = true_range.rolling(window).mean()
        return df

    @staticmethod
    def calculate_volatility_indicators(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate volatility indicators.

        Args:
            df: DataFrame containing returns data with 'returns' column.
            window: Period for volatility calculation (default: 20).

        Returns:
            DataFrame with added 'volatility' column.
        """
        df = df.copy()
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
    df = VolatilityIndicators.calculate_atr(df)
    df = VolumeIndicators.calculate_volume_indicators(df)
    df = MomentumIndicators.calculate_momentum_indicators(df)
    df = VolatilityIndicators.calculate_volatility_indicators(df)

    # Clean up NaN values
    df = df.bfill()

    return df
