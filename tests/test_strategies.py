import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import strategies


class TestStrategies(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create sample price data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
        self.df = pd.DataFrame(
            {
                "open": np.random.normal(100, 1, 100),
                "high": np.random.normal(101, 1, 100),
                "low": np.random.normal(99, 1, 100),
                "close": np.random.normal(100, 1, 100),
                "volume": np.random.normal(1000000, 100000, 100),
            },
            index=dates,
        )

        # Ensure high is highest and low is lowest
        self.df["high"] = self.df[["open", "high", "close"]].max(axis=1)
        self.df["low"] = self.df[["open", "low", "close"]].min(axis=1)

    def test_calculate_price_indicators(self):
        """Test price indicator calculations"""
        df = strategies.calculate_price_indicators(self.df)

        # Check if returns and log_returns are calculated
        self.assertIn("returns", df.columns)
        self.assertIn("log_returns", df.columns)

        # Check if returns are calculated correctly
        expected_returns = self.df["close"].pct_change()
        pd.testing.assert_series_equal(
            df["returns"].reset_index(drop=True), expected_returns.reset_index(drop=True), check_names=False
        )

    def test_calculate_moving_averages(self):
        """Test moving average calculations"""
        df = strategies.calculate_moving_averages(self.df)

        # Check if all moving averages are calculated
        for window in [5, 10, 20, 50, 200]:
            self.assertIn(f"SMA_{window}", df.columns)
            self.assertIn(f"EMA_{window}", df.columns)

            # Check if SMA is calculated correctly
            expected_sma = self.df["close"].rolling(window=window).mean()
            pd.testing.assert_series_equal(
                df[f"SMA_{window}"].reset_index(drop=True), expected_sma.reset_index(drop=True), check_names=False
            )

    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculations"""
        df = strategies.calculate_bollinger_bands(self.df)

        # Check if all Bollinger Bands are calculated
        self.assertIn("BB_middle", df.columns)
        self.assertIn("BB_upper", df.columns)
        self.assertIn("BB_lower", df.columns)
        self.assertIn("BB_std", df.columns)

        # Check if middle band is calculated correctly
        expected_middle = self.df["close"].rolling(window=20).mean()
        pd.testing.assert_series_equal(
            df["BB_middle"].reset_index(drop=True), expected_middle.reset_index(drop=True), check_names=False
        )

        # Check if upper and lower bands are calculated correctly
        expected_std = self.df["close"].rolling(window=20).std()
        expected_upper = expected_middle + (expected_std * 2)
        expected_lower = expected_middle - (expected_std * 2)
        pd.testing.assert_series_equal(
            df["BB_upper"].reset_index(drop=True), expected_upper.reset_index(drop=True), check_names=False
        )
        pd.testing.assert_series_equal(
            df["BB_lower"].reset_index(drop=True), expected_lower.reset_index(drop=True), check_names=False
        )

    def test_calculate_rsi(self):
        """Test RSI calculations"""
        df = strategies.calculate_rsi(self.df)

        # Check if RSI is calculated
        self.assertIn("RSI", df.columns)

        # Check if RSI values are within valid range (0-100)
        rsi_values = df["RSI"].dropna()  # Drop NaN values before checking
        self.assertTrue((rsi_values >= 0).all() and (rsi_values <= 100).all())

    def test_calculate_macd(self):
        """Test MACD calculations"""
        df = strategies.calculate_macd(self.df)

        # Check if MACD components are calculated
        self.assertIn("MACD", df.columns)
        self.assertIn("MACD_signal", df.columns)
        self.assertIn("MACD_hist", df.columns)

        # Check if MACD histogram is calculated correctly
        expected_hist = df["MACD"] - df["MACD_signal"]
        pd.testing.assert_series_equal(
            df["MACD_hist"].reset_index(drop=True), expected_hist.reset_index(drop=True), check_names=False
        )

    def test_calculate_stochastic(self):
        """Test Stochastic Oscillator calculations"""
        df = strategies.calculate_stochastic(self.df)

        # Check if Stochastic components are calculated
        self.assertIn("%K", df.columns)
        self.assertIn("%D", df.columns)

        # Check if Stochastic values are within valid range (0-100)
        k_values = df["%K"].dropna()  # Drop NaN values before checking
        d_values = df["%D"].dropna()  # Drop NaN values before checking
        self.assertTrue((k_values >= 0).all() and (k_values <= 100).all())
        self.assertTrue((d_values >= 0).all() and (d_values <= 100).all())

    def test_calculate_atr(self):
        """Test ATR calculations"""
        df = strategies.calculate_atr(self.df)

        # Check if ATR is calculated
        self.assertIn("ATR", df.columns)

        # Check if ATR values are positive
        atr_values = df["ATR"].dropna()  # Drop NaN values before checking
        self.assertTrue((atr_values >= 0).all())

    def test_calculate_volume_indicators(self):
        """Test volume indicator calculations"""
        df = strategies.calculate_volume_indicators(self.df)

        # Check if volume indicators are calculated
        self.assertIn("volume_ma_20", df.columns)
        self.assertIn("volume_ratio", df.columns)

        # Check if volume MA is calculated correctly
        expected_ma = self.df["volume"].rolling(window=20).mean()
        pd.testing.assert_series_equal(
            df["volume_ma_20"].reset_index(drop=True), expected_ma.reset_index(drop=True), check_names=False
        )

    def test_calculate_momentum_indicators(self):
        """Test momentum indicator calculations"""
        df = strategies.calculate_momentum_indicators(self.df)

        # Check if momentum indicators are calculated
        self.assertIn("momentum", df.columns)
        self.assertIn("rate_of_change", df.columns)

    def test_calculate_volatility_indicators(self):
        """Test volatility indicator calculations"""
        # Calculate returns first
        df = strategies.calculate_price_indicators(self.df)
        df = strategies.calculate_volatility_indicators(df)

        # Check if volatility indicator is calculated
        self.assertIn("volatility", df.columns)

        # Check if volatility values are positive
        volatility_values = df["volatility"].dropna()  # Drop NaN values before checking
        self.assertTrue((volatility_values >= 0).all())

    def test_calculate_indicators(self):
        """Test calculation of all indicators"""
        df = strategies.calculate_indicators(self.df)

        # Check if all indicators are calculated
        expected_columns = [
            "returns",
            "log_returns",
            "SMA_5",
            "SMA_10",
            "SMA_20",
            "SMA_50",
            "SMA_200",
            "EMA_5",
            "EMA_10",
            "EMA_20",
            "EMA_50",
            "EMA_200",
            "BB_middle",
            "BB_upper",
            "BB_lower",
            "BB_std",
            "RSI",
            "MACD",
            "MACD_signal",
            "MACD_hist",
            "%K",
            "%D",
            "ATR",
            "volume_ma_20",
            "volume_ratio",
            "momentum",
            "rate_of_change",
            "volatility",
        ]

        for col in expected_columns:
            self.assertIn(col, df.columns)

    def test_rsi_strategy(self):
        """Test RSI strategy implementation"""
        # Calculate all necessary indicators first
        df = strategies.calculate_indicators(self.df)
        df = strategies.rsi_strategy(df)

        # Check if strategy components are calculated
        self.assertIn("signal", df.columns)
        self.assertIn("position", df.columns)
        self.assertIn("strategy_returns", df.columns)
        self.assertIn("strategy_cumulative_returns", df.columns)
        self.assertIn("portfolio_value", df.columns)

        # Check if signals are valid (-1, 0, 1)
        self.assertTrue(df["signal"].isin([-1, 0, 1]).all())

        # Check if positions match signals
        pd.testing.assert_series_equal(
            df["position"].reset_index(drop=True),
            df["signal"].shift(1).fillna(0).reset_index(drop=True),
            check_names=False,
        )

    def test_format_for_llm(self):
        """Test LLM data formatting"""
        # Calculate indicators first
        df = strategies.calculate_indicators(self.df)

        # Test formatting
        llm_data = strategies.format_for_llm(df)

        # Check if data is formatted correctly
        self.assertIsInstance(llm_data, list)
        self.assertGreater(len(llm_data), 0)

        # Check structure of formatted data
        data_point = llm_data[0]
        self.assertIn("timestamp", data_point)
        self.assertIn("price_data", data_point)
        self.assertIn("trend_indicators", data_point)
        self.assertIn("momentum_indicators", data_point)
        self.assertIn("volatility_indicators", data_point)
        self.assertIn("volume_indicators", data_point)
        self.assertIn("market_context", data_point)
        self.assertIn("historical_context", data_point)


if __name__ == "__main__":
    unittest.main()
