import os
import sys
import unittest

import numpy as np
import pandas as pd

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


from technical_analysis import (
    MomentumIndicators,
    PriceIndicators,
    TechnicalAnalysis,
    VolatilityIndicators,
    VolumeIndicators,
    calculate_all_indicators,
)


class TestTechnicalAnalysis(unittest.TestCase):
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
        test_df = pd.DataFrame({"close": [100, 110, 99, 108.9]})
        df = PriceIndicators.calculate_price_indicators(test_df)

        # Verify columns exist
        self.assertIn("returns", df.columns)
        self.assertIn("log_returns", df.columns)

        # Verify data types
        self.assertTrue(pd.api.types.is_numeric_dtype(df["returns"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(df["log_returns"]))

        # Verify basic properties
        self.assertTrue(df["returns"].notna().any())  # Should have some non-NaN values
        self.assertTrue(df["log_returns"].notna().any())

    def test_calculate_moving_averages(self):
        """Test moving average calculations"""
        # Create a longer price series to ensure we have enough data points
        test_df = pd.DataFrame({"close": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]})
        df = PriceIndicators.calculate_moving_averages(test_df)

        # Verify all expected columns exist
        for window in [5, 10, 20, 50, 200]:
            self.assertIn(f"SMA_{window}", df.columns)
            self.assertIn(f"EMA_{window}", df.columns)

        # For shorter windows, verify we have some non-NaN values
        for window in [5, 10]:
            self.assertTrue(df[f"SMA_{window}"].notna().any(), f"SMA_{window} should have some non-NaN values")
            self.assertTrue(df[f"EMA_{window}"].notna().any(), f"EMA_{window} should have some non-NaN values")

        # For longer windows, verify the data type
        for window in [20, 50, 200]:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[f"SMA_{window}"]))
            self.assertTrue(pd.api.types.is_numeric_dtype(df[f"EMA_{window}"]))

        # Verify that the last values are not NaN for the shorter windows
        self.assertFalse(pd.isna(df["SMA_5"].iloc[-1]))
        self.assertFalse(pd.isna(df["EMA_5"].iloc[-1]))

    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculations"""
        test_df = pd.DataFrame({"close": [100, 102, 98, 103, 97, 101, 99, 104, 96, 102]})
        df = PriceIndicators.calculate_bollinger_bands(test_df, window=3, num_std=2)

        # Verify columns exist
        self.assertIn("BB_middle", df.columns)
        self.assertIn("BB_upper", df.columns)
        self.assertIn("BB_lower", df.columns)
        self.assertIn("BB_std", df.columns)

        # Verify basic properties (ignoring NaN values)
        non_nan_mask = df["BB_middle"].notna()
        self.assertTrue((df.loc[non_nan_mask, "BB_upper"] > df.loc[non_nan_mask, "BB_middle"]).all())
        self.assertTrue((df.loc[non_nan_mask, "BB_middle"] > df.loc[non_nan_mask, "BB_lower"]).all())
        self.assertTrue((df.loc[non_nan_mask, "BB_std"] > 0).all())

        # Verify symmetry (ignoring NaN values)
        upper_diff = df.loc[non_nan_mask, "BB_upper"] - df.loc[non_nan_mask, "BB_middle"]
        lower_diff = df.loc[non_nan_mask, "BB_middle"] - df.loc[non_nan_mask, "BB_lower"]
        pd.testing.assert_series_equal(upper_diff, lower_diff, check_names=False)

    def test_calculate_rsi(self):
        """Test RSI calculations"""
        # Create a longer price series to ensure we have enough data points
        test_df = pd.DataFrame({"close": [100, 110, 105, 115, 110, 120, 115, 125, 130, 135, 140, 145, 150, 155, 160]})
        df = MomentumIndicators.calculate_rsi(test_df, window=2)

        # Verify column exists
        self.assertIn("RSI", df.columns)

        # Verify basic properties
        self.assertTrue(pd.api.types.is_numeric_dtype(df["RSI"]))

        # Only check RSI properties on non-NaN values
        non_nan_mask = df["RSI"].notna()
        self.assertTrue((df.loc[non_nan_mask, "RSI"] >= 0).all())
        self.assertTrue((df.loc[non_nan_mask, "RSI"] <= 100).all())

        # Verify we have some non-NaN values
        self.assertTrue(df["RSI"].notna().any())

        # Verify the last value is not NaN
        self.assertFalse(pd.isna(df["RSI"].iloc[-1]))

    def test_calculate_macd(self):
        """Test MACD calculations"""
        test_df = pd.DataFrame({"close": [100, 110, 120, 130, 140, 150, 160, 170]})
        df = MomentumIndicators.calculate_macd(test_df, fast=2, slow=3, signal=2)

        # Verify columns exist
        self.assertIn("MACD", df.columns)
        self.assertIn("MACD_signal", df.columns)
        self.assertIn("MACD_hist", df.columns)

        # Verify basic properties
        for col in ["MACD", "MACD_signal", "MACD_hist"]:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]))
            self.assertTrue(df[col].notna().any())

        # Verify histogram calculation
        pd.testing.assert_series_equal(
            df["MACD_hist"].dropna(), (df["MACD"] - df["MACD_signal"]).dropna(), check_names=False
        )

    def test_calculate_stochastic(self):
        """Test Stochastic Oscillator calculations"""
        # Create a longer price series to ensure we have enough data points
        test_df = pd.DataFrame(
            {
                "high": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240],
                "low": [90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230],
                "close": [95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205, 215, 225, 235],
            }
        )
        df = MomentumIndicators.calculate_stochastic(test_df, k_window=2, d_window=2)

        # Verify columns exist
        self.assertIn("%K", df.columns)
        self.assertIn("%D", df.columns)

        # Verify basic properties
        for col in ["%K", "%D"]:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]))

            # Only check range on non-NaN values
            non_nan_mask = df[col].notna()
            self.assertTrue((df.loc[non_nan_mask, col] >= 0).all())
            self.assertTrue((df.loc[non_nan_mask, col] <= 100).all())

        # Verify we have some non-NaN values
        self.assertTrue(df["%K"].notna().any())
        self.assertTrue(df["%D"].notna().any())

        # Verify the last values are not NaN
        self.assertFalse(pd.isna(df["%K"].iloc[-1]))
        self.assertFalse(pd.isna(df["%D"].iloc[-1]))

    def test_calculate_volume_indicators(self):
        """Test volume indicator calculations"""
        test_df = pd.DataFrame(
            {
                "close": [100, 102, 101, 103, 102],  # Required for OBV calculation
                "volume": [1000, 2000, 3000, 4000, 5000],
            }
        )
        df = VolumeIndicators.calculate_volume_indicators(test_df, window=2)

        # Verify columns exist
        self.assertIn("volume_ma_20", df.columns)
        self.assertIn("volume_ratio", df.columns)
        self.assertIn("OBV", df.columns)

        # Verify basic properties
        self.assertTrue(pd.api.types.is_numeric_dtype(df["volume_ma_20"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(df["volume_ratio"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(df["OBV"]))

        # Only check volume ratio on non-NaN values
        non_nan_mask = df["volume_ratio"].notna()
        self.assertTrue((df.loc[non_nan_mask, "volume_ratio"] > 0).all())

        # Verify volume MA calculation
        # For window=2, first value should be NaN, then average of previous 2 volumes
        expected_ma = pd.Series([np.nan, 1500, 2500, 3500, 4500], index=test_df.index, dtype=float)
        pd.testing.assert_series_equal(df["volume_ma_20"], expected_ma, check_names=False)

        # Verify OBV calculation
        # TA-Lib's OBV calculation:
        # 1. First value: OBV = volume (1000)
        # 2. Second value: price up (102 > 100), so add volume: 1000 + 2000 = 3000
        # 3. Third value: price down (101 < 102), so subtract volume: 3000 - 3000 = 0
        # 4. Fourth value: price up (103 > 101), so add volume: 0 + 4000 = 4000
        # 5. Fifth value: price down (102 < 103), so subtract volume: 4000 - 5000 = -1000
        expected_obv = pd.Series([1000.0, 3000.0, 0.0, 4000.0, -1000.0], index=test_df.index, dtype=float)
        pd.testing.assert_series_equal(df["OBV"], expected_obv, check_names=False)

    def test_calculate_volatility_indicators(self):
        """Test volatility indicator calculations"""
        # Create test data with known True Range values
        test_df = pd.DataFrame(
            {
                "high": [100, 105, 103, 107, 104],  # First value will have TR = 2 (high-low)
                "low": [
                    98,
                    100,
                    101,
                    102,
                    103,
                ],  # Rest will have TR = max(high-low, |high-prev_close|, |low-prev_close|)
                "close": [99, 102, 101, 105, 103],  # Previous close values for TR calculation
                "returns": [0.01, 0.03, -0.01, 0.04, -0.02],
            }
        )
        df = VolatilityIndicators.calculate_volatility_indicators(test_df, window=2)

        # Verify columns exist
        self.assertIn("ATR", df.columns)
        self.assertIn("NATR", df.columns)
        self.assertIn("TRANGE", df.columns)
        self.assertIn("volatility", df.columns)

        # Verify basic properties
        for col in ["ATR", "NATR", "TRANGE", "volatility"]:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]))

            # Only check non-negativity on non-NaN values
            non_nan_mask = df[col].notna()
            self.assertTrue((df.loc[non_nan_mask, col] >= 0).all())

        # Verify TRANGE calculation
        non_nan_mask = df["TRANGE"].notna()

        # First value: TR = high - low = 100 - 98 = 2
        # Second value: TR = max(105-100, |105-99|, |100-99|) = max(5, 6, 1) = 6
        # Third value: TR = max(103-101, |103-102|, |101-102|) = max(2, 1, 1) = 2
        # Fourth value: TR = max(107-102, |107-101|, |102-101|) = max(5, 6, 1) = 6
        # Fifth value: TR = max(104-103, |104-105|, |103-105|) = max(1, 1, 2) = 2
        expected_trange = pd.Series([2.0, 6.0, 2.0, 6.0, 2.0], index=test_df.index)

        pd.testing.assert_series_equal(df.loc[non_nan_mask, "TRANGE"], expected_trange[non_nan_mask], check_names=False)

        # Verify we have some non-NaN values
        self.assertTrue(df["ATR"].notna().any())
        self.assertTrue(df["NATR"].notna().any())
        self.assertTrue(df["TRANGE"].notna().any())
        self.assertTrue(df["volatility"].notna().any())

        # Verify the last values are not NaN
        self.assertFalse(pd.isna(df["ATR"].iloc[-1]))
        self.assertFalse(pd.isna(df["NATR"].iloc[-1]))
        self.assertFalse(pd.isna(df["TRANGE"].iloc[-1]))
        self.assertFalse(pd.isna(df["volatility"].iloc[-1]))

    def test_calculate_all_indicators(self):
        """Test calculation of all indicators"""
        df = calculate_all_indicators(self.df)

        # Verify all expected columns exist
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
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]))

    def test_get_current_market_data(self):
        """Test get_current_market_data method"""
        df = calculate_all_indicators(self.df)
        market_data = TechnicalAnalysis.get_current_market_data(df)

        # Verify structure
        self.assertIn("timestamp", market_data)
        self.assertIn("price_data", market_data)
        self.assertIn("indicators", market_data)

        # Verify price data
        price_data = market_data["price_data"]
        required_price_fields = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "price_changes",
            "support_level",
            "resistance_level",
        ]
        for field in required_price_fields:
            self.assertIn(field, price_data)

        # Verify indicators
        indicators = market_data["indicators"]
        required_indicator_groups = ["bollinger_bands", "moving_averages", "momentum", "volatility", "volume"]
        for group in required_indicator_groups:
            self.assertIn(group, indicators)

    def test_get_support_resistance_levels(self):
        """Test support and resistance level calculations"""
        df = calculate_all_indicators(self.df)
        current_price = df["close"].iloc[-1]
        support, resistance = TechnicalAnalysis.get_support_resistance_levels(df, current_price)

        # Verify basic properties
        self.assertIsInstance(support, float)
        self.assertIsInstance(resistance, float)
        self.assertLess(support, resistance)
        self.assertLess(support, current_price)
        self.assertGreater(resistance, current_price)

    def test_calculate_pivot_points(self):
        """Test pivot points calculations"""
        test_df = pd.DataFrame({"high": [100, 110], "low": [90, 95], "close": [95, 105]})
        pivot, r1, r2, r3, s1, s2, s3 = TechnicalAnalysis.calculate_pivot_points(test_df)

        # Verify basic properties
        self.assertIsInstance(pivot, float)
        self.assertIsInstance(r1, float)
        self.assertIsInstance(r2, float)
        self.assertIsInstance(r3, float)
        self.assertIsInstance(s1, float)
        self.assertIsInstance(s2, float)
        self.assertIsInstance(s3, float)

        # Verify order of levels
        self.assertLess(s3, s2)
        self.assertLess(s2, s1)
        self.assertLess(s1, pivot)
        self.assertLess(pivot, r1)
        self.assertLess(r1, r2)
        self.assertLess(r2, r3)

    def test_find_nearest_level(self):
        """Test finding nearest support/resistance level"""
        price = 100
        levels = [90, 95, 105, 110]

        nearest = TechnicalAnalysis.find_nearest_level(price, levels)
        self.assertIsInstance(nearest, (int, float))  # Allow both int and float
        self.assertIn(nearest, levels)

        # Test with empty levels list
        with self.assertRaises(ValueError):
            TechnicalAnalysis.find_nearest_level(price, [])

        # Test with float values
        price = 100.5
        levels = [90.5, 95.5, 105.5, 110.5]
        nearest = TechnicalAnalysis.find_nearest_level(price, levels)
        self.assertIsInstance(nearest, (int, float))
        self.assertIn(nearest, levels)

    def test_calculate_price_changes(self):
        """Test price changes calculations"""
        test_df = pd.DataFrame({"close": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]})
        changes = TechnicalAnalysis.calculate_price_changes(test_df)

        # Verify structure
        required_periods = ["1w", "2w", "1m", "6m"]
        for period in required_periods:
            self.assertIn(period, changes)
            self.assertIsInstance(changes[period], float)


if __name__ == "__main__":
    unittest.main()
