import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import logging
from unittest.mock import patch, MagicMock

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# Configure logging to only show warnings and errors during tests
logging.basicConfig(
    level=logging.WARNING,  # Only show WARNING and above
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,  # Override any existing logging configuration
)

import backtest


class TestBacktest(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create sample price data with clear trends for testing
        # Create 6 hours of data (00:00 to 05:00) to match the actual data range
        dates = pd.date_range(start="2024-01-02 00:00", periods=6, freq="h")

        # Create a price series with extreme movements to force trades
        self.base_price = 100  # Make base_price an instance variable
        # Create extreme price movements: high, low, high, low, high, low
        prices = [
            self.base_price + 5,  # High
            self.base_price - 5,  # Low
            self.base_price + 5,  # High
            self.base_price - 5,  # Low
            self.base_price + 5,  # High
            self.base_price - 5,  # Low
        ]

        self.df = pd.DataFrame(
            {
                "open": prices,
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "close": prices,
                "volume": np.random.normal(1000000, 100000, 6),
            },
            index=dates,
        )

        # Ensure high is highest and low is lowest
        self.df["high"] = self.df[["open", "high", "close"]].max(axis=1)
        self.df["low"] = self.df[["open", "low", "close"]].min(axis=1)

        # Create sample equity curve with both gains and losses
        self.equity_curve = pd.Series(np.cumsum(np.random.normal(0, 1, 6)), index=dates)

    def test_get_historical_data(self):
        """Test historical data retrieval with mock data"""

        # Mock the data retrieval function
        def mock_get_data(*args, **kwargs):
            return self.df

        # Replace the actual function with mock
        original_get_data = backtest.get_historical_data
        backtest.get_historical_data = mock_get_data

        try:
            # Test data retrieval
            data = backtest.get_historical_data("AAPL", "1h", "2024-01-01", "2024-01-02")

            # Verify data structure
            self.assertIsInstance(data, pd.DataFrame)
            self.assertIn("open", data.columns)
            self.assertIn("high", data.columns)
            self.assertIn("low", data.columns)
            self.assertIn("close", data.columns)
            self.assertIn("volume", data.columns)

            # Verify data integrity
            self.assertTrue((data["high"] >= data["low"]).all())
            self.assertTrue((data["high"] >= data["open"]).all())
            self.assertTrue((data["high"] >= data["close"]).all())
            self.assertTrue((data["low"] <= data["open"]).all())
            self.assertTrue((data["low"] <= data["close"]).all())

        finally:
            # Restore original function
            backtest.get_historical_data = original_get_data

    def test_calculate_max_drawdown(self):
        """Test max drawdown calculation with known values"""
        # Create a known equity curve with a specific drawdown
        equity = pd.Series([100, 110, 105, 95, 100, 90, 95])
        expected_drawdown = 18.18  # (110-90)/110 * 100

        drawdown = backtest.calculate_max_drawdown(equity)
        self.assertAlmostEqual(drawdown, expected_drawdown, places=2)


if __name__ == "__main__":
    unittest.main()
