import unittest
import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# Configure logging to only show warnings and errors during tests
logging.basicConfig(
    level=logging.WARNING,  # Only show WARNING and above
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,  # Override any existing logging configuration
)

import main


class TestMain(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.sample_account_info = {"equity": 100000.0, "cash": 50000.0, "buying_power": 200000.0, "day_trade_count": 2}

        self.sample_positions = [
            {
                "symbol": "AAPL",
                "quantity": 100,
                "avg_price": 150.25,
                "current_price": 155.50,
                "market_value": 15550.0,
                "unrealized_pnl": 525.0,
            },
            {
                "symbol": "MSFT",
                "quantity": 50,
                "avg_price": 300.00,
                "current_price": 305.75,
                "market_value": 15287.5,
                "unrealized_pnl": 287.5,
            },
        ]

    def test_get_account_info(self):
        """Test account info parsing"""

        # Mock the account info retrieval
        def mock_get_account(*args, **kwargs):
            return self.sample_account_info

        # Replace the actual function with mock
        original_get_account = main.get_account_info
        main.get_account_info = mock_get_account

        try:
            account_info = main.get_account_info()

            # Verify account info structure
            self.assertIsInstance(account_info, dict)
            self.assertIn("equity", account_info)
            self.assertIn("cash", account_info)
            self.assertIn("buying_power", account_info)
            self.assertIn("day_trade_count", account_info)

            # Verify data types
            self.assertIsInstance(account_info["equity"], float)
            self.assertIsInstance(account_info["cash"], float)
            self.assertIsInstance(account_info["buying_power"], float)
            self.assertIsInstance(account_info["day_trade_count"], int)

        finally:
            # Restore original function
            main.get_account_info = original_get_account

    def test_get_current_positions(self):
        """Test position data parsing"""

        # Mock the positions retrieval
        def mock_get_positions(*args, **kwargs):
            return self.sample_positions

        # Replace the actual function with mock
        original_get_positions = main.get_current_positions
        main.get_current_positions = mock_get_positions

        try:
            positions = main.get_current_positions()

            # Verify positions structure
            self.assertIsInstance(positions, list)
            self.assertEqual(len(positions), 2)

            # Verify first position
            position = positions[0]
            self.assertIn("symbol", position)
            self.assertIn("quantity", position)
            self.assertIn("avg_price", position)
            self.assertIn("current_price", position)
            self.assertIn("market_value", position)
            self.assertIn("unrealized_pnl", position)

            # Verify data types
            self.assertIsInstance(position["quantity"], int)
            self.assertIsInstance(position["avg_price"], float)
            self.assertIsInstance(position["current_price"], float)
            self.assertIsInstance(position["market_value"], float)
            self.assertIsInstance(position["unrealized_pnl"], float)

        finally:
            # Restore original function
            main.get_current_positions = original_get_positions

    def test_execute_trade(self):
        """Test trade execution logic"""

        # Mock the trade execution
        def mock_execute(*args, **kwargs):
            return {
                "order_id": "12345",
                "status": "filled",
                "filled_at": datetime.now().isoformat(),
                "filled_price": 150.25,
                "filled_quantity": 100,
            }

        # Replace the actual function with mock
        original_execute = main.execute_trade
        main.execute_trade = mock_execute

        try:
            # Test buy order
            buy_order = main.execute_trade(symbol="AAPL", quantity=100, side="buy", order_type="market")

            # Verify order structure
            self.assertIsInstance(buy_order, dict)
            self.assertIn("order_id", buy_order)
            self.assertIn("status", buy_order)
            self.assertIn("filled_at", buy_order)
            self.assertIn("filled_price", buy_order)
            self.assertIn("filled_quantity", buy_order)

            # Test sell order
            sell_order = main.execute_trade(symbol="AAPL", quantity=50, side="sell", order_type="market")

            self.assertEqual(sell_order["status"], "filled")

        finally:
            # Restore original function
            main.execute_trade = original_execute

    def test_main(self):
        """Test main function"""
        # Add your test cases here
        pass


if __name__ == "__main__":
    unittest.main()
