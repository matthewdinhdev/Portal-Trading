import argparse
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytz
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import (
    OrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)
from dotenv import load_dotenv

import discord_bot
from logger import setup_logger
from trading_enums import TradingEnvironment
from utility import (
    TRADING_STRATEGY,
    MarketDataManager,
)

# Set up logger
logger = setup_logger("paper_trading.log")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run the trading bot with optional market hours bypass.")
parser.add_argument(
    "--env",
    type=str,
    required=True,
    choices=["paper_trading", "live_trading"],
    help="Trading environment (paper_trading, live_trading)",
)
parser.add_argument(
    "--bypass-market-hours", action="store_true", help="Bypass market hours check to allow trading at any time"
)
parser.add_argument("--symbols", type=str, help='Comma-separated list of symbols to trade (e.g., "SPY,AAPL,NVDA")')
parser.add_argument("--dont-save-analysis", action="store_true", help="Skip saving analysis results to files")
args = parser.parse_args()

# Load environment variables
load_dotenv()

# Initialize market data manager
market_data_manager = MarketDataManager()

# Initialize Alpaca clients
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Convert environment string to enum
env = TradingEnvironment(args.env)

# Define trading symbols
TRADING_SYMBOLS = [
    "SPY",  # S&P 500 ETF
    # "AAPL",  # Apple
    # "MSFT",  # Microsoft
    # "GOOGL",  # Alphabet
    # "AMZN",  # Amazon
    "NVDA",  # NVIDIA
    # "META",  # Meta Platforms
    "TSLA",  # Tesla
    # "AMD",  # Advanced Micro Devices
    # "LLY",  # Eli Lilly
    # "JNJ",  # Johnson & Johnson
    "UNH",  # UnitedHealth
    # "JPM",  # JPMorgan Chase
    # "V",  # Visa
    # "MA",  # Mastercard
    # "WMT",  # Walmart
    # "PG",  # Procter & Gamble
    # "XOM",  # ExxonMobil
    # "CVX",  # Chevron
    # "AVGO",  # Broadcom
    # "ADBE",  # Adobe
    # "HIMS",  # Hims and Hers Health
]

# Override symbols if provided via command line
if args.symbols:
    TRADING_SYMBOLS = [symbol.strip().upper() for symbol in args.symbols.split(",")]
    logger.info(f"Using custom symbol list: {TRADING_SYMBOLS}")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file")

# Initialize trading client based on environment
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=env == TradingEnvironment.PAPER)

# Set up timezone
et_tz = pytz.timezone("US/Eastern")


def get_current_et_time() -> datetime:
    """Get current time in ET."""
    return datetime.now(et_tz)


def is_market_open() -> bool:
    """Check if the market is open at the given time.

    Returns:
        bool: True if market is open, False otherwise.
    """
    # If bypass flag is set, always return True
    if args.bypass_market_hours:
        logger.info(" . Market hours check bypassed")
        return True

    logger.info(" . Checking if market is open")

    # Get current market clock
    clock = trading_client.get_clock()

    return clock.is_open


def get_next_market_open() -> Optional[datetime]:
    """Get the next market open time.

    Returns:
        Optional[datetime]: The next market open datetime, or None if there's an error.
    """
    # Get current market clock
    clock = trading_client.get_clock()

    # Return next market open time
    return clock.next_open


def get_account_info() -> Optional[Dict[str, Any]]:
    """Get account information from Alpaca.

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing account information including equity,
            cash, buying power, and trading status. Returns None if there's an error.
    """
    logger.info(" . Getting account information")
    account = trading_client.get_account()

    if not account:
        raise Exception("Failed to get account information")

    return {
        "equity": float(account.equity),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "daytrade_count": account.daytrade_count,
        "initial_margin": float(account.initial_margin),
        "maintenance_margin": float(account.maintenance_margin),
        "last_equity": float(account.last_equity),
        "last_maintenance_margin": float(account.last_maintenance_margin),
        "long_market_value": float(account.long_market_value),
        "short_market_value": float(account.short_market_value),
        "portfolio_value": float(account.portfolio_value),
        "pattern_day_trader": account.pattern_day_trader,
        "trading_blocked": account.trading_blocked,
        "transfers_blocked": account.transfers_blocked,
        "account_blocked": account.account_blocked,
        "created_at": account.created_at.isoformat(),
        "trade_suspended_by_user": account.trade_suspended_by_user,
        "multiplier": account.multiplier,
        "shorting_enabled": account.shorting_enabled,
        "status": account.status,
        "environment": "PAPER" if env == TradingEnvironment.PAPER else "LIVE",
    }


def get_current_positions() -> List[Dict[str, Any]]:
    """Get current positions from Alpaca.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing position information for each
            held position. Each dictionary includes symbol, quantity, entry price, and P/L.
            Returns empty list if there's an error.
    """
    logger.info(" . Getting current positions")
    positions = trading_client.get_all_positions()
    return [
        {
            "symbol": position.symbol,
            "qty": float(position.qty),
            "avg_entry_price": float(position.avg_entry_price),
            "market_value": float(position.market_value),
            "cost_basis": float(position.cost_basis),
            "unrealized_pl": float(position.unrealized_pl),
            "unrealized_plpc": float(position.unrealized_plpc),
            "current_price": float(position.current_price),
            "lastday_price": float(position.lastday_price),
            "change_today": float(position.change_today),
        }
        for position in positions
    ]


def execute_trade(analysis: Dict[str, Any], account_info: Dict[str, Any]) -> None:
    """Execute a trade based on analysis and account information."""
    logger.info(f"Executing trade for {analysis['symbol']}")

    # Send to Discord for all recommendations
    if discord_bot.send_to_discord(analysis):
        logger.info(" . Successfully sent to Discord")

    # Only execute trades for BUY/SELL recommendations
    if analysis["recommendation"] == "HOLD":
        logger.info(f" . HOLD recommendation for {analysis['symbol']} - no trade executed")
        return

    # Get price targets from analysis
    price_targets = analysis["price_targets"]
    stop_loss = price_targets["stop_loss"]
    take_profit = price_targets["take_profit"]

    # Calculate quantity using market data manager
    qty = market_data_manager.calculate_position_size(
        confidence=analysis["confidence"],
        available_cash=float(account_info["cash"]),
        current_price=analysis["current_price"],
    )

    # Setup order request
    stop_loss = round(float(stop_loss), 2)
    take_profit = round(float(take_profit), 2)
    time_in_force = TimeInForce.GTC
    order_request = OrderRequest(
        symbol=analysis["symbol"],
        qty=qty,
        side=OrderSide.BUY if analysis["recommendation"].lower() == "buy" else OrderSide.SELL,
        type="market",
        time_in_force=time_in_force,
        order_class="bracket",
        stop_loss=StopLossRequest(
            stop_price=stop_loss,
            time_in_force=time_in_force,
        ),
        take_profit=TakeProfitRequest(
            limit_price=take_profit,
            time_in_force=time_in_force,
        ),
    )

    # Submit order
    order = trading_client.submit_order(order_request)
    order = trading_client.get_order_by_id(order.id)

    # Return order details
    return {
        "symbol": analysis["symbol"],
        "side": analysis["recommendation"],
        "quantity": qty,
        "status": order.status,
        "filled_at": order.filled_at.isoformat() if order.filled_at else None,
        "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
        "environment": "PAPER" if env == TradingEnvironment.PAPER else "LIVE",
        "price_targets": {"stop_loss": stop_loss, "take_profit": take_profit},
    }


def display_position_updates() -> None:
    """Display current position updates including P/L and current prices.

    Logs position information including quantity, entry price, current price,
    unrealized P/L, and market value for each active position.
    """
    positions = get_current_positions()
    if not positions:
        logger.info(" . No active positions")
        return

    logger.info("Current Position Updates:")
    logger.info("=" * 50)
    for pos in positions:
        pnl_color = "+" if pos["unrealized_pl"] >= 0 else ""
        logger.info(f"Symbol: {pos['symbol']}")
        logger.info(f"  Quantity: {pos['qty']}")
        logger.info(f"  Entry Price: ${pos['avg_entry_price']:.2f}")
        logger.info(f"  Current Price: ${pos['current_price']:.2f}")
        logger.info(
            f"  Unrealized P/L: ${pnl_color}{pos['unrealized_pl']:.2f} ({pnl_color}{pos['unrealized_plpc']*100:.2f}%)"
        )
        logger.info(f"  Market Value: ${pos['market_value']:.2f}")
        logger.info("-" * 50)


def main() -> None:
    """Main function to run the trading bot."""
    logger.info("Starting trading bot...")

    while True:
        current_time = get_current_et_time()

        # Check if market is open
        if not is_market_open():
            next_market_open = get_next_market_open()
            wait_seconds = (next_market_open - current_time).total_seconds()
            logger.info(f"Market is closed. Waiting {wait_seconds/3600:.1f} hours until next market open...")
            time.sleep(wait_seconds)

        # Analyze each symbol and execute trades
        for symbol in TRADING_SYMBOLS:
            account_info = get_account_info()
            if not account_info:
                continue

            # Get historical data
            end_date = current_time - timedelta(days=1)  # Use previous day's close
            start_date = end_date - timedelta(days=TRADING_STRATEGY["lookback_periods"])
            df = market_data_manager.get_historical_data(symbol, start_date, end_date)

            # Get analysis
            result = market_data_manager.analyze_symbol(
                symbol=symbol, data=df, env=env, save_analysis=not args.dont_save_analysis
            )

            if result:
                execute_trade(result["analysis"], account_info)

        # Show position updates every 20 minutes
        wait_seconds = 1200
        while True:
            # if market is closed, wait until next market open
            if not is_market_open():
                next_market_open = get_next_market_open()
                wait_seconds = (next_market_open - current_time).total_seconds()
                logger.info(f"Market is closed. Waiting {wait_seconds/3600:.1f} hours until next market open...")
                time.sleep(wait_seconds)
                continue

            # if market is open, display position updates every 10 minutes
            minutes = int(wait_seconds // 60)
            seconds = int(wait_seconds % 60)
            logger.info(f"Waiting {minutes} minutes and {seconds} seconds until next run time...")
            display_position_updates()
            time.sleep(600)


if __name__ == "__main__":
    main()
