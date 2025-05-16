import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import (
    GetOrdersRequest,
    OrderRequest,
    GetCalendarRequest,
    StopLossRequest,
    TakeProfitRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv
import strategies
import llm
import discord_bot
from logger import setup_logger
from typing import Dict, List, Optional, Any
import pytz
import argparse

# Set up logger
logger = setup_logger("paper_trading.log")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run the trading bot with optional market hours bypass.")
parser.add_argument(
    "--bypass-market-hours", action="store_true", help="Bypass market hours check to allow trading at any time"
)
parser.add_argument("--symbols", type=str, help='Comma-separated list of symbols to trade (e.g., "SPY,AAPL,NVDA")')
args = parser.parse_args()

# Load environment variables
load_dotenv()

# Initialize Alpaca clients
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
CHECK_INTERVAL = 3600  # 1 hour in seconds

# Define trading strategy parameters
TRADING_STRATEGY: Dict[str, Any] = {
    "symbols": [
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
    ],
    "timeframe": TimeFrame.Hour,
    "lookback_periods": 120,  # 5 days of hourly data
}

# Override symbols if provided via command line
if args.symbols:
    TRADING_STRATEGY["symbols"] = [symbol.strip().upper() for symbol in args.symbols.split(",")]
    logger.info(f"Using custom symbol list: {TRADING_STRATEGY['symbols']}")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file")

# Initialize trading client based on environment
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER_TRADING)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Cache for market calendar
_market_calendar_cache = None
_last_calendar_update = None
_calendar_cache_duration = timedelta(hours=1)  # Update calendar cache every hour


def get_market_calendar() -> Optional[List[Dict[str, Any]]]:
    """Get market calendar from Alpaca API with caching.

    Returns:
        Optional[List[Dict[str, Any]]]: List of dictionaries containing market session information.
            Returns None if there's an error.
    """
    global _market_calendar_cache, _last_calendar_update

    # Use timezone-aware datetime
    current_time = datetime.now(timezone.utc)

    # Return cached calendar if it's still valid
    if (
        _market_calendar_cache is not None
        and _last_calendar_update is not None
        and current_time - _last_calendar_update < _calendar_cache_duration
    ):
        return _market_calendar_cache

    try:
        # Get calendar for the next 30 days
        start_date = current_time.date()
        end_date = start_date + timedelta(days=30)

        # Create calendar request with date filters
        calendar_request = GetCalendarRequest(start=start_date, end=end_date)

        # Get calendar using the request
        calendar = trading_client.get_calendar(filters=calendar_request)

        # Update cache
        _market_calendar_cache = calendar
        _last_calendar_update = current_time

        return calendar
    except Exception as e:
        logger.error(f"Error getting market calendar: {str(e)}")
        return None


def is_market_open() -> bool:
    """Check if the market is open at the given time.

    Args:
        current_time: The datetime to check market status for.

    Returns:
        bool: True if market is open, False otherwise.
    """
    # If bypass flag is set, always return True
    if args.bypass_market_hours:
        logger.info(" . Market hours check bypassed")
        return True

    logger.info(" . Checking if market is open")

    try:
        # Get current market clock
        clock = trading_client.get_clock()

        # Log market status
        logger.info(f" . Market is {'open' if clock.is_open else 'closed'}")
        logger.info(f" . Next market open: {clock.next_open.strftime('%Y-%m-%d %H:%M:%S %Z')} (ET)")
        logger.info(f" . Next market close: {clock.next_close.strftime('%Y-%m-%d %H:%M:%S %Z')} (ET)")

        return clock.is_open
    except Exception as e:
        logger.error(f" . Error getting market clock: {str(e)}")
        return False


def get_next_market_open() -> Optional[datetime]:
    """Get the next market open time.

    Args:

        current_time: The current datetime.

    Returns:
        Optional[datetime]: The next market open datetime, or None if there's an error.
    """
    try:
        # Get current market clock
        clock = trading_client.get_clock()

        # Return next market open time
        return clock.next_open
    except Exception as e:
        logger.error(f" . Error getting next market open: {str(e)}")
        return None


def get_account_info() -> Optional[Dict[str, Any]]:
    """Get account information from Alpaca.

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing account information including equity,
            cash, buying power, and trading status. Returns None if there's an error.
    """
    logger.info(" . Getting account information")
    try:
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
            "environment": "PAPER" if PAPER_TRADING else "LIVE",
        }
    except Exception as e:
        logger.error(f". Error fetching account info: {str(e)}")
        return None


def get_current_positions() -> List[Dict[str, Any]]:
    """Get current positions from Alpaca.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing position information for each
            held position. Each dictionary includes symbol, quantity, entry price, and P/L.
            Returns empty list if there's an error.
    """
    logger.info(" . Getting current positions")
    try:
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
    except Exception as e:
        logger.error(f". Error fetching positions: {str(e)}")
        return []


def execute_trade(
    symbol: str,
    side: str,
    qty: float,
    stop_loss: float,
    take_profit: float,
    trade_type: str = "day",
) -> Optional[Dict[str, Any]]:
    """Execute a trade with stop loss and take profit orders."""
    logger.info(f" . Executing {side.upper()} trade for {symbol}")
    logger.info(f"   . Quantity: {qty}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")

    try:
        # Setup order request
        stop_loss = round(float(stop_loss), 2)
        take_profit = round(float(take_profit), 2)
        time_in_force = TimeInForce.DAY if trade_type == "day" else TimeInForce.GTC
        order_request = OrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
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
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "status": order.status,
            "filled_at": order.filled_at.isoformat() if order.filled_at else None,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "environment": "PAPER" if PAPER_TRADING else "LIVE",
            "trade_type": trade_type,
            "price_targets": {"stop_loss": stop_loss, "take_profit": take_profit},
        }

    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")
        return None


def get_historical_data(
    symbol: str, start_date: datetime, end_date: datetime, timeframe: TimeFrame = TimeFrame.Hour
) -> pd.DataFrame:
    """Get historical data for a symbol.

    Args:
        symbol: The trading symbol to get data for.
        start_date: Start date for historical data.
        end_date: End date for historical data.
        timeframe: Timeframe for the data (default: TimeFrame.Hour).

    Returns:
        pd.DataFrame: DataFrame containing historical price data with datetime index.
    """
    logger.info(f" . Getting historical data for {symbol}")
    request_params = StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start_date, end=end_date)

    bars = data_client.get_stock_bars(request_params)
    df = bars.df

    # Reset multi-index to single index and ensure it's a datetime index
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    # Ensure the index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def execute_trading_decision(symbol: str, analysis: Dict[str, Any], account_info: Dict[str, Any]) -> None:
    """Execute trading decision based on analysis and account information."""
    logger.info(f"Executing trading decision for {symbol}")

    # Get price targets from analysis
    price_targets = analysis["price_targets"]
    stop_loss = price_targets["stop_loss"]
    take_profit = price_targets["take_profit"]

    # Get current price for Discord message
    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=datetime.now() - timedelta(minutes=1),
            end=datetime.now(),
        )
        bars = data_client.get_stock_bars(request_params)
        current_price = float(bars.df["close"].iloc[-1])
    except Exception as e:
        logger.error(f"Error getting current price: {str(e)}")
        return

    # Calculate quantity based on position size percentage
    position_size = analysis["position_size"]
    if isinstance(position_size, str):
        position_size_pct = float(position_size.strip("%")) / 100
    else:
        position_size_pct = float(position_size) / 100
    account_value = float(account_info["equity"])
    position_value = account_value * position_size_pct
    qty = int(position_value / current_price)  # Round down to whole shares

    # If quantity is 0, set to 1 share
    if qty == 0:
        qty = 1

    # Execute the trade
    trade_result = execute_trade(
        symbol=symbol,
        side=analysis["recommendation"],
        qty=qty,
        trade_type=analysis["trade_type"],
        stop_loss=stop_loss,
        take_profit=take_profit,
    )

    # Send to Discord if trade was executed
    if trade_result and discord_bot.send_to_discord(analysis, symbol, current_price):
        logger.info("Successfully sent to Discord")


def analyze_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    """Analyze a single symbol and return analysis results."""
    logger.info(f"Analyzing {symbol}")
    try:
        account_info = get_account_info()
        if not account_info:
            return None

        # Check for existing analysis from this hour
        existing_analysis = llm.get_existing_analysis(symbol)
        if existing_analysis:
            analysis = existing_analysis
        else:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=TRADING_STRATEGY["lookback_periods"])
            df = get_historical_data(symbol, start_date, end_date, TRADING_STRATEGY["timeframe"])

            # Calculate indicators and get analysis
            df = strategies.calculate_indicators(df)
            llm_prompt = strategies.get_llm_prompt(df, account_info, get_current_positions())
            if not llm_prompt:
                return None

            analysis = llm.get_trading_analysis(llm_prompt)
            if not analysis:
                return None

            llm.save_analysis(analysis, symbol)

        # Return analysis results if there's a trading recommendation
        if analysis["recommendation"] in ["BUY", "SELL"]:
            return {
                "symbol": symbol,
                "analysis": analysis,
                "account_info": account_info,
            }
        return None
    except Exception as e:
        logger.error(f"Error analyzing symbol {symbol}: {str(e)}")
        return None


def display_position_updates() -> None:
    """Display current position updates including P/L and current prices.

    Logs position information including quantity, entry price, current price,
    unrealized P/L, and market value for each active position.
    """
    positions = get_current_positions()
    if not positions:
        logger.info("No active positions")
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
        try:
            # Calculate next run time (5 minutes after the next hour)
            current_time = datetime.now(timezone.utc)
            next_run = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1, minutes=5)
            wait_seconds = (next_run - current_time).total_seconds()

            # Check if market is open
            if not is_market_open():
                next_market_open = get_next_market_open()
                if next_market_open:
                    wait_seconds = (next_market_open - current_time).total_seconds()
                    if wait_seconds > 0:
                        logger.info(
                            f"Market is closed. Waiting {wait_seconds/3600:.1f} hours until next market open..."
                        )
                        time.sleep(wait_seconds)
                    else:
                        time.sleep(300)
                else:
                    time.sleep(300)
                continue

            # Analyze each symbol and execute trades
            for symbol in TRADING_STRATEGY["symbols"]:
                result = analyze_symbol(symbol)
                if result:
                    execute_trading_decision(result["symbol"], result["analysis"], result["account_info"])

            # Show position updates every 10 minutes
            wait_start = datetime.now(timezone.utc)
            while (datetime.now(timezone.utc) - wait_start).total_seconds() < wait_seconds:
                display_position_updates()
                if not is_market_open():
                    logger.info("Market is closed. Breaking wait loop...")
                    break

                remaining = wait_seconds - (datetime.now(timezone.utc) - wait_start).total_seconds()
                minutes = int(remaining // 60)
                seconds = int(remaining % 60)
                logger.info(f"Waiting {minutes} minutes and {seconds} seconds until next run time...")
                time.sleep(600)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt. Exiting...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            time.sleep(60)


if __name__ == "__main__":
    main()
