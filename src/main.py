import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv
import strategies
import llm
import discord
from logger import setup_logger
from typing import Dict, List, Optional, Any, Union, Tuple

# Set up logger
logger = setup_logger("paper_trading.log")

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
        # "NVDA",  # NVIDIA
        # "META",  # Meta Platforms
        # "TSLA",  # Tesla
        # "AMD",  # Advanced Micro Devices
        # "LLY",  # Eli Lilly
        # "JNJ",  # Johnson & Johnson
        # "UNH",  # UnitedHealth
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

if not API_KEY or not SECRET_KEY:
    raise ValueError("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file")

# Initialize trading client based on environment
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER_TRADING)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


def get_account_info() -> Optional[Dict[str, Any]]:
    """Get account information from Alpaca"""
    try:
        account = trading_client.get_account()
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
        logger.error(f"Error fetching account info: {str(e)}")
        return None


def get_current_positions() -> List[Dict[str, Any]]:
    """Get current positions from Alpaca"""
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
        logger.error(f"Error fetching positions: {str(e)}")
        return []


def execute_trade(
    symbol: str,
    side: str,
    qty: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    trade_type: str = "day",
) -> Optional[Dict[str, Any]]:
    """
    Execute a trade with optional stop loss and take profit orders

    Args:
        symbol: The trading symbol
        side: 'buy' or 'sell'
        qty: Quantity to trade
        stop_loss: Optional stop loss price
        take_profit: Optional take profit price
        trade_type: 'day' or 'swing' - determines order time in force

    Returns:
        Trade execution details or None if execution fails
    """
    try:
        # Create the market order
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY if trade_type == "day" else TimeInForce.GTC,
        )

        # Submit the order
        order = trading_client.submit_order(order_data)

        # Wait for the order to fill
        order = trading_client.get_order_by_id(order.id)

        # Create response dictionary
        trade_details = {
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "status": order.status,
            "filled_at": order.filled_at.isoformat() if order.filled_at else None,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "environment": "PAPER" if PAPER_TRADING else "LIVE",
            "trade_type": trade_type,
        }

        # Add stop loss and take profit if provided
        if stop_loss or take_profit:
            trade_details["additional_orders"] = []

            if stop_loss:
                stop_order = trading_client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.SELL if side.lower() == "buy" else OrderSide.BUY,
                        time_in_force=TimeInForce.GTC,
                        stop_price=stop_loss,
                    )
                )
                trade_details["additional_orders"].append(
                    {"type": "stop_loss", "price": stop_loss, "order_id": stop_order.id}
                )

            if take_profit:
                take_profit_order = trading_client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.SELL if side.lower() == "buy" else OrderSide.BUY,
                        time_in_force=TimeInForce.GTC,
                        limit_price=take_profit,
                    )
                )
                trade_details["additional_orders"].append(
                    {"type": "take_profit", "price": take_profit, "order_id": take_profit_order.id}
                )

        return trade_details

    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")
        return None


def get_historical_data(
    symbol: str, start_date: datetime, end_date: datetime, timeframe: TimeFrame = TimeFrame.Hour
) -> pd.DataFrame:
    """Get historical data for a symbol"""
    request_params = StockBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start_date, end=end_date)

    bars = data_client.get_stock_bars(request_params)
    df = bars.df

    # Reset multi-index to single index
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    return df


def analyze_symbol(symbol: str) -> None:
    """Analyze a single symbol and execute trades if needed"""
    try:
        logger.info(f"\nAnalyzing {symbol}...")
        logger.info("=" * 80)

        # Get account information
        account_info = get_account_info()
        positions = get_current_positions()

        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=TRADING_STRATEGY["lookback_periods"])
        df = get_historical_data(symbol, start_date, end_date, TRADING_STRATEGY["timeframe"])

        # Calculate indicators
        df = strategies.calculate_indicators(df)

        # Generate LLM prompt for the current market conditions
        llm_prompt = strategies.get_llm_prompt(df, account_info, positions)

        # Get LLM analysis
        analysis = llm.get_trading_analysis(llm_prompt)
        if analysis:
            logger.info(llm.format_analysis_for_display(analysis))
            llm.save_analysis(analysis, symbol)

            # Execute trade if recommended
            if analysis["recommendation"] in ["BUY", "SELL"]:
                logger.info("\nExecuting trade based on analysis...")
                # Get position size and trade type from LLM analysis
                portfolio_value = account_info["portfolio_value"]
                position_size = analysis.get("position_size", 0.01)  # Default to 1%
                trade_type = analysis.get("trade_type", "day")

                # Calculate position size in dollars
                position_value = portfolio_value * position_size
                current_price = df["close"].iloc[-1]
                qty = int(position_value / current_price)

                # Execute the trade
                trade_details = execute_trade(
                    symbol=symbol,
                    side=analysis["recommendation"].lower(),
                    qty=qty,
                    stop_loss=analysis["price_targets"].get("stop_loss"),
                    take_profit=analysis["price_targets"].get("take_profit"),
                    trade_type=trade_type,
                )

                if trade_details:
                    logger.info(f"Trade executed successfully: {trade_details}")
                else:
                    logger.error("Failed to execute trade")

            # Send to Discord
            if discord.send_to_discord(analysis, symbol):
                logger.info("Successfully sent to Discord!")
            else:
                logger.error("Failed to send to Discord.")
        else:
            logger.error("Failed to get LLM analysis.")

    except Exception as e:
        logger.error(f"Error analyzing symbol {symbol}: {str(e)}")


def is_market_open(current_time: datetime) -> bool:
    """Check if the market is open"""
    # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
    market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)

    # Check if it's a weekday
    if current_time.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False

    # Check if current time is within market hours
    return market_open <= current_time <= market_close


def main() -> None:
    """Main trading loop"""
    logger.info("Starting trading bot...")
    logger.info(f"Environment: {'PAPER' if PAPER_TRADING else 'LIVE'}")

    while True:
        try:
            current_time = datetime.now()

            # Check if market is open
            if not is_market_open(current_time):
                logger.info("Market is closed. Waiting...")
                time.sleep(CHECK_INTERVAL)
                continue

            # Analyze each symbol
            for symbol in TRADING_STRATEGY["symbols"]:
                analyze_symbol(symbol)

            # Wait for next check
            logger.info(f"Waiting {CHECK_INTERVAL} seconds until next check...")
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
