import os
import pandas as pd
from datetime import datetime, timedelta
import time
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import (
    GetOrdersRequest,
    OrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv
import strategies
import llm
import discord_bot
from logger import setup_logger
from typing import Dict, List, Optional, Any

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

if not API_KEY or not SECRET_KEY:
    raise ValueError("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file")

# Initialize trading client based on environment
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER_TRADING)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


def get_account_info() -> Optional[Dict[str, Any]]:
    """Get account information from Alpaca"""
    logger.info(" . Getting account information")
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
        logger.error(f". Error fetching account info: {str(e)}")
        return None


def get_current_positions() -> List[Dict[str, Any]]:
    """Get current positions from Alpaca"""
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
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    trade_type: str = "day",
) -> Optional[Dict[str, Any]]:
    """Execute a trade with optional stop loss and take profit orders"""
    logger.info(f"Executing {side.upper()} trade for {symbol}")
    logger.info(". Order details:")
    logger.info(f"  . Symbol: {symbol}")
    logger.info(f"  . Side: {side}")
    logger.info(f"  . Quantity: {qty}")
    logger.info(f"  . Stop Loss: {stop_loss}")
    logger.info(f"  . Take Profit: {take_profit}")
    logger.info(f"  . Trade Type: {trade_type}")

    try:
        # Cancel any existing orders for this symbol
        logger.info(". Canceling existing orders")
        try:
            filter = GetOrdersRequest(symbols=[symbol], status="open")
            existing_orders = trading_client.get_orders(filter=filter)
            for order in existing_orders:
                trading_client.cancel_order_by_id(order.id)
        except Exception as e:
            logger.warning(f". Error canceling existing orders: {str(e)}")

        # Get current price to validate stop loss and take profit levels
        try:
            # Get the most recent price using 1-minute bars
            end_date = datetime.now()
            start_date = end_date - timedelta(minutes=1)
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start_date,
                end=end_date,
                limit=1,  # Only get the most recent bar
            )
            bars = data_client.get_stock_bars(request_params)
            current_price = float(bars.df["close"].iloc[-1])
            logger.info(f"  . Current price from latest bar: ${current_price:.4f}")
        except Exception as e:
            logger.error(f". Error getting current price: {str(e)}")
            return None

        # Convert stop loss and take profit to float if they exist
        if stop_loss is not None:
            stop_loss = float(stop_loss)
        if take_profit is not None:
            take_profit = float(take_profit)

        # Create the main order request
        order_request = OrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            type="market",
            time_in_force=TimeInForce.DAY if trade_type == "day" else TimeInForce.GTC,
            order_class="bracket" if (stop_loss or take_profit) else "simple",
        )

        # If we have stop loss or take profit, create a bracket order
        if stop_loss or take_profit:
            # Set stop loss if provided
            if stop_loss is not None:
                # For stop loss orders, we need to set both stop_price and limit_price
                # The stop_price is the trigger price, and limit_price is the execution price
                if side.lower() == "sell":
                    # For SELL orders:
                    # - stop_price must be higher than current price
                    # - limit_price must be higher than stop_price
                    stop_loss_price = round(stop_loss, 2)
                    stop_loss_limit = round(stop_loss + 0.01, 2)
                else:
                    # For BUY orders:
                    # - stop_price must be lower than current price
                    # - limit_price must be lower than stop_price
                    stop_loss_price = round(stop_loss, 2)
                    stop_loss_limit = round(stop_loss - 0.01, 2)

                order_request.stop_loss = {
                    "stop_price": stop_loss_price,
                    "limit_price": stop_loss_limit,
                }

            # Set take profit if provided
            if take_profit is not None:
                take_profit = round(take_profit, 2)
                order_request.take_profit = {
                    "limit_price": take_profit,
                }

        # Submit the order
        logger.info(". Submitting order")
        order = trading_client.submit_order(order_request)

        # Log bracket order details at debug level
        if order.order_class == "bracket" and hasattr(order, "legs") and order.legs:
            logger.debug(". Bracket Order Legs:")
            for i, leg in enumerate(order.legs, 1):
                logger.debug(f"  Leg {i}:")
                logger.debug(f"    . ID: {leg.id}")
                logger.debug(f"    . Type: {leg.type}")
                logger.debug(f"    . Side: {leg.side}")
                logger.debug(f"    . Status: {leg.status}")
                logger.debug(f"    . Limit Price: {leg.limit_price}")
                logger.debug(f"    . Stop Price: {leg.stop_price}")
                logger.debug(f"    . Quantity: {leg.qty}")
                logger.debug(f"    . Filled Quantity: {leg.filled_qty}")
                logger.debug(f"    . Filled Avg Price: {leg.filled_avg_price}")
                logger.debug(f"    . Created At: {leg.created_at}")
                logger.debug(f"    . Updated At: {leg.updated_at}")

        # Wait for the order to fill
        logger.info(". Waiting for order to fill")
        order = trading_client.get_order_by_id(order.id)

        # Log the order response
        logger.info(". Order Response Details:")
        logger.info(f"  . Order ID: {order.id}")
        logger.info(f"  . Status: {order.status}")
        logger.info(f"  . Created At: {order.created_at}")
        logger.info(f"  . Updated At: {order.updated_at}")
        logger.info(f"  . Filled At: {order.filled_at}")
        logger.info(f"  . Filled Avg Price: {order.filled_avg_price}")

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
        if stop_loss is not None or take_profit is not None:
            trade_details["price_targets"] = {"stop_loss": stop_loss, "take_profit": take_profit}

        logger.info(". Trade executed successfully")
        return trade_details

    except Exception as e:
        logger.error(f". Error executing trade: {str(e)}")
        return None


def get_historical_data(
    symbol: str, start_date: datetime, end_date: datetime, timeframe: TimeFrame = TimeFrame.Hour
) -> pd.DataFrame:
    """Get historical data for a symbol"""
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


def analyze_symbol(symbol: str) -> None:
    """Analyze a single symbol and execute trades if needed"""
    logger.info(f"Analyzing {symbol}")
    try:
        # Get account information
        account_info = get_account_info()
        if not account_info:
            logger.error(". Failed to get account information")
            return

        # Check if account has sufficient funds
        if account_info["cash"] <= 0:
            logger.warning(". Insufficient funds in account. Please add funds to your paper trading account.")
            return

        positions = get_current_positions()

        # Check for existing position in this symbol
        existing_position = next((pos for pos in positions if pos["symbol"] == symbol), None)
        if existing_position:
            logger.info(f". Found existing position in {symbol}:")
            logger.info(f"  . Quantity: {existing_position['qty']}")
            logger.info(f"  . Entry Price: ${existing_position['avg_entry_price']:.2f}")
            logger.info(f"  . Current Price: ${existing_position['current_price']:.2f}")
            logger.info(
                f"  . Unrealized P/L: ${existing_position['unrealized_pl']:.2f} ({existing_position['unrealized_plpc']*100:+.2f}%)"
            )

        # Check for existing analysis from this hour
        existing_analysis = llm.get_existing_analysis(symbol)
        if existing_analysis:
            logger.info(". Using existing analysis from this hour")
            analysis = existing_analysis
        else:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=TRADING_STRATEGY["lookback_periods"])
            df = get_historical_data(symbol, start_date, end_date, TRADING_STRATEGY["timeframe"])

            # Calculate indicators
            logger.info(" . Calculating technical indicators")
            df = strategies.calculate_indicators(df)

            # Generate LLM prompt for the current market conditions
            logger.info(" . Generating LLM prompt")
            llm_prompt = strategies.get_llm_prompt(df, account_info, positions)
            if not llm_prompt:
                logger.error(". Failed to generate LLM prompt")
                return

            # Get LLM analysis
            logger.info(" . Getting LLM analysis")
            analysis = llm.get_trading_analysis(llm_prompt)
            if not analysis:
                logger.error(". Failed to get LLM analysis")
                return

            # Save the analysis
            llm.save_analysis(analysis, symbol)

        # Execute trade if recommended
        if analysis["recommendation"] in ["BUY", "SELL"]:
            # Skip if we have a position and the recommendation doesn't match our position
            if existing_position:
                if existing_position["qty"] > 0 and analysis["recommendation"] == "BUY":
                    logger.info(". Skipping BUY recommendation as we already have a long position")
                    return
                elif existing_position["qty"] < 0 and analysis["recommendation"] == "SELL":
                    logger.info(". Skipping SELL recommendation as we already have a short position")
                    return

            # Calculate position size
            position_value = account_info["equity"] * analysis["position_size"]
            trade_type = analysis["trade_type"]

            # Get current price
            try:
                request_params = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    start=datetime.now() - timedelta(minutes=5),
                    end=datetime.now(),
                )
                bars = data_client.get_stock_bars(request_params)
                current_price = float(bars.df["close"].iloc[-1])
            except Exception as e:
                logger.error(f". Error getting current price: {str(e)}")
                return

            qty = int(position_value / current_price)

            # Validate position size
            if qty <= 0:
                logger.warning(f". Calculated position size ({qty}) is invalid. Minimum position size is 1 share.")
                return

            # Check if we have enough buying power
            if position_value > account_info["buying_power"]:
                logger.warning(
                    f". Insufficient buying power. Required: ${position_value:.2f}, Available: ${account_info['buying_power']:.2f}"
                )
                return

            # Execute the trade
            execute_trade(
                symbol=symbol,
                side=analysis["recommendation"],
                qty=qty,
                trade_type=trade_type,
                stop_loss=analysis["price_targets"]["stop_loss"],
                take_profit=analysis["price_targets"]["take_profit"],
            )

        # Send to Discord
        logger.info("Sending analysis to Discord")
        if discord_bot.send_to_discord(analysis, symbol, current_price):
            logger.info(" . Successfully sent to Discord")
        else:
            logger.error(". Failed to send to Discord")

    except Exception as e:
        logger.error(f". Error analyzing symbol {symbol}: {str(e)}")


def is_market_open(current_time: datetime) -> bool:
    """Check if the market is open"""
    logger.info(f" . Checking if market is open at {current_time}")
    # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
    market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)

    # Check if it's a weekday
    if current_time.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        logger.info(". Market is closed (weekend)")
        return False

    # Check if current time is within market hours
    is_open = market_open <= current_time <= market_close
    logger.info(f". Market is {'open' if is_open else 'closed'} (within market hours)")
    return is_open


def display_position_updates() -> None:
    """Display current position updates including P/L and current prices"""
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
    """Main function to run the trading bot"""
    logger.info("Starting trading bot...")

    while True:
        try:
            current_time = datetime.now()
            logger.info(f"Checking market conditions at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Calculate next run time (5 minutes after the next hour)
            next_run = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1, minutes=5)
            wait_seconds = (next_run - current_time).total_seconds()

            # Check if market is open
            if not is_market_open(current_time):
                logger.info("Market is closed. Waiting until next run time...")
                time.sleep(wait_seconds)
                continue

            # Analyze each symbol
            for symbol in TRADING_STRATEGY["symbols"]:
                analyze_symbol(symbol)

            # Display initial position update
            display_position_updates()

            # During the wait time, show position updates every 20 minutes
            wait_start = datetime.now()
            while (datetime.now() - wait_start).total_seconds() < wait_seconds:
                remaining = wait_seconds - (datetime.now() - wait_start).total_seconds()
                logger.info(f"Waiting {remaining:.0f} seconds until next run time...")
                time.sleep(1200)  # Sleep for 20 minutes
                if is_market_open(datetime.now()):  # Only show updates during market hours
                    display_position_updates()

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt. Exiting...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            logger.info("Waiting 60 seconds before retrying...")
            time.sleep(60)


if __name__ == "__main__":
    main()
