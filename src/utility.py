import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

import llm
import strategies
from trading_enums import TradingEnvironment

# Load environment variables
load_dotenv()

# This logger will inherit all settings from the root logger
logger = logging.getLogger(__name__)

# Initialize Alpaca clients
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file")

# Trading Strategy Parameters
TRADING_STRATEGY = {
    "timeframe": TimeFrame.Day,
    "lookback_periods": 365,  # 6 months of trading days
}


def get_historical_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    logger.info(f" . Getting historical data for {symbol}")  # This will go to both file and console

    data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    # Convert ET times to UTC for the API request
    start_date_utc = start_date.astimezone(timezone.utc)

    # Set end date to previous day's close (16:00 ET = 20:00 UTC)
    end_date_utc = end_date.astimezone(timezone.utc).replace(hour=20, minute=0, second=0, microsecond=0)

    # Get historical data
    logger.info(f"   . Start date: {start_date_utc}")
    logger.info(f"   . End date: {end_date_utc}")
    logger.info(f"   . Timeframe: {TRADING_STRATEGY['timeframe']}")
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TRADING_STRATEGY["timeframe"], start=start_date_utc, end=end_date_utc
    )
    bars = data_client.get_stock_bars(request_params)
    df = bars.df

    # Reset multi-index to single index and ensure it's a datetime index
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    logger.info(f" . Retrieved {len(df)} data points for {symbol}")
    return df


def analyze_symbol(
    symbol: str, data: pd.DataFrame, env: TradingEnvironment, save_analysis: bool = True
) -> Optional[Dict[str, Any]]:
    """Analyze a single symbol and return analysis results.

    Args:
        symbol: The trading symbol to analyze
        data: DataFrame containing market data and indicators
        env: Trading environment (backtest, paper, live)
        save_analysis: Whether to save the analysis to a file

    Returns:
        Optional[Dict[str, Any]]: Analysis results if there's a trading recommendation
    """
    logger.info(f"Analyzing {symbol}")

    # create llm instance
    llm_instance = llm.LLMAnalyzer(env=env)

    # Get timestamp from data
    timestamp = data.index[-1]

    # Check for existing analysis from this hour
    existing_analysis = llm_instance.get_existing_analysis(symbol, timestamp.strftime("%Y-%m-%d %H:%M"))
    if existing_analysis:
        analysis = existing_analysis
    else:
        # Calculate indicators and get analysis
        df = strategies.calculate_indicators(data)
        df.name = symbol  # Set the DataFrame's name attribute

        analysis = None
        error = None
        MAX_RETRIES = 3
        for i in range(MAX_RETRIES):
            try:
                analysis = llm_instance.get_llm_response(df, env=env)
                if analysis:
                    break
            except Exception as e:
                error = e
                logger.warning(f"Could not get analysis for {symbol} after {i+1} retries")
                logger.warning(f"Error: {error}")
        if not analysis:
            logger.error(f"Error: {error}")
            raise Exception(f"Failed to get analysis for {symbol} after {MAX_RETRIES} retries: {error}")

        if save_analysis:
            llm_instance.save_analysis(analysis, symbol, timestamp)

    return {
        "symbol": symbol,
        "analysis": analysis,
    }


def calculate_position_size(confidence: float, available_cash: float, current_price: float) -> int:
    """Calculate the position size in number of shares based confidence level

    Args:
        confidence: Confidence level of the trade
        available_cash: Total available cash
        current_price: Current price of the asset

    Returns:
        int: Number of shares to trade (minimum 1 share)
    """
    # Calculate position value and number of shares
    POSITION_SIZE_PCT = confidence
    position_value = available_cash * POSITION_SIZE_PCT
    shares = int(position_value / current_price)

    # Ensure minimum position size of 1 share
    return max(1, shares)
