import os
import openai
import requests
from dotenv import load_dotenv
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from logger import setup_logger
from main import PAPER_TRADING

# Get logger from the calling module
logger = logging.getLogger()

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up logger
logger = setup_logger("paper_trading.log")

# Flag to switch between OpenAI and Ollama
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-llm:7b")

# Define example template for LLM responses
LLM_EXAMPLE_TEMPLATE = {
    "recommendation": "BUY",  # String: Must be exactly "BUY", "SELL", or "HOLD"
    "confidence": 0.85,  # Number: Must be a single decimal between 0 and 1
    "reasoning": "Strong bullish momentum with increasing volume and positive RSI divergence. Price action shows higher lows forming a potential reversal pattern.",  # String: Technical analysis reasoning
    "price_targets": {  # Object: Contains stop loss and take profit
        "stop_loss": 194.00,  # Number: Must be a positive number without $ symbol
        "take_profit": 201.65,  # Number: Must be a positive number without $ symbol
    },
    "position_size": 0.35,  # Number: Must be a decimal between 0 and 1
    "trade_type": "DAY",  # String: Must be exactly "DAY" or "SWING"
}


def format_for_llm(
    df: pd.DataFrame, lookback_periods: int = 48, analysis_date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """Format market data for LLM consumption.

    Creates a structured representation of market data including price,
    technical indicators, and market context for a specific time period.

    Args:
        df: DataFrame containing market data and technical indicators.
        lookback_periods: Number of periods to look back (default: 48).
        analysis_date: Optional specific date to analyze (default: None).

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing formatted
            market data and indicators.

    Raises:
        ValueError: If insufficient data is available for the analysis.
    """
    # If analysis_date is provided, find the index for that date
    if analysis_date is not None:
        if isinstance(analysis_date, datetime):
            analysis_date = analysis_date.date()

        # Get all data up to and including the analysis date
        df = df[df.index.date <= analysis_date]

        if df.empty:
            raise ValueError(f"No data available for analysis date: {analysis_date}")

    # Ensure we have enough data points
    if len(df) < 5:
        logger.warning(f"Insufficient data points. Need at least 5 periods, got {len(df)}")
        return []

    # Create a list to store our formatted data
    llm_data: List[Dict[str, Any]] = []

    # Get the last data point for analysis
    try:
        i = len(df) - 1  # Use the last data point

        current_price = df["close"].iloc[i]
        price_change_1h = float(df["returns"].iloc[i])

        # Calculate multi-period changes safely
        price_change_5h = 0.0
        price_change_20h = 0.0
        price_change_48h = 0.0
        if i >= 5:
            price_change_5h = float(df["close"].iloc[i] / df["close"].iloc[i - 5] - 1)
        if i >= 20:
            price_change_20h = float(df["close"].iloc[i] / df["close"].iloc[i - 20] - 1)
        if i >= 48:
            price_change_48h = float(df["close"].iloc[i] / df["close"].iloc[i - 48] - 1)

        current_data = {
            "timestamp": df.index[i].strftime("%Y-%m-%d %H:%M"),
            "price_data": {
                "open": float(df["open"].iloc[i]),
                "high": float(df["high"].iloc[i]),
                "low": float(df["low"].iloc[i]),
                "close": float(current_price),
                "volume": float(df["volume"].iloc[i]),
                "price_changes": {
                    "1h": price_change_1h,
                    "5h": price_change_5h,
                    "20h": price_change_20h,
                    "48h": price_change_48h,
                },
            },
            "trend_indicators": {
                "sma_20": float(df["SMA_20"].iloc[i]),
                "sma_50": float(df["SMA_50"].iloc[i]),
                "sma_200": float(df["SMA_200"].iloc[i]),
                "ema_20": float(df["EMA_20"].iloc[i]),
                "ema_50": float(df["EMA_50"].iloc[i]),
            },
            "momentum_indicators": {
                "rsi": float(df["RSI"].iloc[i]),
                "macd": float(df["MACD"].iloc[i]),
                "macd_signal": float(df["MACD_signal"].iloc[i]),
                "macd_hist": float(df["MACD_hist"].iloc[i]),
                "stochastic_k": float(df["%K"].iloc[i]),
                "stochastic_d": float(df["%D"].iloc[i]),
            },
            "volatility_indicators": {
                "bollinger_upper": float(df["BB_upper"].iloc[i]),
                "bollinger_middle": float(df["BB_middle"].iloc[i]),
                "bollinger_lower": float(df["BB_lower"].iloc[i]),
                "atr": float(df["ATR"].iloc[i]),
                "volatility": float(df["volatility"].iloc[i]),
            },
            "volume_indicators": {
                "volume_ma_20": float(df["volume_ma_20"].iloc[i]),
                "volume_ratio": float(df["volume_ratio"].iloc[i]),
            },
            "market_context": {
                "price_change_1h": price_change_1h,
                "price_change_5h": price_change_5h,
                "price_change_20h": price_change_20h,
                "price_change_48h": price_change_48h,
                "momentum": float(df["momentum"].iloc[i]),
                "rate_of_change": float(df["rate_of_change"].iloc[i]),
            },
            "historical_context": {
                "price_trend": "up" if df["close"].iloc[i] > df["SMA_20"].iloc[i] else "down",
                "volume_trend": (
                    "high"
                    if df["volume_ratio"].iloc[i] > 1.5
                    else "low"
                    if df["volume_ratio"].iloc[i] < 0.5
                    else "normal"
                ),
                "volatility_state": (
                    "high" if df["volatility"].iloc[i] > df["volatility"].rolling(20).mean().iloc[i] else "low"
                ),
                "rsi_state": (
                    "overbought" if df["RSI"].iloc[i] > 70 else "oversold" if df["RSI"].iloc[i] < 30 else "neutral"
                ),
                "trend_strength": (
                    "strong" if abs(price_change_48h) > 0.1 else "moderate" if abs(price_change_48h) > 0.05 else "weak"
                ),
            },
        }

        # Add previous periods' data for context
        current_data["previous_periods"] = []
        for j in range(1, min(lookback_periods + 1, i + 1)):
            prev_data = {
                "timestamp": df.index[i - j].strftime("%Y-%m-%d %H:%M"),
                "close": float(df["close"].iloc[i - j]),
                "volume": float(df["volume"].iloc[i - j]),
                "rsi": float(df["RSI"].iloc[i - j]),
                "macd": float(df["MACD"].iloc[i - j]),
            }
            current_data["previous_periods"].append(prev_data)

        llm_data.append(current_data)

    except Exception as e:
        logger.error(f"Error processing data point: {str(e)}")
        logger.error(f"DataFrame info:\n{df.info()}")
        logger.error(f"DataFrame head:\n{df.head()}")
        raise ValueError(f"Failed to process data: {str(e)}")

    if not llm_data:
        logger.warning("No valid data points could be processed")
        return []

    return llm_data


def generate_llm_strategy_prompt(
    df: pd.DataFrame,
    account_info: Optional[Dict[str, Any]] = None,
    positions: Optional[List[Dict[str, Any]]] = None,
    lookback_periods: int = 48,
    analysis_date: Optional[datetime] = None,
) -> Optional[str]:
    """Generate a prompt for the LLM based on market data and account information.

    Creates a comprehensive prompt including market data, technical indicators,
    account information, and current positions for the LLM to analyze.

    Args:
        df: DataFrame containing market data and technical indicators.
        account_info: Optional dictionary containing account information.
        positions: Optional list of current positions.
        lookback_periods: Number of periods to look back (default: 48).
        analysis_date: Optional specific date to analyze (default: None).

    Returns:
        Optional[str]: Formatted prompt string for the LLM, or None if there's an error.
    """
    try:
        logger.info(" . Generating LLM prompt")

        if df.empty:
            logger.error("Empty DataFrame provided")
            raise ValueError("No data available for analysis")

        # Get formatted data first
        llm_data = format_for_llm(df, lookback_periods, analysis_date)

        if not llm_data:
            logger.warning("No valid data points could be processed")
            return None

        # Get the most recent data point
        current = llm_data[-1]

        # Calculate support and resistance levels safely
        recent_lows = [period["close"] for period in current["previous_periods"][:24] if "close" in period]
        recent_highs = [period["close"] for period in current["previous_periods"][:24] if "close" in period]

        if not recent_lows or not recent_highs:
            logger.warning("Insufficient data for support/resistance calculation")
            return None

        support_level = min(recent_lows)
        resistance_level = max(recent_highs)

        # Calculate ATR-based price levels
        atr = current["volatility_indicators"]["atr"]
        current_price = current["price_data"]["close"]
        stop_loss_level = current_price - (2 * atr)  # 2 ATR below current price
        take_profit_level = current_price + (4 * atr)

        # Format the prompt with clear sections and better readability
        prompt = [
            f"Trading Analysis ({analysis_date.strftime('%Y-%m-%d') if analysis_date else datetime.now().strftime('%Y-%m-%d')})",
            "=" * 40,
            "Account Information:",
        ]

        # Add account information if available
        if not account_info:
            raise ValueError("Account information not available")
        prompt.extend(
            [
                f"- Portfolio Value: ${account_info['portfolio_value']:,.2f}",
                f"- Available Cash: ${account_info['cash']:,.2f}",
                f"- Buying Power: ${account_info['buying_power']:,.2f}",
                f"- Day Trade Count: {account_info['daytrade_count']}",
                f"- Pattern Day Trader: {'Yes' if account_info['pattern_day_trader'] else 'No'}",
                f"- Shorting Enabled: {'Yes' if account_info['shorting_enabled'] else 'No'}",
                f"- Account Status: {account_info['status']}",
            ]
        )

        # Add current positions if available
        if positions:
            prompt.extend(["", "Current Positions:"])
            for position in positions:
                prompt.extend(
                    [
                        f"- {position['symbol']}:",
                        f"  Quantity: {position['qty']}",
                        f"  Average Entry: ${position['avg_entry_price']:.2f}",
                        f"  Current Price: ${position['current_price']:.2f}",
                        f"  Market Value: ${position['market_value']:,.2f}",
                        f"  Unrealized P/L: ${position['unrealized_pl']:,.2f} ({position['unrealized_plpc']*100:+.2f}%)",
                        f"  Today's Change: {position['change_today']*100:+.2f}%",
                    ]
                )
        else:
            prompt.append("- No current positions")

        # Add market data
        prompt.extend(
            [
                "",
                f"Current Market Conditions ({current['timestamp']}):",
                f"- Price: ${current['price_data']['close']:.2f}",
                f"- Hourly Change: {current['price_data']['price_changes']['1h']*100:+.2f}%",
                f"- 5-Hour Change: {current['price_data']['price_changes']['5h']*100:+.2f}%",
                f"- 20-Hour Change: {current['price_data']['price_changes']['20h']*100:+.2f}%",
                f"- 48-Hour Change: {current['price_data']['price_changes']['48h']*100:+.2f}%",
                "",
                "Price Levels:",
                f"- Support Level: ${support_level:.2f}",
                f"- Resistance Level: ${resistance_level:.2f}",
                f"- ATR-Based Stop Loss: ${stop_loss_level:.2f}",
                f"- ATR-Based Take Profit: ${take_profit_level:.2f}",
                "",
                "Technical Indicators:",
                f"- RSI: {current['momentum_indicators']['rsi']:.2f}",
                f"- MACD: {current['momentum_indicators']['macd']:.2f}",
                f"- MACD Signal: {current['momentum_indicators']['macd_signal']:.2f}",
                f"- MACD Histogram: {current['momentum_indicators']['macd_hist']:.2f}",
                f"- Stochastic K: {current['momentum_indicators']['stochastic_k']:.2f}",
                f"- Stochastic D: {current['momentum_indicators']['stochastic_d']:.2f}",
                f"- Volume Ratio: {current['volume_indicators']['volume_ratio']:.2f}",
                f"- Volatility: {current['volatility_indicators']['volatility']:.2f}",
                "",
                "Market Context:",
                f"- Price Trend: {current['historical_context']['price_trend']}",
                f"- Volume Trend: {current['historical_context']['volume_trend']}",
                f"- Volatility State: {current['historical_context']['volatility_state']}",
                f"- RSI State: {current['historical_context']['rsi_state']}",
                f"- Trend Strength: {current['historical_context']['trend_strength']}",
                "",
                "Recent Price History (Last 48 Hours):",
            ]
        )

        # Add recent price history (last 48 hours)
        for period in current["previous_periods"][:48]:
            prompt.append(
                f"- {period['timestamp']}: ${period['close']:.2f} "
                f"(RSI: {period['rsi']:.2f}, MACD: {period['macd']:.2f})"
            )

        final_prompt = "\n".join(prompt)
        return final_prompt

    except Exception as e:
        logger.error(f"Error generating LLM prompt: {str(e)}")
        logger.error(f"DataFrame info:\n{df.info()}")
        logger.error(f"DataFrame head:\n{df.head()}")
        return None


def query_ollama(prompt: str) -> Optional[str]:
    """Query Ollama API for trading analysis.

    Args:
        prompt: The user prompt containing market data
        system_prompt: The system prompt defining the response format

    Returns:
        Optional[str]: The model's response or None if there's an error
    """
    logger.info(" . Querying Ollama")
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Lower temperature for more consistent JSON
        },
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    # Get the raw response
    raw_response = response.json()["response"]
    logger.debug(f"Raw response: {raw_response}")

    try:
        # Clean the response string
        # Remove any control characters
        cleaned_response = "".join(char for char in raw_response if ord(char) >= 32 or char in "\n\r\t")

        return cleaned_response
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from LLM: {str(e)}")
        return None


def get_llm_response(
    df: pd.DataFrame,
    account_info: Optional[Dict[str, Any]] = None,
    positions: Optional[List[Dict[str, Any]]] = None,
    lookback_periods: int = 20,
    analysis_date: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """Get trading analysis from LLM based on market data and account information.

    Args:
        df: DataFrame containing market data.
        account_info: Optional dictionary containing account information.
        positions: Optional list of dictionaries containing current positions.
        lookback_periods: Number of periods to look back for analysis (default: 20).
        analysis_date: Optional datetime for the analysis (default: current time).

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing LLM's trading analysis and
            recommendations, or None if there's an error.
    """
    logger.info(" . Generating LLM response")

    # Generate prompt
    stat_info = generate_llm_strategy_prompt(df, account_info, positions, lookback_periods, analysis_date)
    if not stat_info:
        logger.error("Failed to generate stat information")
        return None

    # Enhance the system prompt to ensure JSON-only response
    prompt = f"""
    CRITICAL: You are a quantitative trading algorithm with expertise in technical analysis, statistical arbitrage, and market microstructure. 

    Your response MUST use this EXACT structure with these EXACT types:
    {{
        "recommendation": "BUY",  // String: Must be exactly "BUY", "SELL", or "HOLD"
        "confidence": 0.85,       // Number: Must be a single decimal between 0 and 1
        "reasoning": "string",    // String: Technical analysis reasoning
        "price_targets": {{       // Object: Contains stop loss and take profit
            "stop_loss": 194.00,  // Number: Must be a positive number without $ symbol
            "take_profit": 201.65 // Number: Must be a positive number without $ symbol
        }},
        "position_size": 0.35,    // Number: Must be a decimal between 0 and 1
        "trade_type": "DAY"       // String: Must be exactly "DAY" or "SWING"
    }}

    DO NOT:
    - Add any text outside the JSON
    - Add comments or currency symbols
    - Modify the JSON structure or field names
    - Change value cases (e.g., "Swing" vs "SWING")

    TRADING RULES:
    1. Base your analysis on quantitative metrics and technical indicators
    2. Consider market microstructure and order flow
    3. Use statistical significance in your confidence levels
    4. Factor in volatility and market regime
    5. Consider correlation with broader market indices
    6. Account for trading session and time of day
    7. Factor in volume profile and liquidity

    POSITION SIZING CONSIDERATIONS:
    - Position size should be determined by your confidence in the trade and current market conditions
    - Consider the current portfolio value and existing positions
    - Factor in market volatility and liquidity
    - Ensure position size aligns with your confidence level
    - Consider risk management and portfolio diversification
    - Account for the trade type (day vs swing)

    {stat_info}
    """

    if USE_OLLAMA:
        # Get response from Ollama
        response_text = query_ollama(prompt)
        if not response_text:
            logger.error("No response from Ollama")
            return None

        analysis = json.loads(response_text)
        analysis = prompt_validation_and_formatting(analysis)
        logger.info(analysis)

        return analysis
    else:
        # Get response from OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=500,
        )

        analysis = json.loads(response.choices[0].message.content)
        return analysis


def prompt_validation_and_formatting(analysis: Dict[str, Any]) -> Optional[str]:
    """Validate and format the prompt for the LLM.

    Args:
        prompt: The user prompt containing market data

    Returns:
        Optional[str]: The formatted prompt or None if there's an error
    """
    # Validate required fields from LLM_EXAMPLE_TEMPLATE
    required_fields = LLM_EXAMPLE_TEMPLATE.keys()
    for field in required_fields:
        if field not in analysis:
            logger.error(f"Missing required field: {field}")
            return None

    # format if necessary
    # remove $ from price targets
    analysis["price_targets"]["take_profit"] = float(str(analysis["price_targets"]["take_profit"]).replace("$", ""))
    analysis["price_targets"]["stop_loss"] = float(str(analysis["price_targets"]["stop_loss"]).replace("$", ""))

    # remove % from position size
    analysis["position_size"] = float(str(analysis["position_size"]).replace("%", ""))

    # fix capitalization
    analysis["recommendation"] = analysis["recommendation"].upper()
    analysis["trade_type"] = analysis["trade_type"].upper()

    # Validate specific values
    if analysis["recommendation"] not in ["BUY", "SELL", "HOLD"]:
        logger.error(f"Invalid recommendation: {analysis['recommendation']}")
        return None

    if analysis["trade_type"] not in ["DAY", "SWING"]:
        logger.error(f"Invalid trade type: {analysis['trade_type']}")
        return None

    if not 0 < analysis["position_size"] <= 1:
        logger.error(f"Invalid position size: {analysis['position_size']}. Must be between 0 and 1")
        return None

    if "stop_loss" not in analysis["price_targets"] or "take_profit" not in analysis["price_targets"]:
        logger.error("Missing required price targets")
        return None

    return analysis


def save_analysis(analysis: Dict[str, Any], symbol: str) -> None:
    """Save trading analysis to a JSON file.

    Creates a timestamped JSON file containing the trading analysis in the
    appropriate analysis directory.

    Args:
        analysis: Dictionary containing trading analysis and recommendations.
        symbol: Trading symbol (e.g., 'AAPL') for the analysis.
    """
    # Create base directory based on trading mode
    base_dir = "analysis/paper_trading" if PAPER_TRADING else "analysis/live_trading"

    # Create date-based subfolder (YYYY-MM-DD format)
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(base_dir, date_str)

    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create filename with hourly timestamp (floor to hour)
    timestamp = datetime.now().strftime("%Y%m%d_%H00")
    filename = f"{symbol}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Save the analysis
    with open(filepath, "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info(f" . Analysis saved to {filepath}")


def get_existing_analysis(symbol: str, output_dir: str = "analysis") -> Optional[Dict[str, Any]]:
    """Get existing analysis for a symbol from the most recent JSON file.

    Searches for the most recent analysis file for the given symbol in the
    specified directory.

    Args:
        symbol: Trading symbol (e.g., 'AAPL') to get analysis for.
        output_dir: Directory to search for analysis files (default: "analysis").

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing the most recent analysis
            for the symbol, or None if no analysis is found.
    """
    # Determine the output directory based on trading mode
    from main import PAPER_TRADING

    if PAPER_TRADING:
        output_dir = os.path.join(output_dir, "paper_trading")
    else:
        output_dir = os.path.join(output_dir, "live_trading")

    # Create date-based subfolder (YYYY-MM-DD format)
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(output_dir, date_str)

    # Create filename with hourly timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H00")
    filename = f"{symbol}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Check if file exists
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            analysis = json.load(f)
        logger.info(f" . Found existing analysis for {symbol} from {timestamp}")
        return analysis

    logger.debug(f" . No existing analysis found for {symbol} at {filepath}")
    return None


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Load some sample data
    df = pd.read_csv("sample_data.csv", index_col=0, parse_dates=True)

    # Get and display the analysis
    analysis = get_llm_response(df)
    if analysis:
        save_analysis(analysis, "AAPL")
