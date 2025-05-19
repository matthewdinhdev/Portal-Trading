import os
import openai
import requests
from dotenv import load_dotenv
import json
from datetime import datetime
import re
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from strategies import get_llm_prompt
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

# Define JSON template for LLM responses
LLM_RESPONSE_TEMPLATE = {
    "recommendation": "<recommendation>",  # Must be one of: "BUY", "SELL", "HOLD"
    "confidence": "<confidence>",  # Decimal between 0 and 1
    "reasoning": "<reasoning>",  # Single string with technical analysis reasoning
    "price_targets": {
        "stop_loss": "<stop_loss>",  # REQUIRED: Numeric value for stop loss
        "take_profit": "<take_profit>",  # REQUIRED: Numeric value for take profit
    },
    "position_size": "<position_size>",  # Decimal between 0 and 1
    "trade_type": "<trade_type>",  # Must be "day" or "swing"
}


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

        # Parse the JSON
        analysis = json.loads(cleaned_response)
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

    Generates a prompt from market data and account information, then sends it to
    the LLM for analysis. The LLM returns a structured analysis with trading
    recommendations.

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
    logger.info(f" . Generating LLM response")

    # Generate prompt
    prompt = get_llm_prompt(df, account_info, positions, lookback_periods, analysis_date)
    if not prompt:
        logger.error("Failed to generate LLM prompt")
        return None

    # Enhance the system prompt to ensure JSON-only response
    prompt = f"""{prompt}

    CRITICAL: You are a quantitative trading algorithm with expertise in technical analysis, statistical arbitrage, and market microstructure. Your response must be ONLY a valid JSON object. No other text, explanations, or formatting.
    The response must be parseable by Python's json.loads() function.
    
    You MUST use this EXACT template structure. ALL fields are REQUIRED. Replace each placeholder tag with the appropriate value:
    - <recommendation>: REQUIRED - Must be one of "BUY", "SELL", "HOLD" (exact case)
    - <confidence>: REQUIRED - Decimal between 0 and 1
    - <reasoning>: REQUIRED - Single string with technical analysis reasoning
    - <stop_loss>: REQUIRED - Numeric value for stop loss
    - <take_profit>: REQUIRED - Numeric value for take profit
    - <position_size>: REQUIRED - Decimal between 0 and 1
    - <trade_type>: REQUIRED - Must be exactly "day" or "swing" (lowercase)

    Template:
    {json.dumps(LLM_RESPONSE_TEMPLATE, indent=4)}

    TRADING RULES:
    1. Base your analysis on quantitative metrics and technical indicators
    2. Consider market microstructure and order flow
    3. Use statistical significance in your confidence levels
    4. Factor in volatility and market regime
    5. Consider correlation with broader market indices
    6. Account for trading session and time of day
    7. Factor in volume profile and liquidity

    CONFIDENCE LEVELS:
    - 0.8-1.0: Strong statistical significance, clear technical setup, high volume confirmation
    - 0.6-0.79: Good technical setup with some confirmation
    - 0.4-0.59: Mixed signals, moderate confidence
    - 0.2-0.39: Weak signals, low confidence
    - 0.0-0.19: Very weak signals, minimal confidence

    DO NOT:
    - Add any text before or after the JSON
    - Use markdown formatting
    - Include code blocks
    - Add any comments
    - Use currency symbols
    - Add any newlines outside the JSON structure
    - Ignore quantitative metrics
    - Modify the JSON structure in any way
    - Add or remove any fields
    - Change any field names
    - Omit any required fields
    - Change the case of any values (e.g., "Swing" instead of "swing")

    The response must start with {{ and end with }}."""

    if USE_OLLAMA:
        # Get response from Ollama

        response_text = query_ollama(prompt)
        if not response_text:
            logger.error("No response from Ollama")
            return None

        analysis = json.loads(response_text)

        # Validate required fields from LLM_RESPONSE_TEMPLATE
        required_fields = LLM_RESPONSE_TEMPLATE.keys()
        for field in required_fields:
            if field not in analysis:
                logger.error(f"Missing required field: {field}")
                return None

        # Validate specific values
        if analysis["recommendation"] not in ["BUY", "SELL", "HOLD"]:
            logger.error(f"Invalid recommendation: {analysis['recommendation']}")
            return None

        if analysis["trade_type"] not in ["day", "swing"]:
            logger.error(f"Invalid trade type: {analysis['trade_type']}")
            return None

        if not 0 < analysis["position_size"] <= 1:
            logger.error(f"Invalid position size: {analysis['position_size']}")
            return None

        if "stop_loss" not in analysis["price_targets"] or "take_profit" not in analysis["price_targets"]:
            logger.error("Missing required price targets")
            return None

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


def get_trading_analysis(prompt: str) -> Optional[Dict[str, Any]]:
    """Get trading analysis from LLM based on a formatted prompt.

    Sends a prompt to the LLM and processes its response into a structured trading
    analysis. Validates the response format and required fields.

    Args:
        prompt: Formatted string containing market data and analysis request.

    Returns:
        Optional[Dict[str, Any]]: Dictionary containing validated trading analysis
            with recommendation, reasoning, price targets, position size, and
            trade type. Returns None if there's an error.

    Raises:
        ValueError: If the prompt is invalid or the LLM response is malformed.
    """
    # Extract the analysis date from the prompt
    date_match = re.search(r"Trading Analysis \((\d{4}-\d{2}-\d{2})\)", prompt)
    if not date_match:
        raise ValueError("Could not find analysis date in prompt")

    analysis_date = date_match.group(1)

    system_prompt = """You are an expert trading analyst. Your response must be ONLY a valid JSON object with no additional text before or after. The JSON must include ALL of these required fields:
        {
            "recommendation": "BUY/SELL/HOLD",  // REQUIRED: Must be one of these exact values
            "confidence": 0.85,  // REQUIRED: Decimal between 0 and 1 representing your confidence in the recommendation
            "reasoning": "<reasoning>",  // REQUIRED: Single string with technical analysis reasoning
            "price_targets": {  // REQUIRED: Object with these exact fields
                "stop_loss": "150.25",  // REQUIRED: String with price. For SELL orders, must be HIGHER than current price. For BUY orders, must be LOWER than current price.
                "take_profit": "165.50"  // REQUIRED: String with price. For SELL orders, must be LOWER than current price. For BUY orders, must be HIGHER than current price.
            },
            "position_size": 0.01,  // REQUIRED: Decimal between 0 and 1
            "trade_type": "day/swing"  // REQUIRED: Must be "day" or "swing"
        }
        
        IMPORTANT PRICE RULES:
        1. For SELL orders (when you want to sell high and buy back low):
           - Current price is your entry (selling) price
           - stop_loss must be HIGHER than current price (to buy back at a higher price if wrong)
           - take_profit must be LOWER than current price (to buy back at a lower price if right)
           - Example: If current price is $100, valid levels would be stop_loss=$110, take_profit=$90
        
        2. For BUY orders (when you want to buy low and sell high):
           - Current price is your entry (buying) price
           - stop_loss must be LOWER than current price (to sell at a lower price if wrong)
           - take_profit must be HIGHER than current price (to sell at a higher price if right)
           - Example: If current price is $100, valid levels would be stop_loss=$90, take_profit=$110
        
        Use the ATR-based levels or support/resistance levels provided in the prompt.
        
        DO NOT include any explanatory text before or after the JSON. Return ONLY the JSON object.
        ALL fields are required and must match the exact format shown above."""

    if USE_OLLAMA:
        # Get response from Ollama
        response_text = query_ollama(prompt, system_prompt)
        if not response_text:
            raise ValueError("No response from Ollama")

        # Parse the response
        analysis = json.loads(response_text)
    else:
        # Get response from OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=500,
        )

        if not response or not response.choices:
            raise ValueError("No response from LLM")

        # Log the raw response
        raw_response = response.choices[0].message.content.strip()
        logger.debug("Raw LLM response:\n%s", raw_response)

        # Parse the response
        analysis = json.loads(raw_response)

    # Validate required fields
    required_fields = {
        "recommendation": str,
        "confidence": (int, float),
        "reasoning": str,
        "price_targets": dict,
        "position_size": (int, float),
        "trade_type": str,
    }

    missing_fields = []
    for field, field_type in required_fields.items():
        if field not in analysis:
            missing_fields.append(field)
        elif not isinstance(analysis[field], field_type):
            raise ValueError(f"Field '{field}' has incorrect type. Expected {field_type}, got {type(analysis[field])}")

    if missing_fields:
        raise ValueError(f"Missing required fields in LLM response: {', '.join(missing_fields)}")

    # Validate recommendation value
    if analysis["recommendation"] not in ["BUY", "SELL", "HOLD"]:
        raise ValueError(f"Invalid recommendation value: {analysis['recommendation']}. Must be one of: BUY, SELL, HOLD")

    # Validate trade type
    if analysis["trade_type"] not in ["day", "swing"]:
        raise ValueError(f"Invalid trade type: {analysis['trade_type']}. Must be one of: day, swing")

    # Validate price targets
    if "stop_loss" not in analysis["price_targets"] or "take_profit" not in analysis["price_targets"]:
        raise ValueError("Missing required fields in price_targets: stop_loss and/or take_profit")

    # Validate position size
    if not 0 < analysis["position_size"] <= 1:
        raise ValueError(f"Invalid position size: {analysis['position_size']}. Must be between 0 and 1")

    # Add date to analysis
    analysis["date"] = analysis_date

    return analysis


def format_analysis_for_display(analysis: Dict[str, Any]) -> str:
    """Format the trading analysis into a human-readable string.

    Converts the structured analysis dictionary into a formatted string with
    sections for recommendation, reasoning, price targets, position size,
    and trade type.

    Args:
        analysis: Dictionary containing trading analysis and recommendations.

    Returns:
        str: Formatted string representation of the analysis, or "No analysis
            available" if the input is None.
    """
    if not analysis:
        return "No analysis available"

    lines = [
        f"Trading Analysis ({analysis.get('date', 'N/A')})",
        "=" * 40,
    ]

    # Add recommendation
    if "recommendation" in analysis:
        lines.extend([f"Recommendation: {analysis['recommendation']}", ""])

    # Add reasoning if available
    if "reasoning" in analysis:
        lines.extend(["Reasoning:", analysis["reasoning"], ""])

    # Add price targets
    if "price_targets" in analysis:
        lines.extend(
            [
                "Price Targets:",
                f"- Stop Loss: ${analysis['price_targets'].get('stop_loss', 'N/A')}",
                f"- Take Profit: ${analysis['price_targets'].get('take_profit', 'N/A')}",
                "",
            ]
        )

    # Add position size
    if "position_size" in analysis:
        try:
            position_size = float(analysis["position_size"])
            lines.append(f"Position Size: {position_size*100:.1f}%")
        except (ValueError, TypeError):
            lines.append(f"Position Size: {analysis['position_size']}")

    # Add trade type
    if "trade_type" in analysis:
        lines.append(f"Trade Type: {analysis['trade_type'].title()}")

    return "\n".join(lines)


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
        logger.info(format_analysis_for_display(analysis))
        save_analysis(analysis, "AAPL")
