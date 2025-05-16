import os
import openai
from dotenv import load_dotenv
import json
from datetime import datetime
import re
import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from strategies import format_for_llm

# Get logger from the calling module
logger = logging.getLogger()

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Enable API call logging
os.environ["OPENAI_LOG"] = "debug"


def get_llm_response(
    df: pd.DataFrame,
    account_info: Optional[Dict[str, Any]] = None,
    positions: Optional[List[Dict[str, Any]]] = None,
    lookback_periods: int = 20,
    analysis_date: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get trading analysis from LLM.
    """
    try:
        # Generate prompt
        prompt = get_llm_prompt(df, account_info, positions, lookback_periods, analysis_date)
        if not prompt:
            logger.error("Failed to generate LLM prompt")
            return None

        # Get response from OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert trading analyst. Analyze the market data and provide:
                    1. Trading recommendation (BUY/SELL/HOLD)
                    2. Position size (as a decimal, e.g., 0.01 for 1%)
                    3. Trade type (day/swing)
                    4. Price targets (stop loss and take profit)
                    Format your response as a JSON object with these fields.""",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=500,
        )

        # Parse response
        try:
            analysis = json.loads(response.choices[0].message.content)
            return analysis
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            return None

    except Exception as e:
        logger.error(f"Error getting LLM response: {str(e)}")
        return None


def get_llm_prompt(
    df: pd.DataFrame,
    account_info: Optional[Dict[str, Any]] = None,
    positions: Optional[List[Dict[str, Any]]] = None,
    lookback_periods: int = 20,
    analysis_date: Optional[datetime] = None,
) -> Optional[str]:
    """
    Generate a prompt for the LLM based on the current market conditions.
    Uses 20 periods of historical data by default.
    """
    try:
        if df.empty:
            raise ValueError("No data available for analysis")

        # Ensure we have enough data points
        if len(df) < 5:
            raise ValueError(f"Insufficient data points. Need at least 5 periods, got {len(df)}")

        # Get formatted data
        llm_data = format_for_llm(df, lookback_periods, analysis_date)
        if not llm_data:
            raise ValueError("No valid data points could be processed")

        # Get the most recent data point
        current = llm_data[-1]

        # Calculate support and resistance levels safely
        recent_lows = [period["close"] for period in current["previous_periods"][:10] if "close" in period]
        recent_highs = [period["close"] for period in current["previous_periods"][:10] if "close" in period]

        if not recent_lows or not recent_highs:
            raise ValueError("Insufficient data for support/resistance calculation")

        support_level = min(recent_lows)
        resistance_level = max(recent_highs)

        # Calculate ATR-based price levels
        atr = current["volatility_indicators"]["atr"]
        current_price = current["price_data"]["close"]
        stop_loss_level = current_price - (2 * atr)  # 2 ATR below current price
        take_profit_level = current_price + (4 * atr)  # 4 ATR above current price

        # Format the prompt with clear sections and better readability
        prompt = [
            f"Trading Analysis ({analysis_date.strftime('%Y-%m-%d')})",
            "=" * 40,
            "",
            "Based on the following market data and account information, provide a trading signal (BUY, SELL, or HOLD) and explain your reasoning:",
            "",
            "Account Information:",
        ]

        # Add account information if available
        if account_info:
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
        else:
            prompt.append("- Account information not available")

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
                "Recent Price History (Last 5 Hours):",
            ]
        )

        # Add recent price history (last 5 hours)
        for period in current["previous_periods"][:5]:
            prompt.append(
                f"- {period['timestamp']}: ${period['close']:.2f} "
                f"(RSI: {period['rsi']:.2f}, MACD: {period['macd']:.2f})"
            )

        prompt.extend(
            [
                "",
                "Please provide your analysis and trading recommendation, considering both technical indicators and market context.",
                "Include:",
                "1. Overall market sentiment",
                "2. Key technical signals",
                "3. Risk assessment",
                "4. Clear trading recommendation (BUY/SELL/HOLD)",
                "5. Brief explanation of your reasoning",
                "6. Specific price targets based on support/resistance levels and ATR",
                "7. Position sizing recommendation based on account equity and risk tolerance",
                "",
                "For price targets, use the following format:",
                "- stop_loss: Use either the ATR-based stop loss or the nearest support level",
                "- take_profit: Use either the ATR-based take profit or the nearest resistance level",
                "Example: stop_loss: '150.25', take_profit: '165.50'",
            ]
        )

        return "\n".join(prompt)

    except Exception as e:
        logger.error(f"Error generating LLM prompt: {str(e)}")
        return None


def get_trading_analysis(prompt):
    """
    Get trading analysis from LLM.

    Args:
        prompt (str): The prompt to send to the LLM

    Returns:
        dict: The LLM's analysis and trading recommendation
    """
    try:
        # Extract the analysis date from the prompt
        date_match = re.search(r"Trading Analysis \((\d{4}-\d{2}-\d{2})\)", prompt)
        if not date_match:
            raise ValueError("Could not find analysis date in prompt")

        analysis_date = date_match.group(1)

        # Get response from LLM
        response = get_llm_response(prompt)

        if not response:
            raise ValueError("No response from LLM")

        # Parse the response
        analysis = {
            "date": analysis_date,
            "sentiment": None,
            "recommendation": None,
            "reasoning": [],
            "price_targets": {},
            "position_size": None,
            "trade_type": None,
        }

        # Extract sentiment
        sentiment_match = re.search(r"Market Sentiment:\s*(\w+)", response)
        if sentiment_match:
            analysis["sentiment"] = sentiment_match.group(1).upper()

        # Extract recommendation
        recommendation_match = re.search(r"recommendation:\s*(BUY|SELL|HOLD)", response, re.IGNORECASE)
        if recommendation_match:
            analysis["recommendation"] = recommendation_match.group(1).upper()

        # Extract reasoning
        reasoning_section = re.search(r"Reasoning:(.*?)(?=Price Targets:|$)", response, re.DOTALL)
        if reasoning_section:
            reasoning = reasoning_section.group(1).strip()
            analysis["reasoning"] = [line.strip() for line in reasoning.split("\n") if line.strip()]

        # Extract price targets
        stop_loss_match = re.search(r'stop_loss:\s*[\'"]?(\d+\.?\d*)[\'"]?', response)
        take_profit_match = re.search(r'take_profit:\s*[\'"]?(\d+\.?\d*)[\'"]?', response)

        if stop_loss_match:
            analysis["price_targets"]["stop_loss"] = float(stop_loss_match.group(1))
        if take_profit_match:
            analysis["price_targets"]["take_profit"] = float(take_profit_match.group(1))

        # Extract position size
        position_size_match = re.search(r"position size:\s*(\d+\.?\d*)%", response, re.IGNORECASE)
        if position_size_match:
            analysis["position_size"] = float(position_size_match.group(1)) / 100

        # Extract trade type
        trade_type_match = re.search(r"trade type:\s*(day|swing)", response, re.IGNORECASE)
        if trade_type_match:
            analysis["trade_type"] = trade_type_match.group(1).lower()

        return analysis

    except Exception as e:
        logger.error(f"Error getting trading analysis: {str(e)}")
        return None


def format_analysis_for_display(analysis):
    """
    Format the analysis for display.

    Args:
        analysis (dict): The analysis to format

    Returns:
        str: Formatted analysis
    """
    if not analysis:
        return "No analysis available"

    lines = [
        f"Trading Analysis ({analysis['date']})",
        "=" * 40,
        f"Market Sentiment: {analysis['sentiment']}",
        "",
        f"Recommendation: {analysis['recommendation']}",
        "",
        "Reasoning:",
    ]

    for reason in analysis["reasoning"]:
        lines.append(f"- {reason}")

    if analysis["price_targets"]:
        lines.extend(
            [
                "",
                "Price Targets:",
                f"- Stop Loss: ${analysis['price_targets'].get('stop_loss', 'N/A'):.2f}",
                f"- Take Profit: ${analysis['price_targets'].get('take_profit', 'N/A'):.2f}",
            ]
        )

    if analysis["position_size"]:
        lines.append(f"\nPosition Size: {analysis['position_size']*100:.1f}%")

    if analysis["trade_type"]:
        lines.append(f"Trade Type: {analysis['trade_type'].title()}")

    return "\n".join(lines)


def save_analysis(analysis, symbol, output_dir="analysis"):
    """
    Save the analysis to a JSON file.

    Args:
        analysis (dict): The analysis to save
        symbol (str): The trading symbol
        output_dir (str): Directory to save the analysis
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{symbol}_{timestamp}.json"

        # Save the analysis
        with open(filename, "w") as f:
            json.dump(analysis, f, indent=4)

        logger.info(f"Analysis saved to {filename}")

    except Exception as e:
        logger.error(f"Error saving analysis: {str(e)}")


if __name__ == "__main__":
    # Example usage
    from strategies import get_llm_prompt
    import pandas as pd

    # Load some sample data
    df = pd.read_csv("sample_data.csv", index_col=0, parse_dates=True)

    # Get the prompt
    prompt = get_llm_prompt(df)

    # Get and display the analysis
    analysis = get_trading_analysis(prompt)
    if analysis:
        logger.info(format_analysis_for_display(analysis))
        save_analysis(analysis, "AAPL")
