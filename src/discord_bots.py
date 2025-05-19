from typing import Dict, Any
from datetime import datetime


def get_recommendation_emoji(recommendation: str) -> str:
    """Get emoji based on trading recommendation."""
    emojis = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}
    return emojis.get(recommendation, "⚪")


def format_trade_message(analysis: Dict[str, Any], symbol: str) -> str:
    """Format trade analysis into a Discord message."""
    # Get emoji based on recommendation
    emoji = get_recommendation_emoji(analysis["recommendation"])

    # Format confidence level with appropriate emoji
    confidence = analysis.get("confidence", 0)
    confidence_emoji = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.4 else "🔴"
    confidence_text = f"{confidence * 100:.1f}%"

    # Format price targets
    stop_loss = analysis["price_targets"]["stop_loss"]
    take_profit = analysis["price_targets"]["take_profit"]

    # Format position size as percentage
    position_size = f"{analysis['position_size'] * 100:.1f}%"

    # Create message
    message = f"""**{emoji} {symbol} {analysis['recommendation']} Signal** {emoji}
{confidence_emoji} Confidence: {confidence_text}

**Analysis:**
{chr(10).join(f"• {reason}" for reason in analysis['reasoning'])}

**Price Targets:**
• Stop Loss: ${stop_loss:.2f}
• Take Profit: ${take_profit:.2f}

**Trade Details:**
• Position Size: {position_size}
• Trade Type: {analysis['trade_type'].title()}

*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"""

    return message


def format_analysis_message(analysis: Dict[str, Any], symbol: str) -> str:
    """Format analysis into a Discord message."""
    # Get emoji based on recommendation
    emoji = get_recommendation_emoji(analysis["recommendation"])

    # Format confidence level with appropriate emoji
    confidence = analysis.get("confidence", 0)
    confidence_emoji = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.4 else "🔴"
    confidence_text = f"{confidence * 100:.1f}%"

    # Create message
    message = f"""**{emoji} {symbol} Analysis** {emoji}
{confidence_emoji} Confidence: {confidence_text}

**Recommendation:** {analysis['recommendation']}

**Analysis:**
{chr(10).join(f"• {reason}" for reason in analysis['reasoning'])}

**Price Targets:**
• Stop Loss: ${analysis['price_targets']['stop_loss']:.2f}
• Take Profit: ${analysis['price_targets']['take_profit']:.2f}

**Trade Details:**
• Position Size: {analysis['position_size'] * 100:.1f}%
• Trade Type: {analysis['trade_type'].title()}

*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"""

    return message
