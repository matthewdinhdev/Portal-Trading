import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from trading_enums import TradingEnvironment

# This logger will inherit all settings from the root logger
logger = logging.getLogger(__name__)


class LLMAnalyzer:
    # Private class variables
    _volatility_window = 20
    _volatility_threshold = 1.0
    _rsi_overbought = 70
    _rsi_oversold = 30
    _trend_strong = 0.1
    _trend_moderate = 0.05
    _model_backtest = "deepseek-llm:7b"
    # _model_backtest = "deepseek-r1:14b"
    _model_live = "deepseek-r1:14b"
    _historical_window = 48

    def __init__(self, env: TradingEnvironment):
        """Initialize the LLM Analyzer."""
        self.env = env

    def calculate_pivot_points(self, df: pd.DataFrame) -> Tuple[float, float, float, float, float, float, float]:
        """Calculate pivot points and support/resistance levels.

        Args:
            df: DataFrame with OHLC data

        Returns:
            Tuple of (Pivot, R1, R2, R3, S1, S2, S3)
        """
        # Get previous day's data
        prev_high = df["high"].iloc[-2]
        prev_low = df["low"].iloc[-2]
        prev_close = df["close"].iloc[-2]

        # Calculate pivot point
        pivot = (prev_high + prev_low + prev_close) / 3

        # Calculate support and resistance levels
        r1 = (2 * pivot) - prev_low
        s1 = (2 * pivot) - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)

        return pivot, r1, r2, r3, s1, s2, s3

    def find_nearest_level(self, price: float, levels: List[float]) -> float:
        """Find the nearest support/resistance level to the current price.

        Args:
            price: Current price
            levels: List of support/resistance levels

        Returns:
            Nearest level to the current price
        """
        return min(levels, key=lambda x: abs(x - price))

    def format_for_llm(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Format market data for LLM analysis."""
        llm_data: List[Dict[str, Any]] = []

        try:
            i = len(df) - 1
            current_price = df["close"].iloc[i]

            # Calculate multi-period changes
            price_change_1w = 0.0
            price_change_2w = 0.0
            price_change_1m = 0.0
            price_change_6m = 0.0

            if i >= 5:
                price_change_1w = float(df["close"].iloc[i] / df["close"].iloc[i - 5] - 1)
            if i >= 10:
                price_change_2w = float(df["close"].iloc[i] / df["close"].iloc[i - 10] - 1)
            if i >= 21:
                price_change_1m = float(df["close"].iloc[i] / df["close"].iloc[i - 21] - 1)
            if i >= 126:
                price_change_6m = float(df["close"].iloc[i] / df["close"].iloc[i - 126] - 1)
            else:
                logger.error(f"Not enough data points for 6 month change. Need 126 points, have {i + 1}")

            current_data = {
                "timestamp": df.index[i].strftime("%Y-%m-%d %H:%M"),
                "price_data": {
                    "open": float(df["open"].iloc[i]),
                    "high": float(df["high"].iloc[i]),
                    "low": float(df["low"].iloc[i]),
                    "close": float(current_price),
                    "volume": float(df["volume"].iloc[i]),
                    "price_changes": {
                        "1w": price_change_1w,
                        "2w": price_change_2w,
                        "1m": price_change_1m,
                        "6m": price_change_6m,
                    },
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
                        "high"
                        if df["volatility"].iloc[i] > df["volatility"].rolling(self._volatility_window).mean().iloc[i]
                        else "low"
                    ),
                    "rsi_state": (
                        "overbought"
                        if df["RSI"].iloc[i] > self._rsi_overbought
                        else "oversold"
                        if df["RSI"].iloc[i] < self._rsi_oversold
                        else "neutral"
                    ),
                    "trend_strength": (
                        "strong"
                        if abs(price_change_6m) > self._trend_strong
                        else "moderate"
                        if abs(price_change_6m) > self._trend_moderate
                        else "weak"
                    ),
                },
            }

            # Add previous periods' data
            current_data["previous_periods"] = []
            for j in range(1, min(self._historical_window + 1, i + 1)):
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

    def query_ollama(self, prompt: str, env: TradingEnvironment) -> Optional[str]:
        """Query Ollama API for trading analysis."""
        logger.info("   . Querying Ollama")
        url = "http://localhost:11434/api/generate"

        # live and paper trading use same model
        model = self._model_backtest if env == TradingEnvironment.BACKTEST else self._model_live

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        raw_response = response.json()["response"]

        cleaned_response = "".join(char for char in raw_response if ord(char) >= 32 or char in "\n\r\t")
        return cleaned_response

    def get_llm_response(
        self, market_data_analysis_df: Dict[str, Any], env: TradingEnvironment
    ) -> Optional[Dict[str, Any]]:
        """Get trading analysis from LLM."""
        logger.info(" . Generating LLM response")

        # generate stat information for prompt
        stat_info = self.generate_statistical_context(market_data_analysis_df)
        if not stat_info:
            raise ValueError("Failed to generate stat information")

        # generate prompt
        prompt = f"""
        CRITICAL INSTRUCTIONS:
        1. You are a quantitative trading algorithm. Analyze the market data and generate a trading recommendation.
        2. Your response MUST be ONLY the JSON object below
        3. The response must start with {{ and end with }}
        4. Use this EXACT structure:
        {{
            "recommendation": "BUY|SELL|HOLD",
            "confidence": "0.0|0.1|0.2|0.3|0.4|0.5|0.6|0.7|0.8|0.9|1.0",
            "reasoning": "string",
            "price_targets": {{
                "stop_loss": "positive_number (for BUY: below entry, for SELL: above entry)",
                "take_profit": "positive_number (for BUY: above entry, for SELL: below entry)"
            }}
        }}
        5. DO NOT include any thinking process or tags like <thinking> or </thinking>. IT MUST FOLLOW THE EXACT STRUCTURE ABOVE.

        TRADING GUIDELINES:
        1. Analyze the provided market data thoroughly, including:
           - Price action and trends
           - Volume patterns
           - Technical indicators
           - Market context and conditions
           - Support and resistance levels
           - Volatility patterns

        2. Make trading decisions based on your analysis:
           - BUY when you identify ANY bullish opportunity with 10%+ profit potential
           - SELL when you identify ANY bearish opportunity with 10%+ profit potential
           - HOLD only when market conditions are extremely unfavorable or unpredictable
           - Take trades more frequently, even with moderate confidence
           - Look for momentum and trend continuation opportunities
           - Don't be afraid to trade against the trend if you see a reversal opportunity

        3. Size your position based on:
           - Your confidence in the trade
           - Risk/reward ratio (aim for at least 1.5:1)
           - Market volatility
           - Overall market conditions

        4. Set price targets that:
           - For BUY trades:
             * Stop loss must be BELOW entry price
             * Take profit must be ABOVE entry price
             * Target at least 10% profit potential
             * Be more aggressive with take profit levels
           - For SELL trades:
             * Stop loss must be ABOVE entry price
             * Take profit must be BELOW entry price
             * Target at least 10% profit potential
             * Be more aggressive with take profit levels
           - Account for volatility
           - Provide favorable risk/reward ratios (minimum 1.5:1)
           - Consider support/resistance levels
           - Set stop loss at logical support/resistance levels

        {stat_info}
        """

        logger.debug(f"Prompt: {prompt}")
        response_text = self.query_ollama(prompt, env)
        if not response_text:
            logger.error("No response from Ollama")
            return None

        analysis = self.load_analysis_to_json(response_text)
        analysis["current_price"] = market_data_analysis_df["price_data"]["close"]
        analysis["symbol"] = market_data_analysis_df.get("symbol")
        logger.debug(f"Raw response: {analysis}")
        analysis = self.prompt_validation_and_formatting(analysis)
        logger.debug(f"LLM response: {analysis}")

        return analysis

    def load_analysis_to_json(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Load analysis to JSON."""
        logger.info("   . Loading analysis to JSON")
        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            json_str = response_text[start_idx:end_idx]
            analysis = json.loads(json_str)
            return analysis
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON response: {e}")

    def prompt_validation_and_formatting(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and format LLM response."""
        logger.info("   . Validating and formatting LLM JSON response")

        # check if all required fields are present
        required_fields = [
            "recommendation",
            "confidence",
            "reasoning",
            "price_targets",
            "current_price",
        ]
        for field in required_fields:
            if field not in analysis:
                raise ValueError(f"Missing required field: {field}")

        # convert fields to correct types
        analysis["confidence"] = float(analysis["confidence"])
        analysis["recommendation"] = analysis["recommendation"].upper()
        analysis["current_price"] = float(str(analysis["current_price"]).replace("$", ""))

        # format field values
        analysis["price_targets"]["take_profit"] = float(str(analysis["price_targets"]["take_profit"]).replace("$", ""))
        analysis["price_targets"]["stop_loss"] = float(str(analysis["price_targets"]["stop_loss"]).replace("$", ""))

        # check if recommendation is valid
        if analysis["recommendation"] not in ["BUY", "SELL", "HOLD"]:
            raise ValueError(f"Invalid recommendation: {analysis['recommendation']}")

        # check if confidence is valid
        if not 0 <= analysis["confidence"] <= 1:
            raise ValueError(f"Invalid confidence: {analysis['confidence']}. Must be between 0 and 1")

        # check if price targets are valid
        if "stop_loss" not in analysis["price_targets"] or "take_profit" not in analysis["price_targets"]:
            raise ValueError("Missing required price targets")

        # check if stop loss is less than take profit for BUY
        if (
            analysis["recommendation"] == "BUY"
            and analysis["price_targets"]["stop_loss"] >= analysis["price_targets"]["take_profit"]
        ):
            raise ValueError("Stop loss must be less than take profit for a BUY trade")

        # check if stop loss is greater than take profit for SELL
        if (
            analysis["recommendation"] == "SELL"
            and analysis["price_targets"]["stop_loss"] <= analysis["price_targets"]["take_profit"]
        ):
            raise ValueError("Stop loss must be greater than take profit for a SELL trade")

        return analysis

    def save_analysis(self, analysis: Dict[str, Any], symbol: str, timestamp: Optional[datetime] = None) -> None:
        """Save trading analysis to a JSON file.

        Args:
            analysis: The analysis dictionary to save
            symbol: Trading symbol
            timestamp: Optional datetime to use for the file
        """
        base_dir = os.path.join("analysis", self.env.value)

        # Use provided datetime or current time
        dt = timestamp or datetime.now()
        date_str = dt.strftime("%Y-%m-%d")
        hour_str = dt.strftime("%H00")

        output_dir = os.path.join(base_dir, date_str)
        os.makedirs(output_dir, exist_ok=True)

        # Create filename
        filename = f"{symbol}_{date_str.replace('-', '')}_{hour_str}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            json.dump(analysis, f, indent=2)

        logger.info(f" . Analysis saved to {filepath}")

    def get_existing_analysis(self, symbol: str, timestamp: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get existing analysis for a symbol from the most recent JSON file.

        Args:
            symbol: Trading symbol to get analysis for
            timestamp: Optional timestamp to use for the analysis file (format: YYYY-MM-DD HH:MM)
        """
        logger.info(" . Checking for existing analysis")

        # define output directory
        base_dir = os.path.join("analysis", self.env.value)

        if timestamp:
            # Parse the timestamp to get the date
            date_obj = datetime.strptime(timestamp, "%Y-%m-%d %H:%M")
            date_str = date_obj.strftime("%Y-%m-%d")
            hour_str = date_obj.strftime("%H00")
        else:
            # Fallback to current date if no timestamp provided
            date_str = datetime.now().strftime("%Y-%m-%d")
            hour_str = datetime.now().strftime("%H00")

        output_dir = os.path.join(base_dir, date_str)

        # define filename
        filename = f"{symbol}_{date_str.replace('-', '')}_{hour_str}.json"
        filepath = os.path.join(output_dir, filename)

        # check if file exists
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                analysis = json.load(f)
            logger.info(f" . Found existing analysis for {symbol} at {filepath}")
            return analysis

        logger.info(f" . No existing analysis found for {symbol} at {filepath}")
        return None

    def generate_statistical_context(self, market_data_analysis_df: Dict[str, Any]) -> Optional[str]:
        """Format market data and generate trading analysis prompt for LLM.

        Args:
            market_data: Dictionary containing market data and indicators

        Returns:
            Formatted prompt string or None if error
        """
        try:
            logger.info("   . Generating LLM strategy prompt")

            if not market_data_analysis_df:
                raise ValueError("   . No data available for strategy analysis")

            # Format the prompt with clear sections and better readability
            prompt = f"""Statistical Context ({market_data_analysis_df['timestamp']})
            {'=' * 40}

            Current Market Conditions ({market_data_analysis_df['timestamp']}):
            - Price: ${market_data_analysis_df['price_data']['close']:.2f}
            - 1-Week Change: {market_data_analysis_df['price_data']['price_changes']['1w']*100:+.2f}%
            - 2-Week Change: {market_data_analysis_df['price_data']['price_changes']['2w']*100:+.2f}%
            - 1-Month Change: {market_data_analysis_df['price_data']['price_changes']['1m']*100:+.2f}%
            - 6-Month Change: {market_data_analysis_df['price_data']['price_changes']['6m']*100:+.2f}%

            Price Levels:
            - Support Level: ${market_data_analysis_df['price_data']['support_level']:.2f}
            - Resistance Level: ${market_data_analysis_df['price_data']['resistance_level']:.2f}

            Technical Indicators:
            - RSI: {market_data_analysis_df['indicators']['momentum']['rsi']:.2f}
            - MACD: {market_data_analysis_df['indicators']['momentum']['macd']:.2f}
            - MACD Signal: {market_data_analysis_df['indicators']['momentum']['macd_signal']:.2f}
            - MACD Histogram: {market_data_analysis_df['indicators']['momentum']['macd_hist']:.2f}
            - Stochastic K: {market_data_analysis_df['indicators']['momentum']['stoch_k']:.2f}
            - Stochastic D: {market_data_analysis_df['indicators']['momentum']['stoch_d']:.2f}
            - Volume Ratio: {market_data_analysis_df['indicators']['volume']['volume_ratio']:.2f}
            - Volatility: {market_data_analysis_df['indicators']['volatility']['volatility']:.2f}

            Moving Averages:
            - SMA 20: ${market_data_analysis_df['indicators']['moving_averages']['sma_20']:.2f}
            - SMA 50: ${market_data_analysis_df['indicators']['moving_averages']['sma_50']:.2f}
            - SMA 200: ${market_data_analysis_df['indicators']['moving_averages']['sma_200']:.2f}
            - EMA 20: ${market_data_analysis_df['indicators']['moving_averages']['ema_20']:.2f}
            - EMA 50: ${market_data_analysis_df['indicators']['moving_averages']['ema_50']:.2f}
            - EMA 200: ${market_data_analysis_df['indicators']['moving_averages']['ema_200']:.2f}

            Bollinger Bands:
            - Upper: ${market_data_analysis_df['indicators']['bollinger_bands']['upper']:.2f}
            - Middle: ${market_data_analysis_df['indicators']['bollinger_bands']['middle']:.2f}
            - Lower: ${market_data_analysis_df['indicators']['bollinger_bands']['lower']:.2f}"""

            return prompt

        except Exception as e:
            logger.error(f"Error generating LLM prompt: {str(e)}")
            return None
