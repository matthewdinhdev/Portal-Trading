import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import requests

from technical_analysis import TechnicalAnalysis
from trading_enums import TradingEnvironment

# This logger will inherit all settings from the root logger
logger = logging.getLogger(__name__)


class LLMAnalyzer:
    # Private class variables
    _model_backtest = "deepseek-llm:7b"
    _model_live = "deepseek-r1:14b"
    _min_risk_reward_ratio = 1.5
    _current_stat_info = None
    _original_analysis = None

    def __init__(self, env: TradingEnvironment):
        """Initialize the LLM Analyzer."""
        self.env = env
        self._original_analysis = None

    def query_ollama_generate(
        self, prompt: str, env: TradingEnvironment, system_message: Optional[str] = None
    ) -> Optional[str]:
        """Query Ollama API for initial trading analysis using generate endpoint."""
        logger.info("   . Querying Ollama (generate)")
        url = "http://localhost:11434/api/generate"

        model = self._model_backtest if env == TradingEnvironment.BACKTEST else self._model_live

        # Combine system message and prompt
        full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
        logger.debug(f"Full prompt for generate:\n{full_prompt}")

        payload = {
            "model": model,
            "prompt": full_prompt,
            "options": {
                "temperature": 0.1,
            },
        }

        response = requests.post(url, json=payload)
        logger.debug(f"Response status: {response.status_code}")
        response.raise_for_status()

        try:
            # Handle streaming response
            full_content = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        full_content += chunk["response"]
                    if chunk.get("done", False):
                        break

            logger.debug(f"Full content: {full_content}")
            return full_content
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            logger.error(f"Response text: {response.text}")
            raise

    def get_llm_response(
        self, market_data_analysis: Dict[str, Any], env: TradingEnvironment
    ) -> Optional[Dict[str, Any]]:
        """Get trading analysis from LLM."""
        logger.info(" . Generating LLM response")

        # generate stat information for prompt
        self._current_stat_info = self.generate_statistical_context(market_data_analysis)
        if not self._current_stat_info:
            raise ValueError("Failed to generate stat information")

        # Reset original analysis for new analysis
        self._original_analysis = None

        # generate prompt
        system_message = """
        CRITICAL INSTRUCTIONS:
        1. You are a quantitative trading algorithm. Analyze the market data and generate a trading recommendation.
        2. Your response MUST be ONLY the JSON object below
        3. The response must start with { and end with }
        4. Use this EXACT structure:
        {
            "recommendation": "BUY|SELL|HOLD",
            "confidence": 0.0|0.1|0.2|0.3|0.4|0.5|0.6|0.7|0.8|0.9|1.0,
            "reasoning": "string"
        }

        TRADING GUIDELINES:
        1. Analyze the provided market data thoroughly, including:
           - Price action and trends
           - Volume patterns
           - Technical indicators
           - Market context and conditions
           - Support and resistance levels
           - Volatility patterns

        2. Make trading decisions based on your analysis:
           - BUY when you identify ANY bullish opportunity
           - SELL when you identify ANY bearish opportunity 
           - HOLD only when market conditions are extremely unfavorable or unpredictable
           - Take trades more frequently, even with moderate confidence
           - Look for momentum and trend continuation opportunities
           - Don't be afraid to trade against the trend if you see a reversal opportunity
        """

        user_message = f"""
        Please analyze this market data and provide a trading recommendation:
        {self._current_stat_info}
        """

        # Store the original analysis prompt
        self._original_analysis = {"system": system_message, "prompt": user_message}

        response_text = self.query_ollama_generate(user_message, env, system_message)
        if not response_text:
            logger.error("No response from Ollama")
            return None

        # Store the original analysis response
        self._original_analysis["response"] = response_text

        analysis = self.load_analysis_to_json(response_text)

        # Add additional parameters before validation
        analysis = self.add_analysis_parameters(analysis, market_data_analysis)

        # Validate and format the complete analysis
        analysis = self.analysis_validation_and_formatting(analysis)
        logger.debug(f"LLM response: {analysis}")

        return analysis

    def add_analysis_parameters(self, analysis: Dict[str, Any], market_data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Add additional parameters to the analysis, including price targets.

        Args:
            analysis: The base analysis dictionary
            market_data_analysis: The market data dictionary

        Returns:
            Dictionary with additional parameters added
        """
        logger.info("   . Adding additional analysis parameters")

        # Add current price and symbol
        analysis["current_price"] = market_data_analysis["price_data"]["close"]
        analysis["symbol"] = market_data_analysis.get("symbol")

        # If HOLD, we don't need price targets
        if analysis["recommendation"] == "HOLD":
            return analysis

        # Get Fibonacci levels and ATR from market data
        fib_levels = market_data_analysis["price_data"]["fibonacci_levels"]
        atr = market_data_analysis["indicators"]["volatility"]["atr"]

        if not fib_levels:
            raise ValueError("No Fibonacci levels available in market data")
        if not atr or atr <= 0:
            raise ValueError("Invalid ATR value in market data")

        # Add ATR to fib_levels dict for fallback
        fib_levels["atr"] = atr

        # Calculate price targets
        price_targets = TechnicalAnalysis.calculate_price_targets_from_fib(
            current_price=analysis["current_price"],
            fib_levels=fib_levels,
            recommendation=analysis["recommendation"],
            min_risk_reward=self._min_risk_reward_ratio,
        )

        # Add price targets to analysis
        analysis["price_targets"] = price_targets

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

    def analysis_validation_and_formatting(self, analysis: Dict[str, Any], retry_count: int = 0) -> Dict[str, Any]:
        """Validate and format LLM response."""
        logger.info("   . Validating and formatting LLM JSON response")

        # check if all required fields are present
        required_fields = [
            "recommendation",
            "confidence",
            "reasoning",
            "current_price",
        ]
        for field in required_fields:
            if field not in analysis:
                raise ValueError(f"Missing required field: {field}")

        # convert fields to correct types
        analysis["confidence"] = float(analysis["confidence"])
        analysis["recommendation"] = analysis["recommendation"].upper()
        analysis["current_price"] = float(str(analysis["current_price"]).replace("$", ""))

        # check if recommendation is valid
        if analysis["recommendation"] not in ["BUY", "SELL", "HOLD"]:
            raise ValueError(f"Invalid recommendation: {analysis['recommendation']}")

        # check if confidence is valid
        if not 0 <= analysis["confidence"] <= 1:
            raise ValueError(f"Invalid confidence: {analysis['confidence']}. Must be between 0 and 1")

        # If HOLD, we don't need price targets
        if analysis["recommendation"] == "HOLD":
            return analysis

        # Price targets should already be added by add_analysis_parameters
        if "price_targets" not in analysis:
            raise ValueError("Price targets not found in analysis")

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
        """Generate statistical context for LLM prompt."""
        try:
            # Get price data
            price_data = market_data_analysis_df["price_data"]
            current_price = price_data["close"]

            # Format support and resistance levels
            support_levels = price_data["support_levels"]
            resistance_levels = price_data["resistance_levels"]
            fib_levels = price_data["fibonacci_levels"]

            # Format the context
            context = f"""
            MARKET DATA:
            Current Price: ${current_price:.2f}
            
            Support Levels:
            - Short-term (30 days): {', '.join([f'${s:.2f}' for s in support_levels['short_term']])}
            - Medium-term (90 days): {', '.join([f'${s:.2f}' for s in support_levels['medium_term']])}
            - Long-term (180 days): {', '.join([f'${s:.2f}' for s in support_levels['long_term']])}
            
            Resistance Levels:
            - Short-term (30 days): {', '.join([f'${r:.2f}' for r in resistance_levels['short_term']])}
            - Medium-term (90 days): {', '.join([f'${r:.2f}' for r in resistance_levels['medium_term']])}
            - Long-term (180 days): {', '.join([f'${r:.2f}' for r in resistance_levels['long_term']])}
            
            Fibonacci Levels:
            - 0.0: ${fib_levels['0.0']:.2f}
            - 0.236: ${fib_levels['0.236']:.2f}
            - 0.382: ${fib_levels['0.382']:.2f}
            - 0.5: ${fib_levels['0.5']:.2f}
            - 0.618: ${fib_levels['0.618']:.2f}
            - 0.786: ${fib_levels['0.786']:.2f}
            - 1.0: ${fib_levels['1.0']:.2f}
            
            Technical Indicators:
            - RSI: {market_data_analysis_df['indicators']['momentum']['rsi']:.2f}
            - MACD: {market_data_analysis_df['indicators']['momentum']['macd']:.2f}
            - MACD Signal: {market_data_analysis_df['indicators']['momentum']['macd_signal']:.2f}
            - MACD Histogram: {market_data_analysis_df['indicators']['momentum']['macd_hist']:.2f}
            - Stochastic K: {market_data_analysis_df['indicators']['momentum']['stoch_k']:.2f}
            - Stochastic D: {market_data_analysis_df['indicators']['momentum']['stoch_d']:.2f}
            - Volume Ratio: {market_data_analysis_df['indicators']['volume']['volume_ratio']:.2f}
            - Volatility: {market_data_analysis_df['indicators']['volatility']['volatility']:.2f}
            - ATR: {market_data_analysis_df['indicators']['volatility']['atr']:.2f}
            
            Moving Averages:
            - SMA 20: {market_data_analysis_df['indicators']['moving_averages']['sma_20']:.2f}
            - SMA 50: {market_data_analysis_df['indicators']['moving_averages']['sma_50']:.2f}
            - SMA 200: {market_data_analysis_df['indicators']['moving_averages']['sma_200']:.2f}
            - EMA 20: {market_data_analysis_df['indicators']['moving_averages']['ema_20']:.2f}
            - EMA 50: {market_data_analysis_df['indicators']['moving_averages']['ema_50']:.2f}
            - EMA 200: {market_data_analysis_df['indicators']['moving_averages']['ema_200']:.2f}
            
            Bollinger Bands:
            - Upper: {market_data_analysis_df['indicators']['bollinger_bands']['upper']:.2f}
            - Middle: {market_data_analysis_df['indicators']['bollinger_bands']['middle']:.2f}
            - Lower: {market_data_analysis_df['indicators']['bollinger_bands']['lower']:.2f}
            """

            return context

        except Exception as e:
            logger.error(f"Error generating statistical context: {str(e)}")
            return None
