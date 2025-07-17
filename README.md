# Portal Trading

A Python-based algorithmic trading platform that integrates with Alpaca for trading execution, Discord for notifications and control, and DeepSeek LLMs for advanced trading analysis.

## Features

- Automated trading strategies implementation
- Backtesting capabilities
- Discord bot integration for real-time notifications and control
- LLM-powered analysis and decision making (DeepSeek via Ollama)
- Comprehensive logging system

## Prerequisites

- Python 3.8 or higher
- Alpaca trading account
- Discord bot token
- **Ollama server running DeepSeek models** (see DeepSeek section below)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/matthewdinhdev/Portal-Trading.git
cd Portal-Trading
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
DISCORD_TOKEN=your_discord_bot_token
DISCORD_WEBHOOK_URL=your_discord_webhook_url
```

## Project Structure

```
Portal-Trading/
├── src/                    # Source code
│   ├── backtest.py         # Backtesting implementation and performance metrics
│   ├── discord_bot.py      # Discord bot and webhook notification logic
│   ├── llm.py              # LLM integration (DeepSeek via Ollama)
│   ├── logger.py           # Logging system setup
│   ├── main.py             # Main application logic and trading loop
│   ├── technical_analysis.py # Technical analysis indicators and calculations
│   ├── trading_enums.py    # Enum definitions for trading environments
│   ├── utility.py          # Market data management and utility functions
│   └── __init__.py         # Package initialization
├── tests/                  # Test files
├── analysis/               # Analysis results (gitignored)
├── logs/                   # Application logs (gitignored)
└── requirements.txt        # Project dependencies
```

## DeepSeek LLM Integration

This project uses DeepSeek LLMs (e.g., `deepseek-llm:7b`, `deepseek-r1:14b`) for advanced trading analysis and decision-making. The LLMs are accessed via a local [Ollama](https://ollama.com/) server. You must:

1. Install Ollama and pull the required DeepSeek models:
   ```bash
   ollama pull deepseek-llm:7b
   ollama pull deepseek-r1:14b
   ```
2. Ensure Ollama is running locally (default: `http://localhost:11434`).
3. The LLM is queried for trading recommendations, confidence, and reasoning, which are then used for automated trading decisions.

## Usage

1. Start the trading bot:
```bash
python src/main.py --env paper_trading
python src/backtest.py
```

2. Use Discord commands to interact with the bot:
- `/status` - Check current trading status
- `/start` - Start trading
- `/stop` - Stop trading
- `/backtest` - Run backtesting

## Development

### Running Tests
```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Alpaca for the trading API
- Discord.py for the bot framework
- DeepSeek for advanced LLM models
- Ollama for local LLM serving 