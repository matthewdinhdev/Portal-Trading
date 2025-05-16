# Portal Trading

A Python-based algorithmic trading platform that integrates with Alpaca for trading execution and Discord for notifications and control.

## Features

- Automated trading strategies implementation
- Backtesting capabilities
- Discord bot integration for real-time notifications and control
- LLM-powered analysis and decision making
- Comprehensive logging system

## Prerequisites

- Python 3.8 or higher
- Alpaca trading account
- Discord bot token
- OpenAI API key (for LLM features)

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
OPENAI_API_KEY=your_openai_api_key
```

## Project Structure

```
Portal-Trading/
├── src/                    # Source code
│   ├── backtest.py        # Backtesting implementation
│   ├── discord_bot.py     # Discord bot functionality
│   ├── llm.py            # LLM integration
│   ├── logger.py         # Logging system
│   ├── main.py           # Main application logic
│   └── strategies.py     # Trading strategies
├── tests/                 # Test files
├── analysis/             # Analysis results (gitignored)
├── logs/                 # Application logs (gitignored)
└── requirements.txt      # Project dependencies
```

## Usage

1. Start the trading bot:
```bash
python src/main.py
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
`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Alpaca for the trading API
- Discord.py for the bot framework
- OpenAI for LLM capabilities 