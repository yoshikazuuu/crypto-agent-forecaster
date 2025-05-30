# 🚀 CryptoAgentForecaster

**Multimodal AI-Driven Cryptocurrency Price Forecasting System**

An advanced cryptocurrency forecasting tool that leverages hosted Large Language Models (LLMs), multi-agent architecture, and novel data sources including 4chan's /biz/ board for sentiment analysis.

## 🌟 Features

- **🤖 Multi-Agent Architecture**: Specialized AI agents for data collection, sentiment analysis, technical analysis, and forecasting
- **📊 Comprehensive Data Sources**:
  - CoinGecko API for market data (OHLCV, volume, market cap)
  - 4chan /biz/ board for raw sentiment analysis
  - Technical indicators and candlestick patterns
- **🧠 Hosted LLM Integration**: Support for OpenAI GPT, Anthropic Claude, and Google Gemini
- **📈 Advanced Analysis**:
  - FUD (Fear, Uncertainty, Doubt) detection
  - Shill detection and manipulation analysis
  - Technical pattern recognition
  - Multimodal signal fusion
- **💻 User-Friendly CLI**: Rich terminal interface with beautiful output formatting
- **🗂️ Automatic Result Management**:
  - Auto-saves all results to organized folders
  - Technical analysis charts saved as PNG files
  - Complete run logs with sanitized output
  - Clean logging (no verbose JSON or base64 spam)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd crypto-agent-forecaster

# Install dependencies using uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### 2. Setup API Keys

Copy the environment template and add your API keys:

```bash
cp env_example .env
```

Edit `.env` file with your API keys:

```bash
# At least one LLM provider is required
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here  
GOOGLE_API_KEY=your_google_api_key_here

# Optional but recommended for higher rate limits
COINGECKO_API_KEY=your_coingecko_api_key_here

# LLM Configuration
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4o-mini
```

### 3. Get API Keys

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **Google**: https://aistudio.google.com/app/apikey
- **CoinGecko** (optional): https://www.coingecko.com/en/api/pricing

### 4. Test the Setup

```bash
python main.py config
python main.py test --quick
```

### 5. Run Your First Forecast

```bash
python main.py forecast bitcoin
```

## 📖 Usage Guide

### Basic Commands

```bash
# Generate a forecast for Bitcoin
python main.py forecast bitcoin

# Forecast Ethereum with 3-day horizon
python main.py forecast ethereum --horizon "3 days"

# Use specific LLM provider
python main.py forecast solana --provider anthropic

# All results are automatically saved to results/ folder
# with charts, logs, and structured data

# Get help
python main.py --help
```

### Available Cryptocurrencies

Use `list-cryptos` to see popular options:

```bash
python main.py list-cryptos
```

Popular cryptocurrencies include:
- `bitcoin` (Bitcoin - BTC)
- `ethereum` (Ethereum - ETH)
- `solana` (Solana - SOL)
- `cardano` (Cardano - ADA)
- And many more...

### Configuration Management

```bash
# Check current configuration
python main.py config

# Test system components
python main.py test

# Quick test (fewer API calls)
python main.py test --quick
```

## 🏗️ Architecture

The CryptoAgentForecaster uses a multi-agent architecture built with CrewAI:

### Agents

1. **🔍 CryptoMarketDataAgent**
   - Fetches OHLCV data from CoinGecko
   - Handles rate limiting and data quality
   - Provides foundation for technical analysis

2. **💭 CryptoSentimentAnalysisAgent**
   - Analyzes 4chan /biz/ discussions
   - Detects FUD and shilling attempts
   - Extracts market sentiment and narratives

3. **📊 TechnicalAnalysisAgent**
   - Calculates technical indicators (RSI, MACD, MA, BB)
   - Identifies candlestick patterns
   - Generates technical outlook

4. **🎯 CryptoForecastingAgent**
   - Fuses sentiment and technical analysis
   - Generates final forecast with confidence
   - Provides detailed reasoning

### Data Flow

```
CoinGecko API → Market Data → Technical Analysis
                                        ↓
4chan /biz/ → Sentiment Analysis → Multimodal Fusion → Final Forecast
                                        ↓
                              Hosted LLM Processing
```

## 🔧 Advanced Configuration

### LLM Provider Settings

You can specify different providers and models:

```bash
# Use Anthropic Claude
python main.py forecast bitcoin --provider anthropic --model claude-3-5-sonnet-20241022

# Use Google Gemini
python main.py forecast ethereum --provider google --model gemini-1.5-pro

# Use OpenAI GPT-4
python main.py forecast solana --provider openai --model gpt-4o
```

### Environment Variables

All configuration options available in `.env`:

```bash
# LLM Configuration
DEFAULT_LLM_PROVIDER=openai  # openai, anthropic, google
DEFAULT_LLM_MODEL=gpt-4o-mini

# Rate Limiting
API_RATE_LIMIT_DELAY=1.0

# Logging
LOG_LEVEL=INFO
```

## 📊 Output Format

The system provides structured forecasts including:

- **Direction**: UP/DOWN/NEUTRAL
- **Confidence**: HIGH/MEDIUM/LOW  
- **Detailed Explanation**: Reasoning and key factors
- **Technical Analysis**: Indicators and patterns
- **Sentiment Analysis**: Market mood and narratives
- **Risk Considerations**: Caveats and uncertainties

Example output:
```
📊 Forecast Results for BITCOIN
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
║ Metric           ║ Value                         ║
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Direction        │ 🟢 UP (Bullish)              │
│ Confidence       │ MEDIUM                        │
│ Forecast Horizon │ 24 hours                      │
│ Timestamp        │ 2024-01-15T10:30:00           │
│ Charts Generated │ ✅ 1                          │
└──────────────────┴───────────────────────────────┘

🧠 Analysis & Reasoning
Technical analysis shows bullish momentum with RSI at 45 
and MACD crossing above signal line. 4chan sentiment 
reveals moderate optimism with limited FUD detection...
```

## 📁 Results Management

Every forecast run automatically creates a dedicated folder in `results/` with:

### Folder Structure
```
results/
└── bitcoin_20241215_143052/
    ├── README.md                # Summary with key metrics
    ├── forecast_results.json    # Complete structured data
    ├── run_logs.txt            # Sanitized execution logs
    └── charts/
        └── technical_analysis_chart.png  # Generated charts
```

### Features
- **🧹 Clean Logging**: No verbose JSON or base64 spam in console
- **📊 Chart Generation**: Technical analysis charts saved as PNG
- **📄 Structured Data**: Complete results in JSON format
- **📝 Run Logs**: Full execution history with timestamps
- **📋 Summary**: Markdown summary with embedded charts

## ⚠️ Important Considerations

### Risks & Limitations

- **Not Financial Advice**: This tool is for research and educational purposes only
- **Market Volatility**: Cryptocurrency markets are highly volatile and unpredictable  
- **Data Quality**: 4chan data is noisy and may contain manipulation attempts
- **LLM Limitations**: AI models can hallucinate or misinterpret data
- **API Dependencies**: System relies on external APIs that may have downtime

### Ethical Use

- Respect API rate limits and terms of service
- Use 4chan data responsibly and ethically
- Do not use for market manipulation
- Always conduct additional research before making financial decisions

## 🛠️ Development

### Project Structure

```
crypto-agent-forecaster/
├── src/crypto_agent_forecaster/
│   ├── agents/          # CrewAI agent definitions
│   ├── tools/           # LangChain tools for data fetching
│   ├── prompts/         # LLM prompt templates
│   ├── config.py        # Configuration management
│   └── llm_factory.py   # LLM provider factory
├── main.py              # CLI application
├── pyproject.toml       # Dependencies
└── README.md           # This file
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable  
5. Submit a pull request

## 📚 Research Background

This project implements concepts from the research paper "Agent-Driven Cryptocurrency Forecasting: A Multimodal Approach with Hosted LLMs, CoinGecko, and 4chan/biz Sentiment Analysis." Key innovations include:

- Novel use of 4chan /biz/ for sentiment analysis
- Multimodal fusion of technical and sentiment signals
- Agent-based architecture for scalable analysis
- Hosted LLM integration for rapid iteration

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- CoinGecko API for market data
- 4chan for providing public API access
- OpenAI, Anthropic, and Google for LLM APIs
- CrewAI and LangChain communities
- The broader cryptocurrency and AI research communities

---

**⚠️ Disclaimer**: This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Always do your own research and never invest more than you can afford to lose.
