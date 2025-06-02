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
git clone https://github.com/yoshikazuuu/crypto-agent-forecaster.git
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

### System Overview

![System Overview Flow](.github/System%20Overview%20Flow.svg)

*Complete system architecture showing the flow from user interface through multi-agent processing to final outputs*

### Technology Stack

![Technology Stack Overview](.github/Technology%20Stack%20Overview.svg)

*Comprehensive view of all technologies, frameworks, and APIs used in the system*

### Data Flow Sequence

![Detailed Data Flow Sequence](.github/Detailed%20Data%20Flow%20Sequence.svg)

*Step-by-step sequence diagram showing how data flows through each component*

### Component Interactions

![Component Interaction Matrix](.github/Component%20Interaction%20Matrix.svg)

*Matrix view of how different system components interact with each other*

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

## Features

- Real-time cryptocurrency data retrieval
- Advanced technical analysis with visual charts
- **NEW: Multimodal chart analysis using AI agents**
- Sentiment analysis from multiple sources
- Comprehensive market forecasting

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Technical Analysis

```python
from crypto_agent_forecaster.tools.technical_analysis_tool import technical_analysis_tool
from crypto_agent_forecaster.tools.chart_analysis_tool import chart_analysis_tool

# Perform technical analysis (fetches data automatically and generates chart)
analysis = technical_analysis_tool(crypto_name="bitcoin", days=7)
print(analysis)

# Analyze the generated chart using AI (multimodal analysis)
chart_analysis = chart_analysis_tool(crypto_name="bitcoin", analysis_context="Focus on 24-hour price prediction")
print(chart_analysis)
```

### Using with CrewAI Flows (Recommended)

```python
from crewai.flow.flow import Flow, listen, start
from crewai import Agent, Task, Crew
from pydantic import BaseModel

class CryptoAnalysisState(BaseModel):
    crypto_name: str = ""
    technical_analysis: str = ""
    chart_analysis: str = ""
    final_forecast: str = ""

class CryptoAnalysisFlow(Flow[CryptoAnalysisState]):
    
    @start()
    def initialize_analysis(self):
        self.state.crypto_name = "bitcoin"
        return "Analysis initialized"
    
    @listen(initialize_analysis)
    def perform_technical_analysis(self, _):
        # Use technical_analysis_tool (automatically fetches fresh data)
        from crypto_agent_forecaster.tools.technical_analysis_tool import technical_analysis_tool
        
        result = technical_analysis_tool(
            crypto_name=self.state.crypto_name,
            days=30  # Fetch 30 days of data for analysis
        )
        self.state.technical_analysis = result
        return "Technical analysis completed"

# Run the flow
flow = CryptoAnalysisFlow()
forecast = flow.kickoff()
print(f"Final Forecast: {forecast}")
```

## Key Features of the Updated Tools

### Technical Analysis Tool (`technical_analysis_tool`)

- **Enhanced Parameter Handling**: Now accepts both JSON strings and dict objects
- **Improved Chart Generation**: Creates high-quality TradingView-style charts
- **File-based Chart Storage**: Saves charts as temporary files for multimodal access
- **Comprehensive Indicators**: RSI, MACD, Moving Averages, Bollinger Bands, Volume analysis

### Chart Analysis Tool (`chart_analysis_tool`) - NEW MULTIMODAL

- **AI-Powered Visual Analysis**: Uses CrewAI multimodal agents to actually "see" and analyze charts
- **Computer Vision Capabilities**: Recognizes patterns, trends, and technical formations visually
- **Contextual Analysis**: Accepts specific analysis context for targeted insights
- **Expert Agent Integration**: Creates specialized chart analysis agents with domain expertise

### Multimodal Agent Features

The chart analysis tool now uses CrewAI's multimodal capabilities:

```python
# Create a multimodal agent
chart_analyst = Agent(
    role="Expert Technical Chart Analyst",
    goal="Analyze technical charts with visual recognition",
    backstory="World-class technical analyst with pattern recognition expertise",
    multimodal=True  # This enables image analysis capabilities
)

# The agent can now actually "see" and interpret chart images
task = Task(
    description="Analyze the chart image and identify key patterns, trends, and trading opportunities",
    expected_output="Detailed visual chart analysis with specific insights",
    agent=chart_analyst
)
```

## Error Handling and Fallbacks

Both tools include comprehensive error handling:

- **Parameter validation** with clear error messages
- **Graceful degradation** when chart generation fails
- **Fallback charts** when advanced charting libraries aren't available
- **Multimodal fallbacks** when AI chart analysis fails

## Advanced Usage

### Custom Risk-Adjusted Analysis

```python
from crypto_agent_forecaster.tools.chart_analysis_tool import analyze_chart_with_context

# Risk-adjusted analysis
analysis = analyze_chart_with_context(
    crypto_name="bitcoin",
    specific_questions=[
        "What are the key support levels for the next 24 hours?",
        "Is this a good entry point for a long position?",
        "What's the probability of a breakout above current resistance?"
    ],
    risk_tolerance="conservative"  # conservative, moderate, or aggressive
)
```

### Integration with External Data

```python
# Example with CoinGecko data
from crypto_agent_forecaster.tools.coingecko_tool import coingecko_tool

# Fetch fresh data
data_result = coingecko_tool(query="bitcoin ohlcv 24 hours horizon")
ohlcv_data = data_result["ohlcv_data"]

# Perform analysis
technical_result = technical_analysis_tool(
    ohlcv_data=ohlcv_data,
    crypto_name="bitcoin"
)

# AI chart analysis
chart_result = chart_analysis_tool(
    crypto_name="bitcoin",
    analysis_context="Focus on momentum indicators and volume confirmation"
)
```

## Requirements

- Python 3.8+
- CrewAI with multimodal support
- matplotlib, mplfinance for charting
- pandas, numpy for data processing
- ta (Technical Analysis library)

## Configuration

Configure technical analysis parameters in `src/crypto_agent_forecaster/config.py`:

```python
TA_INDICATORS = {
    "sma_periods": [20, 50],
    "ema_periods": [12, 26],
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2
}
```

## License

MIT License
