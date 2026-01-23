"""
CryptoAgentForecaster: A multimodal LLM-driven cryptocurrency forecasting system.

This package implements a sophisticated multi-agent forecasting framework featuring:
- Multi-agent architecture using CrewAI (4 specialized agents)
- Integration with CoinGecko API for comprehensive market data
- Novel sentiment analysis from 4chan /biz/ board
- Advanced technical analysis using 50+ indicators
- LLM-based multimodal data fusion for directional forecasting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PACKAGE STRUCTURE
-----------------

agents/
├── crew_manager.py         - Orchestrates multi-agent workflow and task execution
├── market_data_agent.py    - Collects OHLCV and market statistics from CoinGecko
├── sentiment_agent.py      - Analyzes 4chan /biz/ for FUD/shill detection
├── technical_agent.py      - Performs technical analysis with 50+ indicators
└── forecasting_agent.py    - Synthesizes all data into actionable forecasts

tools/
├── coingecko_tool.py       - Market data collection (OHLCV, volume, market cap)
├── fourchan_tool.py        - Real-time 4chan /biz/ sentiment scraping
├── warosu_tool.py          - Historical 4chan archive for backtesting
├── technical_analysis_tool.py  - 50+ indicators, patterns, chart generation
└── chart_analysis_tool.py  - Multimodal chart analysis using LLM vision

prompts/
├── fusion_prompts.py       - Forecasting agent synthesis prompts
├── sentiment_prompts.py    - FUD/shill detection prompts
├── technical_prompts.py    - Indicator interpretation prompts
└── market_prompts.py       - Market data analysis prompts

core/
├── config_base.py          - Base configuration and environment management
├── agent_config.py         - Per-agent LLM configuration
├── analysis_config.py      - Technical/sentiment analysis parameters
├── validation.py           - Configuration validation utilities
├── logging_config.py       - Logging setup and management
└── error_handling.py       - Error handling and recovery

cli/
├── commands.py             - Command handlers for all CLI operations
├── constants.py            - CLI constants and configuration
├── display.py              - Rich terminal UI formatting
└── validators.py           - Input validation and sanitation

backtesting/
├── framework.py            - Backtesting orchestration and workflow
├── methods.py              - Prediction method implementations (agentic, image, sentiment)
├── data_collector.py       - Historical data collection for backtesting
└── analyzer.py             - Statistical analysis and report generation

Other Core Modules:
├── config.py               - Unified configuration interface
├── llm_factory.py          - LLM instance creation (OpenAI, Anthropic, Google)
└── utils.py                - Utility functions (logging, result saving, sanitization)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AGENT WORKFLOW
--------------
1. Market Data Agent → Collects 30 days OHLCV data, market stats
2. Sentiment Agent   → Analyzes 4chan /biz/ for sentiment signals
3. Technical Agent   → Performs indicator analysis + chart generation
4. Forecasting Agent → Synthesizes data into final prediction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATA FLOW
---------
CoinGecko API → Market Data Agent → LLM Analysis
4chan API     → Sentiment Agent   → FUD/Shill Detection
TA Library    → Technical Agent   → Indicator Calculation → Chart Generation
Charts + Data → Forecasting Agent → Final Forecast (Direction + Confidence)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LLM INTEGRATION
---------------
Supports multiple LLM providers with per-agent optimization:
- OpenAI:    GPT-4o, GPT-4o-mini (general analysis, cost-effective)
- Anthropic: Claude-3.5-Sonnet (nuanced reasoning, sentiment)
- Google:    Gemini-1.5-Pro, Gemini-2.0-Flash (large context windows)

Each agent uses task-specific LLM configuration for optimal performance.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For detailed documentation:
- Architecture: see ARCHITECTURE.md
- Usage: see README.md  
- CLI: run 'python main.py help'
"""

__version__ = "0.1.0"
__author__ = "CryptoAgentForecaster Development Team" 