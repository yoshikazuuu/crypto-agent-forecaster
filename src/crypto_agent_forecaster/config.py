"""
Configuration management for CryptoAgentForecaster.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for CryptoAgentForecaster."""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    COINGECKO_API_KEY: Optional[str] = os.getenv("COINGECKO_API_KEY")
    
    # Default LLM Configuration
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o")
    
    # LLM-specific configurations for different agents
    LLM_AGENT_CONFIGS: Dict[str, Dict[str, Any]] = {
        "market_data": {
            "temperature": 0.0,  # Very deterministic for data processing
            "max_tokens": 2000,
            "preferred_provider": "openai",
            "preferred_model": "gpt-4o-mini"
        },
        "sentiment": {
            "temperature": 0.2,  # Slightly creative for nuanced sentiment
            "max_tokens": 3000,
            "preferred_provider": "anthropic", 
            "preferred_model": "claude-3-5-sonnet-20241022"
        },
        "technical": {
            "temperature": 0.1,  # Precise for technical analysis
            "max_tokens": 2500,
            "preferred_provider": "openai",
            "preferred_model": "gpt-4o"
        },
        "forecasting": {
            "temperature": 0.3,  # Balanced for reasoning and synthesis
            "max_tokens": 4000,
            "preferred_provider": "google",
            "preferred_model": "gemini-1.5-pro"
        }
    }
    
    # CrewAI specific settings
    CREW_SETTINGS: Dict[str, Any] = {
        "verbose": True,
        "memory": False,
        "cache": True,
        "max_iter": 3,
        "max_execution_time": 300,  # 5 minutes max per forecast
    }
    
    # Rate Limiting
    API_RATE_LIMIT_DELAY: float = float(os.getenv("API_RATE_LIMIT_DELAY", "1.0"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # CoinGecko Configuration
    COINGECKO_BASE_URL: str = "https://api.coingecko.com/api/v3"
    
    # 4chan Configuration
    FOURCHAN_BASE_URL: str = "https://a.4cdn.org"
    FOURCHAN_RATE_LIMIT: float = 1.0  # Max 1 request per second
    
    # Default cryptocurrencies for testing
    DEFAULT_CRYPTOS: list = ["bitcoin", "ethereum", "solana"]
    
    # Technical Analysis Configuration
    TA_INDICATORS: dict = {
        "sma_periods": [20, 50, 200],
        "ema_periods": [12, 26],
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_std": 2
    }
    
    # Sentiment Analysis Configuration
    SENTIMENT_CONFIG: dict = {
        "fud_keywords": [
            "crash", "dump", "scam", "rug", "ponzi", "bubble", 
            "dead", "worthless", "exit liquidity", "bagholders"
        ],
        "shill_keywords": [
            "moon", "lambo", "diamond hands", "to the moon", "hodl",
            "pump", "buy the dip", "1000x", "next bitcoin", "early"
        ],
        "max_posts_per_analysis": 100,
        "sentiment_aggregation_window": "24h"
    }
    
    @classmethod
    def get_agent_llm_config(cls, agent_type: str) -> Dict[str, Any]:
        """Get LLM configuration for a specific agent type."""
        return cls.LLM_AGENT_CONFIGS.get(agent_type, {
            "temperature": 0.1,
            "max_tokens": 2000,
            "preferred_provider": cls.DEFAULT_LLM_PROVIDER,
            "preferred_model": cls.DEFAULT_LLM_MODEL
        })
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        required_keys = []
        
        # Check if at least one LLM provider is configured
        llm_providers = [cls.OPENAI_API_KEY, cls.ANTHROPIC_API_KEY, cls.GOOGLE_API_KEY]
        if not any(llm_providers):
            print("Warning: No LLM API keys configured. You'll need at least one to run the system.")
            return False
            
        return True 