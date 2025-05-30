"""
Configuration management for CryptoAgentForecaster.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Ensure environment is loaded only once
_env_loaded = False
if not _env_loaded:
    load_dotenv(override=False)  # Don't override existing env vars
    _env_loaded = True

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
            "temperature": 0.05,  # Very low creativity for consistent sentiment analysis
            "max_tokens": 3000,
            "preferred_provider": "openai",  # Temporarily using OpenAI instead of Anthropic
            "preferred_model": "gpt-4o"
        },
        "technical": {
            "temperature": 0.0,  # Maximum precision for technical analysis
            "max_tokens": 2500,
            "preferred_provider": "openai",
            "preferred_model": "gpt-4o"
        },
        "forecasting": {
            "temperature": 0.05,  # Much lower for consistent forecasts
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
        "sma_periods": [20, 50, 100, 200],  # Extended SMA periods
        "ema_periods": [9, 12, 26, 50],     # Extended EMA periods for professional analysis
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "bb_period": 20,
        "bb_std": 2,
        "volume_sma_period": 20,             # Volume SMA overlay
        "professional_colors": {             # Professional color scheme matching TradingView
            "ema_9": "#00D4AA",              # Cyan
            "ema_12": "#00CED1",             # Light cyan  
            "ema_21": "#20B2AA",             # Light sea green
            "sma_20": "#FFD700",             # Gold
            "ema_26": "#FF8C00",             # Orange
            "sma_50": "#DA70D6",             # Purple
            "sma_100": "#9370DB",            # Medium purple
            "sma_200": "#8A2BE2",            # Blue violet
            "bollinger_bands": "#87CEEB",    # Sky blue
            "rsi_line": "#F59E0B",           # Amber
            "rsi_overbought": "#DC2626",     # Red
            "rsi_oversold": "#059669",       # Green
            "macd_line": "#00D4AA",          # Cyan
            "macd_signal": "#FF6B6B",        # Light red
            "macd_histogram_positive": "#10B981", # Green
            "macd_histogram_negative": "#EF4444", # Red
            "volume_bullish": "#10B981",     # Green
            "volume_bearish": "#EF4444",     # Red
            "volume_sma": "#FFD700",         # Gold
            "pattern_annotation_bg": "#2A2E39",    # Dark annotation background
            "pattern_annotation_border": "#363A45", # Annotation border
            "text_primary": "#D1D4DC",       # Primary text color
            "text_secondary": "#9CA3AF"      # Secondary text color
        }
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
            "temperature": 0.0,  # Lower default temperature for consistency
            "max_tokens": 2000,
            "preferred_provider": cls.DEFAULT_LLM_PROVIDER,
            "preferred_model": cls.DEFAULT_LLM_MODEL
        })
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        # Check if at least one LLM provider is configured
        llm_providers = [cls.OPENAI_API_KEY, cls.ANTHROPIC_API_KEY, cls.GOOGLE_API_KEY]
        if not any(llm_providers):
            print("Warning: No LLM API keys configured. You'll need at least one to run the system.")
            return False
            
        return True
    
    @classmethod
    def debug_environment(cls) -> Dict[str, Any]:
        """Debug environment configuration for troubleshooting."""
        debug_info = {
            "environment_loaded": _env_loaded,
            "api_keys_configured": {
                "openai": bool(cls.OPENAI_API_KEY),
                "anthropic": bool(cls.ANTHROPIC_API_KEY),
                "google": bool(cls.GOOGLE_API_KEY),
                "coingecko": bool(cls.COINGECKO_API_KEY)
            },
            "default_provider": cls.DEFAULT_LLM_PROVIDER,
            "default_model": cls.DEFAULT_LLM_MODEL,
            "env_vars_present": {
                "OPENAI_API_KEY": "OPENAI_API_KEY" in os.environ,
                "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY" in os.environ,
                "GOOGLE_API_KEY": "GOOGLE_API_KEY" in os.environ,
                "COINGECKO_API_KEY": "COINGECKO_API_KEY" in os.environ
            }
        }
        return debug_info
    
    @classmethod 
    def validate_llm_config(cls, agent_type: str) -> bool:
        """Validate LLM configuration for a specific agent type."""
        try:
            config = cls.get_agent_llm_config(agent_type)
            provider = config.get("preferred_provider")
            model = config.get("preferred_model")
            
            if not provider or not model:
                print(f"❌ Invalid LLM config for {agent_type}: missing provider or model")
                return False
            
            # Check if the provider's API key is available
            if provider == "openai" and not cls.OPENAI_API_KEY:
                print(f"❌ OpenAI API key required for {agent_type} agent")
                return False
            elif provider == "anthropic" and not cls.ANTHROPIC_API_KEY:
                print(f"❌ Anthropic API key required for {agent_type} agent")
                return False
            elif provider == "google" and not cls.GOOGLE_API_KEY:
                print(f"❌ Google API key required for {agent_type} agent")
                return False
            
            print(f"✅ LLM config valid for {agent_type}: {provider}/{model}")
            return True
            
        except Exception as e:
            print(f"❌ Error validating LLM config for {agent_type}: {e}")
            return False 