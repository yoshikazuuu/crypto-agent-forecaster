"""
Configuration management for CryptoAgentForecaster.
"""

import os
from typing import Optional
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
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
    
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