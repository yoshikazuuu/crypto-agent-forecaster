"""
Base configuration management for CryptoAgentForecaster.
"""

import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Ensure environment is loaded only once
_env_loaded = False
if not _env_loaded:
    load_dotenv(override=False)  # Don't override existing env vars
    _env_loaded = True


class BaseConfig:
    """Base configuration class with core settings."""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    COINGECKO_API_KEY: Optional[str] = os.getenv("COINGECKO_API_KEY")
    
    # Default LLM Configuration
    # Default to None to trigger interactive model selection
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "")
    DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "")
    
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
    
    @classmethod
    def get_api_key_status(cls) -> Dict[str, bool]:
        """Get the status of all API keys."""
        return {
            "openai": bool(cls.OPENAI_API_KEY),
            "anthropic": bool(cls.ANTHROPIC_API_KEY),
            "google": bool(cls.GOOGLE_API_KEY),
            "coingecko": bool(cls.COINGECKO_API_KEY)
        }
    
    @classmethod
    def has_any_llm_provider(cls) -> bool:
        """Check if at least one LLM provider is configured."""
        return any([
            cls.OPENAI_API_KEY,
            cls.ANTHROPIC_API_KEY,
            cls.GOOGLE_API_KEY
        ])
    
    @classmethod
    def get_configured_providers(cls) -> list:
        """Get list of configured LLM providers."""
        providers = []
        if cls.OPENAI_API_KEY:
            providers.append("openai")
        if cls.ANTHROPIC_API_KEY:
            providers.append("anthropic")
        if cls.GOOGLE_API_KEY:
            providers.append("google")
        return providers
    
    @classmethod
    def validate_environment(cls) -> bool:
        """Validate basic environment configuration."""
        try:
            # Check if at least one LLM provider is configured
            if not cls.has_any_llm_provider():
                logger.error("No LLM providers configured")
                return False
            
            # Validate environment variables are accessible
            providers = cls.get_configured_providers()
            logger.info(f"Environment validated: {len(providers)} LLM providers configured")
            return True
            
        except Exception as e:
            logger.error(f"Environment validation failed: {str(e)}")
            return False 