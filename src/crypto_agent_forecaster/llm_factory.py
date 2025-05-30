"""
LLM Factory for creating different hosted LLM instances.
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import Config


class LLMFactory:
    """Factory for creating hosted LLM instances."""
    
    @staticmethod
    def create_llm(provider: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.1):
        """
        Create an LLM instance based on provider and model.
        
        Args:
            provider: LLM provider ('openai', 'anthropic', 'google')
            model: Model name
            temperature: Sampling temperature
            
        Returns:
            LLM instance
        """
        provider = provider or Config.DEFAULT_LLM_PROVIDER
        model = model or Config.DEFAULT_LLM_MODEL
        
        if provider == "openai":
            if not Config.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not configured")
            return ChatOpenAI(
                api_key=Config.OPENAI_API_KEY,
                model=model,
                temperature=temperature
            )
        
        elif provider == "anthropic":
            if not Config.ANTHROPIC_API_KEY:
                raise ValueError("Anthropic API key not configured")
            return ChatAnthropic(
                api_key=Config.ANTHROPIC_API_KEY,
                model=model,
                temperature=temperature
            )
        
        elif provider == "google":
            if not Config.GOOGLE_API_KEY:
                raise ValueError("Google API key not configured")
            return ChatGoogleGenerativeAI(
                google_api_key=Config.GOOGLE_API_KEY,
                model=model,
                temperature=temperature
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def get_available_providers():
        """Get list of available LLM providers based on configured API keys."""
        providers = []
        
        if Config.OPENAI_API_KEY:
            providers.append("openai")
        if Config.ANTHROPIC_API_KEY:
            providers.append("anthropic")
        if Config.GOOGLE_API_KEY:
            providers.append("google")
            
        return providers
    
    @staticmethod
    def validate_configuration():
        """Validate that at least one LLM provider is configured."""
        providers = LLMFactory.get_available_providers()
        if not providers:
            raise ValueError(
                "No LLM providers configured. Please set at least one of: "
                "OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY"
            )
        return True 