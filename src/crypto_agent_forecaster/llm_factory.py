"""
LLM Factory for creating different hosted LLM instances using CrewAI's native approach.
"""

from typing import Optional, Dict, Any
import logging
from crewai import LLM
import google.generativeai as genai

from .config import Config

logger = logging.getLogger(__name__)

class LLMFactory:
    """Factory for creating hosted LLM instances with CrewAI's native LLM class."""
    
    # Model specifications for better validation and context management
    MODEL_SPECS = {
        "openai": {
            "gpt-4o": {"max_tokens": 128000, "cost_per_1k_tokens": {"input": 0.005, "output": 0.015}},
            "gpt-4o-mini": {"max_tokens": 128000, "cost_per_1k_tokens": {"input": 0.00015, "output": 0.0006}},
            "gpt-4": {"max_tokens": 8192, "cost_per_1k_tokens": {"input": 0.03, "output": 0.06}},
        },
        "anthropic": {
            "claude-3-5-sonnet-20241022": {"max_tokens": 200000, "cost_per_1k_tokens": {"input": 0.003, "output": 0.015}},
            "claude-3-haiku-20240307": {"max_tokens": 200000, "cost_per_1k_tokens": {"input": 0.00025, "output": 0.00125}},
        },
        "google": {
            "gemini-1.5-pro": {"max_tokens": 1000000, "cost_per_1k_tokens": {"input": 0.0035, "output": 0.0105}},
            "gemini-1.5-flash": {"max_tokens": 1000000, "cost_per_1k_tokens": {"input": 0.000075, "output": 0.0003}},
            "gemini-2.0-flash-lite": {"max_tokens": 1000000, "cost_per_1k_tokens": {"input": 0.000075, "output": 0.0003}},
        }
    }
    
    @staticmethod
    def create_llm(
        provider: Optional[str] = None, 
        model: Optional[str] = None, 
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Create an LLM instance based on provider and model using CrewAI's native LLM class.
        
        Args:
            provider: LLM provider ('openai', 'anthropic', 'google')
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens for response
            **kwargs: Additional model-specific parameters
            
        Returns:
            CrewAI LLM instance
        """
        provider = provider or Config.DEFAULT_LLM_PROVIDER
        model = model or Config.DEFAULT_LLM_MODEL
        
        # Enhanced validation and debugging
        logger.info(f"Creating LLM: {provider}/{model} with temp={temperature}")
        
        # Validate provider and model
        if not LLMFactory._validate_model(provider, model):
            logger.warning(f"Model {model} not in known specs for {provider}, proceeding anyway")
        
        # Pre-creation validation
        try:
            LLMFactory._validate_provider_config(provider)
        except Exception as e:
            logger.error(f"Provider validation failed for {provider}: {str(e)}")
            raise ValueError(f"Invalid configuration for {provider}: {str(e)}")

        try:
            if provider == "openai":
                return LLMFactory._create_openai_llm(model, temperature, max_tokens, **kwargs)
            elif provider == "anthropic":
                return LLMFactory._create_anthropic_llm(model, temperature, max_tokens, **kwargs)
            elif provider == "google":
                return LLMFactory._create_google_llm(model, temperature, max_tokens, **kwargs)
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
                
        except Exception as e:
            logger.error(f"Failed to create LLM {provider}/{model}: {str(e)}")
            # Enhanced error message for common issues
            if "env already loaded" in str(e).lower():
                raise ValueError(f"Environment configuration conflict detected. Try restarting the application or check for multiple dotenv loads. Original error: {str(e)}")
            elif "api" in str(e).lower() and "key" in str(e).lower():
                raise ValueError(f"API key issue for {provider}. Please check your .env file and ensure {provider.upper()}_API_KEY is set. Original error: {str(e)}")
            else:
                raise
    
    @staticmethod
    def _create_openai_llm(model: str, temperature: float, max_tokens: Optional[int], **kwargs):
        """Create OpenAI LLM instance using CrewAI's LLM class."""
        if not Config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not configured")
        
        params = {
            "model": f"openai/{model}",
            "api_key": Config.OPENAI_API_KEY,
            "temperature": temperature,
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        # Add additional parameters
        if kwargs.get("top_p"):
            params["top_p"] = kwargs["top_p"]
        if kwargs.get("frequency_penalty"):
            params["frequency_penalty"] = kwargs["frequency_penalty"]
        if kwargs.get("presence_penalty"):
            params["presence_penalty"] = kwargs["presence_penalty"]
            
        return LLM(**params)
    
    @staticmethod
    def _create_anthropic_llm(model: str, temperature: float, max_tokens: Optional[int], **kwargs):
        """Create Anthropic LLM instance using CrewAI's LLM class."""
        if not Config.ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key not configured")
        
        params = {
            "model": f"anthropic/{model}",
            "api_key": Config.ANTHROPIC_API_KEY,
            "temperature": temperature,
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        else:
            # Set reasonable default for crypto analysis
            params["max_tokens"] = 4000
            
        return LLM(**params)
    
    @staticmethod
    def _create_google_llm(model: str, temperature: float, max_tokens: Optional[int], **kwargs):
        """Create Google LLM instance using CrewAI's LLM class and configure generativeai."""
        if not Config.GOOGLE_API_KEY:
            raise ValueError("Google API key not configured")
        
        # Configure Google GenerativeAI client directly
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        
        # For CrewAI's LLM class, use the gemini/ prefix for LiteLLM compatibility
        if not model.startswith("gemini/"):
            model = f"gemini/{model}"
        
        params = {
            "model": model,
            "api_key": Config.GOOGLE_API_KEY,
            "temperature": temperature,
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        # Add Google-specific parameters
        if kwargs.get("top_p"):
            params["top_p"] = kwargs["top_p"]
        if kwargs.get("top_k"):
            params["top_k"] = kwargs["top_k"]
            
        return LLM(**params)
    
    @staticmethod
    def create_google_genai_client():
        """Create a direct Google GenerativeAI client for advanced usage."""
        if not Config.GOOGLE_API_KEY:
            raise ValueError("Google API key not configured")
            
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        return genai
    
    @staticmethod
    def _validate_model(provider: str, model: str) -> bool:
        """Validate if model is known for the provider."""
        return (provider in LLMFactory.MODEL_SPECS and 
                model in LLMFactory.MODEL_SPECS[provider])
    
    @staticmethod
    def _validate_provider_config(provider: str) -> bool:
        """Validate provider configuration before creating LLM."""
        if provider == "openai":
            if not Config.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
            if not Config.OPENAI_API_KEY.startswith('sk-'):
                logger.warning("OpenAI API key format looks incorrect (should start with 'sk-')")
        elif provider == "anthropic":
            if not Config.ANTHROPIC_API_KEY:
                raise ValueError("Anthropic API key not configured. Please set ANTHROPIC_API_KEY in your .env file.")
            if not Config.ANTHROPIC_API_KEY.startswith('sk-ant-'):
                logger.warning("Anthropic API key format looks incorrect (should start with 'sk-ant-')")
        elif provider == "google":
            if not Config.GOOGLE_API_KEY:
                raise ValueError("Google API key not configured. Please set GOOGLE_API_KEY in your .env file.")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        logger.info(f"Provider {provider} configuration validated successfully")
        return True
    
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
    def get_model_info(provider: str, model: str) -> Dict[str, Any]:
        """Get model specifications and capabilities."""
        if provider in LLMFactory.MODEL_SPECS:
            return LLMFactory.MODEL_SPECS[provider].get(model, {})
        return {}
    
    @staticmethod
    def get_recommended_model_for_task(task_type: str) -> Dict[str, str]:
        """Get recommended model configurations for different task types."""
        recommendations = {
            "sentiment_analysis": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "reason": "Excellent at nuanced text interpretation and context understanding"
            },
            "technical_analysis": {
                "provider": "openai", 
                "model": "gpt-4o-mini",
                "reason": "Strong analytical capabilities and structured output"
            },
            "multimodal_fusion": {
                "provider": "openai",
                "model": "gpt-4o-mini", 
                "reason": "Large context window for complex data synthesis"
            },
            "cost_optimized": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "reason": "Best balance of performance and cost"
            }
        }
        return recommendations.get(task_type, recommendations["cost_optimized"])
    
    @staticmethod
    def validate_configuration():
        """Validate that at least one LLM provider is configured."""
        providers = LLMFactory.get_available_providers()
        if not providers:
            raise ValueError(
                "No LLM providers configured. Please set at least one of: "
                "OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY"
            )
        
        logger.info(f"Available LLM providers: {', '.join(providers)}")
        return True 