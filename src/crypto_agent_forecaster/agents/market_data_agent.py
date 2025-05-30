"""
Market Data Agent for fetching cryptocurrency data from CoinGecko.
"""

from crewai import Agent
from ..tools import create_coingecko_tool
from ..llm_factory import LLMFactory
from ..config import Config


def create_crypto_market_data_agent() -> Agent:
    """Create the CryptoMarketDataAgent."""
    
    # Create the tool instance
    coingecko_tool = create_coingecko_tool()
    
    # Get agent-specific LLM configuration
    agent_config = Config.get_agent_llm_config("market_data")
    
    # Use agent-specific provider and model instead of global defaults
    preferred_provider = agent_config.get("preferred_provider", Config.DEFAULT_LLM_PROVIDER)
    preferred_model = agent_config.get("preferred_model", Config.DEFAULT_LLM_MODEL)
    
    # Create LLM with configured settings
    llm = LLMFactory.create_llm(
        provider=preferred_provider,
        model=preferred_model,
        temperature=agent_config.get("temperature", 0.0),
        max_tokens=agent_config.get("max_tokens", 2000)
    )
    
    return Agent(
        role='Crypto Market Data Analyst',
        goal='Collect comprehensive historical market data for cryptocurrency analysis',
        backstory="""You are an expert cryptocurrency market data analyst with deep knowledge of 
        market dynamics, trading patterns, and data sources. You specialize in collecting, validating, 
        and organizing historical market data to provide accurate foundation for analysis.""",
        tools=[coingecko_tool],
        verbose=False,  # Reduced verbosity for cleaner logs
        allow_delegation=False,
        llm=llm
    )


class CryptoMarketDataAgent:
    """Wrapper class for the market data agent."""
    
    def __init__(self):
        self.agent = create_crypto_market_data_agent()
    
    def get_agent(self) -> Agent:
        """Get the CrewAI agent instance."""
        return self.agent 