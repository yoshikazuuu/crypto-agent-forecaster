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
    
    # Get agent-specific LLM configuration but force use of default provider/model
    agent_config = Config.get_agent_llm_config("market_data")
    
    # Always use the default provider and model from environment, ignoring agent preferences
    provider = Config.DEFAULT_LLM_PROVIDER
    model = Config.DEFAULT_LLM_MODEL
    
    # Create LLM with configured settings using defaults
    llm = LLMFactory.create_llm(
        provider=provider,
        model=model,
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