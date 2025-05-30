"""
Market Data Agent for fetching cryptocurrency data from CoinGecko.
"""

from crewai import Agent
from ..tools import create_coingecko_tool
from ..llm_factory import LLMFactory


def create_crypto_market_data_agent() -> Agent:
    """Create the CryptoMarketDataAgent."""
    
    return Agent(
        role="Cryptocurrency Market Data Retrieval Specialist",
        goal="To fetch accurate, timely, and relevant historical and current market data "
             "(OHLCV, volume, market capitalization, etc.) for specified cryptocurrencies "
             "from the CoinGecko API.",
        backstory="""You are an expert at gathering cryptocurrency market data from the 
        CoinGecko API. You understand the intricacies of crypto market data, including 
        OHLCV (Open, High, Low, Close, Volume) data, market capitalization, and trading 
        volumes. You are meticulous about data quality and always ensure that you fetch 
        the most relevant and recent data for analysis.
        
        You handle API rate limits responsibly and can adapt to different data granularities 
        based on the analysis requirements. You understand that accurate market data is the 
        foundation for all subsequent technical analysis and forecasting.""",
        verbose=True,
        allow_delegation=False,
        tools=[create_coingecko_tool()],
        llm=LLMFactory.create_llm()
    )


class CryptoMarketDataAgent:
    """Wrapper class for the market data agent."""
    
    def __init__(self):
        self.agent = create_crypto_market_data_agent()
    
    def get_agent(self) -> Agent:
        """Get the CrewAI agent instance."""
        return self.agent 