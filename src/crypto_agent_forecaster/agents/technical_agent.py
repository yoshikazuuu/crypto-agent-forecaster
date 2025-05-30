"""
Technical Analysis Agent for cryptocurrency chart patterns and indicators.
"""

from crewai import Agent
from ..tools import create_technical_analysis_tool
from ..llm_factory import LLMFactory


def create_technical_analysis_agent() -> Agent:
    """Create the TechnicalAnalysisAgent."""
    
    return Agent(
        role="Cryptocurrency Technical Analysis and Chart Pattern Specialist",
        goal="To process historical OHLCV data, calculate key technical indicators, "
             "identify significant candlestick patterns, and generate a concise textual "
             "summary of the technical outlook for a target cryptocurrency.",
        backstory="""You are a seasoned technical analyst with extensive experience in 
        cryptocurrency markets. You understand the nuances of crypto price action, which 
        can be more volatile and sentiment-driven than traditional markets.
        
        Your expertise includes identifying classic chart patterns (triangles, flags, 
        head and shoulders, etc.), candlestick formations (engulfing patterns, dojis, 
        hammers, shooting stars), and interpreting momentum indicators (RSI, MACD), 
        trend indicators (moving averages), and volatility measures (Bollinger Bands).
        
        You excel at translating complex technical data into clear, actionable insights. 
        You understand that in crypto markets, technical analysis must be combined with 
        fundamental and sentiment analysis for optimal results. You're particularly adept 
        at identifying confluence points where multiple technical signals align.""",
        verbose=True,
        allow_delegation=False,
        tools=[create_technical_analysis_tool()],
        llm=LLMFactory.create_llm(temperature=0.1)  # Lower temperature for precise technical analysis
    )


class TechnicalAnalysisAgent:
    """Wrapper class for the technical analysis agent."""
    
    def __init__(self):
        self.agent = create_technical_analysis_agent()
    
    def get_agent(self) -> Agent:
        """Get the CrewAI agent instance."""
        return self.agent 