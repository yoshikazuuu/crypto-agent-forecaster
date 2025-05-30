"""
Technical Analysis Agent for cryptocurrency chart patterns and indicators.
"""

from crewai import Agent
from ..tools import create_technical_analysis_tool, create_chart_analysis_tool
from ..llm_factory import LLMFactory
from ..config import Config


def create_technical_analysis_agent() -> Agent:
    """Create the TechnicalAnalysisAgent."""
    
    # Create the tool instances
    technical_tool = create_technical_analysis_tool()
    chart_analysis_tool = create_chart_analysis_tool()
    
    # Get agent-specific LLM configuration but force use of default provider/model
    agent_config = Config.get_agent_llm_config("technical")
    
    # Always use the default provider and model from environment, ignoring agent preferences
    provider = Config.DEFAULT_LLM_PROVIDER
    model = Config.DEFAULT_LLM_MODEL
    
    # Create LLM with configured settings using defaults
    llm = LLMFactory.create_llm(
        provider=provider,
        model=model,
        temperature=agent_config.get("temperature", 0.1),
        max_tokens=agent_config.get("max_tokens", 2500)
    )
    
    return Agent(
        role="Cryptocurrency Technical Analysis and Chart Pattern Specialist",
        goal="To fetch fresh OHLCV data for any cryptocurrency, process it to calculate key technical indicators, "
             "identify significant candlestick patterns, generate comprehensive visual charts, "
             "and provide both numerical and visual analysis insights.",
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
        at identifying confluence points where multiple technical signals align.
        
        You can fetch fresh market data for any cryptocurrency and always generate visual 
        charts to accompany your analysis. Your technical analysis tool automatically 
        retrieves the latest OHLCV data, so you can analyze any cryptocurrency by simply 
        specifying its name and the desired analysis period. Your visual analysis skills 
        help identify patterns that may not be immediately apparent from numerical 
        indicators alone.""",
        verbose=False,
        allow_delegation=False,
        tools=[technical_tool, chart_analysis_tool],
        llm=llm
    )


class TechnicalAnalysisAgent:
    """Wrapper class for the technical analysis agent."""
    
    def __init__(self):
        self.agent = create_technical_analysis_agent()
    
    def get_agent(self) -> Agent:
        """Get the CrewAI agent instance."""
        return self.agent 