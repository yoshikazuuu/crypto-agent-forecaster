"""
Forecasting Agent that orchestrates and fuses sentiment and technical analysis.
"""

from crewai import Agent
from ..llm_factory import LLMFactory
from ..config import Config


def create_crypto_forecasting_agent() -> Agent:
    """Create the CryptoForecastingAgent."""
    
    # Get agent-specific LLM configuration
    agent_config = Config.get_agent_llm_config("forecasting")
    
    # Create LLM with configured settings
    llm = LLMFactory.create_llm(
        provider=Config.DEFAULT_LLM_PROVIDER,
        model=Config.DEFAULT_LLM_MODEL,
        temperature=agent_config.get("temperature", 0.3),
        max_tokens=agent_config.get("max_tokens", 4000)
    )
    
    return Agent(
        role="Lead Multimodal Crypto Forecaster and Synthesizer",
        goal="To integrate the outputs from sentiment analysis and technical analysis, "
             "generate a final directional forecast (UP/DOWN/NEUTRAL) for a given "
             "cryptocurrency, assign a confidence score, and provide an explanation "
             "for the forecast.",
        backstory="""You are the chief strategist and decision-maker of the crypto 
        forecasting team. You possess a unique ability to synthesize complex, 
        multimodal information streams and extract actionable trading insights.
        
        Your expertise spans both quantitative technical analysis and qualitative 
        sentiment interpretation. You understand how market psychology, as reflected 
        in social media discussions and news sentiment, interacts with price action 
        and technical indicators to drive cryptocurrency movements.
        
        You are particularly skilled at handling conflicting signals - when technical 
        analysis suggests one direction but sentiment analysis suggests another. You 
        weigh evidence carefully, considering factors like data source reliability, 
        market volatility, and the current macro environment.
        
        Your forecasts are always accompanied by clear reasoning and appropriate 
        confidence levels. You understand that in the volatile crypto market, 
        humility and risk awareness are as important as analytical skill.""",
        verbose=False,
        allow_delegation=False,
        tools=[],  # Uses other agents' outputs rather than direct tools
        llm=llm
    )


class CryptoForecastingAgent:
    """Wrapper class for the forecasting agent."""
    
    def __init__(self):
        self.agent = create_crypto_forecasting_agent()
    
    def get_agent(self) -> Agent:
        """Get the CrewAI agent instance."""
        return self.agent 