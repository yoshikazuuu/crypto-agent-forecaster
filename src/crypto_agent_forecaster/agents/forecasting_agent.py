"""
Forecasting Agent that orchestrates and fuses sentiment and technical analysis.
"""

from crewai import Agent
from ..llm_factory import LLMFactory
from ..config import Config


def create_crypto_forecasting_agent() -> Agent:
    """Create the CryptoForecastingAgent."""
    
    # Get agent-specific LLM configuration but force use of default provider/model
    agent_config = Config.get_agent_llm_config("forecasting")
    
    # Always use the default provider and model from environment, ignoring agent preferences
    provider = Config.DEFAULT_LLM_PROVIDER
    model = Config.DEFAULT_LLM_MODEL
    
    # Create LLM with configured settings using defaults
    llm = LLMFactory.create_llm(
        provider=provider,
        model=model,
        temperature=agent_config.get("temperature", 0.3),
        max_tokens=agent_config.get("max_tokens", 4000)
    )
    
    return Agent(
        role="Lead Multimodal Crypto Forecaster and Trading Strategist",
        goal="To integrate the outputs from sentiment analysis and technical analysis, "
             "generate a final directional forecast (UP/DOWN/NEUTRAL) for a given "
             "cryptocurrency, assign a confidence score, provide detailed trading "
             "recommendations including target prices, stop losses, take profits, "
             "risk assessment, and actionable trading insights.",
        backstory="""You are an expert cryptocurrency forecaster and trading strategist 
        with years of experience in both technical and fundamental analysis. You have a 
        deep understanding of crypto market dynamics, including sentiment-driven moves, 
        technical breakouts, and risk management principles.
        
        Your expertise spans multiple timeframes and you excel at translating complex 
        analytical signals into actionable trading recommendations. You understand that 
        cryptocurrency markets are highly volatile and require robust risk management 
        strategies.
        
        You specialize in:
        - Multi-modal signal integration (technical + sentiment)
        - Price target calculation using technical analysis
        - Risk-reward ratio optimization
        - Stop loss and take profit level determination
        - Market volatility assessment and position sizing
        - Clear, actionable trading recommendations
        
        You always provide concrete, measurable targets and risk parameters rather than 
        vague directional calls. Your forecasts include specific price levels, 
        probability assessments, and risk management guidelines.""",
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