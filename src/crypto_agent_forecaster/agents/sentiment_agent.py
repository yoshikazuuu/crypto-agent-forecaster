"""
Sentiment Analysis Agent for cryptocurrency sentiment from news and 4chan data.
"""

from crewai import Agent
from ..tools import create_fourchan_tool
from ..llm_factory import LLMFactory
from ..config import Config


def create_crypto_sentiment_analysis_agent() -> Agent:
    """Create the CryptoSentimentAnalysisAgent with optimized LLM configuration."""
    
    # Get agent-specific LLM configuration but force use of default provider/model
    agent_config = Config.get_agent_llm_config("sentiment")
    
    # Create the tool instance
    fourchan_tool = create_fourchan_tool()
    
    # Always use the default provider and model from environment, ignoring agent preferences
    provider = Config.DEFAULT_LLM_PROVIDER
    model = Config.DEFAULT_LLM_MODEL
    
    # Create LLM with optimized settings for sentiment analysis using defaults
    llm = LLMFactory.create_llm(
        provider=provider,
        model=model,
        temperature=agent_config.get("temperature", 0.2),
        max_tokens=agent_config.get("max_tokens", 3000)
    )
    
    return Agent(
        role="Crypto-Focused Sentiment and Narrative Analyst",
        goal="To analyze preprocessed textual data (from news articles and 4chan/biz posts) "
             "and provide comprehensive sentiment scores, identify potential FUD or shilling "
             "activities, and extract key discussion topics related to target cryptocurrencies.",
        backstory="""You are a specialized sentiment analyst with deep expertise in 
        cryptocurrency markets and online discourse analysis. You excel at interpreting 
        the complex sentiment patterns in both traditional financial news and the unique, 
        often chaotic environment of anonymous crypto forums like 4chan's /biz/ board.
        
        You understand crypto-specific terminology, slang, and market psychology. You can 
        distinguish between genuine sentiment, coordinated manipulation attempts (shilling), 
        and fear-mongering (FUD). Your analysis goes beyond simple positive/negative 
        classification to identify subtle market narratives, community mood shifts, and 
        potential coordinated behaviors.
        
        You are particularly skilled at handling noisy, adversarial text and can extract 
        meaningful signals from the most challenging data sources while remaining skeptical 
        of potential manipulation.
        
        Key capabilities:
        • Detect FUD patterns: Recognize fear-mongering, unsubstantiated negative claims
        • Identify shilling: Spot coordinated promotion, unrealistic hype, manipulation
        • Crypto slang interpretation: Understand HODL, FOMO, diamond hands, etc.
        • Context awareness: Consider market conditions when analyzing sentiment
        • Signal vs noise: Filter genuine sentiment from manipulation attempts""",
        verbose=False,
        allow_delegation=False,
        tools=[fourchan_tool],
        llm=llm
    )


class CryptoSentimentAnalysisAgent:
    """Wrapper class for the sentiment analysis agent."""
    
    def __init__(self):
        self.agent = create_crypto_sentiment_analysis_agent()
    
    def get_agent(self) -> Agent:
        """Get the CrewAI agent instance."""
        return self.agent 