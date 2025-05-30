"""
Sentiment Analysis Agent for cryptocurrency sentiment from news and 4chan data.
"""

from crewai import Agent
from ..tools import create_fourchan_tool
from ..llm_factory import LLMFactory


def create_crypto_sentiment_analysis_agent() -> Agent:
    """Create the CryptoSentimentAnalysisAgent."""
    
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
        of potential manipulation.""",
        verbose=True,
        allow_delegation=False,
        tools=[create_fourchan_tool()],
        llm=LLMFactory.create_llm(temperature=0.2)  # Slightly higher temperature for nuanced analysis
    )


class CryptoSentimentAnalysisAgent:
    """Wrapper class for the sentiment analysis agent."""
    
    def __init__(self):
        self.agent = create_crypto_sentiment_analysis_agent()
    
    def get_agent(self) -> Agent:
        """Get the CrewAI agent instance."""
        return self.agent 