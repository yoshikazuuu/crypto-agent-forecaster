"""
Prompts package for CryptoAgentForecaster.
"""

from .sentiment_prompts import get_sentiment_prompts
from .technical_prompts import get_technical_prompts  
from .fusion_prompts import get_fusion_prompts


def get_task_prompts():
    """Get task descriptions for CrewAI agents."""
    return {
        "market_data_task": """
        Fetch comprehensive market data for {crypto_name} to support the {forecast_horizon} forecast.
        
        Your task:
        1. Get historical OHLCV data for the past 30 days with daily granularity
        2. Fetch current market statistics (price, volume, market cap)
        3. Ensure data quality and handle any API errors gracefully
        4. Format the data in JSON structure suitable for technical analysis
        
        Focus on accuracy and completeness of the data as it forms the foundation for all subsequent analysis.
        """,
        
        "sentiment_analysis_task": """
        Analyze sentiment and market narratives for {crypto_name} over the {forecast_horizon} forecast period.
        
        Your task:
        1. Gather recent discussions about {crypto_name} from 4chan /biz/ board
        2. Search for keywords: [{crypto_name}, btc, bitcoin, crypto, coin, moon, pump, dump]
        3. Analyze sentiment with special focus on:
           - Overall sentiment (positive/negative/neutral)
           - FUD detection (fear, uncertainty, doubt campaigns)
           - Shilling detection (artificial promotion attempts)
           - Key narrative themes and topics
        4. Provide sentiment scores and confidence levels
        5. Identify any coordinated or suspicious activity patterns
        
        Be especially careful to distinguish genuine sentiment from manipulation attempts.
        """,
        
        "technical_analysis_task": """
        Perform comprehensive technical analysis on {crypto_name} for the {forecast_horizon} forecast.
        
        Your task:
        1. Use the OHLCV data from the market data agent
        2. Calculate key technical indicators (RSI, MACD, Moving Averages, Bollinger Bands)
        3. Identify significant candlestick patterns
        4. Assess trend direction and momentum
        5. Determine support and resistance levels
        6. Provide an overall technical outlook (bullish/bearish/neutral)
        
        Generate a clear textual summary that can be easily understood and integrated with sentiment analysis.
        """,
        
        "forecasting_task": """
        Generate the final {forecast_horizon} forecast for {crypto_name} by integrating sentiment and technical analysis.
        
        Your task:
        1. Review the sentiment analysis findings, paying attention to:
           - Overall sentiment scores
           - FUD/shill detection results
           - Key narrative themes
           - Source reliability considerations
        
        2. Review the technical analysis findings, focusing on:
           - Technical indicator signals
           - Chart pattern implications
           - Trend and momentum assessment
           - Overall technical outlook
        
        3. Synthesize the information considering:
           - Agreement or conflict between sentiment and technical signals
           - Reliability and strength of each signal type
           - Market volatility and uncertainty factors
           - Appropriate confidence levels
        
        4. Provide your final forecast including:
           - Direction: UP, DOWN, or NEUTRAL
           - Confidence level: HIGH, MEDIUM, or LOW
           - Detailed explanation of your reasoning
           - Key factors that influenced your decision
           - Risk considerations and caveats
        
        Remember: In crypto markets, sentiment can drive short-term movements, but technical analysis often provides better risk/reward insights. Weight your analysis accordingly.
        """
    }


__all__ = [
    "get_sentiment_prompts",
    "get_technical_prompts", 
    "get_fusion_prompts",
    "get_task_prompts",
] 