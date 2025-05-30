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
        1. Use the coingecko_tool to get historical OHLCV data optimized for the {forecast_horizon} forecast horizon:
           - For hour-based forecasts (1hr, 4hr, 12hr): Get 3-14 days of data
           - For day-based forecasts (1d, 3d, 7d): Get 1-3 months of data  
           - For week-based forecasts: Get 3-4 months of data
           - For month-based forecasts: Get 1 year of data
        2. Fetch current market statistics (price, volume, market cap)
        3. Analyze the data quality and key metrics
        
        IMPORTANT: Use the query format: "{crypto_name} ohlcv {forecast_horizon} horizon" to get optimal data amount.
        Example: coingecko_tool(query="{crypto_name} ohlcv {forecast_horizon} horizon")
        
        CRITICAL: Do NOT include the full OHLCV dataset in your response. Instead, provide only a CONCISE SUMMARY containing:
        - Current market price with timestamp
        - Data period covered (start date to end date)
        - Number of data points collected
        - Key price levels (recent high, low, average)
        - Volume trends and market cap information
        - Data quality assessment
        - Any notable patterns or anomalies in the dataset
        
        Your response should be a brief, informative summary (under 500 words) that provides context for analysis without overwhelming other agents with raw data.
        
        Make sure to clearly state the current price at the end of your response in the format:
        "Current market price: $[PRICE]"
        """,
        
        "sentiment_analysis_task": """
        Analyze sentiment and market narratives for {crypto_name} over the {forecast_horizon} forecast period.
        
        Your task:
        1. Gather recent discussions about {crypto_name} from 4chan /biz/ board using the fourchan_tool
        2. Use the fourchan_tool with these parameters:
           - keywords: A list of relevant search terms like ["{crypto_name}", "btc", "bitcoin", "crypto", "coin", "moon", "pump", "dump"]
           - max_threads: 5 (default, can be adjusted based on needs)
           - max_posts_per_thread: 20 (default, can be adjusted based on needs)
        3. Analyze sentiment with special focus on:
           - Overall sentiment (positive/negative/neutral)
           - FUD detection (fear, uncertainty, doubt campaigns)
           - Shilling detection (artificial promotion attempts)
           - Key narrative themes and topics
        4. Provide sentiment scores and confidence levels
        5. Identify any coordinated or suspicious activity patterns
        
        Example tool usage:
        fourchan_tool(keywords=["{crypto_name}", "btc", "bitcoin", "crypto", "coin", "moon", "pump", "dump"], max_threads=5, max_posts_per_thread=20)
        
        Be especially careful to distinguish genuine sentiment from manipulation attempts.
        """,
        
        "technical_analysis_task": """
        Perform comprehensive technical analysis on {crypto_name} for the {forecast_horizon} forecast.
        
        Your task:
        1. Use the technical_analysis_tool directly with the cryptocurrency name:
           - The tool will automatically fetch fresh OHLCV data internally
           - It will fetch the optimal amount of historical data for analysis
           - Specify the number of days for the analysis if needed (default is 30 days)
        
        2. Process the technical analysis results and create a comprehensive summary that includes:
           - Key technical indicators (RSI, MACD, Moving Averages, Bollinger Bands) and their signals
           - Significant candlestick patterns identified
           - Trend direction and momentum assessment
           - Critical support and resistance levels
           - Overall technical outlook (bullish/bearish/neutral)
           - Chart patterns and their implications
        
        CRITICAL: 
        - Use technical_analysis_tool(crypto_name="{crypto_name}", days=30) or similar
        - The tool handles all data fetching internally
        - Focus on ANALYSIS and INSIGHTS, not data replication
        - Provide a clear, actionable technical summary
        - Include specific price levels and percentages where relevant
        
        Example tool usage:
        technical_analysis_tool(crypto_name="{crypto_name}", days=30)
        
        Your response should focus on what the technical indicators are telling us about future price movement, with specific actionable insights for the {forecast_horizon} timeframe.
        """,
        
        "forecasting_task": """
        Generate the final comprehensive {forecast_horizon} forecast and trading strategy for {crypto_name} by integrating sentiment and technical analysis.
        
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
           - Support and resistance levels
           - Overall technical outlook
        
        3. Synthesize the information considering:
           - Agreement or conflict between sentiment and technical signals
           - Reliability and strength of each signal type
           - Market volatility and uncertainty factors
           - Appropriate confidence levels
        
        4. Provide your final comprehensive forecast in this EXACT format:
        
        **Direction**: [UP/DOWN/NEUTRAL]
        **Confidence level**: [HIGH/MEDIUM/LOW]
        **Current Price**: $[price]
        **Target Price(s)**:
        - **Primary Target**: $[price] (Probability: [%]%) - [reasoning]
        - **Secondary Target**: $[price] (Probability: [%]%) - [reasoning]
        
        **Stop Loss Level**: $[price] - [reasoning]
        **Take Profit Level(s)**:
        - **Take Profit 1**: $[price] - [reasoning]
        - **Take Profit 2**: $[price] - [reasoning]
        
        **Risk-Reward Ratio**: [ratio] ([calculation])
        **Position Size Recommendation**: [percentage]% of portfolio - [reasoning]
        **Time Horizon**: [timeframe] - [reasoning]
        **Key Catalysts**: [list of events/factors]
        **Risk Factors**: [list of risks]
        **Entry Strategy**: [detailed entry plan]
        **Exit Strategy**: [detailed exit plan]
        **Market Context**: [broader market analysis]
        
        **Detailed Analysis & Reasoning**:
        [Provide comprehensive step-by-step analysis that supports your direction, targets, and risk levels. Explain how you integrated sentiment and technical signals. Be specific about what indicators or sentiment factors led to your conclusion.]
        
        IMPORTANT REQUIREMENTS:
        - The **Direction** field MUST match your overall conclusion in the detailed analysis
        - Use specific price levels, not ranges or vague terms
        - Include probability estimates for targets
        - Provide clear risk management guidelines
        - Your detailed analysis should support and align with your direction choice
        - Be consistent: if you say UP in direction, your analysis should primarily support bullish outlook
        - If conflicted signals, choose NEUTRAL and explain the uncertainty
        """
    }


__all__ = [
    "get_sentiment_prompts",
    "get_technical_prompts", 
    "get_fusion_prompts",
    "get_task_prompts",
] 