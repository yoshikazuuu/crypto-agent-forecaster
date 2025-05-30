"""
Technical analysis prompts for CryptoAgentForecaster.
"""

class TechnicalPrompts:
    """Collection of prompt templates for technical analysis tasks."""
    
    TECHNICAL_SUMMARY_GENERATOR = """
    You are a professional cryptocurrency technical analyst. Generate a comprehensive 
    technical analysis summary based on the provided data.
    
    Cryptocurrency: {cryptocurrency}
    Current Price: ${current_price}
    
    Technical Indicators:
    {technical_indicators}
    
    Candlestick Patterns (if any):
    {candlestick_patterns}
    
    Volume Information:
    {volume_info}
    
    Price Action Summary:
    {price_action}
    
    Generate a comprehensive technical analysis summary that includes:
    1. Current trend direction (Bullish/Bearish/Neutral)
    2. Key support and resistance levels
    3. Momentum indicators interpretation
    4. Volume analysis
    5. Notable chart patterns
    6. Overall technical outlook
    7. Risk factors and key levels to watch
    
    Provide your analysis in the following JSON format:
    {{
        "trend_direction": "Bullish/Bearish/Neutral",
        "trend_strength": "Strong/Moderate/Weak",
        "support_levels": [level1, level2],
        "resistance_levels": [level1, level2],
        "momentum": "Bullish/Bearish/Neutral",
        "volume_analysis": "summary",
        "key_patterns": ["pattern1", "pattern2"],
        "technical_score": 0.0,
        "outlook": "short description",
        "risk_factors": ["risk1", "risk2"],
        "key_levels": [level1, level2],
        "summary": "Comprehensive technical analysis summary"
    }}
    """
    
    CANDLESTICK_PATTERN_INTERPRETER = """
    You are a technical analysis expert specializing in candlestick patterns.
    
    Based on the OHLCV data provided, identify and interpret any significant 
    candlestick patterns for {cryptocurrency}.
    
    Recent Candlestick Data:
    {ohlcv_data}
    
    For each identified pattern, provide:
    1. Pattern name
    2. Formation date
    3. Bullish/Bearish implication
    4. Reliability/Strength
    5. Expected price target or continuation
    
    Common patterns to look for:
    - Doji, Hammer, Shooting Star
    - Engulfing patterns (Bullish/Bearish)
    - Morning Star, Evening Star
    - Three White Soldiers, Three Black Crows
    - Hanging Man, Inverted Hammer
    
    Respond in JSON format:
    {{
        "patterns_found": [
            {{
                "name": "pattern_name",
                "date": "YYYY-MM-DD",
                "type": "Bullish/Bearish/Neutral",
                "strength": "Strong/Moderate/Weak",
                "description": "detailed description",
                "implication": "expected market reaction"
            }}
        ],
        "overall_pattern_sentiment": "Bullish/Bearish/Neutral",
        "pattern_summary": "summary of all patterns"
    }}
    """
    
    INDICATOR_ANALYSIS = """
    You are analyzing technical indicators for {cryptocurrency} trading decisions.
    
    Current Technical Indicators:
    
    Moving Averages:
    - SMA 20: {sma_20}
    - SMA 50: {sma_50}  
    - SMA 200: {sma_200}
    - EMA 12: {ema_12}
    - EMA 26: {ema_26}
    
    Momentum Indicators:
    - RSI (14): {rsi}
    - MACD: {macd_line}
    - MACD Signal: {macd_signal}
    - MACD Histogram: {macd_histogram}
    
    Volatility:
    - Bollinger Bands Upper: {bb_upper}
    - Bollinger Bands Lower: {bb_lower}
    - Bollinger Bands %B: {bb_percent_b}
    
    Current Price: {current_price}
    
    Analyze these indicators and provide:
    1. Moving average trend analysis
    2. Momentum interpretation (RSI, MACD)
    3. Volatility assessment (Bollinger Bands)
    4. Cross-over signals
    5. Overbought/oversold conditions
    6. Overall technical score (-1 to +1)
    
    Respond in JSON format:
    {{
        "ma_analysis": "moving average trend analysis",
        "momentum_analysis": "RSI and MACD interpretation", 
        "volatility_analysis": "Bollinger Bands assessment",
        "signals": ["signal1", "signal2"],
        "overbought_oversold": "assessment",
        "technical_score": 0.0,
        "summary": "overall technical indicators summary"
    }}
    """
    
    VOLUME_PRICE_ANALYSIS = """
    You are analyzing volume and price relationship for {cryptocurrency}.
    
    Price Data:
    {price_data}
    
    Volume Data:
    {volume_data}
    
    Volume Indicators:
    - Average Volume (20-day): {avg_volume}
    - Volume Ratio (current/average): {volume_ratio}
    - On-Balance Volume: {obv}
    
    Analyze:
    1. Volume trend vs price trend
    2. Volume spikes and their significance
    3. Price-volume divergences
    4. Volume confirmation of price moves
    5. Accumulation/Distribution signals
    
    Respond in JSON format:
    {{
        "volume_trend": "Increasing/Decreasing/Stable",
        "price_volume_relationship": "Confirmed/Diverging/Neutral",
        "volume_spikes": ["date: significance"],
        "accumulation_distribution": "Accumulation/Distribution/Neutral",
        "volume_analysis": "comprehensive volume analysis",
        "volume_score": 0.0
    }}
    """

def get_technical_prompts():
    """Get technical analysis prompt templates."""
    return {
        "technical_summary_generator": TechnicalPrompts.TECHNICAL_SUMMARY_GENERATOR,
        "candlestick_pattern_interpreter": TechnicalPrompts.CANDLESTICK_PATTERN_INTERPRETER,
        "indicator_analysis": TechnicalPrompts.INDICATOR_ANALYSIS,
        "volume_price_analysis": TechnicalPrompts.VOLUME_PRICE_ANALYSIS,
    }

__all__ = ["TechnicalPrompts", "get_technical_prompts"] 