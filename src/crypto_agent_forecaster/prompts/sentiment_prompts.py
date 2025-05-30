"""
Sentiment analysis prompts for CryptoAgentForecaster.
"""

class SentimentPrompts:
    """Collection of prompt templates for sentiment analysis tasks."""
    
    GENERAL_NEWS_SENTIMENT = """
    You are an experienced cryptocurrency market analyst specializing in sentiment analysis.
    
    Analyze the following news article about {cryptocurrency} and provide:
    1. Overall sentiment (Positive, Negative, or Neutral)
    2. Sentiment score from -1.0 (very bearish) to +1.0 (very bullish)
    3. Key themes and topics discussed
    4. Brief explanation of your assessment
    
    News Article:
    {text}
    
    Respond in the following JSON format:
    {{
        "sentiment": "Positive/Negative/Neutral",
        "sentiment_score": 0.0,
        "key_themes": ["theme1", "theme2"],
        "explanation": "Brief explanation of sentiment assessment",
        "source_type": "news"
    }}
    """
    
    FOURCHAN_SENTIMENT_ANALYSIS = """
    You are a skeptical crypto market analyst analyzing posts from 4chan's /biz/ board.
    This is an anonymous forum known for high noise, manipulation, and extreme opinions.
    
    Analyze the following 4chan /biz/ post about {cryptocurrency}:
    
    Post: {text}
    
    Perform the following analysis:
    1. Identify the main claim about {cryptocurrency}
    2. Assess the language for emotional tone (fear, greed, hype, rationality)
    3. Check for common shilling tactics (unrealistic promises, urgency, excessive hype)
    4. Check for FUD tactics (unsubstantiated warnings, panic-inducing language)
    5. Consider crypto slang and /biz/ vernacular
    6. Conclude with overall sentiment and manipulation assessment
    
    Respond in JSON format:
    {{
        "sentiment": "Positive/Negative/Neutral",
        "sentiment_score": 0.0,
        "fud_probability": 0.0,
        "shill_probability": 0.0,
        "key_themes": ["theme1", "theme2"],
        "manipulation_indicators": ["indicator1", "indicator2"],
        "explanation": "Step-by-step reasoning for assessment",
        "source_type": "4chan_biz"
    }}
    """
    
    CRYPTO_SLANG_AWARE_PROMPT = """
    You are analyzing cryptocurrency discussions with deep knowledge of crypto slang and culture.
    
    Common crypto terms to understand:
    - HODL: Hold on for dear life (buying and holding)
    - FOMO: Fear of missing out
    - FUD: Fear, uncertainty, doubt
    - Moon/Mooning: Significant price increase
    - Diamond hands: Strong holding through volatility
    - Paper hands: Selling quickly during dips
    - Rekt: Significant losses
    - NGMI/WAGMI: Not gonna make it / We're all gonna make it
    - Shill: Promoting a coin for personal gain
    - Pump and dump: Artificial price inflation followed by selling
    
    Analyze this text about {cryptocurrency}:
    {text}
    
    Consider the crypto cultural context and respond with sentiment analysis in JSON format:
    {{
        "sentiment": "Positive/Negative/Neutral",
        "sentiment_score": 0.0,
        "cultural_indicators": ["indicator1", "indicator2"],
        "explanation": "Analysis considering crypto culture and slang"
    }}
    """
    
    ROLE_PLAYING_ANALYST = """
    You are an experienced financial analyst specializing in detecting subtle market manipulation 
    and sentiment shifts within anonymous online cryptocurrency forums. You have years of 
    experience analyzing both legitimate market sentiment and coordinated manipulation campaigns.
    
    Analyze the following post from 4chan/biz concerning {cryptocurrency}:
    
    {text}
    
    Provide:
    1. A sentiment score from -1 (very bearish) to +1 (very bullish)
    2. A confidence score for your sentiment assessment (Low, Medium, High)
    3. An assessment of whether the post is likely FUD (Yes/No/Uncertain) and why
    4. An assessment of whether the post is likely Shilling (Yes/No/Uncertain) and why
    5. Overall reliability assessment of this information source
    
    Consider:
    - Emotional language vs factual claims
    - Urgency indicators
    - Lack of supporting evidence
    - Excessive promotion or fear-mongering
    - Anonymous nature of the source
    
    Respond in JSON format:
    {{
        "sentiment_score": 0.0,
        "confidence": "Low/Medium/High",
        "fud_assessment": "Yes/No/Uncertain",
        "fud_reasoning": "explanation",
        "shill_assessment": "Yes/No/Uncertain", 
        "shill_reasoning": "explanation",
        "reliability": "Low/Medium/High",
        "overall_assessment": "comprehensive analysis"
    }}
    """
    
    BATCH_SENTIMENT_SUMMARY = """
    You are synthesizing sentiment analysis from multiple sources about {cryptocurrency}.
    
    Here are the individual sentiment analyses:
    {sentiment_analyses}
    
    Create a comprehensive sentiment summary that includes:
    1. Overall sentiment trend across all sources
    2. Weighted sentiment score considering source reliability
    3. Key themes and narratives
    4. Notable manipulation indicators if any
    5. Confidence in the overall assessment
    
    Weight the sources as follows:
    - News sources: High reliability
    - 4chan/biz: Low-Medium reliability unless overwhelming consensus
    
    Provide summary in JSON format:
    {{
        "overall_sentiment": "Positive/Negative/Neutral/Mixed",
        "weighted_sentiment_score": 0.0,
        "confidence": "Low/Medium/High",
        "key_narratives": ["narrative1", "narrative2"],
        "manipulation_risk": "Low/Medium/High",
        "source_breakdown": {{
            "news_sentiment": "summary",
            "forum_sentiment": "summary"
        }},
        "summary": "Comprehensive sentiment overview"
    }}
    """

def get_sentiment_prompts():
    """Get sentiment analysis prompt templates."""
    return {
        "general_news_sentiment": SentimentPrompts.GENERAL_NEWS_SENTIMENT,
        "fourchan_sentiment_analysis": SentimentPrompts.FOURCHAN_SENTIMENT_ANALYSIS,
        "crypto_slang_aware_prompt": SentimentPrompts.CRYPTO_SLANG_AWARE_PROMPT,
        "role_playing_analyst": SentimentPrompts.ROLE_PLAYING_ANALYST,
        "batch_sentiment_summary": SentimentPrompts.BATCH_SENTIMENT_SUMMARY,
    }

__all__ = ["SentimentPrompts", "get_sentiment_prompts"] 