"""
Fusion prompts for combining sentiment and technical analysis in CryptoAgentForecaster.
"""

class FusionPrompts:
    """Collection of prompt templates for multimodal fusion tasks."""
    
    MASTER_FUSION_PROMPT = """
    You are the Lead Cryptocurrency Forecaster for a sophisticated AI trading system. 
    Your task is to synthesize sentiment analysis and technical analysis to predict 
    the price direction of {cryptocurrency} over the next {time_horizon}.
    
    === TECHNICAL ANALYSIS SUMMARY ===
    {technical_summary}
    
    === SENTIMENT ANALYSIS SUMMARY ===
    {sentiment_summary}
    
    === CURRENT MARKET CONTEXT ===
    Current Price: ${current_price}
    24h Volume: ${volume_24h}
    Market Volatility: {volatility_level}
    
    === FUSION INSTRUCTIONS ===
    Consider both the sentiment analysis and technical analysis provided. Follow these guidelines:
    
    1. If signals strongly agree, reflect high confidence in your prediction
    2. If signals conflict, critically evaluate the reliability of each source:
       - Technical patterns confirmed by high volume should be given significant weight
       - Sentiment from verified news may be more reliable than anonymous forum posts
       - However, overwhelming forum sentiment consistency can indicate emerging trends
    3. Consider the current market volatility in your confidence assessment
    4. Factor in the source reliability (news vs 4chan/biz forums)
    5. Provide step-by-step reasoning for your forecast
    
    === REQUIRED OUTPUT ===
    Predict the price direction for {cryptocurrency} and provide:
    
    1. **Directional Forecast**: UP, DOWN, or NEUTRAL
    2. **Confidence Score**: Low, Medium, or High
    3. **Reasoning**: Step-by-step explanation of how you weighed the evidence
    4. **Key Factors**: Most influential factors in your decision
    5. **Risk Assessment**: Potential risks to your forecast
    6. **Numerical Confidence**: 0.0 to 1.0 score
    
    Respond in the following JSON format:
    {{
        "forecast": "UP/DOWN/NEUTRAL",
        "confidence": "Low/Medium/High", 
        "confidence_score": 0.0,
        "reasoning": "Detailed step-by-step reasoning",
        "key_factors": ["factor1", "factor2", "factor3"],
        "technical_weight": 0.0,
        "sentiment_weight": 0.0,
        "risk_factors": ["risk1", "risk2"],
        "time_horizon": "{time_horizon}",
        "summary": "Brief forecast summary"
    }}
    """
    
    FS_REASONING_ADAPTED_PROMPT = """
    You are a cryptocurrency forecasting system implementing Fact-Subjectivity Aware Reasoning.
    
    Analyze the following inputs for {cryptocurrency} by explicitly separating factual 
    and subjective information streams:
    
    === FACTUAL DATA STREAM (Technical Analysis) ===
    {technical_summary}
    
    === MODERATELY SUBJECTIVE STREAM (News Sentiment) ===  
    {news_sentiment}
    
    === HIGHLY SUBJECTIVE STREAM (4chan/biz Sentiment) ===
    {forum_sentiment}
    
    === ANALYSIS FRAMEWORK ===
    
    **Step 1: Factual Analysis**
    Based on the technical data, what is the objective price trend indicated?
    
    **Step 2: News Sentiment Analysis** 
    What is the prevailing mood from curated financial news sources?
    
    **Step 3: Forum Sentiment Analysis**
    What signals emerge from 4chan/biz, noting its speculative and potentially manipulative nature?
    
    **Step 4: Synthesis**
    Combine these three perspectives, giving appropriate weight based on reliability:
    - Technical data: High weight (factual, objective)
    - News sentiment: Medium weight (evidence-based but subjective)
    - Forum sentiment: Low-Medium weight (highly speculative, potential manipulation)
    
    However, if forum sentiment shows overwhelming consensus that contradicts other signals, 
    consider it as a potential early indicator.
    
    **Step 5: Final Forecast**
    Predict the price direction for {cryptocurrency} over {time_horizon}.
    
    Respond in JSON format:
    {{
        "factual_analysis": "objective technical trend assessment",
        "news_sentiment_analysis": "curated news mood assessment", 
        "forum_sentiment_analysis": "4chan/biz sentiment with manipulation notes",
        "weight_allocation": {{
            "technical": 0.0,
            "news": 0.0, 
            "forum": 0.0
        }},
        "synthesis": "how the three streams were combined",
        "forecast": "UP/DOWN/NEUTRAL",
        "confidence": "Low/Medium/High",
        "reasoning": "final reasoning for the forecast"
    }}
    """
    
    CHAIN_OF_THOUGHT_FUSION = """
    You are making a cryptocurrency price prediction for {cryptocurrency}. 
    Think through this step by step using chain-of-thought reasoning.
    
    **Available Information:**
    
    Technical Analysis:
    {technical_summary}
    
    Sentiment Analysis:  
    {sentiment_summary}
    
    Market Context:
    Current Price: ${current_price}
    Volatility: {volatility_level}
    
    **Let me think through this step by step:**
    
    **Step 1:** What does the technical analysis tell us?
    - Analyze the trend, indicators, and patterns
    - Assess the strength of technical signals
    - Consider volume confirmation
    
    **Step 2:** What does the sentiment analysis reveal?
    - Evaluate news sentiment reliability and themes
    - Assess forum sentiment while considering manipulation risk
    - Weight different sentiment sources appropriately
    
    **Step 3:** How do technical and sentiment signals compare?
    - Do they agree or conflict?
    - Which signals are stronger/more reliable?
    - Are there any contradictions that need resolution?
    
    **Step 4:** What market context factors should I consider?
    - Current volatility levels
    - General market conditions
    - Risk factors specific to {cryptocurrency}
    
    **Step 5:** What is my final assessment?
    - Integrate all information streams
    - Assign appropriate weights to different signals
    - Make final directional forecast with confidence level
    
    **Final Output:**
    Based on this analysis, provide your forecast in JSON format:
    {{
        "step1_technical": "technical analysis interpretation",
        "step2_sentiment": "sentiment analysis interpretation", 
        "step3_comparison": "comparison of signals",
        "step4_context": "market context considerations",
        "step5_integration": "final integration reasoning",
        "forecast": "UP/DOWN/NEUTRAL",
        "confidence": "Low/Medium/High",
        "primary_reasoning": "main factors driving the forecast"
    }}
    """
    
    CONFLICTING_SIGNALS_RESOLVER = """
    You are analyzing conflicting signals for {cryptocurrency} price prediction.
    
    **Conflicting Information Detected:**
    
    Technical Analysis suggests: {technical_direction}
    Sentiment Analysis suggests: {sentiment_direction}
    
    **Detailed Data:**
    Technical Summary: {technical_summary}
    Sentiment Summary: {sentiment_summary}
    
    **Conflict Resolution Framework:**
    
    When technical and sentiment signals conflict, consider:
    
    1. **Signal Strength**: Which signal is stronger/more definitive?
    2. **Volume Confirmation**: Are technical patterns confirmed by volume?
    3. **Sentiment Source Quality**: News vs forum sentiment reliability
    4. **Market Regime**: Is this a technical or sentiment-driven market?
    5. **Time Horizon**: Short-term sentiment vs longer-term technicals
    6. **Historical Precedent**: How have similar conflicts resolved before?
    
    **Your Task:**
    Resolve this conflict and make a reasoned forecast for {cryptocurrency} 
    over {time_horizon}.
    
    Provide your analysis in JSON format:
    {{
        "conflict_summary": "description of the conflicting signals",
        "technical_strength": "assessment of technical signal strength",
        "sentiment_strength": "assessment of sentiment signal strength",
        "resolution_logic": "how you resolved the conflict",
        "dominant_factor": "which signal type dominated and why",
        "forecast": "UP/DOWN/NEUTRAL",
        "confidence": "Low/Medium/High",
        "rationale": "detailed rationale for the final decision",
        "risks": ["key risks to this forecast"]
    }}
    """
    
    CONFIDENCE_CALIBRATOR = """
    You are calibrating the confidence level for a {cryptocurrency} price forecast.
    
    **Forecast Details:**
    Direction: {forecast_direction}
    Technical Score: {technical_score}
    Sentiment Score: {sentiment_score}
    
    **Evidence Strength:**
    {evidence_summary}
    
    **Market Context:**
    Volatility: {volatility_level}
    Recent Price Action: {price_action}
    
    **Confidence Calibration Factors:**
    
    **High Confidence (0.8-1.0):**
    - Strong agreement between technical and sentiment
    - High-volume technical confirmations
    - Clear, consistent sentiment themes
    - Low market volatility
    
    **Medium Confidence (0.5-0.8):**
    - Moderate agreement between signals
    - Some conflicting indicators
    - Mixed sentiment sources
    - Normal market volatility
    
    **Low Confidence (0.0-0.5):**
    - Conflicting signals
    - Weak technical patterns
    - Manipulated or unclear sentiment
    - High market volatility
    
    Calibrate the confidence for this forecast:
    
    {{
        "original_confidence": "{original_confidence}",
        "calibrated_confidence": "Low/Medium/High",
        "confidence_score": 0.0,
        "calibration_reasoning": "why confidence was adjusted",
        "key_uncertainty_factors": ["factor1", "factor2"],
        "confidence_drivers": ["what supports confidence"],
        "final_assessment": "overall confidence assessment"
    }}
    """

def get_fusion_prompts():
    """Get fusion analysis prompt templates."""
    return {
        "master_fusion_prompt": FusionPrompts.MASTER_FUSION_PROMPT,
        "fs_reasoning_adapted_prompt": FusionPrompts.FS_REASONING_ADAPTED_PROMPT,
        "chain_of_thought_fusion": FusionPrompts.CHAIN_OF_THOUGHT_FUSION,
        "conflicting_signals_resolver": FusionPrompts.CONFLICTING_SIGNALS_RESOLVER,
        "confidence_calibrator": FusionPrompts.CONFIDENCE_CALIBRATOR,
    }

__all__ = ["FusionPrompts", "get_fusion_prompts"] 