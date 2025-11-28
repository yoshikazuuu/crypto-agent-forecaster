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
    ⚠️ CRITICAL - USE THIS EXACT CURRENT PRICE AS YOUR REFERENCE POINT ⚠️
    
    Current Price: ${current_price}
    24h Volume: ${volume_24h}
    Market Volatility: {volatility_level}
    
    **MANDATORY PRICE TARGET RULES:**
    - ALL price targets MUST be calculated relative to the current price of ${current_price}
    - For UP predictions: ALL targets must be ABOVE ${current_price} (typically 1-15% higher)
    - For DOWN predictions: ALL targets must be BELOW ${current_price} (typically 1-10% lower)
    - Stop loss must be on the OPPOSITE side of your direction from current price
    - DO NOT use historical Bitcoin prices from 2023-2024 or any other period
    - Your targets should reflect realistic {time_horizon} price movement from ${current_price}
    
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
    6. When signals are mixed or unclear, choose the direction supported by the most reliable signals
    
    === REQUIRED OUTPUT ===
    Predict the price direction for {cryptocurrency} using this EXACT format:
    
    **Direction**: [UP or DOWN]
    **Confidence**: [High, Medium, or Low]
    **Primary Target**: $[price level]
    **Secondary Target**: $[price level] (if applicable)
    **Stop Loss**: $[price level]
    **Take Profit 1**: $[price level]
    **Take Profit 2**: $[price level] (if applicable)
    **Risk-Reward Ratio**: [ratio like 1:2 or 1:3]
    **Position Size**: [Small, Medium, Large, or percentage like 5%]
    **Time Horizon**: {time_horizon}
    **Key Catalysts**: 
    - [catalyst 1]
    - [catalyst 2]
    - [catalyst 3]
    **Risk Factors**:
    - [risk factor 1]
    - [risk factor 2] 
    - [risk factor 3]

    **Analysis Summary:**
    [Provide detailed step-by-step reasoning of how you weighed the evidence, including:
    - Technical analysis weight and reasoning: {technical_weight}%
    - Sentiment analysis weight and reasoning: {sentiment_weight}%
    - Key factors that influenced your decision
    - Risk assessment and mitigation strategies]
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
    Make a binary decision: UP or DOWN for {cryptocurrency} over {time_horizon}.
    When signals are mixed, choose the direction supported by the most reliable data.
    
    Respond using this EXACT format:
    
    **Direction**: [UP or DOWN]
    **Confidence**: [High, Medium, or Low]
    **Primary Target**: $[price level]
    **Secondary Target**: $[price level] (if applicable)
    **Stop Loss**: $[price level]
    **Take Profit 1**: $[price level]
    **Take Profit 2**: $[price level] (if applicable)
    **Risk-Reward Ratio**: [ratio like 1:2 or 1:3]
    **Position Size**: [Small, Medium, Large, or percentage like 5%]
    **Time Horizon**: {time_horizon}
    **Key Catalysts**: 
    - [catalyst 1]
    - [catalyst 2]
    - [catalyst 3]
    **Risk Factors**:
    - [risk factor 1]
    - [risk factor 2] 
    - [risk factor 3]

    **Analysis Summary:**
    Factual Analysis: [objective technical trend assessment]
    News Sentiment: [curated news mood assessment]
    Forum Sentiment: [4chan/biz sentiment with manipulation notes]
    Weight Allocation: Technical {technical_weight}%, News {news_weight}%, Forum {forum_weight}%
    Synthesis: [how the three streams were combined]
    Final Reasoning: [final reasoning for the forecast]
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
    ⚠️ CRITICAL - CURRENT PRICE ANCHOR ⚠️
    Current Price: ${current_price}
    Volatility: {volatility_level}
    
    **IMPORTANT:** All your price targets MUST be based on ${current_price}.
    Do NOT use any historical prices. Calculate targets as percentages from ${current_price}.
    
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
    - Make final directional forecast: UP or DOWN (no neutral allowed)
    - When signals are mixed, lean toward the more reliable source
    
    **Final Output:**
    Based on this analysis, provide your forecast using this EXACT format:
    
    **Direction**: [UP or DOWN]
    **Confidence**: [High, Medium, or Low]
    **Primary Target**: $[price level]
    **Secondary Target**: $[price level] (if applicable)
    **Stop Loss**: $[price level]
    **Take Profit 1**: $[price level]
    **Take Profit 2**: $[price level] (if applicable)
    **Risk-Reward Ratio**: [ratio like 1:2 or 1:3]
    **Position Size**: [Small, Medium, Large, or percentage like 5%]
    **Time Horizon**: [forecast horizon]
    **Key Catalysts**: 
    - [catalyst 1]
    - [catalyst 2]
    - [catalyst 3]
    **Risk Factors**:
    - [risk factor 1]
    - [risk factor 2] 
    - [risk factor 3]

    **Analysis Summary:**
    Step 1 - Technical: [technical analysis interpretation]
    Step 2 - Sentiment: [sentiment analysis interpretation]
    Step 3 - Comparison: [comparison of signals]
    Step 4 - Context: [market context considerations]
    Step 5 - Integration: [final integration reasoning and main factors driving the forecast]
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
    over {time_horizon}. You must choose either UP or DOWN - no neutral allowed.
    When truly uncertain, lean toward the more reliable signal source.
    
    Provide your analysis using this EXACT format:
    
    **Direction**: [UP or DOWN]
    **Confidence**: [High, Medium, or Low]
    **Primary Target**: $[price level]
    **Secondary Target**: $[price level] (if applicable)
    **Stop Loss**: $[price level]
    **Take Profit 1**: $[price level]
    **Take Profit 2**: $[price level] (if applicable)
    **Risk-Reward Ratio**: [ratio like 1:2 or 1:3]
    **Position Size**: [Small, Medium, Large, or percentage like 5%]
    **Time Horizon**: [forecast horizon]
    **Key Catalysts**: 
    - [catalyst 1]
    - [catalyst 2]
    - [catalyst 3]
    **Risk Factors**:
    - [risk factor 1]
    - [risk factor 2] 
    - [risk factor 3]

    **Analysis Summary:**
    Conflict Summary: [description of the conflicting signals]
    Technical Strength: [assessment of technical signal strength]
    Sentiment Strength: [assessment of sentiment signal strength]
    Resolution Logic: [how you resolved the conflict]
    Dominant Factor: [which signal type dominated and why]
    Detailed Rationale: [detailed rationale for the final decision]
    """
    
    CONFIDENCE_CALIBRATOR = """
    You are calibrating confidence levels for cryptocurrency price predictions.
    
    **Prediction Details:**
    Cryptocurrency: {cryptocurrency}
    Forecast: {forecast_direction}
    Time Horizon: {time_horizon}
    
    **Signal Strength Assessment:**
    Technical Signals: {technical_strength}
    Sentiment Signals: {sentiment_strength}
    Signal Agreement: {signal_agreement}
    
    **Confidence Calibration Guidelines:**
    
    **HIGH Confidence (0.8-1.0):**
    - Technical and sentiment strongly agree
    - Multiple confirming indicators
    - Clear trend with volume support
    - Low market volatility
    - Historical pattern reliability high
    
    **MEDIUM Confidence (0.5-0.7):**
    - Moderate signal agreement
    - Some confirming indicators
    - Trend present but with some uncertainty
    - Normal market volatility
    - Mixed historical patterns
    
    **LOW Confidence (0.2-0.4):**
    - Conflicting signals between technical/sentiment
    - Weak or unclear indicators
    - High market volatility
    - Uncertain market conditions
    - Limited reliable data
    
    **Your Task:**
    Assess the confidence level for this {forecast_direction} prediction 
    and provide detailed calibration reasoning.
    
    Respond using this EXACT format:
    
    **Direction**: [{forecast_direction}]
    **Confidence**: [High, Medium, or Low]
    **Primary Target**: $[price level]
    **Secondary Target**: $[price level] (if applicable)
    **Stop Loss**: $[price level]
    **Take Profit 1**: $[price level]
    **Take Profit 2**: $[price level] (if applicable)
    **Risk-Reward Ratio**: [ratio like 1:2 or 1:3]
    **Position Size**: [Small, Medium, Large, or percentage like 5%]
    **Time Horizon**: {time_horizon}
    **Key Catalysts**: 
    - [catalyst 1]
    - [catalyst 2]
    - [catalyst 3]
    **Risk Factors**:
    - [risk factor 1]
    - [risk factor 2] 
    - [risk factor 3]

    **Analysis Summary:**
    Confidence Assessment: [detailed explanation of confidence calibration]
    Supporting Factors: [factors supporting this confidence level]
    Signal Agreement: [assessment of how well signals agree]
    Forecast Reliability: [assessment of prediction reliability]
    """
    
    MULTI_TIMEFRAME_FUSION = """
    You are analyzing {cryptocurrency} across multiple timeframes for trend consistency.
    
    **Multi-Timeframe Analysis:**
    Short-term (4h-1d): {short_term_signals}
    Medium-term (1d-1w): {medium_term_signals}  
    Long-term (1w-1m): {long_term_signals}
    
    **Timeframe Alignment Assessment:**
    Analyze how signals align across different timeframes:
    - Do all timeframes suggest the same direction?
    - Are there any conflicts between timeframes?
    - Which timeframe is most relevant for {time_horizon} prediction?
    
    **Fusion Strategy:**
    1. Identify the dominant trend across timeframes
    2. Weight timeframes based on prediction horizon
    3. Resolve any timeframe conflicts
    4. Make final UP/DOWN decision (no neutral allowed)
    
    Respond using this EXACT format:
    
    **Direction**: [UP or DOWN]
    **Confidence**: [High, Medium, or Low]
    **Primary Target**: $[price level]
    **Secondary Target**: $[price level] (if applicable)
    **Stop Loss**: $[price level]
    **Take Profit 1**: $[price level]
    **Take Profit 2**: $[price level] (if applicable)
    **Risk-Reward Ratio**: [ratio like 1:2 or 1:3]
    **Position Size**: [Small, Medium, Large, or percentage like 5%]
    **Time Horizon**: [forecast horizon]
    **Key Catalysts**: 
    - [catalyst 1]
    - [catalyst 2]
    - [catalyst 3]
    **Risk Factors**:
    - [risk factor 1]
    - [risk factor 2] 
    - [risk factor 3]

    **Analysis Summary:**
    Timeframe Alignment: [Strong/Moderate/Weak trend alignment]
    Dominant Timeframe: [short/medium/long term driving the forecast]
    Trend Consistency: [description of trend alignment across timeframes]
    Conflict Resolution: [how timeframe conflicts were resolved]
    Timeframe Weights: Short {short_weight}%, Medium {medium_weight}%, Long {long_weight}%
    Multi-Timeframe Reasoning: [comprehensive multi-timeframe analysis reasoning]
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