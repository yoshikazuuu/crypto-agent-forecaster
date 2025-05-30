"""
Chart analysis tool for providing multimodal analysis of generated technical analysis charts.
"""

import base64
import os
from typing import Optional, Union
from crewai.tools import tool
from crewai import Agent, Task, Crew
from .technical_analysis_tool import get_current_chart_path, get_current_chart_data
from ..llm_factory import LLMFactory
from ..config import Config


@tool("chart_analysis_tool")
def chart_analysis_tool(analysis_context: str = "") -> str:
    """
    Analyzes the most recently generated technical analysis chart using multimodal AI capabilities.
    
    Args:
        analysis_context: Additional context or specific aspects to focus on in the analysis.
                         Can include crypto_name by prefixing with "crypto_name:bitcoin," or similar.
    
    Returns:
        Detailed AI-powered chart analysis with visual insights and actionable recommendations
    """
    
    # Extract crypto_name from analysis_context if provided
    crypto_name = "Cryptocurrency"
    if analysis_context and "crypto_name:" in analysis_context:
        try:
            context_parts = analysis_context.split("crypto_name:")
            if len(context_parts) > 1:
                name_part = context_parts[1].split(",")[0].strip()
                if name_part:
                    crypto_name = name_part
                    # Remove the crypto_name part from analysis_context
                    analysis_context = context_parts[0] + ",".join(context_parts[1].split(",")[1:])
                    analysis_context = analysis_context.strip().strip(",").strip()
        except:
            pass  # If parsing fails, use default
    
    # Get the current chart path for multimodal analysis
    chart_path = get_current_chart_path()
    chart_data = get_current_chart_data()
    
    if not chart_path or not os.path.exists(chart_path):
        return f"❌ No chart data available for {crypto_name}. Please run technical analysis first to generate a chart."
    
    try:
        # Create LLM instance using default configuration
        llm = LLMFactory.create_llm(
            provider=Config.DEFAULT_LLM_PROVIDER,
            model=Config.DEFAULT_LLM_MODEL,
            temperature=0.1,
            max_tokens=3000
        )
        
        # Create a multimodal agent for chart analysis
        chart_analyst = Agent(
            role="Expert Technical Chart Analyst",
            goal=f"Analyze the technical analysis chart for {crypto_name} and provide detailed visual insights",
            backstory="""You are a world-class technical analyst with over 20 years of experience in 
            cryptocurrency markets. You specialize in visual chart pattern recognition, trend analysis, 
            support/resistance identification, and providing actionable trading insights. You have an 
            exceptional ability to interpret complex multi-panel technical charts including candlesticks, 
            volume, RSI, MACD, and moving averages.""",
            multimodal=True,  # Enable multimodal capabilities
            verbose=True,
            llm=llm  # Use explicitly configured LLM
        )
        
        # Define the analysis focus based on context
        analysis_focus = analysis_context if analysis_context else """
        Comprehensive technical analysis focusing on:
        1. Price action and trend direction
        2. Support and resistance levels
        3. Technical indicator signals
        4. Entry/exit opportunities
        5. Risk management levels
        """
        
        # Create a detailed analysis task
        analysis_task = Task(
            description=f"""
            Analyze the technical analysis chart for {crypto_name} located at: {chart_path}
            
            Please provide a comprehensive analysis covering:
            
            **Chart Structure Analysis:**
            - Overall trend direction (bullish/bearish/sideways)
            - Key support and resistance levels visible on the chart
            - Price position relative to moving averages
            - Volume patterns and confirmation signals
            
            **Technical Indicator Analysis:**
            - RSI readings and momentum signals
            - MACD crossovers and divergences
            - Moving average positioning and crossovers
            - Bollinger Band position and squeeze/expansion patterns
            
            **Pattern Recognition:**
            - Candlestick patterns and formations
            - Chart patterns (triangles, channels, flags, etc.)
            - Breakout or breakdown signals
            - Confluence zones where multiple indicators align
            
            **Trading Insights:**
            - Immediate price direction bias (next 24 hours)
            - Key levels to watch for entries and exits
            - Stop-loss placement recommendations
            - Risk-reward assessment
            - Potential price targets
            
            **Market Context:**
            - Current market phase (trending vs ranging)
            - Volatility assessment
            - Strength of current move
            - Potential reversal or continuation signals
            
            Analysis Focus: {analysis_focus}
            
            Provide specific, actionable insights based on what you observe in the chart.
            Use actual price levels and percentages where visible.
            """,
            expected_output="""A detailed professional chart analysis report with:
            - Clear trend assessment
            - Specific support/resistance levels
            - Technical indicator interpretations
            - Pattern recognition insights
            - Actionable trading recommendations
            - Risk management guidance""",
            agent=chart_analyst
        )
        
        # Create and run the crew for chart analysis
        analysis_crew = Crew(
            agents=[chart_analyst],
            tasks=[analysis_task],
            verbose=True
        )
        
        # Execute the multimodal analysis
        print(f"🔍 Starting AI-powered chart analysis for {crypto_name}...")
        result = analysis_crew.kickoff()
        
        # Format the response with additional metadata
        formatted_response = f"""
# 📊 AI-Powered Chart Analysis for {crypto_name}

## 🎯 Analysis Summary
{result.raw if hasattr(result, 'raw') else str(result)}

---

## 📈 Chart Analysis Metadata
- **Chart File:** {os.path.basename(chart_path)}
- **Analysis Method:** CrewAI Multimodal Agent with Visual Recognition
- **Chart Data Available:** {len(chart_data)} bytes (base64 encoded)
- **Analysis Context:** {analysis_context if analysis_context else "Comprehensive technical analysis"}

## 🔍 Visual Analysis Capabilities Applied
- ✅ Multi-panel chart interpretation (Price, RSI, Volume)
- ✅ Candlestick pattern recognition
- ✅ Technical indicator signal analysis
- ✅ Support/resistance level identification
- ✅ Trend and momentum assessment
- ✅ Visual pattern recognition
- ✅ Color-coded indicator analysis

## 💡 Next Steps
1. Consider the technical levels and signals identified
2. Monitor the key price levels mentioned for entry/exit opportunities
3. Apply proper risk management based on the recommendations
4. Re-analyze if market conditions change significantly

*Analysis powered by CrewAI multimodal agents with advanced visual recognition capabilities.*
"""
        
        print(f"✅ Completed AI chart analysis for {crypto_name}")
        return formatted_response
        
    except Exception as e:
        error_msg = f"""
❌ **Error in AI Chart Analysis for {crypto_name}**

An error occurred during the multimodal chart analysis: {str(e)}

**Fallback Analysis Available:**
- Chart data is available ({len(chart_data)} bytes) but AI analysis failed
- Chart file location: {chart_path}
- You can try running the analysis again or check the chart manually

**Basic Chart Information:**
- Cryptocurrency: {crypto_name}
- Chart Type: Multi-panel technical analysis (Candlesticks + Indicators)
- Contains: Price action, RSI, Volume, Moving Averages
- Analysis Context: {analysis_context if analysis_context else "General technical analysis"}

**Recommendations:**
1. Verify the chart file exists and is accessible
2. Check system requirements for multimodal analysis
3. Try running technical analysis again to regenerate the chart
4. Consider manual chart interpretation if AI analysis continues to fail
"""
        print(f"❌ Error in chart analysis: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return error_msg


def create_multimodal_chart_analyst(crypto_name: str) -> Agent:
    """
    Create a specialized multimodal agent for chart analysis.
    
    Args:
        crypto_name: Name of the cryptocurrency for personalized analysis
        
    Returns:
        Configured multimodal Agent for chart analysis
    """
    # Create LLM instance using default configuration
    llm = LLMFactory.create_llm(
        provider=Config.DEFAULT_LLM_PROVIDER,
        model=Config.DEFAULT_LLM_MODEL,
        temperature=0.1,
        max_tokens=3000
    )
    
    return Agent(
        role=f"{crypto_name} Technical Chart Specialist",
        goal=f"Provide expert visual analysis of {crypto_name} technical charts with actionable insights",
        backstory=f"""You are a specialized technical analyst focused on {crypto_name} with deep expertise in:
        
        📊 **Chart Analysis Skills:**
        - Advanced candlestick pattern recognition
        - Multi-timeframe trend analysis  
        - Support/resistance level identification
        - Volume profile analysis
        - Technical indicator interpretation
        
        🎯 **Specializations:**
        - {crypto_name} market behavior patterns
        - Cryptocurrency-specific volatility analysis
        - DeFi and institutional flow analysis
        - Risk management in crypto markets
        - Entry/exit timing optimization
        
        🔍 **Visual Recognition Capabilities:**
        - Chart pattern detection and classification
        - Color-coded indicator analysis
        - Multi-panel chart interpretation
        - Price action confluence identification
        - Market structure analysis
        
        You provide clear, actionable insights that help traders make informed decisions.""",
        multimodal=True,
        verbose=True,
        allow_delegation=False,
        llm=llm  # Use explicitly configured LLM
    )


def analyze_chart_with_context(crypto_name: str, 
                             specific_questions: list = None,
                             risk_tolerance: str = "moderate") -> str:
    """
    Enhanced chart analysis with specific questions and risk context.
    
    Args:
        crypto_name: Cryptocurrency to analyze
        specific_questions: List of specific questions to address
        risk_tolerance: Risk tolerance level (conservative, moderate, aggressive)
        
    Returns:
        Targeted chart analysis addressing specific concerns
    """
    
    chart_path = get_current_chart_path()
    if not chart_path or not os.path.exists(chart_path):
        return f"No chart available for {crypto_name}. Generate technical analysis first."
    
    # Create specialized agent
    analyst = create_multimodal_chart_analyst(crypto_name)
    
    # Build context-specific questions
    questions_text = ""
    if specific_questions:
        questions_text = "**Specific Questions to Address:**\n"
        for i, question in enumerate(specific_questions, 1):
            questions_text += f"{i}. {question}\n"
    
    # Risk-adjusted analysis prompt
    risk_context = {
        "conservative": "Focus on high-probability setups with strong confirmation signals. Emphasize capital preservation.",
        "moderate": "Balance opportunity recognition with risk management. Consider both aggressive and conservative entries.",
        "aggressive": "Identify high-reward opportunities even with higher risk. Focus on momentum and breakout trades."
    }
    
    risk_guidance = risk_context.get(risk_tolerance, risk_context["moderate"])
    
    # Create targeted analysis task
    task = Task(
        description=f"""
        Analyze the {crypto_name} chart at {chart_path} with specific focus on:
        
        {questions_text}
        
        **Risk Profile:** {risk_tolerance.title()}
        **Risk Guidance:** {risk_guidance}
        
        Provide chart analysis that specifically addresses the questions above while
        considering the risk tolerance level. Include visual observations from the chart
        and specific price levels or percentages where applicable.
        """,
        expected_output=f"Targeted chart analysis for {crypto_name} addressing specific questions with risk-appropriate recommendations",
        agent=analyst
    )
    
    # Execute analysis
    crew = Crew(agents=[analyst], tasks=[task], verbose=True)
    result = crew.kickoff()
    
    return f"""
# 🎯 Targeted Chart Analysis: {crypto_name}

## Risk Profile: {risk_tolerance.title()}

{result.raw if hasattr(result, 'raw') else str(result)}

---
*Analysis tailored for {risk_tolerance} risk tolerance with multimodal chart interpretation*
"""


class ChartAnalysisTool:
    """Wrapper class for the enhanced multimodal chart analysis tool."""
    
    def __init__(self):
        self.name = "chart_analysis_tool"
        self.description = """
        Advanced multimodal chart analysis tool that uses AI agents to interpret technical analysis charts.
        Provides visual pattern recognition, technical indicator analysis, support/resistance identification,
        and actionable trading insights by analyzing generated chart images using computer vision capabilities.
        """
    
    def _run(self, crypto_name: str, analysis_context: str = "") -> str:
        """Legacy interface for the tool."""
        return chart_analysis_tool.func(analysis_context)


def create_chart_analysis_tool():
    """Create and return a multimodal chart analysis tool instance."""
    return chart_analysis_tool 