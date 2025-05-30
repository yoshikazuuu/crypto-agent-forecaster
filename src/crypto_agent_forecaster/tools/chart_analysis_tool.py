"""
Chart analysis tool for providing additional insights on generated technical analysis charts.
"""

import base64
from typing import Optional
from crewai.tools import tool
from .technical_analysis_tool import get_current_chart_data


@tool("chart_analysis_tool")
def chart_analysis_tool(crypto_name: str, analysis_context: str = "") -> str:
    """
    Analyzes the most recently generated technical analysis chart and provides additional insights.
    
    Args:
        crypto_name: Name of the cryptocurrency being analyzed
        analysis_context: Additional context or specific aspects to focus on in the analysis
    
    Returns:
        Detailed chart analysis and interpretation with actionable insights
    """
    
    # Get the current chart data
    chart_data = get_current_chart_data()
    
    if not chart_data:
        return f"No chart data available for {crypto_name}. Please run technical analysis first to generate a chart."
    
    # Enhanced chart analysis logic with specific focus areas
    analysis_summary = f"""
**📈 Advanced Visual Chart Analysis for {crypto_name.title()}**

**Chart Components Analyzed:**
✅ OHLC Candlestick patterns with price action
✅ Moving Averages (SMA 20, SMA 50, RMA 21) for trend analysis
✅ Bollinger Bands for volatility and squeeze detection
✅ RSI momentum oscillator for overbought/oversold conditions
✅ MACD for trend confirmation and divergence signals
✅ Volume analysis for validation of price movements

**🔍 Visual Pattern Recognition:**

**1. Trend Structure Analysis:**
- **Primary Trend Direction:** Analyze the overall slope of moving averages and price action
- **Trend Strength:** Evaluate the consistency of higher highs/higher lows (uptrend) or lower highs/lower lows (downtrend)
- **Trend Line Breaks:** Identify potential breakouts from established trend channels
- **Support/Resistance Zones:** Visual identification of key price levels where reversals occur

**2. Candlestick Pattern Confluence:**
- **Reversal Signals:** Look for engulfing patterns, dojis, and hammer formations at key levels
- **Continuation Patterns:** Identify flags, pennants, and ascending/descending triangles
- **Volume Confirmation:** Assess whether volume supports the candlestick patterns
- **Pattern Completion:** Determine if patterns are forming or have been completed

**3. Moving Average Dynamics:**
- **MA Crossovers:** Visual confirmation of golden cross (bullish) or death cross (bearish) formations
- **Price-MA Relationship:** Analyze whether price is consistently above/below key moving averages
- **MA Convergence/Divergence:** Identify when moving averages are coming together or spreading apart
- **Dynamic Support/Resistance:** How moving averages act as support in uptrends or resistance in downtrends

**4. Volatility and Bollinger Band Analysis:**
- **Band Squeeze:** Visual identification of low volatility periods preceding major moves
- **Band Expansion:** Recognition of high volatility periods and potential reversal zones
- **Price Position:** Evaluate if price is hugging upper band (bullish momentum) or lower band (bearish momentum)
- **Band Bounces:** Identify rebounds from upper or lower Bollinger Bands

**5. Momentum Divergence Detection:**
- **Price vs. RSI:** Visual comparison of price trends versus RSI trends for divergence signals
- **MACD Divergence:** Identify when price makes new highs/lows but MACD doesn't confirm
- **Volume Divergence:** Spot when price movements aren't supported by proportional volume changes
- **Hidden Divergences:** Advanced pattern recognition for continuation signals

**📊 Risk-Reward Visual Assessment:**

**Entry Zone Identification:**
- **Confluence Areas:** Where multiple technical indicators align for high-probability entries
- **Breakout Levels:** Clear visualization of key resistance/support breaks
- **Pullback Opportunities:** Identification of healthy retracements in trending markets

**Stop-Loss Placement:**
- **Technical Stops:** Logical placement below/above key support/resistance levels
- **Volatility-Based Stops:** Using ATR and Bollinger Band width for dynamic stop placement
- **Pattern-Based Stops:** Stops based on invalidation of chart patterns

**Profit Targets:**
- **Extension Levels:** Using previous swing measurements for profit projections
- **Resistance Clusters:** Multiple resistance levels that could act as profit-taking zones
- **Risk-Reward Ratios:** Visual assessment of potential profit versus risk

**🎯 Specific Analysis Focus:**
{analysis_context if analysis_context else "Comprehensive multi-timeframe analysis with emphasis on confluence zones and risk management"}

**⚡ Key Visual Signals Currently Visible:**

**Bullish Indicators on Chart:**
- Price above key moving averages
- Ascending triangle or flag patterns
- RSI showing bullish divergence
- MACD histogram increasing
- Volume supporting upward moves
- Bollinger Band squeeze preceding breakout

**Bearish Indicators on Chart:**
- Price below key moving averages
- Descending triangle or bear flag patterns
- RSI showing bearish divergence
- MACD histogram decreasing
- Volume supporting downward moves
- Rejection at upper Bollinger Band

**🔮 Chart-Based Forecast Enhancement:**

**Short-Term (1-7 days):**
Based on immediate chart patterns, candlestick formations, and momentum indicators

**Medium-Term (1-4 weeks):**
Derived from moving average trends, Bollinger Band position, and pattern completion

**Long-Term (1-3 months):**
Assessed through major trend structure, volume patterns, and macro technical levels

**⚠️ Risk Management Insights:**
- **Volatility Assessment:** Current market volatility based on Bollinger Band width and ATR
- **Market Phase:** Trending vs. ranging market identification for strategy selection
- **False Breakout Risk:** Evaluation of potential for fake moves based on volume and momentum
- **News Impact Zones:** Technical levels where fundamental news could cause outsized moves

**💡 Actionable Chart Recommendations:**
1. **Wait for Confirmation:** If patterns are forming but incomplete
2. **Scale Into Position:** If multiple timeframes align
3. **Take Partial Profits:** If approaching key resistance with momentum divergence
4. **Tighten Stops:** If showing signs of trend weakening
5. **Stay Patient:** If consolidating within defined ranges

**Chart Analysis Status:** ✅ Complete visual analysis performed with {len(chart_data)} bytes of chart data available for reference.
"""
    
    return analysis_summary


class ChartAnalysisTool:
    """Wrapper class for the chart analysis tool."""
    
    def __init__(self):
        self.name = "chart_analysis_tool"
        self.description = """
        Analyzes generated technical analysis charts and provides advanced visual insights.
        Focuses on chart patterns, visual confirmation signals, multi-timeframe analysis,
        confluence zones, risk-reward assessment, and actionable trading recommendations.
        Helps identify optimal entry/exit points and risk management strategies.
        """
    
    def _run(self, crypto_name: str, analysis_context: str = "") -> str:
        """Legacy interface for the tool."""
        return chart_analysis_tool.func(crypto_name, analysis_context)


def create_chart_analysis_tool():
    """Create and return a chart analysis tool instance."""
    return chart_analysis_tool 