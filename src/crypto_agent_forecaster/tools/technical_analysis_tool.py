"""
Technical analysis tool for cryptocurrency price data.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import ta

from ..config import Config


class TechnicalAnalysisInput(BaseModel):
    """Input for technical analysis tool."""
    ohlcv_data: str = Field(description="JSON string containing OHLCV data with columns: timestamp, open, high, low, close, volume")
    crypto_name: str = Field(description="Name of the cryptocurrency being analyzed")


class TechnicalAnalysisTool(BaseTool):
    """Tool for performing technical analysis on cryptocurrency price data."""
    
    name: str = "technical_analysis_tool"
    description: str = """
    Performs comprehensive technical analysis on cryptocurrency OHLCV data.
    Calculates various technical indicators and identifies candlestick patterns.
    Returns a textual summary suitable for LLM processing.
    """
    args_schema: type[BaseModel] = TechnicalAnalysisInput
    
    def __init__(self):
        super().__init__()
        self.ta_config = Config.TA_INDICATORS
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators from OHLCV data."""
        indicators = {}
        
        try:
            # Moving Averages
            for period in self.ta_config["sma_periods"]:
                indicators[f"sma_{period}"] = ta.trend.SMAIndicator(df['close'], window=period).sma_indicator().iloc[-1]
            
            for period in self.ta_config["ema_periods"]:
                indicators[f"ema_{period}"] = ta.trend.EMAIndicator(df['close'], window=period).ema_indicator().iloc[-1]
            
            # RSI
            rsi_period = self.ta_config["rsi_period"]
            indicators["rsi"] = ta.momentum.RSIIndicator(df['close'], window=rsi_period).rsi().iloc[-1]
            
            # MACD
            macd_indicator = ta.trend.MACD(
                df['close'],
                window_slow=self.ta_config["macd_slow"],
                window_fast=self.ta_config["macd_fast"],
                window_sign=self.ta_config["macd_signal"]
            )
            indicators["macd"] = macd_indicator.macd().iloc[-1]
            indicators["macd_signal"] = macd_indicator.macd_signal().iloc[-1]
            indicators["macd_histogram"] = macd_indicator.macd_diff().iloc[-1]
            
            # Bollinger Bands
            bb_indicator = ta.volatility.BollingerBands(
                df['close'],
                window=self.ta_config["bb_period"],
                window_dev=self.ta_config["bb_std"]
            )
            indicators["bb_upper"] = bb_indicator.bollinger_hband().iloc[-1]
            indicators["bb_middle"] = bb_indicator.bollinger_mavg().iloc[-1]
            indicators["bb_lower"] = bb_indicator.bollinger_lband().iloc[-1]
            
            # Volume indicators
            indicators["volume_sma"] = ta.volume.VolumeSMAIndicator(df['close'], df['volume'], window=20).volume_sma().iloc[-1]
            
            # Price position relative to bands
            current_price = df['close'].iloc[-1]
            indicators["bb_position"] = (current_price - indicators["bb_lower"]) / (indicators["bb_upper"] - indicators["bb_lower"])
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            indicators["error"] = str(e)
        
        return indicators
    
    def _identify_candlestick_patterns(self, df: pd.DataFrame) -> List[str]:
        """Identify candlestick patterns in the data."""
        patterns = []
        
        if len(df) < 3:
            return patterns
        
        try:
            # Get last few candles
            last_3 = df.tail(3)
            last_2 = df.tail(2)
            current = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else current
            
            # Bullish Engulfing
            if (len(last_2) >= 2 and
                previous['close'] < previous['open'] and  # Previous was bearish
                current['close'] > current['open'] and   # Current is bullish
                current['open'] < previous['close'] and  # Current opens below previous close
                current['close'] > previous['open']):    # Current closes above previous open
                patterns.append("Bullish Engulfing")
            
            # Bearish Engulfing
            if (len(last_2) >= 2 and
                previous['close'] > previous['open'] and  # Previous was bullish
                current['close'] < current['open'] and   # Current is bearish
                current['open'] > previous['close'] and  # Current opens above previous close
                current['close'] < previous['open']):    # Current closes below previous open
                patterns.append("Bearish Engulfing")
            
            # Doji (open and close are very close)
            body_size = abs(current['close'] - current['open'])
            candle_range = current['high'] - current['low']
            if candle_range > 0 and body_size / candle_range < 0.1:
                patterns.append("Doji")
            
            # Hammer (bullish reversal)
            if (current['close'] > current['open'] and  # Bullish candle
                body_size > 0 and
                (current['close'] - current['low']) > 2 * body_size and  # Long lower shadow
                (current['high'] - current['close']) < body_size):  # Short upper shadow
                patterns.append("Hammer")
            
            # Shooting Star (bearish reversal)
            if (current['close'] < current['open'] and  # Bearish candle
                body_size > 0 and
                (current['high'] - current['open']) > 2 * body_size and  # Long upper shadow
                (current['close'] - current['low']) < body_size):  # Short lower shadow
                patterns.append("Shooting Star")
            
            # Morning Star (3-candle bullish reversal)
            if (len(last_3) >= 3 and
                last_3.iloc[0]['close'] < last_3.iloc[0]['open'] and  # First: bearish
                abs(last_3.iloc[1]['close'] - last_3.iloc[1]['open']) < body_size * 0.5 and  # Second: small body
                last_3.iloc[2]['close'] > last_3.iloc[2]['open'] and  # Third: bullish
                last_3.iloc[2]['close'] > (last_3.iloc[0]['open'] + last_3.iloc[0]['close']) / 2):  # Third closes above midpoint of first
                patterns.append("Morning Star")
            
            # Evening Star (3-candle bearish reversal)
            if (len(last_3) >= 3 and
                last_3.iloc[0]['close'] > last_3.iloc[0]['open'] and  # First: bullish
                abs(last_3.iloc[1]['close'] - last_3.iloc[1]['open']) < body_size * 0.5 and  # Second: small body
                last_3.iloc[2]['close'] < last_3.iloc[2]['open'] and  # Third: bearish
                last_3.iloc[2]['close'] < (last_3.iloc[0]['open'] + last_3.iloc[0]['close']) / 2):  # Third closes below midpoint of first
                patterns.append("Evening Star")
                
        except Exception as e:
            print(f"Error identifying patterns: {e}")
            patterns.append(f"Pattern analysis error: {str(e)}")
        
        return patterns if patterns else ["No significant patterns detected"]
    
    def _interpret_indicators(self, indicators: Dict[str, Any], current_price: float) -> Dict[str, str]:
        """Interpret indicator values and provide qualitative assessments."""
        interpretations = {}
        
        try:
            # RSI interpretation
            rsi = indicators.get("rsi", 50)
            if rsi > 70:
                interpretations["rsi"] = f"Overbought (RSI: {rsi:.1f})"
            elif rsi < 30:
                interpretations["rsi"] = f"Oversold (RSI: {rsi:.1f})"
            else:
                interpretations["rsi"] = f"Neutral (RSI: {rsi:.1f})"
            
            # MACD interpretation
            macd = indicators.get("macd", 0)
            macd_signal = indicators.get("macd_signal", 0)
            macd_hist = indicators.get("macd_histogram", 0)
            
            if macd > macd_signal:
                macd_trend = "Bullish"
            else:
                macd_trend = "Bearish"
            
            interpretations["macd"] = f"{macd_trend} (MACD: {macd:.4f}, Signal: {macd_signal:.4f})"
            
            # Moving Average interpretation
            sma_20 = indicators.get("sma_20")
            sma_50 = indicators.get("sma_50")
            
            if sma_20 and sma_50:
                if sma_20 > sma_50:
                    ma_trend = "Bullish (20 SMA > 50 SMA)"
                else:
                    ma_trend = "Bearish (20 SMA < 50 SMA)"
                
                price_vs_ma20 = "above" if current_price > sma_20 else "below"
                interpretations["moving_averages"] = f"{ma_trend}, Price {price_vs_ma20} 20 SMA"
            
            # Bollinger Bands interpretation
            bb_position = indicators.get("bb_position", 0.5)
            if bb_position > 0.8:
                interpretations["bollinger_bands"] = "Near upper band (potentially overbought)"
            elif bb_position < 0.2:
                interpretations["bollinger_bands"] = "Near lower band (potentially oversold)"
            else:
                interpretations["bollinger_bands"] = f"Within bands (position: {bb_position:.2f})"
                
        except Exception as e:
            interpretations["error"] = f"Interpretation error: {str(e)}"
        
        return interpretations
    
    def _generate_summary(self, crypto_name: str, indicators: Dict[str, Any], 
                         patterns: List[str], interpretations: Dict[str, str],
                         current_price: float) -> str:
        """Generate a comprehensive textual summary of technical analysis."""
        
        summary_parts = [
            f"**Technical Analysis for {crypto_name.title()}**",
            f"Current Price: ${current_price:.4f}",
            "",
            "**Candlestick Patterns:**"
        ]
        
        for pattern in patterns:
            summary_parts.append(f"- {pattern}")
        
        summary_parts.extend([
            "",
            "**Technical Indicators:**"
        ])
        
        for indicator, interpretation in interpretations.items():
            summary_parts.append(f"- {indicator.replace('_', ' ').title()}: {interpretation}")
        
        # Overall assessment
        bullish_signals = 0
        bearish_signals = 0
        
        # Count signals
        if indicators.get("rsi", 50) < 30:
            bullish_signals += 1
        elif indicators.get("rsi", 50) > 70:
            bearish_signals += 1
        
        if indicators.get("macd", 0) > indicators.get("macd_signal", 0):
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        sma_20 = indicators.get("sma_20", current_price)
        if current_price > sma_20:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        bullish_patterns = ["Bullish Engulfing", "Hammer", "Morning Star"]
        bearish_patterns = ["Bearish Engulfing", "Shooting Star", "Evening Star"]
        
        for pattern in patterns:
            if any(bp in pattern for bp in bullish_patterns):
                bullish_signals += 1
            elif any(bp in pattern for bp in bearish_patterns):
                bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            overall = "Bullish"
        elif bearish_signals > bullish_signals:
            overall = "Bearish"
        else:
            overall = "Neutral"
        
        summary_parts.extend([
            "",
            f"**Overall Technical Outlook: {overall}**",
            f"(Bullish signals: {bullish_signals}, Bearish signals: {bearish_signals})"
        ])
        
        return "\n".join(summary_parts)
    
    def _run(self, ohlcv_data: str, crypto_name: str) -> str:
        """
        Perform technical analysis on OHLCV data.
        
        Args:
            ohlcv_data: JSON string containing OHLCV data
            crypto_name: Name of the cryptocurrency
            
        Returns:
            Textual summary of technical analysis
        """
        try:
            # Parse the OHLCV data
            data = json.loads(ohlcv_data)
            df = pd.DataFrame(data)
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    return f"Error: Missing required column '{col}' in OHLCV data"
            
            # Convert to numeric
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            if len(df) < 2:
                return f"Error: Insufficient data for technical analysis (need at least 2 data points, got {len(df)})"
            
            current_price = df['close'].iloc[-1]
            
            # Calculate indicators
            indicators = self._calculate_indicators(df)
            
            # Identify patterns
            patterns = self._identify_candlestick_patterns(df)
            
            # Interpret indicators
            interpretations = self._interpret_indicators(indicators, current_price)
            
            # Generate summary
            summary = self._generate_summary(crypto_name, indicators, patterns, interpretations, current_price)
            
            return summary
            
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON in OHLCV data: {str(e)}"
        except Exception as e:
            return f"Error performing technical analysis: {str(e)}"


def create_technical_analysis_tool() -> TechnicalAnalysisTool:
    """Create and return a technical analysis tool instance."""
    return TechnicalAnalysisTool() 