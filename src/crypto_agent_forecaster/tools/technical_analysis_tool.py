"""
Technical analysis tool for cryptocurrency price data with chart generation.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Any, Optional
from crewai.tools import tool
import ta
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ..config import Config

# Global variable to store chart data for the crew manager
_current_chart_data = None


def get_current_chart_data() -> Optional[str]:
    """Get the current chart data (base64 encoded)."""
    global _current_chart_data
    return _current_chart_data


def clear_chart_data():
    """Clear the current chart data."""
    global _current_chart_data
    _current_chart_data = None


@tool("technical_analysis_tool")
def technical_analysis_tool(ohlcv_data: str, crypto_name: str, generate_chart: bool = True) -> str:
    """
    Performs comprehensive technical analysis on cryptocurrency OHLCV data and generates visual charts.
    
    Args:
        ohlcv_data: JSON string containing OHLCV data with columns: timestamp, open, high, low, close, volume
        crypto_name: Name of the cryptocurrency being analyzed
        generate_chart: Whether to generate a visual chart (default: True)
    
    Returns:
        Textual summary with chart image encoded as base64 if generate_chart=True
    """
    
    def _calculate_rma(series: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Moving Average (RMA) also known as Modified Moving Average."""
        alpha = 1.0 / period
        rma = series.ewm(alpha=alpha, adjust=False).mean()
        return rma
    
    def _calculate_indicators(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators from OHLCV data."""
        indicators = {}
        ta_config = Config.TA_INDICATORS
        
        try:
            # Moving Averages
            for period in ta_config["sma_periods"]:
                indicators[f"sma_{period}"] = ta.trend.SMAIndicator(df['close'], window=period).sma_indicator().iloc[-1]
            
            for period in ta_config["ema_periods"]:
                indicators[f"ema_{period}"] = ta.trend.EMAIndicator(df['close'], window=period).ema_indicator().iloc[-1]
            
            # RMA (Relative Moving Average)
            for period in [14, 21, 50]:
                rma_values = _calculate_rma(df['close'], period)
                indicators[f"rma_{period}"] = rma_values.iloc[-1]
            
            # RSI
            rsi_period = ta_config["rsi_period"]
            indicators["rsi"] = ta.momentum.RSIIndicator(df['close'], window=rsi_period).rsi().iloc[-1]
            
            # Stochastic RSI
            stoch_rsi = ta.momentum.StochRSIIndicator(df['close'], window=14, smooth1=3, smooth2=3)
            indicators["stoch_rsi_k"] = stoch_rsi.stochrsi_k().iloc[-1]
            indicators["stoch_rsi_d"] = stoch_rsi.stochrsi_d().iloc[-1]
            
            # Williams %R
            williams_r = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14)
            indicators["williams_r"] = williams_r.williams_r().iloc[-1]
            
            # MACD
            macd_indicator = ta.trend.MACD(
                df['close'],
                window_slow=ta_config["macd_slow"],
                window_fast=ta_config["macd_fast"],
                window_sign=ta_config["macd_signal"]
            )
            indicators["macd"] = macd_indicator.macd().iloc[-1]
            indicators["macd_signal"] = macd_indicator.macd_signal().iloc[-1]
            indicators["macd_histogram"] = macd_indicator.macd_diff().iloc[-1]
            
            # Bollinger Bands
            bb_indicator = ta.volatility.BollingerBands(
                df['close'],
                window=ta_config["bb_period"],
                window_dev=ta_config["bb_std"]
            )
            indicators["bb_upper"] = bb_indicator.bollinger_hband().iloc[-1]
            indicators["bb_middle"] = bb_indicator.bollinger_mavg().iloc[-1]
            indicators["bb_lower"] = bb_indicator.bollinger_lband().iloc[-1]
            
            # Additional volatility indicators
            # ATR (Average True Range)
            atr_indicator = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
            indicators["atr"] = atr_indicator.average_true_range().iloc[-1]
            
            # Keltner Channels
            keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'], window=20)
            indicators["keltner_upper"] = keltner.keltner_channel_hband().iloc[-1]
            indicators["keltner_middle"] = keltner.keltner_channel_mband().iloc[-1]
            indicators["keltner_lower"] = keltner.keltner_channel_lband().iloc[-1]
            
            # Volume indicators
            indicators["volume_sma"] = ta.trend.SMAIndicator(df['volume'], window=20).sma_indicator().iloc[-1]
            
            # On Balance Volume (OBV)
            obv_indicator = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
            indicators["obv"] = obv_indicator.on_balance_volume().iloc[-1]
            
            # Commodity Channel Index (CCI)
            cci_indicator = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20)
            indicators["cci"] = cci_indicator.cci().iloc[-1]
            
            # Price position relative to bands
            current_price = df['close'].iloc[-1]
            indicators["bb_position"] = (current_price - indicators["bb_lower"]) / (indicators["bb_upper"] - indicators["bb_lower"])
            indicators["keltner_position"] = (current_price - indicators["keltner_lower"]) / (indicators["keltner_upper"] - indicators["keltner_lower"])
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            indicators["error"] = str(e)
        
        return indicators
    
    def _create_technical_chart(df: pd.DataFrame, indicators: Dict[str, Any], crypto_name: str) -> str:
        """Create a comprehensive technical analysis chart and return as base64 encoded image."""
        try:
            # Set up the plot style
            plt.style.use('dark_background')
            fig, axes = plt.subplots(4, 1, figsize=(16, 20), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
            fig.suptitle(f'{crypto_name.title()} - Technical Analysis', fontsize=20, fontweight='bold', color='white')
            
            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                df['datetime'] = pd.to_datetime(df.index)
            
            # Main price chart with moving averages and Bollinger Bands
            ax1 = axes[0]
            ax1.plot(df['datetime'], df['close'], color='#00D4AA', linewidth=2, label='Close Price', alpha=0.9)
            
            # Moving averages
            if len(df) >= 20:
                sma_20 = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
                ax1.plot(df['datetime'], sma_20, color='#FFB800', linewidth=1.5, label='SMA 20', alpha=0.7)
            
            if len(df) >= 50:
                sma_50 = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
                ax1.plot(df['datetime'], sma_50, color='#FF6B6B', linewidth=1.5, label='SMA 50', alpha=0.7)
            
            # RMA
            if len(df) >= 21:
                rma_21 = _calculate_rma(df['close'], 21)
                ax1.plot(df['datetime'], rma_21, color='#4ECDC4', linewidth=1.5, label='RMA 21', alpha=0.7, linestyle='--')
            
            # Bollinger Bands
            if len(df) >= 20:
                bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
                bb_upper = bb_indicator.bollinger_hband()
                bb_lower = bb_indicator.bollinger_lband()
                bb_middle = bb_indicator.bollinger_mavg()
                
                ax1.fill_between(df['datetime'], bb_upper, bb_lower, alpha=0.1, color='#8B5CF6', label='Bollinger Bands')
                ax1.plot(df['datetime'], bb_upper, color='#8B5CF6', linewidth=1, alpha=0.6)
                ax1.plot(df['datetime'], bb_lower, color='#8B5CF6', linewidth=1, alpha=0.6)
                ax1.plot(df['datetime'], bb_middle, color='#8B5CF6', linewidth=1, alpha=0.8, linestyle=':')
            
            ax1.set_ylabel('Price ($)', fontsize=12, color='white')
            ax1.legend(loc='upper left', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(colors='white')
            
            # RSI with Stochastic RSI
            ax2 = axes[1]
            if len(df) >= 14:
                rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
                ax2.plot(df['datetime'], rsi, color='#F59E0B', linewidth=2, label='RSI')
                
                # Stochastic RSI
                stoch_rsi = ta.momentum.StochRSIIndicator(df['close'], window=14, smooth1=3, smooth2=3)
                stoch_k = stoch_rsi.stochrsi_k() * 100  # Convert to 0-100 scale
                stoch_d = stoch_rsi.stochrsi_d() * 100
                ax2.plot(df['datetime'], stoch_k, color='#06B6D4', linewidth=1.5, label='Stoch RSI %K', alpha=0.8)
                ax2.plot(df['datetime'], stoch_d, color='#8B5CF6', linewidth=1.5, label='Stoch RSI %D', alpha=0.8)
                
                # Williams %R
                williams_r = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()
                ax2_twin = ax2.twinx()
                ax2_twin.plot(df['datetime'], williams_r, color='#EF4444', linewidth=1, label='Williams %R', alpha=0.6)
                ax2_twin.set_ylabel('Williams %R', fontsize=10, color='#EF4444')
                ax2_twin.tick_params(colors='#EF4444')
                ax2_twin.set_ylim(-100, 0)
            
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax2.axhline(y=50, color='white', linestyle='-', alpha=0.3)
            ax2.set_ylabel('RSI / Stoch RSI', fontsize=12, color='white')
            ax2.set_ylim(0, 100)
            ax2.legend(loc='upper left', fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(colors='white')
            
            # MACD
            ax3 = axes[2]
            if len(df) >= 26:
                macd_indicator = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
                macd_line = macd_indicator.macd()
                macd_signal = macd_indicator.macd_signal()
                macd_histogram = macd_indicator.macd_diff()
                
                ax3.plot(df['datetime'], macd_line, color='#00D4AA', linewidth=2, label='MACD')
                ax3.plot(df['datetime'], macd_signal, color='#FF6B6B', linewidth=2, label='Signal')
                
                # MACD Histogram
                colors = ['#10B981' if x >= 0 else '#EF4444' for x in macd_histogram]
                ax3.bar(df['datetime'], macd_histogram, color=colors, alpha=0.6, width=0.8)
            
            ax3.axhline(y=0, color='white', linestyle='-', alpha=0.3)
            ax3.set_ylabel('MACD', fontsize=12, color='white')
            ax3.legend(loc='upper left', fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(colors='white')
            
            # Volume with OBV
            ax4 = axes[3]
            volume_colors = ['#10B981' if df['close'].iloc[i] >= df['open'].iloc[i] else '#EF4444' 
                           for i in range(len(df))]
            ax4.bar(df['datetime'], df['volume'], color=volume_colors, alpha=0.7)
            
            # OBV on secondary y-axis
            if len(df) >= 2:
                obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
                ax4_twin = ax4.twinx()
                ax4_twin.plot(df['datetime'], obv, color='#F59E0B', linewidth=2, alpha=0.8, label='OBV')
                ax4_twin.set_ylabel('OBV', fontsize=10, color='#F59E0B')
                ax4_twin.tick_params(colors='#F59E0B')
                ax4_twin.legend(loc='upper right', fontsize=9)
            
            ax4.set_ylabel('Volume', fontsize=12, color='white')
            ax4.set_xlabel('Date', fontsize=12, color='white')
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(colors='white')
            
            # Format x-axis
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df)//10)))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a1a', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            print(f"Error creating chart: {e}")
            return ""
    
    def _identify_candlestick_patterns(df: pd.DataFrame) -> List[str]:
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
    
    def _interpret_indicators(indicators: Dict[str, Any], current_price: float) -> Dict[str, str]:
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
            
            # Stochastic RSI interpretation
            stoch_k = indicators.get("stoch_rsi_k", 0.5) * 100
            stoch_d = indicators.get("stoch_rsi_d", 0.5) * 100
            if stoch_k > 80 and stoch_d > 80:
                interpretations["stoch_rsi"] = f"Overbought (K: {stoch_k:.1f}, D: {stoch_d:.1f})"
            elif stoch_k < 20 and stoch_d < 20:
                interpretations["stoch_rsi"] = f"Oversold (K: {stoch_k:.1f}, D: {stoch_d:.1f})"
            else:
                interpretations["stoch_rsi"] = f"Neutral (K: {stoch_k:.1f}, D: {stoch_d:.1f})"
            
            # Williams %R interpretation
            williams_r = indicators.get("williams_r", -50)
            if williams_r > -20:
                interpretations["williams_r"] = f"Overbought (%R: {williams_r:.1f})"
            elif williams_r < -80:
                interpretations["williams_r"] = f"Oversold (%R: {williams_r:.1f})"
            else:
                interpretations["williams_r"] = f"Neutral (%R: {williams_r:.1f})"
            
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
            rma_21 = indicators.get("rma_21")
            
            ma_trends = []
            if sma_20 and sma_50:
                if sma_20 > sma_50:
                    ma_trends.append("SMA(20>50): Bullish")
                else:
                    ma_trends.append("SMA(20<50): Bearish")
            
            if rma_21:
                price_vs_rma = "above" if current_price > rma_21 else "below"
                ma_trends.append(f"Price {price_vs_rma} RMA 21")
            
            interpretations["moving_averages"] = ", ".join(ma_trends) if ma_trends else "Insufficient data"
            
            # Bollinger Bands interpretation
            bb_position = indicators.get("bb_position", 0.5)
            if bb_position > 0.8:
                interpretations["bollinger_bands"] = "Near upper band (potentially overbought)"
            elif bb_position < 0.2:
                interpretations["bollinger_bands"] = "Near lower band (potentially oversold)"
            else:
                interpretations["bollinger_bands"] = f"Within bands (position: {bb_position:.2f})"
            
            # CCI interpretation
            cci = indicators.get("cci", 0)
            if cci > 100:
                interpretations["cci"] = f"Overbought (CCI: {cci:.1f})"
            elif cci < -100:
                interpretations["cci"] = f"Oversold (CCI: {cci:.1f})"
            else:
                interpretations["cci"] = f"Neutral (CCI: {cci:.1f})"
                
        except Exception as e:
            interpretations["error"] = f"Interpretation error: {str(e)}"
        
        return interpretations
    
    def _generate_summary(crypto_name: str, indicators: Dict[str, Any], 
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
    
    # Main execution
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
        indicators = _calculate_indicators(df)
        
        # Identify patterns
        patterns = _identify_candlestick_patterns(df)
        
        # Interpret indicators
        interpretations = _interpret_indicators(indicators, current_price)
        
        # Generate summary
        summary = _generate_summary(crypto_name, indicators, patterns, interpretations, current_price)
        
        # Create chart if requested
        if generate_chart:
            chart_image = _create_technical_chart(df, indicators, crypto_name)
            if chart_image:
                # Store chart data globally for crew manager to access
                global _current_chart_data
                _current_chart_data = chart_image
                
                summary += f"\n\n**Technical Analysis Chart:** Generated and available for saving."
        
        return summary
        
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in OHLCV data: {str(e)}"
    except Exception as e:
        return f"Error performing technical analysis: {str(e)}"


# Legacy wrapper for backward compatibility
class TechnicalAnalysisTool:
    """Legacy wrapper for the technical_analysis_tool function."""
    
    def __init__(self):
        self.name = "technical_analysis_tool"
        self.description = """
        Performs comprehensive technical analysis on cryptocurrency OHLCV data.
        Calculates various technical indicators including RMA, Stochastic RSI, Williams %R, CCI, ATR, OBV.
        Identifies candlestick patterns and generates visual charts.
        Returns a textual summary with optional chart image encoded as base64.
        """
        self.ta_config = Config.TA_INDICATORS
    
    def _run(self, ohlcv_data: str, crypto_name: str, generate_chart: bool = True) -> str:
        """Legacy interface for the tool."""
        return technical_analysis_tool.func(ohlcv_data, crypto_name, generate_chart)


def create_technical_analysis_tool():
    """Create and return a technical analysis tool instance."""
    return technical_analysis_tool 