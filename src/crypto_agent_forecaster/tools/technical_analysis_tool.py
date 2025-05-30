"""
Technical analysis tool for cryptocurrency price data with chart generation.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
from crewai.tools import tool
import ta
import io
import base64
from datetime import datetime
import warnings
import tempfile
import os
warnings.filterwarnings('ignore')

from ..config import Config
from .coingecko_tool import coingecko_tool

# Global variable to store chart data for multimodal access
_current_chart_data = None
_current_chart_path = None


def get_current_chart_data() -> Optional[str]:
    """Get the current chart data (base64 encoded)."""
    global _current_chart_data
    return _current_chart_data


def get_current_chart_path() -> Optional[str]:
    """Get the current chart file path for multimodal access."""
    global _current_chart_path
    return _current_chart_path


def clear_chart_data():
    """Clear the current chart data."""
    global _current_chart_data, _current_chart_path
    _current_chart_data = None
    if _current_chart_path and os.path.exists(_current_chart_path):
        try:
            os.remove(_current_chart_path)
        except:
            pass  # Ignore cleanup errors
    _current_chart_path = None


@tool("technical_analysis_tool")
def technical_analysis_tool(crypto_name: str, days: int = 30) -> str:
    """
    Performs comprehensive technical analysis on cryptocurrency data by fetching fresh OHLCV data 
    and generating visual charts.
    
    Args:
        crypto_name: Name or symbol of the cryptocurrency to analyze (e.g., 'bitcoin', 'ethereum', 'BTC')
        days: Number of days of historical data to fetch for analysis (default: 30)
    
    Returns:
        Textual summary with chart generation status and analysis insights
    """
    
    # Clear any existing chart data to prevent caching issues
    clear_chart_data()
    
    try:
        # Fetch OHLCV data using the coingecko tool
        print(f"🔍 Fetching {days} days of OHLCV data for {crypto_name}...")
        
        # Use the coingecko tool to fetch data
        query = f"{crypto_name} ohlcv {days} days"
        coingecko_result = coingecko_tool.func(query)
        
        # Parse the result
        if isinstance(coingecko_result, str):
            try:
                coingecko_data = json.loads(coingecko_result)
            except json.JSONDecodeError as e:
                return f"Error: Failed to parse coingecko data: {str(e)}"
        else:
            coingecko_data = coingecko_result
        
        # Check for errors in the response
        if "error" in coingecko_data:
            return f"Error fetching data: {coingecko_data['error']}"
        
        # Extract OHLCV data from the response
        ohlcv_list = None
        if 'ohlcv_data' in coingecko_data and isinstance(coingecko_data['ohlcv_data'], list):
            ohlcv_list = coingecko_data['ohlcv_data']
            # Use the cryptocurrency name from the response if available
            if 'cryptocurrency' in coingecko_data:
                crypto_name = coingecko_data['cryptocurrency']
        elif isinstance(coingecko_data, list):
            ohlcv_list = coingecko_data
        else:
            return f"Error: Cannot find OHLCV data in coingecko response. Available keys: {list(coingecko_data.keys()) if isinstance(coingecko_data, dict) else 'Not a dict'}"
        
        if not ohlcv_list:
            return f"Error: Empty OHLCV data received for {crypto_name}"
        
        print(f"✅ Fetched {len(ohlcv_list)} data points for {crypto_name}")
        
        # Create DataFrame from the fetched data
        df = pd.DataFrame(ohlcv_list)
        
        if df.empty:
            return f"Error: No OHLCV data provided for {crypto_name}"
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return f"Error: Missing required columns: {missing_columns}. Available columns: {list(df.columns)}"
        
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
        
        # Generate chart and save for multimodal access
        chart_success = _create_technical_chart(df, indicators, crypto_name)
        
        if chart_success:
            summary += f"\n\n**Technical Analysis Chart:** Generated successfully and saved for multimodal analysis."
        else:
            summary += f"\n\n**Technical Analysis Chart:** Chart generation failed - see logs for details."
        
        return summary
        
    except Exception as e:
        return f"Error performing technical analysis: {str(e)}"


def _calculate_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive technical indicators from OHLCV data."""
    indicators = {}
    ta_config = Config.TA_INDICATORS
    
    def _calculate_rma(series: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Moving Average (RMA) also known as Modified Moving Average."""
        alpha = 1.0 / period
        rma = series.ewm(alpha=alpha, adjust=False).mean()
        return rma
    
    try:
        # Moving Averages
        for period in ta_config["sma_periods"]:
            if len(df) >= period:
                indicators[f"sma_{period}"] = ta.trend.SMAIndicator(df['close'], window=period).sma_indicator().iloc[-1]
        
        for period in ta_config["ema_periods"]:
            if len(df) >= period:
                indicators[f"ema_{period}"] = ta.trend.EMAIndicator(df['close'], window=period).ema_indicator().iloc[-1]
        
        # RMA (Relative Moving Average)
        for period in [14, 21, 50]:
            if len(df) >= period:
                rma_values = _calculate_rma(df['close'], period)
                indicators[f"rma_{period}"] = rma_values.iloc[-1]
        
        # RSI
        rsi_period = min(ta_config["rsi_period"], len(df) - 1)
        if rsi_period >= 2:
            indicators["rsi"] = ta.momentum.RSIIndicator(df['close'], window=rsi_period).rsi().iloc[-1]
        
        # Stochastic RSI
        if len(df) >= 14:
            stoch_rsi = ta.momentum.StochRSIIndicator(df['close'], window=14, smooth1=3, smooth2=3)
            indicators["stoch_rsi_k"] = stoch_rsi.stochrsi_k().iloc[-1]
            indicators["stoch_rsi_d"] = stoch_rsi.stochrsi_d().iloc[-1]
        
        # Williams %R
        if len(df) >= 14:
            williams_r = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14)
            indicators["williams_r"] = williams_r.williams_r().iloc[-1]
        
        # MACD
        if len(df) >= max(ta_config["macd_slow"], ta_config["macd_fast"]):
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
        if len(df) >= ta_config["bb_period"]:
            bb_indicator = ta.volatility.BollingerBands(
                df['close'],
                window=ta_config["bb_period"],
                window_dev=ta_config["bb_std"]
            )
            indicators["bb_upper"] = bb_indicator.bollinger_hband().iloc[-1]
            indicators["bb_middle"] = bb_indicator.bollinger_mavg().iloc[-1]
            indicators["bb_lower"] = bb_indicator.bollinger_lband().iloc[-1]
        
        # Additional indicators (with data length checks)
        if len(df) >= 14:
            # ATR
            atr_indicator = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
            indicators["atr"] = atr_indicator.average_true_range().iloc[-1]
            
            # Keltner Channels
            keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'], window=20)
            indicators["keltner_upper"] = keltner.keltner_channel_hband().iloc[-1]
            indicators["keltner_middle"] = keltner.keltner_channel_mband().iloc[-1]
            indicators["keltner_lower"] = keltner.keltner_channel_lband().iloc[-1]
            
            # Volume indicators
            indicators["volume_sma"] = ta.trend.SMAIndicator(df['volume'], window=min(20, len(df))).sma_indicator().iloc[-1]
            
            # On Balance Volume (OBV)
            obv_indicator = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
            indicators["obv"] = obv_indicator.on_balance_volume().iloc[-1]
            
            # Commodity Channel Index (CCI)
            cci_indicator = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=min(20, len(df)))
            indicators["cci"] = cci_indicator.cci().iloc[-1]
        
        # Price position calculations
        current_price = df['close'].iloc[-1]
        if "bb_upper" in indicators and "bb_lower" in indicators:
            indicators["bb_position"] = (current_price - indicators["bb_lower"]) / (indicators["bb_upper"] - indicators["bb_lower"])
        if "keltner_upper" in indicators and "keltner_lower" in indicators:
            indicators["keltner_position"] = (current_price - indicators["keltner_lower"]) / (indicators["keltner_upper"] - indicators["keltner_lower"])
            
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        indicators["error"] = str(e)
    
    return indicators


def _create_technical_chart(df: pd.DataFrame, indicators: Dict[str, Any], crypto_name: str) -> bool:
    """Create a comprehensive technical analysis chart and save it for multimodal access."""
    global _current_chart_data, _current_chart_path
    
    try:
        # Validate inputs
        if df.empty or len(df) < 2:
            return False
        
        # Import mplfinance for candlestick charts
        try:
            import mplfinance as mpf
        except ImportError:
            return _create_fallback_line_chart(df, indicators, crypto_name)
        
        # Prepare data for mplfinance
        if 'timestamp' in df.columns:
            # Handle both string and numeric timestamps
            if df['timestamp'].dtype == 'object':
                df['datetime'] = pd.to_datetime(df['timestamp'])
            else:
                # Handle millisecond vs second timestamps correctly
                if df['timestamp'].max() > 1e10:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
            df['datetime'] = pd.to_datetime(df.index)
        
        # Ensure data is sorted chronologically
        df = df.sort_values('datetime')
        
        # Set datetime as index for mplfinance
        chart_df = df.set_index('datetime')
        
        # Calculate current price and price change
        current_price = df['close'].iloc[-1]
        price_change = ((current_price - df['close'].iloc[0]) / df['close'].iloc[0] * 100) if len(df) > 1 else 0
        
        # Get date range
        actual_start_date = df['datetime'].min().strftime('%Y-%m-%d')
        actual_end_date = df['datetime'].max().strftime('%Y-%m-%d')
        
        # Enhanced title
        enhanced_title = (f'{crypto_name.title()} - Technical Analysis\n'
                        f'Data Range: {actual_start_date} to {actual_end_date} | '
                        f'${current_price:.2f} ({price_change:+.2f}%) | '
                        f'{len(df)} Candles')
        
        # Create TradingView-style configuration
        custom_style = mpf.make_mpf_style(
            base_mpf_style='charles',
            marketcolors=mpf.make_marketcolors(
                up='#26A69A',
                down='#EF5350',
                edge='inherit',
                wick={'up': '#26A69A', 'down': '#EF5350'},
                volume={'up': '#26A69A', 'down': '#EF5350'}
            ),
            facecolor='#131722',
            edgecolor='#2A2E39',
            gridcolor='#363A45',
            gridstyle='-',
            y_on_right=True
        )
        
        # Build additional plots
        apd = []
        panel_count = 1
        
        # Add moving averages if data is sufficient
        min_data_points = len(df)
        if min_data_points >= 12:
            ema_12 = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
            if not ema_12.empty and not ema_12.isna().all():
                apd.append(mpf.make_addplot(ema_12, color='#00CED1', width=2, alpha=0.8))
        
        if min_data_points >= 20:
            sma_20 = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            if not sma_20.empty and not sma_20.isna().all():
                apd.append(mpf.make_addplot(sma_20, color='#FFD700', width=2, alpha=0.8))
        
        # Add RSI panel
        if min_data_points >= 14:
            rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            if not rsi.empty and not rsi.isna().all():
                panel_count += 1
                apd.append(mpf.make_addplot(rsi, panel=1, color='#F59E0B', width=3, ylabel='RSI'))
                
                # RSI levels
                rsi_70 = pd.Series([70]*len(rsi), index=rsi.index)
                rsi_30 = pd.Series([30]*len(rsi), index=rsi.index)
                apd.extend([
                    mpf.make_addplot(rsi_70, panel=1, color='#DC2626', width=1, linestyle='--'),
                    mpf.make_addplot(rsi_30, panel=1, color='#059669', width=1, linestyle='--')
                ])
        
        # Add volume if available
        if 'volume' in chart_df.columns and (chart_df['volume'] > 0).any():
            panel_count += 1
            volume_colors = ['#10B981' if row['close'] >= row['open'] else '#EF4444' 
                           for _, row in chart_df.iterrows()]
            apd.append(mpf.make_addplot(chart_df['volume'], panel=panel_count-1, type='bar', 
                                     color=volume_colors, alpha=0.7, ylabel='Volume'))
        
        # Set panel ratios
        if panel_count == 1:
            panel_ratios = None
            figsize = (16, 10)
        elif panel_count == 2:
            panel_ratios = (3, 1)
            figsize = (16, 12)
        else:
            panel_ratios = (3, 1, 1.2)
            figsize = (16, 14)
        
        # Create the plot
        plot_kwargs = {
            'data': chart_df[['open', 'high', 'low', 'close']],
            'type': 'candle',
            'style': custom_style,
            'volume': False,
            'title': enhanced_title,
            'ylabel': 'Price ($)',
            'figsize': figsize,
            'returnfig': True,
            'tight_layout': True
        }
        
        if apd:
            plot_kwargs['addplot'] = apd
        if panel_ratios:
            plot_kwargs['panel_ratios'] = panel_ratios
        
        fig, axes = mpf.plot(**plot_kwargs)
        
        # Enhanced styling
        fig.patch.set_facecolor('#131722')
        fig.suptitle(enhanced_title, fontsize=14, fontweight='bold', color='#D1D4DC', y=0.98)
        
        # Configure legends with white font color
        for ax in fig.get_axes():
            if ax.get_legend():
                ax.get_legend().set_facecolor('#131722')
                for text in ax.get_legend().get_texts():
                    text.set_color('white')
        
        # Add custom legend for moving averages on main price panel
        if len(apd) > 0:
            # Create custom legend entries for moving averages
            legend_elements = []
            legend_labels = []
            
            if min_data_points >= 12:
                from matplotlib.lines import Line2D
                legend_elements.append(Line2D([0], [0], color='#00CED1', lw=2, alpha=0.8))
                legend_labels.append('EMA 12')
            
            if min_data_points >= 20:
                from matplotlib.lines import Line2D
                legend_elements.append(Line2D([0], [0], color='#FFD700', lw=2, alpha=0.8))
                legend_labels.append('SMA 20')
            
            if legend_elements:
                main_ax = fig.get_axes()[0]  # First axis is the main price chart
                legend = main_ax.legend(legend_elements, legend_labels, 
                                      loc='upper left', framealpha=0.9, 
                                      facecolor='#131722', edgecolor='#363A45')
                for text in legend.get_texts():
                    text.set_color('white')
                    text.set_fontsize(10)
        
        # Save chart to temporary file for multimodal access
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', prefix=f'chart_{crypto_name}_')
        fig.savefig(temp_file.name, format='png', dpi=200, bbox_inches='tight', 
                   facecolor='#131722', edgecolor='none', pad_inches=0.3)
        plt.close(fig)
        
        # Store paths and data globally
        _current_chart_path = temp_file.name
        
        # Also create base64 for backwards compatibility
        with open(temp_file.name, 'rb') as f:
            image_data = f.read()
            _current_chart_data = base64.b64encode(image_data).decode()
        
        return True
        
    except Exception as e:
        return _create_fallback_line_chart(df, indicators, crypto_name)


def _create_fallback_line_chart(df: pd.DataFrame, indicators: Dict[str, Any], crypto_name: str) -> bool:
    """Create a simple fallback chart if advanced charting fails."""
    global _current_chart_data, _current_chart_path
    
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s' if df['timestamp'].dtype in ['int64', 'float64'] else None)
        else:
            df['datetime'] = pd.to_datetime(df.index)
        
        current_price = df['close'].iloc[-1]
        price_change = ((current_price - df['close'].iloc[0]) / df['close'].iloc[0] * 100) if len(df) > 1 else 0
        
        ax.plot(df['datetime'], df['close'], color='#00D4AA', linewidth=2, label='Close Price')
        ax.set_title(f'{crypto_name.title()} - Price Chart (Fallback)\nCurrent: ${current_price:.2f} ({price_change:+.2f}%)', 
                    fontsize=14, color='white')
        ax.set_ylabel('Price ($)', color='white')
        
        # Create legend with white font color
        legend = ax.legend(facecolor='#1a1a1a', edgecolor='#444444', framealpha=0.9)
        for text in legend.get_texts():
            text.set_color('white')
        
        ax.grid(True, alpha=0.3)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', prefix=f'fallback_chart_{crypto_name}_')
        fig.savefig(temp_file.name, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='#1a1a1a', edgecolor='none')
        plt.close(fig)
        
        _current_chart_path = temp_file.name
        
        # Create base64 for backwards compatibility
        with open(temp_file.name, 'rb') as f:
            image_data = f.read()
            _current_chart_data = base64.b64encode(image_data).decode()
        
        return True
        
    except Exception as e:
        return False


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
            previous['close'] < previous['open'] and
            current['close'] > current['open'] and
            current['open'] < previous['close'] and
            current['close'] > previous['open']):
            patterns.append("Bullish Engulfing")
        
        # Bearish Engulfing
        if (len(last_2) >= 2 and
            previous['close'] > previous['open'] and
            current['close'] < current['open'] and
            current['open'] > previous['close'] and
            current['close'] < previous['open']):
            patterns.append("Bearish Engulfing")
        
        # Doji
        body_size = abs(current['close'] - current['open'])
        candle_range = current['high'] - current['low']
        if candle_range > 0 and body_size / candle_range < 0.1:
            patterns.append("Doji")
        
        # Hammer
        if (current['close'] > current['open'] and
            body_size > 0 and
            (current['close'] - current['low']) > 2 * body_size and
            (current['high'] - current['close']) < body_size):
            patterns.append("Hammer")
        
        # Shooting Star
        if (current['close'] < current['open'] and
            body_size > 0 and
            (current['high'] - current['open']) > 2 * body_size and
            (current['close'] - current['low']) < body_size):
            patterns.append("Shooting Star")
            
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
        
        # MACD interpretation
        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)
        
        if macd > macd_signal:
            macd_trend = "Bullish"
        else:
            macd_trend = "Bearish"
        
        interpretations["macd"] = f"{macd_trend} (MACD: {macd:.4f}, Signal: {macd_signal:.4f})"
        
        # Moving Average interpretation
        sma_20 = indicators.get("sma_20")
        sma_50 = indicators.get("sma_50")
        
        ma_trends = []
        if sma_20 and sma_50:
            if sma_20 > sma_50:
                ma_trends.append("SMA(20>50): Bullish")
            else:
                ma_trends.append("SMA(20<50): Bearish")
        
        interpretations["moving_averages"] = ", ".join(ma_trends) if ma_trends else "Insufficient data"
        
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


# Legacy wrapper for backward compatibility
class TechnicalAnalysisTool:
    """Legacy wrapper for the technical_analysis_tool function."""
    
    def __init__(self):
        self.name = "technical_analysis_tool"
        self.description = """
        Performs comprehensive technical analysis on cryptocurrency data by fetching fresh OHLCV data.
        Calculates various technical indicators and generates visual charts.
        Returns a textual summary with chart generation status.
        """
        self.ta_config = Config.TA_INDICATORS
    
    def _run(self, crypto_name: str, days: int = 30) -> str:
        """Legacy interface for the tool."""
        return technical_analysis_tool.func(crypto_name, days)


def create_technical_analysis_tool():
    """Create and return a technical analysis tool instance."""
    return technical_analysis_tool 