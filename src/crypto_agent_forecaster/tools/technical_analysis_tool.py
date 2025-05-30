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
        
        # Calculate key indicators for title
        rsi_value = indicators.get('rsi', 0)
        rsi_status = "Bullish" if rsi_value < 40 else "Bearish" if rsi_value > 60 else "Neutral"
        
        macd_val = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_status = "Bullish" if macd_val > macd_signal else "Bearish"
        
        # Enhanced professional title matching the debug chart
        enhanced_title = (f'{crypto_name.title()} - Professional Technical Analysis\n'
                        f'Data Range: {actual_start_date} to {actual_end_date} | '
                        f'${current_price:.2f} ({price_change:+.2f}%) | '
                        f'RSI: {rsi_value:.1f} | MACD: {macd_status} | {len(df)} Candles')
        
        # Create professional TradingView-style configuration
        custom_style = mpf.make_mpf_style(
            base_mpf_style='charles',
            marketcolors=mpf.make_marketcolors(
                up='#26A69A',      # Green for bullish candles
                down='#EF5350',    # Red for bearish candles
                edge='inherit',
                wick={'up': '#26A69A', 'down': '#EF5350'},
                volume={'up': '#26A69A40', 'down': '#EF535040'}  # Semi-transparent
            ),
            facecolor='#131722',   # Dark background
            edgecolor='#2A2E39',   # Chart edges
            gridcolor='#363A45',   # Grid lines
            gridstyle='-',
            y_on_right=True        # Price axis on right like TradingView
        )
        
        # Build additional plots
        apd = []
        panel_count = 1
        min_data_points = len(df)
        
        # Get professional colors from config
        colors = Config.TA_INDICATORS.get("professional_colors", {})
        
        # Add multiple moving averages with professional colors
        if min_data_points >= 9:
            ema_9 = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            if not ema_9.empty and not ema_9.isna().all():
                apd.append(mpf.make_addplot(ema_9, color=colors.get('ema_9', '#00D4AA'), width=2, alpha=0.9))  # Cyan
        
        if min_data_points >= 12:
            ema_12 = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
            if not ema_12.empty and not ema_12.isna().all():
                apd.append(mpf.make_addplot(ema_12, color=colors.get('ema_12', '#00CED1'), width=2, alpha=0.8))  # Light cyan
        
        if min_data_points >= 20:
            sma_20 = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            if not sma_20.empty and not sma_20.isna().all():
                apd.append(mpf.make_addplot(sma_20, color=colors.get('sma_20', '#FFD700'), width=2, alpha=0.8))  # Gold
        
        if min_data_points >= 26:
            ema_26 = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
            if not ema_26.empty and not ema_26.isna().all():
                apd.append(mpf.make_addplot(ema_26, color=colors.get('ema_26', '#FF8C00'), width=2, alpha=0.8))  # Orange
        
        if min_data_points >= 50:
            sma_50 = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
            if not sma_50.empty and not sma_50.isna().all():
                apd.append(mpf.make_addplot(sma_50, color=colors.get('sma_50', '#DA70D6'), width=2, alpha=0.8))  # Purple
        
        if min_data_points >= 100:
            sma_100 = ta.trend.SMAIndicator(df['close'], window=100).sma_indicator()
            if not sma_100.empty and not sma_100.isna().all():
                apd.append(mpf.make_addplot(sma_100, color=colors.get('sma_100', '#9370DB'), width=2, alpha=0.7))  # Medium purple
        
        if min_data_points >= 200:
            sma_200 = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
            if not sma_200.empty and not sma_200.isna().all():
                apd.append(mpf.make_addplot(sma_200, color=colors.get('sma_200', '#8A2BE2'), width=3, alpha=0.7))  # Blue violet
        
        # Add Bollinger Bands if available
        if min_data_points >= 20:
            bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            bb_upper = bb_indicator.bollinger_hband()
            bb_lower = bb_indicator.bollinger_lband()
            if not bb_upper.empty and not bb_lower.empty:
                bb_color = colors.get('bollinger_bands', '#87CEEB')
                apd.extend([
                    mpf.make_addplot(bb_upper, color=bb_color, width=1, alpha=0.5, linestyle='--'),
                    mpf.make_addplot(bb_lower, color=bb_color, width=1, alpha=0.5, linestyle='--')
                ])
        
        # Add RSI panel (panel 1)
        if min_data_points >= 14:
            rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            if not rsi.empty and not rsi.isna().all():
                panel_count += 1
                apd.append(mpf.make_addplot(rsi, panel=1, color=colors.get('rsi_line', '#F59E0B'), width=3, ylabel='RSI (14)'))
                
                # RSI overbought/oversold levels
                rsi_80 = pd.Series([80]*len(rsi), index=rsi.index)
                rsi_70 = pd.Series([70]*len(rsi), index=rsi.index)
                rsi_50 = pd.Series([50]*len(rsi), index=rsi.index)
                rsi_30 = pd.Series([30]*len(rsi), index=rsi.index)
                rsi_20 = pd.Series([20]*len(rsi), index=rsi.index)
                
                overbought_color = colors.get('rsi_overbought', '#DC2626')
                oversold_color = colors.get('rsi_oversold', '#059669')
                
                apd.extend([
                    mpf.make_addplot(rsi_80, panel=1, color=overbought_color, width=0.8, linestyle='--', alpha=0.6),
                    mpf.make_addplot(rsi_70, panel=1, color=overbought_color, width=1, linestyle='--', alpha=0.8),
                    mpf.make_addplot(rsi_50, panel=1, color='#6B7280', width=0.8, linestyle='-', alpha=0.5),
                    mpf.make_addplot(rsi_30, panel=1, color=oversold_color, width=1, linestyle='--', alpha=0.8),
                    mpf.make_addplot(rsi_20, panel=1, color=oversold_color, width=0.8, linestyle='--', alpha=0.6)
                ])
        
        # Add MACD panel (panel 2) with histogram
        if min_data_points >= 26:
            macd_indicator = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
            macd_line = macd_indicator.macd()
            macd_signal_line = macd_indicator.macd_signal()
            macd_histogram = macd_indicator.macd_diff()
            
            if not macd_line.empty and not macd_signal_line.empty and not macd_histogram.empty:
                panel_count += 1
                
                # MACD line and signal
                apd.extend([
                    mpf.make_addplot(macd_line, panel=2, color=colors.get('macd_line', '#00D4AA'), width=2, ylabel='MACD (12,26,9)'),
                    mpf.make_addplot(macd_signal_line, panel=2, color=colors.get('macd_signal', '#FF6B6B'), width=2)
                ])
                
                # MACD histogram with conditional colors
                pos_color = colors.get('macd_histogram_positive', '#10B981')
                neg_color = colors.get('macd_histogram_negative', '#EF4444')
                macd_hist_colors = [pos_color if val > 0 else neg_color for val in macd_histogram]
                apd.append(mpf.make_addplot(macd_histogram, panel=2, type='bar', 
                                         color=macd_hist_colors, alpha=0.6, width=0.8))
                
                # Zero line
                macd_zero = pd.Series([0]*len(macd_line), index=macd_line.index)
                apd.append(mpf.make_addplot(macd_zero, panel=2, color='#6B7280', width=0.8, 
                                         linestyle='-', alpha=0.5))
        
        # Add Volume with SMA overlay (panel 3)
        if 'volume' in chart_df.columns and (chart_df['volume'] > 0).any():
            panel_count += 1
            
            # Volume bars with conditional colors
            bull_vol_color = colors.get('volume_bullish', '#10B981')
            bear_vol_color = colors.get('volume_bearish', '#EF4444')
            volume_colors = [bull_vol_color if row['close'] >= row['open'] else bear_vol_color 
                           for _, row in chart_df.iterrows()]
            apd.append(mpf.make_addplot(chart_df['volume'], panel=panel_count-1, type='bar', 
                                     color=volume_colors, alpha=0.7, ylabel='Volume & SMA(20)'))
            
            # Volume SMA overlay
            if min_data_points >= 20:
                volume_sma = ta.trend.SMAIndicator(df['volume'], window=20).sma_indicator()
                if not volume_sma.empty and not volume_sma.isna().all():
                    apd.append(mpf.make_addplot(volume_sma, panel=panel_count-1, color=colors.get('volume_sma', '#FFD700'), 
                                             width=2, alpha=0.8))
        
        # Set panel ratios for professional layout
        if panel_count == 1:
            panel_ratios = None
            figsize = (20, 12)
        elif panel_count == 2:
            panel_ratios = (4, 1)
            figsize = (20, 14)
        elif panel_count == 3:
            panel_ratios = (4, 1, 1)
            figsize = (20, 16)
        else:
            panel_ratios = (4, 1, 1, 1.2)
            figsize = (20, 18)
        
        # Create the professional plot
        plot_kwargs = {
            'data': chart_df[['open', 'high', 'low', 'close']],
            'type': 'candle',
            'style': custom_style,
            'volume': False,  # We handle volume separately
            'title': enhanced_title,
            'ylabel': 'Price ($)',
            'figsize': figsize,
            'returnfig': True,
            'tight_layout': True,
            'show_nontrading': False
        }
        
        if apd:
            plot_kwargs['addplot'] = apd
        if panel_ratios:
            plot_kwargs['panel_ratios'] = panel_ratios
        
        fig, axes = mpf.plot(**plot_kwargs)
        
        # Enhanced professional styling
        fig.patch.set_facecolor('#131722')
        fig.suptitle(enhanced_title, fontsize=16, fontweight='bold', color='#D1D4DC', y=0.98)
        
        # Configure legends and professional appearance
        for ax in fig.get_axes():
            ax.set_facecolor('#131722')
            ax.tick_params(colors='#D1D4DC', which='both')
            ax.xaxis.label.set_color('#D1D4DC')
            ax.yaxis.label.set_color('#D1D4DC')
            
            # Style the grid
            ax.grid(True, color='#363A45', linestyle='-', linewidth=0.5, alpha=0.3)
            
            if ax.get_legend():
                ax.get_legend().set_facecolor('#131722')
                ax.get_legend().set_edgecolor('#363A45')
                for text in ax.get_legend().get_texts():
                    text.set_color('#D1D4DC')
        
        # Add custom legend for moving averages on main price panel
        if len(apd) > 0:
            # Create custom legend entries for moving averages
            legend_elements = []
            legend_labels = []
            
            from matplotlib.lines import Line2D
            
            if min_data_points >= 9:
                legend_elements.append(Line2D([0], [0], color=colors.get('ema_9', '#00D4AA'), lw=2, alpha=0.9))
                legend_labels.append('EMA 9')
            
            if min_data_points >= 12:
                legend_elements.append(Line2D([0], [0], color=colors.get('ema_12', '#00CED1'), lw=2, alpha=0.8))
                legend_labels.append('EMA 12')
            
            if min_data_points >= 20:
                legend_elements.append(Line2D([0], [0], color=colors.get('sma_20', '#FFD700'), lw=2, alpha=0.8))
                legend_labels.append('SMA 20')
            
            if min_data_points >= 26:
                legend_elements.append(Line2D([0], [0], color=colors.get('ema_26', '#FF8C00'), lw=2, alpha=0.8))
                legend_labels.append('EMA 26')
            
            if min_data_points >= 50:
                legend_elements.append(Line2D([0], [0], color=colors.get('sma_50', '#DA70D6'), lw=2, alpha=0.8))
                legend_labels.append('SMA 50')
            
            if legend_elements:
                main_ax = fig.get_axes()[0]  # First axis is the main price chart
                legend = main_ax.legend(legend_elements, legend_labels, 
                                      loc='upper left', framealpha=0.9, 
                                      facecolor='#131722', edgecolor='#363A45')
                for text in legend.get_texts():
                    text.set_color('#D1D4DC')
                    text.set_fontsize(10)
        
        # Save chart to temporary file for multimodal access
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', prefix=f'chart_{crypto_name}_')
        fig.savefig(temp_file.name, format='png', dpi=300, bbox_inches='tight', 
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
        print(f"Error creating professional chart: {e}")
        return _create_fallback_line_chart(df, indicators, crypto_name)


def _create_fallback_line_chart(df: pd.DataFrame, indicators: Dict[str, Any], crypto_name: str) -> bool:
    """Create a professional fallback chart if advanced charting fails."""
    global _current_chart_data, _current_chart_path
    
    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Set dark theme colors
        fig.patch.set_facecolor('#131722')
        for ax in [ax1, ax2]:
            ax.set_facecolor('#131722')
            ax.tick_params(colors='#D1D4DC', which='both')
            ax.xaxis.label.set_color('#D1D4DC')
            ax.yaxis.label.set_color('#D1D4DC')
            ax.grid(True, color='#363A45', linestyle='-', linewidth=0.5, alpha=0.3)
        
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s' if df['timestamp'].dtype in ['int64', 'float64'] else None)
        else:
            df['datetime'] = pd.to_datetime(df.index)
        
        current_price = df['close'].iloc[-1]
        price_change = ((current_price - df['close'].iloc[0]) / df['close'].iloc[0] * 100) if len(df) > 1 else 0
        rsi_value = indicators.get('rsi', 0)
        
        # Get professional colors from config
        colors = Config.TA_INDICATORS.get("professional_colors", {})
        
        # Main price chart
        ax1.plot(df['datetime'], df['close'], color='#00D4AA', linewidth=3, label='Close Price')
        
        # Add moving averages if available
        if len(df) >= 20:
            sma_20 = df['close'].rolling(window=20).mean()
            ax1.plot(df['datetime'], sma_20, color=colors.get('sma_20', '#FFD700'), linewidth=2, alpha=0.8, label='SMA 20')
        
        if len(df) >= 50:
            sma_50 = df['close'].rolling(window=50).mean()
            ax1.plot(df['datetime'], sma_50, color=colors.get('sma_50', '#DA70D6'), linewidth=2, alpha=0.8, label='SMA 50')
        
        ax1.set_title(f'{crypto_name.title()} - Professional Technical Analysis (Fallback)\n'
                     f'Current: ${current_price:.2f} ({price_change:+.2f}%) | RSI: {rsi_value:.1f}', 
                     fontsize=14, color='#D1D4DC', fontweight='bold')
        ax1.set_ylabel('Price ($)', color='#D1D4DC')
        
        # Create legend with professional style
        legend1 = ax1.legend(facecolor='#131722', edgecolor='#363A45', framealpha=0.9)
        for text in legend1.get_texts():
            text.set_color('#D1D4DC')
        
        # RSI subplot if available
        if 'rsi' in indicators and len(df) >= 14:
            # Calculate RSI for visualization
            rsi_series = pd.Series(index=df.index, dtype=float)
            rsi_series.iloc[-1] = indicators['rsi']
            
            # Simple RSI approximation for visualization
            for i in range(len(df) - 2, -1, -1):
                if i >= 13:  # Only calculate when we have enough data
                    gains = df['close'].diff().clip(lower=0)
                    losses = -df['close'].diff().clip(upper=0)
                    avg_gain = gains.rolling(window=14).mean().iloc[i]
                    avg_loss = losses.rolling(window=14).mean().iloc[i]
                    if avg_loss != 0:
                        rs = avg_gain / avg_loss
                        rsi_series.iloc[i] = 100 - (100 / (1 + rs))
            
            rsi_series = rsi_series.dropna()
            if not rsi_series.empty:
                ax2.plot(df['datetime'][-len(rsi_series):], rsi_series, color=colors.get('rsi_line', '#F59E0B'), linewidth=3, label='RSI')
                ax2.axhline(y=70, color=colors.get('rsi_overbought', '#DC2626'), linestyle='--', alpha=0.8, linewidth=1)
                ax2.axhline(y=30, color=colors.get('rsi_oversold', '#059669'), linestyle='--', alpha=0.8, linewidth=1)
                ax2.axhline(y=50, color='#6B7280', linestyle='-', alpha=0.5, linewidth=1)
                ax2.set_ylabel('RSI (14)', color='#D1D4DC')
                ax2.set_ylim(0, 100)
                
                legend2 = ax2.legend(facecolor='#131722', edgecolor='#363A45', framealpha=0.9)
                for text in legend2.get_texts():
                    text.set_color('#D1D4DC')
        else:
            # Remove the second subplot if no RSI
            ax2.remove()
            fig, ax1 = plt.subplots(1, 1, figsize=(16, 10))
            fig.patch.set_facecolor('#131722')
            ax1.set_facecolor('#131722')
            ax1.tick_params(colors='#D1D4DC', which='both')
            ax1.xaxis.label.set_color('#D1D4DC')
            ax1.yaxis.label.set_color('#D1D4DC')
            ax1.grid(True, color='#363A45', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Replot on single axis
            ax1.plot(df['datetime'], df['close'], color='#00D4AA', linewidth=3, label='Close Price')
            if len(df) >= 20:
                sma_20 = df['close'].rolling(window=20).mean()
                ax1.plot(df['datetime'], sma_20, color=colors.get('sma_20', '#FFD700'), linewidth=2, alpha=0.8, label='SMA 20')
            
            ax1.set_title(f'{crypto_name.title()} - Professional Technical Analysis (Fallback)\n'
                         f'Current: ${current_price:.2f} ({price_change:+.2f}%)', 
                         fontsize=14, color='#D1D4DC', fontweight='bold')
            ax1.set_ylabel('Price ($)', color='#D1D4DC')
            
            legend1 = ax1.legend(facecolor='#131722', edgecolor='#363A45', framealpha=0.9)
            for text in legend1.get_texts():
                text.set_color('#D1D4DC')
        
        plt.tight_layout()
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', prefix=f'fallback_chart_{crypto_name}_')
        fig.savefig(temp_file.name, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='#131722', edgecolor='none', pad_inches=0.3)
        plt.close(fig)
        
        _current_chart_path = temp_file.name
        
        # Create base64 for backwards compatibility
        with open(temp_file.name, 'rb') as f:
            image_data = f.read()
            _current_chart_data = base64.b64encode(image_data).decode()
        
        return True
        
    except Exception as e:
        print(f"Error creating fallback chart: {e}")
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