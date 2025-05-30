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
def technical_analysis_tool(ohlcv_data: str, crypto_name: str = "Cryptocurrency") -> str:
    """
    Performs comprehensive technical analysis on cryptocurrency OHLCV data and always generates visual charts.
    
    Args:
        ohlcv_data: JSON string containing OHLCV data with columns: timestamp, open, high, low, close, volume
        crypto_name: Name of the cryptocurrency being analyzed (defaults to "Cryptocurrency")
    
    Returns:
        Textual summary with chart image always encoded as base64
    """
    
    # Clear any existing chart data to prevent caching issues
    global _current_chart_data
    _current_chart_data = None
    
    print(f"🔍 DEBUG: Technical analysis inputs - ohlcv_data type: {type(ohlcv_data)}, crypto_name: {crypto_name}")
    
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
            # Validate inputs
            if df.empty or len(df) < 2:
                print("Error: Insufficient data for chart generation")
                return ""
            
            # Import mplfinance for candlestick charts
            try:
                import mplfinance as mpf
                from matplotlib.patches import Rectangle
            except ImportError:
                print("mplfinance not available, falling back to line chart")
                # Fallback to line chart if mplfinance not available
                return _create_fallback_line_chart(df, indicators, crypto_name)
            
            # Prepare data for mplfinance
            if 'timestamp' in df.columns:
                # Handle both string and numeric timestamps
                if df['timestamp'].dtype == 'object':
                    # For ISO format strings like "2025-05-30T12:00:00"
                    df['datetime'] = pd.to_datetime(df['timestamp'])
                else:
                    # Handle millisecond vs second timestamps correctly
                    if df['timestamp'].max() > 1e10:
                        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    else:
                        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                df['datetime'] = pd.to_datetime(df.index)
            
            # Ensure data is sorted chronologically (oldest first)
            df = df.sort_values('datetime')
            
            # Debug: Print detailed timestamp information
            print(f"Raw timestamp sample: {df['timestamp'].iloc[0] if 'timestamp' in df.columns else 'N/A'}")
            print(f"Converted datetime sample: {df['datetime'].iloc[0]}")
            print(f"Chart data range: {df['datetime'].min()} to {df['datetime'].max()}")
            print(f"Total data points: {len(df)}")
            
            # Set datetime as index for mplfinance
            chart_df = df.set_index('datetime')
            
            # Ensure we have minimum required data points for indicators
            min_data_points = len(df)
            
            # Start with basic candlestick chart
            apd = []
            panel_count = 1
            
            # Adaptive periods based on available data
            ema_12_period = min(12, max(3, min_data_points // 4))
            sma_20_period = min(20, max(5, min_data_points // 3))
            ema_26_period = min(26, max(7, min_data_points // 2))
            sma_50_period = min(50, max(10, min_data_points - 5))
            rma_21_period = min(21, max(5, min_data_points // 3))
            
            # Add moving averages with adaptive periods
            if min_data_points >= 5:
                try:
                    ema_12 = ta.trend.EMAIndicator(df['close'], window=ema_12_period).ema_indicator()
                    if not ema_12.empty and not ema_12.isna().all():
                        apd.append(mpf.make_addplot(ema_12, color='#00CED1', width=2, alpha=0.8))
                except Exception as e:
                    print(f"Error adding EMA: {e}")
            
            if min_data_points >= 7:
                try:
                    sma_20 = ta.trend.SMAIndicator(df['close'], window=sma_20_period).sma_indicator()
                    ema_26 = ta.trend.EMAIndicator(df['close'], window=ema_26_period).ema_indicator()
                    if not sma_20.empty and not sma_20.isna().all():
                        apd.append(mpf.make_addplot(sma_20, color='#FFD700', width=2, alpha=0.8))
                    if not ema_26.empty and not ema_26.isna().all():
                        apd.append(mpf.make_addplot(ema_26, color='#FF69B4', width=2, alpha=0.8))
                except Exception as e:
                    print(f"Error adding SMA/EMA: {e}")
            
            if min_data_points >= 12:
                try:
                    sma_50 = ta.trend.SMAIndicator(df['close'], window=sma_50_period).sma_indicator()
                    if not sma_50.empty and not sma_50.isna().all():
                        apd.append(mpf.make_addplot(sma_50, color='#FF4500', width=2.5, alpha=0.8))
                except Exception as e:
                    print(f"Error adding SMA 50: {e}")
            
            # Add RMA with adaptive period
            if min_data_points >= 7:
                try:
                    rma_21 = _calculate_rma(df['close'], rma_21_period)
                    if not rma_21.empty and not rma_21.isna().all():
                        apd.append(mpf.make_addplot(rma_21, color='#4ECDC4', width=2, alpha=0.7, linestyle='--'))
                except Exception as e:
                    print(f"Error adding RMA: {e}")
            
            # Add Bollinger Bands with adaptive period
            if min_data_points >= 10:
                try:
                    bb_period = min(20, max(5, min_data_points // 2))
                    bb_indicator = ta.volatility.BollingerBands(df['close'], window=bb_period, window_dev=2)
                    bb_upper = bb_indicator.bollinger_hband()
                    bb_lower = bb_indicator.bollinger_lband()
                    bb_middle = bb_indicator.bollinger_mavg()
                    
                    if not bb_upper.empty and not bb_upper.isna().all():
                        apd.extend([
                            mpf.make_addplot(bb_upper, color='#8B5CF6', width=1.5, alpha=0.6, linestyle='-'),
                            mpf.make_addplot(bb_lower, color='#8B5CF6', width=1.5, alpha=0.6, linestyle='-'),
                            mpf.make_addplot(bb_middle, color='#8B5CF6', width=1.5, alpha=0.8, linestyle=':')
                        ])
                except Exception as e:
                    print(f"Error adding Bollinger Bands: {e}")
            
            # Enhanced RSI panel with adaptive period - Always try to show RSI
            if min_data_points >= 5:  # Lowered threshold
                try:
                    rsi_period = min(14, max(3, min_data_points // 2))
                    rsi = ta.momentum.RSIIndicator(df['close'], window=rsi_period).rsi()
                    if not rsi.empty and not rsi.isna().all():
                        panel_count += 1
                        rsi_panel = panel_count - 1
                        
                        # Main RSI line with better styling
                        apd.append(mpf.make_addplot(rsi, panel=rsi_panel, color='#F59E0B', width=3, ylabel='RSI'))
                        
                        # Add RSI levels with TradingView-like appearance
                        rsi_70 = pd.Series([70]*len(rsi), index=rsi.index)
                        rsi_50 = pd.Series([50]*len(rsi), index=rsi.index)
                        rsi_30 = pd.Series([30]*len(rsi), index=rsi.index)
                        
                        # Overbought/Oversold zones with filled areas
                        rsi_80 = pd.Series([80]*len(rsi), index=rsi.index)
                        rsi_20 = pd.Series([20]*len(rsi), index=rsi.index)
                        
                        apd.extend([
                            # Key levels
                            mpf.make_addplot(rsi_70, panel=rsi_panel, color='#DC2626', width=1.5, alpha=0.8, linestyle='--'),
                            mpf.make_addplot(rsi_50, panel=rsi_panel, color='#6B7280', width=1, alpha=0.6, linestyle=':'),
                            mpf.make_addplot(rsi_30, panel=rsi_panel, color='#059669', width=1.5, alpha=0.8, linestyle='--'),
                            # Extreme levels
                            mpf.make_addplot(rsi_80, panel=rsi_panel, color='#991B1B', width=1, alpha=0.5, linestyle='-'),
                            mpf.make_addplot(rsi_20, panel=rsi_panel, color='#065F46', width=1, alpha=0.5, linestyle='-')
                        ])
                        
                        print(f"RSI panel added with period {rsi_period}")
                except Exception as e:
                    print(f"Error adding RSI: {e}")
            
            # Enhanced MACD panel with adaptive periods - Always try to show MACD
            if min_data_points >= 10:  # Lowered threshold
                try:
                    # Adaptive MACD periods
                    fast_period = min(12, max(3, min_data_points // 4))
                    slow_period = min(26, max(6, min_data_points // 2))
                    signal_period = min(9, max(3, min_data_points // 5))
                    
                    macd_indicator = ta.trend.MACD(df['close'], window_slow=slow_period, window_fast=fast_period, window_sign=signal_period)
                    macd_line = macd_indicator.macd()
                    macd_signal = macd_indicator.macd_signal()
                    macd_histogram = macd_indicator.macd_diff()
                    
                    if not macd_line.empty and not macd_line.isna().all():
                        panel_count += 1
                        macd_panel = panel_count - 1
                        
                        # MACD line with enhanced styling
                        apd.append(mpf.make_addplot(macd_line, panel=macd_panel, color='#00D4AA', width=3, ylabel='MACD'))
                        
                        # Signal line
                        if not macd_signal.empty and not macd_signal.isna().all():
                            apd.append(mpf.make_addplot(macd_signal, panel=macd_panel, color='#FF6B6B', width=2.5))
                        
                        # Enhanced histogram with better colors
                        if not macd_histogram.empty and not macd_histogram.isna().all():
                            # Color histogram bars based on value and momentum
                            hist_colors = []
                            for i, val in enumerate(macd_histogram):
                                if pd.isna(val):
                                    hist_colors.append('#6B7280')
                                elif val >= 0:
                                    # Green gradient based on momentum
                                    if i > 0 and val > macd_histogram.iloc[i-1]:
                                        hist_colors.append('#10B981')  # Bright green (increasing)
                                    else:
                                        hist_colors.append('#059669')  # Dark green (decreasing)
                                else:
                                    # Red gradient based on momentum
                                    if i > 0 and val < macd_histogram.iloc[i-1]:
                                        hist_colors.append('#EF4444')  # Bright red (decreasing)
                                    else:
                                        hist_colors.append('#DC2626')  # Dark red (increasing)
                            
                            apd.append(mpf.make_addplot(macd_histogram, panel=macd_panel, type='bar', 
                                                      color=hist_colors, alpha=0.8, width=0.8))
                        
                        # Zero line for MACD
                        zero_line = pd.Series([0]*len(macd_line), index=macd_line.index)
                        apd.append(mpf.make_addplot(zero_line, panel=macd_panel, color='#6B7280', width=1, alpha=0.7, linestyle='-'))
                        
                        print(f"MACD panel added with periods {fast_period}/{slow_period}/{signal_period}")
                except Exception as e:
                    print(f"Error adding MACD: {e}")
            
            # Enhanced Volume panel with volume indicators
            if 'volume' in chart_df.columns:
                # Check if we have real volume data (not all zeros)
                has_volume = not chart_df['volume'].isna().all() and (chart_df['volume'] > 0).any()
                
                if has_volume:
                    panel_count += 1
                    volume_panel = panel_count - 1
                    
                    # Enhanced volume colors with gradient
                    volume_colors = []
                    max_volume = chart_df['volume'].max()
                    for _, row in chart_df.iterrows():
                        vol_intensity = min(row['volume'] / max_volume, 1.0) if max_volume > 0 else 0.5
                        if row['close'] >= row['open']:
                            # Green with intensity based on volume
                            alpha = 0.4 + (vol_intensity * 0.6)
                            volume_colors.append(f'rgba(16, 185, 129, {alpha})')
                        else:
                            # Red with intensity based on volume
                            alpha = 0.4 + (vol_intensity * 0.6)
                            volume_colors.append(f'rgba(239, 68, 68, {alpha})')
                    
                    # Fallback to simple colors for mplfinance compatibility
                    simple_colors = ['#10B981' if row['close'] >= row['open'] else '#EF4444' 
                                   for _, row in chart_df.iterrows()]
                    
                    # Add volume bars
                    apd.append(mpf.make_addplot(chart_df['volume'], panel=volume_panel, type='bar', 
                                             color=simple_colors, alpha=0.7, ylabel='Volume'))
                    
                    # Add volume moving average
                    if min_data_points >= 10:
                        try:
                            vol_sma_period = min(20, max(5, min_data_points // 2))
                            volume_sma = ta.trend.SMAIndicator(df['volume'], window=vol_sma_period).sma_indicator()
                            if not volume_sma.empty and not volume_sma.isna().all():
                                apd.append(mpf.make_addplot(volume_sma, panel=volume_panel, 
                                                          color='#FFA500', width=2.5, alpha=0.9, linestyle='-'))
                        except Exception as e:
                            print(f"Error adding volume SMA: {e}")
                    
                    # Enhanced volume spike detection
                    try:
                        if min_data_points >= 5:
                            vol_window = min(10, max(3, min_data_points // 3))
                            volume_mean = df['volume'].rolling(window=vol_window).mean()
                            volume_std = df['volume'].rolling(window=vol_window).std()
                            volume_threshold = volume_mean + (2 * volume_std)
                            
                            # Mark volume spikes
                            volume_spikes = df['volume'] > volume_threshold
                            if volume_spikes.any():
                                spike_volumes = df['volume'].where(volume_spikes)
                                apd.append(mpf.make_addplot(spike_volumes, panel=volume_panel, type='scatter',
                                                          color='#FFD700', markersize=40, alpha=0.9))
                    except Exception as e:
                        print(f"Error adding volume spike detection: {e}")
                else:
                    print("Volume data appears to be empty or all zeros - not displaying volume panel")
            
            # Calculate price change for title
            current_price = df['close'].iloc[-1] if not df['close'].empty else 0
            price_change = ((current_price - df['close'].iloc[0]) / df['close'].iloc[0] * 100) if len(df) > 1 else 0
            
            # Get actual date range from the data for verification
            actual_start_date = df['datetime'].min().strftime('%Y-%m-%d')
            actual_end_date = df['datetime'].max().strftime('%Y-%m-%d')
            
            # Create dynamic panel ratios optimized for TradingView-like appearance
            if panel_count == 1:
                panel_ratios = None  # Just main price panel
                figsize = (24, 14)
            elif panel_count == 2:
                panel_ratios = (5, 2)  # Price gets more space, one indicator
                figsize = (24, 16)
            elif panel_count == 3:
                panel_ratios = (5, 1.5, 1.5)  # Price, RSI, Volume or MACD
                figsize = (24, 18)
            elif panel_count == 4:
                panel_ratios = (5, 1.5, 1.5, 1.8)  # Price, RSI, MACD, Volume
                figsize = (24, 22)
            else:
                panel_ratios = (5, 1.5, 1.5, 1.5, 1.8)  # All panels
                figsize = (24, 24)
            
            # Enhanced TradingView-like style
            custom_style = mpf.make_mpf_style(
                base_mpf_style='charles',
                marketcolors=mpf.make_marketcolors(
                    up='#26A69A',    # TradingView green
                    down='#EF5350',  # TradingView red
                    edge='inherit',
                    wick={'up': '#26A69A', 'down': '#EF5350'},
                    volume={'up': '#26A69A', 'down': '#EF5350'}
                ),
                facecolor='#131722',  # TradingView dark background
                edgecolor='#2A2E39',
                gridcolor='#363A45',
                gridstyle='-',
                y_on_right=True  # TradingView style
            )
            
            # Create enhanced title with key indicator values and EXPLICIT date range
            current_rsi = indicators.get('rsi', float('nan'))
            macd_value = indicators.get('macd', 0)
            macd_signal_value = indicators.get('macd_signal', 0)
            macd_signal_strength = "Bullish" if macd_value > macd_signal_value else "Bearish"
            
            # Format indicator values for display
            rsi_display = f"{current_rsi:.1f}" if not pd.isna(current_rsi) else "N/A"
            
            # Enhanced title with explicit date verification
            enhanced_title = (f'{crypto_name.title()} - Professional Technical Analysis\n'
                            f'Data Range: {actual_start_date} to {actual_end_date} | '
                            f'${current_price:.2f} ({price_change:+.2f}%) | '
                            f'RSI: {rsi_display} | MACD: {macd_signal_strength} | '
                            f'{min_data_points} Candles')
            
            # Log the actual date range being used for debugging
            print(f"🔍 CHART DEBUG: Title date range - {actual_start_date} to {actual_end_date}")
            print(f"🔍 CHART DEBUG: Datetime range - {df['datetime'].min()} to {df['datetime'].max()}")
            print(f"🔍 CHART DEBUG: Chart title: {enhanced_title.split(chr(10))[0]}")  # First line only
            
            # Create the plot with TradingView-style configuration
            plot_kwargs = {
                'data': chart_df[['open', 'high', 'low', 'close']],
                'type': 'candle',
                'style': custom_style,
                'volume': False,  # We'll add volume as a separate panel
                'title': enhanced_title,
                'ylabel': 'Price ($)',
                'figsize': figsize,
                'returnfig': True,
                'tight_layout': True,
                'warn_too_much_data': len(df) * panel_count,
                'scale_width_adjustment': dict(volume=0.7, candle=1.0)  # TradingView proportions
            }
            
            # Only add addplot and panel_ratios if we have additional plots
            if apd:
                plot_kwargs['addplot'] = apd
            if panel_ratios:
                plot_kwargs['panel_ratios'] = panel_ratios
            
            fig, axes = mpf.plot(**plot_kwargs)
            
            # Enhanced TradingView-style customization
            fig.patch.set_facecolor('#131722')
            
            # Customize each axis with TradingView styling
            if isinstance(axes, list):
                for i, ax in enumerate(axes):
                    ax.set_facecolor('#131722')
                    ax.tick_params(colors='#D1D4DC', labelsize=10)
                    ax.xaxis.label.set_color('#D1D4DC')
                    ax.yaxis.label.set_color('#D1D4DC')
                    ax.grid(True, alpha=0.1, color='#363A45', linewidth=0.5)
                    
                    # Add professional panel labels
                    if i == 0:
                        ax.text(0.01, 0.98, '📈 Price & Indicators', transform=ax.transAxes,
                               fontsize=11, color='#D1D4DC', alpha=0.8, weight='bold',
                               verticalalignment='top')
                    elif 'RSI' in str(ax.get_ylabel()):
                        ax.text(0.01, 0.95, f'📊 RSI ({rsi_period})', transform=ax.transAxes,
                               fontsize=10, color='#F59E0B', alpha=0.9, weight='bold')
                        ax.set_ylim(0, 100)  # Standard RSI range
                    elif 'MACD' in str(ax.get_ylabel()):
                        ax.text(0.01, 0.95, f'📈 MACD ({fast_period},{slow_period},{signal_period})', transform=ax.transAxes,
                               fontsize=10, color='#00D4AA', alpha=0.9, weight='bold')
                    elif 'Volume' in str(ax.get_ylabel()):
                        ax.text(0.01, 0.95, f'📊 Volume & SMA({vol_sma_period if min_data_points >= 10 else 'N/A'})', transform=ax.transAxes,
                               fontsize=10, color='#FFA500', alpha=0.9, weight='bold')
            else:
                axes.set_facecolor('#131722')
                axes.tick_params(colors='#D1D4DC')
                axes.xaxis.label.set_color('#D1D4DC')
                axes.yaxis.label.set_color('#D1D4DC')
                axes.grid(True, alpha=0.1, color='#363A45')
            
            # Set enhanced title styling
            fig.suptitle(enhanced_title, fontsize=16, fontweight='bold', color='#D1D4DC', y=0.98)
            
            # Add professional footer with indicator details
            indicator_legend = (f"Indicators: SMA({sma_20_period},{sma_50_period}), EMA({ema_12_period},{ema_26_period}), "
                              f"RMA({rma_21_period}), Bollinger Bands, RSI({rsi_period}), "
                              f"MACD({fast_period},{slow_period},{signal_period}), Volume SMA({vol_sma_period if min_data_points >= 10 else 'N/A'})")
            fig.text(0.5, 0.01, indicator_legend, ha='center', fontsize=9, 
                    color='#787B86', alpha=0.8)
            
            # Convert to base64
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=200, bbox_inches='tight', 
                       facecolor='#131722', edgecolor='none', pad_inches=0.3)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            print(f"Successfully generated TradingView-style chart for {crypto_name} with {len(df)} data points and {panel_count} panels")
            return image_base64
            
        except Exception as e:
            print(f"Error creating enhanced OHLC chart: {e}")
            import traceback
            print(f"Chart generation traceback: {traceback.format_exc()}")
            # Fallback to the original line chart
            return _create_fallback_line_chart(df, indicators, crypto_name)
    
    def _create_fallback_line_chart(df: pd.DataFrame, indicators: Dict[str, Any], crypto_name: str) -> str:
        """Fallback line chart if OHLC chart fails."""
        try:
            # Original line chart code as fallback
            plt.style.use('dark_background')
            fig, axes = plt.subplots(4, 1, figsize=(16, 20), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
            
            current_price = df['close'].iloc[-1] if not df['close'].empty else 0
            price_change = ((current_price - df['close'].iloc[0]) / df['close'].iloc[0] * 100) if len(df) > 1 else 0
            
            fig.suptitle(f'{crypto_name.title()} - Technical Analysis (Fallback)\nCurrent: ${current_price:.2f} ({price_change:+.2f}%) | Data Points: {len(df)}', 
                        fontsize=16, fontweight='bold', color='white')
            
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s' if df['timestamp'].dtype in ['int64', 'float64'] else None)
            else:
                df['datetime'] = pd.to_datetime(df.index)
            
            # Simple line chart
            ax1 = axes[0]
            ax1.plot(df['datetime'], df['close'], color='#00D4AA', linewidth=2, label='Close Price', alpha=0.9)
            ax1.set_ylabel('Price ($)', fontsize=12, color='white')
            ax1.legend(loc='upper left', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(colors='white')
            
            # Simple placeholder for other panels
            for i, ax in enumerate(axes[1:], 1):
                ax.text(0.5, 0.5, f'Panel {i+1}', transform=ax.transAxes, 
                       ha='center', va='center', color='white', fontsize=12)
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a1a', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            print(f"Generated fallback line chart for {crypto_name}")
            return image_base64
            
        except Exception as e:
            print(f"Error creating fallback chart: {e}")
            import traceback
            print(f"Fallback chart traceback: {traceback.format_exc()}")
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
        # Handle different input formats from CrewAI
        print(f"🔍 DEBUG: Technical analysis inputs - ohlcv_data type: {type(ohlcv_data)}, crypto_name: {crypto_name}")
        
        # Validate inputs
        if not ohlcv_data:
            return f"Error: No OHLCV data provided"
        
        if not crypto_name or crypto_name.strip() == "":
            crypto_name = "Cryptocurrency"  # Default fallback
        
        # Ensure ohlcv_data is a string
        if isinstance(ohlcv_data, dict):
            # If we received a dict, try to extract the OHLCV data
            if 'ohlcv_data' in ohlcv_data:
                ohlcv_data = ohlcv_data['ohlcv_data']
            else:
                ohlcv_data = json.dumps(ohlcv_data)
        elif not isinstance(ohlcv_data, str):
            ohlcv_data = str(ohlcv_data)
        
        # Parse JSON data
        print(f"🔍 DEBUG: Processing {crypto_name} with {len(ohlcv_data)} characters of OHLCV data")
        ohlcv_list = json.loads(ohlcv_data)
        
        # Create DataFrame
        df = pd.DataFrame(ohlcv_list)
        print(f"🔍 DEBUG: Created DataFrame with {len(df)} rows and columns: {list(df.columns)}")
        
        # Timestamp verification for debugging
        if 'timestamp' in df.columns and not df.empty:
            first_timestamp = df['timestamp'].iloc[0]
            last_timestamp = df['timestamp'].iloc[-1]
            print(f"🔍 TIMESTAMP VERIFICATION: First={first_timestamp}, Last={last_timestamp}")
            
            # Additional check for date format
            if isinstance(first_timestamp, str) and 'T' in first_timestamp:
                # ISO format - extract date
                first_date = first_timestamp.split('T')[0]
                last_date = last_timestamp.split('T')[0]
                print(f"🔍 DATE VERIFICATION: Data range {first_date} to {last_date}")
                
                # Check if we're getting 2025 data (current year)
                if '2025' in first_date and '2025' in last_date:
                    print(f"✅ TIMESTAMP CHECK: Using current 2025 data ✅")
                else:
                    print(f"⚠️ TIMESTAMP WARNING: Data may be from wrong year! ⚠️")
        
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
        original_length = len(df)
        df = df.dropna()
        if len(df) < original_length:
            print(f"🔍 DEBUG: Removed {original_length - len(df)} rows with NaN values")
        
        if len(df) < 2:
            return f"Error: Insufficient data for technical analysis (need at least 2 data points, got {len(df)})"
        
        current_price = df['close'].iloc[-1]
        print(f"🔍 DEBUG: Current price from data: ${current_price:,.2f}")
        
        # Calculate indicators
        print("🔍 DEBUG: Calculating technical indicators...")
        indicators = _calculate_indicators(df)
        
        # Identify patterns
        print("🔍 DEBUG: Identifying candlestick patterns...")
        patterns = _identify_candlestick_patterns(df)
        
        # Interpret indicators
        print("🔍 DEBUG: Interpreting indicators...")
        interpretations = _interpret_indicators(indicators, current_price)
        
        # Generate summary
        print("🔍 DEBUG: Generating analysis summary...")
        summary = _generate_summary(crypto_name, indicators, patterns, interpretations, current_price)
        
        # Always generate chart - this is now mandatory
        print("🔍 DEBUG: Generating technical analysis chart...")
        chart_image = _create_technical_chart(df, indicators, crypto_name)
        if chart_image:
            _current_chart_data = chart_image
            summary += f"\n\n**Technical Analysis Chart:** Generated successfully and ready for analysis."
            print(f"✅ Chart generated successfully ({len(chart_image)} characters)")
        else:
            summary += f"\n\n**Technical Analysis Chart:** Chart generation failed - see logs for details."
            print(f"⚠️ Chart generation failed")
        
        print(f"🔍 DEBUG: Technical analysis completed for {crypto_name}")
        return summary
        
    except json.JSONDecodeError as e:
        error_msg = f"Error: Invalid JSON in OHLCV data: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Error performing technical analysis: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return error_msg


# Legacy wrapper for backward compatibility
class TechnicalAnalysisTool:
    """Legacy wrapper for the technical_analysis_tool function."""
    
    def __init__(self):
        self.name = "technical_analysis_tool"
        self.description = """
        Performs comprehensive technical analysis on cryptocurrency OHLCV data.
        Calculates various technical indicators including RMA, Stochastic RSI, Williams %R, CCI, ATR, OBV.
        Identifies candlestick patterns and always generates visual charts.
        Returns a textual summary with chart image encoded as base64.
        """
        self.ta_config = Config.TA_INDICATORS
    
    def _run(self, ohlcv_data: str, crypto_name: str) -> str:
        """Legacy interface for the tool."""
        return technical_analysis_tool.func(ohlcv_data, crypto_name)


def create_technical_analysis_tool():
    """Create and return a technical analysis tool instance."""
    return technical_analysis_tool 