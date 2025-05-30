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
_current_results_path = None  # Store the current results directory


def get_current_chart_data() -> Optional[str]:
    """Get the current chart data (base64 encoded)."""
    global _current_chart_data
    return _current_chart_data


def get_current_chart_path() -> Optional[str]:
    """Get the current chart file path for multimodal access."""
    global _current_chart_path
    return _current_chart_path


def set_results_directory(results_dir: str):
    """Set the results directory for saving charts."""
    global _current_results_path
    _current_results_path = results_dir


def save_chart_to_results(chart_name: str = "technical_analysis_chart") -> Optional[str]:
    """
    Save the current chart to the results directory.
    
    Args:
        chart_name: Name for the saved chart file
        
    Returns:
        Path to the saved chart file, or None if saving failed
    """
    global _current_chart_path, _current_results_path
    
    if not _current_chart_path or not os.path.exists(_current_chart_path):
        return None
        
    if not _current_results_path:
        return None
    
    try:
        # Create charts directory in results
        charts_dir = os.path.join(_current_results_path, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # Copy chart file to results directory
        chart_filename = f"{chart_name}.png"
        results_chart_path = os.path.join(charts_dir, chart_filename)
        
        # Copy the chart file
        import shutil
        shutil.copy2(_current_chart_path, results_chart_path)
        
        print(f"✅ Chart saved to: {results_chart_path}")
        return results_chart_path
        
    except Exception as e:
        print(f"❌ Error saving chart to results: {e}")
        return None


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
def technical_analysis_tool(crypto_name: str, forecast_horizon: str = "24 hours", days: Optional[int] = None) -> str:
    """
    Performs comprehensive technical analysis on cryptocurrency data by fetching fresh OHLCV data 
    and generating visual charts optimized for the forecast horizon.
    
    Args:
        crypto_name: Name or symbol of the cryptocurrency to analyze (e.g., 'bitcoin', 'ethereum', 'BTC')
        forecast_horizon: The forecast time horizon to optimize analysis for (e.g., "1 hour", "24 hours", "3 days", "1 week")
        days: Number of days of historical data to fetch for analysis (optional, will be optimized for forecast_horizon if not provided)
    
    Returns:
        Textual summary with chart generation status and analysis insights
    """
    
    # Clear any existing chart data to prevent caching issues
    clear_chart_data()
    
    try:
        # Optimize data range based on forecast horizon if days not provided
        if days is None:
            days = _get_optimal_days_for_horizon(forecast_horizon)
        
        print(f"🔍 Fetching {days} days of OHLCV data for {crypto_name} (optimized for {forecast_horizon} forecast)...")
        
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
        
        # Identify patterns with horizon context
        patterns = _identify_candlestick_patterns(df, forecast_horizon)
        
        # Interpret indicators
        interpretations = _interpret_indicators(indicators, current_price)
        
        # Generate summary
        summary = _generate_summary(crypto_name, indicators, patterns, interpretations, current_price, forecast_horizon)
        
        # Generate enhanced chart with pattern annotations and horizon optimization
        chart_success = _create_enhanced_technical_chart(df, indicators, patterns, crypto_name, forecast_horizon)
        
        if chart_success:
            summary += f"\n\n**Enhanced Technical Analysis Chart:** Generated successfully with pattern annotations and {forecast_horizon} optimization."
        else:
            summary += f"\n\n**Technical Analysis Chart:** Chart generation failed - see logs for details."
        
        return summary
        
    except Exception as e:
        return f"Error performing technical analysis: {str(e)}"


def _get_optimal_days_for_horizon(forecast_horizon: str) -> int:
    """Get optimal historical data days based on forecast horizon for better technical analysis."""
    horizon_lower = forecast_horizon.lower()
    
    # Extract numeric value and time unit - optimized for technical indicators
    if "hour" in horizon_lower:
        if "1 hour" in horizon_lower or "1hr" in horizon_lower:
            return 14  # 2 weeks for 1 hour forecast
        elif "4 hour" in horizon_lower or "4hr" in horizon_lower:
            return 21  # 3 weeks for 4 hour forecast
        elif "12 hour" in horizon_lower or "12hr" in horizon_lower:
            return 30  # 1 month for 12 hour forecast
        else:
            return 21  # Default 3 weeks for hour-based forecasts
    elif "day" in horizon_lower:
        if "1 day" in horizon_lower:
            return 90  # 3 months for 1 day forecast
        elif "3 day" in horizon_lower:
            return 120  # 4 months for 3 day forecast
        elif "7 day" in horizon_lower:
            return 180  # 6 months for 1 week forecast
        else:
            return 90  # Default 3 months for day-based forecasts
    elif "week" in horizon_lower:
        if "1 week" in horizon_lower:
            return 180  # 6 months for 1 week forecast
        elif "2 week" in horizon_lower:
            return 270  # 9 months for 2 week forecast
        else:
            return 180  # Default 6 months for week-based forecasts
    elif "month" in horizon_lower:
        return 365  # 1 year for month-based forecasts
    else:
        return 90  # Default to 3 months for better indicator data


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
            print(f"⚠️ Insufficient data for chart: {len(df)} rows")
            return False
        
        # Import mplfinance for candlestick charts
        try:
            import mplfinance as mpf
        except ImportError:
            print("⚠️ mplfinance not available, using fallback chart")
            return _create_fallback_line_chart(df, indicators, crypto_name)
        
        # Debug: Print original data structure
        print(f"📊 Chart data structure - Rows: {len(df)}, Columns: {list(df.columns)}")
        if 'timestamp' in df.columns:
            print(f"📊 Timestamp sample: {df['timestamp'].iloc[0] if len(df) > 0 else 'No data'}")
            print(f"📊 Timestamp dtype: {df['timestamp'].dtype}")
        
        # Prepare data for mplfinance with better validation
        if 'timestamp' in df.columns:
            # Handle both string and numeric timestamps
            if df['timestamp'].dtype == 'object':
                # Try parsing as ISO format first (e.g., "2025-05-17T16:00:00")
                try:
                    df['datetime'] = pd.to_datetime(df['timestamp'])
                except Exception as e:
                    print(f"⚠️ Error parsing string timestamps: {e}")
                    # Fallback: try to parse as various other formats
                    df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
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
        
        # Validate that we have valid datetime data after parsing
        if df['datetime'].isna().any():
            print(f"⚠️ Found {df['datetime'].isna().sum()} invalid timestamps, removing...")
            df = df.dropna(subset=['datetime'])
            if len(df) < 2:
                print("❌ Insufficient valid data after timestamp validation")
                return _create_fallback_line_chart(df, indicators, crypto_name)
        
        # Set datetime as index for mplfinance - this is critical for proper candlestick rendering
        chart_df = df.set_index('datetime').copy()
        
        # Validate OHLC data integrity
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in chart_df.columns]
        if missing_cols:
            print(f"❌ Missing required OHLC columns: {missing_cols}")
            return _create_fallback_line_chart(df, indicators, crypto_name)
        
        # Check for invalid OHLC relationships
        invalid_data = (
            (chart_df['high'] < chart_df['low']) |
            (chart_df['high'] < chart_df['open']) |
            (chart_df['high'] < chart_df['close']) |
            (chart_df['low'] > chart_df['open']) |
            (chart_df['low'] > chart_df['close'])
        )
        if invalid_data.any():
            print(f"⚠️ Found {invalid_data.sum()} invalid OHLC relationships, fixing...")
            # Fix invalid relationships
            chart_df.loc[invalid_data, 'high'] = chart_df.loc[invalid_data, ['open', 'close']].max(axis=1)
            chart_df.loc[invalid_data, 'low'] = chart_df.loc[invalid_data, ['open', 'close']].min(axis=1)
        
        # Check for NaN values in OHLC data
        ohlc_nan_count = chart_df[required_cols].isnull().sum().sum()
        if ohlc_nan_count > 0:
            print(f"⚠️ Found {ohlc_nan_count} NaN values in OHLC data, forward filling...")
            chart_df[required_cols] = chart_df[required_cols].fillna(method='ffill')
        
        # Ensure all OHLC values are positive
        negative_values = (chart_df[required_cols] <= 0).any(axis=1)
        if negative_values.any():
            print(f"⚠️ Found {negative_values.sum()} rows with non-positive values, fixing...")
            chart_df = chart_df[~negative_values]
            if len(chart_df) < 2:
                print("❌ Insufficient valid data after cleaning")
                return _create_fallback_line_chart(df, indicators, crypto_name)
        
        # Ensure proper data types for mplfinance
        for col in required_cols:
            chart_df[col] = pd.to_numeric(chart_df[col], errors='coerce')
        
        # Remove any remaining NaN values
        chart_df = chart_df.dropna(subset=required_cols)
        
        if len(chart_df) < 2:
            print("❌ Insufficient valid OHLC data after cleaning")
            return _create_fallback_line_chart(df, indicators, crypto_name)
        
        # Debug: Validate final chart data
        print(f"✅ Chart data validated - {len(chart_df)} candles, OHLC range: "
              f"${chart_df['low'].min():.2f} - ${chart_df['high'].max():.2f}")
        
        # Check if all candles are identical (would appear as dots)
        price_variance = chart_df[required_cols].var().sum()
        if price_variance < 1e-10:
            print("⚠️ Very low price variance detected - candles may appear as dots")
            # Add small artificial variance to prevent dot appearance
            chart_df['high'] = chart_df['high'] * 1.0001
            chart_df['low'] = chart_df['low'] * 0.9999
        
        # Calculate current price and price change
        current_price = chart_df['close'].iloc[-1]
        price_change = ((current_price - chart_df['close'].iloc[0]) / chart_df['close'].iloc[0] * 100) if len(chart_df) > 1 else 0
        
        # Get date range
        actual_start_date = chart_df.index.min().strftime('%Y-%m-%d')
        actual_end_date = chart_df.index.max().strftime('%Y-%m-%d')
        
        # Calculate key indicators for title
        rsi_value = indicators.get('rsi', 0)
        rsi_status = "Bullish" if rsi_value < 40 else "Bearish" if rsi_value > 60 else "Neutral"
        
        macd_val = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_status = "Bullish" if macd_val > macd_signal else "Bearish"
        
        # Enhanced professional title
        enhanced_title = (f'{crypto_name.title()} - Professional Technical Analysis\n'
                        f'Data Range: {actual_start_date} to {actual_end_date} | '
                        f'${current_price:.2f} ({price_change:+.2f}%) | '
                        f'RSI: {rsi_value:.1f} | MACD: {macd_status} | {len(chart_df)} Candles')
        
        # Create professional TradingView-style configuration with proper candle sizing
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
            y_on_right=True,        # Price axis on right like TradingView
            rc={'font.size': 10}    # Better font size for readability
        )
        
        # Build additional plots
        apd = []
        panel_count = 1
        min_data_points = len(chart_df)
        
        # Get professional colors from config
        colors = Config.TA_INDICATORS.get("professional_colors", {})
        
        # Add multiple moving averages with professional colors
        if min_data_points >= 9:
            ema_9 = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            if not ema_9.empty and not ema_9.isna().all():
                # Align with chart_df index
                ema_9_aligned = ema_9.reindex(chart_df.index).fillna(method='ffill')
                apd.append(mpf.make_addplot(ema_9_aligned, color=colors.get('ema_9', '#00D4AA'), width=2, alpha=0.9))
        
        if min_data_points >= 12:
            ema_12 = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
            if not ema_12.empty and not ema_12.isna().all():
                ema_12_aligned = ema_12.reindex(chart_df.index).fillna(method='ffill')
                apd.append(mpf.make_addplot(ema_12_aligned, color=colors.get('ema_12', '#00CED1'), width=2, alpha=0.8))
        
        if min_data_points >= 20:
            sma_20 = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            if not sma_20.empty and not sma_20.isna().all():
                sma_20_aligned = sma_20.reindex(chart_df.index).fillna(method='ffill')
                apd.append(mpf.make_addplot(sma_20_aligned, color=colors.get('sma_20', '#FFD700'), width=2, alpha=0.8))
        
        if min_data_points >= 26:
            ema_26 = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
            if not ema_26.empty and not ema_26.isna().all():
                ema_26_aligned = ema_26.reindex(chart_df.index).fillna(method='ffill')
                apd.append(mpf.make_addplot(ema_26_aligned, color=colors.get('ema_26', '#FF8C00'), width=2, alpha=0.8))
        
        if min_data_points >= 50:
            sma_50 = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
            if not sma_50.empty and not sma_50.isna().all():
                sma_50_aligned = sma_50.reindex(chart_df.index).fillna(method='ffill')
                apd.append(mpf.make_addplot(sma_50_aligned, color=colors.get('sma_50', '#DA70D6'), width=2, alpha=0.8))
        
        if min_data_points >= 100:
            sma_100 = ta.trend.SMAIndicator(df['close'], window=100).sma_indicator()
            if not sma_100.empty and not sma_100.isna().all():
                sma_100_aligned = sma_100.reindex(chart_df.index).fillna(method='ffill')
                apd.append(mpf.make_addplot(sma_100_aligned, color=colors.get('sma_100', '#9370DB'), width=2, alpha=0.7))
        
        if min_data_points >= 200:
            sma_200 = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
            if not sma_200.empty and not sma_200.isna().all():
                sma_200_aligned = sma_200.reindex(chart_df.index).fillna(method='ffill')
                apd.append(mpf.make_addplot(sma_200_aligned, color=colors.get('sma_200', '#8A2BE2'), width=3, alpha=0.7))
        
        # Add Bollinger Bands if available
        if min_data_points >= 20:
            bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            bb_upper = bb_indicator.bollinger_hband()
            bb_lower = bb_indicator.bollinger_lband()
            if not bb_upper.empty and not bb_lower.empty:
                bb_color = colors.get('bollinger_bands', '#87CEEB')
                bb_upper_aligned = bb_upper.reindex(chart_df.index).fillna(method='ffill')
                bb_lower_aligned = bb_lower.reindex(chart_df.index).fillna(method='ffill')
                apd.extend([
                    mpf.make_addplot(bb_upper_aligned, color=bb_color, width=1, alpha=0.5, linestyle='--'),
                    mpf.make_addplot(bb_lower_aligned, color=bb_color, width=1, alpha=0.5, linestyle='--')
                ])
        
        # Add RSI panel (panel 1)
        if min_data_points >= 14:
            rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            if not rsi.empty and not rsi.isna().all():
                panel_count += 1
                rsi_aligned = rsi.reindex(chart_df.index).fillna(method='ffill')
                apd.append(mpf.make_addplot(rsi_aligned, panel=1, color=colors.get('rsi_line', '#F59E0B'), width=3, ylabel='RSI (14)'))
                
                # RSI overbought/oversold levels
                rsi_len = len(rsi_aligned)
                rsi_80 = pd.Series([80]*rsi_len, index=chart_df.index)
                rsi_70 = pd.Series([70]*rsi_len, index=chart_df.index)
                rsi_50 = pd.Series([50]*rsi_len, index=chart_df.index)
                rsi_30 = pd.Series([30]*rsi_len, index=chart_df.index)
                rsi_20 = pd.Series([20]*rsi_len, index=chart_df.index)
                
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
                
                # Align MACD indicators with chart index
                macd_line_aligned = macd_line.reindex(chart_df.index).fillna(method='ffill')
                macd_signal_aligned = macd_signal_line.reindex(chart_df.index).fillna(method='ffill')
                macd_hist_aligned = macd_histogram.reindex(chart_df.index).fillna(0)
                
                # MACD line and signal
                apd.extend([
                    mpf.make_addplot(macd_line_aligned, panel=2, color=colors.get('macd_line', '#00D4AA'), width=2, ylabel='MACD (12,26,9)'),
                    mpf.make_addplot(macd_signal_aligned, panel=2, color=colors.get('macd_signal', '#FF6B6B'), width=2)
                ])
                
                # MACD histogram with conditional colors
                pos_color = colors.get('macd_histogram_positive', '#10B981')
                neg_color = colors.get('macd_histogram_negative', '#EF4444')
                macd_hist_colors = [pos_color if val > 0 else neg_color for val in macd_hist_aligned]
                apd.append(mpf.make_addplot(macd_hist_aligned, panel=2, type='bar', 
                                         color=macd_hist_colors, alpha=0.6, width=0.8))
                
                # Zero line
                macd_zero = pd.Series([0]*len(macd_line_aligned), index=chart_df.index)
                apd.append(mpf.make_addplot(macd_zero, panel=2, color='#6B7280', width=0.8, 
                                         linestyle='-', alpha=0.5))
        
        # Add Volume with SMA overlay (panel 3)
        if 'volume' in chart_df.columns and (chart_df['volume'] > 0).any():
            panel_count += 1
            
            # Enhanced volume bars
            bull_vol_color = colors.get('volume_bullish', '#10B981')
            bear_vol_color = colors.get('volume_bearish', '#EF4444')
            volume_colors = [bull_vol_color if row['close'] >= row['open'] else bear_vol_color 
                           for _, row in chart_df.iterrows()]
            apd.append(mpf.make_addplot(chart_df['volume'], panel=panel_count-1, type='bar', 
                                     color=volume_colors, alpha=0.7, ylabel='Volume & SMA(20)'))
            
            # Volume SMA overlay with thicker line
            if min_data_points >= 20:
                volume_sma = ta.trend.SMAIndicator(df['volume'], window=20).sma_indicator()
                if not volume_sma.empty and not volume_sma.isna().all():
                    volume_sma_aligned = volume_sma.reindex(chart_df.index).fillna(method='ffill')
                    apd.append(mpf.make_addplot(volume_sma_aligned, panel=panel_count-1, color=colors.get('volume_sma', '#FFD700'), 
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
        
        # Create the professional plot with explicit OHLC data
        plot_kwargs = {
            'data': chart_df[['open', 'high', 'low', 'close']],  # Explicit OHLC order
            'type': 'candle',
            'style': custom_style,
            'volume': False,  # We handle volume separately
            'title': enhanced_title,
            'ylabel': 'Price ($)',
            'figsize': figsize,
            'returnfig': True,
            'tight_layout': True,
            'show_nontrading': False,
            'scale_padding': {'left': 0.3, 'top': 0.8, 'right': 1.0, 'bottom': 0.3}  # More space for legend
        }
        
        if apd:
            plot_kwargs['addplot'] = apd
        if panel_ratios:
            plot_kwargs['panel_ratios'] = panel_ratios
        
        # Debug: Print final plot parameters
        print(f"📊 Creating mplfinance plot with {len(chart_df)} candles")
        
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
        
        # Add custom legend for moving averages - Fixed positioning to avoid overlap
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
                
                # Fix legend positioning - use bottom left to avoid title overlap
                legend = main_ax.legend(legend_elements, legend_labels, 
                                      loc='lower left', framealpha=0.9, 
                                      facecolor='#131722', edgecolor='#363A45',
                                      bbox_to_anchor=(0.02, 0.02))  # Position in bottom left with padding
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
        
        print(f"✅ Chart created successfully with {len(chart_df)} candlesticks")
        return True
        
    except Exception as e:
        print(f"❌ Error creating professional chart: {e}")
        import traceback
        traceback.print_exc()
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
            # Handle both string and numeric timestamps properly
            if df['timestamp'].dtype == 'object':
                # Try parsing as ISO format first (e.g., "2025-05-17T16:00:00")
                try:
                    df['datetime'] = pd.to_datetime(df['timestamp'])
                except Exception as e:
                    print(f"⚠️ Error parsing string timestamps in fallback: {e}")
                    # Fallback: try to parse as various other formats
                    df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
            else:
                # Handle numeric timestamps
                if df['timestamp'].max() > 1e10:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
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


def _identify_candlestick_patterns(df: pd.DataFrame, forecast_horizon: str) -> List[str]:
    """Identify candlestick patterns in the data with forecast horizon context."""
    patterns = []
    
    try:
        if len(df) < 3:
            return patterns
        
        # Get the most recent candlesticks for pattern analysis
        recent_count = min(10, len(df))  # Look at last 10 candles
        recent_df = df.tail(recent_count).copy()
        
        # Calculate body and shadow ratios
        recent_df['body'] = abs(recent_df['close'] - recent_df['open'])
        recent_df['upper_shadow'] = recent_df['high'] - recent_df[['open', 'close']].max(axis=1)
        recent_df['lower_shadow'] = recent_df[['open', 'close']].min(axis=1) - recent_df['low']
        recent_df['total_range'] = recent_df['high'] - recent_df['low']
        
        # Avoid division by zero
        recent_df['body_ratio'] = recent_df['body'] / (recent_df['total_range'] + 1e-10)
        recent_df['upper_shadow_ratio'] = recent_df['upper_shadow'] / (recent_df['total_range'] + 1e-10)
        recent_df['lower_shadow_ratio'] = recent_df['lower_shadow'] / (recent_df['total_range'] + 1e-10)
        
        # Enhanced pattern detection with horizon context
        horizon_context = "short-term" if "hour" in forecast_horizon.lower() else "medium-term" if "day" in forecast_horizon.lower() else "long-term"
        
        for i in range(len(recent_df)):
            row = recent_df.iloc[i]
            
            # Doji patterns (small body)
            if row['body_ratio'] < 0.1:
                if row['upper_shadow_ratio'] > 0.4 and row['lower_shadow_ratio'] < 0.2:
                    patterns.append(f"🕯️ Dragonfly Doji (bullish reversal signal for {horizon_context} trends)")
                elif row['lower_shadow_ratio'] > 0.4 and row['upper_shadow_ratio'] < 0.2:
                    patterns.append(f"🕯️ Gravestone Doji (bearish reversal signal for {horizon_context} trends)")
                elif row['upper_shadow_ratio'] > 0.3 and row['lower_shadow_ratio'] > 0.3:
                    patterns.append(f"🕯️ Long-legged Doji (market indecision for {horizon_context} outlook)")
                else:
                    patterns.append(f"🕯️ Standard Doji (trend uncertainty for {horizon_context} forecast)")
            
            # Hammer patterns
            elif (row['lower_shadow_ratio'] > 0.5 and row['upper_shadow_ratio'] < 0.1 and 
                  row['close'] > row['open']):
                patterns.append(f"🔨 Hammer (strong bullish reversal for {horizon_context} forecast)")
            
            # Shooting star patterns
            elif (row['upper_shadow_ratio'] > 0.5 and row['lower_shadow_ratio'] < 0.1 and 
                  row['close'] < row['open']):
                patterns.append(f"⭐ Shooting Star (bearish reversal warning for {horizon_context} forecast)")
            
            # Large body patterns
            elif row['body_ratio'] > 0.7:
                if row['close'] > row['open']:
                    patterns.append(f"📈 Strong Bullish Candle (momentum continuation for {horizon_context})")
                else:
                    patterns.append(f"📉 Strong Bearish Candle (downward pressure for {horizon_context})")
        
        # Multi-candle patterns (if we have enough data)
        if len(recent_df) >= 3:
            # Engulfing patterns
            for i in range(1, len(recent_df)):
                prev_row = recent_df.iloc[i-1]
                curr_row = recent_df.iloc[i]
                
                # Bullish engulfing
                if (prev_row['close'] < prev_row['open'] and curr_row['close'] > curr_row['open'] and
                    curr_row['open'] < prev_row['close'] and curr_row['close'] > prev_row['open']):
                    patterns.append(f"🐂 Bullish Engulfing (strong buy signal for {horizon_context} outlook)")
                
                # Bearish engulfing
                elif (prev_row['close'] > prev_row['open'] and curr_row['close'] < curr_row['open'] and
                      curr_row['open'] > prev_row['close'] and curr_row['close'] < prev_row['open']):
                    patterns.append(f"🐻 Bearish Engulfing (strong sell signal for {horizon_context} outlook)")
        
        # Three-candle patterns
        if len(recent_df) >= 3:
            for i in range(2, len(recent_df)):
                first = recent_df.iloc[i-2]
                second = recent_df.iloc[i-1]
                third = recent_df.iloc[i]
                
                # Morning star pattern
                if (first['close'] < first['open'] and  # First candle bearish
                    second['body_ratio'] < 0.3 and  # Second candle small body (star)
                    third['close'] > third['open'] and  # Third candle bullish
                    third['close'] > (first['open'] + first['close']) / 2):  # Third closes above first's midpoint
                    patterns.append(f"🌅 Morning Star (powerful bullish reversal for {horizon_context} forecast)")
                
                # Evening star pattern
                elif (first['close'] > first['open'] and  # First candle bullish
                      second['body_ratio'] < 0.3 and  # Second candle small body (star)
                      third['close'] < third['open'] and  # Third candle bearish
                      third['close'] < (first['open'] + first['close']) / 2):  # Third closes below first's midpoint
                    patterns.append(f"🌆 Evening Star (strong bearish reversal for {horizon_context} forecast)")
        
        # Add pattern significance based on forecast horizon
        if patterns:
            if "hour" in forecast_horizon.lower():
                patterns.append(f"⚡ Short-term patterns detected - High relevance for {forecast_horizon} trading")
            elif "day" in forecast_horizon.lower():
                patterns.append(f"📅 Medium-term patterns - Moderate impact on {forecast_horizon} outlook")
            else:
                patterns.append(f"📊 Long-term patterns - Trend context for {forecast_horizon} forecast")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_patterns = []
        for pattern in patterns:
            if pattern not in seen:
                seen.add(pattern)
                unique_patterns.append(pattern)
        
        return unique_patterns[:5]  # Limit to 5 most relevant patterns
        
    except Exception as e:
        print(f"Error in pattern identification: {e}")
        return [f"⚠️ Pattern analysis error for {forecast_horizon} forecast"]


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
                     current_price: float, forecast_horizon: str) -> str:
    """Generate a comprehensive textual summary of technical analysis with forecast horizon context."""
    
    summary_parts = [
        f"**Enhanced Technical Analysis for {crypto_name.title()} - {forecast_horizon} Forecast**",
        f"Current Price: ${current_price:.4f}",
        f"Analysis Optimized for: {forecast_horizon} trading horizon",
        "",
        "**📊 Chart Patterns Detected:**"
    ]
    
    if patterns:
        for pattern in patterns:
            summary_parts.append(f"- {pattern}")
    else:
        summary_parts.append("- No significant patterns detected for this timeframe")
    
    summary_parts.extend([
        "",
        "**📈 Technical Indicators Analysis:**"
    ])
    
    for indicator, interpretation in interpretations.items():
        summary_parts.append(f"- {indicator.replace('_', ' ').title()}: {interpretation}")
    
    # Enhanced signal analysis based on forecast horizon
    bullish_signals = 0
    bearish_signals = 0
    neutral_signals = 0
    
    # RSI analysis with horizon context
    rsi = indicators.get("rsi", 50)
    if rsi < 30:
        bullish_signals += 2 if "hour" in forecast_horizon.lower() else 1  # More weight for short-term
        signal_strength = "Strong"
    elif rsi < 40:
        bullish_signals += 1
        signal_strength = "Moderate"
    elif rsi > 70:
        bearish_signals += 2 if "hour" in forecast_horizon.lower() else 1
        signal_strength = "Strong"
    elif rsi > 60:
        bearish_signals += 1
        signal_strength = "Moderate"
    else:
        neutral_signals += 1
        signal_strength = "Neutral"
    
    # MACD analysis
    macd = indicators.get("macd", 0)
    macd_signal = indicators.get("macd_signal", 0)
    macd_histogram = indicators.get("macd_histogram", 0)
    
    if macd > macd_signal:
        if macd_histogram > 0:
            bullish_signals += 2  # Strong bullish
        else:
            bullish_signals += 1  # Moderate bullish
    else:
        if macd_histogram < 0:
            bearish_signals += 2  # Strong bearish
        else:
            bearish_signals += 1  # Moderate bearish
    
    # Moving Average analysis with horizon optimization
    sma_20 = indicators.get("sma_20")
    sma_50 = indicators.get("sma_50")
    ema_9 = indicators.get("ema_9")
    ema_12 = indicators.get("ema_12")
    
    # Choose relevant MAs based on forecast horizon
    if "hour" in forecast_horizon.lower():
        # Short-term: Focus on EMA 9 and EMA 21
        if ema_9 and current_price > ema_9:
            bullish_signals += 1
        elif ema_9 and current_price < ema_9:
            bearish_signals += 1
    elif "day" in forecast_horizon.lower():
        # Medium-term: Focus on EMA 12, EMA 26, SMA 50
        if ema_12 and current_price > ema_12:
            bullish_signals += 1
        elif ema_12 and current_price < ema_12:
            bearish_signals += 1
    else:
        # Long-term: Focus on SMA 50, SMA 200
        if sma_20 and sma_50:
            if sma_20 > sma_50 and current_price > sma_20:
                bullish_signals += 2
            elif sma_20 < sma_50 and current_price < sma_20:
                bearish_signals += 2
            else:
                neutral_signals += 1
    
    # Bollinger Bands analysis
    bb_position = indicators.get("bb_position", 0.5)
    if bb_position > 0.8:
        bearish_signals += 1  # Potentially overbought
    elif bb_position < 0.2:
        bullish_signals += 1  # Potentially oversold
    else:
        neutral_signals += 1
    
    # Pattern signal analysis
    bullish_pattern_keywords = ["bullish", "hammer", "morning", "buy", "reversal"]
    bearish_pattern_keywords = ["bearish", "shooting", "evening", "sell", "down"]
    
    for pattern in patterns:
        pattern_lower = pattern.lower()
        if any(keyword in pattern_lower for keyword in bullish_pattern_keywords):
            if "strong" in pattern_lower or "powerful" in pattern_lower:
                bullish_signals += 2
            else:
                bullish_signals += 1
        elif any(keyword in pattern_lower for keyword in bearish_pattern_keywords):
            if "strong" in pattern_lower or "powerful" in pattern_lower:
                bearish_signals += 2
            else:
                bearish_signals += 1
    
    # Determine overall outlook with confidence
    total_signals = bullish_signals + bearish_signals + neutral_signals
    if total_signals > 0:
        bullish_percentage = (bullish_signals / total_signals) * 100
        bearish_percentage = (bearish_signals / total_signals) * 100
        neutral_percentage = (neutral_signals / total_signals) * 100
        
        if bullish_signals > bearish_signals + 1:
            if bullish_percentage > 70:
                overall = "Strong Bullish"
                confidence = "High"
            else:
                overall = "Bullish"
                confidence = "Medium"
        elif bearish_signals > bullish_signals + 1:
            if bearish_percentage > 70:
                overall = "Strong Bearish"
                confidence = "High"
            else:
                overall = "Bearish"
                confidence = "Medium"
        else:
            overall = "Neutral/Mixed"
            confidence = "Low" if abs(bullish_signals - bearish_signals) <= 1 else "Medium"
    else:
        overall = "Insufficient Data"
        confidence = "Low"
        bullish_percentage = bearish_percentage = neutral_percentage = 0
    
    # Add horizon-specific recommendations
    horizon_recommendations = []
    if "hour" in forecast_horizon.lower():
        horizon_recommendations = [
            "Focus on short-term momentum indicators (RSI, EMA 9)",
            "Watch for rapid price movements and volume spikes",
            "Consider scalping opportunities on pattern confirmations"
        ]
    elif "day" in forecast_horizon.lower():
        horizon_recommendations = [
            "Monitor medium-term trends and EMA crossovers",
            "Consider swing trading setups on pattern breaks",
            "Watch daily support/resistance levels"
        ]
    else:
        horizon_recommendations = [
            "Focus on long-term trend confirmation",
            "Monitor major moving average crossovers",
            "Consider position trading opportunities"
        ]
    
    summary_parts.extend([
        "",
        f"**🎯 {forecast_horizon} Technical Outlook: {overall}**",
        f"**Confidence Level: {confidence}**",
        f"Signal Distribution: {bullish_signals} Bullish | {bearish_signals} Bearish | {neutral_signals} Neutral",
        f"Bullish: {bullish_percentage:.1f}% | Bearish: {bearish_percentage:.1f}% | Neutral: {neutral_percentage:.1f}%",
        "",
        f"**📋 {forecast_horizon} Recommendations:**"
    ])
    
    for rec in horizon_recommendations:
        summary_parts.append(f"- {rec}")
    
    summary_parts.extend([
        "",
        f"**⚠️ Key Levels to Watch ({forecast_horizon}):**"
    ])
    
    # Add key levels based on indicators
    key_levels = []
    if sma_20:
        key_levels.append(f"SMA 20: ${sma_20:.2f} (medium-term support/resistance)")
    if sma_50:
        key_levels.append(f"SMA 50: ${sma_50:.2f} (major trend indicator)")
    
    bb_upper = indicators.get("bb_upper")
    bb_lower = indicators.get("bb_lower")
    if bb_upper and bb_lower:
        key_levels.append(f"Bollinger Upper: ${bb_upper:.2f} (resistance)")
        key_levels.append(f"Bollinger Lower: ${bb_lower:.2f} (support)")
    
    for level in key_levels[:4]:  # Limit to 4 key levels
        summary_parts.append(f"- {level}")
    
    return "\n".join(summary_parts)


def _create_enhanced_technical_chart(df: pd.DataFrame, indicators: Dict[str, Any], patterns: List[str], 
                                   crypto_name: str, forecast_horizon: str) -> bool:
    """Create an enhanced technical analysis chart with larger text, pattern annotations, and horizon optimization."""
    global _current_chart_data, _current_chart_path
    
    try:
        # Validate inputs
        if df.empty or len(df) < 2:
            return False
        
        # Import mplfinance for candlestick charts
        try:
            import mplfinance as mpf
            import matplotlib.patches as patches
        except ImportError:
            return _create_fallback_line_chart(df, indicators, crypto_name)
        
        # Prepare data for mplfinance
        if 'timestamp' in df.columns:
            # Handle both string and numeric timestamps
            if df['timestamp'].dtype == 'object':
                # Try parsing as ISO format first (e.g., "2025-05-17T16:00:00")
                try:
                    df['datetime'] = pd.to_datetime(df['timestamp'])
                except Exception as e:
                    print(f"⚠️ Error parsing string timestamps: {e}")
                    # Fallback: try to parse as various other formats
                    df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
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
        
        # Validate that we have valid datetime data after parsing
        if df['datetime'].isna().any():
            print(f"⚠️ Found {df['datetime'].isna().sum()} invalid timestamps, removing...")
            df = df.dropna(subset=['datetime'])
            if len(df) < 2:
                print("❌ Insufficient valid data after timestamp validation")
                return _create_fallback_line_chart(df, indicators, crypto_name)
        
        # Set datetime as index for mplfinance - this is critical for proper candlestick rendering
        chart_df = df.set_index('datetime').copy()
        
        # Validate OHLC data integrity
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in chart_df.columns]
        if missing_cols:
            print(f"❌ Missing required OHLC columns: {missing_cols}")
            return _create_fallback_line_chart(df, indicators, crypto_name)
        
        # Check for invalid OHLC relationships
        invalid_data = (
            (chart_df['high'] < chart_df['low']) |
            (chart_df['high'] < chart_df['open']) |
            (chart_df['high'] < chart_df['close']) |
            (chart_df['low'] > chart_df['open']) |
            (chart_df['low'] > chart_df['close'])
        )
        if invalid_data.any():
            print(f"⚠️ Found {invalid_data.sum()} invalid OHLC relationships, fixing...")
            # Fix invalid relationships
            chart_df.loc[invalid_data, 'high'] = chart_df.loc[invalid_data, ['open', 'close']].max(axis=1)
            chart_df.loc[invalid_data, 'low'] = chart_df.loc[invalid_data, ['open', 'close']].min(axis=1)
        
        # Check for NaN values in OHLC data
        ohlc_nan_count = chart_df[required_cols].isnull().sum().sum()
        if ohlc_nan_count > 0:
            print(f"⚠️ Found {ohlc_nan_count} NaN values in OHLC data, forward filling...")
            chart_df[required_cols] = chart_df[required_cols].fillna(method='ffill')
        
        # Ensure all OHLC values are positive
        negative_values = (chart_df[required_cols] <= 0).any(axis=1)
        if negative_values.any():
            print(f"⚠️ Found {negative_values.sum()} rows with non-positive values, fixing...")
            chart_df = chart_df[~negative_values]
            if len(chart_df) < 2:
                print("❌ Insufficient valid data after cleaning")
                return _create_fallback_line_chart(df, indicators, crypto_name)
        
        # Ensure proper data types for mplfinance
        for col in required_cols:
            chart_df[col] = pd.to_numeric(chart_df[col], errors='coerce')
        
        # Remove any remaining NaN values
        chart_df = chart_df.dropna(subset=required_cols)
        
        if len(chart_df) < 2:
            print("❌ Insufficient valid OHLC data after cleaning")
            return _create_fallback_line_chart(df, indicators, crypto_name)
        
        # Debug: Validate final chart data
        print(f"✅ Chart data validated - {len(chart_df)} candles, OHLC range: "
              f"${chart_df['low'].min():.2f} - ${chart_df['high'].max():.2f}")
        
        # Check if all candles are identical (would appear as dots)
        price_variance = chart_df[required_cols].var().sum()
        if price_variance < 1e-10:
            print("⚠️ Very low price variance detected - candles may appear as dots")
            # Add small artificial variance to prevent dot appearance
            chart_df['high'] = chart_df['high'] * 1.0001
            chart_df['low'] = chart_df['low'] * 0.9999
        
        # Calculate current price and price change
        current_price = chart_df['close'].iloc[-1]
        price_change = ((current_price - chart_df['close'].iloc[0]) / chart_df['close'].iloc[0] * 100) if len(chart_df) > 1 else 0
        
        # Get date range
        actual_start_date = df['datetime'].min().strftime('%Y-%m-%d')
        actual_end_date = df['datetime'].max().strftime('%Y-%m-%d')
        
        # Calculate key indicators for title
        rsi_value = indicators.get('rsi', 0)
        rsi_status = "Bullish" if rsi_value < 40 else "Bearish" if rsi_value > 60 else "Neutral"
        
        macd_val = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_status = "Bullish" if macd_val > macd_signal else "Bearish"
        
        # Enhanced professional title with forecast horizon
        enhanced_title = (f'{crypto_name.title()} - Enhanced Technical Analysis ({forecast_horizon} Forecast)\n'
                        f'Data: {actual_start_date} to {actual_end_date} | '
                        f'${current_price:.2f} ({price_change:+.2f}%) | '
                        f'RSI: {rsi_value:.1f} | MACD: {macd_status} | {len(df)} Candles')
        
        # Create professional TradingView-style configuration with larger fonts
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
        
        # Build additional plots with horizon-optimized indicators
        apd = []
        panel_count = 1
        min_data_points = len(df)
        
        # Get professional colors from config
        colors = Config.TA_INDICATORS.get("professional_colors", {})
        
        # Optimize indicators based on forecast horizon
        ma_periods = _get_horizon_optimized_indicators(forecast_horizon, min_data_points)
        
        # Add moving averages with optimized periods
        for ma_type, period in ma_periods.items():
            if min_data_points >= period:
                if ma_type.startswith('ema'):
                    ma_line = ta.trend.EMAIndicator(df['close'], window=period).ema_indicator()
                    color = colors.get(f'ema_{period}', '#00D4AA')
                else:
                    ma_line = ta.trend.SMAIndicator(df['close'], window=period).sma_indicator()
                    color = colors.get(f'sma_{period}', '#FFD700')
                
                if not ma_line.empty and not ma_line.isna().all():
                    width = 3 if period <= 20 else 2  # Thicker lines for shorter-term MAs
                    apd.append(mpf.make_addplot(ma_line, color=color, width=width, alpha=0.9))
        
        # Add Bollinger Bands if we have enough data
        if min_data_points >= 20:
            bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            bb_upper = bb_indicator.bollinger_hband()
            bb_lower = bb_indicator.bollinger_lband()
            if not bb_upper.empty and not bb_lower.empty:
                bb_color = colors.get('bollinger_bands', '#87CEEB')
                apd.extend([
                    mpf.make_addplot(bb_upper, color=bb_color, width=2, alpha=0.6, linestyle='--'),
                    mpf.make_addplot(bb_lower, color=bb_color, width=2, alpha=0.6, linestyle='--')
                ])
        
        # Add RSI panel with larger fonts
        if min_data_points >= 14:
            rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            if not rsi.empty and not rsi.isna().all():
                panel_count += 1
                apd.append(mpf.make_addplot(rsi, panel=1, color=colors.get('rsi_line', '#F59E0B'), 
                                         width=3, ylabel='RSI (14)'))
                
                # RSI levels with better visibility
                for level, color, alpha in [(80, '#DC2626', 0.7), (70, '#DC2626', 0.9), 
                                          (50, '#6B7280', 0.6), (30, '#059669', 0.9), (20, '#059669', 0.7)]:
                    rsi_level = pd.Series([level]*len(rsi), index=rsi.index)
                    apd.append(mpf.make_addplot(rsi_level, panel=1, color=color, 
                                             width=1.5, linestyle='--', alpha=alpha))
        
        # Add MACD panel with enhanced visibility
        if min_data_points >= 26:
            macd_indicator = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
            macd_line = macd_indicator.macd()
            macd_signal_line = macd_indicator.macd_signal()
            macd_histogram = macd_indicator.macd_diff()
            
            if not macd_line.empty and not macd_signal_line.empty and not macd_histogram.empty:
                panel_count += 1
                
                # MACD lines with thicker appearance
                apd.extend([
                    mpf.make_addplot(macd_line, panel=2, color=colors.get('macd_line', '#00D4AA'), 
                                   width=3, ylabel='MACD (12,26,9)'),
                    mpf.make_addplot(macd_signal_line, panel=2, color=colors.get('macd_signal', '#FF6B6B'), 
                                   width=3)
                ])
                
                # Enhanced MACD histogram
                pos_color = colors.get('macd_histogram_positive', '#10B981')
                neg_color = colors.get('macd_histogram_negative', '#EF4444')
                macd_hist_colors = [pos_color if val > 0 else neg_color for val in macd_histogram]
                apd.append(mpf.make_addplot(macd_histogram, panel=2, type='bar', 
                                         color=macd_hist_colors, alpha=0.7, width=1.0))
                
                # Zero line
                macd_zero = pd.Series([0]*len(macd_line), index=macd_line.index)
                apd.append(mpf.make_addplot(macd_zero, panel=2, color='#6B7280', width=1.5, 
                                         linestyle='-', alpha=0.6))
        
        # Add Volume panel with better visualization
        if 'volume' in chart_df.columns and (chart_df['volume'] > 0).any():
            panel_count += 1
            
            # Enhanced volume bars
            bull_vol_color = colors.get('volume_bullish', '#10B981')
            bear_vol_color = colors.get('volume_bearish', '#EF4444')
            volume_colors = [bull_vol_color if row['close'] >= row['open'] else bear_vol_color 
                           for _, row in chart_df.iterrows()]
            apd.append(mpf.make_addplot(chart_df['volume'], panel=panel_count-1, type='bar', 
                                     color=volume_colors, alpha=0.8, ylabel='Volume & SMA(20)'))
            
            # Volume SMA overlay with thicker line
            if min_data_points >= 20:
                volume_sma = ta.trend.SMAIndicator(df['volume'], window=20).sma_indicator()
                if not volume_sma.empty and not volume_sma.isna().all():
                    apd.append(mpf.make_addplot(volume_sma, panel=panel_count-1, 
                                             color=colors.get('volume_sma', '#FFD700'), 
                                             width=3, alpha=0.9))
        
        # Set panel ratios for enhanced layout
        if panel_count == 1:
            panel_ratios = None
            figsize = (24, 16)  # Larger figure size
        elif panel_count == 2:
            panel_ratios = (4, 1)
            figsize = (24, 18)
        elif panel_count == 3:
            panel_ratios = (4, 1, 1)
            figsize = (24, 20)
        else:
            panel_ratios = (4, 1, 1, 1.2)
            figsize = (24, 22)
        
        # Create the enhanced plot with larger text
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
        
        # Enhanced professional styling with larger fonts
        fig.patch.set_facecolor('#131722')
        fig.suptitle(enhanced_title, fontsize=20, fontweight='bold', color='#D1D4DC', y=0.98)
        
        # Configure enhanced legends and text sizes
        for ax in fig.get_axes():
            ax.set_facecolor('#131722')
            ax.tick_params(colors='#D1D4DC', which='both', labelsize=14)  # Larger tick labels
            ax.xaxis.label.set_color('#D1D4DC')
            ax.yaxis.label.set_color('#D1D4DC')
            ax.xaxis.label.set_fontsize(16)  # Larger axis labels
            ax.yaxis.label.set_fontsize(16)
            
            # Enhanced grid
            ax.grid(True, color='#363A45', linestyle='-', linewidth=0.7, alpha=0.4)
            
            if ax.get_legend():
                ax.get_legend().set_facecolor('#131722')
                ax.get_legend().set_edgecolor('#363A45')
                for text in ax.get_legend().get_texts():
                    text.set_color('#D1D4DC')
                    text.set_fontsize(12)  # Larger legend text
        
        # Add enhanced pattern annotations
        _add_pattern_annotations(fig, df, patterns, forecast_horizon)
        
        # Add custom legend for moving averages with larger text
        _add_enhanced_legend(fig, ma_periods, colors, min_data_points)
        
        # Save enhanced chart
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', prefix=f'enhanced_chart_{crypto_name}_')
        fig.savefig(temp_file.name, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='#131722', edgecolor='none', pad_inches=0.5)
        plt.close(fig)
        
        # Store paths and data globally
        _current_chart_path = temp_file.name
        
        # Create base64 for backwards compatibility
        with open(temp_file.name, 'rb') as f:
            image_data = f.read()
            _current_chart_data = base64.b64encode(image_data).decode()
        
        return True
        
    except Exception as e:
        print(f"Error creating enhanced chart: {e}")
        return _create_fallback_line_chart(df, indicators, crypto_name)


def _get_horizon_optimized_indicators(forecast_horizon: str, data_points: int) -> Dict[str, int]:
    """Get optimized moving average periods based on forecast horizon."""
    horizon_lower = forecast_horizon.lower()
    ma_periods = {}
    
    if "hour" in horizon_lower:
        # Short-term focus for hour-based forecasts
        if data_points >= 9:
            ma_periods['ema_9'] = 9
        if data_points >= 21:
            ma_periods['ema_21'] = 21
        if data_points >= 50:
            ma_periods['sma_50'] = 50
    elif "day" in horizon_lower and ("1 day" in horizon_lower or "3 day" in horizon_lower):
        # Medium-term focus for 1-3 day forecasts
        if data_points >= 12:
            ma_periods['ema_12'] = 12
        if data_points >= 26:
            ma_periods['ema_26'] = 26
        if data_points >= 50:
            ma_periods['sma_50'] = 50
        if data_points >= 100:
            ma_periods['sma_100'] = 100
    else:
        # Long-term focus for week+ forecasts
        if data_points >= 20:
            ma_periods['sma_20'] = 20
        if data_points >= 50:
            ma_periods['sma_50'] = 50
        if data_points >= 100:
            ma_periods['sma_100'] = 100
        if data_points >= 200:
            ma_periods['sma_200'] = 200
    
    return ma_periods


def _add_pattern_annotations(fig, df: pd.DataFrame, patterns: List[str], forecast_horizon: str):
    """Add pattern annotations to the chart with context."""
    if not patterns or fig.get_axes() is None:
        return
    
    main_ax = fig.get_axes()[0]  # Price chart axis
    
    # Add pattern annotations text box
    if patterns:
        pattern_text = f"📊 Chart Patterns ({forecast_horizon} Forecast):\n"
        for i, pattern in enumerate(patterns[:3]):  # Limit to 3 patterns for readability
            pattern_text += f"• {pattern}\n"
        
        # Add text box with pattern information
        main_ax.text(0.02, 0.95, pattern_text, transform=main_ax.transAxes,
                    fontsize=13, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                    facecolor='#2A2E39', edgecolor='#363A45', alpha=0.9),
                    color='#D1D4DC', fontweight='bold')
    
    # Add forecast horizon indicator
    horizon_text = f"🎯 Optimized for {forecast_horizon} forecast\n📈 {len(df)} data points analyzed"
    main_ax.text(0.98, 0.95, horizon_text, transform=main_ax.transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#2A2E39', 
                         edgecolor='#363A45', alpha=0.9),
                color='#D1D4DC')


def _add_enhanced_legend(fig, ma_periods: Dict[str, int], colors: Dict[str, str], min_data_points: int):
    """Add enhanced legend with larger text for moving averages."""
    if not ma_periods or fig.get_axes() is None:
        return
    
    main_ax = fig.get_axes()[0]
    legend_elements = []
    legend_labels = []
    
    from matplotlib.lines import Line2D
    
    for ma_type, period in ma_periods.items():
        if min_data_points >= period:
            if ma_type.startswith('ema'):
                color = colors.get(f'ema_{period}', '#00D4AA')
                label = f'EMA {period}'
            else:
                color = colors.get(f'sma_{period}', '#FFD700')
                label = f'SMA {period}'
            
            legend_elements.append(Line2D([0], [0], color=color, lw=3, alpha=0.9))
            legend_labels.append(label)
    
    if legend_elements:
        legend = main_ax.legend(legend_elements, legend_labels, 
                              loc='upper left', framealpha=0.9, 
                              facecolor='#131722', edgecolor='#363A45')
        for text in legend.get_texts():
            text.set_color('#D1D4DC')
            text.set_fontsize(14)  # Larger legend text


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
    
    def _run(self, crypto_name: str, forecast_horizon: str = "24 hours", days: Optional[int] = None) -> str:
        """Legacy interface for the tool with proper parameter handling."""
        return technical_analysis_tool.func(crypto_name, forecast_horizon, days)


def create_technical_analysis_tool():
    """Create and return a technical analysis tool instance."""
    return technical_analysis_tool 