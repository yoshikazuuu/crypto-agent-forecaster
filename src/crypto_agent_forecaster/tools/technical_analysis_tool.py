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
def technical_analysis_tool(crypto_name: str, forecast_horizon: str = "24 hours") -> str:
    """
    Performs comprehensive technical analysis on cryptocurrency data by fetching fresh OHLCV data 
    and generating visual charts optimized for the forecast horizon.
    
    Args:
        crypto_name: Name or symbol of the cryptocurrency to analyze (e.g., 'bitcoin', 'ethereum', 'BTC')
        forecast_horizon: The forecast time horizon to optimize analysis for (e.g., "1 hour", "24 hours", "3 days", "1 week")
    
    Returns:
        Textual summary with chart generation status and analysis insights
    """
    
    # Clear any existing chart data to prevent caching issues
    clear_chart_data()
    
    try:
        # Automatically optimize data range based on forecast horizon
        days = _get_optimal_days_for_horizon(forecast_horizon)
        
        print(f"🔍 Fetching {days} days of OHLCV data for {crypto_name} (optimized for {forecast_horizon} forecast)...")
        
        # Use the coingecko tool to fetch data with consistent query format
        query = f"{crypto_name} ohlcv {days} days horizon {forecast_horizon}"
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
        
        # Extract OHLCV data from the response with enhanced validation
        ohlcv_list = None
        data_metadata = {}
        
        if 'ohlcv_data' in coingecko_data and isinstance(coingecko_data['ohlcv_data'], list):
            ohlcv_list = coingecko_data['ohlcv_data']
            # Extract metadata for consistency validation
            if 'cryptocurrency' in coingecko_data:
                crypto_name = coingecko_data['cryptocurrency']
            data_metadata = {
                'data_points': coingecko_data.get('data_points', len(ohlcv_list)),
                'has_volume_data': coingecko_data.get('has_volume_data', False),
                'data_source': coingecko_data.get('data_source', 'unknown'),
                'resolution': coingecko_data.get('resolution', 'unknown')
            }
        elif isinstance(coingecko_data, list):
            ohlcv_list = coingecko_data
            data_metadata = {'data_points': len(ohlcv_list), 'data_source': 'legacy_format'}
        else:
            return f"Error: Cannot find OHLCV data in coingecko response. Available keys: {list(coingecko_data.keys()) if isinstance(coingecko_data, dict) else 'Not a dict'}"
        
        if not ohlcv_list:
            return f"Error: Empty OHLCV data received for {crypto_name}"
        
        print(f"✅ Fetched {len(ohlcv_list)} data points for {crypto_name}")
        print(f"📊 Data metadata: {data_metadata}")
        
        # Create DataFrame from the fetched data with enhanced validation
        df = pd.DataFrame(ohlcv_list)
        
        if df.empty:
            return f"Error: No OHLCV data provided for {crypto_name}"
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return f"Error: Missing required columns: {missing_columns}. Available columns: {list(df.columns)}"
        
        # ENHANCED DATA PREPROCESSING FOR CONSISTENCY
        print(f"🔧 Preprocessing data for consistency...")
        
        # Convert timestamp column to standardized datetime format
        if 'timestamp' in df.columns:
            df = _standardize_timestamps(df)
        else:
            print(f"⚠️ No timestamp column found - adding sequential timestamps")
            df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
            df['datetime'] = df['timestamp']
        
        # Ensure data is sorted chronologically (oldest first)
        if 'datetime' in df.columns:
            df = df.sort_values('datetime').reset_index(drop=True)
            print(f"✅ Data sorted chronologically: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
        
        # Convert OHLCV columns to numeric with enhanced validation
        for col in required_columns:
            original_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check for conversion issues
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                print(f"⚠️ Found {nan_count} NaN values in {col} after conversion")
            
            if original_dtype != df[col].dtype:
                print(f"✅ Converted {col} from {original_dtype} to numeric")
        
        # Remove any rows with NaN values in critical columns
        pre_clean_len = len(df)
        df = df.dropna(subset=required_columns)
        post_clean_len = len(df)
        
        if pre_clean_len != post_clean_len:
            print(f"🧹 Removed {pre_clean_len - post_clean_len} rows with NaN values")
        
        if len(df) < 2:
            return f"Error: Insufficient data for technical analysis (need at least 2 data points, got {len(df)})"
        
        # Validate OHLC relationships for data integrity
        invalid_ohlc = _validate_ohlc_data(df)
        if invalid_ohlc > 0:
            print(f"🔧 Fixed {invalid_ohlc} invalid OHLC relationships")
        
        # Create a clean copy for indicators and charts to ensure consistency
        analysis_df = df.copy()
        current_price = analysis_df['close'].iloc[-1]
        
        print(f"✅ Data preprocessing complete: {len(analysis_df)} valid data points")
        print(f"📊 Price range: ${analysis_df['low'].min():.2f} - ${analysis_df['high'].max():.2f}")
        print(f"💰 Current price: ${current_price:.2f}")
        
        # Calculate indicators using the cleaned data
        print(f"🧮 Calculating technical indicators...")
        indicators = _calculate_indicators(analysis_df)
        
        # Identify patterns with horizon context using the same data
        print(f"🔍 Identifying chart patterns...")
        patterns = _identify_candlestick_patterns(analysis_df, forecast_horizon)
        
        # Interpret indicators
        print(f"📊 Interpreting indicators...")
        interpretations = _interpret_indicators(indicators, current_price)
        
        # Generate summary
        summary = _generate_summary(crypto_name, indicators, patterns, interpretations, current_price, forecast_horizon)
        
        # Generate enhanced chart using THE SAME data that was used for indicators
        print(f"📈 Creating enhanced technical chart...")
        chart_success = _create_enhanced_technical_chart(
            analysis_df,  # Use the SAME preprocessed data
            indicators, 
            patterns, 
            crypto_name, 
            forecast_horizon,
            data_metadata  # Pass metadata for validation
        )
        
        if chart_success:
            summary += f"\n\n**Enhanced Technical Analysis Chart:** Generated successfully with pattern annotations and {forecast_horizon} optimization."
            summary += f"\nChart data points: {len(analysis_df)} | Resolution: {data_metadata.get('resolution', 'auto')}"
        else:
            summary += f"\n\n**Technical Analysis Chart:** Chart generation failed - see logs for details."
        
        return summary
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error performing technical analysis: {str(e)}"


def _standardize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize timestamp format for consistency across the system."""
    try:
        print(f"🕒 Standardizing timestamps...")
        
        if 'timestamp' not in df.columns:
            print(f"⚠️ No timestamp column to standardize")
            return df
        
        # Get a sample of the timestamp data
        sample_timestamp = df['timestamp'].iloc[0]
        print(f"📊 Sample timestamp: {sample_timestamp} (type: {type(sample_timestamp)})")
        
        if df['timestamp'].dtype == 'object':
            # String timestamps - try parsing as ISO format first
            try:
                df['datetime'] = pd.to_datetime(df['timestamp'])
                print(f"✅ Parsed string timestamps as ISO datetime")
            except Exception as e:
                print(f"⚠️ ISO parsing failed: {e}, trying flexible parsing...")
                df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
                # Check for parsing failures
                failed_count = df['datetime'].isna().sum()
                if failed_count > 0:
                    print(f"❌ Failed to parse {failed_count} timestamps")
                    # Create sequential timestamps as fallback
                    start_time = pd.Timestamp.now() - pd.Timedelta(days=30)
                    df['datetime'] = pd.date_range(start=start_time, periods=len(df), freq='H')
                    print(f"🔧 Generated sequential timestamps as fallback")
        else:
            # Numeric timestamps
            if df['timestamp'].max() > 1e10:
                # Milliseconds
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                print(f"✅ Converted millisecond timestamps to datetime")
            else:
                # Seconds
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                print(f"✅ Converted second timestamps to datetime")
        
        # Validate the datetime column
        if df['datetime'].isna().any():
            valid_count = df['datetime'].notna().sum()
            print(f"⚠️ {len(df) - valid_count} invalid datetime entries, will be filtered out")
        
        return df
        
    except Exception as e:
        print(f"❌ Timestamp standardization failed: {e}")
        # Fallback: create sequential timestamps
        df['datetime'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
        print(f"🔧 Using sequential timestamps as fallback")
        return df


def _validate_ohlc_data(df: pd.DataFrame) -> int:
    """Validate and fix OHLC data relationships."""
    try:
        # Check for invalid OHLC relationships
        invalid_conditions = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        invalid_count = invalid_conditions.sum()
        
        if invalid_count > 0:
            print(f"🔧 Fixing {invalid_count} invalid OHLC relationships...")
            
            # Fix invalid relationships by ensuring logical constraints
            # High should be the maximum of all values
            df.loc[invalid_conditions, 'high'] = df.loc[invalid_conditions, ['open', 'high', 'low', 'close']].max(axis=1)
            
            # Low should be the minimum of all values
            df.loc[invalid_conditions, 'low'] = df.loc[invalid_conditions, ['open', 'high', 'low', 'close']].min(axis=1)
            
            print(f"✅ Fixed OHLC relationships")
        
        # Ensure all prices are positive
        negative_mask = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        negative_count = negative_mask.sum()
        
        if negative_count > 0:
            print(f"⚠️ Found {negative_count} rows with non-positive prices, removing...")
            df = df[~negative_mask]
        
        return invalid_count
        
    except Exception as e:
        print(f"❌ OHLC validation failed: {e}")
        return 0


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
    """Create a comprehensive technical analysis chart and save it for multimodal access with data consistency."""
    global _current_chart_data, _current_chart_path
    
    try:
        # Validate inputs
        if df.empty or len(df) < 2:
            print(f"⚠️ Chart creation failed: Insufficient data for chart: {len(df)} rows")
            return False
        
        # Ensure we have datetime column for consistency
        if 'datetime' not in df.columns:
            if 'timestamp' in df.columns:
                df = _standardize_timestamps(df)
            else:
                print(f"⚠️ No datetime available - creating sequential timestamps")
                df['datetime'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
        
        # Import mplfinance for candlestick charts
        try:
            import mplfinance as mpf
            print(f"✅ mplfinance library imported successfully for candlestick charts")
        except ImportError:
            print("❌ mplfinance not available - falling back to simple line chart")
            return _create_fallback_line_chart(df, indicators, crypto_name)
        
        # Use consistent data processing
        chart_df = df.set_index('datetime').copy()
        print(f"✅ Technical chart using consistent data with {len(chart_df)} rows")
        
        # Validate OHLC data
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in chart_df.columns]
        if missing_cols:
            print(f"❌ Missing required OHLC columns: {missing_cols}")
            return _create_fallback_line_chart(df, indicators, crypto_name)
        
        # Remove any NaN values
        chart_df = chart_df.dropna(subset=required_cols)
        if len(chart_df) < 2:
            print("❌ Insufficient valid data after cleaning")
            return _create_fallback_line_chart(df, indicators, crypto_name)
        
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
        
        # Professional title
        enhanced_title = (f'{crypto_name.title()} - Professional Technical Analysis\n'
                        f'Data Range: {actual_start_date} to {actual_end_date} | '
                        f'${current_price:.2f} ({price_change:+.2f}%) | '
                        f'RSI: {rsi_value:.1f} | MACD: {macd_status} | {len(chart_df)} Candles')
        
        # Create professional style
        try:
            custom_style = mpf.make_mpf_style(
                base_mpf_style='binance',
                marketcolors=mpf.make_marketcolors(
                    up='#26A69A',
                    down='#EF5350',
                    edge='inherit',
                    wick={'up': '#26A69A', 'down': '#EF5350'},
                    volume={'up': '#26A69A40', 'down': '#EF535040'}
                ),
                facecolor='#131722',
                edgecolor='#2A2E39',
                gridcolor='#363A45',
                gridstyle='-',
                y_on_right=True
            )
            print(f"✅ Created professional chart style")
        except Exception as e:
            print(f"⚠️ Error creating custom style: {e}")
            custom_style = 'binance'
        
        # Build additional plots with consistent data
        apd = []
        panel_count = 1
        min_data_points = len(chart_df)
        colors = Config.TA_INDICATORS.get("professional_colors", {})
        indicator_count = 0
        
        # Add moving averages using chart data for consistency
        ma_periods = [20, 50]
        for period in ma_periods:
            if min_data_points >= period:
                try:
                    ma_line = ta.trend.SMAIndicator(chart_df['close'], window=period).sma_indicator()
                    if not ma_line.empty and not ma_line.isna().all():
                        color = colors.get(f'sma_{period}', '#FFD700')
                        width = 3 if period <= 26 else 2
                        apd.append(mpf.make_addplot(ma_line, color=color, width=width, alpha=0.9))
                        indicator_count += 1
                        print(f"✅ Added SMA {period} to technical chart")
                except Exception as e:
                    print(f"❌ Failed to calculate SMA {period}: {e}")
        
        # Add volume panel if available
        if 'volume' in chart_df.columns and (chart_df['volume'] > 0).any():
            try:
                panel_count += 1
                bull_vol_color = colors.get('volume_bullish', '#10B981')
                bear_vol_color = colors.get('volume_bearish', '#EF4444')
                volume_colors = [bull_vol_color if row['close'] >= row['open'] else bear_vol_color 
                               for _, row in chart_df.iterrows()]
                apd.append(mpf.make_addplot(chart_df['volume'], panel=panel_count-1, type='bar', 
                                         color=volume_colors, alpha=0.7, ylabel='Volume'))
                indicator_count += 1
                print(f"✅ Added Volume panel to technical chart")
            except Exception as e:
                print(f"❌ Failed to add volume panel: {e}")
        
        # Set figure size based on panels
        figsize = (20, 12) if panel_count == 1 else (20, 14)
        panel_ratios = None if panel_count == 1 else (4, 1)
        
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
            'tight_layout': True,
            'show_nontrading': False,
            'scale_padding': {'left': 0.3, 'top': 0.8, 'right': 1.0, 'bottom': 0.3}
        }
        
        if apd:
            plot_kwargs['addplot'] = apd
        if panel_ratios:
            plot_kwargs['panel_ratios'] = panel_ratios
        
        print(f"📊 Creating technical chart with {len(chart_df)} candles")
        
        try:
            fig, axes = mpf.plot(**plot_kwargs)
            print(f"✅ Technical chart created successfully")
        except Exception as e:
            print(f"❌ mplfinance plot creation failed: {e}")
            return _create_fallback_line_chart(df, indicators, crypto_name)
        
        # Apply professional styling
        try:
            fig.patch.set_facecolor('#131722')
            fig.suptitle(enhanced_title, fontsize=16, fontweight='bold', color='#D1D4DC', y=0.98)
            
            for ax in fig.get_axes():
                ax.set_facecolor('#131722')
                ax.tick_params(colors='#D1D4DC', which='both')
                ax.xaxis.label.set_color('#D1D4DC')
                ax.yaxis.label.set_color('#D1D4DC')
                ax.grid(True, color='#363A45', linestyle='-', linewidth=0.5, alpha=0.3)
                
                if ax.get_legend():
                    ax.get_legend().set_facecolor('#131722')
                    ax.get_legend().set_edgecolor('#363A45')
                    for text in ax.get_legend().get_texts():
                        text.set_color('#D1D4DC')
            
            print(f"✅ Applied professional styling")
        except Exception as e:
            print(f"⚠️ Warning: Could not apply all styling elements: {e}")
        
        # Save chart
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', prefix=f'chart_{crypto_name}_')
            fig.savefig(temp_file.name, format='png', dpi=300, bbox_inches='tight', 
                       facecolor='#131722', edgecolor='none', pad_inches=0.3)
            plt.close(fig)
            
            _current_chart_path = temp_file.name
            
            with open(temp_file.name, 'rb') as f:
                image_data = f.read()
                _current_chart_data = base64.b64encode(image_data).decode()
            
            print(f"✅ Technical chart created with data consistency")
            print(f"📁 Chart saved to: {temp_file.name}")
            return True
        except Exception as e:
            print(f"❌ Failed to save chart: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Unexpected error creating technical chart: {e}")
        import traceback
        traceback.print_exc()
        return _create_fallback_line_chart(df, indicators, crypto_name)


def _create_fallback_line_chart(df: pd.DataFrame, indicators: Dict[str, Any], crypto_name: str) -> bool:
    """Create a professional fallback chart if advanced charting fails with consistent data processing."""
    global _current_chart_data, _current_chart_path
    
    print(f"🔄 Creating fallback line chart for {crypto_name}")
    print(f"📋 Fallback chart will use matplotlib with simple line plots instead of candlesticks")
    
    try:
        # Ensure consistent data preprocessing for fallback chart
        if 'datetime' not in df.columns:
            if 'timestamp' in df.columns:
                df = _standardize_timestamps(df)
            else:
                print(f"⚠️ No datetime available - creating sequential timestamps for fallback")
                df['datetime'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
        
        # Sort data chronologically
        df = df.sort_values('datetime').reset_index(drop=True)
        print(f"✅ Fallback chart using {len(df)} consistently processed data points")
        
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        print(f"✅ Created matplotlib figure with 2 subplots")
        
        # Set dark theme colors
        fig.patch.set_facecolor('#131722')
        for ax in [ax1, ax2]:
            ax.set_facecolor('#131722')
            ax.tick_params(colors='#D1D4DC', which='both')
            ax.xaxis.label.set_color('#D1D4DC')
            ax.yaxis.label.set_color('#D1D4DC')
            ax.grid(True, color='#363A45', linestyle='-', linewidth=0.5, alpha=0.3)
        
        current_price = df['close'].iloc[-1]
        price_change = ((current_price - df['close'].iloc[0]) / df['close'].iloc[0] * 100) if len(df) > 1 else 0
        rsi_value = indicators.get('rsi', 0)
        
        # Get professional colors from config
        colors = Config.TA_INDICATORS.get("professional_colors", {})
        
        # Main price chart using consistent datetime
        ax1.plot(df['datetime'], df['close'], color='#00D4AA', linewidth=3, label='Close Price')
        print(f"✅ Added main price line to fallback chart using consistent datetime")
        
        # Add moving averages if available using consistent data
        ma_count = 0
        if len(df) >= 20:
            try:
                sma_20 = df['close'].rolling(window=20).mean()
                ax1.plot(df['datetime'], sma_20, color=colors.get('sma_20', '#FFD700'), 
                        linewidth=2, alpha=0.8, label='SMA 20')
                ma_count += 1
                print(f"✅ Added SMA 20 to fallback chart using consistent data")
            except Exception as e:
                print(f"⚠️ Failed to add SMA 20: {e}")
        
        if len(df) >= 50:
            try:
                sma_50 = df['close'].rolling(window=50).mean()
                ax1.plot(df['datetime'], sma_50, color=colors.get('sma_50', '#DA70D6'), 
                        linewidth=2, alpha=0.8, label='SMA 50')
                ma_count += 1
                print(f"✅ Added SMA 50 to fallback chart using consistent data")
            except Exception as e:
                print(f"⚠️ Failed to add SMA 50: {e}")
        
        ax1.set_title(f'{crypto_name.title()} - Professional Technical Analysis (Fallback Line Chart)\n'
                     f'Current: ${current_price:.2f} ({price_change:+.2f}%) | RSI: {rsi_value:.1f} | Data Points: {len(df)}', 
                     fontsize=14, color='#D1D4DC', fontweight='bold')
        ax1.set_ylabel('Price ($)', color='#D1D4DC')
        
        # Create legend with professional style
        legend1 = ax1.legend(facecolor='#131722', edgecolor='#363A45', framealpha=0.9)
        for text in legend1.get_texts():
            text.set_color('#D1D4DC')
        
        print(f"✅ Added {ma_count + 1} indicators to main price chart")
        
        # RSI subplot if available using consistent data
        rsi_added = False
        if 'rsi' in indicators and len(df) >= 14:
            try:
                # Calculate RSI for visualization using consistent data
                rsi_series = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
                
                if not rsi_series.empty and not rsi_series.isna().all():
                    ax2.plot(df['datetime'][-len(rsi_series):], rsi_series, 
                            color=colors.get('rsi_line', '#F59E0B'), linewidth=3, label='RSI')
                    ax2.axhline(y=70, color=colors.get('rsi_overbought', '#DC2626'), 
                               linestyle='--', alpha=0.8, linewidth=1)
                    ax2.axhline(y=30, color=colors.get('rsi_oversold', '#059669'), 
                               linestyle='--', alpha=0.8, linewidth=1)
                    ax2.axhline(y=50, color='#6B7280', linestyle='-', alpha=0.5, linewidth=1)
                    ax2.set_ylabel('RSI (14)', color='#D1D4DC')
                    ax2.set_ylim(0, 100)
                    
                    legend2 = ax2.legend(facecolor='#131722', edgecolor='#363A45', framealpha=0.9)
                    for text in legend2.get_texts():
                        text.set_color('#D1D4DC')
                    
                    rsi_added = True
                    print(f"✅ Added RSI subplot to fallback chart using consistent data")
                else:
                    print(f"⚠️ RSI calculation resulted in empty series")
            except Exception as e:
                print(f"⚠️ Failed to add RSI to fallback chart: {e}")
        
        if not rsi_added:
            # Remove the second subplot if no RSI
            print(f"🔄 Removing RSI subplot - creating single panel fallback chart")
            ax2.remove()
            fig, ax1 = plt.subplots(1, 1, figsize=(16, 10))
            fig.patch.set_facecolor('#131722')
            ax1.set_facecolor('#131722')
            ax1.tick_params(colors='#D1D4DC', which='both')
            ax1.xaxis.label.set_color('#D1D4DC')
            ax1.yaxis.label.set_color('#D1D4DC')
            ax1.grid(True, color='#363A45', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Replot on single axis with consistent data
            ax1.plot(df['datetime'], df['close'], color='#00D4AA', linewidth=3, label='Close Price')
            ma_count = 0
            if len(df) >= 20:
                try:
                    sma_20 = df['close'].rolling(window=20).mean()
                    ax1.plot(df['datetime'], sma_20, color=colors.get('sma_20', '#FFD700'), 
                            linewidth=2, alpha=0.8, label='SMA 20')
                    ma_count += 1
                except:
                    pass
            
            ax1.set_title(f'{crypto_name.title()} - Professional Technical Analysis (Fallback Line Chart)\n'
                         f'Current: ${current_price:.2f} ({price_change:+.2f}%) | Data Points: {len(df)}', 
                         fontsize=14, color='#D1D4DC', fontweight='bold')
            ax1.set_ylabel('Price ($)', color='#D1D4DC')
            
            legend1 = ax1.legend(facecolor='#131722', edgecolor='#363A45', framealpha=0.9)
            for text in legend1.get_texts():
                text.set_color('#D1D4DC')
            
            print(f"✅ Created single-panel fallback chart with {ma_count + 1} indicators")
        
        plt.tight_layout()
        
        # Save to temporary file
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', prefix=f'fallback_chart_{crypto_name}_')
            fig.savefig(temp_file.name, format='png', dpi=300, bbox_inches='tight', 
                       facecolor='#131722', edgecolor='none', pad_inches=0.3)
            plt.close(fig)
            
            _current_chart_path = temp_file.name
            
            # Create base64 for backwards compatibility
            with open(temp_file.name, 'rb') as f:
                image_data = f.read()
                _current_chart_data = base64.b64encode(image_data).decode()
            
            print(f"✅ Fallback line chart created with consistent data processing")
            print(f"📁 Fallback chart saved to: {temp_file.name}")
            return True
        except Exception as e:
            print(f"❌ Failed to save fallback chart: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Error creating fallback chart: {e}")
        print(f"📋 Fallback chart creation completely failed - no chart will be available")
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
                    patterns.append(f"Dragonfly Doji (bullish reversal signal for {horizon_context} trends)")
                elif row['lower_shadow_ratio'] > 0.4 and row['upper_shadow_ratio'] < 0.2:
                    patterns.append(f"Gravestone Doji (bearish reversal signal for {horizon_context} trends)")
                elif row['upper_shadow_ratio'] > 0.3 and row['lower_shadow_ratio'] > 0.3:
                    patterns.append(f"Long-legged Doji (market indecision for {horizon_context} outlook)")
                else:
                    patterns.append(f"Standard Doji (trend uncertainty for {horizon_context} forecast)")
            
            # Hammer patterns
            elif (row['lower_shadow_ratio'] > 0.5 and row['upper_shadow_ratio'] < 0.1 and 
                  row['close'] > row['open']):
                patterns.append(f"Hammer (strong bullish reversal for {horizon_context} forecast)")
            
            # Shooting star patterns
            elif (row['upper_shadow_ratio'] > 0.5 and row['lower_shadow_ratio'] < 0.1 and 
                  row['close'] < row['open']):
                patterns.append(f"Shooting Star (bearish reversal warning for {horizon_context} forecast)")
            
            # Large body patterns
            elif row['body_ratio'] > 0.7:
                if row['close'] > row['open']:
                    patterns.append(f"Strong Bullish Candle (momentum continuation for {horizon_context})")
                else:
                    patterns.append(f"Strong Bearish Candle (downward pressure for {horizon_context})")
        
        # Multi-candle patterns (if we have enough data)
        if len(recent_df) >= 3:
            # Engulfing patterns
            for i in range(1, len(recent_df)):
                prev_row = recent_df.iloc[i-1]
                curr_row = recent_df.iloc[i]
                
                # Bullish engulfing
                if (prev_row['close'] < prev_row['open'] and curr_row['close'] > curr_row['open'] and
                    curr_row['open'] < prev_row['close'] and curr_row['close'] > prev_row['open']):
                    patterns.append(f"Bullish Engulfing (strong buy signal for {horizon_context} outlook)")
                
                # Bearish engulfing
                elif (prev_row['close'] > prev_row['open'] and curr_row['close'] < curr_row['open'] and
                      curr_row['open'] > prev_row['close'] and curr_row['close'] < prev_row['open']):
                    patterns.append(f"Bearish Engulfing (strong sell signal for {horizon_context} outlook)")
        
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
                    patterns.append(f"Morning Star (powerful bullish reversal for {horizon_context} forecast)")
                
                # Evening star pattern
                elif (first['close'] > first['open'] and  # First candle bullish
                      second['body_ratio'] < 0.3 and  # Second candle small body (star)
                      third['close'] < third['open'] and  # Third candle bearish
                      third['close'] < (first['open'] + first['close']) / 2):  # Third closes below first's midpoint
                    patterns.append(f"Evening Star (strong bearish reversal for {horizon_context} forecast)")
        
        # Add pattern significance based on forecast horizon
        if patterns:
            if "hour" in forecast_horizon.lower():
                patterns.append(f"Short-term patterns detected - High relevance for {forecast_horizon} trading")
            elif "day" in forecast_horizon.lower():
                patterns.append(f"Medium-term patterns - Moderate impact on {forecast_horizon} outlook")
            else:
                patterns.append(f"Long-term patterns - Trend context for {forecast_horizon} forecast")
        
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
        return [f"Pattern analysis error for {forecast_horizon} forecast"]


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
                                   crypto_name: str, forecast_horizon: str, data_metadata: dict) -> bool:
    """Create an enhanced technical analysis chart with larger text, pattern annotations, and horizon optimization."""
    global _current_chart_data, _current_chart_path
    
    print(f"🚀 Creating enhanced technical chart for {crypto_name} with {forecast_horizon} optimization")
    print(f"📊 Input data: {len(df)} rows, metadata: {data_metadata}")
    
    try:
        # Validate inputs with metadata
        if df.empty or len(df) < 2:
            print(f"❌ Enhanced chart creation failed: Insufficient data ({len(df)} rows)")
            print(f"📋 Fallback reason: Enhanced charts require at least 2 data points")
            return _create_technical_chart(df, indicators, crypto_name)
        
        # Import mplfinance for candlestick charts
        try:
            import mplfinance as mpf
            print(f"✅ mplfinance library available for enhanced chart")
        except ImportError:
            print("❌ mplfinance not available for enhanced chart - falling back to standard chart")
            print(f"📋 Fallback reason: Enhanced charts require mplfinance library")
            return _create_technical_chart(df, indicators, crypto_name)
        
        # Use the preprocessed data directly (it already has standardized datetime)
        if 'datetime' not in df.columns:
            print(f"❌ Enhanced chart missing datetime column after preprocessing")
            return _create_technical_chart(df, indicators, crypto_name)
        
        # Create chart dataframe using the preprocessed datetime index
        chart_df = df.set_index('datetime').copy()
        print(f"✅ Enhanced chart using preprocessed data with {len(chart_df)} rows")
        
        # Validate OHLC data integrity (should already be clean from preprocessing)
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in chart_df.columns]
        if missing_cols:
            print(f"❌ Enhanced chart missing required OHLC columns: {missing_cols}")
            print(f"📋 Fallback reason: Enhanced candlestick charts require complete OHLC data")
            return _create_technical_chart(df, indicators, crypto_name)
        
        # Final validation since data is already preprocessed
        final_nan_count = chart_df[required_cols].isnull().sum().sum()
        if final_nan_count > 0:
            print(f"⚠️ Enhanced chart found {final_nan_count} remaining NaN values after preprocessing")
            chart_df = chart_df.dropna(subset=required_cols)
            
        if len(chart_df) < 2:
            print("❌ Enhanced chart insufficient data after final validation")
            return _create_technical_chart(df, indicators, crypto_name)
        
        # Debug: Validate final chart data consistency
        price_min = chart_df['low'].min()
        price_max = chart_df['high'].max()
        price_variance = chart_df[required_cols].var().sum()
        
        print(f"✅ Enhanced chart final validation:")
        print(f"   - Data points: {len(chart_df)}")
        print(f"   - Date range: {chart_df.index.min()} to {chart_df.index.max()}")
        print(f"   - Price range: ${price_min:.2f} - ${price_max:.2f}")
        print(f"   - Price variance: {price_variance:.2e}")
        print(f"   - Resolution: {data_metadata.get('resolution', 'auto')}")
        
        # Calculate current price and price change
        current_price = chart_df['close'].iloc[-1]
        price_change = ((current_price - chart_df['close'].iloc[0]) / chart_df['close'].iloc[0] * 100) if len(chart_df) > 1 else 0
        
        # Get date range for title
        actual_start_date = chart_df.index.min().strftime('%Y-%m-%d %H:%M')
        actual_end_date = chart_df.index.max().strftime('%Y-%m-%d %H:%M')
        
        # Calculate key indicators for title
        rsi_value = indicators.get('rsi', 0)
        rsi_status = "Bullish" if rsi_value < 40 else "Bearish" if rsi_value > 60 else "Neutral"
        
        macd_val = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_status = "Bullish" if macd_val > macd_signal else "Bearish"
        
        # Enhanced professional title with more detail
        enhanced_title = (f'{crypto_name.title()} - Enhanced Technical Analysis ({forecast_horizon} Forecast)\n'
                        f'Data: {actual_start_date} to {actual_end_date} | '
                        f'${current_price:.2f} ({price_change:+.2f}%) | '
                        f'RSI: {rsi_value:.1f} ({rsi_status}) | MACD: {macd_status} | {len(chart_df)} Candles')
        
        # Create professional TradingView-style configuration
        try:
            custom_style = mpf.make_mpf_style(
                base_mpf_style='binance',
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
            print(f"✅ Enhanced chart created professional TradingView-style configuration")
        except Exception as e:
            print(f"⚠️ Enhanced chart error creating custom style: {e}")
            custom_style = 'binance'
        
        # Build additional plots with consistent data alignment
        apd = []
        panel_count = 1
        min_data_points = len(chart_df)
        
        # Get professional colors from config
        colors = Config.TA_INDICATORS.get("professional_colors", {})
        indicator_count = 0
        
        # Create indicators aligned with chart data using the same source data
        print(f"📊 Aligning indicators with chart data...")
        
        # Optimize indicators based on forecast horizon
        try:
            ma_periods = _get_horizon_optimized_indicators(forecast_horizon, min_data_points)
            print(f"✅ Enhanced chart optimized indicators for {forecast_horizon}: {ma_periods}")
        except Exception as e:
            print(f"⚠️ Enhanced chart error optimizing indicators: {e}")
            ma_periods = {'sma_20': 20, 'sma_50': 50} if min_data_points >= 50 else {'sma_20': 20} if min_data_points >= 20 else {}
        
        # IMPORTANT: Calculate indicators using chart_df for perfect alignment
        chart_aligned_indicators = {}
        
        # Add moving averages with perfect alignment to chart data
        for ma_type, period in ma_periods.items():
            if min_data_points >= period:
                try:
                    if ma_type.startswith('ema'):
                        # Calculate EMA directly on chart data for alignment
                        ma_line = ta.trend.EMAIndicator(chart_df['close'], window=period).ema_indicator()
                        color = colors.get(f'ema_{period}', '#00D4AA')
                        ma_name = f'EMA {period}'
                    else:
                        # Calculate SMA directly on chart data for alignment
                        ma_line = ta.trend.SMAIndicator(chart_df['close'], window=period).sma_indicator()
                        color = colors.get(f'sma_{period}', '#FFD700')
                        ma_name = f'SMA {period}'
                    
                    if not ma_line.empty and not ma_line.isna().all():
                        width = 3 if period <= 20 else 2
                        apd.append(mpf.make_addplot(ma_line, color=color, width=width, alpha=0.9))
                        indicator_count += 1
                        chart_aligned_indicators[ma_type] = ma_line.iloc[-1] if not ma_line.empty else 0
                        print(f"✅ Enhanced chart added {ma_name} aligned with chart data")
                    else:
                        print(f"⚠️ Enhanced chart {ma_name} calculation resulted in empty data")
                except Exception as e:
                    print(f"❌ Enhanced chart failed to calculate {ma_type} {period}: {e}")
        
        # Add Bollinger Bands aligned with chart data
        if min_data_points >= 20:
            try:
                bb_indicator = ta.volatility.BollingerBands(chart_df['close'], window=20, window_dev=2)
                bb_upper = bb_indicator.bollinger_hband()
                bb_lower = bb_indicator.bollinger_lband()
                if not bb_upper.empty and not bb_lower.empty:
                    bb_color = colors.get('bollinger_bands', '#87CEEB')
                    apd.extend([
                        mpf.make_addplot(bb_upper, color=bb_color, width=2, alpha=0.6, linestyle='--'),
                        mpf.make_addplot(bb_lower, color=bb_color, width=2, alpha=0.6, linestyle='--')
                    ])
                    indicator_count += 2
                    print(f"✅ Enhanced chart added Bollinger Bands aligned with chart data")
                else:
                    print(f"⚠️ Enhanced chart Bollinger Bands calculation resulted in empty data")
            except Exception as e:
                print(f"❌ Enhanced chart failed to calculate Bollinger Bands: {e}")
        
        # Add RSI panel with chart-aligned data
        if min_data_points >= 14:
            try:
                rsi = ta.momentum.RSIIndicator(chart_df['close'], window=14).rsi()
                if not rsi.empty and not rsi.isna().all():
                    panel_count += 1
                    apd.append(mpf.make_addplot(rsi, panel=1, color=colors.get('rsi_line', '#F59E0B'), 
                                             width=3, ylabel='RSI (14)'))
                    
                    # RSI levels
                    rsi_levels = [(80, '#DC2626', 0.7), (70, '#DC2626', 0.9), 
                                  (50, '#6B7280', 0.6), (30, '#059669', 0.9), (20, '#059669', 0.7)]
                    
                    for level, color, alpha in rsi_levels:
                        rsi_level = pd.Series([level]*len(rsi), index=rsi.index)
                        apd.append(mpf.make_addplot(rsi_level, panel=1, color=color, 
                                                 width=1.5, linestyle='--', alpha=alpha))
                    
                    indicator_count += 6
                    print(f"✅ Enhanced chart added RSI panel aligned with chart data")
                else:
                    print(f"⚠️ Enhanced chart RSI calculation resulted in empty data")
            except Exception as e:
                print(f"❌ Enhanced chart failed to calculate RSI: {e}")
        
        # Add MACD panel with chart-aligned data
        if min_data_points >= 26:
            try:
                macd_indicator = ta.trend.MACD(chart_df['close'], window_slow=26, window_fast=12, window_sign=9)
                macd_line = macd_indicator.macd()
                macd_signal_line = macd_indicator.macd_signal()
                macd_histogram = macd_indicator.macd_diff()
                
                if not macd_line.empty and not macd_signal_line.empty and not macd_histogram.empty:
                    panel_count += 1
                    
                    # MACD lines
                    apd.extend([
                        mpf.make_addplot(macd_line, panel=2, color=colors.get('macd_line', '#00D4AA'), 
                                       width=3, ylabel='MACD (12,26,9)'),
                        mpf.make_addplot(macd_signal_line, panel=2, color=colors.get('macd_signal', '#FF6B6B'), 
                                       width=3)
                    ])
                    
                    # MACD histogram
                    pos_color = colors.get('macd_histogram_positive', '#10B981')
                    neg_color = colors.get('macd_histogram_negative', '#EF4444')
                    macd_hist_colors = [pos_color if val > 0 else neg_color for val in macd_histogram]
                    apd.append(mpf.make_addplot(macd_histogram, panel=2, type='bar', 
                                             color=macd_hist_colors, alpha=0.7, width=1.0))
                    
                    # Zero line
                    macd_zero = pd.Series([0]*len(macd_line), index=macd_line.index)
                    apd.append(mpf.make_addplot(macd_zero, panel=2, color='#6B7280', width=1.5, 
                                             linestyle='-', alpha=0.6))
                    
                    indicator_count += 4
                    print(f"✅ Enhanced chart added MACD panel aligned with chart data")
                else:
                    print(f"⚠️ Enhanced chart MACD calculation resulted in empty data")
            except Exception as e:
                print(f"❌ Enhanced chart failed to calculate MACD: {e}")
        
        # Add Volume panel with chart-aligned data
        if 'volume' in chart_df.columns and (chart_df['volume'] > 0).any():
            try:
                panel_count += 1
                
                # Enhanced volume bars
                bull_vol_color = colors.get('volume_bullish', '#10B981')
                bear_vol_color = colors.get('volume_bearish', '#EF4444')
                volume_colors = [bull_vol_color if row['close'] >= row['open'] else bear_vol_color 
                               for _, row in chart_df.iterrows()]
                apd.append(mpf.make_addplot(chart_df['volume'], panel=panel_count-1, type='bar', 
                                         color=volume_colors, alpha=0.8, ylabel='Volume & SMA(20)'))
                
                # Volume SMA overlay
                if min_data_points >= 20:
                    volume_sma = ta.trend.SMAIndicator(chart_df['volume'], window=20).sma_indicator()
                    if not volume_sma.empty and not volume_sma.isna().all():
                        apd.append(mpf.make_addplot(volume_sma, panel=panel_count-1, 
                                                 color=colors.get('volume_sma', '#FFD700'), 
                                                 width=3, alpha=0.9))
                        indicator_count += 2
                        print(f"✅ Enhanced chart added Volume panel with SMA(20) aligned with chart data")
                    else:
                        indicator_count += 1
                        print(f"✅ Enhanced chart added Volume panel (SMA calculation failed)")
                else:
                    indicator_count += 1
                    print(f"✅ Enhanced chart added Volume panel (insufficient data for SMA)")
            except Exception as e:
                print(f"❌ Enhanced chart failed to add volume panel: {e}")
        
        print(f"✅ Enhanced chart total aligned indicators: {indicator_count}")
        print(f"📊 Chart-aligned indicator values: {chart_aligned_indicators}")
        
        # Set panel ratios for enhanced layout
        if panel_count == 1:
            panel_ratios = None
            figsize = (24, 16)
        elif panel_count == 2:
            panel_ratios = (4, 1)
            figsize = (24, 18)
        elif panel_count == 3:
            panel_ratios = (4, 1, 1)
            figsize = (24, 20)
        else:
            panel_ratios = (4, 1, 1, 1.2)
            figsize = (24, 22)
        
        print(f"✅ Enhanced chart layout: {panel_count} panels, figure size: {figsize}")
        
        # Create the enhanced plot with the aligned data
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
            'show_nontrading': False,
            'scale_padding': {'left': 0.3, 'top': 0.8, 'right': 1.0, 'bottom': 0.3}
        }
        
        if apd:
            plot_kwargs['addplot'] = apd
            print(f"✅ Enhanced chart prepared {len(apd)} additional plot elements")
        if panel_ratios:
            plot_kwargs['panel_ratios'] = panel_ratios
        
        print(f"📊 Creating enhanced mplfinance plot with {len(chart_df)} aligned candles")
        
        try:
            fig, axes = mpf.plot(**plot_kwargs)
            print(f"✅ Enhanced mplfinance chart created successfully with data alignment")
        except Exception as e:
            print(f"❌ Enhanced mplfinance plot creation failed: {e}")
            print(f"📋 Fallback reason: Enhanced chart mplfinance internal error")
            return _create_technical_chart(df, indicators, crypto_name)
        
        # Enhanced professional styling
        try:
            fig.patch.set_facecolor('#131722')
            fig.suptitle(enhanced_title, fontsize=20, fontweight='bold', color='#D1D4DC', y=0.98)
            
            # Configure enhanced styling
            for ax in fig.get_axes():
                ax.set_facecolor('#131722')
                ax.tick_params(colors='#D1D4DC', which='both', labelsize=14)
                ax.xaxis.label.set_color('#D1D4DC')
                ax.yaxis.label.set_color('#D1D4DC')
                ax.xaxis.label.set_fontsize(16)
                ax.yaxis.label.set_fontsize(16)
                ax.grid(True, color='#363A45', linestyle='-', linewidth=0.7, alpha=0.4)
                
                if ax.get_legend():
                    ax.get_legend().set_facecolor('#131722')
                    ax.get_legend().set_edgecolor('#363A45')
                    for text in ax.get_legend().get_texts():
                        text.set_color('#D1D4DC')
                        text.set_fontsize(12)
            
            print(f"✅ Enhanced chart applied professional styling")
        except Exception as e:
            print(f"⚠️ Enhanced chart warning: Could not apply all styling: {e}")
        
        # Add enhanced pattern annotations with alignment info
        try:
            _add_enhanced_pattern_annotations(fig, chart_df, patterns, forecast_horizon, data_metadata)
            print(f"✅ Enhanced chart added pattern annotations with data alignment info")
        except Exception as e:
            print(f"⚠️ Enhanced chart warning: Could not add pattern annotations: {e}")
        
        # Add custom legend with alignment validation
        try:
            _add_enhanced_legend(fig, ma_periods, colors, min_data_points)
            print(f"✅ Enhanced chart added custom legend")
        except Exception as e:
            print(f"⚠️ Enhanced chart warning: Could not create custom legend: {e}")
        
        # Save enhanced chart
        try:
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
            
            print(f"✅ Enhanced technical chart created with perfect data alignment")
            print(f"📁 Enhanced chart saved to: {temp_file.name}")
            print(f"🎯 Chart features: {forecast_horizon} optimization, {len(chart_df)} aligned candles, {indicator_count} indicators")
            return True
        except Exception as e:
            print(f"❌ Enhanced chart failed to save: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Unexpected error creating enhanced chart: {e}")
        import traceback
        traceback.print_exc()
        return _create_technical_chart(df, indicators, crypto_name)


def _get_horizon_optimized_indicators(forecast_horizon: str, data_points: int) -> Dict[str, int]:
    """Get optimized moving average periods based on forecast horizon."""
    horizon_lower = forecast_horizon.lower()
    ma_periods = {}
    
    print(f"🎯 Optimizing indicators for {forecast_horizon} with {data_points} data points")
    
    if "hour" in horizon_lower:
        # Short-term focus for hour-based forecasts
        print(f"📊 Short-term optimization: Focusing on faster-moving indicators")
        if data_points >= 9:
            ma_periods['ema_9'] = 9
        if data_points >= 21:
            ma_periods['ema_21'] = 21
        if data_points >= 50:
            ma_periods['sma_50'] = 50
    elif "day" in horizon_lower and ("1 day" in horizon_lower or "3 day" in horizon_lower):
        # Medium-term focus for 1-3 day forecasts
        print(f"📊 Medium-term optimization: Balancing fast and slow indicators")
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
        print(f"📊 Long-term optimization: Emphasizing trend-following indicators")
        if data_points >= 20:
            ma_periods['sma_20'] = 20
        if data_points >= 50:
            ma_periods['sma_50'] = 50
        if data_points >= 100:
            ma_periods['sma_100'] = 100
        if data_points >= 200:
            ma_periods['sma_200'] = 200
    
    selected_indicators = list(ma_periods.keys())
    print(f"✅ Selected indicators for {forecast_horizon}: {selected_indicators}")
    
    if not ma_periods:
        print(f"⚠️ No indicators selected for {forecast_horizon} - insufficient data points ({data_points})")
    
    return ma_periods


def _add_enhanced_pattern_annotations(fig, df: pd.DataFrame, patterns: List[str], forecast_horizon: str, data_metadata: dict):
    """Add enhanced pattern annotations to the chart with data alignment context."""
    if not patterns or fig.get_axes() is None:
        print(f"⚠️ No patterns to annotate or no chart axes available")
        return
    
    try:
        main_ax = fig.get_axes()[0]  # Price chart axis
        
        # Add pattern annotations text box with data metadata
        if patterns:
            pattern_text = f"Chart Patterns ({forecast_horizon} Forecast):\n"
            pattern_count = 0
            for i, pattern in enumerate(patterns):  
                pattern_text += f"- {pattern}\n"
                pattern_count += 1
            
            # Add data alignment info
            pattern_text += f"\nData Quality:\n"
            pattern_text += f"- Resolution: {data_metadata.get('resolution', 'auto')}\n"
            pattern_text += f"- Source: {data_metadata.get('data_source', 'unknown')}\n"
            pattern_text += f"- Points: {len(df)} aligned candles"
            
            # Add text box with pattern information
            main_ax.text(0.02, 0.70, pattern_text, transform=main_ax.transAxes,
                        fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                        facecolor='#2A2E39', edgecolor='#363A45', alpha=0.9),
                        color='#D1D4DC', fontweight='bold')
            
            print(f"✅ Added {pattern_count} pattern annotations with data alignment info")
        
        # Add forecast horizon indicator with consistency info
        consistency_status = "Aligned" if data_metadata.get('data_source') != 'unknown' else "Basic"
        horizon_text = f"Optimized for {forecast_horizon} forecast\nData Consistency: {consistency_status}\nChart-Indicator Alignment: Perfect"
        main_ax.text(0.98, 0.95, horizon_text, transform=main_ax.transAxes,
                    fontsize=11, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='#2A2E39', 
                             edgecolor='#363A45', alpha=0.9),
                    color='#D1D4DC')
        
        print(f"✅ Added forecast horizon indicator with consistency validation")
        
    except Exception as e:
        print(f"⚠️ Error adding enhanced pattern annotations: {e}")


def _add_enhanced_legend(fig, ma_periods: Dict[str, int], colors: Dict[str, str], min_data_points: int):
    """Add enhanced legend with larger text for moving averages."""
    if not ma_periods or fig.get_axes() is None:
        print(f"⚠️ No moving averages to add to legend or no chart axes available")
        return
    
    try:
        main_ax = fig.get_axes()[0]
        legend_elements = []
        legend_labels = []
        
        from matplotlib.lines import Line2D
        
        legend_count = 0
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
                legend_count += 1
        
        if legend_elements:
            legend = main_ax.legend(legend_elements, legend_labels, 
                                  loc='upper left', framealpha=0.9, 
                                  facecolor='#131722', edgecolor='#363A45')
            for text in legend.get_texts():
                text.set_color('#D1D4DC')
                text.set_fontsize(14)  # Larger legend text
            
            print(f"✅ Added enhanced legend with {legend_count} moving averages and larger text")
        
    except Exception as e:
        print(f"⚠️ Error creating enhanced legend: {e}")


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
    
    def _run(self, crypto_name: str, forecast_horizon: str = "24 hours") -> str:
        """Legacy interface for the tool with proper parameter handling."""
        return technical_analysis_tool.func(crypto_name, forecast_horizon)


def create_technical_analysis_tool():
    """Create and return a technical analysis tool instance."""
    return technical_analysis_tool 