"""
CoinGecko API tool for fetching cryptocurrency market data.
"""

import time
import requests
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, Optional
from crewai.tools import tool

from ..config import Config
from ..utils import APIRateLimiter, APICache


@tool("coingecko_tool")
def coingecko_tool(query: str, historical_date: str = "") -> str:
    """
    Fetches cryptocurrency market data from CoinGecko API.
    
    Args:
        query: Query for cryptocurrency data (e.g., 'bitcoin current price', 'ethereum ohlcv 30 days')
               Can include horizon information for optimal data fetching (e.g., 'bitcoin ohlcv 24 hours horizon')
        historical_date: Optional date in YYYY-MM-DD format for historical backtesting mode. Leave empty for current data.
    
    Returns:
        JSON string containing the requested cryptocurrency data
    """
    
    def _get_session():
        """Create a requests session with API key if available."""
        session = requests.Session()
        api_key = Config.COINGECKO_API_KEY or ""
        if api_key:
            session.headers.update({"x-cg-demo-api-key": api_key})
        return session
    
    def _extract_crypto_id(query: str) -> str:
        """Extract cryptocurrency ID from query."""
        query_lower = query.lower()
        # Map common names to CoinGecko IDs
        crypto_map = {
            "bitcoin": "bitcoin",
            "btc": "bitcoin",
            "ethereum": "ethereum", 
            "eth": "ethereum",
            "solana": "solana",
            "sol": "solana",
            "cardano": "cardano",
            "ada": "cardano",
            "polkadot": "polkadot",
            "dot": "polkadot",
            "chainlink": "chainlink",
            "link": "chainlink"
        }
        
        for name, coin_id in crypto_map.items():
            if name in query_lower:
                return coin_id
        
        # Default to bitcoin if no specific crypto found
        return "bitcoin"
    
    def _extract_days(query: str) -> int:
        """Extract number of days from query."""
        import re
        
        # Try to extract explicit day numbers from query
        day_match = re.search(r'(\d+)\s*days?', query.lower())
        if day_match:
            return int(day_match.group(1))
        
        # Look for common time periods
        if "30 days" in query.lower() or "month" in query.lower():
            return 30
        elif "7 days" in query.lower() or "week" in query.lower():
            return 7
        elif "365 days" in query.lower() or "year" in query.lower():
            return 365
        elif "horizon" in query.lower():
            # If horizon is mentioned, use optimal days for technical analysis
            return 90  # Default 3 months for good technical indicators
        else:
            return 90  # Default to 90 days for better technical analysis
    
    def _get_optimal_days_for_horizon(horizon: str) -> int:
        """Get optimal historical data days based on forecast horizon."""
        horizon_lower = horizon.lower()
        
        # Extract numeric value and time unit - increased data amounts for better TA
        if "hour" in horizon_lower:
            if "1 hour" in horizon_lower or "1hr" in horizon_lower:
                return 7  # 1 week for 1 hour forecast (more data for indicators)
            elif "4 hour" in horizon_lower or "4hr" in horizon_lower:
                return 14  # 2 weeks for 4 hour forecast
            elif "12 hour" in horizon_lower or "12hr" in horizon_lower:
                return 30  # 1 month for 12 hour forecast
            else:
                return 14  # Default 2 weeks for hour-based forecasts
        elif "day" in horizon_lower:
            if "1 day" in horizon_lower:
                return 60  # 2 months for 1 day forecast (need more data for hourly analysis)
            elif "3 day" in horizon_lower:
                return 90  # 3 months for 3 day forecast
            elif "7 day" in horizon_lower:
                return 120  # 4 months for 1 week forecast
            else:
                return 60  # Default 2 months for day-based forecasts
        elif "week" in horizon_lower:
            if "1 week" in horizon_lower:
                return 120  # 4 months for 1 week forecast
            elif "2 week" in horizon_lower:
                return 180  # 6 months for 2 week forecast
            else:
                return 120  # Default 4 months for week-based forecasts
        elif "month" in horizon_lower:
            return 365  # 1 year for month-based forecasts
        else:
            return 60  # Default to 2 months for better indicator data
    
    def _convert_timestamps(df, column_name='timestamp'):
        """Simplified and improved timestamp conversion."""
        if column_name not in df.columns:
            return df
            
        try:
            first_ts = df[column_name].iloc[0]
            
            if isinstance(first_ts, str):
                # ISO format string
                df['datetime'] = pd.to_datetime(df[column_name])
            elif pd.api.types.is_numeric_dtype(df[column_name]):
                # Check if it's milliseconds or seconds
                if first_ts > 1e10:
                    # Milliseconds
                    df['datetime'] = pd.to_datetime(df[column_name], unit='ms')
                else:
                    # Seconds
                    df['datetime'] = pd.to_datetime(df[column_name], unit='s')
            else:
                # Fallback: try to parse as string
                df['datetime'] = pd.to_datetime(df[column_name])
                
            # Log the conversion result
            
        except Exception as e:
            print(f"⚠️ ERROR: Timestamp conversion failed: {e}")
            # Fallback to current time
            df['datetime'] = pd.Timestamp.now()
        
        return df
    
    def _get_current_price(crypto_id: str, session: requests.Session) -> Dict[str, Any]:
        """Get current price and basic market data."""
        try:
            url = f"{Config.COINGECKO_BASE_URL}/simple/price"
            params = {
                "ids": crypto_id,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_24hr_vol": "true",
                "include_market_cap": "true"
            }
            
            response = session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if crypto_id in data:
                current_price = data[crypto_id].get("usd", 0)
                
                # Sanity check for Bitcoin specifically
                if crypto_id == "bitcoin" and (current_price < 20000 or current_price > 200000):
                    print(f"⚠️ WARNING: Bitcoin price ${current_price:,} seems unrealistic!")
                
                return {
                    "cryptocurrency": crypto_id,
                    "current_price": current_price,
                    "market_cap": data[crypto_id].get("usd_market_cap", 0),
                    "volume_24h": data[crypto_id].get("usd_24h_vol", 0),
                    "price_change_24h": data[crypto_id].get("usd_24h_change", 0),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"error": f"Cryptocurrency {crypto_id} not found"}
                
        except requests.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
    
    def _get_ohlcv_data(crypto_id: str, days: int, session: requests.Session) -> Dict[str, Any]:
        """Get historical OHLCV data with proper volume information."""
        try:
            # Always use market_chart endpoint for better volume data
            url = f"{Config.COINGECKO_BASE_URL}/coins/{crypto_id}/market_chart"
            params = {"vs_currency": "usd", "days": days}
            
            
            response = session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Market chart endpoint returns prices, market_caps, total_volumes
            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])
            market_caps = data.get("market_caps", [])
            
            if not prices:
                return {"error": "No price data available"}
            
            
            # Convert to OHLCV format with proper volume data
            df_data = []
            
            # Group prices by day to create proper OHLC data
            price_df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            
            # Convert timestamps to datetime using improved function
            price_df = _convert_timestamps(price_df, 'timestamp')
            volume_df = _convert_timestamps(volume_df, 'timestamp')
            
            # Merge price and volume data on datetime
            merged_df = pd.merge(price_df, volume_df, on='datetime', how='left', suffixes=('_price', '_volume'))
            merged_df['volume'] = merged_df['volume'].fillna(0)
            
            # Use timestamp from price data as the primary timestamp
            merged_df['timestamp'] = merged_df['timestamp_price']
            merged_df = merged_df.drop(['timestamp_price', 'timestamp_volume'], axis=1, errors='ignore')
            
            
            # Enhanced data grouping for better technical analysis - FIXED VERSION
            # The key insight: CoinGecko provides price points every ~5 minutes, but when we group
            # them into hourly periods with only 1-2 data points, we get flat OHLC.
            # Solution: Use wider time periods or handle single-point periods better.
            
            if days <= 1:
                # For 1 day or less, use 15-minute intervals
                merged_df['period'] = merged_df['datetime'].dt.floor('15min')
                resolution = "15min"
            elif days <= 7:
                # FIXED: For 1 week or less, use 2-hour intervals instead of 1-hour
                merged_df['period'] = merged_df['datetime'].dt.floor('2H')
                resolution = "2hourly"
            elif days <= 14:
                # FIXED: For 2 weeks or less, use 4-hour intervals
                merged_df['period'] = merged_df['datetime'].dt.floor('4H')
                resolution = "4hourly"
            elif days <= 30:
                # For 1 month or less, use 6-hour intervals
                merged_df['period'] = merged_df['datetime'].dt.floor('6H')
                resolution = "6hourly"
            elif days <= 60:
                # For 2 months or less, use 12-hour intervals
                merged_df['period'] = merged_df['datetime'].dt.floor('12H')
                resolution = "12hourly"
            elif days <= 90:
                # For 3 months or less, use daily intervals
                merged_df['period'] = merged_df['datetime'].dt.floor('D')
                resolution = "daily"
            else:
                # For longer periods, use daily intervals
                merged_df['period'] = merged_df['datetime'].dt.floor('D')
                resolution = "daily"
            
            # Group data by the determined period
            # IMPROVED: Better handling of single data points per period
            grouped = merged_df.groupby('period').agg({
                'price': ['first', 'max', 'min', 'last', 'std', 'count'],
                'volume': 'sum',
                'timestamp': 'first'
            })
            
            # Flatten column names
            grouped.columns = ['open', 'high', 'low', 'close', 'price_std', 'price_count', 'volume', 'timestamp']
            grouped = grouped.reset_index()
            
            # ABORT: Do not create artificial OHLC spreads - this creates misleading data
            # If periods have flat OHLC or insufficient volatility, use the actual data as-is
            # Technical analysis should work with real market data, not artificially enhanced data
            
            single_point_mask = grouped['price_count'] == 1
            if single_point_mask.sum() > 0:
                print(f"WARNING: Found {single_point_mask.sum()} periods with single data points (flat OHLC)")
                print("Using actual flat OHLC data - no artificial spreads will be created")
            
            # Check for periods with very low volatility but do not modify them
            low_volatility_count = 0
            for idx in grouped.index:
                high = grouped.loc[idx, 'high']
                low = grouped.loc[idx, 'low']
                close = grouped.loc[idx, 'close']
                
                current_spread = (high - low) / close if close > 0 else 0
                if current_spread < 0.0001:  # Less than 0.01%
                    low_volatility_count += 1
            
            if low_volatility_count > 0:
                print(f"WARNING: Found {low_volatility_count} periods with very low volatility")
                print("Using actual low-volatility OHLC data - no artificial enhancement will be applied")
            
            # Sort by period to ensure chronological order
            grouped = grouped.sort_values('period')
            
            if not grouped.empty:
                # Ensure we have enough data points for technical indicators
                # If we don't have enough, try to get more data
                min_required_points = 50  # Need at least 50 points for proper MACD (26 + 24 buffer)
                if len(grouped) < min_required_points and days < 90:
                    # Recursively try to get more data
                    return _get_ohlcv_data(crypto_id, min(days * 2, 90), session)
                
                # Convert to serializable format
                ohlcv_data = []
                for _, row in grouped.iterrows():
                    try:
                        # Use the period directly for timestamp (it's already a datetime)
                        timestamp_str = row['period'].isoformat()
                        
                        ohlcv_data.append({
                            "timestamp": timestamp_str,
                            "open": float(row["open"]) if pd.notna(row["open"]) else 0.0,
                            "high": float(row["high"]) if pd.notna(row["high"]) else 0.0,
                            "low": float(row["low"]) if pd.notna(row["low"]) else 0.0,
                            "close": float(row["close"]) if pd.notna(row["close"]) else 0.0,
                            "volume": float(row["volume"]) if pd.notna(row["volume"]) else 0.0
                        })
                    except Exception as e:
                        print(f"⚠️ ERROR: Error processing row: {e}")
                        continue
                
                # Sort by timestamp to ensure chronological order (oldest first, standard for financial charts)
                ohlcv_data.sort(key=lambda x: x['timestamp'])
                
                if ohlcv_data:
                    # Add some metadata about the data quality
                    total_volume = sum([item['volume'] for item in ohlcv_data])
                    has_volume_data = total_volume > 0
                    avg_volume = total_volume / len(ohlcv_data) if ohlcv_data else 0
                    
                    return {
                        "cryptocurrency": crypto_id,
                        "days": days,
                        "ohlcv_data": ohlcv_data,
                        "data_points": len(ohlcv_data),
                        "has_volume_data": has_volume_data,
                        "total_volume": total_volume,
                        "average_volume": avg_volume,
                        "data_source": "market_chart_api",
                        "resolution": resolution
                    }
            
            # If we reach here, there was no valid data
            return {"error": "No valid OHLCV data found"}
            
        except requests.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
        except Exception as e:
            print(f"⚠️ ERROR: Data processing error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Data processing error: {str(e)}"}
    
    def _get_market_stats(crypto_id: str, session: requests.Session) -> Dict[str, Any]:
        """Get comprehensive market statistics."""
        try:
            url = f"{Config.COINGECKO_BASE_URL}/coins/{crypto_id}"
            response = session.get(url)
            response.raise_for_status()
            
            data = response.json()
            market_data = data.get("market_data", {})
            
            return {
                "cryptocurrency": crypto_id,
                "name": data.get("name", ""),
                "symbol": data.get("symbol", "").upper(),
                "current_price": market_data.get("current_price", {}).get("usd", 0),
                "market_cap": market_data.get("market_cap", {}).get("usd", 0),
                "market_cap_rank": market_data.get("market_cap_rank", 0),
                "volume_24h": market_data.get("total_volume", {}).get("usd", 0),
                "price_change_24h": market_data.get("price_change_percentage_24h", 0),
                "price_change_7d": market_data.get("price_change_percentage_7d", 0),
                "price_change_30d": market_data.get("price_change_percentage_30d", 0),
                "circulating_supply": market_data.get("circulating_supply", 0),
                "total_supply": market_data.get("total_supply", 0),
                "all_time_high": market_data.get("ath", {}).get("usd", 0),
                "all_time_low": market_data.get("atl", {}).get("usd", 0),
                "last_updated": data.get("last_updated", "")
            }
            
        except requests.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
    
    def _handle_historical_mode(query: str, historical_date: str) -> str:
        """Handle historical backtesting mode by fetching data for the specific date."""
        try:
            from datetime import datetime, timedelta
            from pathlib import Path
            
            # Extract crypto ID from query
            crypto_id = _extract_crypto_id(query)
            
            # Initialize rate limiter and cache (similar to backtesting framework)
            rate_limiter = APIRateLimiter(max_requests_per_minute=30, safety_buffer=0.1)
            
            # Create cache in a temporary location for historical data
            cache_file = Path("temp_historical_cache.json")
            api_cache = APICache(cache_file=str(cache_file), max_age_hours=24*365)  # Long cache for historical data
            
            current_date = datetime.strptime(historical_date, '%Y-%m-%d')
            next_date = current_date + timedelta(days=1)
            
            # Check cache first
            cache_key = f"{crypto_id}_{historical_date}"
            cached_data = api_cache.get(cache_key)
            if cached_data:
                return _format_historical_data_for_query(cached_data, query)
            
            # Fetch historical data using CoinGecko historical API
            session = _get_session()
            date_formatted = current_date.strftime('%d-%m-%Y')
            next_date_formatted = next_date.strftime('%d-%m-%Y')
            
            # Wait for rate limit
            rate_limiter.wait_if_needed()
            
            # Get current day data
            url = f"{Config.COINGECKO_BASE_URL}/coins/{crypto_id}/history"
            params = {
                "date": date_formatted,
                "localization": "false"
            }
            
            response = session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if "market_data" not in data:
                return json.dumps({"error": f"No market data available for {historical_date}"})
            
            market_data = data["market_data"]
            current_price = market_data.get("current_price", {}).get("usd")
            
            if not current_price:
                return json.dumps({"error": f"No USD price available for {historical_date}"})
            
            # Get next day data for price movement calculation
            next_day_price = None
            try:
                rate_limiter.wait_if_needed()
                next_response = session.get(
                    f"{Config.COINGECKO_BASE_URL}/coins/{crypto_id}/history",
                    params={"date": next_date_formatted, "localization": "false"},
                    timeout=15
                )
                if next_response.status_code == 200:
                    next_data = next_response.json()
                    if "market_data" in next_data:
                        next_day_price = next_data["market_data"].get("current_price", {}).get("usd")
            except:
                pass  # Next day data is optional
            
            # Calculate price movement
            actual_movement = None
            if next_day_price:
                actual_movement = ((next_day_price - current_price) / current_price) * 100
            
            # Create comprehensive historical data
            historical_result = {
                "date": historical_date,
                "cryptocurrency": crypto_id,
                "current_price": current_price,
                "next_day_price": next_day_price,
                "actual_movement_24h": actual_movement,
                "market_cap": market_data.get("market_cap", {}).get("usd", 0),
                "volume_24h": market_data.get("total_volume", {}).get("usd", 0),
                "ohlcv": {
                    "timestamp": current_date.isoformat(),
                    "open": current_price,
                    "high": current_price,
                    "low": current_price,
                    "close": current_price,
                    "volume": market_data.get("total_volume", {}).get("usd", 0)
                },
                "historical_mode": True,
                "data_source": "coingecko_historical_api"
            }
            
            # Cache the result
            api_cache.set(cache_key, historical_result)
            
            # Format according to the original query type
            return _format_historical_data_for_query(historical_result, query)
            
        except Exception as e:
            return json.dumps({"error": f"Historical mode error: {str(e)}"})



    def _format_historical_data_for_query(historical_data: dict, query: str) -> str:
        """Format historical data according to the query type."""
        try:
            crypto_id = historical_data["cryptocurrency"]
            historical_date = historical_data["date"]
            
            if "current_price" in query.lower() or "price" in query.lower():
                # Return current price format
                result = {
                    "cryptocurrency": historical_data["cryptocurrency"],
                    "current_price": historical_data["current_price"],
                    "market_cap": historical_data.get("market_cap", 0),
                    "volume_24h": historical_data.get("volume_24h", 0),
                    "historical_date": historical_data["date"],
                    "data_source": "historical_mode"
                }
            elif "ohlcv" in query.lower() or "historical" in query.lower():
                # For backtesting: get OHLCV data from BEFORE the prediction date
                # If predicting 2024-12-01, get data from 7 days before: 2024-11-24 to 2024-11-30
                from datetime import datetime, timedelta
                
                prediction_date = datetime.strptime(historical_date, '%Y-%m-%d')
                # Get 7 days of data ending the day BEFORE the prediction date
                end_date = prediction_date - timedelta(days=1)  # Day before prediction
                days_to_fetch = 7  # Always use 7 days for backtesting
                
                print(f"Historical OHLCV: Predicting for {historical_date}, fetching {days_to_fetch} days ending {end_date.strftime('%Y-%m-%d')}")
                
                # Use market chart API to get proper OHLCV data for the historical period
                session = _get_session()
                
                # Calculate days from end_date backwards
                start_date = end_date - timedelta(days=days_to_fetch-1)
                
                # Use market_chart endpoint with specific date range
                url = f"{Config.COINGECKO_BASE_URL}/coins/{crypto_id}/market_chart/range"
                params = {
                    "vs_currency": "usd",
                    "from": int(start_date.timestamp()),
                    "to": int(end_date.timestamp() + 86400)  # Add 1 day to include end_date
                }
                
                try:
                    response = session.get(url, params=params, timeout=15)
                    response.raise_for_status()
                    
                    data = response.json()
                    prices = data.get("prices", [])
                    volumes = data.get("total_volumes", [])
                    
                    if not prices:
                        print(f"No price data available for range {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                        ohlcv_data = []
                    else:
                        # Process the data using the same logic as _get_ohlcv_data
                        import pandas as pd
                        
                        # Convert to DataFrame
                        price_df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                        volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                        
                        # Convert timestamps
                        price_df = _convert_timestamps(price_df, 'timestamp')
                        volume_df = _convert_timestamps(volume_df, 'timestamp')
                        
                        # Merge data
                        merged_df = pd.merge(price_df, volume_df, on='datetime', how='left', suffixes=('_price', '_volume'))
                        merged_df['volume'] = merged_df['volume'].fillna(0)
                        merged_df['timestamp'] = merged_df['timestamp_price']
                        merged_df = merged_df.drop(['timestamp_price', 'timestamp_volume'], axis=1, errors='ignore')
                        
                        # Group by day for daily OHLCV
                        merged_df['period'] = merged_df['datetime'].dt.floor('D')
                        
                        grouped = merged_df.groupby('period').agg({
                            'price': ['first', 'max', 'min', 'last'],
                            'volume': 'sum',
                            'timestamp': 'first'
                        })
                        
                        grouped.columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
                        grouped = grouped.reset_index()
                        
                        # Convert to OHLCV format
                        ohlcv_data = []
                        for _, row in grouped.iterrows():
                            ohlcv_data.append({
                                "timestamp": row['period'].isoformat(),
                                "open": float(row["open"]),
                                "high": float(row["high"]),
                                "low": float(row["low"]),
                                "close": float(row["close"]),
                                "volume": float(row["volume"])
                            })
                        
                        print(f"Successfully fetched {len(ohlcv_data)} days of historical OHLCV data")
                
                except Exception as e:
                    print(f"Error fetching historical OHLCV data: {e}")
                    ohlcv_data = []
                
                result = {
                    "cryptocurrency": historical_data["cryptocurrency"],
                    "ohlcv_data": ohlcv_data,
                    "data_points": len(ohlcv_data),
                    "has_volume_data": len(ohlcv_data) > 0,
                    "resolution": "daily",
                    "data_source": "historical_market_chart_api",
                    "historical_date": historical_data["date"],
                    "data_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                    "days_fetched": len(ohlcv_data),
                    "note": f"Historical backtesting: {len(ohlcv_data)} days BEFORE prediction date {historical_date}"
                }
            else:
                # Return comprehensive data
                result = historical_data
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"error": f"Error formatting historical data: {str(e)}"})

    # Main execution
    try:
        # Check if we're in historical backtesting mode
        if historical_date and historical_date.strip():
            return _handle_historical_mode(query, historical_date)
        
        session = _get_session()
        crypto_id = _extract_crypto_id(query)
        
        # Check if horizon is specified for optimal data fetching
        if "horizon" in query.lower():
            # Extract horizon information
            import re
            horizon_match = re.search(r'(\d+\s*(?:hour|hr|day|week|month)s?)\s+horizon', query.lower())
            if horizon_match:
                horizon = horizon_match.group(1)
                days = _get_optimal_days_for_horizon(horizon)
            else:
                days = _extract_days(query)
        else:
            days = _extract_days(query)
        
        # Parse the query to determine the operation
        if "current_price" in query.lower() or "price" in query.lower():
            result = _get_current_price(crypto_id, session)
        elif "ohlcv" in query.lower() or "historical" in query.lower():
            result = _get_ohlcv_data(crypto_id, days, session)
        elif "market_stats" in query.lower() or "stats" in query.lower():
            result = _get_market_stats(crypto_id, session)
        else:
            # Default comprehensive data
            current_data = _get_current_price(crypto_id, session)
            market_stats = _get_market_stats(crypto_id, session)
            ohlcv_data = _get_ohlcv_data(crypto_id, days, session)
            
            # Add rate limiting
            time.sleep(Config.API_RATE_LIMIT_DELAY)
            
            result = {
                "current_data": current_data,
                "market_stats": market_stats,
                "recent_ohlcv": ohlcv_data
            }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"CoinGecko API error: {str(e)}"})


# Legacy wrapper for backward compatibility
class CoinGeckoTool:
    """Legacy wrapper for the coingecko_tool function."""
    
    def __init__(self):
        self.name = "coingecko_tool"
        self.description = """
        Fetches cryptocurrency market data from CoinGecko API.
        Can get current prices, historical OHLCV data, and market statistics.
        Supports queries like 'bitcoin current price', 'ethereum ohlcv 7 days', 'solana market stats'.
        """
        self.base_url = Config.COINGECKO_BASE_URL
        self.api_key = Config.COINGECKO_API_KEY or ""
        self._session = requests.Session()
        if self.api_key:
            self._session.headers.update({"x-cg-demo-api-key": self.api_key})
    
    @property  
    def session(self):
        """Get the requests session."""
        if not hasattr(self, '_session'):
            self._session = requests.Session()
            if self.api_key:
                self._session.headers.update({"x-cg-demo-api-key": self.api_key})
        return self._session
    
    def _run(self, query: Optional[str] = None, crypto_id: Optional[str] = None, days: Optional[int] = None, historical_date: Optional[str] = None) -> str:
        """Legacy interface for the tool."""
        if crypto_id is not None:
            if days is not None:
                query = f"{crypto_id} ohlcv {days} days"
            else:
                query = f"{crypto_id} comprehensive data"
        elif query is None:
            query = "bitcoin current price"
            
        # Access the underlying function from the Tool object
        return coingecko_tool.func(query, historical_date)


def create_coingecko_tool():
    """Create and return a CoinGecko tool instance."""
    return coingecko_tool 