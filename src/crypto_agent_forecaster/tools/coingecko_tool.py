"""
CoinGecko API tool for fetching cryptocurrency market data.
"""

import time
import requests
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any
from crewai.tools import tool

from ..config import Config


@tool("coingecko_tool")
def coingecko_tool(query: str) -> str:
    """
    Fetches cryptocurrency market data from CoinGecko API.
    
    Args:
        query: Query for cryptocurrency data (e.g., 'bitcoin current price', 'ethereum ohlcv 30 days')
               Can include horizon information for optimal data fetching (e.g., 'bitcoin ohlcv 24 hours horizon')
    
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
        if "30 days" in query.lower() or "month" in query.lower():
            return 30
        elif "7 days" in query.lower() or "week" in query.lower():
            return 7
        elif "365 days" in query.lower() or "year" in query.lower():
            return 365
        else:
            return 7  # Default to 7 days
    
    def _get_optimal_days_for_horizon(horizon: str) -> int:
        """Get optimal historical data days based on forecast horizon."""
        horizon_lower = horizon.lower()
        
        # Extract numeric value and time unit
        if "hour" in horizon_lower:
            if "1 hour" in horizon_lower or "1hr" in horizon_lower:
                return 3  # 3 days for 1 hour forecast
            elif "4 hour" in horizon_lower or "4hr" in horizon_lower:
                return 7  # 1 week for 4 hour forecast
            elif "12 hour" in horizon_lower or "12hr" in horizon_lower:
                return 14  # 2 weeks for 12 hour forecast
            else:
                return 7  # Default 1 week for hour-based forecasts
        elif "day" in horizon_lower:
            if "1 day" in horizon_lower:
                return 30  # 1 month for 1 day forecast
            elif "3 day" in horizon_lower:
                return 60  # 2 months for 3 day forecast
            elif "7 day" in horizon_lower:
                return 90  # 3 months for 1 week forecast
            else:
                return 30  # Default 1 month for day-based forecasts
        elif "week" in horizon_lower:
            if "1 week" in horizon_lower:
                return 90  # 3 months for 1 week forecast
            elif "2 week" in horizon_lower:
                return 120  # 4 months for 2 week forecast
            else:
                return 90  # Default 3 months for week-based forecasts
        elif "month" in horizon_lower:
            return 365  # 1 year for month-based forecasts
        else:
            return 30  # Default to 1 month
    
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
                return {
                    "cryptocurrency": crypto_id,
                    "current_price": data[crypto_id].get("usd", 0),
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
        """Get historical OHLCV data."""
        try:
            # Use different endpoints based on days requested
            if days <= 1:
                # For very recent data, use market_chart
                url = f"{Config.COINGECKO_BASE_URL}/coins/{crypto_id}/market_chart"
                params = {"vs_currency": "usd", "days": days}
            else:
                # For longer periods, use OHLC endpoint
                url = f"{Config.COINGECKO_BASE_URL}/coins/{crypto_id}/ohlc"
                params = {"vs_currency": "usd", "days": days}
            
            response = session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if "ohlc" in url:
                # OHLC endpoint returns [[timestamp, open, high, low, close], ...]
                df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["volume"] = 0  # OHLC endpoint doesn't include volume
            else:
                # Market chart endpoint
                prices = data.get("prices", [])
                volumes = data.get("total_volumes", [])
                
                if not prices:
                    return {"error": "No price data available"}
                
                # Convert to OHLCV format (simplified - using price as all OHLC values)
                df_data = []
                for i, (timestamp, price) in enumerate(prices):
                    volume = volumes[i][1] if i < len(volumes) else 0
                    df_data.append({
                        "timestamp": pd.to_datetime(timestamp, unit="ms"),
                        "open": price,
                        "high": price,
                        "low": price, 
                        "close": price,
                        "volume": volume
                    })
                df = pd.DataFrame(df_data)
            
            # Convert to serializable format
            ohlcv_data = []
            for _, row in df.iterrows():
                ohlcv_data.append({
                    "timestamp": row["timestamp"].isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"])
                })
            
            return {
                "cryptocurrency": crypto_id,
                "days": days,
                "ohlcv_data": ohlcv_data,
                "data_points": len(ohlcv_data)
            }
            
        except requests.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
        except Exception as e:
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
    
    # Main execution
    try:
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
    
    def _run(self, query: str = None, crypto_id: str = None, days: int = None) -> str:
        """Legacy interface for the tool."""
        if crypto_id is not None:
            if days is not None:
                query = f"{crypto_id} ohlcv {days} days"
            else:
                query = f"{crypto_id} comprehensive data"
        elif query is None:
            query = "bitcoin current price"
            
        # Access the underlying function from the Tool object
        return coingecko_tool.func(query)


def create_coingecko_tool():
    """Create and return a CoinGecko tool instance."""
    return coingecko_tool 