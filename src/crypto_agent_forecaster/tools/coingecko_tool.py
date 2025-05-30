"""
CoinGecko API tool for fetching cryptocurrency market data.
"""

import time
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..config import Config


class CoinGeckoInput(BaseModel):
    """Input for CoinGecko tool."""
    query: str = Field(description="Query for cryptocurrency data (e.g., 'bitcoin current price', 'ethereum ohlcv 30 days')")


class CoinGeckoTool(BaseTool):
    """Tool for fetching cryptocurrency data from CoinGecko API."""
    
    name: str = "coingecko_tool"
    description: str = """
    Fetches cryptocurrency market data from CoinGecko API.
    Can get current prices, historical OHLCV data, and market statistics.
    Supports queries like 'bitcoin current price', 'ethereum ohlcv 7 days', 'solana market stats'.
    """
    args_schema: type[BaseModel] = CoinGeckoInput
    
    def __init__(self):
        super().__init__()
        self.base_url = Config.COINGECKO_BASE_URL
        self.api_key = Config.COINGECKO_API_KEY
        self.session = requests.Session()
        
        # Add API key to headers if available
        if self.api_key:
            self.session.headers.update({"x-cg-demo-api-key": self.api_key})
    
    def _run(self, query: str = None, crypto_id: str = None, days: int = None) -> str:
        """Execute the CoinGecko API query."""
        try:
            # Handle direct parameters (for internal use)
            if crypto_id is not None:
                if days is not None:
                    result = self.get_ohlcv_data(crypto_id, days)
                else:
                    result = self.get_comprehensive_data(crypto_id)
                # Return JSON string for technical analysis tool
                if "ohlcv_data" in result:
                    import json
                    return json.dumps(result["ohlcv_data"])
                else:
                    import json
                    return json.dumps(result)
            
            # Handle query string (for LLM use)
            if query:
                # Parse the query to determine the operation
                if "current_price" in query.lower():
                    crypto_id = self._extract_crypto_id(query)
                    result = self.get_current_price(crypto_id)
                elif "ohlcv" in query.lower() or "historical" in query.lower():
                    crypto_id = self._extract_crypto_id(query)
                    days = self._extract_days(query)
                    result = self.get_ohlcv_data(crypto_id, days)
                elif "market_stats" in query.lower():
                    crypto_id = self._extract_crypto_id(query)
                    result = self.get_market_stats(crypto_id)
                else:
                    crypto_id = self._extract_crypto_id(query)
                    result = self.get_comprehensive_data(crypto_id)
                
                import json
                return json.dumps(result, indent=2)
            
            return json.dumps({"error": "No query or crypto_id provided"})
            
        except Exception as e:
            import json
            return json.dumps({"error": f"CoinGecko API error: {str(e)}"})
    
    def _extract_crypto_id(self, query: str) -> str:
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
            "ada": "cardano"
        }
        
        for name, coin_id in crypto_map.items():
            if name in query_lower:
                return coin_id
        
        # Default to bitcoin if no specific crypto found
        return "bitcoin"
    
    def _extract_days(self, query: str) -> int:
        """Extract number of days from query."""
        if "30 days" in query.lower() or "month" in query.lower():
            return 30
        elif "7 days" in query.lower() or "week" in query.lower():
            return 7
        elif "365 days" in query.lower() or "year" in query.lower():
            return 365
        else:
            return 7  # Default to 7 days
    
    def get_current_price(self, crypto_id: str) -> Dict[str, Any]:
        """Get current price and basic market data."""
        try:
            url = f"{self.base_url}/simple/price"
            params = {
                "ids": crypto_id,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_24hr_vol": "true",
                "include_market_cap": "true"
            }
            
            response = self.session.get(url, params=params)
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
    
    def get_ohlcv_data(self, crypto_id: str, days: int = 7) -> Dict[str, Any]:
        """Get historical OHLCV data."""
        try:
            # Use different endpoints based on days requested
            if days <= 1:
                # For very recent data, use market_chart
                url = f"{self.base_url}/coins/{crypto_id}/market_chart"
                params = {"vs_currency": "usd", "days": days}
            else:
                # For longer periods, use OHLC endpoint
                url = f"{self.base_url}/coins/{crypto_id}/ohlc"
                params = {"vs_currency": "usd", "days": days}
            
            response = self.session.get(url, params=params)
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
    
    def get_market_stats(self, crypto_id: str) -> Dict[str, Any]:
        """Get comprehensive market statistics."""
        try:
            url = f"{self.base_url}/coins/{crypto_id}"
            response = self.session.get(url)
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
    
    def get_comprehensive_data(self, crypto_id: str) -> Dict[str, Any]:
        """Get both current stats and recent OHLCV data."""
        current_data = self.get_current_price(crypto_id)
        market_stats = self.get_market_stats(crypto_id)
        ohlcv_data = self.get_ohlcv_data(crypto_id, days=7)
        
        # Add rate limiting
        time.sleep(Config.API_RATE_LIMIT_DELAY)
        
        return {
            "current_data": current_data,
            "market_stats": market_stats,
            "recent_ohlcv": ohlcv_data
        }


def create_coingecko_tool() -> CoinGeckoTool:
    """Create and return a CoinGecko tool instance."""
    return CoinGeckoTool() 