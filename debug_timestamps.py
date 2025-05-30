#!/usr/bin/env python3

from src.crypto_agent_forecaster.tools.coingecko_tool import coingecko_tool
import json

# Test current data fetching
result = coingecko_tool.func('bitcoin ohlcv 7 days')
data = json.loads(result)

print("=== CoinGecko Timestamp Debug ===")
print(f"Data structure keys: {list(data.keys())}")

# Handle different data structures
ohlcv_data = None
if 'ohlcv_data' in data:
    ohlcv_data = data['ohlcv_data']
elif 'recent_ohlcv' in data:
    recent = data['recent_ohlcv']
    if 'ohlcv_data' in recent:
        ohlcv_data = recent['ohlcv_data']

if ohlcv_data:
    print(f"Total data points: {len(ohlcv_data)}")
    print(f"First timestamp: {ohlcv_data[0]['timestamp']}")
    print(f"Last timestamp: {ohlcv_data[-1]['timestamp']}")
    print(f"First close price: ${ohlcv_data[0]['close']:.2f}")
    print(f"Last close price: ${ohlcv_data[-1]['close']:.2f}")
    
    # Show some sample records
    print("\nFirst 3 records:")
    for i in range(min(3, len(ohlcv_data))):
        record = ohlcv_data[i]
        print(f"  {record['timestamp']} - Close: ${record['close']:.2f}")
    
    print("\nLast 3 records:")
    for i in range(max(0, len(ohlcv_data)-3), len(ohlcv_data)):
        record = ohlcv_data[i]
        print(f"  {record['timestamp']} - Close: ${record['close']:.2f}")
else:
    print("No OHLCV data found!")
    print(f"Full data: {json.dumps(data, indent=2)[:500]}...") 