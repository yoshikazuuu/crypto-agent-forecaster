#!/usr/bin/env python3

from src.crypto_agent_forecaster.tools.coingecko_tool import coingecko_tool
from src.crypto_agent_forecaster.tools.technical_analysis_tool import technical_analysis_tool
import json
import base64
import os

# Test complete chart generation pipeline
print("=== Chart Generation Debug ===")

# Step 1: Get fresh data
print("\n1. Fetching fresh Bitcoin data...")
result = coingecko_tool.func('bitcoin ohlcv 7 days')
data = json.loads(result)

if 'ohlcv_data' in data:
    ohlcv_data = data['ohlcv_data']
    print(f"✅ Got {len(ohlcv_data)} data points")
    print(f"Date range: {ohlcv_data[0]['timestamp']} to {ohlcv_data[-1]['timestamp']}")
    
    # Step 2: Generate technical analysis with chart
    print("\n2. Running technical analysis with chart generation...")
    
    # Convert to JSON string for the tool
    ohlcv_json = json.dumps(ohlcv_data)
    
    # Run technical analysis
    analysis_result = technical_analysis_tool.func(ohlcv_json, "Bitcoin")
    
    print("\n3. Analysis result:")
    print(analysis_result[:500] + "..." if len(analysis_result) > 500 else analysis_result)
    
    # Step 3: Check if chart was generated
    from src.crypto_agent_forecaster.tools.technical_analysis_tool import get_current_chart_data
    chart_data = get_current_chart_data()
    
    if chart_data:
        print(f"\n✅ Chart generated successfully! ({len(chart_data)} characters)")
        
        # Save chart to debug file
        chart_bytes = base64.b64decode(chart_data)
        with open("debug_chart.png", "wb") as f:
            f.write(chart_bytes)
        print("📊 Chart saved as debug_chart.png")
        
        # Check file size
        file_size = os.path.getsize("debug_chart.png")
        print(f"Chart file size: {file_size:,} bytes")
        
    else:
        print("❌ No chart data generated")

else:
    print("❌ Failed to get OHLCV data")
    print(f"Data keys: {list(data.keys())}") 