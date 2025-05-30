"""
Crew Manager for orchestrating the CryptoAgentForecaster system.
"""

import io
import contextlib
import sys
from typing import Dict, Any, Optional, List
from crewai import Crew, Task
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .market_data_agent import create_crypto_market_data_agent
from .sentiment_agent import create_crypto_sentiment_analysis_agent
from .technical_agent import create_technical_analysis_agent
from .forecasting_agent import create_crypto_forecasting_agent
from ..prompts import get_task_prompts
from ..utils import LogCapture, save_run_results, sanitize_for_logging, create_run_directory
from ..tools.technical_analysis_tool import get_current_chart_data, clear_chart_data, set_results_directory, save_chart_to_results


class CryptoForecastingCrew:
    """Main crew manager for the CryptoAgentForecaster system."""
    
    def __init__(self, verbose: bool = False):
        self.console = Console()
        self.verbose = verbose
        
        # Create agents
        self.market_agent = create_crypto_market_data_agent()
        self.sentiment_agent = create_crypto_sentiment_analysis_agent()
        self.technical_agent = create_technical_analysis_agent()
        self.forecasting_agent = create_crypto_forecasting_agent()
        
        # Get prompts
        self.prompts = get_task_prompts()
        
        # Storage for run data
        self.current_run_charts = {}
        self.current_run_market_data = {}  # Store current market data
        
        # Agent and tool information
        self.agents_info = {
            "Market Data Agent": {
                "agent": self.market_agent,
                "tools": ["CoinGeckoTool"],
                "description": "Collects historical OHLCV data and current market statistics"
            },
            "Sentiment Analysis Agent": {
                "agent": self.sentiment_agent,
                "tools": ["FourChanBizTool"],
                "description": "Analyzes sentiment from 4chan /biz/ discussions"
            },
            "Technical Analysis Agent": {
                "agent": self.technical_agent,
                "tools": ["TechnicalAnalysisTool"],
                "description": "Performs technical analysis and generates charts"
            },
            "Forecasting Agent": {
                "agent": self.forecasting_agent,
                "tools": ["Data fusion and synthesis"],
                "description": "Synthesizes all data to create final forecast"
            }
        }
        
        if self.verbose:
            self._display_initialization_info()
        else:
            self.console.print(Panel.fit("🚀 CryptoAgentForecaster Crew Initialized", style="bold green"))
    
    def _display_initialization_info(self):
        """Display detailed initialization information when verbose mode is enabled."""
        self.console.print(Panel.fit("🚀 CryptoAgentForecaster Crew Initialized - VERBOSE MODE", style="bold green"))
        
        # Display agents and tools table
        agents_table = Table(title="🤖 Agents & Tools Configuration")
        agents_table.add_column("Agent", style="cyan")
        agents_table.add_column("Tools", style="yellow")
        agents_table.add_column("Description", style="white")
        
        for agent_name, info in self.agents_info.items():
            tools_str = ", ".join(info["tools"])
            agents_table.add_row(agent_name, tools_str, info["description"])
        
        self.console.print(agents_table)
        
        # Display LLM configuration
        llm_info = Panel(
            f"🧠 LLM Configuration\n"
            f"All agents use the configured LLM provider and model settings\n"
            f"Verbose output enabled - detailed logs will be shown",
            title="🔧 Configuration",
            expand=False
        )
        self.console.print(llm_info)
    
    def create_tasks(self, crypto_name: str, forecast_horizon: str = "24 hours") -> list[Task]:
        """Create tasks for the forecasting workflow."""
        
        tasks = []
        
        # Task 1: Market Data Collection
        market_data_task = Task(
            description=self.prompts["market_data_task"].format(
                crypto_name=crypto_name,
                forecast_horizon=forecast_horizon
            ),
            agent=self.market_agent,
            expected_output="Concise market data summary including current price, data coverage, key metrics, and quality assessment (NO raw OHLCV data)"
        )
        tasks.append(market_data_task)
        
        # Task 2: Sentiment Analysis
        sentiment_task = Task(
            description=self.prompts["sentiment_analysis_task"].format(
                crypto_name=crypto_name,
                forecast_horizon=forecast_horizon
            ),
            agent=self.sentiment_agent,
            expected_output="Comprehensive sentiment analysis including scores, FUD/shill detection, and key narrative themes"
        )
        tasks.append(sentiment_task)
        
        # Task 3: Technical Analysis
        technical_task = Task(
            description=self.prompts["technical_analysis_task"].format(
                crypto_name=crypto_name,
                forecast_horizon=forecast_horizon
            ),
            agent=self.technical_agent,
            expected_output="Technical analysis insights summary with indicator signals, patterns, support/resistance levels, and outlook (NO raw data reproduction)"
            # Removed context dependency - technical analysis fetches its own data
        )
        tasks.append(technical_task)
        
        # Task 4: Forecasting and Fusion
        forecasting_task = Task(
            description=self.prompts["forecasting_task"].format(
                crypto_name=crypto_name,
                forecast_horizon=forecast_horizon
            ),
            agent=self.forecasting_agent,
            expected_output="Final forecast with direction (UP/DOWN/NEUTRAL), confidence score, and detailed explanation",
            context=[sentiment_task, technical_task]  # Depends on both analyses
        )
        tasks.append(forecasting_task)
        
        return tasks
    
    def run_forecast(self, crypto_name: str, forecast_horizon: str = "24 hours") -> Dict[str, Any]:
        """
        Run the complete forecasting workflow.
        
        Args:
            crypto_name: Name of the cryptocurrency to analyze
            forecast_horizon: Time horizon for the forecast
            
        Returns:
            Dictionary containing all results
        """
        self.console.print(f"\n🔮 Starting forecast for {crypto_name.upper()}")
        self.console.print(f"📅 Forecast horizon: {forecast_horizon}")
        
        if self.verbose:
            self.console.print(f"🔧 Verbose mode enabled - detailed execution logs will be shown")
        
        # Create results directory early and set it for technical analysis tool
        from datetime import datetime
        results_dir = create_run_directory(crypto_name, datetime.now())
        set_results_directory(str(results_dir))
        
        # Reset charts storage and capture current market data for this run
        self.current_run_charts = {}
        self.current_run_market_data = {}
        self._capture_current_market_data(crypto_name)
        
        # Capture logs for this forecast run
        with LogCapture() as log_capture:
            try:
                # Create tasks for this specific forecast
                tasks = self.create_tasks(crypto_name, forecast_horizon)
                
                if self.verbose:
                    self._display_workflow_plan(tasks)
                
                # Create crew with configurable verbosity
                crew = Crew(
                    agents=[
                        self.market_agent,
                        self.sentiment_agent, 
                        self.technical_agent,
                        self.forecasting_agent
                    ],
                    tasks=tasks,
                    verbose=True,  # Always enable crew verbose for complete logging
                    memory=False
                )
                
                # Run the crew with logging strategy based on user preference
                self.console.print("\n🚀 Executing forecasting workflow...")
                log_capture.log("Executing forecasting workflow...")
                
                if self.verbose:
                    # In verbose mode, show everything to user and capture to logs
                    self.console.print("🔧 Verbose mode: Showing real-time crew execution...")
                    
                    # Capture crew output to logs while still showing to user
                    original_stdout = sys.stdout
                    captured_output = io.StringIO()
                    
                    class TeeOutput:
                        def __init__(self, file1, file2):
                            self.file1 = file1
                            self.file2 = file2
                        
                        def write(self, data):
                            self.file1.write(data)
                            self.file2.write(data)
                        
                        def flush(self):
                            self.file1.flush() 
                            self.file2.flush()
                    
                    try:
                        # Tee output to both console and capture buffer
                        tee = TeeOutput(original_stdout, captured_output)
                        sys.stdout = tee
                        result = crew.kickoff()
                        sys.stdout = original_stdout
                        
                        # Log all crew output for permanent record
                        crew_output = captured_output.getvalue()
                        if crew_output:
                            sanitized_output = sanitize_for_logging(crew_output)
                            log_capture.log(f"Crew verbose execution output:\n{sanitized_output}")
                            
                    except Exception as capture_error:
                        sys.stdout = original_stdout
                        self.console.print(f"⚠️ Logging capture issue: {capture_error}", style="yellow")
                        # Fall back to normal execution without capture
                        result = crew.kickoff()
                        log_capture.log("Crew execution completed (capture failed)")
                        
                else:
                    # In non-verbose mode, capture crew output but don't show to user
                    self.console.print("🔇 Non-verbose mode: Capturing crew logs without terminal output...")
                    
                    try:
                        captured_output = io.StringIO()
                        with contextlib.redirect_stdout(captured_output):
                            result = crew.kickoff()
                        
                        # Always log the complete crew output for record keeping
                        crew_output = captured_output.getvalue()
                        if crew_output:
                            sanitized_output = sanitize_for_logging(crew_output)
                            log_capture.log(f"Crew execution output (non-verbose mode):\n{sanitized_output}")
                            self.console.print("✅ Crew execution completed (logs captured)", style="green")
                        else:
                            log_capture.log("Crew execution completed (no output captured)")
                            
                    except Exception as capture_error:
                        # If stdout capture fails (e.g., due to chart generation), fall back to normal execution
                        self.console.print(f"⚠️ Stdout capture failed, running without capture for compatibility: {capture_error}", style="yellow")
                        result = crew.kickoff()
                        log_capture.log(f"Crew execution completed with capture failure: {capture_error}")
                
                # Parse and format results
                formatted_result = self._format_results(result, crypto_name, forecast_horizon)
                log_capture.log(f"Forecast completed: Direction={formatted_result.get('direction')}, Confidence={formatted_result.get('confidence')}")
                
                # Add execution summary
                formatted_result['execution_summary'] = {
                    'agents_used': list(self.agents_info.keys()),
                    'tools_used': [tool for info in self.agents_info.values() for tool in info['tools']],
                    'verbose_mode': self.verbose
                }
                
                # Save results to dedicated folder (use existing directory)
                save_run_results(
                    results=formatted_result,
                    charts=self.current_run_charts,
                    logs=log_capture.get_logs(),
                    verbose=self.verbose,
                    existing_dir=results_dir  # Use the pre-created directory
                )
                
                # Add save path to results
                formatted_result['saved_to'] = str(results_dir)
                
                # Display results
                self._display_results(formatted_result)
                
                # Notify about saved results
                self.console.print(f"\n💾 Complete results saved to: {results_dir}", style="bold green")
                
                return formatted_result
                
            except Exception as e:
                error_msg = f"Error during forecasting: {str(e)}"
                log_capture.log(f"ERROR: {error_msg}")
                self.console.print(f"❌ {error_msg}", style="bold red")
                
                # Save error results too
                error_result = {
                    "error": error_msg,
                    "crypto_name": crypto_name,
                    "forecast_horizon": forecast_horizon,
                    "timestamp": self._get_timestamp(),
                    'execution_summary': {
                        'agents_used': list(self.agents_info.keys()),
                        'tools_used': [tool for info in self.agents_info.values() for tool in info['tools']],
                        'verbose_mode': self.verbose,
                        'status': 'error'
                    }
                }
                
                save_run_results(
                    results=error_result,
                    charts=self.current_run_charts,
                    logs=log_capture.get_logs(),
                    verbose=self.verbose,
                    existing_dir=results_dir  # Use the pre-created directory
                )
                error_result['saved_to'] = str(results_dir)
                
                return error_result
    
    def _capture_current_market_data(self, crypto_name: str):
        """Capture current market data from OHLCV data for consistency."""
        try:
            from ..tools.coingecko_tool import coingecko_tool
            import json
            import time
            
            print(f"🔍 Fetching fresh market data for {crypto_name}...")
            
            # Get current price data first for comparison
            current_price_result = coingecko_tool.func(f"{crypto_name} current price")
            current_price_data = json.loads(current_price_result)
            
            # Add small delay to ensure we don't get cached data
            time.sleep(1)
            
            # Get OHLCV data for consistency - using consistent query format
            ohlcv_result = coingecko_tool.func(f"{crypto_name} ohlcv 7 days")
            ohlcv_data = json.loads(ohlcv_result)
            
            if "error" not in current_price_data and "current_price" in current_price_data:
                api_current_price = current_price_data["current_price"]
                
                # ENHANCED SANITY CHECKS for different cryptocurrencies
                is_price_reasonable = True
                if crypto_name.lower() in ["bitcoin", "btc"]:
                    if api_current_price < 20000 or api_current_price > 200000:
                        print(f"⚠️ WARNING: Bitcoin price ${api_current_price:,} seems unrealistic!")
                        is_price_reasonable = False
                elif crypto_name.lower() in ["ethereum", "eth"]:
                    if api_current_price < 500 or api_current_price > 20000:
                        print(f"⚠️ WARNING: Ethereum price ${api_current_price:,} seems unrealistic!")
                        is_price_reasonable = False
                elif api_current_price <= 0:
                    print(f"⚠️ WARNING: Invalid price ${api_current_price} for {crypto_name}")
                    is_price_reasonable = False
                
                if is_price_reasonable:
                    print(f"✅ Current price ${api_current_price:,.2f} seems reasonable for {crypto_name}")
                else:
                    print(f"🚨 Price validation failed for {crypto_name} - may affect forecast quality")
            else:
                print(f"❌ Failed to fetch current price data for {crypto_name}")
                return
            
            if "error" not in ohlcv_data and "ohlcv_data" in ohlcv_data:
                # Extract current price from the most recent OHLCV data
                recent_data = ohlcv_data["ohlcv_data"]
                if recent_data:
                    latest_candle = recent_data[-1]  # Most recent candle
                    ohlcv_current_price = latest_candle["close"]
                    
                    print(f"📊 Data timestamps - Latest OHLCV: {latest_candle.get('timestamp', 'Unknown')}")
                    
                    # Compare API current price vs OHLCV close price with improved tolerance
                    if "error" not in current_price_data and "current_price" in current_price_data:
                        api_current_price = current_price_data["current_price"]
                        price_diff = abs(api_current_price - ohlcv_current_price)
                        price_diff_percent = (price_diff / api_current_price) * 100 if api_current_price > 0 else 0
                        
                        print(f"📊 Price Comparison: API ${api_current_price:,.2f} vs OHLCV ${ohlcv_current_price:,.2f}")
                        print(f"📊 Price Difference: ${price_diff:,.2f} ({price_diff_percent:.2f}%)")
                        
                        # Improved price difference tolerance - be more lenient for normal market movement
                        if price_diff_percent > 5:  # More than 5% difference is suspicious
                            print(f"⚠️ WARNING: Significant price difference ({price_diff_percent:.2f}%) detected!")
                            
                            # For moderate differences (5-15%), use API price but warn
                            if price_diff_percent <= 15:
                                final_price = api_current_price
                                print(f"🔧 Using more current API price: ${final_price:,.2f}")
                            else:
                                # For large differences (>15%), this indicates serious data issues
                                print(f"🚨 Large price difference detected! OHLCV data may be stale or incorrect.")
                                print(f"🚨 This could cause significant analysis inconsistencies.")
                                final_price = api_current_price
                                print(f"🔧 Using API price, but flagging for validation: ${final_price:,.2f}")
                        else:
                            print(f"✅ Price difference is acceptable, using OHLCV close price")
                            final_price = ohlcv_current_price
                    else:
                        final_price = ohlcv_current_price
                        print(f"🔧 Using OHLCV close price: ${final_price:,.2f}")
                    
                    # Calculate 24h price change if we have enough data
                    price_change_24h = 0
                    if len(recent_data) > 1:
                        # Find candle from ~24 hours ago (depending on resolution)
                        current_price = latest_candle["close"]
                        # Get price from 24 hours ago (look back approximately 24 data points for hourly data)
                        lookback_index = min(24, len(recent_data) - 1)
                        past_candle = recent_data[-(lookback_index + 1)]
                        past_price = past_candle["close"]
                        if past_price > 0:
                            price_change_24h = ((current_price - past_price) / past_price) * 100
                    
                    # Store comprehensive market data with validation flags
                    self.current_run_market_data = {
                        "current_price": final_price,
                        "volume_24h": latest_candle["volume"],
                        "timestamp": latest_candle["timestamp"],
                        "cryptocurrency": crypto_name,
                        "price_change_24h": price_change_24h,
                        "data_source": "validated_mixed",
                        "data_quality": {
                            "api_ohlcv_diff_percent": price_diff_percent if 'price_diff_percent' in locals() else 0,
                            "is_price_reasonable": is_price_reasonable,
                            "data_freshness": "fresh" if price_diff_percent < 5 else "potentially_stale",
                            "ohlcv_data_points": len(recent_data)
                        },
                        "price_validation": {
                            "api_price": current_price_data.get("current_price") if "error" not in current_price_data else None,
                            "ohlcv_price": ohlcv_current_price,
                            "price_diff_percent": price_diff_percent if 'price_diff_percent' in locals() else 0,
                            "validation_timestamp": time.time()
                        }
                    }
                    
                    data_quality = self.current_run_market_data["data_quality"]["data_freshness"]
                    quality_icon = "✅" if data_quality == "fresh" else "⚠️"
                    
                    print(f"{quality_icon} Market data captured: ${self.current_run_market_data['current_price']:,.2f} ({price_change_24h:+.2f}% 24h)")
                    print(f"📊 Data quality: {data_quality}")
                    
                    # If data quality is poor, warn about potential issues
                    if data_quality == "potentially_stale":
                        print(f"⚠️ Data quality warning: Price differences detected. Analysis may be affected.")
                        print(f"⚠️ Consider re-running the forecast if results seem inconsistent.")
                    
                else:
                    print(f"⚠️ No OHLCV data available")
                    self.current_run_market_data = {}
            else:
                print(f"⚠️ Could not capture OHLCV data: {ohlcv_data.get('error', 'Unknown error')}")
                # Store basic data from API only
                if "error" not in current_price_data and "current_price" in current_price_data:
                    self.current_run_market_data = {
                        "current_price": current_price_data["current_price"],
                        "cryptocurrency": crypto_name,
                        "data_source": "api_only",
                        "timestamp": time.time(),
                        "data_quality": {
                            "data_freshness": "api_only",
                            "ohlcv_available": False
                        }
                    }
                    print(f"⚠️ Using API-only data: ${current_price_data['current_price']:,.2f}")
                else:
                    self.current_run_market_data = {}
                
        except Exception as e:
            print(f"⚠️ Error capturing current market data: {str(e)}")
            import traceback
            traceback.print_exc()
            self.current_run_market_data = {}
    
    def _format_results(self, raw_result: Any, crypto_name: str, forecast_horizon: str) -> Dict[str, Any]:
        """Format the raw crew results into a structured output."""
        
        # Extract final forecast from the last task
        final_forecast = str(raw_result)
        
        # Extract any charts that were generated during the process
        charts_info = self._extract_charts_from_forecast(final_forecast)
        
        # Validate price consistency between captured data and forecast text
        captured_price = None
        price_consistency_error = False
        
        if self.current_run_market_data and "current_price" in self.current_run_market_data:
            captured_price = self.current_run_market_data["current_price"]
            print(f"✅ Market data captured price: ${captured_price:,.2f}")
        
        # Check for price mentions in forecast text that might be inconsistent
        import re
        text_price_matches = re.findall(r'\$([0-9,]+\.?[0-9]*)', final_forecast)
        
        if text_price_matches and captured_price:
            inconsistent_prices = []
            analysis_prices = []
            
            for price_str in text_price_matches:
                try:
                    text_price = float(price_str.replace(',', ''))
                    # Skip very small values (likely percentages or other data)
                    if text_price < 1:
                        continue
                    
                    price_diff = abs(text_price - captured_price)
                    price_diff_percent = (price_diff / captured_price) * 100
                    
                    # Be more selective about which prices to validate
                    # Only check prices that seem like they could be current market prices
                    price_ratio = text_price / captured_price
                    if 0.1 <= price_ratio <= 10:  # Price is within 10x range (reasonable for analysis)
                        analysis_prices.append((text_price, price_diff_percent))
                        
                        # Only flag as inconsistent if difference is very large AND price seems like current price
                        if price_diff_percent > 30:  # More than 30% difference
                            # Additional check: only flag if this looks like a current price mention
                            # (not a target price or historical reference)
                            text_context = final_forecast.lower()
                            price_mentions = [
                                f"current.*{price_str}", f"price.*{price_str}", 
                                f"{price_str}.*current", f"trading.*{price_str}"
                            ]
                            
                            is_current_price_mention = any(
                                re.search(pattern, text_context) for pattern in price_mentions
                            )
                            
                            if is_current_price_mention or price_diff_percent > 50:  # Very large difference
                                inconsistent_prices.append((text_price, price_diff_percent))
                        
                except (ValueError, ZeroDivisionError):
                    continue
            
            # Detect if analysis was done with seriously wrong price data
            major_inconsistencies = [p for p in inconsistent_prices if p[1] > 50]  # More than 50% difference
            
            if major_inconsistencies:
                print(f"🚨 CRITICAL ERROR: Major price inconsistency detected!")
                print(f"🚨 Analysis appears to be based on price ~${major_inconsistencies[0][0]:,.0f}")
                print(f"🚨 But current market price is ${captured_price:,.2f}")
                print(f"🚨 Difference: {major_inconsistencies[0][1]:.1f}%")
                
                # This forecast is unreliable - mark it as an error
                price_consistency_error = True
                
                error_explanation = f"""
🚨 **FORECAST INVALIDATED DUE TO PRICE INCONSISTENCY** 🚨

**Problem Detected:**
- Analysis was performed using price: ~${major_inconsistencies[0][0]:,.0f}
- Current market price: ${captured_price:,.2f}
- Price difference: {major_inconsistencies[0][1]:.1f}%

**Why This Happened:**
The AI agents received seriously stale or incorrect price data during analysis, making 
price targets, stop losses, and recommendations unreliable for the current market price.

**What This Means:**
- Price targets may be wrong
- Stop loss levels may be wrong  
- Risk/reward calculations may be affected
- Use forecast directional guidance with caution

**Recommended Action:**
Run the forecast again to get fresh data and consistent analysis.
"""
                
            elif inconsistent_prices:
                print(f"⚠️ Warning: Found {len(inconsistent_prices)} potentially inconsistent price(s)")
                for price, diff_pct in inconsistent_prices[:3]:  # Show top 3
                    print(f"   - ${price:,.0f} (differs by {diff_pct:.1f}%)")
                print(f"✅ Using verified market price: ${captured_price:,.2f}")
                print(f"ℹ️ Price differences may be due to targets or historical references")
            else:
                print(f"✅ Price consistency validated: Analysis aligns with current market data")
        
        # Extract forecast components with error handling for price consistency
        if price_consistency_error:
            # Return error result instead of invalid forecast
            forecast_data = {
                "crypto_name": crypto_name,
                "forecast_horizon": forecast_horizon,
                "error": "Price consistency validation failed",
                "error_details": {
                    "analysis_price": major_inconsistencies[0][0] if major_inconsistencies else "Unknown",
                    "current_market_price": captured_price,
                    "price_difference_percent": major_inconsistencies[0][1] if major_inconsistencies else 0,
                    "reason": "Analysis based on stale/incorrect price data"
                },
                "forecast": error_explanation,
                "direction": "ERROR",
                "confidence": "INVALID",
                "current_price": f"${captured_price:,.2f}" if captured_price else "Unknown",
                "targets": {"error": "Invalid due to price inconsistency"},
                "stop_loss": "Invalid due to price inconsistency",
                "take_profits": {"error": "Invalid due to price inconsistency"},
                "risk_reward_ratio": "Invalid",
                "position_size": "Do not trade",
                "time_horizon": forecast_horizon,
                "key_catalysts": ["Price data inconsistency detected"],
                "risk_factors": ["Analysis based on incorrect price data", "All recommendations are invalid"],
                "explanation": error_explanation,
                "timestamp": self._get_timestamp(),
                "charts_generated": len(self.current_run_charts) > 0,
                "charts_info": charts_info,
                "price_validation": {
                    "status": "FAILED",
                    "captured_price": captured_price,
                    "analysis_prices": analysis_prices,
                    "major_inconsistencies": major_inconsistencies
                }
            }
        else:
            # Normal processing for consistent data
            forecast_data = {
                "crypto_name": crypto_name,
                "forecast_horizon": forecast_horizon,
                "forecast": self._clean_forecast_text(final_forecast),
                "direction": self._extract_direction(final_forecast),
                "confidence": self._extract_confidence(final_forecast),
                "current_price": self._extract_current_price(final_forecast),
                "targets": self._extract_targets(final_forecast),
                "stop_loss": self._extract_stop_loss(final_forecast),
                "take_profits": self._extract_take_profits(final_forecast),
                "risk_reward_ratio": self._extract_risk_reward_ratio(final_forecast),
                "position_size": self._extract_position_size(final_forecast),
                "time_horizon": self._extract_time_horizon(final_forecast),
                "key_catalysts": self._extract_key_catalysts(final_forecast),
                "risk_factors": self._extract_risk_factors(final_forecast),
                "explanation": self._extract_explanation(final_forecast),
                "timestamp": self._get_timestamp(),
                "charts_generated": len(self.current_run_charts) > 0,
                "charts_info": charts_info,
                "price_validation": {
                    "status": "PASSED",
                    "captured_price": captured_price,
                    "inconsistencies_found": len(inconsistent_prices) if inconsistent_prices else 0
                }
            }
        
        return forecast_data
    
    def _extract_charts_from_forecast(self, forecast_text: str) -> Dict[str, str]:
        """Extract chart information from forecast text and store charts."""
        charts_info = {}
        
        # Get chart data from technical analysis tool
        chart_data = get_current_chart_data()
        if chart_data:
            chart_name = "technical_analysis_chart"
            charts_info[chart_name] = "Enhanced Technical Analysis Chart with Pattern Annotations"
            
            # Store the chart data for saving (base64)
            self.current_run_charts[chart_name] = chart_data
            print(f"✅ Chart data captured: {len(chart_data)} characters")
            
            # Save the chart file to results directory
            saved_chart_path = save_chart_to_results(chart_name)
            if saved_chart_path:
                charts_info[f"{chart_name}_file"] = saved_chart_path
            
            # Clear the global chart data
            clear_chart_data()
        else:
            # Check if forecast mentions chart generation
            if "chart" in forecast_text.lower() and "generated" in forecast_text.lower():
                print("⚠️ Chart mentioned in forecast but no chart data found")
            else:
                print("ℹ️ No chart data available from technical analysis")
        
        return charts_info
    
    def _clean_forecast_text(self, forecast_text: str) -> str:
        """Clean forecast text by removing base64 data for display."""
        from ..utils import hide_base64_from_logs
        return hide_base64_from_logs(forecast_text)
    
    def _extract_direction(self, forecast_text: str) -> str:
        """Extract direction from forecast text."""
        import re
        
        # First try to find explicit direction declarations with highest priority
        direction_patterns = [
            r'\*\*Direction\*\*:\s*(UP|DOWN|NEUTRAL)',
            r'\*\*Direction:\*\*\s*(UP|DOWN|NEUTRAL)',
            r'Direction:\s*(UP|DOWN|NEUTRAL)',
            r'\*\s*\*\*Direction\*\*\s*(UP|DOWN|NEUTRAL)',
            r'Direction\s*[:\-]\s*(UP|DOWN|NEUTRAL)',
        ]
        
        for pattern in direction_patterns:
            match = re.search(pattern, forecast_text, re.IGNORECASE)
            if match:
                direction = match.group(1).upper()
                print(f"✅ Direction extracted from explicit field: {direction}")
                return direction
        
        # Secondary patterns for less explicit mentions
        fallback_patterns = [
            r'forecast["\s:]*["\']?(UP|DOWN|NEUTRAL)',
            r'direction["\s:]*["\']?(UP|DOWN|NEUTRAL)',
            r'overall\s+(?:direction|forecast)["\s:]*["\']?(UP|DOWN|NEUTRAL)',
        ]
        
        for pattern in fallback_patterns:
            match = re.search(pattern, forecast_text, re.IGNORECASE)
            if match:
                direction = match.group(1).upper()
                print(f"ℹ️ Direction extracted from secondary pattern: {direction}")
                return direction
        
        # Final fallback to counting directional words, but with careful weighting
        text_upper = forecast_text.upper()
        
        # Count strong directional indicators
        strong_bullish = (
            text_upper.count("BULLISH") + 
            text_upper.count("BUY SIGNAL") + 
            text_upper.count("STRONG UPWARD") +
            text_upper.count("STRONG POSITIVE")
        )
        
        strong_bearish = (
            text_upper.count("BEARISH") + 
            text_upper.count("SELL SIGNAL") + 
            text_upper.count("STRONG DOWNWARD") +
            text_upper.count("STRONG NEGATIVE")
        )
        
        # Count neutral indicators
        neutral_indicators = (
            text_upper.count("NEUTRAL") +
            text_upper.count("MIXED SIGNALS") +
            text_upper.count("CONSOLIDATION") +
            text_upper.count("UNCERTAIN") +
            text_upper.count("CONFLICTING")
        )
        
        # Check for explicit forecasts
        forecast_up = len(re.findall(r'forecast.*?(?:up|bullish|positive)', text_upper))
        forecast_down = len(re.findall(r'forecast.*?(?:down|bearish|negative)', text_upper))
        forecast_neutral = len(re.findall(r'forecast.*?(?:neutral|mixed|uncertain)', text_upper))
        
        # Calculate totals
        total_bullish = strong_bullish + forecast_up
        total_bearish = strong_bearish + forecast_down  
        total_neutral = neutral_indicators + forecast_neutral
        
        # Make decision with preference for explicit neutral signals
        if total_neutral > max(total_bullish, total_bearish):
            print(f"ℹ️ Direction extracted from neutral signals: NEUTRAL (neutral: {total_neutral}, bullish: {total_bullish}, bearish: {total_bearish})")
            return "NEUTRAL"
        elif total_bullish > total_bearish:
            print(f"ℹ️ Direction extracted from sentiment analysis: UP (bullish: {total_bullish}, bearish: {total_bearish}, neutral: {total_neutral})")
            return "UP"
        elif total_bearish > total_bullish:
            print(f"ℹ️ Direction extracted from sentiment analysis: DOWN (bearish: {total_bearish}, bullish: {total_bullish}, neutral: {total_neutral})")
            return "DOWN"
        else:
            print(f"ℹ️ Direction defaulting to NEUTRAL due to balanced signals (bullish: {total_bullish}, bearish: {total_bearish}, neutral: {total_neutral})")
            return "NEUTRAL"
    
    def _extract_confidence(self, forecast_text: str) -> str:
        """Extract confidence from forecast text."""
        import re
        text_upper = forecast_text.upper()
        
        # More precise pattern matching to avoid false positives
        confidence_patterns = [
            # Look for explicit confidence level declarations first
            r'\*\*CONFIDENCE LEVEL\*\*:\s*(HIGH|MEDIUM|LOW)',
            r'\*\*CONFIDENCE:\*\*\s*(HIGH|MEDIUM|LOW)', 
            r'CONFIDENCE LEVEL:\s*(HIGH|MEDIUM|LOW)',
            r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)',
            
            # Look for confidence descriptions
            r'(HIGH|VERY HIGH)\s+CONFIDENCE',
            r'(MEDIUM|MODERATE|MODERATE)\s+CONFIDENCE',
            r'(LOW|VERY LOW)\s+CONFIDENCE',
            
            # Additional patterns
            r'CONFIDENCE\s+(?:IS\s+|LEVEL\s+(?:IS\s+)?)(HIGH|MEDIUM|LOW)',
        ]
        
        # Try each pattern in order of specificity
        for pattern in confidence_patterns:
            match = re.search(pattern, text_upper)
            if match:
                confidence_level = match.group(1).upper()
                print(f"✅ Confidence extracted using pattern '{pattern}': {confidence_level}")
                
                # Normalize variations
                if confidence_level in ['MODERATE', 'MEDIUM']:
                    return "MEDIUM"
                elif confidence_level in ['VERY HIGH']:
                    return "HIGH"
                elif confidence_level in ['VERY LOW']:
                    return "LOW"
                else:
                    return confidence_level
        
        # Fallback: Count confidence-related words with more precision
        high_indicators = len(re.findall(r'\b(?:HIGH|STRONG|VERY\s+CONFIDENT)\b', text_upper))
        medium_indicators = len(re.findall(r'\b(?:MEDIUM|MODERATE|MODERATELY\s+CONFIDENT)\b', text_upper))
        low_indicators = len(re.findall(r'\b(?:LOW|WEAK|UNCERTAIN|MIXED\s+SIGNALS)\b', text_upper))
        
        print(f"ℹ️ Confidence indicators found - High: {high_indicators}, Medium: {medium_indicators}, Low: {low_indicators}")
        
        # Make decision based on strongest signal
        if high_indicators > max(medium_indicators, low_indicators):
            print("ℹ️ Confidence extracted from high indicators: HIGH")
            return "HIGH"
        elif low_indicators > max(high_indicators, medium_indicators):
            print("ℹ️ Confidence extracted from low indicators: LOW")
            return "LOW"
        else:
            print("ℹ️ Confidence defaulting to MEDIUM")
            return "MEDIUM"
    
    def _extract_current_price(self, forecast_text: str) -> str:
        """Extract current price from forecast text or use captured market data."""
        # Always prioritize captured market data (most reliable and up-to-date)
        if self.current_run_market_data and "current_price" in self.current_run_market_data:
            current_price = self.current_run_market_data["current_price"]
            print(f"✅ Using captured market data current price: ${current_price:,.2f}")
            return f"${current_price:,.2f}"
        
        # If no captured data, the system has a problem - don't guess from text
        print("❌ No captured market data available - this should not happen")
        return "Data consistency error"
    
    def _extract_targets(self, forecast_text: str) -> Dict[str, str]:
        """Extract target prices from forecast text."""
        import re
        targets = {}
        
        # Look for primary and secondary targets
        primary_patterns = [
            r'\*\*Primary Target\*\*:\s*\$?([0-9,]+\.?[0-9]*)\s*\(Probability:\s*([0-9]+)%\)',
            r'Target 1.*?\$([0-9,]+\.?[0-9]*)\s*\(Probability:\s*([0-9]+)%\)',
            r'Primary.*?\$([0-9,]+\.?[0-9]*)\s*\(.*?([0-9]+)%\)',
        ]
        
        secondary_patterns = [
            r'\*\*Secondary Target\*\*:\s*\$?([0-9,]+\.?[0-9]*)\s*\(Probability:\s*([0-9]+)%\)',
            r'Target 2.*?\$([0-9,]+\.?[0-9]*)\s*\(Probability:\s*([0-9]+)%\)',
            r'Secondary.*?\$([0-9,]+\.?[0-9]*)\s*\(.*?([0-9]+)%\)',
        ]
        
        # Try to find primary target
        for pattern in primary_patterns:
            match = re.search(pattern, forecast_text, re.IGNORECASE | re.DOTALL)
            if match:
                targets['primary'] = f"${match.group(1)} ({match.group(2)}%)"
                break
        
        # Try to find secondary target  
        for pattern in secondary_patterns:
            match = re.search(pattern, forecast_text, re.IGNORECASE | re.DOTALL)
            if match:
                targets['secondary'] = f"${match.group(1)} ({match.group(2)}%)"
                break
        
        return targets
    
    def _extract_stop_loss(self, forecast_text: str) -> str:
        """Extract stop loss level from forecast text."""
        import re
        patterns = [
            r'\*\*Stop Loss Level\*\*:\s*\$?([0-9,]+\.?[0-9]*)',
            r'Stop Loss:\s*\$?([0-9,]+\.?[0-9]*)',
            r'Stop-Loss:\s*\$?([0-9,]+\.?[0-9]*)',
            r'stop.loss.*?\$([0-9,]+\.?[0-9]*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, forecast_text, re.IGNORECASE)
            if match:
                return f"${match.group(1)}"
        return "Not specified"
    
    def _extract_take_profits(self, forecast_text: str) -> Dict[str, str]:
        """Extract take profit levels from forecast text."""
        import re
        take_profits = {}
        
        # Look for take profit patterns
        tp1_patterns = [
            r'\*\*Take Profit 1\*\*:\s*\$?([0-9,]+\.?[0-9]*)',
            r'Take Profit 1.*?\$([0-9,]+\.?[0-9]*)',
            r'TP1.*?\$([0-9,]+\.?[0-9]*)',
        ]
        
        tp2_patterns = [
            r'\*\*Take Profit 2\*\*:\s*\$?([0-9,]+\.?[0-9]*)',
            r'Take Profit 2.*?\$([0-9,]+\.?[0-9]*)',
            r'TP2.*?\$([0-9,]+\.?[0-9]*)',
        ]
        
        # Try to find TP1
        for pattern in tp1_patterns:
            match = re.search(pattern, forecast_text, re.IGNORECASE)
            if match:
                take_profits['tp1'] = f"${match.group(1)}"
                break
        
        # Try to find TP2
        for pattern in tp2_patterns:
            match = re.search(pattern, forecast_text, re.IGNORECASE)
            if match:
                take_profits['tp2'] = f"${match.group(1)}"
                break
        
        return take_profits
    
    def _extract_risk_reward_ratio(self, forecast_text: str) -> str:
        """Extract risk-reward ratio from forecast text."""
        import re
        patterns = [
            r'\*\*Risk-Reward Ratio\*\*:\s*([0-9.:]+)',
            r'Risk-Reward Ratio:\s*([0-9.:]+)',
            r'Risk-Reward:\s*([0-9.:]+)',
            r'Risk.Reward.*?([0-9]+:[0-9.]+)',
            r'Ratio:\s*([0-9]+:[0-9.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, forecast_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return "Not specified"
    
    def _extract_position_size(self, forecast_text: str) -> str:
        """Extract position size recommendation from forecast text."""
        import re
        patterns = [
            r'\*\*Position Size Recommendation\*\*:\s*([0-9]+)%',
            r'Position Size:\s*([0-9]+)%',
            r'position.size.*?([0-9]+)%',
            r'([0-9]+)%.*?portfolio',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, forecast_text, re.IGNORECASE)
            if match:
                return f"{match.group(1)}%"
        return "Not specified"
    
    def _extract_time_horizon(self, forecast_text: str) -> str:
        """Extract time horizon from forecast text."""
        import re
        patterns = [
            r'\*\*Time Horizon\*\*:\s*([^-\n]+)',
            r'Time Horizon:\s*([^-\n]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, forecast_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "Not specified"
    
    def _extract_key_catalysts(self, forecast_text: str) -> List[str]:
        """Extract key catalysts from forecast text."""
        import re
        patterns = [
            r'\*\*Key Catalysts\*\*:\s*([^*]+?)(?=\*\*|$)',
            r'Key Catalysts:\s*([^*]+?)(?=\*\*|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, forecast_text, re.IGNORECASE | re.DOTALL)
            if match:
                catalysts_text = match.group(1).strip()
                # Split by common delimiters
                catalysts = [c.strip() for c in re.split(r'[,;-]|\n-', catalysts_text) if c.strip()]
                return catalysts[:3]  # Limit to top 3
        return ["Not specified"]
    
    def _extract_risk_factors(self, forecast_text: str) -> List[str]:
        """Extract risk factors from forecast text."""
        import re
        patterns = [
            r'\*\*Risk Factors\*\*:\s*([^*]+?)(?=\*\*|$)',
            r'Risk Factors:\s*([^*]+?)(?=\*\*|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, forecast_text, re.IGNORECASE | re.DOTALL)
            if match:
                risks_text = match.group(1).strip()
                # Split by common delimiters
                risks = [r.strip() for r in re.split(r'[,;-]|\n-', risks_text) if r.strip()]
                return risks[:3]  # Limit to top 3
        return ["Not specified"]
    
    def _extract_explanation(self, forecast_text: str) -> str:
        """Extract explanation from forecast text."""
        # Clean the text and return it as explanation
        return self._clean_forecast_text(forecast_text)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _display_workflow_plan(self, tasks: list[Task]):
        """Display the workflow plan when in verbose mode."""
        workflow_table = Table(title="📋 Execution Workflow Plan")
        workflow_table.add_column("Step", style="cyan")
        workflow_table.add_column("Task", style="yellow")
        workflow_table.add_column("Agent", style="green")
        workflow_table.add_column("Dependencies", style="white")
        
        task_names = [
            "Market Data Collection",
            "Sentiment Analysis", 
            "Technical Analysis",
            "Forecasting & Fusion"
        ]
        
        agent_names = [
            "Market Data Agent",
            "Sentiment Analysis Agent",
            "Technical Analysis Agent", 
            "Forecasting Agent"
        ]
        
        dependencies = [
            "None",
            "None",
            "None",
            "Sentiment + Technical Analysis"
        ]
        
        for i, (task_name, agent_name, dep) in enumerate(zip(task_names, agent_names, dependencies)):
            workflow_table.add_row(str(i+1), task_name, agent_name, dep)
        
        self.console.print(workflow_table)
    
    def _display_results(self, results: Dict[str, Any]):
        """Display formatted results to console."""
        
        # Create main results table
        table = Table(title=f"📊 Forecast Results for {results['crypto_name'].upper()}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Direction", self._format_direction(results['direction']))
        table.add_row("Confidence", results['confidence'])
        table.add_row("Current Price", results.get('current_price', 'Not specified'))
        table.add_row("Forecast Horizon", results['forecast_horizon'])
        
        # Add target information
        targets = results.get('targets', {})
        if 'primary' in targets:
            table.add_row("Primary Target", targets['primary'])
        if 'secondary' in targets:
            table.add_row("Secondary Target", targets['secondary'])
        
        # Add risk management info
        stop_loss = results.get('stop_loss', 'Not specified')
        table.add_row("Stop Loss", stop_loss)
        
        take_profits = results.get('take_profits', {})
        if 'tp1' in take_profits:
            table.add_row("Take Profit 1", take_profits['tp1'])
        if 'tp2' in take_profits:
            table.add_row("Take Profit 2", take_profits['tp2'])
        
        # Add other trading metrics
        risk_reward = results.get('risk_reward_ratio', 'Not specified')
        table.add_row("Risk-Reward Ratio", risk_reward)
        
        position_size = results.get('position_size', 'Not specified')
        table.add_row("Position Size", position_size)
        
        time_horizon = results.get('time_horizon', 'Not specified')
        table.add_row("Time to Target", time_horizon)
        
        table.add_row("Timestamp", results['timestamp'])
        
        if results.get('charts_generated'):
            table.add_row("Charts Generated", f"✅ {len(results.get('charts_info', {}))}")
        
        # Add execution summary if verbose
        if self.verbose and 'execution_summary' in results:
            table.add_row("Agents Used", f"{len(results['execution_summary']['agents_used'])}")
            table.add_row("Tools Used", f"{len(results['execution_summary']['tools_used'])}")
            table.add_row("Verbose Mode", "✅ Enabled")
        
        self.console.print(table)
        
        # Show execution details in verbose mode
        if self.verbose and 'execution_summary' in results:
            summary = results['execution_summary']
            
            exec_table = Table(title="🔧 Execution Summary")
            exec_table.add_column("Component", style="cyan")
            exec_table.add_column("Details", style="white")
            
            exec_table.add_row("Agents", ", ".join(summary['agents_used']))
            exec_table.add_row("Tools", ", ".join(summary['tools_used']))
            exec_table.add_row("Mode", "Verbose" if summary['verbose_mode'] else "Standard")
            
            self.console.print(exec_table)
        
        # Display key catalysts and risk factors
        catalysts = results.get('key_catalysts', ['Not specified'])
        risk_factors = results.get('risk_factors', ['Not specified'])
        
        if catalysts != ['Not specified'] or risk_factors != ['Not specified']:
            catalyst_risk_table = Table(title="🎯 Key Factors")
            catalyst_risk_table.add_column("Type", style="cyan")
            catalyst_risk_table.add_column("Factors", style="white")
            
            catalyst_risk_table.add_row("📈 Key Catalysts", "\n• ".join([""] + catalysts[:3]))
            catalyst_risk_table.add_row("⚠️ Risk Factors", "\n• ".join([""] + risk_factors[:3]))
            
            self.console.print(catalyst_risk_table)
        
        # Display explanation (cleaned)
        explanation_panel = Panel(
            results['explanation'][:1500] + "..." if len(results['explanation']) > 1500 else results['explanation'],
            title="🧠 Analysis & Reasoning",
            expand=False
        )
        self.console.print(explanation_panel)
    
    def _format_direction(self, direction: str) -> str:
        """Format direction with appropriate styling."""
        if direction == "UP":
            return "🟢 UP (Bullish)"
        elif direction == "DOWN":
            return "🔴 DOWN (Bearish)"
        else:
            return "🟡 NEUTRAL" 