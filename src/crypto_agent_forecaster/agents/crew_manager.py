"""
Crew Manager for orchestrating the CryptoAgentForecaster system.
"""

from typing import Dict, Any, Optional
from crewai import Crew, Task
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .market_data_agent import create_crypto_market_data_agent
from .sentiment_agent import create_crypto_sentiment_analysis_agent
from .technical_agent import create_technical_analysis_agent
from .forecasting_agent import create_crypto_forecasting_agent
from ..prompts import get_task_prompts


class CryptoForecastingCrew:
    """Main crew manager for the CryptoAgentForecaster system."""
    
    def __init__(self):
        self.console = Console()
        
        # Create agents
        self.market_agent = create_crypto_market_data_agent()
        self.sentiment_agent = create_crypto_sentiment_analysis_agent()
        self.technical_agent = create_technical_analysis_agent()
        self.forecasting_agent = create_crypto_forecasting_agent()
        
        # Get prompts
        self.prompts = get_task_prompts()
        
        self.console.print(Panel.fit("🚀 CryptoAgentForecaster Crew Initialized", style="bold green"))
    
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
            expected_output="Historical OHLCV data for the past 30 days and current market statistics in JSON format"
        )
        tasks.append(market_data_task)
        
        # Task 2: Sentiment Analysis
        sentiment_task = Task(
            description=self.prompts["sentiment_analysis_task"].format(
                crypto_name=crypto_name,
                forecast_horizon=forecast_horizon
            ),
            agent=self.sentiment_agent,
            expected_output="Comprehensive sentiment analysis including scores, FUD/shill detection, and key topics"
        )
        tasks.append(sentiment_task)
        
        # Task 3: Technical Analysis
        technical_task = Task(
            description=self.prompts["technical_analysis_task"].format(
                crypto_name=crypto_name,
                forecast_horizon=forecast_horizon
            ),
            agent=self.technical_agent,
            expected_output="Technical analysis summary with indicators, patterns, and overall outlook",
            context=[market_data_task]  # Depends on market data
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
        
        try:
            # Create tasks
            tasks = self.create_tasks(crypto_name, forecast_horizon)
            
            # Create crew
            crew = Crew(
                agents=[
                    self.market_agent,
                    self.sentiment_agent, 
                    self.technical_agent,
                    self.forecasting_agent
                ],
                tasks=tasks,
                verbose=2,
                memory=True
            )
            
            # Run the crew
            self.console.print("\n🚀 Executing forecasting workflow...")
            result = crew.kickoff()
            
            # Parse and format results
            formatted_result = self._format_results(result, crypto_name, forecast_horizon)
            
            # Display results
            self._display_results(formatted_result)
            
            return formatted_result
            
        except Exception as e:
            error_msg = f"Error during forecasting: {str(e)}"
            self.console.print(f"❌ {error_msg}", style="bold red")
            return {
                "error": error_msg,
                "crypto_name": crypto_name,
                "forecast_horizon": forecast_horizon
            }
    
    def _format_results(self, raw_result: Any, crypto_name: str, forecast_horizon: str) -> Dict[str, Any]:
        """Format the raw crew results into a structured output."""
        
        # Extract final forecast from the last task
        final_forecast = str(raw_result)
        
        # Try to parse forecast components
        forecast_data = {
            "crypto_name": crypto_name,
            "forecast_horizon": forecast_horizon,
            "forecast": final_forecast,
            "direction": self._extract_direction(final_forecast),
            "confidence": self._extract_confidence(final_forecast),
            "explanation": self._extract_explanation(final_forecast),
            "timestamp": self._get_timestamp()
        }
        
        return forecast_data
    
    def _extract_direction(self, forecast_text: str) -> str:
        """Extract direction from forecast text."""
        text_upper = forecast_text.upper()
        
        if "UP" in text_upper or "BULLISH" in text_upper or "BUY" in text_upper:
            return "UP"
        elif "DOWN" in text_upper or "BEARISH" in text_upper or "SELL" in text_upper:
            return "DOWN"
        else:
            return "NEUTRAL"
    
    def _extract_confidence(self, forecast_text: str) -> str:
        """Extract confidence from forecast text."""
        text_upper = forecast_text.upper()
        
        if "HIGH CONFIDENCE" in text_upper or "VERY CONFIDENT" in text_upper:
            return "HIGH"
        elif "LOW CONFIDENCE" in text_upper or "LOW" in text_upper:
            return "LOW"
        else:
            return "MEDIUM"
    
    def _extract_explanation(self, forecast_text: str) -> str:
        """Extract explanation from forecast text."""
        # For now, return the full text as explanation
        # Could be enhanced with more sophisticated parsing
        return forecast_text
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _display_results(self, results: Dict[str, Any]):
        """Display formatted results to console."""
        
        # Create results table
        table = Table(title=f"📊 Forecast Results for {results['crypto_name'].upper()}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Direction", self._format_direction(results['direction']))
        table.add_row("Confidence", results['confidence'])
        table.add_row("Forecast Horizon", results['forecast_horizon'])
        table.add_row("Timestamp", results['timestamp'])
        
        self.console.print(table)
        
        # Display explanation
        explanation_panel = Panel(
            results['explanation'],
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