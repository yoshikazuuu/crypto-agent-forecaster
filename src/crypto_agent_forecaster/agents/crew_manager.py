"""
Crew Manager for orchestrating the CryptoAgentForecaster system.
"""

import io
import contextlib
from typing import Dict, Any, Optional
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
from ..utils import LogCapture, save_run_results, sanitize_for_logging
from ..tools.technical_analysis_tool import get_current_chart_data, clear_chart_data


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
        
        if self.verbose:
            self.console.print(f"🔧 Verbose mode enabled - detailed execution logs will be shown")
        
        # Reset charts storage for this run
        self.current_run_charts = {}
        
        # Capture logs during execution
        with LogCapture() as log_capture:
            try:
                # Log start of process
                log_capture.log(f"Starting forecast for {crypto_name} with horizon {forecast_horizon}")
                log_capture.log(f"Verbose mode: {self.verbose}")
                
                # Log agents and tools being used
                for agent_name, info in self.agents_info.items():
                    log_capture.log(f"Agent: {agent_name} - Tools: {', '.join(info['tools'])}")
                
                # Create tasks
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
                    verbose=self.verbose,  # Now configurable based on verbose flag
                    memory=False
                )
                
                # Run the crew
                self.console.print("\n🚀 Executing forecasting workflow...")
                log_capture.log("Executing forecasting workflow...")
                
                if self.verbose:
                    # In verbose mode, don't capture stdout so users can see crew outputs
                    result = crew.kickoff()
                else:
                    # In non-verbose mode, capture stdout to log crew outputs
                    captured_output = io.StringIO()
                    with contextlib.redirect_stdout(captured_output):
                        result = crew.kickoff()
                    
                    # Log the sanitized crew output
                    crew_output = captured_output.getvalue()
                    if crew_output:
                        sanitized_output = sanitize_for_logging(crew_output, max_json_length=300)
                        log_capture.log(f"Crew execution output: {sanitized_output}")
                
                # Parse and format results
                formatted_result = self._format_results(result, crypto_name, forecast_horizon)
                log_capture.log(f"Forecast completed: Direction={formatted_result.get('direction')}, Confidence={formatted_result.get('confidence')}")
                
                # Add execution summary
                formatted_result['execution_summary'] = {
                    'agents_used': list(self.agents_info.keys()),
                    'tools_used': [tool for info in self.agents_info.values() for tool in info['tools']],
                    'verbose_mode': self.verbose
                }
                
                # Save results to dedicated folder
                saved_path = save_run_results(
                    results=formatted_result,
                    charts=self.current_run_charts,
                    logs=log_capture.get_logs(),
                    verbose=self.verbose
                )
                
                # Add save path to results
                formatted_result['saved_to'] = str(saved_path)
                
                # Display results
                self._display_results(formatted_result)
                
                # Notify about saved results
                self.console.print(f"\n💾 Complete results saved to: {saved_path}", style="bold green")
                
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
                
                saved_path = save_run_results(
                    results=error_result,
                    charts=self.current_run_charts,
                    logs=log_capture.get_logs(),
                    verbose=self.verbose
                )
                error_result['saved_to'] = str(saved_path)
                
                return error_result
    
    def _format_results(self, raw_result: Any, crypto_name: str, forecast_horizon: str) -> Dict[str, Any]:
        """Format the raw crew results into a structured output."""
        
        # Extract final forecast from the last task
        final_forecast = str(raw_result)
        
        # Extract any charts that were generated during the process
        charts_info = self._extract_charts_from_forecast(final_forecast)
        
        # Try to parse forecast components
        forecast_data = {
            "crypto_name": crypto_name,
            "forecast_horizon": forecast_horizon,
            "forecast": self._clean_forecast_text(final_forecast),  # Clean base64 from display text
            "direction": self._extract_direction(final_forecast),
            "confidence": self._extract_confidence(final_forecast),
            "explanation": self._extract_explanation(final_forecast),
            "timestamp": self._get_timestamp(),
            "charts_generated": len(self.current_run_charts) > 0,
            "charts_info": charts_info
        }
        
        return forecast_data
    
    def _extract_charts_from_forecast(self, forecast_text: str) -> Dict[str, str]:
        """Extract chart information from forecast text and store charts."""
        charts_info = {}
        
        # Get chart data from technical analysis tool
        chart_data = get_current_chart_data()
        if chart_data:
            chart_name = "technical_analysis_chart"
            charts_info[chart_name] = "Technical Analysis Chart"
            
            # Store the chart data for saving
            self.current_run_charts[chart_name] = chart_data
            
            # Clear the global chart data
            clear_chart_data()
        
        return charts_info
    
    def _clean_forecast_text(self, forecast_text: str) -> str:
        """Clean forecast text by removing base64 data for display."""
        from ..utils import hide_base64_from_logs
        return hide_base64_from_logs(forecast_text)
    
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
            "Market Data",
            "Sentiment + Technical Analysis"
        ]
        
        for i, (task_name, agent_name, dep) in enumerate(zip(task_names, agent_names, dependencies)):
            workflow_table.add_row(str(i+1), task_name, agent_name, dep)
        
        self.console.print(workflow_table)
    
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
        
        # Display explanation (cleaned)
        explanation_panel = Panel(
            results['explanation'][:1000] + "..." if len(results['explanation']) > 1000 else results['explanation'],
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