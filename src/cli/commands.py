"""
Command handlers for the CLI application.
"""

import logging
import traceback
from typing import Optional
from datetime import datetime, timedelta

from .constants import ERROR_MESSAGES, SUCCESS_MESSAGES, DEFAULT_HORIZON
from .output import OutputManager
from .validation import Validator
from ..config import Config
from ..llm_factory import LLMFactory
from ..agents import CryptoForecastingCrew

logger = logging.getLogger(__name__)


class CommandHandler:
    """Handles all CLI command operations."""
    
    def __init__(self):
        self.output = OutputManager()
        self.validator = Validator(self.output)
    
    def handle_forecast(self, crypto: str, horizon: str = DEFAULT_HORIZON, 
                       provider: Optional[str] = None, model: Optional[str] = None,
                       verbose: bool = False, yes: bool = False) -> bool:
        """
        Handle forecast command.
        
        Args:
            crypto: Cryptocurrency name
            horizon: Forecast time horizon
            provider: LLM provider (optional)
            model: Model name (optional)
            verbose: Enable verbose output
            yes: Skip confirmation prompt
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.output.display_banner()
        
        # Validate configuration
        if not self.validator.validate_configuration():
            return False
        
        # Validate inputs
        if not self.validator.validate_forecast_inputs(crypto, horizon, provider, model):
            return False
        
        # Validate provider if specified
        if not self.validator.validate_provider(provider):
            return False
        
        # If no provider/model specified and defaults are empty, prompt for selection
        if not provider and not model and (not Config.DEFAULT_LLM_PROVIDER or not Config.DEFAULT_LLM_MODEL):
            provider, model = self.validator.prompt_model_selection()
        
        # Update configuration if provider/model specified
        if provider:
            Config.DEFAULT_LLM_PROVIDER = provider
        if model:
            Config.DEFAULT_LLM_MODEL = model
        
        # Display configuration
        self.output.display_forecast_configuration(
            crypto, horizon, Config.DEFAULT_LLM_PROVIDER, Config.DEFAULT_LLM_MODEL
        )
        
        # Confirm before proceeding
        if not yes and not self.validator.get_user_confirmation("\nProceed with forecast?", default=True):
            self.output.print("Forecast cancelled.")
            return False
        
        try:
            # Initialize the forecasting crew
            crew = CryptoForecastingCrew(verbose=verbose)
            
            # Run the forecast
            results = crew.run_forecast(crypto, horizon)
            
            # Check results
            if "error" not in results:
                self.output.display_success(SUCCESS_MESSAGES["forecast_completed"])
                return True
            else:
                self.output.display_error(f"{ERROR_MESSAGES['forecast_failed'].format(results['error'])}")
                return False
                
        except Exception as e:
            self.output.display_error(f"Unexpected error: {str(e)}")
            if verbose:
                self.output.print(traceback.format_exc())
            logger.error(f"Forecast command failed: {str(e)}")
            return False
    
    def handle_config(self) -> bool:
        """
        Handle config command.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.output.display_banner()
        
        try:
            is_valid = self.output.display_configuration_status()
            self.output.display_setup_instructions()
            
            if is_valid:
                logger.info("Configuration command completed successfully")
            else:
                logger.warning("Configuration issues detected")
            
            return is_valid
            
        except Exception as e:
            self.output.display_error(f"Configuration check failed: {str(e)}")
            logger.error(f"Config command failed: {str(e)}")
            return False
    
    def handle_debug(self) -> bool:
        """
        Handle debug command.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.output.display_banner()
        
        try:
            # Get debug information
            debug_info = self.validator.validate_debug_environment()
            
            # Display debug information
            self.output.display_debug_info(debug_info)
            
            # Display agent configuration status
            self.output.display_agent_configuration_status()
            
            # Display available providers
            self.output.display_available_providers()
            
            # Display troubleshooting tips
            self.output.display_troubleshooting_tips()
            
            logger.info("Debug command completed")
            return True
            
        except Exception as e:
            self.output.display_error(ERROR_MESSAGES["debug_failed"].format(str(e)))
            self.output.print(traceback.format_exc())
            logger.error(f"Debug command failed: {str(e)}")
            return False
    
    def handle_test(self, crypto: str = "bitcoin", quick: bool = False) -> bool:
        """
        Handle test command.
        
        Args:
            crypto: Cryptocurrency to test with
            quick: Whether to run quick test
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.output.display_banner()
        
        if not self.validator.validate_configuration():
            return False
        
        # Validate and normalize test parameters
        crypto, days = self.validator.validate_test_parameters(crypto, quick)
        
        self.output.print(f"\nTesting system with {crypto.upper()}...")
        
        try:
            # Import tools for testing
            from ..tools.coingecko_tool import CoinGeckoTool
            from ..tools.fourchan_tool import FourChanBizTool
            from ..tools.technical_analysis_tool import TechnicalAnalysisTool
            
            # Test CoinGecko tool
            self.output.print("1. Testing CoinGecko API...")
            coingecko_tool = CoinGeckoTool()
            market_data = coingecko_tool._run(query=f"{crypto} ohlcv {days} days")
            self.output.print("   ✅ CoinGecko API working")
            
            # Test 4chan tool (if not quick mode)
            if not quick:
                self.output.print("2. Testing 4chan /biz/ API...")
                fourchan_tool = FourChanBizTool()
                biz_data = fourchan_tool._run(keywords=[crypto, "crypto"], max_threads=2, max_posts_per_thread=5)
                self.output.print("   ✅ 4chan API working")
            else:
                self.output.print("2. Skipping 4chan test (quick mode)")
            
            # Test technical analysis
            self.output.print("3. Testing technical analysis...")
            tech_tool = TechnicalAnalysisTool()
            forecast_horizon = "7 days" if quick else "30 days"
            tech_analysis = tech_tool._run(crypto_name=crypto, forecast_horizon=forecast_horizon)
            self.output.print("   ✅ Technical analysis working")
            
            self.output.display_success(SUCCESS_MESSAGES["all_tests_passed"])
            logger.info(f"Test command completed successfully for {crypto}")
            return True
            
        except Exception as e:
            self.output.display_error(ERROR_MESSAGES["test_failed"].format(str(e)))
            logger.error(f"Test command failed: {str(e)}")
            return False
    
    def handle_list_cryptos(self) -> bool:
        """
        Handle list-cryptos command.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.output.display_crypto_list()
            logger.info("List cryptos command completed")
            return True
            
        except Exception as e:
            self.output.display_error(f"Failed to display crypto list: {str(e)}")
            logger.error(f"List cryptos command failed: {str(e)}")
            return False
    
    def handle_models(self) -> bool:
        """
        Handle models command.
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.output.display_banner()
        
        if not self.validator.validate_configuration():
            return False
        
        try:
            self.output.display_models_info()
            logger.info("Models command completed")
            return True
            
        except Exception as e:
            self.output.display_error(f"Failed to display models info: {str(e)}")
            logger.error(f"Models command failed: {str(e)}")
            return False
    
    def handle_help(self) -> bool:
        """
        Handle help command.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.output.display_banner()
            self.output.display_help_overview()
            logger.info("Help command completed")
            return True
            
        except Exception as e:
            self.output.display_error(f"Failed to display help: {str(e)}")
            logger.error(f"Help command failed: {str(e)}")
            return False

    def handle_backtest(self, crypto: str, start_date: Optional[str], end_date: Optional[str], 
                       methods: str, data_dir: str, resume: bool, quick_test: bool) -> bool:
        """
        Handle backtest command.
        
        Args:
            crypto: Cryptocurrency to backtest
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            methods: Prediction methods to test
            data_dir: Directory to store results
            resume: Whether to resume existing backtest
            quick_test: Whether to run quick test
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.output.display_banner()
        self.output.print(f"\n🧪 Starting backtest experiment for {crypto.upper()}")
        
        # Prompt for model selection if defaults are empty
        if not Config.DEFAULT_LLM_PROVIDER or not Config.DEFAULT_LLM_MODEL:
            provider, model = self.validator.prompt_model_selection()
            Config.DEFAULT_LLM_PROVIDER = provider
            Config.DEFAULT_LLM_MODEL = model
        
        try:
            # Import backtesting modules
            from ..backtesting.framework import BacktestingFramework
            from ..backtesting.analyzer import ThesisAnalyzer
            
            # Handle quick test mode - only override dates if not explicitly provided
            if quick_test and not start_date and not end_date:
                # Use a fixed recent date range that we know has real historical data
                # Set to December 2024 which should have real CoinGecko data available
                end_dt = datetime(2024, 12, 25)  # Christmas day 2024
                start_dt = end_dt - timedelta(days=7)  # Week before
                start_date = start_dt.strftime("%Y-%m-%d")
                end_date = end_dt.strftime("%Y-%m-%d")
                self.output.print(f"Quick test mode: Using {start_date} to {end_date} (7 days, real historical data)")
            elif quick_test and (start_date or end_date):
                # If dates are provided with quick test, limit the range to prevent long experiments
                if start_date and end_date:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    # Limit to 7 days max for quick test
                    if (end_dt - start_dt).days > 7:
                        end_dt = start_dt + timedelta(days=7)
                        end_date = end_dt.strftime("%Y-%m-%d")
                        self.output.print(f"Quick test mode: Limited date range to {start_date} to {end_date} (7 days max)")
            
            # Set default dates if not provided - use fixed dates that we know have real historical data
            if not start_date:
                # Use a known good date range (1 year ending December 2024)
                start_dt = datetime(2024, 1, 1)  # Start of 2024
                start_date = start_dt.strftime("%Y-%m-%d")
            if not end_date:
                # End at Christmas 2024, ensuring we have real historical data
                end_dt = datetime(2024, 12, 25)
                end_date = end_dt.strftime("%Y-%m-%d")
            
            # Parse and validate date format
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Basic sanity check - end date should be after start date
            if end_dt <= start_dt:
                self.output.display_error("End date must be after start date")
                return False
            
            # Parse methods
            if methods.lower() == "all":
                methods_list = ["full_agentic", "image_only", "sentiment_only"]
            else:
                method_mapping = {
                    "agentic": "full_agentic",
                    "full_agentic": "full_agentic",
                    "image": "image_only",
                    "image_only": "image_only",
                    "sentiment": "sentiment_only",
                    "sentiment_only": "sentiment_only"
                }
                methods_list = []
                for method in methods.split(","):
                    method = method.strip().lower()
                    if method in method_mapping:
                        methods_list.append(method_mapping[method])
                    else:
                        self.output.display_error(f"Unknown method: {method}")
                        return False
            
            self.output.print(f"Methods to test: {', '.join(methods_list)}")
            self.output.print(f"Date range: {start_date} to {end_date}")
            self.output.print(f"Data directory: {data_dir}")
            self.output.print(f"Resume existing: {resume}")
            
            # Initialize framework with crypto symbol
            framework = BacktestingFramework(crypto_symbol=crypto, data_dir=data_dir)
            
            # Run backtest
            self.output.print("\n🚀 Starting backtesting...")
            results = framework.run_backtest(
                start_date=start_date,
                end_date=end_date,
                methods=methods_list,
                skip_existing=resume
            )
            
            if not results:
                self.output.display_error("Backtesting failed")
                return False
            
            self.output.print("✅ Backtesting completed!")
            
            # Run analysis
            self.output.print("\n📊 Generating analysis...")
            logger.info("CLI: Initializing ThesisAnalyzer...")
            analyzer = ThesisAnalyzer(data_dir=data_dir)
            
            logger.info("CLI: Running prediction accuracy analysis...")
            analysis_results = analyzer.analyze_prediction_accuracy(results)
            
            if "error" in analysis_results:
                logger.error(f"CLI: Analysis failed: {analysis_results['error']}")
                self.output.print("❌ Analysis failed!")
                return False
            
            logger.info("CLI: Generating comprehensive thesis report...")
            report_path = analyzer.generate_thesis_report(results, crypto)
            
            if report_path and report_path.exists():
                self.output.print("✅ Analysis completed!")
                self.output.print(f"\n📁 Results saved to: {data_dir}/")
                self.output.print(f"📈 Charts: {data_dir}/charts/")
                self.output.print(f"📊 Analysis data: {data_dir}/analysis/")
                self.output.print(f"📝 Report: {data_dir}/analysis/{crypto}_thesis_report.md")
                logger.info(f"CLI: ✅ Complete analysis report generated at: {report_path}")
            else:
                self.output.print("⚠️  Analysis completed but report generation failed")
                logger.error("CLI: ❌ Report generation failed or file not created")
            
            logger.info(f"Backtest command completed successfully for {crypto}")
            return True
            
        except Exception as e:
            self.output.display_error(f"Backtesting failed: {str(e)}")
            logger.error(f"Backtest command failed: {str(e)}")
            if hasattr(self, 'output'):
                self.output.print(traceback.format_exc())
            return False 