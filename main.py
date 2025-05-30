#!/usr/bin/env python3
"""
CryptoAgentForecaster - CLI Application

A multimodal cryptocurrency forecasting system using hosted LLMs and agent-based architecture.

Features:
- Clean, truncated logging (no verbose JSON or base64 spam)
- Automatic result saving to organized folders for each run
- Technical analysis charts saved as PNG files
- Complete run logs and forecast results in structured directories
"""

import typer
import json
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from src.crypto_agent_forecaster.config import Config
from src.crypto_agent_forecaster.llm_factory import LLMFactory
from src.crypto_agent_forecaster.agents import CryptoForecastingCrew

# Initialize CLI app and console
app = typer.Typer(
    name="crypto-agent-forecaster",
    help="""🔮 CryptoAgentForecaster - Advanced AI-Driven Cryptocurrency Forecasting

A sophisticated multi-agent system that combines market data analysis, sentiment analysis, 
technical analysis, and LLM-powered forecasting to predict cryptocurrency price movements.

🚀 KEY FEATURES:
• Multi-agent AI analysis with 4 specialized agents
• Real-time market data from CoinGecko API  
• Social sentiment analysis from 4chan /biz/
• Advanced technical analysis with chart generation
• Multiple LLM provider support (OpenAI, Anthropic, Google)
• Interactive verbose mode for detailed execution tracking
• Automatic result saving with charts and logs
• Professional forecast reports in Markdown format

📊 AGENTS & TOOLS:
• Market Data Agent → CoinGecko API integration
• Sentiment Agent → 4chan /biz/ social sentiment analysis  
• Technical Agent → TA indicators + chart generation
• Forecasting Agent → Multi-modal data fusion & prediction

🎯 OUTPUTS:
• Price direction prediction (UP/DOWN/NEUTRAL)
• Confidence scoring (HIGH/MEDIUM/LOW)
• Technical analysis charts (PNG format)
• Detailed reasoning and explanation
• Complete execution logs and metadata
• Professional markdown reports with embedded charts

Use 'crypto-agent-forecaster COMMAND --help' for detailed command information.
""",
    rich_markup_mode="rich"
)
console = Console()


def display_banner():
    """Display application banner."""
    banner_text = """
🚀 CryptoAgentForecaster
Multimodal AI-Driven Cryptocurrency Price Forecasting

Powered by:
• 📊 CoinGecko API for market data
• 🤖 Hosted LLMs (OpenAI, Anthropic, Google)
• 🧠 Multi-agent analysis system
• 📈 Technical analysis & sentiment fusion
• 💬 4chan /biz/ sentiment analysis
"""
    console.print(Panel(banner_text, style="bold blue"))


def check_configuration():
    """Check and display configuration status."""
    console.print("\n🔧 Configuration Status:", style="bold")
    
    # Check LLM providers
    providers = LLMFactory.get_available_providers()
    
    provider_table = Table(title="LLM Providers")
    provider_table.add_column("Provider", style="cyan")
    provider_table.add_column("Status", style="white")
    provider_table.add_column("Model", style="yellow")
    
    provider_table.add_row(
        "OpenAI", 
        "✅ Configured" if Config.OPENAI_API_KEY else "❌ Not configured",
        "gpt-4o-mini / gpt-4o"
    )
    provider_table.add_row(
        "Anthropic", 
        "✅ Configured" if Config.ANTHROPIC_API_KEY else "❌ Not configured",
        "claude-3-5-sonnet-20241022"
    )
    provider_table.add_row(
        "Google", 
        "✅ Configured" if Config.GOOGLE_API_KEY else "❌ Not configured",
        "gemini-1.5-pro"
    )
    
    console.print(provider_table)
    
    # Check CoinGecko
    coingecko_status = "✅ Configured (Pro)" if Config.COINGECKO_API_KEY else "⚠️  Free tier (limited)"
    console.print(f"CoinGecko API: {coingecko_status}")
    
    if not providers:
        console.print("❌ No LLM providers configured! Please set up API keys.", style="bold red")
        console.print("Create a .env file with your API keys. See env_example for template.")
        return False
    
    console.print(f"✅ Ready with {len(providers)} LLM provider(s): {', '.join(providers)}", style="bold green")
    return True


@app.command()
def forecast(
    crypto: str = typer.Argument(
        ..., 
        help="Cryptocurrency name (use CoinGecko ID). Examples: bitcoin, ethereum, solana, cardano",
        metavar="CRYPTO_NAME"
    ),
    horizon: str = typer.Option(
        "24 hours", 
        "--horizon", "-h", 
        help="Forecast time horizon. Examples: '24 hours', '3 days', '1 week'",
        metavar="TIME_PERIOD"
    ),
    provider: Optional[str] = typer.Option(
        None, 
        "--provider", "-p", 
        help="LLM provider to use. Options: 'openai', 'anthropic', 'google'",
        metavar="PROVIDER"
    ),
    model: Optional[str] = typer.Option(
        None, 
        "--model", "-m", 
        help="Specific model to use. Examples: 'gpt-4o', 'claude-3-5-sonnet-20241022', 'gemini-1.5-pro'",
        metavar="MODEL_NAME"
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", "-v", 
        help="Enable verbose output with detailed execution tracking, agent interactions, and real-time progress"
    ),
    yes: bool = typer.Option(
        False, 
        "--yes", "-y", 
        help="Skip confirmation prompt and proceed automatically"
    )
):
    """
    🔮 Generate a comprehensive cryptocurrency price forecast using AI agents.
    
    This command orchestrates a sophisticated forecasting workflow using 4 specialized AI agents:
    
    📊 WORKFLOW PROCESS:
    1. Market Data Agent collects 30 days of OHLCV data + current market stats
    2. Sentiment Agent analyzes social sentiment from 4chan /biz/ discussions  
    3. Technical Agent performs TA analysis and generates interactive charts
    4. Forecasting Agent synthesizes all data into final prediction
    
    🎯 OUTPUTS GENERATED:
    • Forecast direction (UP/DOWN/NEUTRAL) with confidence score
    • Technical analysis charts saved as PNG files
    • Complete execution logs and agent interactions
    • Professional markdown report with embedded charts
    • Structured JSON data for programmatic access
    
    📁 RESULTS LOCATION:
    All results are automatically saved to 'results/CRYPTO_TIMESTAMP/' containing:
    • forecast_results.json - Complete forecast data
    • charts/ - Technical analysis charts (PNG format)
    • run_logs.txt - Detailed execution logs  
    • README.md - Professional report with embedded charts
    
    🔧 VERBOSE MODE:
    Use --verbose flag for interactive execution tracking including:
    • Real-time agent task progress
    • Tool usage monitoring (CoinGecko, 4chan, TA tools)
    • Detailed workflow visualization
    • Agent communication logs
    • Enhanced error diagnostics
    
    💡 EXAMPLES:
        # Basic forecast with default settings
        crypto-agent-forecaster forecast bitcoin
        
        # Verbose mode with detailed tracking
        crypto-agent-forecaster forecast ethereum --verbose
        
        # Custom time horizon
        crypto-agent-forecaster forecast solana --horizon "3 days"
        
        # Specify LLM provider and model
        crypto-agent-forecaster forecast cardano --provider anthropic --model claude-3-5-sonnet-20241022
        
        # Full customization with verbose output
        crypto-agent-forecaster forecast bitcoin --horizon "1 week" --provider openai --model gpt-4o --verbose
        
        # Quick analysis for altcoins
        crypto-agent-forecaster forecast chainlink --horizon "12 hours" --verbose
        
        # Automated execution without confirmation prompt
        crypto-agent-forecaster forecast bitcoin --yes
        
        # Fully automated with custom settings
        crypto-agent-forecaster forecast ethereum --horizon "3 days" --provider anthropic --verbose --yes
    
    📋 SUPPORTED CRYPTOCURRENCIES:
    Use 'crypto-agent-forecaster list-cryptos' to see popular options, or any valid CoinGecko ID.
    Common examples: bitcoin, ethereum, solana, cardano, polkadot, chainlink, avalanche-2
    
    ⚙️  LLM CONFIGURATION:
    Use 'crypto-agent-forecaster models' to see available providers and recommendations.
    Default provider/model can be set in .env file or overridden with --provider/--model flags.
    
    🛠️  TROUBLESHOOTING:
    • Use 'crypto-agent-forecaster config' to check API key configuration
    • Use 'crypto-agent-forecaster test' to verify system components
    • Enable --verbose for detailed error diagnostics
    • Check results/ folder for saved outputs even on partial failures
    """
    display_banner()
    
    # Check configuration
    if not check_configuration():
        raise typer.Exit(1)
    
    # Validate provider if specified
    if provider and provider not in LLMFactory.get_available_providers():
        console.print(f"❌ Provider '{provider}' not available or not configured", style="bold red")
        raise typer.Exit(1)
    
    # Update configuration if provider/model specified
    if provider:
        Config.DEFAULT_LLM_PROVIDER = provider
    if model:
        Config.DEFAULT_LLM_MODEL = model
    
    console.print(f"\n📋 Forecast Configuration:")
    console.print(f"• Cryptocurrency: {crypto.upper()}")
    console.print(f"• Time Horizon: {horizon}")
    console.print(f"• LLM Provider: {Config.DEFAULT_LLM_PROVIDER}")
    console.print(f"• Model: {Config.DEFAULT_LLM_MODEL}")
    
    # Confirm before proceeding
    if not yes and not Confirm.ask("\nProceed with forecast?", default=True):
        console.print("Forecast cancelled.")
        raise typer.Exit(0)
    
    try:
        # Initialize the forecasting crew
        crew = CryptoForecastingCrew(verbose=verbose)
        
        # Run the forecast
        results = crew.run_forecast(crypto, horizon)
        
        # Display success message
        if "error" not in results:
            console.print("\n✅ Forecast completed successfully!", style="bold green")
        else:
            console.print(f"\n❌ Forecast failed: {results['error']}", style="bold red")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"\n❌ Unexpected error: {str(e)}", style="bold red")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def config():
    """
    🔧 Display current configuration and setup information.
    
    This command shows the current system configuration including:
    • LLM provider API key status (OpenAI, Anthropic, Google)
    • Available models and their specifications
    • CoinGecko API configuration
    • Default provider and model settings
    • Setup instructions for missing configurations
    
    💡 CONFIGURATION FILES:
    • .env - Main configuration file with API keys
    • .env.example - Template file with configuration examples
    
    🔑 REQUIRED API KEYS:
    At minimum, you need ONE LLM provider configured:
    • OPENAI_API_KEY - For GPT models (gpt-4o, gpt-4o-mini)
    • ANTHROPIC_API_KEY - For Claude models (claude-3-5-sonnet)  
    • GOOGLE_API_KEY - For Gemini models (gemini-1.5-pro, gemini-2.0-flash)
    
    📊 OPTIONAL API KEYS:
    • COINGECKO_API_KEY - For higher rate limits and pro features
    
    🚀 SETUP PROCESS:
    1. Copy '.env.example' to '.env'
    2. Add your API keys to the .env file
    3. Run 'crypto-agent-forecaster config' to verify
    4. Use 'crypto-agent-forecaster test' to validate functionality
    
    💰 COST OPTIMIZATION:
    Use 'crypto-agent-forecaster models' to see cost comparisons and select
    the most cost-effective model for your use case.
    """
    display_banner()
    check_configuration()
    
    console.print("\n📝 Setup Instructions:", style="bold")
    console.print("1. Copy 'env_example' to '.env'")
    console.print("2. Add your API keys to the .env file")
    console.print("3. You need at least one LLM provider (OpenAI, Anthropic, or Google)")
    console.print("4. CoinGecko API key is optional but recommended")
    
    console.print("\n🔗 Get API Keys:")
    console.print("• OpenAI: https://platform.openai.com/api-keys")
    console.print("• Anthropic: https://console.anthropic.com/")
    console.print("• Google: https://aistudio.google.com/app/apikey")
    console.print("• CoinGecko: https://www.coingecko.com/en/api/pricing")


@app.command()
def test(
    crypto: str = typer.Option(
        "bitcoin", 
        help="Cryptocurrency to test with (use CoinGecko ID)",
        metavar="CRYPTO_NAME"
    ),
    quick: bool = typer.Option(
        False, 
        "--quick", 
        help="Quick test mode (fewer data points, faster execution)"
    )
):
    """
    🧪 Test system components and validate configuration.
    
    This command performs comprehensive system testing to ensure all components
    are working correctly before running forecasts.
    
    🔍 TESTS PERFORMED:
    1. CoinGecko API connectivity and data retrieval
    2. 4chan /biz/ API access and sentiment data collection  
    3. Technical analysis tool functionality and chart generation
    4. Data processing and integration workflows
    
    📊 TEST MODES:
    • Standard Mode: Full testing with 30 days of data
    • Quick Mode (--quick): Faster testing with 7 days of data
    
    💡 TROUBLESHOOTING:
    If tests fail, check:
    • Internet connectivity
    • API key configuration (use 'crypto-agent-forecaster config')
    • Rate limiting (wait a few minutes and retry)
    • Firewall settings blocking API access
    
    🎯 USE CASES:
    • Initial setup validation
    • Debugging forecast issues  
    • Testing new cryptocurrencies
    • Verifying system health after configuration changes
    
    📝 EXAMPLES:
        # Test with default Bitcoin
        crypto-agent-forecaster test
        
        # Quick test for faster validation
        crypto-agent-forecaster test --quick
        
        # Test specific cryptocurrency
        crypto-agent-forecaster test --crypto ethereum
        
        # Quick test with custom crypto
        crypto-agent-forecaster test --crypto solana --quick
    """
    display_banner()
    
    if not check_configuration():
        raise typer.Exit(1)
    
    console.print(f"\n🧪 Testing system with {crypto.upper()}...")
    
    try:
        from src.crypto_agent_forecaster.tools.coingecko_tool import CoinGeckoTool
        from src.crypto_agent_forecaster.tools.fourchan_tool import FourChanBizTool
        from src.crypto_agent_forecaster.tools.technical_analysis_tool import TechnicalAnalysisTool
        
        # Test CoinGecko tool
        console.print("1. Testing CoinGecko API...")
        coingecko_tool = CoinGeckoTool()
        
        # Use the legacy interface for testing
        days = 7 if quick else 30
        market_data = coingecko_tool._run(query=f"{crypto} ohlcv {days} days")
        console.print("   ✅ CoinGecko API working")
        
        # Test 4chan tool (if not quick mode)
        if not quick:
            console.print("2. Testing 4chan /biz/ API...")
            fourchan_tool = FourChanBizTool()
            biz_data = fourchan_tool._run(keywords=[crypto, "crypto"], max_threads=2, max_posts_per_thread=5)
            console.print("   ✅ 4chan API working")
        else:
            console.print("2. Skipping 4chan test (quick mode)")
        
        # Test technical analysis
        console.print("3. Testing technical analysis...")
        tech_tool = TechnicalAnalysisTool()
        tech_analysis = tech_tool._run(crypto_name=crypto, days=days)
        console.print("   ✅ Technical analysis working")
        
        console.print("\n✅ All tests passed!", style="bold green")
        
    except Exception as e:
        console.print(f"\n❌ Test failed: {str(e)}", style="bold red")
        raise typer.Exit(1)


@app.command(name="list-cryptos")
def list_cryptos():
    """
    📋 List popular cryptocurrencies available for analysis.
    
    This command displays a curated list of popular cryptocurrencies that work
    well with the forecasting system, along with their CoinGecko IDs and symbols.
    
    🎯 CRYPTOCURRENCY SELECTION:
    The listed cryptocurrencies are selected based on:
    • High trading volume and market liquidity
    • Active social media presence (for sentiment analysis)
    • Strong technical analysis signal availability
    • Reliable historical data from CoinGecko
    
    💡 USAGE TIPS:
    • Use the 'ID' column values as the crypto parameter for forecasts
    • All major cryptocurrencies beyond this list are also supported
    • Use any valid CoinGecko ID (check coinapi.com for full list)
    • Case-insensitive: 'Bitcoin', 'bitcoin', and 'BITCOIN' all work
    
    🔍 FINDING OTHER CRYPTOCURRENCIES:
    For cryptocurrencies not listed here:
    1. Visit coinapi.com or api.coingecko.com
    2. Search for your desired cryptocurrency
    3. Use the 'id' field from the API response
    4. Test with 'crypto-agent-forecaster test --crypto YOUR_CRYPTO_ID'
    
    📊 ANALYSIS QUALITY:
    Popular cryptocurrencies typically provide:
    • More social sentiment data from 4chan /biz/
    • Better technical analysis patterns
    • More reliable price predictions
    • Higher quality chart generation
    
    🚀 EXAMPLES:
        # Forecast popular cryptocurrencies
        crypto-agent-forecaster forecast bitcoin
        crypto-agent-forecaster forecast ethereum --verbose
        crypto-agent-forecaster forecast solana --horizon "3 days"
    """
    console.print("📋 Popular cryptocurrencies available for analysis:\n")
    
    popular_cryptos = [
        ("bitcoin", "Bitcoin", "BTC"),
        ("ethereum", "Ethereum", "ETH"), 
        ("solana", "Solana", "SOL"),
        ("cardano", "Cardano", "ADA"),
        ("polkadot", "Polkadot", "DOT"),
        ("chainlink", "Chainlink", "LINK"),
        ("polygon", "Polygon", "MATIC"),
        ("avalanche-2", "Avalanche", "AVAX"),
        ("dogecoin", "Dogecoin", "DOGE"),
        ("shiba-inu", "Shiba Inu", "SHIB")
    ]
    
    crypto_table = Table(title="Popular Cryptocurrencies")
    crypto_table.add_column("ID (use this)", style="cyan")
    crypto_table.add_column("Name", style="white")
    crypto_table.add_column("Symbol", style="yellow")
    
    for crypto_id, name, symbol in popular_cryptos:
        crypto_table.add_row(crypto_id, name, symbol)
    
    console.print(crypto_table)
    console.print("\n💡 Use the ID column when running forecasts")
    console.print("   Example: crypto-agent-forecaster forecast bitcoin")


@app.command()
def models():
    """
    📋 Display available LLM models, specifications, and recommendations.
    
    This command provides comprehensive information about available Language Models
    including costs, capabilities, and task-specific recommendations.
    
    🤖 SUPPORTED PROVIDERS:
    • OpenAI: GPT-4o, GPT-4o-mini (balanced performance and cost)
    • Anthropic: Claude-3.5-Sonnet (excellent reasoning and analysis)
    • Google: Gemini-1.5-Pro, Gemini-2.0-Flash (large context windows)
    
    💰 COST INFORMATION:
    • Input costs: Price per 1,000 input tokens
    • Output costs: Price per 1,000 output tokens  
    • Typical forecast costs: $0.01 - $0.10 depending on model choice
    • Use cheaper models for testing, premium models for production
    
    🎯 TASK-SPECIFIC RECOMMENDATIONS:
    • Sentiment Analysis: Claude models excel at nuanced sentiment understanding
    • Technical Analysis: GPT models provide structured technical insights
    • Data Fusion: All models perform well, choose based on cost preference
    • Chart Analysis: Models with vision capabilities (GPT-4o) for future multimodal features
    
    ⚙️  CONFIGURATION:
    • Set default provider/model in .env file:
      DEFAULT_LLM_PROVIDER=openai
      DEFAULT_LLM_MODEL=gpt-4o-mini
    • Override per-forecast with --provider and --model flags
    • Use 'crypto-agent-forecaster config' to check current settings
    
    🚀 PERFORMANCE TIPS:
    • Start with gpt-4o-mini for cost-effective testing
    • Upgrade to claude-3-5-sonnet for production forecasts
    • Use gemini models for experimental features and large context needs
    • Enable --verbose to see model performance in real-time
    
    📊 MODEL COMPARISON:
    • Speed: Gemini-2.0-Flash > GPT-4o-mini > Claude-3.5-Sonnet > GPT-4o
    • Quality: Claude-3.5-Sonnet ≥ GPT-4o > GPT-4o-mini > Gemini models
    • Cost: GPT-4o-mini < Gemini < Claude < GPT-4o
    • Context: Gemini (2M tokens) > Claude (200K) > GPT (128K)
    
    💡 EXAMPLES:
        # Use specific model for forecast
        crypto-agent-forecaster forecast bitcoin --provider openai --model gpt-4o
        
        # Cost-effective option
        crypto-agent-forecaster forecast ethereum --provider openai --model gpt-4o-mini
        
        # Premium analysis
        crypto-agent-forecaster forecast solana --provider anthropic --model claude-3-5-sonnet-20241022
    """
    display_banner()
    
    if not check_configuration():
        return
    
    console.print("\n🤖 LLM Model Information & Recommendations", style="bold")
    
    # Display available providers and their models
    providers = LLMFactory.get_available_providers()
    
    for provider in providers:
        console.print(f"\n {provider.upper()} Models:", style="bold cyan")
        
        models_table = Table(title=f"{provider.capitalize()} Model Specifications")
        models_table.add_column("Model", style="yellow")
        models_table.add_column("Max Tokens", style="cyan")
        models_table.add_column("Cost/1K Input", style="green")
        models_table.add_column("Cost/1K Output", style="red")
        
        model_specs = LLMFactory.MODEL_SPECS.get(provider, {})
        for model_name, specs in model_specs.items():
            cost_info = specs.get("cost_per_1k_tokens", {})
            models_table.add_row(
                model_name,
                f"{specs.get('max_tokens', 'Unknown'):,}",
                f"${cost_info.get('input', 0):.6f}",
                f"${cost_info.get('output', 0):.6f}"
            )
        
        console.print(models_table)
    
    # Display task recommendations
    console.print("\n🎯 Task-Specific Model Recommendations:", style="bold")
    
    tasks = ["sentiment_analysis", "technical_analysis", "multimodal_fusion", "cost_optimized"]
    
    rec_table = Table(title="Recommended Models by Task")
    rec_table.add_column("Task", style="cyan")
    rec_table.add_column("Provider", style="yellow") 
    rec_table.add_column("Model", style="green")
    rec_table.add_column("Reason", style="white")
    
    for task in tasks:
        rec = LLMFactory.get_recommended_model_for_task(task)
        rec_table.add_row(
            task.replace("_", " ").title(),
            rec["provider"].capitalize(),
            rec["model"],
            rec["reason"]
        )
    
    console.print(rec_table)
    
    console.print("\n💡 Usage Tips:", style="bold yellow")
    console.print("• Use --provider and --model flags to specify different models")
    console.print("• Claude models excel at nuanced sentiment analysis")
    console.print("• GPT models are strong for structured technical analysis") 
    console.print("• Gemini models have the largest context windows")
    console.print("• Consider cost vs. performance for your use case")


@app.command()
def help():
    """
    ❓ Quick help and usage examples for common tasks.
    
    This command provides a quick reference guide for the most common use cases
    and workflow examples to get you started quickly.
    
    🚀 QUICK START:
    1. Check configuration: crypto-agent-forecaster config
    2. Test system: crypto-agent-forecaster test --quick  
    3. Run first forecast: crypto-agent-forecaster forecast bitcoin --verbose
    4. Check results in the 'results/' folder
    
    💡 COMMON WORKFLOWS:
    
    📊 BASIC FORECASTING:
        # Simple Bitcoin forecast
        crypto-agent-forecaster forecast bitcoin
        
        # Ethereum with custom timeframe
        crypto-agent-forecaster forecast ethereum --horizon "3 days"
        
        # Verbose mode for learning/debugging
        crypto-agent-forecaster forecast solana --verbose
    
    🔧 CONFIGURATION & TESTING:
        # Check API keys and setup
        crypto-agent-forecaster config
        
        # Quick system validation
        crypto-agent-forecaster test --quick
        
        # Full system test with specific crypto
        crypto-agent-forecaster test --crypto cardano
    
    🤖 MODEL SELECTION:
        # Use cost-effective model
        crypto-agent-forecaster forecast bitcoin --provider openai --model gpt-4o-mini
        
        # Premium quality analysis
        crypto-agent-forecaster forecast ethereum --provider anthropic
        
        # See all available models
        crypto-agent-forecaster models
    
    📋 CRYPTOCURRENCY DISCOVERY:
        # See popular cryptocurrencies
        crypto-agent-forecaster list-cryptos
        
        # Try different altcoins
        crypto-agent-forecaster forecast chainlink --verbose
        crypto-agent-forecaster forecast avalanche-2 --horizon "1 week"
    
    🎯 PRODUCTION WORKFLOWS:
        # Daily BTC analysis with high-quality model
        crypto-agent-forecaster forecast bitcoin --provider anthropic --horizon "24 hours" --verbose --yes
        
        # Weekly portfolio review
        crypto-agent-forecaster forecast ethereum --horizon "1 week" --provider openai --model gpt-4o --yes
        
        # Quick market sentiment check
        crypto-agent-forecaster forecast bitcoin --horizon "4 hours" --provider openai --model gpt-4o-mini --yes
    
    📁 OUTPUT LOCATIONS:
        # All results saved to: results/CRYPTO_TIMESTAMP/
        # ├── forecast_results.json    # Complete data
        # ├── charts/                  # Technical analysis charts (PNG)
        # ├── run_logs.txt            # Execution logs
        # └── README.md               # Professional report with charts
    
    🔍 TROUBLESHOOTING:
        # Debug configuration issues
        crypto-agent-forecaster config
        
        # Test specific components
        crypto-agent-forecaster test --crypto bitcoin --quick
        
        # Get detailed error info
        crypto-agent-forecaster forecast bitcoin --verbose
        
        # Check model availability
        crypto-agent-forecaster models
    
    ⚡ PRO TIPS:
    • Use --verbose flag to understand what's happening
    • Start with gpt-4o-mini for cost-effective testing
    • Check 'results/' folder for all saved outputs
    • Popular cryptos (BTC, ETH) have better sentiment data
    • Set default provider in .env file to avoid typing --provider every time
    • Use shorter time horizons (4-24 hours) for higher accuracy
    
    📚 FULL DOCUMENTATION:
    Use 'crypto-agent-forecaster COMMAND --help' for detailed command documentation.
    """
    display_banner()
    
    console.print("🎯 Quick Reference Guide", style="bold blue")
    console.print("\nThis overview shows the most common usage patterns.")
    console.print("Use the commands above to get started quickly!")
    
    console.print("\n💡 Next Steps:", style="bold yellow")
    console.print("1. Run: crypto-agent-forecaster config")
    console.print("2. Run: crypto-agent-forecaster test --quick")  
    console.print("3. Run: crypto-agent-forecaster forecast bitcoin --verbose")
    console.print("4. Check the 'results/' folder for outputs")
    
    console.print("\n📖 For detailed help on any command:")
    console.print("crypto-agent-forecaster COMMAND --help")


if __name__ == "__main__":
    app()
