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
    help="🔮 Multimodal cryptocurrency forecasting with AI agents",
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
    crypto: str = typer.Argument(..., help="Cryptocurrency name (e.g., bitcoin, ethereum)"),
    horizon: str = typer.Option("24 hours", "--horizon", "-h", help="Forecast time horizon"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider (openai/anthropic/google)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific model to use"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """
    🔮 Generate a cryptocurrency price forecast.
    
    Results are automatically saved to the 'results/' folder with charts and logs.
    
    Examples:
        crypto-agent-forecaster forecast bitcoin
        crypto-agent-forecaster forecast ethereum --horizon "3 days"
        crypto-agent-forecaster forecast solana --provider anthropic
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
    if not Confirm.ask("\nProceed with forecast?", default=True):
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
    crypto: str = typer.Option("bitcoin", help="Cryptocurrency to test with"),
    quick: bool = typer.Option(False, "--quick", help="Quick test (fewer data points)")
):
    """
    🧪 Test the system components.
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
        if market_data and "error" not in market_data:
            tech_tool = TechnicalAnalysisTool()
            tech_analysis = tech_tool._run(ohlcv_data=market_data, crypto_name=crypto)
            console.print("   ✅ Technical analysis working")
        else:
            console.print("   ❌ Technical analysis failed (no market data)")
        
        console.print("\n✅ All tests passed!", style="bold green")
        
    except Exception as e:
        console.print(f"\n❌ Test failed: {str(e)}", style="bold red")
        raise typer.Exit(1)


@app.command()
def list_cryptos():
    """
    📋 List available cryptocurrencies from CoinGecko.
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
    📋 Display available LLM models and recommendations.
    """
    display_banner()
    
    if not check_configuration():
        return
    
    console.print("\n🤖 LLM Model Information & Recommendations", style="bold")
    
    # Display available providers and their models
    providers = LLMFactory.get_available_providers()
    
    for provider in providers:
        console.print(f"\n📊 {provider.upper()} Models:", style="bold cyan")
        
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


if __name__ == "__main__":
    app()
