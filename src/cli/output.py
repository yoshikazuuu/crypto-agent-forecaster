"""
Output and display utilities for the CLI application.
"""

from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .constants import (
    BANNER_TEXT, POPULAR_CRYPTOS, API_KEY_URLS, SETUP_INSTRUCTIONS,
    TROUBLESHOOTING_TIPS, NEXT_STEPS, TABLE_STYLES, AGENT_TYPES
)
from ..config import Config
from ..llm_factory import LLMFactory


class OutputManager:
    """Manages all console output and display formatting."""
    
    def __init__(self):
        self.console = Console()
    
    def display_banner(self) -> None:
        """Display application banner."""
        self.console.print(Panel(BANNER_TEXT, style="bold blue"))
    
    def display_initialization_info(self) -> None:
        """Display initialization information for verbose mode."""
        self.console.print(Panel.fit("CryptoAgentForecaster Crew Initialized", style="bold green"))
    
    def display_configuration_status(self) -> bool:
        """Check and display configuration status."""
        self.console.print("\nConfiguration Status:", style="bold")
        
        # Check LLM providers
        providers = LLMFactory.get_available_providers()
        
        provider_table = Table(title="LLM Providers")
        provider_table.add_column("Provider", style=TABLE_STYLES["provider_column"])
        provider_table.add_column("Status", style=TABLE_STYLES["status_column"])
        provider_table.add_column("Model", style=TABLE_STYLES["model_column"])
        
        provider_table.add_row(
            "OpenAI", 
            "✅ Configured" if Config.OPENAI_API_KEY else "❌ Not configured",
            "gpt-5-nano / gpt-4o-mini / gpt-4o"
        )
        provider_table.add_row(
            "Anthropic", 
            "✅ Configured" if Config.ANTHROPIC_API_KEY else "❌ Not configured",
            "claude-3-5-sonnet-20241022"
        )
        provider_table.add_row(
            "Google", 
            "✅ Configured" if Config.GOOGLE_API_KEY else "❌ Not configured",
            "gemini-2.5-flash / gemini-1.5-pro"
        )
        
        self.console.print(provider_table)
        
        # Check CoinGecko
        coingecko_status = "✅ Configured (Pro)" if Config.COINGECKO_API_KEY else "⚠️  Free tier (limited)"
        self.console.print(f"CoinGecko API: {coingecko_status}")
        
        if not providers:
            self.console.print("❌ No LLM providers configured! Please set up API keys.", style="bold red")
            self.console.print("Create a .env file with your API keys. See env_example for template.")
            return False
        
        self.console.print(f"✅ Ready with {len(providers)} LLM provider(s): {', '.join(providers)}", style="bold green")
        return True
    
    def display_setup_instructions(self) -> None:
        """Display setup instructions."""
        self.console.print("\nSetup Instructions:", style="bold")
        for i, instruction in enumerate(SETUP_INSTRUCTIONS, 1):
            self.console.print(f"{i}. {instruction}")
        
        self.console.print("\nGet API Keys:")
        for provider, url in API_KEY_URLS.items():
            self.console.print(f"• {provider}: {url}")
    
    def display_forecast_configuration(self, crypto: str, horizon: str, provider: str, model: str) -> None:
        """Display forecast configuration."""
        self.console.print(f"\nForecast Configuration:")
        self.console.print(f"• Cryptocurrency: {crypto.upper()}")
        self.console.print(f"• Time Horizon: {horizon}")
        self.console.print(f"• LLM Provider: {provider}")
        self.console.print(f"• Model: {model}")
    
    def display_crypto_list(self) -> None:
        """Display list of popular cryptocurrencies."""
        self.console.print("Popular cryptocurrencies available for analysis:\n")
        
        crypto_table = Table(title="Popular Cryptocurrencies")
        crypto_table.add_column("ID (use this)", style=TABLE_STYLES["provider_column"])
        crypto_table.add_column("Name", style=TABLE_STYLES["status_column"])
        crypto_table.add_column("Symbol", style=TABLE_STYLES["model_column"])
        
        for crypto_id, name, symbol in POPULAR_CRYPTOS:
            crypto_table.add_row(crypto_id, name, symbol)
        
        self.console.print(crypto_table)
        self.console.print("\nUse the ID column when running forecasts")
        self.console.print("   Example: crypto-agent-forecaster forecast bitcoin")
    
    def display_debug_info(self, debug_info: Dict[str, Any]) -> None:
        """Display debug information."""
        self.console.print("Environment & Configuration Debug Information\n", style="bold blue")
        
        # Display environment status
        self.console.print("Environment Status:", style="bold")
        env_table = Table(title="Environment Configuration")
        env_table.add_column("Setting", style=TABLE_STYLES["provider_column"])
        env_table.add_column("Status", style=TABLE_STYLES["status_column"])
        env_table.add_column("Value/Details", style=TABLE_STYLES["model_column"])
        
        env_table.add_row(
            "Environment Loaded", 
            "✅ Yes" if debug_info["environment_loaded"] else "❌ No",
            "dotenv loading status"
        )
        env_table.add_row(
            "Default Provider", 
            "✅ Set" if debug_info["default_provider"] else "❌ Not set",
            str(debug_info["default_provider"])
        )
        env_table.add_row(
            "Default Model", 
            "✅ Set" if debug_info["default_model"] else "❌ Not set",
            str(debug_info["default_model"])
        )
        
        self.console.print(env_table)
        
        # Display API key status
        self.console.print("\nAPI Key Configuration:", style="bold")
        api_table = Table(title="API Key Status")
        api_table.add_column("Provider", style=TABLE_STYLES["provider_column"])
        api_table.add_column("Configured", style=TABLE_STYLES["status_column"])
        api_table.add_column("In Environment", style=TABLE_STYLES["model_column"])
        api_table.add_column("Status", style=TABLE_STYLES["green_style"])
        
        for provider, configured in debug_info["api_keys_configured"].items():
            env_present = debug_info["env_vars_present"].get(f"{provider.upper()}_API_KEY", False)
            status = "✅ Ready" if configured and env_present else "❌ Issue"
            
            api_table.add_row(
                provider.title(),
                "✅ Yes" if configured else "❌ No",
                "✅ Yes" if env_present else "❌ No", 
                status
            )
        
        self.console.print(api_table)
    
    def display_agent_configuration_status(self) -> None:
        """Display agent LLM configuration status."""
        self.console.print("\nAgent LLM Configuration:", style="bold")
        
        for agent_type in AGENT_TYPES:
            is_valid = Config.validate_llm_config(agent_type)
            if not is_valid:
                self.console.print(f"❌ {agent_type} agent has configuration issues", style="red")
    
    def display_available_providers(self) -> None:
        """Display available LLM providers."""
        available_providers = LLMFactory.get_available_providers()
        self.console.print(f"\n✅ Available LLM Providers: {', '.join(available_providers) if available_providers else 'None'}")
        
        if not available_providers:
            self.console.print("❌ No LLM providers available! Please configure at least one API key.", style="bold red")
            self.console.print("💡 Use 'crypto-agent-forecaster config' for setup instructions.")
    
    def display_troubleshooting_tips(self) -> None:
        """Display common troubleshooting tips."""
        self.console.print("\nCommon Issues & Fixes:", style="bold")
        for tip in TROUBLESHOOTING_TIPS:
            self.console.print(f"• {tip}")
    
    def display_models_info(self) -> None:
        """Display available LLM models and their specifications."""
        self.console.print("\nLLM Model Information & Recommendations", style="bold")
        
        providers = LLMFactory.get_available_providers()
        
        for provider in providers:
            self.console.print(f"\n {provider.upper()} Models:", style="bold cyan")
            
            models_table = Table(title=f"{provider.capitalize()} Model Specifications")
            models_table.add_column("Model", style=TABLE_STYLES["model_column"])
            models_table.add_column("Max Tokens", style=TABLE_STYLES["provider_column"])
            models_table.add_column("Cost/1K Input", style=TABLE_STYLES["green_style"])
            models_table.add_column("Cost/1K Output", style=TABLE_STYLES["red_style"])
            
            model_specs = LLMFactory.MODEL_SPECS.get(provider, {})
            for model_name, specs in model_specs.items():
                cost_info = specs.get("cost_per_1k_tokens", {})
                models_table.add_row(
                    model_name,
                    f"{specs.get('max_tokens', 'Unknown'):,}",
                    f"${cost_info.get('input', 0):.6f}",
                    f"${cost_info.get('output', 0):.6f}"
                )
            
            self.console.print(models_table)
        
        # Display task recommendations
        self.console.print("\nTask-Specific Model Recommendations:", style="bold")
        
        tasks = ["sentiment_analysis", "technical_analysis", "multimodal_fusion", "cost_optimized"]
        
        rec_table = Table(title="Recommended Models by Task")
        rec_table.add_column("Task", style=TABLE_STYLES["provider_column"])
        rec_table.add_column("Provider", style=TABLE_STYLES["model_column"]) 
        rec_table.add_column("Model", style=TABLE_STYLES["green_style"])
        rec_table.add_column("Reason", style=TABLE_STYLES["status_column"])
        
        for task in tasks:
            rec = LLMFactory.get_recommended_model_for_task(task)
            rec_table.add_row(
                task.replace("_", " ").title(),
                rec["provider"].capitalize(),
                rec["model"],
                rec["reason"]
            )
        
        self.console.print(rec_table)
        
        self.console.print("\nUsage Tips:", style="bold yellow")
        self.console.print("• Use --provider and --model flags to specify different models")
        self.console.print("• Claude models excel at nuanced sentiment analysis")
        self.console.print("• GPT models are strong for structured technical analysis") 
        self.console.print("• Gemini models have the largest context windows")
        self.console.print("• Consider cost vs. performance for your use case")
    
    def display_help_overview(self) -> None:
        """Display quick help and usage examples."""
        self.console.print("Quick Reference Guide", style="bold blue")
        self.console.print("\nThis overview shows the most common usage patterns.")
        self.console.print("Use the commands above to get started quickly!")
        
        self.console.print("\nNext Steps:", style="bold yellow")
        for i, step in enumerate(NEXT_STEPS, 1):
            self.console.print(f"{i}. {step}")
        
        self.console.print("\nFor detailed help on any command:")
        self.console.print("crypto-agent-forecaster COMMAND --help")
    
    def display_success(self, message: str) -> None:
        """Display success message."""
        self.console.print(f"\n✅ {message}", style="bold green")
    
    def display_error(self, message: str) -> None:
        """Display error message."""
        self.console.print(f"\n❌ {message}", style="bold red")
    
    def display_warning(self, message: str) -> None:
        """Display warning message."""
        self.console.print(f"\n⚠️ {message}", style="bold yellow")
    
    def print(self, text: str, style: str = None) -> None:
        """Print text with optional styling."""
        self.console.print(text, style=style) 