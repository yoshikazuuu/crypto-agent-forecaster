#!/usr/bin/env python3
"""
Crypto Agent Forecaster Validation CLI

Simple command-line interface for running validation tests and generating reports.
"""

import asyncio
import sys
from pathlib import Path
import typer
from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from validation.validator import CryptoValidator
from validation.analytics import ValidationAnalytics
from validation.vps_deployment import VPSDeploymentManager, install_dependencies

app = typer.Typer(
    name="crypto-validator",
    help="🔮 Crypto Agent Forecaster Validation Toolkit",
    rich_markup_mode="rich"
)
console = Console()


@app.command()
def live(
    duration: int = typer.Option(24, "--duration", "-d", help="Duration in hours"),
    interval: int = typer.Option(1, "--interval", "-i", help="Forecast interval in hours"),
    coins: List[str] = typer.Option(["bitcoin"], "--coins", "-c", help="Cryptocurrencies to test"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    🔴 Run live validation by making real forecasts and tracking their accuracy.
    
    This will make actual forecasts using your main forecasting tool and then wait
    to check if they were correct. Perfect for testing real-world performance.
    
    Examples:
        crypto-validator live --duration 24 --interval 1 --coins bitcoin ethereum
        crypto-validator live -d 6 -i 2 -c solana cardano
    """
    console.print(Panel.fit(
        f"🔴 Starting Live Validation\n"
        f"Duration: {duration} hours\n"
        f"Interval: {interval} hours\n"
        f"Coins: {', '.join(coins)}",
        title="Live Validation",
        style="bold red"
    ))
    
    async def run_live():
        validator = CryptoValidator()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running live validation...", total=None)
            
            try:
                metrics = await validator.run_live_validation(
                    duration_hours=duration,
                    interval_hours=interval,
                    coins=coins
                )
                
                progress.update(task, description="✅ Live validation completed!")
                
                # Convert ValidationMetrics to dictionary format for display
                metrics_dict = validator._metrics_to_dict(metrics) if metrics else {}
                
                # Display results
                _display_metrics(metrics_dict)
                
            except KeyboardInterrupt:
                progress.update(task, description="⏹️ Stopped by user")
                console.print("Live validation stopped by user", style="yellow")
            except Exception as e:
                progress.update(task, description="❌ Error occurred")
                console.print(f"Error: {e}", style="red")
    
    asyncio.run(run_live())


@app.command()
def backtest(
    days: int = typer.Option(30, "--days", "-d", help="Days of historical data"),
    coins: List[str] = typer.Option(
        ["bitcoin", "ethereum", "solana", "cardano", "polygon"],
        "--coins", "-c", 
        help="Cryptocurrencies to test"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    📊 Run backtesting validation using historical data.
    
    This simulates forecasts using historical data to quickly test performance
    without waiting for real-time results.
    
    Examples:
        crypto-validator backtest --days 30 --coins bitcoin ethereum
        crypto-validator backtest -d 7 -c solana
    """
    console.print(Panel.fit(
        f"📊 Starting Backtesting Validation\n"
        f"Historical days: {days}\n"
        f"Coins: {', '.join(coins)}",
        title="Backtesting Validation",
        style="bold blue"
    ))
    
    async def run_backtest():
        validator = CryptoValidator()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running backtesting...", total=None)
            
            try:
                metrics = await validator.run_backtesting_validation(
                    days_back=days,
                    coins=coins
                )
                
                progress.update(task, description="✅ Backtesting completed!")
                
                # Convert ValidationMetrics to dictionary format for display
                metrics_dict = validator._metrics_to_dict(metrics) if metrics else {}
                
                # Display results
                _display_metrics(metrics_dict)
                
            except Exception as e:
                progress.update(task, description="❌ Error occurred")
                console.print(f"Error: {e}", style="red")
    
    asyncio.run(run_backtest())


@app.command()
def report(
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    open_browser: bool = typer.Option(True, "--open", help="Open report in browser")
):
    """
    📋 Generate comprehensive validation report from existing results.
    
    Analyzes all validation results and creates a detailed HTML report with charts,
    statistics, and insights.
    
    Examples:
        crypto-validator report
        crypto-validator report --output ./reports --no-open
    """
    console.print(Panel.fit(
        "📋 Generating Comprehensive Report",
        title="Report Generation",
        style="bold green"
    ))
    
    try:
        # Initialize analytics
        results_dir = output_dir or "validation_results"
        analytics = ValidationAnalytics(results_dir)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating report...", total=None)
            
            # Generate report
            report_file = analytics.generate_comprehensive_report()
            csv_file = analytics.export_metrics_csv()
            
            progress.update(task, description="✅ Report generated!")
            
            if report_file:
                console.print(f"✅ HTML Report: {report_file}", style="green")
                
                if open_browser:
                    import webbrowser
                    webbrowser.open(f"file://{Path(report_file).absolute()}")
                    console.print("📖 Report opened in browser", style="blue")
            
            if csv_file:
                console.print(f"📊 CSV Export: {csv_file}", style="green")
            
            if not report_file and not csv_file:
                console.print("⚠️ No validation results found", style="yellow")
    
    except Exception as e:
        console.print(f"Error generating report: {e}", style="red")


@app.command()
def deploy(
    config_file: str = typer.Option("vps_config.ini", "--config", "-c", help="Configuration file"),
    install_deps: bool = typer.Option(False, "--install-deps", help="Install dependencies first"),
    create_service: bool = typer.Option(False, "--create-service", help="Create systemd service")
):
    """
    🚀 Deploy continuous validation to VPS.
    
    Sets up and runs continuous validation on a VPS server with automated reporting,
    monitoring, and error recovery.
    
    Examples:
        crypto-validator deploy --install-deps --create-service
        crypto-validator deploy --config my_config.ini
    """
    console.print(Panel.fit(
        "🚀 VPS Deployment Manager",
        title="VPS Deployment",
        style="bold purple"
    ))
    
    try:
        if install_deps:
            console.print("📦 Installing dependencies...")
            install_dependencies()
            console.print("✅ Dependencies installed", style="green")
        
        if create_service:
            console.print("⚙️ Creating systemd service...")
            from validation.vps_deployment import create_systemd_service
            if create_systemd_service():
                console.print("✅ Systemd service created", style="green")
            else:
                console.print("❌ Failed to create systemd service", style="red")
                return
        
        # Start deployment manager
        console.print("🔄 Starting continuous validation...")
        manager = VPSDeploymentManager(config_file)
        asyncio.run(manager.start_continuous_validation())
    
    except KeyboardInterrupt:
        console.print("\n🛑 Deployment stopped by user", style="yellow")
    except Exception as e:
        console.print(f"❌ Deployment error: {e}", style="red")


@app.command()
def status():
    """
    📊 Show validation status and recent results.
    
    Displays a summary of recent validation runs, accuracy metrics,
    and system status.
    """
    console.print(Panel.fit(
        "📊 Validation Status",
        title="Status Check",
        style="bold cyan"
    ))
    
    try:
        # Load recent results
        analytics = ValidationAnalytics()
        
        if not analytics.results_data:
            console.print("⚠️ No validation results found", style="yellow")
            console.print("\nTo get started, run one of these commands:")
            console.print("• uv run python cli.py live --duration 6 --coins bitcoin")
            console.print("• uv run python cli.py backtest --days 30")
            return
        
        # Show summary
        _show_status_summary(analytics)
    
    except Exception as e:
        console.print(f"Error checking status: {e}", style="red")


def _display_metrics(metrics):
    """Display validation metrics in a nice format"""
    if not metrics or not isinstance(metrics, dict):
        console.print("⚠️ No metrics available", style="yellow")
        return
    
    # Create metrics table
    table = Table(title="📊 Validation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    # Add key metrics
    if 'accuracy_percentage' in metrics:
        accuracy_color = "green" if metrics['accuracy_percentage'] > 50 else "red"
        table.add_row("Overall Accuracy", f"[{accuracy_color}]{metrics['accuracy_percentage']:.1f}%[/{accuracy_color}]")
    
    if 'total_predictions' in metrics:
        table.add_row("Total Predictions", str(metrics['total_predictions']))
    
    if 'correct_predictions' in metrics:
        table.add_row("Correct Predictions", str(metrics['correct_predictions']))
    
    if 'average_return' in metrics:
        return_color = "green" if metrics['average_return'] > 0 else "red"
        table.add_row("Average Return", f"[{return_color}]{metrics['average_return']:.2f}%[/{return_color}]")
    
    if 'sharpe_ratio' in metrics:
        sharpe_color = "green" if metrics['sharpe_ratio'] > 0.5 else "red"
        table.add_row("Sharpe Ratio", f"[{sharpe_color}]{metrics['sharpe_ratio']:.2f}[/{sharpe_color}]")
    
    if 'win_rate' in metrics:
        win_color = "green" if metrics['win_rate'] > 50 else "red"
        table.add_row("Win Rate", f"[{win_color}]{metrics['win_rate']:.1f}%[/{win_color}]")
    
    console.print(table)


def _show_status_summary(analytics: ValidationAnalytics):
    """Show status summary"""
    
    # Combine all results
    all_forecasts = []
    for result_set in analytics.results_data:
        for forecast in result_set.get('results', []):
            forecast['validation_type'] = result_set.get('validation_type', 'unknown')
            all_forecasts.append(forecast)
    
    if not all_forecasts:
        console.print("📊 No forecast data found", style="yellow")
        return
    
    # Calculate summary stats
    total_predictions = len(all_forecasts)
    correct_predictions = sum(1 for f in all_forecasts if f.get('accuracy') is True)
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    # Recent predictions (last 10)
    recent_forecasts = sorted(all_forecasts, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
    
    # Status summary
    console.print(f"📊 Total Predictions: {total_predictions}")
    console.print(f"✅ Correct: {correct_predictions}")
    
    accuracy_color = "green" if accuracy > 50 else "red"
    console.print(f"🎯 Accuracy: [{accuracy_color}]{accuracy:.1f}%[/{accuracy_color}]")
    
    # Recent forecasts table
    if recent_forecasts:
        console.print("\n📋 Recent Forecasts:")
        
        table = Table()
        table.add_column("Time", style="dim")
        table.add_column("Crypto", style="cyan")
        table.add_column("Direction", style="white")
        table.add_column("Result", style="white")
        
        for forecast in recent_forecasts[:5]:  # Show only last 5
            timestamp = forecast.get('timestamp', 'Unknown')[:16]  # Show date and hour
            crypto = forecast.get('crypto', 'Unknown')
            direction = forecast.get('forecast_direction', 'Unknown')
            accuracy = forecast.get('accuracy')
            
            if accuracy is True:
                result = "✅ Correct"
            elif accuracy is False:
                result = "❌ Wrong"
            else:
                result = "⏳ Pending"
            
            table.add_row(timestamp, crypto, direction, result)
        
        console.print(table)


@app.command()
def full_test():
    """
    🔥 Run comprehensive 6-hour validation test with all supported cryptocurrencies.
    
    Tests Bitcoin, Ethereum, Solana, Cardano, Polygon, Chainlink, Avalanche, 
    Polkadot, Uniswap, and Litecoin for 6 hours with 1-hour intervals.
    """
    # All supported coins from the validator
    all_coins = [
        "bitcoin", "ethereum", "solana", "cardano", "polygon",
        "chainlink", "avalanche-2", "polkadot", "uniswap", "litecoin"
    ]
    
    console.print(Panel.fit(
        f"🔥 Comprehensive 6-Hour Multi-Coin Validation\n"
        f"Duration: 6 hours\n"
        f"Interval: 1 hour\n"
        f"Testing {len(all_coins)} cryptocurrencies:\n"
        f"• {', '.join(all_coins[:5])}\n"
        f"• {', '.join(all_coins[5:])}",
        title="Full Validation Test",
        style="bold magenta"
    ))
    
    console.print("\n⚠️  [yellow]This test will take 6 hours to complete![/yellow]")
    console.print("💡 [blue]Tip: Use 'backtest' for instant results with historical data[/blue]\n")
    
    async def run_full_test():
        validator = CryptoValidator()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running comprehensive validation...", total=None)
            
            try:
                metrics = await validator.run_live_validation(
                    duration_hours=6,
                    interval_hours=1,
                    coins=all_coins
                )
                
                progress.update(task, description="✅ Comprehensive validation completed!")
                
                # Convert ValidationMetrics to dictionary format for display
                metrics_dict = validator._metrics_to_dict(metrics) if metrics else {}
                
                # Display results
                _display_metrics(metrics_dict)
                
                # Show coin-specific summary
                console.print("\n🎯 [bold]Test Summary:[/bold]")
                console.print(f"• Total coins tested: {len(all_coins)}")
                console.print(f"• Test duration: 6 hours")
                console.print(f"• Forecasts per coin: 6 (every hour)")
                console.print(f"• Total forecasts made: {len(all_coins) * 6}")
                
            except KeyboardInterrupt:
                progress.update(task, description="⏹️ Stopped by user")
                console.print("Comprehensive validation stopped by user", style="yellow")
            except Exception as e:
                progress.update(task, description="❌ Error occurred")
                console.print(f"Error: {e}", style="red")
    
    asyncio.run(run_full_test())


@app.command()
def quick_test():
    """
    ⚡ Run a quick validation test with Bitcoin.
    
    Performs a 6-hour live validation with Bitcoin to quickly test the system.
    """
    console.print(Panel.fit(
        "⚡ Quick Test - 6 Hour Bitcoin Validation",
        style="bold yellow"
    ))
    
    async def run_quick_test():
        validator = CryptoValidator()
        
        try:
            console.print("🚀 Starting 6-hour Bitcoin validation...")
            metrics = await validator.run_live_validation(
                duration_hours=6,
                interval_hours=1,
                coins=["bitcoin"]
            )
            
            console.print("✅ Quick test completed!")
            # Convert ValidationMetrics to dictionary format for display
            metrics_dict = validator._metrics_to_dict(metrics) if metrics else {}
            _display_metrics(metrics_dict)
            
        except Exception as e:
            console.print(f"❌ Quick test failed: {e}", style="red")
    
    asyncio.run(run_quick_test())


@app.command()
def full_backtest(
    days: int = typer.Option(30, "--days", "-d", help="Days of historical data")
):
    """
    📊 Run comprehensive backtesting with all supported cryptocurrencies.
    
    Tests Bitcoin, Ethereum, Solana, Cardano, Polygon, Chainlink, Avalanche, 
    Polkadot, Uniswap, and Litecoin using historical data. Results are instant!
    
    Examples:
        crypto-validator full-backtest --days 30
        crypto-validator full-backtest -d 7
    """
    # All supported coins from the validator
    all_coins = [
        "bitcoin", "ethereum", "solana", "cardano", "polygon",
        "chainlink", "avalanche-2", "polkadot", "uniswap", "litecoin"
    ]
    
    console.print(Panel.fit(
        f"📊 Comprehensive Multi-Coin Backtesting\n"
        f"Historical days: {days}\n"
        f"Testing {len(all_coins)} cryptocurrencies:\n"
        f"• {', '.join(all_coins[:5])}\n"
        f"• {', '.join(all_coins[5:])}",
        title="Full Backtest Validation",
        style="bold cyan"
    ))
    
    async def run_full_backtest():
        validator = CryptoValidator()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running comprehensive backtesting...", total=None)
            
            try:
                metrics = await validator.run_backtesting_validation(
                    days_back=days,
                    coins=all_coins
                )
                
                progress.update(task, description="✅ Comprehensive backtesting completed!")
                
                # Convert ValidationMetrics to dictionary format for display
                metrics_dict = validator._metrics_to_dict(metrics) if metrics else {}
                
                # Display results
                _display_metrics(metrics_dict)
                
                # Show test summary
                console.print("\n🎯 [bold]Backtest Summary:[/bold]")
                console.print(f"• Total coins tested: {len(all_coins)}")
                console.print(f"• Historical days: {days}")
                console.print(f"• Estimated forecasts: {len(all_coins) * (days // 2)} (approx)")
                
            except Exception as e:
                progress.update(task, description="❌ Error occurred")
                console.print(f"Error: {e}", style="red")
    
    asyncio.run(run_full_backtest())


def main():
    """Main entry point"""
    console.print(Panel.fit(
        "🔮 Crypto Agent Forecaster Validation Toolkit\n"
        "Scientific validation and testing for crypto forecasting models",
        title="Welcome",
        style="bold blue"
    ))
    
    app()


if __name__ == "__main__":
    main() 