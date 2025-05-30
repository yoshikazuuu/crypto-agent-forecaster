#!/usr/bin/env python3
"""
Example Usage of Crypto Agent Forecaster Validation Framework

This script demonstrates various ways to use the validation framework
for testing your crypto forecasting models.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path so we can import the validation modules
sys.path.append(str(Path(__file__).parent.parent))

from validation.validator import CryptoValidator
from validation.analytics import ValidationAnalytics


async def example_backtesting():
    """Example: Run backtesting validation"""
    print("🔄 Running Backtesting Example")
    print("=" * 50)
    
    # Initialize validator
    validator = CryptoValidator()
    
    # Run backtesting with 30 days of data for top cryptocurrencies
    print("📊 Starting 30-day backtest for Bitcoin, Ethereum, and Solana...")
    
    metrics = await validator.run_backtesting_validation(
        days_back=30,
        coins=["bitcoin", "ethereum", "solana"]
    )
    
    # Display results
    print("\n📈 Results:")
    print(f"Total Predictions: {metrics.total_predictions}")
    print(f"Correct Predictions: {metrics.correct_predictions}")
    print(f"Accuracy: {metrics.accuracy_percentage:.1f}%")
    print(f"Average Return: {metrics.average_return:.2f}%")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2f}%")
    
    return metrics


async def example_live_validation():
    """Example: Run live validation (short duration for demo)"""
    print("\n🔴 Running Live Validation Example")
    print("=" * 50)
    
    # Initialize validator
    validator = CryptoValidator()
    
    # Run live validation for 2 hours with Bitcoin
    print("⏰ Starting 2-hour live validation with Bitcoin...")
    print("⚠️  This will make actual forecasts and wait for results!")
    
    try:
        metrics = await validator.run_live_validation(
            duration_hours=2,
            interval_hours=1,
            coins=["bitcoin"]
        )
        
        # Display results
        print("\n📈 Live Validation Results:")
        print(f"Total Predictions: {metrics.total_predictions}")
        print(f"Accuracy: {metrics.accuracy_percentage:.1f}%")
        
        return metrics
        
    except KeyboardInterrupt:
        print("\n⏹️ Live validation stopped by user")
        return None


def example_analytics():
    """Example: Generate analytics report"""
    print("\n📋 Generating Analytics Report")
    print("=" * 50)
    
    # Initialize analytics
    analytics = ValidationAnalytics()
    
    # Check if we have any results
    if not analytics.results_data:
        print("⚠️ No validation results found.")
        print("Run backtesting or live validation first to generate data.")
        return
    
    # Generate comprehensive report
    print("📊 Generating comprehensive HTML report...")
    report_file = analytics.generate_comprehensive_report()
    
    # Export CSV data
    print("📋 Exporting CSV data...")
    csv_file = analytics.export_metrics_csv()
    
    if report_file:
        print(f"✅ HTML Report generated: {report_file}")
    
    if csv_file:
        print(f"✅ CSV data exported: {csv_file}")


async def example_custom_analysis():
    """Example: Custom analysis of validation results"""
    print("\n🔍 Custom Analysis Example")
    print("=" * 50)
    
    # Load and analyze results manually
    analytics = ValidationAnalytics()
    
    if not analytics.results_data:
        print("⚠️ No validation results found for custom analysis.")
        return
    
    # Combine all forecast results
    all_forecasts = []
    for result_set in analytics.results_data:
        for forecast in result_set.get('results', []):
            forecast['validation_type'] = result_set.get('validation_type', 'unknown')
            all_forecasts.append(forecast)
    
    if not all_forecasts:
        print("📊 No forecast data found")
        return
    
    # Custom analysis
    print(f"📊 Total forecasts analyzed: {len(all_forecasts)}")
    
    # Analyze by cryptocurrency
    crypto_performance = {}
    for forecast in all_forecasts:
        crypto = forecast.get('crypto', 'unknown')
        if crypto not in crypto_performance:
            crypto_performance[crypto] = {'total': 0, 'correct': 0}
        
        crypto_performance[crypto]['total'] += 1
        if forecast.get('accuracy') is True:
            crypto_performance[crypto]['correct'] += 1
    
    print("\n📈 Performance by Cryptocurrency:")
    for crypto, stats in crypto_performance.items():
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {crypto}: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
    
    # Analyze by direction
    direction_performance = {}
    for forecast in all_forecasts:
        direction = forecast.get('forecast_direction', 'unknown')
        if direction not in direction_performance:
            direction_performance[direction] = {'total': 0, 'correct': 0}
        
        direction_performance[direction]['total'] += 1
        if forecast.get('accuracy') is True:
            direction_performance[direction]['correct'] += 1
    
    print("\n📊 Performance by Direction:")
    for direction, stats in direction_performance.items():
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {direction}: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")


async def main():
    """Run all examples"""
    print("🔮 Crypto Agent Forecaster Validation Framework Examples")
    print("=" * 60)
    print()
    
    try:
        # Example 1: Backtesting (always safe to run)
        await example_backtesting()
        
        # Example 2: Analytics (if we have data)
        example_analytics()
        
        # Example 3: Custom analysis
        await example_custom_analysis()
        
        # Example 4: Live validation (optional - commented out by default)
        # Uncomment the line below to run live validation
        # await example_live_validation()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("\nNext steps:")
        print("• Check the validation_results/ directory for generated data")
        print("• Open the HTML report in your browser")
        print("• Try the CLI tool: uv run python validation/cli.py --help")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure your crypto forecasting tool is properly configured.")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main()) 