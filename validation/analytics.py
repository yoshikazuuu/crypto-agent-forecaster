#!/usr/bin/env python3
"""
Validation Analytics and Reporting Module

Generates comprehensive reports, visualizations, and statistical analysis
of the crypto forecasting validation results.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ValidationAnalytics:
    """Analytics and reporting for validation results"""
    
    def __init__(self, results_dir: str = "validation_results"):
        self.results_dir = Path(results_dir)
        self.reports_dir = self.results_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Load all validation results
        self.results_data = self._load_all_results()
    
    def _load_all_results(self) -> List[Dict[str, Any]]:
        """Load all validation result files"""
        results = []
        
        for file_path in self.results_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['file_path'] = str(file_path)
                    results.append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return results
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive HTML report"""
        if not self.results_data:
            print("No validation results found")
            return ""
        
        # Combine all results into a DataFrame
        all_forecasts = []
        for result_set in self.results_data:
            for forecast in result_set.get('results', []):
                forecast['validation_type'] = result_set.get('validation_type', 'unknown')
                forecast['file_source'] = result_set.get('file_path', 'unknown')
                all_forecasts.append(forecast)
        
        if not all_forecasts:
            print("No forecast results found")
            return ""
        
        df = pd.DataFrame(all_forecasts)
        
        # Generate report
        report_html = self._create_html_report(df)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.reports_dir / f"validation_report_{timestamp}.html"
        
        with open(report_file, 'w') as f:
            f.write(report_html)
        
        print(f"Comprehensive report saved to: {report_file}")
        return str(report_file)
    
    def _create_html_report(self, df: pd.DataFrame) -> str:
        """Create comprehensive HTML report"""
        
        # Clean and prepare data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['accuracy'] = df['accuracy'].astype(bool)
        df['return_pct'] = pd.to_numeric(df['return_pct'], errors='coerce')
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(df)
        
        # Generate visualizations
        charts_html = self._generate_charts(df)
        
        # Create HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CryptoAgentForecaster Validation Report</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h3 {{ color: #2c3e50; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
                .metric-value {{ font-size: 2em; font-weight: bold; margin: 10px 0; }}
                .metric-label {{ font-size: 0.9em; opacity: 0.9; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .chart-container {{ margin: 30px 0; text-align: center; }}
                .positive {{ color: #27ae60; font-weight: bold; }}
                .negative {{ color: #e74c3c; font-weight: bold; }}
                .neutral {{ color: #f39c12; font-weight: bold; }}
                .summary-section {{ background-color: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔮 CryptoAgentForecaster Validation Report</h1>
                <p style="text-align: center; color: #7f8c8d; font-style: italic;">
                    Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
                
                <div class="summary-section">
                    <h2>📊 Executive Summary</h2>
                    {self._create_executive_summary(summary_stats, df)}
                </div>
                
                <h2>📈 Key Performance Metrics</h2>
                <div class="metric-grid">
                    {self._create_metric_cards(summary_stats)}
                </div>
                
                <h2>📋 Detailed Statistics</h2>
                {self._create_statistics_tables(summary_stats, df)}
                
                <h2>📊 Visualizations</h2>
                {charts_html}
                
                <h2>📋 Forecast History</h2>
                {self._create_forecast_table(df)}
                
                <h2>🔍 Analysis & Insights</h2>
                {self._create_insights_section(summary_stats, df)}
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _calculate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics"""
        stats = {}
        
        # Basic accuracy metrics
        total_predictions = len(df)
        correct_predictions = df['accuracy'].sum()
        accuracy_percentage = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        stats['total_predictions'] = total_predictions
        stats['correct_predictions'] = correct_predictions
        stats['accuracy_percentage'] = accuracy_percentage
        
        # Accuracy by direction
        direction_accuracy = df.groupby('forecast_direction')['accuracy'].agg(['count', 'sum', 'mean']).reset_index()
        direction_accuracy['accuracy_pct'] = direction_accuracy['mean'] * 100
        stats['direction_accuracy'] = direction_accuracy
        
        # Accuracy by confidence
        confidence_accuracy = df.groupby('confidence')['accuracy'].agg(['count', 'sum', 'mean']).reset_index()
        confidence_accuracy['accuracy_pct'] = confidence_accuracy['mean'] * 100
        stats['confidence_accuracy'] = confidence_accuracy
        
        # Accuracy by crypto
        crypto_accuracy = df.groupby('crypto')['accuracy'].agg(['count', 'sum', 'mean']).reset_index()
        crypto_accuracy['accuracy_pct'] = crypto_accuracy['mean'] * 100
        stats['crypto_accuracy'] = crypto_accuracy
        
        # Financial metrics
        returns = df['return_pct'].dropna()
        if len(returns) > 0:
            stats['average_return'] = returns.mean()
            stats['median_return'] = returns.median()
            stats['std_return'] = returns.std()
            stats['sharpe_ratio'] = returns.mean() / returns.std() if returns.std() > 0 else 0
            stats['max_return'] = returns.max()
            stats['min_return'] = returns.min()
            
            # Win/loss metrics
            winning_trades = returns[returns > 0]
            losing_trades = returns[returns < 0]
            
            stats['winning_trades'] = len(winning_trades)
            stats['losing_trades'] = len(losing_trades)
            stats['win_rate'] = len(winning_trades) / len(returns) * 100 if len(returns) > 0 else 0
            stats['avg_winning_return'] = winning_trades.mean() if len(winning_trades) > 0 else 0
            stats['avg_losing_return'] = losing_trades.mean() if len(losing_trades) > 0 else 0
            
            # Risk metrics
            cumulative_returns = returns.cumsum()
            peak = cumulative_returns.expanding(min_periods=1).max()
            drawdown = cumulative_returns - peak
            stats['max_drawdown'] = drawdown.min()
            
            # Value at Risk (95%)
            stats['var_95'] = np.percentile(returns, 5)
            
        else:
            # Default values if no return data
            for key in ['average_return', 'median_return', 'std_return', 'sharpe_ratio', 
                       'max_return', 'min_return', 'winning_trades', 'losing_trades', 
                       'win_rate', 'avg_winning_return', 'avg_losing_return', 'max_drawdown', 'var_95']:
                stats[key] = 0
        
        # Time-based analysis
        if 'timestamp' in df.columns:
            df_time = df.copy()
            df_time['hour'] = df_time['timestamp'].dt.hour
            df_time['day_of_week'] = df_time['timestamp'].dt.day_name()
            
            hourly_accuracy = df_time.groupby('hour')['accuracy'].mean() * 100
            daily_accuracy = df_time.groupby('day_of_week')['accuracy'].mean() * 100
            
            stats['hourly_accuracy'] = hourly_accuracy.to_dict()
            stats['daily_accuracy'] = daily_accuracy.to_dict()
        
        return stats
    
    def _create_executive_summary(self, stats: Dict[str, Any], df: pd.DataFrame) -> str:
        """Create executive summary"""
        
        # Determine performance level
        accuracy = stats['accuracy_percentage']
        if accuracy >= 60:
            performance_level = "Excellent"
            performance_color = "#27ae60"
        elif accuracy >= 50:
            performance_level = "Good"
            performance_color = "#f39c12"
        else:
            performance_level = "Needs Improvement"
            performance_color = "#e74c3c"
        
        # Calculate trend
        if len(df) > 10:
            recent_accuracy = df.tail(10)['accuracy'].mean() * 100
            trend = "improving" if recent_accuracy > accuracy else "declining"
        else:
            trend = "stable"
        
        summary = f"""
        <p><strong>Overall Performance:</strong> <span style="color: {performance_color};">{performance_level}</span> 
        ({accuracy:.1f}% accuracy across {stats['total_predictions']} predictions)</p>
        
        <p><strong>Financial Performance:</strong> Average return of {stats['average_return']:.2f}% per prediction 
        with a Sharpe ratio of {stats['sharpe_ratio']:.2f}</p>
        
        <p><strong>Risk Profile:</strong> Maximum drawdown of {stats['max_drawdown']:.2f}% 
        with 95% VaR at {stats['var_95']:.2f}%</p>
        
        <p><strong>Recent Trend:</strong> Performance appears to be {trend} based on recent predictions</p>
        
        <p><strong>Best Performing Asset:</strong> {stats['crypto_accuracy'].loc[stats['crypto_accuracy']['accuracy_pct'].idxmax(), 'crypto']} 
        ({stats['crypto_accuracy']['accuracy_pct'].max():.1f}% accuracy)</p>
        """
        
        return summary
    
    def _create_metric_cards(self, stats: Dict[str, Any]) -> str:
        """Create metric cards HTML"""
        cards = f"""
        <div class="metric-card">
            <div class="metric-value">{stats['accuracy_percentage']:.1f}%</div>
            <div class="metric-label">Overall Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{stats['total_predictions']}</div>
            <div class="metric-label">Total Predictions</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{stats['average_return']:.2f}%</div>
            <div class="metric-label">Avg Return</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{stats['sharpe_ratio']:.2f}</div>
            <div class="metric-label">Sharpe Ratio</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{stats['win_rate']:.1f}%</div>
            <div class="metric-label">Win Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{stats['max_drawdown']:.2f}%</div>
            <div class="metric-label">Max Drawdown</div>
        </div>
        """
        return cards
    
    def _create_statistics_tables(self, stats: Dict[str, Any], df: pd.DataFrame) -> str:
        """Create detailed statistics tables"""
        
        # Direction accuracy table
        direction_table = stats['direction_accuracy'].to_html(
            columns=['forecast_direction', 'count', 'sum', 'accuracy_pct'],
            index=False,
            table_id="direction-table",
            classes="table",
            escape=False,
            formatters={'accuracy_pct': lambda x: f"{x:.1f}%"}
        )
        
        # Confidence accuracy table  
        confidence_table = stats['confidence_accuracy'].to_html(
            columns=['confidence', 'count', 'sum', 'accuracy_pct'],
            index=False,
            table_id="confidence-table", 
            classes="table",
            escape=False,
            formatters={'accuracy_pct': lambda x: f"{x:.1f}%"}
        )
        
        # Crypto accuracy table
        crypto_table = stats['crypto_accuracy'].to_html(
            columns=['crypto', 'count', 'sum', 'accuracy_pct'],
            index=False,
            table_id="crypto-table",
            classes="table", 
            escape=False,
            formatters={'accuracy_pct': lambda x: f"{x:.1f}%"}
        )
        
        tables_html = f"""
        <h3>Accuracy by Forecast Direction</h3>
        {direction_table}
        
        <h3>Accuracy by Confidence Level</h3>
        {confidence_table}
        
        <h3>Accuracy by Cryptocurrency</h3>
        {crypto_table}
        """
        
        return tables_html
    
    def _generate_charts(self, df: pd.DataFrame) -> str:
        """Generate visualization charts"""
        charts_html = ""
        
        # Accuracy over time
        if 'timestamp' in df.columns and len(df) > 1:
            fig_accuracy = self._create_accuracy_over_time_chart(df)
            charts_html += f'<div class="chart-container">{fig_accuracy}</div>'
        
        # Return distribution
        returns = df['return_pct'].dropna()
        if len(returns) > 0:
            fig_returns = self._create_return_distribution_chart(returns)
            charts_html += f'<div class="chart-container">{fig_returns}</div>'
        
        # Accuracy by crypto
        if len(df['crypto'].unique()) > 1:
            fig_crypto = self._create_crypto_accuracy_chart(df)
            charts_html += f'<div class="chart-container">{fig_crypto}</div>'
        
        return charts_html
    
    def _create_accuracy_over_time_chart(self, df: pd.DataFrame) -> str:
        """Create accuracy over time chart"""
        # Create rolling accuracy
        df_sorted = df.sort_values('timestamp')
        df_sorted['rolling_accuracy'] = df_sorted['accuracy'].rolling(window=10, min_periods=1).mean() * 100
        
        plt.figure(figsize=(12, 6))
        plt.plot(df_sorted['timestamp'], df_sorted['rolling_accuracy'], linewidth=2, color='#3498db')
        plt.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random Chance (50%)')
        plt.title('Accuracy Over Time (10-prediction rolling average)', fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save and return HTML
        chart_file = self.reports_dir / 'accuracy_over_time.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f'<img src="{chart_file.name}" alt="Accuracy Over Time" style="max-width: 100%;">'
    
    def _create_return_distribution_chart(self, returns: pd.Series) -> str:
        """Create return distribution chart"""
        plt.figure(figsize=(12, 6))
        
        # Histogram
        plt.hist(returns, bins=30, alpha=0.7, color='#2ecc71', edgecolor='black')
        plt.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
        plt.axvline(returns.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {returns.median():.2f}%')
        
        plt.title('Distribution of Returns', fontsize=14, fontweight='bold')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save and return HTML
        chart_file = self.reports_dir / 'return_distribution.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f'<img src="{chart_file.name}" alt="Return Distribution" style="max-width: 100%;">'
    
    def _create_crypto_accuracy_chart(self, df: pd.DataFrame) -> str:
        """Create cryptocurrency accuracy comparison chart"""
        crypto_stats = df.groupby('crypto')['accuracy'].agg(['count', 'mean']).reset_index()
        crypto_stats = crypto_stats[crypto_stats['count'] >= 3]  # Only show cryptos with 3+ predictions
        crypto_stats['accuracy_pct'] = crypto_stats['mean'] * 100
        crypto_stats = crypto_stats.sort_values('accuracy_pct', ascending=True)
        
        plt.figure(figsize=(12, 6))
        bars = plt.barh(crypto_stats['crypto'], crypto_stats['accuracy_pct'], color='#9b59b6')
        plt.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='Random Chance (50%)')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center')
        
        plt.title('Accuracy by Cryptocurrency', fontsize=14, fontweight='bold')
        plt.xlabel('Accuracy (%)')
        plt.ylabel('Cryptocurrency')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        # Save and return HTML
        chart_file = self.reports_dir / 'crypto_accuracy.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f'<img src="{chart_file.name}" alt="Crypto Accuracy Comparison" style="max-width: 100%;">'
    
    def _create_forecast_table(self, df: pd.DataFrame) -> str:
        """Create forecast history table"""
        # Show only recent 50 forecasts
        recent_df = df.sort_values('timestamp', ascending=False).head(50)
        
        # Format the dataframe for display
        display_df = recent_df[['timestamp', 'crypto', 'forecast_direction', 'confidence', 
                               'return_pct', 'accuracy']].copy()
        
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['return_pct'] = display_df['return_pct'].round(2)
        display_df['accuracy'] = display_df['accuracy'].map({True: '✅', False: '❌'})
        
        return display_df.to_html(index=False, classes="table", escape=False)
    
    def _create_insights_section(self, stats: Dict[str, Any], df: pd.DataFrame) -> str:
        """Create insights and recommendations section"""
        
        insights = []
        
        # Accuracy insights
        if stats['accuracy_percentage'] > 55:
            insights.append("✅ <strong>Strong Performance:</strong> The model shows above-random accuracy, indicating genuine predictive power.")
        elif stats['accuracy_percentage'] > 50:
            insights.append("⚠️ <strong>Marginal Performance:</strong> The model shows slight edge over random chance.")
        else:
            insights.append("❌ <strong>Poor Performance:</strong> The model is performing worse than random chance.")
        
        # Direction bias insights
        direction_stats = stats['direction_accuracy']
        if not direction_stats.empty:
            best_direction = direction_stats.loc[direction_stats['accuracy_pct'].idxmax()]
            worst_direction = direction_stats.loc[direction_stats['accuracy_pct'].idxmin()]
            
            if best_direction['accuracy_pct'] - worst_direction['accuracy_pct'] > 10:
                insights.append(f"📊 <strong>Direction Bias:</strong> The model performs significantly better on {best_direction['forecast_direction']} predictions ({best_direction['accuracy_pct']:.1f}%) vs {worst_direction['forecast_direction']} ({worst_direction['accuracy_pct']:.1f}%).")
        
        # Financial performance insights
        if stats['sharpe_ratio'] > 1:
            insights.append(f"💰 <strong>Excellent Risk-Adjusted Returns:</strong> Sharpe ratio of {stats['sharpe_ratio']:.2f} indicates strong risk-adjusted performance.")
        elif stats['sharpe_ratio'] > 0.5:
            insights.append(f"💼 <strong>Good Risk-Adjusted Returns:</strong> Sharpe ratio of {stats['sharpe_ratio']:.2f} shows decent performance.")
        else:
            insights.append(f"⚠️ <strong>Poor Risk-Adjusted Returns:</strong> Low Sharpe ratio of {stats['sharpe_ratio']:.2f} suggests high risk relative to returns.")
        
        # Crypto-specific insights
        crypto_stats = stats['crypto_accuracy']
        if not crypto_stats.empty and len(crypto_stats) > 1:
            best_crypto = crypto_stats.loc[crypto_stats['accuracy_pct'].idxmax()]
            worst_crypto = crypto_stats.loc[crypto_stats['accuracy_pct'].idxmin()]
            
            if best_crypto['accuracy_pct'] - worst_crypto['accuracy_pct'] > 15:
                insights.append(f"🎯 <strong>Asset-Specific Performance:</strong> Model works much better on {best_crypto['crypto']} ({best_crypto['accuracy_pct']:.1f}%) than {worst_crypto['crypto']} ({worst_crypto['accuracy_pct']:.1f}%).")
        
        # Recommendations
        recommendations = []
        
        if stats['accuracy_percentage'] < 55:
            recommendations.append("🔧 Consider tuning model parameters or incorporating additional data sources")
        
        if stats['max_drawdown'] < -10:
            recommendations.append("🛡️ Implement better risk management strategies to reduce drawdown")
        
        if stats['win_rate'] < 40:
            recommendations.append("📈 Focus on improving prediction quality over quantity")
        
        # Combine insights and recommendations
        insights_html = "<h3>📋 Key Insights</h3><ul>"
        for insight in insights:
            insights_html += f"<li>{insight}</li>"
        insights_html += "</ul>"
        
        if recommendations:
            insights_html += "<h3>💡 Recommendations</h3><ul>"
            for rec in recommendations:
                insights_html += f"<li>{rec}</li>"
            insights_html += "</ul>"
        
        return insights_html
    
    def export_metrics_csv(self) -> str:
        """Export detailed metrics to CSV"""
        if not self.results_data:
            return ""
        
        # Combine all results
        all_forecasts = []
        for result_set in self.results_data:
            for forecast in result_set.get('results', []):
                forecast['validation_type'] = result_set.get('validation_type', 'unknown')
                all_forecasts.append(forecast)
        
        df = pd.DataFrame(all_forecasts)
        
        # Export to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = self.reports_dir / f"validation_metrics_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"Metrics exported to: {csv_file}")
        return str(csv_file)


if __name__ == "__main__":
    analytics = ValidationAnalytics()
    
    # Generate comprehensive report
    report_file = analytics.generate_comprehensive_report()
    
    # Export CSV
    csv_file = analytics.export_metrics_csv()
    
    print(f"Analysis complete!")
    print(f"HTML Report: {report_file}")
    print(f"CSV Export: {csv_file}") 