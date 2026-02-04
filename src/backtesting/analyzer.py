"""
Thesis analysis module for statistical comparison of prediction methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from scipy import stats
import json

logger = logging.getLogger(__name__)


class ThesisAnalyzer:
    """Analyzes and compares prediction methods for thesis research."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.analysis_dir = self.data_dir / "analysis"
        self.charts_dir = self.data_dir / "charts"
        
        # Ensure directories exist (create parent directories if needed)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
    
    def load_backtest_data(self, crypto_symbol: str) -> pd.DataFrame:
        """Load and process backtest data into a comprehensive DataFrame."""
        results_file = self.data_dir / f"{crypto_symbol}_backtest_results.json"
        
        if not results_file.exists():
            logger.error(f"No backtest results found for {crypto_symbol}")
            return pd.DataFrame()
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        return self._create_analysis_dataframe(results)
    
    def analyze_prediction_accuracy(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze prediction accuracy across all methods."""
        try:
            logger.info("=== ANALYZER: Starting prediction accuracy analysis ===")
            logger.info(f"Input results keys: {list(results.keys()) if results else 'None'}")
            
            if not results:
                logger.error("ANALYZER: No results provided to analyzer")
                return {"error": "No results provided"}
            
            # Log the structure of daily_predictions
            daily_predictions = results.get("daily_predictions", {})
            logger.info(f"ANALYZER: Found {len(daily_predictions)} days of predictions")
            
            if daily_predictions:
                sample_date = list(daily_predictions.keys())[0]
                sample_data = daily_predictions[sample_date]
                logger.info(f"ANALYZER: Sample date {sample_date} has methods: {list(sample_data.keys())}")
                
                # Log a sample prediction structure
                if sample_data:
                    sample_method = list(sample_data.keys())[0]
                    sample_pred = sample_data[sample_method]
                    logger.info(f"ANALYZER: Sample prediction keys: {list(sample_pred.keys()) if isinstance(sample_pred, dict) else 'Not a dict'}")
                    logger.info(f"ANALYZER: Sample prediction success: {sample_pred.get('success') if isinstance(sample_pred, dict) else 'N/A'}")
            
            analysis = {
                "overall_stats": {},
                "method_comparison": {},
                "confidence_analysis": {},
                "temporal_analysis": {},
                "detailed_metrics": {}
            }
            
            # Create comparison dataset
            logger.info("ANALYZER: Creating analysis dataframe...")
            df = self._create_analysis_dataframe(results)
            logger.info(f"ANALYZER: Created dataframe with {len(df)} rows")
            
            if df.empty:
                logger.error("ANALYZER: Dataframe is empty - no valid data for analysis")
                return {"error": "No valid data for analysis"}
            
            logger.info(f"ANALYZER: Dataframe columns: {list(df.columns)}")
            logger.info(f"ANALYZER: Sample dataframe data:\n{df.head()}")
            
            # Overall statistics
            logger.info("ANALYZER: Calculating overall stats...")
            analysis["overall_stats"] = self._calculate_overall_stats(df)
            logger.info(f"ANALYZER: Overall stats: {analysis['overall_stats']}")
            
            # Method comparison
            logger.info("ANALYZER: Comparing methods...")
            analysis["method_comparison"] = self._compare_methods(df)
            
            # Confidence analysis
            logger.info("ANALYZER: Analyzing confidence correlation...")
            analysis["confidence_analysis"] = self._analyze_confidence_correlation(df)
            
            # Temporal analysis
            logger.info("ANALYZER: Analyzing temporal patterns...")
            analysis["temporal_analysis"] = self._analyze_temporal_patterns(df)
            
            # Detailed metrics
            logger.info("ANALYZER: Calculating detailed metrics...")
            analysis["detailed_metrics"] = self._calculate_detailed_metrics(df)
            
            # Statistical significance tests
            logger.info("ANALYZER: Performing statistical tests...")
            analysis["statistical_tests"] = self._perform_statistical_tests(df)
            
            logger.info("=== ANALYZER: Analysis completed successfully ===")
            return analysis
            
        except Exception as e:
            logger.error(f"ANALYZER: Failed to analyze prediction accuracy: {e}")
            import traceback
            logger.error(f"ANALYZER: Traceback: {traceback.format_exc()}")
            return {"error": str(e)}
    
    def _create_analysis_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create a comprehensive dataframe for analysis."""
        records = []
        
        for date, predictions in results.get("daily_predictions", {}).items():
            # Get actual outcome
            actual_movement = None
            actual_direction = None
            actual_price = None
            
            # Find any successful prediction to get actual data
            for method, prediction in predictions.items():
                if prediction.get("success") and prediction.get("actual_movement_24h") is not None:
                    actual_movement = prediction["actual_movement_24h"]
                    actual_price = prediction.get("actual_price", 0)
                    
                    # Determine actual direction based on price movement
                    if actual_movement > 1.0:  # > 1% movement threshold
                        actual_direction = "UP"
                        actual_numerical = 1
                    elif actual_movement < -1.0:  # < -1% movement threshold
                        actual_direction = "DOWN"
                        actual_numerical = -1
                    else:
                        # For small movements, assign to DOWN (conservative approach)
                        actual_direction = "DOWN"
                        actual_numerical = -1
                    break
            
            if actual_direction is None:
                continue
            
            # Create records for each method
            for method in ["full_agentic", "image_only", "sentiment_only"]:
                if method in predictions:
                    prediction = predictions[method]
                    
                    record = {
                        "date": pd.to_datetime(date),
                        "method": method,
                        "actual_direction": actual_direction,
                        "actual_movement": actual_movement,
                        "actual_price": actual_price,
                        "predicted_direction": prediction.get("predicted_direction", "FAILED"),
                        "confidence": prediction.get("confidence", "NONE"),
                        "success": prediction.get("success", False),
                        "correct_prediction": (
                            prediction.get("predicted_direction") == actual_direction 
                            if prediction.get("success") else False
                        ),
                        "execution_time": prediction.get("execution_time"),
                        "error": prediction.get("error", "")
                    }
                    
                    records.append(record)
        
        df = pd.DataFrame(records)
        
        if len(df) > 0:
            # Add additional temporal features for enhanced analysis
            df['day_of_week'] = df['date'].dt.day_name()
            df['month'] = df['date'].dt.month
            df['week_of_year'] = df['date'].dt.isocalendar().week
            df['volatility'] = abs(df['actual_movement'].fillna(0))
            
            # Add numerical direction columns for correlation analysis
            df['actual_numerical'] = df['actual_direction'].map({'UP': 1, 'DOWN': -1})
            df['predicted_numerical'] = df['predicted_direction'].map({'UP': 1, 'DOWN': -1, 'FAILED': -1})
            
            # Sort by date and method for easier analysis
            df = df.sort_values(['date', 'method']).reset_index(drop=True)
        
        return df
    
    def _calculate_overall_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall statistics across all methods."""
        return {
            "total_predictions": len(df),
            "successful_predictions": df["success"].sum(),
            "overall_accuracy": df["correct_prediction"].mean() if len(df) > 0 else 0,
            "success_rate": df["success"].mean() if len(df) > 0 else 0,
            "date_range": {
                "start": df["date"].min().strftime("%Y-%m-%d") if len(df) > 0 else None,
                "end": df["date"].max().strftime("%Y-%m-%d") if len(df) > 0 else None,
                "total_days": df["date"].nunique()
            },
            "actual_direction_distribution": df["actual_direction"].value_counts().to_dict(),
            "average_actual_movement": df["actual_movement"].mean()
        }
    
    def _compare_methods(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare accuracy across different methods."""
        method_stats = {}
        
        for method in ["full_agentic", "image_only", "sentiment_only"]:
            method_df = df[df["method"] == method]
            
            if len(method_df) > 0:
                successful_df = method_df[method_df["success"] == True]
                
                method_stats[method] = {
                    "total_attempts": len(method_df),
                    "successful_predictions": len(successful_df),
                    "success_rate": len(successful_df) / len(method_df),
                    "accuracy": successful_df["correct_prediction"].mean() if len(successful_df) > 0 else 0,
                    "correct_predictions": successful_df["correct_prediction"].sum(),
                    "direction_accuracy": {
                        direction: successful_df[successful_df["actual_direction"] == direction]["correct_prediction"].mean()
                        for direction in ["UP", "DOWN"]
                        if len(successful_df[successful_df["actual_direction"] == direction]) > 0
                    },
                    "confidence_distribution": successful_df["confidence"].value_counts().to_dict(),
                    "avg_execution_time": method_df["execution_time"].mean() if method_df["execution_time"].notna().any() else None
                }
        
        # Calculate relative performance
        accuracies = {method: stats["accuracy"] for method, stats in method_stats.items()}
        best_method = max(accuracies.keys(), key=lambda k: accuracies[k]) if accuracies else None
        
        method_stats["comparison_summary"] = {
            "best_method": best_method,
            "accuracy_ranking": sorted(accuracies.items(), key=lambda x: x[1], reverse=True),
            "accuracy_differences": {
                f"{m1}_vs_{m2}": accuracies[m1] - accuracies[m2]
                for m1 in accuracies for m2 in accuracies if m1 != m2
            }
        }
        
        return method_stats
    
    def _analyze_confidence_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation between confidence levels and accuracy."""
        confidence_analysis = {}
        
        for method in ["full_agentic", "image_only", "sentiment_only"]:
            method_df = df[(df["method"] == method) & (df["success"] == True)]
            
            if len(method_df) > 0:
                confidence_stats = {}
                
                for confidence in ["HIGH", "MEDIUM", "LOW"]:
                    conf_df = method_df[method_df["confidence"] == confidence]
                    if len(conf_df) > 0:
                        confidence_stats[confidence] = {
                            "count": len(conf_df),
                            "accuracy": conf_df["correct_prediction"].mean(),
                            "correct_predictions": conf_df["correct_prediction"].sum()
                        }
                
                confidence_analysis[method] = confidence_stats
        
        return confidence_analysis
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in predictions."""
        temporal_analysis = {}
        
        if len(df) > 0:
            df["month"] = df["date"].dt.month
            df["day_of_week"] = df["date"].dt.dayofweek
            df["week_of_year"] = df["date"].dt.isocalendar().week
            
            # Monthly analysis
            monthly_accuracy = df.groupby(["method", "month"])["correct_prediction"].mean().unstack(level=0)
            temporal_analysis["monthly_accuracy"] = monthly_accuracy.to_dict() if not monthly_accuracy.empty else {}
            
            # Day of week analysis
            dow_accuracy = df.groupby(["method", "day_of_week"])["correct_prediction"].mean().unstack(level=0)
            temporal_analysis["day_of_week_accuracy"] = dow_accuracy.to_dict() if not dow_accuracy.empty else {}
            
            # Trend analysis over time
            df_sorted = df.sort_values("date")
            rolling_accuracy = df_sorted.groupby("method")["correct_prediction"].rolling(window=30, min_periods=10).mean()
            temporal_analysis["accuracy_trend"] = rolling_accuracy.to_dict() if not rolling_accuracy.empty else {}
        
        return temporal_analysis
    
    def _calculate_detailed_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed performance metrics."""
        detailed_metrics = {}
        
        for method in ["full_agentic", "image_only", "sentiment_only"]:
            method_df = df[(df["method"] == method) & (df["success"] == True)]
            
            if len(method_df) > 0:
                # Confusion matrix components
                tp = len(method_df[(method_df["predicted_direction"] == "UP") & (method_df["actual_direction"] == "UP")])
                tn = len(method_df[(method_df["predicted_direction"] == "DOWN") & (method_df["actual_direction"] == "DOWN")])
                fp = len(method_df[(method_df["predicted_direction"] == "UP") & (method_df["actual_direction"] != "UP")])
                fn = len(method_df[(method_df["predicted_direction"] == "DOWN") & (method_df["actual_direction"] != "DOWN")])
                
                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                detailed_metrics[method] = {
                    "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
                    "balanced_accuracy": (recall + (tn / (tn + fp) if (tn + fp) > 0 else 0)) / 2
                }
        
        return detailed_metrics
    
    def _perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        statistical_tests = {}
        
        try:
            methods = ["full_agentic", "image_only", "sentiment_only"]
            
            # Chi-square test for independence
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    method1_df = df[(df["method"] == method1) & (df["success"] == True)]
                    method2_df = df[(df["method"] == method2) & (df["success"] == True)]
                    
                    if len(method1_df) > 10 and len(method2_df) > 10:
                        # Create contingency table
                        contingency = pd.crosstab(
                            pd.concat([method1_df["correct_prediction"], method2_df["correct_prediction"]]),
                            pd.concat([pd.Series([method1]*len(method1_df)), pd.Series([method2]*len(method2_df))])
                        )
                        
                        if contingency.shape == (2, 2):
                            chi2, p_value = stats.chi2_contingency(contingency)[:2]
                            
                            statistical_tests[f"{method1}_vs_{method2}"] = {
                                "test": "chi_square",
                                "chi2_statistic": chi2,
                                "p_value": p_value,
                                "significant": p_value < 0.05,
                                "contingency_table": contingency.to_dict()
                            }
            
            # McNemar's test for paired comparisons (if same dates available)
            # This would be more complex to implement and requires careful date matching
            
        except Exception as e:
            statistical_tests["error"] = str(e)
        
        return statistical_tests
    
    def generate_thesis_visualizations(self, results: Dict[str, Any], crypto_symbol: str) -> List[Path]:
        """Generate comprehensive visualizations for thesis including trading performance."""
        try:
            logger.info("=== ANALYZER: Starting chart generation ===")
            logger.info(f"CHARTS: Charts directory: {self.charts_dir}")
            logger.info(f"CHARTS: Charts directory exists: {self.charts_dir.exists()}")
            
            df = self._create_analysis_dataframe(results)
            if df.empty:
                logger.error("CHARTS: Cannot generate charts - dataframe is empty")
                return []
            
            logger.info(f"CHARTS: Using dataframe with {len(df)} rows for chart generation")
            
            chart_paths = []
            
            # 1. Overall Accuracy Comparison
            logger.info("CHARTS: Generating accuracy comparison chart...")
            chart_path = self._plot_accuracy_comparison(df, crypto_symbol)
            if chart_path:
                logger.info(f"CHARTS: Accuracy comparison saved to: {chart_path}")
            else:
                logger.error("CHARTS: Failed to generate accuracy comparison chart")
            chart_paths.append(chart_path)
            
            # 2. Confidence vs Accuracy Analysis
            logger.info("CHARTS: Generating confidence accuracy chart...")
            chart_path = self._plot_confidence_accuracy(df, crypto_symbol)
            if chart_path:
                logger.info(f"CHARTS: Confidence accuracy saved to: {chart_path}")
            else:
                logger.error("CHARTS: Failed to generate confidence accuracy chart")
            chart_paths.append(chart_path)
            
            # 3. Temporal Performance Analysis
            logger.info("CHARTS: Generating temporal analysis chart...")
            chart_path = self._plot_temporal_analysis(df, crypto_symbol)
            if chart_path:
                logger.info(f"CHARTS: Temporal analysis saved to: {chart_path}")
            else:
                logger.error("CHARTS: Failed to generate temporal analysis chart")
            chart_paths.append(chart_path)
            
            # 4. Confusion Matrix Heatmaps
            logger.info("CHARTS: Generating confusion matrices chart...")
            chart_path = self._plot_confusion_matrices(df, crypto_symbol)
            if chart_path:
                logger.info(f"CHARTS: Confusion matrices saved to: {chart_path}")
            else:
                logger.error("CHARTS: Failed to generate confusion matrices chart")
            chart_paths.append(chart_path)
            
            # 5. Method Performance Distribution
            logger.info("CHARTS: Generating performance distribution chart...")
            chart_path = self._plot_performance_distribution(df, crypto_symbol)
            if chart_path:
                logger.info(f"CHARTS: Performance distribution saved to: {chart_path}")
            else:
                logger.error("CHARTS: Failed to generate performance distribution chart")
            chart_paths.append(chart_path)
            
            # 6. Trading Performance Analysis (NEW)
            logger.info("CHARTS: Generating trading performance chart...")
            chart_path = self._plot_trading_performance(results, crypto_symbol)
            if chart_path:
                logger.info(f"CHARTS: Trading performance saved to: {chart_path}")
            else:
                logger.error("CHARTS: Failed to generate trading performance chart")
            chart_paths.append(chart_path)
            
            # 7. Portfolio Performance Over Time (NEW)
            logger.info("CHARTS: Generating portfolio performance chart...")
            chart_path = self._plot_portfolio_performance(results, crypto_symbol)
            if chart_path:
                logger.info(f"CHARTS: Portfolio performance saved to: {chart_path}")
            else:
                logger.error("CHARTS: Failed to generate portfolio performance chart")
            chart_paths.append(chart_path)
            
            # 8. Risk-Return Analysis (NEW)
            logger.info("CHARTS: Generating risk-return analysis chart...")
            chart_path = self._plot_risk_return_analysis(results, crypto_symbol)
            if chart_path:
                logger.info(f"CHARTS: Risk-return analysis saved to: {chart_path}")
            else:
                logger.error("CHARTS: Failed to generate risk-return analysis chart")
            chart_paths.append(chart_path)
            
            # 9. Stop Loss / Take Profit Analysis (NEW)
            logger.info("CHARTS: Generating stop loss/take profit chart...")
            chart_path = self._plot_stop_loss_take_profit_analysis(results, crypto_symbol)
            if chart_path:
                logger.info(f"CHARTS: Stop loss/take profit saved to: {chart_path}")
            else:
                logger.error("CHARTS: Failed to generate stop loss/take profit chart")
            chart_paths.append(chart_path)
            
            valid_charts = [path for path in chart_paths if path]
            logger.info(f"CHARTS: Generated {len(valid_charts)} valid charts out of {len(chart_paths)} attempts")
            
            return valid_charts
            
        except Exception as e:
            logger.error(f"CHARTS: Failed to generate visualizations: {e}")
            import traceback
            logger.error(f"CHARTS: Traceback: {traceback.format_exc()}")
            return []
    
    def _plot_accuracy_comparison(self, df: pd.DataFrame, crypto_symbol: str) -> Optional[Path]:
        """Plot accuracy comparison between methods."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Calculate accuracy by method
            method_accuracy = df[df["success"] == True].groupby("method")["correct_prediction"].mean()
            method_counts = df[df["success"] == True].groupby("method").size()
            
            # Create subplot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Accuracy bar plot
            bars = ax1.bar(method_accuracy.index, method_accuracy.values, alpha=0.8)
            ax1.set_title(f'Prediction Accuracy by Method - {crypto_symbol.upper()}', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Accuracy Rate', fontsize=12)
            ax1.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, method_accuracy.values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Sample size bar plot
            bars2 = ax2.bar(method_counts.index, method_counts.values, alpha=0.8, color='orange')
            ax2.set_title(f'Sample Sizes by Method - {crypto_symbol.upper()}', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Number of Successful Predictions', fontsize=12)
            
            # Add value labels
            for bar, value in zip(bars2, method_counts.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_accuracy_comparison.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to plot accuracy comparison: {e}")
            return None
    
    def _plot_confidence_accuracy(self, df: pd.DataFrame, crypto_symbol: str) -> Optional[Path]:
        """Plot confidence vs accuracy analysis."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            methods = ["full_agentic", "image_only", "sentiment_only"]
            
            for i, method in enumerate(methods):
                method_df = df[(df["method"] == method) & (df["success"] == True)]
                
                if len(method_df) > 0:
                    conf_accuracy = method_df.groupby("confidence")["correct_prediction"].mean()
                    conf_counts = method_df.groupby("confidence").size()
                    
                    # Create grouped bar chart
                    x_pos = np.arange(len(conf_accuracy))
                    bars = axes[i].bar(x_pos, conf_accuracy.values, alpha=0.8)
                    
                    axes[i].set_title(f'{method.replace("_", " ").title()}\nConfidence vs Accuracy', 
                                    fontsize=12, fontweight='bold')
                    axes[i].set_ylabel('Accuracy Rate')
                    axes[i].set_xlabel('Confidence Level')
                    axes[i].set_xticks(x_pos)
                    axes[i].set_xticklabels(conf_accuracy.index)
                    axes[i].set_ylim(0, 1)
                    
                    # Add value and count labels
                    for j, (bar, acc, count) in enumerate(zip(bars, conf_accuracy.values, conf_counts.values)):
                        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                                   f'{acc:.3f}\n(n={count})', ha='center', va='bottom', fontweight='bold')
                else:
                    axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{method.replace("_", " ").title()}\nNo Data Available')
            
            plt.suptitle(f'Confidence vs Accuracy Analysis - {crypto_symbol.upper()}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_confidence_accuracy.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to plot confidence accuracy: {e}")
            return None
    
    def _plot_temporal_analysis(self, df: pd.DataFrame, crypto_symbol: str) -> Optional[Path]:
        """Plot temporal performance analysis."""
        try:
            if len(df) == 0:
                return None
                
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Accuracy over time (rolling average)
            df_success = df[df["success"] == True].copy()
            df_success = df_success.sort_values("date")
            
            for method in ["full_agentic", "image_only", "sentiment_only"]:
                method_df = df_success[df_success["method"] == method]
                if len(method_df) > 10:
                    rolling_acc = method_df.set_index("date")["correct_prediction"].rolling(window=14, min_periods=5).mean()
                    axes[0,0].plot(rolling_acc.index, rolling_acc.values, label=method.replace("_", " ").title(), linewidth=2)
            
            axes[0,0].set_title('Accuracy Trend Over Time (14-day Rolling Average)', fontweight='bold')
            axes[0,0].set_ylabel('Accuracy Rate')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # 2. Monthly performance
            df_success["month"] = df_success["date"].dt.month
            monthly_acc = df_success.groupby(["method", "month"])["correct_prediction"].mean().unstack(level=0, fill_value=0)
            
            if not monthly_acc.empty:
                monthly_acc.plot(kind='bar', ax=axes[0,1], alpha=0.8)
                axes[0,1].set_title('Monthly Accuracy Comparison', fontweight='bold')
                axes[0,1].set_ylabel('Accuracy Rate')
                axes[0,1].set_xlabel('Month')
                axes[0,1].legend(title='Method')
                axes[0,1].tick_params(axis='x', rotation=0)
            
            # 3. Day of week performance
            df_success["day_of_week"] = df_success["date"].dt.dayofweek
            dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            dow_acc = df_success.groupby(["method", "day_of_week"])["correct_prediction"].mean().unstack(level=0, fill_value=0)
            
            if not dow_acc.empty:
                dow_acc.index = [dow_names[i] for i in dow_acc.index]
                dow_acc.plot(kind='bar', ax=axes[1,0], alpha=0.8)
                axes[1,0].set_title('Day of Week Accuracy', fontweight='bold')
                axes[1,0].set_ylabel('Accuracy Rate')
                axes[1,0].set_xlabel('Day of Week')
                axes[1,0].legend(title='Method')
                axes[1,0].tick_params(axis='x', rotation=45)
            
            # 4. Sample size over time
            sample_counts = df_success.groupby([df_success["date"].dt.to_period("M"), "method"]).size().unstack(level=1, fill_value=0)
            
            if not sample_counts.empty:
                sample_counts.plot(kind='line', ax=axes[1,1], marker='o')
                axes[1,1].set_title('Sample Sizes Over Time', fontweight='bold')
                axes[1,1].set_ylabel('Number of Predictions')
                axes[1,1].set_xlabel('Month')
                axes[1,1].legend(title='Method')
                axes[1,1].grid(True, alpha=0.3)
            
            plt.suptitle(f'Temporal Analysis - {crypto_symbol.upper()}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_temporal_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to plot temporal analysis: {e}")
            return None
    
    def _plot_confusion_matrices(self, df: pd.DataFrame, crypto_symbol: str) -> Optional[Path]:
        """Plot confusion matrices for each method."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            methods = ["full_agentic", "image_only", "sentiment_only"]
            
            for i, method in enumerate(methods):
                method_df = df[(df["method"] == method) & (df["success"] == True)]
                
                if len(method_df) > 0:
                    # Create confusion matrix
                    conf_matrix = pd.crosstab(
                        method_df["actual_direction"], 
                        method_df["predicted_direction"],
                        normalize='index'
                    )
                    
                    # Plot heatmap
                    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', 
                              ax=axes[i], cbar_kws={'label': 'Proportion'})
                    axes[i].set_title(f'{method.replace("_", " ").title()}\nConfusion Matrix', fontweight='bold')
                    axes[i].set_ylabel('Actual Direction')
                    axes[i].set_xlabel('Predicted Direction')
                else:
                    axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{method.replace("_", " ").title()}\nNo Data Available')
            
            plt.suptitle(f'Confusion Matrices - {crypto_symbol.upper()}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_confusion_matrices.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to plot confusion matrices: {e}")
            return None
    
    def _plot_performance_distribution(self, df: pd.DataFrame, crypto_symbol: str) -> Optional[Path]:
        """Plot performance distribution analysis."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            df_success = df[df["success"] == True]
            
            # 1. Success rate by method
            success_rates = df.groupby("method")["success"].mean()
            bars1 = axes[0,0].bar(success_rates.index, success_rates.values, alpha=0.8, color='lightcoral')
            axes[0,0].set_title('Success Rate by Method', fontweight='bold')
            axes[0,0].set_ylabel('Success Rate')
            axes[0,0].set_ylim(0, 1)
            
            for bar, value in zip(bars1, success_rates.values):
                axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                              f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 2. Prediction distribution by actual direction
            pred_dist = df_success.groupby(["method", "actual_direction"])["correct_prediction"].mean().unstack(level=1, fill_value=0)
            
            if not pred_dist.empty:
                pred_dist.plot(kind='bar', ax=axes[0,1], alpha=0.8)
                axes[0,1].set_title('Accuracy by Actual Market Direction', fontweight='bold')
                axes[0,1].set_ylabel('Accuracy Rate')
                axes[0,1].legend(title='Actual Direction')
                axes[0,1].tick_params(axis='x', rotation=45)
            
            # 3. Confidence distribution
            conf_dist = df_success["confidence"].value_counts()
            axes[1,0].pie(conf_dist.values, labels=conf_dist.index, autopct='%1.1f%%', startangle=90)
            axes[1,0].set_title('Overall Confidence Distribution', fontweight='bold')
            
            # 4. Method vs Direction heatmap
            method_direction = pd.crosstab(df_success["method"], df_success["predicted_direction"])
            sns.heatmap(method_direction, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1,1])
            axes[1,1].set_title('Prediction Direction by Method', fontweight='bold')
            axes[1,1].set_ylabel('Method')
            axes[1,1].set_xlabel('Predicted Direction')
            
            plt.suptitle(f'Performance Distribution Analysis - {crypto_symbol.upper()}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_performance_distribution.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to plot performance distribution: {e}")
            return None
    
    def _plot_trading_performance(self, results: Dict[str, Any], crypto_symbol: str) -> Optional[Path]:
        """Plot comprehensive trading performance metrics."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'Trading Performance Analysis - {crypto_symbol.upper()}', fontsize=16, fontweight='bold')
            
            methods = ["full_agentic", "image_only", "sentiment_only"]
            trading_data = {}
            
            # Extract trading data from results
            for method in methods:
                method_trades = []
                for date, predictions in results.get("daily_predictions", {}).items():
                    if method in predictions and predictions[method].get("success"):
                        pred = predictions[method]
                        if all(key in pred for key in ["target_price", "take_profit", "stop_loss"]):
                            # Calculate trade performance
                            current_price = pred.get("historical_price", 0)
                            actual_movement = pred.get("actual_movement_24h", 0)
                            next_day_price = pred.get("next_day_price")
                            
                            if next_day_price and current_price > 0:
                                predicted_direction = pred.get("predicted_direction", "DOWN")
                                if predicted_direction == "UP":
                                    pnl_pct = ((next_day_price - current_price) / current_price) * 100
                                else:
                                    pnl_pct = ((current_price - next_day_price) / current_price) * 100
                                
                                method_trades.append({
                                    'date': date,
                                    'pnl_pct': pnl_pct,
                                    'confidence': pred.get('confidence', 'MEDIUM'),
                                    'target_pct': pred.get('target_percentage', 0),
                                    'position_size': pred.get('position_size', 'MEDIUM')
                                })
                
                trading_data[method] = method_trades
            
            # 1. P&L Distribution by Method
            for i, method in enumerate(methods):
                if trading_data[method]:
                    pnl_values = [trade['pnl_pct'] for trade in trading_data[method]]
                    axes[0, 0].hist(pnl_values, alpha=0.6, label=method.replace('_', ' ').title(), bins=20)
            
            axes[0, 0].set_title('P&L Distribution by Method', fontweight='bold')
            axes[0, 0].set_xlabel('P&L Percentage')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.8)
            
            # 2. Win Rate by Method
            win_rates = []
            method_names = []
            method_colors = []
            color_map = {'full_agentic': '#2E86C1', 'image_only': '#E74C3C', 'sentiment_only': '#F39C12'}
            
            for method in methods:
                if trading_data[method]:
                    wins = len([t for t in trading_data[method] if t['pnl_pct'] > 0])
                    total = len(trading_data[method])
                    win_rates.append(wins / total if total > 0 else 0)
                    method_names.append(method.replace('_', ' ').title())
                    method_colors.append(color_map[method])
            
            if win_rates:
                bars = axes[0, 1].bar(method_names, win_rates, alpha=0.8, color=method_colors)
                axes[0, 1].set_title('Win Rate by Method', fontweight='bold')
                axes[0, 1].set_ylabel('Win Rate')
                axes[0, 1].set_ylim(0, 1)
                
                for bar, rate in zip(bars, win_rates):
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                                   f'{rate:.2%}', ha='center', va='bottom', fontweight='bold')
            
            # 3. Average Return by Method
            avg_returns = []
            avg_return_colors = []
            for method in methods:
                if trading_data[method]:
                    avg_return = sum(t['pnl_pct'] for t in trading_data[method]) / len(trading_data[method])
                    avg_returns.append(avg_return)
                    avg_return_colors.append(color_map[method])
            
            if avg_returns and method_names:
                bars = axes[0, 2].bar(method_names, avg_returns, alpha=0.8, color=avg_return_colors)
                axes[0, 2].set_title('Average Return per Trade', fontweight='bold')
                axes[0, 2].set_ylabel('Average Return (%)')
                axes[0, 2].axhline(y=0, color='red', linestyle='--', alpha=0.8)
                
                for bar, ret in zip(bars, avg_returns):
                    axes[0, 2].text(bar.get_x() + bar.get_width()/2, 
                                   bar.get_height() + (0.1 if ret >= 0 else -0.3),
                                   f'{ret:.2f}%', ha='center', va='bottom' if ret >= 0 else 'top', fontweight='bold')
            
            # 4. Confidence vs Performance
            for method in methods:
                if trading_data[method]:
                    conf_performance = {}
                    for conf_level in ['HIGH', 'MEDIUM', 'LOW']:
                        conf_trades = [t for t in trading_data[method] if t['confidence'] == conf_level]
                        if conf_trades:
                            avg_pnl = sum(t['pnl_pct'] for t in conf_trades) / len(conf_trades)
                            conf_performance[conf_level] = avg_pnl
                    
                    if conf_performance:
                        x_pos = range(len(conf_performance))
                        axes[1, 0].plot(x_pos, list(conf_performance.values()), 
                                      marker='o', label=method.replace('_', ' ').title(), linewidth=2)
            
            axes[1, 0].set_title('Performance by Confidence Level', fontweight='bold')
            axes[1, 0].set_xlabel('Confidence Level')
            axes[1, 0].set_ylabel('Average P&L (%)')
            axes[1, 0].set_xticks(range(3))
            axes[1, 0].set_xticklabels(['HIGH', 'MEDIUM', 'LOW'])
            axes[1, 0].legend()
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Best vs Worst Trades
            best_worst_data = []
            for method in methods:
                if trading_data[method]:
                    pnl_values = [t['pnl_pct'] for t in trading_data[method]]
                    best_worst_data.append({
                        'method': method.replace('_', ' ').title(),
                        'best': max(pnl_values),
                        'worst': min(pnl_values)
                    })
            
            if best_worst_data:
                methods_clean = [d['method'] for d in best_worst_data]
                best_trades = [d['best'] for d in best_worst_data]
                worst_trades = [d['worst'] for d in best_worst_data]
                
                x = range(len(methods_clean))
                width = 0.35
                
                axes[1, 1].bar([i - width/2 for i in x], best_trades, width, 
                             label='Best Trade', alpha=0.8, color='green')
                axes[1, 1].bar([i + width/2 for i in x], worst_trades, width, 
                             label='Worst Trade', alpha=0.8, color='red')
                
                axes[1, 1].set_title('Best vs Worst Trades', fontweight='bold')
                axes[1, 1].set_ylabel('P&L Percentage')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(methods_clean)
                axes[1, 1].legend()
                axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # 6. Risk-Adjusted Returns (Sharpe-like ratio)
            risk_adjusted = []
            risk_colors = []
            for method in methods:
                if trading_data[method] and len(trading_data[method]) > 1:
                    pnl_values = [t['pnl_pct'] for t in trading_data[method]]
                    avg_return = sum(pnl_values) / len(pnl_values)
                    volatility = (sum((x - avg_return) ** 2 for x in pnl_values) / len(pnl_values)) ** 0.5
                    sharpe = avg_return / volatility if volatility > 0 else 0
                    risk_adjusted.append(sharpe)
                    risk_colors.append(color_map[method])
            
            if risk_adjusted and method_names:
                bars = axes[1, 2].bar(method_names, risk_adjusted, alpha=0.8, color=risk_colors)
                axes[1, 2].set_title('Risk-Adjusted Returns (Sharpe-like)', fontweight='bold')
                axes[1, 2].set_ylabel('Risk-Adjusted Return')
                axes[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.8)
                
                for bar, ratio in zip(bars, risk_adjusted):
                    axes[1, 2].text(bar.get_x() + bar.get_width()/2, 
                                   bar.get_height() + (0.01 if ratio >= 0 else -0.02),
                                   f'{ratio:.3f}', ha='center', va='bottom' if ratio >= 0 else 'top', fontweight='bold')
            
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_trading_performance.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to plot trading performance: {e}")
            return None
    
    def _plot_portfolio_performance(self, results: Dict[str, Any], crypto_symbol: str) -> Optional[Path]:
        """Plot portfolio performance over time."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Portfolio Performance Over Time - {crypto_symbol.upper()}', fontsize=16, fontweight='bold')
            
            methods = ["full_agentic", "image_only", "sentiment_only"]
            colors = ['#2E86C1', '#E74C3C', '#F39C12']
            
            # Calculate cumulative portfolio performance for each method
            for method_idx, method in enumerate(methods):
                portfolio_values = [10000]  # Starting with $10,000
                dates = []
                cumulative_returns = [0]
                
                for date, predictions in sorted(results.get("daily_predictions", {}).items()):
                    if method in predictions and predictions[method].get("success"):
                        pred = predictions[method]
                        if all(key in pred for key in ["target_price", "take_profit", "stop_loss"]):
                            current_price = pred.get("historical_price", 0)
                            next_day_price = pred.get("next_day_price")
                            
                            if next_day_price and current_price > 0:
                                predicted_direction = pred.get("predicted_direction", "DOWN")
                                position_size = pred.get("position_size", "MEDIUM")
                                
                                # Calculate position size multiplier
                                multipliers = {"SMALL": 0.25, "MEDIUM": 0.5, "LARGE": 1.0}
                                size_mult = multipliers.get(position_size, 0.5)
                                
                                # Calculate P&L
                                if predicted_direction == "UP":
                                    pnl_pct = ((next_day_price - current_price) / current_price) * 100
                                else:
                                    pnl_pct = ((current_price - next_day_price) / current_price) * 100
                                
                                # Update portfolio
                                current_portfolio = portfolio_values[-1]
                                portfolio_change = current_portfolio * (pnl_pct / 100) * size_mult
                                new_portfolio = current_portfolio + portfolio_change
                                
                                portfolio_values.append(new_portfolio)
                                dates.append(date)
                                cumulative_return = ((new_portfolio - 10000) / 10000) * 100
                                cumulative_returns.append(cumulative_return)
                
                # Plot portfolio value over time
                if len(dates) > 0:
                    axes[0, 0].plot(range(len(portfolio_values)), portfolio_values, 
                                  color=colors[method_idx], label=method.replace('_', ' ').title(), 
                                  linewidth=2, marker='o', markersize=3)
                    
                    # Plot cumulative returns
                    axes[0, 1].plot(range(len(cumulative_returns)), cumulative_returns, 
                                  color=colors[method_idx], label=method.replace('_', ' ').title(), 
                                  linewidth=2, marker='o', markersize=3)
            
            # 1. Portfolio Value Over Time
            axes[0, 0].set_title('Portfolio Value Over Time', fontweight='bold')
            axes[0, 0].set_xlabel('Trade Number')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axhline(y=10000, color='black', linestyle='--', alpha=0.5, label='Starting Value')
            
            # 2. Cumulative Returns Over Time
            axes[0, 1].set_title('Cumulative Returns Over Time', fontweight='bold')
            axes[0, 1].set_xlabel('Trade Number')
            axes[0, 1].set_ylabel('Cumulative Return (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # 3. Drawdown Analysis
            for method_idx, method in enumerate(methods):
                portfolio_values = [10000]
                peak = 10000
                drawdowns = [0]
                
                for date, predictions in sorted(results.get("daily_predictions", {}).items()):
                    if method in predictions and predictions[method].get("success"):
                        pred = predictions[method]
                        if all(key in pred for key in ["target_price", "take_profit", "stop_loss"]):
                            current_price = pred.get("historical_price", 0)
                            next_day_price = pred.get("next_day_price")
                            
                            if next_day_price and current_price > 0:
                                predicted_direction = pred.get("predicted_direction", "DOWN")
                                position_size = pred.get("position_size", "MEDIUM")
                                
                                multipliers = {"SMALL": 0.25, "MEDIUM": 0.5, "LARGE": 1.0}
                                size_mult = multipliers.get(position_size, 0.5)
                                
                                if predicted_direction == "UP":
                                    pnl_pct = ((next_day_price - current_price) / current_price) * 100
                                else:
                                    pnl_pct = ((current_price - next_day_price) / current_price) * 100
                                
                                current_portfolio = portfolio_values[-1]
                                portfolio_change = current_portfolio * (pnl_pct / 100) * size_mult
                                new_portfolio = current_portfolio + portfolio_change
                                
                                portfolio_values.append(new_portfolio)
                                
                                # Calculate drawdown
                                if new_portfolio > peak:
                                    peak = new_portfolio
                                drawdown = ((peak - new_portfolio) / peak) * 100
                                drawdowns.append(drawdown)
                
                if len(drawdowns) > 1:
                    axes[1, 0].plot(range(len(drawdowns)), drawdowns, 
                                  color=colors[method_idx], label=method.replace('_', ' ').title(), 
                                  linewidth=2)
            
            axes[1, 0].set_title('Drawdown Analysis', fontweight='bold')
            axes[1, 0].set_xlabel('Trade Number')
            axes[1, 0].set_ylabel('Drawdown (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].fill_between(range(len(drawdowns)), drawdowns, 0, alpha=0.3, color='red')
            
            # 4. Risk-Return Scatter
            risk_return_data = []
            for method in methods:
                returns = []
                for date, predictions in results.get("daily_predictions", {}).items():
                    if method in predictions and predictions[method].get("success"):
                        pred = predictions[method]
                        if all(key in pred for key in ["target_price", "take_profit", "stop_loss"]):
                            current_price = pred.get("historical_price", 0)
                            next_day_price = pred.get("next_day_price")
                            
                            if next_day_price and current_price > 0:
                                predicted_direction = pred.get("predicted_direction", "DOWN")
                                if predicted_direction == "UP":
                                    pnl_pct = ((next_day_price - current_price) / current_price) * 100
                                else:
                                    pnl_pct = ((current_price - next_day_price) / current_price) * 100
                                returns.append(pnl_pct)
                
                if len(returns) > 1:
                    avg_return = sum(returns) / len(returns)
                    volatility = (sum((x - avg_return) ** 2 for x in returns) / len(returns)) ** 0.5
                    risk_return_data.append((volatility, avg_return, method))
            
            if risk_return_data:
                color_map = {'full_agentic': '#2E86C1', 'image_only': '#E74C3C', 'sentiment_only': '#F39C12'}
                for i, (risk, ret, method) in enumerate(risk_return_data):
                    axes[1, 1].scatter(risk, ret, s=100, color=color_map[method], 
                                     label=method.replace('_', ' ').title())
                    axes[1, 1].annotate(method.replace('_', ' ').title(), 
                                      (risk, ret), xytext=(5, 5), textcoords='offset points')
                
                axes[1, 1].set_title('Risk-Return Profile', fontweight='bold')
                axes[1, 1].set_xlabel('Volatility (Risk)')
                axes[1, 1].set_ylabel('Average Return (%)')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[1, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_portfolio_performance.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to plot portfolio performance: {e}")
            return None
    
    def _plot_risk_return_analysis(self, results: Dict[str, Any], crypto_symbol: str) -> Optional[Path]:
        """Plot comprehensive risk-return analysis."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Risk-Return Analysis - {crypto_symbol.upper()}', fontsize=16, fontweight='bold')
            
            methods = ["full_agentic", "image_only", "sentiment_only"]
            colors = ['#2E86C1', '#E74C3C', '#F39C12']
            
            # Calculate comprehensive risk metrics
            method_metrics = {}
            
            for method in methods:
                returns = []
                winning_trades = []
                losing_trades = []
                
                for date, predictions in results.get("daily_predictions", {}).items():
                    if method in predictions and predictions[method].get("success"):
                        pred = predictions[method]
                        if all(key in pred for key in ["target_price", "take_profit", "stop_loss"]):
                            current_price = pred.get("historical_price", 0)
                            next_day_price = pred.get("next_day_price")
                            
                            if next_day_price and current_price > 0:
                                predicted_direction = pred.get("predicted_direction", "DOWN")
                                if predicted_direction == "UP":
                                    pnl_pct = ((next_day_price - current_price) / current_price) * 100
                                else:
                                    pnl_pct = ((current_price - next_day_price) / current_price) * 100
                                
                                returns.append(pnl_pct)
                                if pnl_pct > 0:
                                    winning_trades.append(pnl_pct)
                                else:
                                    losing_trades.append(pnl_pct)
                
                if returns:
                    avg_return = sum(returns) / len(returns)
                    volatility = (sum((x - avg_return) ** 2 for x in returns) / len(returns)) ** 0.5
                    sharpe = avg_return / volatility if volatility > 0 else 0
                    win_rate = len(winning_trades) / len(returns)
                    avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
                    avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
                    profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades)) if losing_trades and avg_loss != 0 else float('inf')
                    
                    method_metrics[method] = {
                        'returns': returns,
                        'avg_return': avg_return,
                        'volatility': volatility,
                        'sharpe': sharpe,
                        'win_rate': win_rate,
                        'avg_win': avg_win,
                        'avg_loss': avg_loss,
                        'profit_factor': profit_factor,
                        'max_win': max(returns),
                        'max_loss': min(returns)
                    }
            
            # 1. Return Distribution Comparison
            for i, method in enumerate(methods):
                if method in method_metrics:
                    returns = method_metrics[method]['returns']
                    axes[0, 0].hist(returns, alpha=0.6, label=method.replace('_', ' ').title(), 
                                  bins=20, color=colors[i])
            
            axes[0, 0].set_title('Return Distribution Comparison', fontweight='bold')
            axes[0, 0].set_xlabel('Return (%)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.8)
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Risk Metrics Comparison
            metrics_to_plot = ['volatility', 'sharpe', 'win_rate']
            metric_names = ['Volatility', 'Sharpe Ratio', 'Win Rate']
            
            x = np.arange(len(methods))
            width = 0.25
            
            for i, metric in enumerate(metrics_to_plot):
                values = [method_metrics.get(method, {}).get(metric, 0) for method in methods]
                axes[0, 1].bar(x + i*width, values, width, label=metric_names[i], alpha=0.8)
            
            axes[0, 1].set_title('Risk Metrics Comparison', fontweight='bold')
            axes[0, 1].set_xlabel('Methods')
            axes[0, 1].set_ylabel('Metric Value')
            axes[0, 1].set_xticks(x + width)
            axes[0, 1].set_xticklabels([m.replace('_', ' ').title() for m in methods])
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Profit Factor Analysis
            profit_factors = []
            method_names_clean = []
            
            for method in methods:
                if method in method_metrics:
                    pf = method_metrics[method]['profit_factor']
                    if pf != float('inf'):
                        profit_factors.append(pf)
                        method_names_clean.append(method.replace('_', ' ').title())
            
            if profit_factors:
                bars = axes[1, 0].bar(method_names_clean, profit_factors, alpha=0.8, color=colors[:len(profit_factors)])
                axes[1, 0].set_title('Profit Factor by Method', fontweight='bold')
                axes[1, 0].set_ylabel('Profit Factor')
                axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.8, label='Break-even')
                axes[1, 0].legend()
                
                for bar, pf in zip(bars, profit_factors):
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                   f'{pf:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # 4. Risk-Adjusted Performance Radar
            categories = ['Avg Return', 'Win Rate', 'Sharpe Ratio', 'Profit Factor']
            
            # Normalize metrics for radar chart
            for method_idx, method in enumerate(methods):
                if method in method_metrics:
                    metrics = method_metrics[method]
                    
                    # Normalize values (0-1 scale)
                    normalized_values = [
                        max(0, min(1, (metrics['avg_return'] + 10) / 20)),  # Assuming -10% to +10% range
                        metrics['win_rate'],
                        max(0, min(1, (metrics['sharpe'] + 2) / 4)),  # Assuming -2 to +2 range
                        max(0, min(1, metrics['profit_factor'] / 5)) if metrics['profit_factor'] != float('inf') else 1
                    ]
                    
                    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                    angles += angles[:1]  # Complete the circle
                    normalized_values += normalized_values[:1]  # Complete the circle
                    
                    axes[1, 1].plot(angles, normalized_values, 'o-', linewidth=2, 
                                  label=method.replace('_', ' ').title(), color=colors[method_idx])
                    axes[1, 1].fill(angles, normalized_values, alpha=0.25, color=colors[method_idx])
            
            axes[1, 1].set_xticks(angles[:-1])
            axes[1, 1].set_xticklabels(categories)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_title('Risk-Adjusted Performance Profile', fontweight='bold')
            axes[1, 1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_risk_return_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to plot risk-return analysis: {e}")
            return None
    
    def _plot_stop_loss_take_profit_analysis(self, results: Dict[str, Any], crypto_symbol: str) -> Optional[Path]:
        """Plot stop loss and take profit analysis."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Stop Loss / Take Profit Analysis - {crypto_symbol.upper()}', fontsize=16, fontweight='bold')
            
            methods = ["full_agentic", "image_only", "sentiment_only"]
            colors = ['#2E86C1', '#E74C3C', '#F39C12']
            
            # Analyze stop loss and take profit effectiveness
            method_sl_tp_data = {}
            
            for method in methods:
                sl_hits = 0
                tp_hits = 0
                total_trades = 0
                sl_prevented_losses = []
                tp_secured_profits = []
                
                for date, predictions in results.get("daily_predictions", {}).items():
                    if method in predictions and predictions[method].get("success"):
                        pred = predictions[method]
                        if all(key in pred for key in ["target_price", "take_profit", "stop_loss"]):
                            current_price = pred.get("historical_price", 0)
                            next_day_price = pred.get("next_day_price")
                            
                            if next_day_price and current_price > 0:
                                total_trades += 1
                                predicted_direction = pred.get("predicted_direction", "DOWN")
                                take_profit = pred.get("take_profit", current_price)
                                stop_loss = pred.get("stop_loss", current_price)
                                
                                if predicted_direction == "UP":
                                    # Long position
                                    if next_day_price <= stop_loss:
                                        sl_hits += 1
                                        loss_prevented = ((current_price - next_day_price) / current_price) * 100
                                        sl_prevented_losses.append(loss_prevented)
                                    elif next_day_price >= take_profit:
                                        tp_hits += 1
                                        profit_secured = ((take_profit - current_price) / current_price) * 100
                                        tp_secured_profits.append(profit_secured)
                                else:
                                    # Short position
                                    if next_day_price >= stop_loss:
                                        sl_hits += 1
                                        loss_prevented = ((next_day_price - current_price) / current_price) * 100
                                        sl_prevented_losses.append(loss_prevented)
                                    elif next_day_price <= take_profit:
                                        tp_hits += 1
                                        profit_secured = ((current_price - take_profit) / current_price) * 100
                                        tp_secured_profits.append(profit_secured)
                
                method_sl_tp_data[method] = {
                    'sl_hits': sl_hits,
                    'tp_hits': tp_hits,
                    'total_trades': total_trades,
                    'sl_rate': sl_hits / total_trades if total_trades > 0 else 0,
                    'tp_rate': tp_hits / total_trades if total_trades > 0 else 0,
                    'sl_prevented_losses': sl_prevented_losses,
                    'tp_secured_profits': tp_secured_profits
                }
            
            # 1. Stop Loss Hit Rates
            sl_rates = [method_sl_tp_data.get(method, {}).get('sl_rate', 0) for method in methods]
            method_names = [method.replace('_', ' ').title() for method in methods]
            
            bars = axes[0, 0].bar(method_names, sl_rates, alpha=0.8, color=colors)
            axes[0, 0].set_title('Stop Loss Hit Rate', fontweight='bold')
            axes[0, 0].set_ylabel('Stop Loss Hit Rate')
            axes[0, 0].set_ylim(0, 1)
            
            for bar, rate in zip(bars, sl_rates):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{rate:.2%}', ha='center', va='bottom', fontweight='bold')
            
            # 2. Take Profit Hit Rates
            tp_rates = [method_sl_tp_data.get(method, {}).get('tp_rate', 0) for method in methods]
            
            bars = axes[0, 1].bar(method_names, tp_rates, alpha=0.8, color=colors)
            axes[0, 1].set_title('Take Profit Hit Rate', fontweight='bold')
            axes[0, 1].set_ylabel('Take Profit Hit Rate')
            axes[0, 1].set_ylim(0, 1)
            
            for bar, rate in zip(bars, tp_rates):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{rate:.2%}', ha='center', va='bottom', fontweight='bold')
            
            # 3. SL vs TP Effectiveness Comparison
            x = np.arange(len(methods))
            width = 0.35
            
            axes[0, 2].bar(x - width/2, sl_rates, width, label='Stop Loss Hit Rate', alpha=0.8, color='red')
            axes[0, 2].bar(x + width/2, tp_rates, width, label='Take Profit Hit Rate', alpha=0.8, color='green')
            
            axes[0, 2].set_title('SL vs TP Hit Rates Comparison', fontweight='bold')
            axes[0, 2].set_ylabel('Hit Rate')
            axes[0, 2].set_xticks(x)
            axes[0, 2].set_xticklabels(method_names)
            axes[0, 2].legend()
            axes[0, 2].set_ylim(0, 1)
            
            # 4. Average Loss Prevented by Stop Loss
            avg_losses_prevented = []
            for method in methods:
                losses = method_sl_tp_data.get(method, {}).get('sl_prevented_losses', [])
                avg_loss = sum(losses) / len(losses) if losses else 0
                avg_losses_prevented.append(avg_loss)
            
            bars = axes[1, 0].bar(method_names, avg_losses_prevented, alpha=0.8, color='red')
            axes[1, 0].set_title('Average Loss Prevented by SL', fontweight='bold')
            axes[1, 0].set_ylabel('Average Loss Prevented (%)')
            
            for bar, loss in zip(bars, avg_losses_prevented):
                if loss > 0:
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                   f'{loss:.2f}%', ha='center', va='bottom', fontweight='bold')
            
            # 5. Average Profit Secured by Take Profit
            avg_profits_secured = []
            for method in methods:
                profits = method_sl_tp_data.get(method, {}).get('tp_secured_profits', [])
                avg_profit = sum(profits) / len(profits) if profits else 0
                avg_profits_secured.append(avg_profit)
            
            bars = axes[1, 1].bar(method_names, avg_profits_secured, alpha=0.8, color='green')
            axes[1, 1].set_title('Average Profit Secured by TP', fontweight='bold')
            axes[1, 1].set_ylabel('Average Profit Secured (%)')
            
            for bar, profit in zip(bars, avg_profits_secured):
                if profit > 0:
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                   f'{profit:.2f}%', ha='center', va='bottom', fontweight='bold')
            
            # 6. Risk Management Effectiveness Score
            # Combine SL and TP effectiveness with weights
            risk_mgmt_scores = []
            for i, method in enumerate(methods):
                sl_rate = sl_rates[i]
                tp_rate = tp_rates[i]
                avg_loss_prevented = avg_losses_prevented[i]
                avg_profit_secured = avg_profits_secured[i]
                
                # Calculate a composite risk management score
                # Higher SL rate is generally bad (more losses), higher TP rate is good
                # But prevented losses and secured profits are good
                score = (tp_rate * 40) + (avg_profit_secured * 2) + (avg_loss_prevented * 1.5) - (sl_rate * 20)
                risk_mgmt_scores.append(max(0, score))  # Ensure non-negative
            
            bars = axes[1, 2].bar(method_names, risk_mgmt_scores, alpha=0.8, color=colors)
            axes[1, 2].set_title('Risk Management Effectiveness Score', fontweight='bold')
            axes[1, 2].set_ylabel('Effectiveness Score')
            
            for bar, score in zip(bars, risk_mgmt_scores):
                axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            chart_path = self.charts_dir / f"{crypto_symbol}_stop_loss_take_profit_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Failed to plot stop loss take profit analysis: {e}")
            return None
    
    def generate_thesis_report(self, results: Dict[str, Any], crypto_symbol: str) -> Path:
        """Generate comprehensive thesis report."""
        try:
            logger.info("=== ANALYZER: Generating thesis report ===")
            logger.info(f"REPORT: Analysis directory: {self.analysis_dir}")
            logger.info(f"REPORT: Analysis directory exists: {self.analysis_dir.exists()}")
            
            logger.info("REPORT: Running prediction accuracy analysis...")
            analysis = self.analyze_prediction_accuracy(results)
            logger.info(f"REPORT: Analysis completed, has error: {'error' in analysis}")
            
            if "error" in analysis:
                logger.error(f"REPORT: Analysis failed with error: {analysis['error']}")
                return None
            
            logger.info("REPORT: Generating visualizations...")
            chart_paths = self.generate_thesis_visualizations(results, crypto_symbol)
            logger.info(f"REPORT: Generated {len(chart_paths)} chart files")
            
            # Create markdown report
            report_path = self.analysis_dir / f"{crypto_symbol}_thesis_report.md"
            logger.info(f"REPORT: Creating markdown report at: {report_path}")
            
            logger.info("REPORT: Generating markdown content...")
            markdown_content = self._generate_markdown_report(analysis, chart_paths, crypto_symbol)
            logger.info(f"REPORT: Generated {len(markdown_content)} characters of markdown")
            
            logger.info("REPORT: Writing markdown file...")
            with open(report_path, 'w') as f:
                f.write(markdown_content)
            logger.info(f"REPORT: Markdown report saved to: {report_path}")
            
            # Also save as JSON
            json_path = self.analysis_dir / f"{crypto_symbol}_thesis_analysis.json"
            logger.info(f"REPORT: Saving JSON analysis to: {json_path}")
            
            # Clean the analysis data for JSON serialization
            logger.info("REPORT: Cleaning analysis data for JSON serialization...")
            cleaned_analysis = self._clean_for_json_serialization(analysis)
            
            with open(json_path, 'w') as f:
                json.dump(cleaned_analysis, f, indent=2, default=str)
            logger.info(f"REPORT: JSON analysis saved to: {json_path}")
            
            # Verify files were created
            if report_path.exists():
                logger.info(f"REPORT: Markdown file verified: {report_path.stat().st_size} bytes")
            else:
                logger.error(f"REPORT: Markdown file not found after creation: {report_path}")
                
            if json_path.exists():
                logger.info(f"REPORT: JSON file verified: {json_path.stat().st_size} bytes")
            else:
                logger.error(f"REPORT: JSON file not found after creation: {json_path}")
            
            logger.info("=== ANALYZER: Thesis report generation completed ===")
            return report_path
            
        except Exception as e:
            logger.error(f"REPORT: Failed to generate thesis report: {e}")
            import traceback
            logger.error(f"REPORT: Traceback: {traceback.format_exc()}")
            return None
    
    def _clean_for_json_serialization(self, obj):
        """Clean data structure for JSON serialization by converting tuple keys to strings."""
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                # Convert tuple keys to string
                if isinstance(key, tuple):
                    key_str = "_".join(str(k) for k in key)
                    cleaned[key_str] = self._clean_for_json_serialization(value)
                else:
                    cleaned[str(key)] = self._clean_for_json_serialization(value)
            return cleaned
        elif isinstance(obj, list):
            return [self._clean_for_json_serialization(item) for item in obj]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            # Convert pandas objects to dict with string keys
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            else:
                return obj.to_dict()
        elif hasattr(obj, 'to_dict'):
            # Handle other objects with to_dict method
            return self._clean_for_json_serialization(obj.to_dict())
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Convert other types to string
            return str(obj)
    
    def _generate_markdown_report(self, analysis: Dict[str, Any], chart_paths: List[Path], crypto_symbol: str) -> str:
        """Generate comprehensive markdown thesis report with trading performance metrics."""
        report = f"""# Comprehensive Thesis Analysis Report: {crypto_symbol.upper()}

## Multi-Agent vs One-Shot LLM Prediction Comparison with Trading Performance Analysis

**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report compares the performance of three cryptocurrency price prediction approaches with comprehensive trading performance analysis:

1. **Full Agentic**: Multi-agent system using technical analysis, sentiment analysis, and data fusion
2. **Image Only**: One-shot LLM analysis of technical charts only  
3. **Sentiment Only**: One-shot LLM analysis of social sentiment only

## Overall Statistics

- **Total Predictions:** {analysis.get('overall_stats', {}).get('total_predictions', 0)}
- **Overall Accuracy:** {analysis.get('overall_stats', {}).get('overall_accuracy', 0):.3f}
- **Date Range:** {analysis.get('overall_stats', {}).get('date_range', {}).get('start', 'N/A')} to {analysis.get('overall_stats', {}).get('date_range', {}).get('end', 'N/A')}

## Method Comparison Analysis

"""
        
        # Add detailed method comparison with trading metrics
        method_comp = analysis.get('method_comparison', {})
        for method, stats in method_comp.items():
            if method != 'comparison_summary' and isinstance(stats, dict):
                report += f"""### {method.replace('_', ' ').title()}

**Basic Performance:**
- **Success Rate:** {stats.get('success_rate', 0):.3f}
- **Accuracy:** {stats.get('accuracy', 0):.3f}
- **Total Attempts:** {stats.get('total_attempts', 0)}
- **Successful Predictions:** {stats.get('successful_predictions', 0)}

**Trading Performance:**"""
                
                # Add trading performance if available
                trading_perf = stats.get('trading_performance', {})
                if trading_perf and not trading_perf.get('no_trading_data'):
                    report += f"""
- **Portfolio Return:** {stats.get('portfolio_return', 0):.2f}%
- **Win Rate:** {trading_perf.get('win_rate', 0):.2%}
- **Total Trades:** {trading_perf.get('total_trades', 0)}
- **Average Return per Trade:** {trading_perf.get('avg_return_per_trade', 0):.2f}%
- **Best Trade:** {trading_perf.get('best_trade_pct', 0):.2f}%
- **Worst Trade:** {trading_perf.get('worst_trade_pct', 0):.2f}%
- **Sharpe Ratio:** {trading_perf.get('sharpe_ratio', 0):.3f}
- **Maximum Drawdown:** {trading_perf.get('max_drawdown_pct', 0):.2f}%
- **Profit Factor:** {trading_perf.get('profit_factor', 0):.2f}

**Risk Management:**
- **Stop Loss Hit Rate:** {trading_perf.get('stop_loss_hit_rate', 0):.2%}
- **Take Profit Hit Rate:** {trading_perf.get('take_profit_hit_rate', 0):.2%}
- **Target Hit Rate:** {trading_perf.get('target_hit_rate', 0):.2%}
- **High Confidence Win Rate:** {trading_perf.get('high_confidence_win_rate', 0):.2%}
"""
                else:
                    report += "\n- **No trading data available for this method**\n"

                report += "\n"

        # Add statistical significance
        if 'statistical_tests' in analysis:
            report += "\n## Statistical Significance Tests\n\n"
            for test_name, test_result in analysis['statistical_tests'].items():
                if isinstance(test_result, dict) and 'p_value' in test_result:
                    significance = "**Significant**" if test_result['p_value'] < 0.05 else "Not Significant"
                    report += f"- **{test_name}:** p-value = {test_result['p_value']:.4f} ({significance})\n"

        # Add trading performance summary
        report += """
## Trading Performance Summary

### Portfolio Performance Comparison

The trading performance analysis simulates actual trading based on the predictions with the following assumptions:
- **Starting Capital:** $10,000
- **Position Sizing:** Variable (Small: 25%, Medium: 50%, Large: 100% of capital)
- **Risk Management:** Stop loss and take profit levels as predicted by each method
- **Transaction Costs:** Not included (would reduce actual returns)

### Key Trading Metrics Explained

- **Win Rate:** Percentage of profitable trades
- **Profit Factor:** Ratio of gross profit to gross loss (>1.0 indicates profitability)
- **Sharpe Ratio:** Risk-adjusted return measure (higher is better)
- **Maximum Drawdown:** Largest peak-to-trough decline in portfolio value
- **Stop Loss Hit Rate:** How often stop losses were triggered (lower generally better)
- **Take Profit Hit Rate:** How often take profit targets were reached (higher generally better)

"""

        # Add charts section
        report += "\n## Comprehensive Visualizations\n\n"
        
        chart_categories = {
            "accuracy_comparison": "Accuracy Comparison Analysis",
            "confidence_accuracy": "Confidence vs Accuracy Analysis", 
            "temporal_analysis": "Temporal Performance Patterns",
            "confusion_matrices": "Prediction Confusion Matrices",
            "performance_distribution": "Performance Distribution Analysis",
            "trading_performance": "Trading Performance Metrics",
            "portfolio_performance": "Portfolio Performance Over Time",
            "risk_return_analysis": "Risk-Return Analysis",
            "stop_loss_take_profit_analysis": "Stop Loss / Take Profit Analysis"
        }
        
        for chart_path in chart_paths:
            if chart_path and chart_path.exists():
                chart_filename = chart_path.name
                # Try to categorize the chart
                chart_category = "Analysis Chart"
                for key, category in chart_categories.items():
                    if key in chart_filename:
                        chart_category = category
                        break
                
                report += f"### {chart_category}\n\n![{chart_category}]({chart_filename})\n\n"

        # Add comprehensive conclusions
        report += """## Comprehensive Analysis & Conclusions

### Key Findings

#### Prediction Accuracy
1. **Method Performance Ranking:** [Analyze which method had highest accuracy]
2. **Confidence Correlation:** [Analyze relationship between confidence levels and accuracy]
3. **Temporal Stability:** [Analyze performance consistency over time]

#### Trading Performance
1. **Profitability Ranking:** [Analyze which method generated highest returns]
2. **Risk-Adjusted Performance:** [Compare Sharpe ratios and drawdowns]
3. **Risk Management Effectiveness:** [Analyze stop loss and take profit performance]

#### Multi-Agent vs One-Shot Comparison
1. **Complexity vs Performance:** [Compare complex multi-agent vs simple one-shot approaches]
2. **Resource Efficiency:** [Consider computational cost vs performance gains]
3. **Practical Implementation:** [Assess real-world viability of each approach]

### Trading Insights

#### Position Sizing Impact
- **Small Positions (25%):** Lower risk but also lower returns
- **Medium Positions (50%):** Balanced risk-return profile
- **Large Positions (100%):** Higher returns but increased volatility

#### Risk Management Analysis
- **Stop Loss Effectiveness:** [Analyze how well stop losses protected capital]
- **Take Profit Optimization:** [Assess target achievement rates]
- **Confidence-Based Sizing:** [Evaluate if confidence levels correlated with performance]

### Implications for Cryptocurrency Trading

#### Practical Applications
1. **Signal Generation:** [Assess viability for automated trading signals]
2. **Risk Management:** [Evaluate stop loss and take profit strategies]
3. **Portfolio Integration:** [Consider how to incorporate predictions into broader strategy]

#### Limitations and Considerations
1. **Market Conditions:** [Analyze performance across different market regimes]
2. **Slippage and Costs:** [Consider real-world trading friction]
3. **Scalability:** [Assess performance with different position sizes]

### Future Research Directions

#### Methodology Improvements
1. **Extended Time Periods:** Longer backtesting periods for more robust statistics
2. **Multiple Asset Classes:** Test across different cryptocurrencies and market caps
3. **Market Regime Analysis:** Performance during bull/bear markets and high/low volatility
4. **Transaction Cost Integration:** Include realistic trading costs and slippage

#### Advanced Analytics
1. **Machine Learning Enhancement:** Incorporate ML for prediction post-processing
2. **Dynamic Position Sizing:** Adaptive position sizing based on market conditions
3. **Multi-Timeframe Analysis:** Incorporate multiple prediction horizons
4. **Ensemble Methods:** Combine multiple prediction approaches

#### Risk Management Optimization
1. **Dynamic Stop Loss/Take Profit:** Adaptive risk management levels
2. **Correlation Analysis:** Account for correlation with other assets
3. **Volatility Adjustment:** Risk management based on current market volatility
4. **Drawdown Protection:** Enhanced capital preservation strategies

### Final Recommendations

#### For Academic Research
- Focus on statistical significance testing with larger sample sizes
- Incorporate economic significance alongside statistical significance
- Consider regime-dependent analysis
- Validate findings across multiple cryptocurrencies

#### For Practical Implementation
- Start with paper trading to validate real-world performance
- Implement conservative position sizing initially
- Monitor performance across different market conditions
- Consider hybrid approaches combining multiple methods

---

*This comprehensive analysis was generated automatically by the CryptoAgentForecaster thesis analysis system, incorporating both traditional accuracy metrics and practical trading performance evaluation.*

## Disclaimer

This analysis is for research and educational purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always conduct your own research and consider consulting with financial professionals before making investment decisions.
"""
        
        return report 