#!/usr/bin/env python3
"""
CryptoAgentForecaster Validation Framework

A comprehensive validation system to test the accuracy and performance of the crypto forecasting tool.
Supports backtesting, live validation, and multi-coin analysis with scientific metrics.
"""

import asyncio
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import subprocess
import time
from dataclasses import dataclass
from enum import Enum

from src.crypto_agent_forecaster.tools.coingecko_tool import CoinGeckoTool


class ForecastDirection(Enum):
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


class ConfidenceLevel(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class ForecastResult:
    """Represents a single forecast result"""
    crypto: str
    timestamp: datetime
    forecast_direction: ForecastDirection
    confidence: ConfidenceLevel
    predicted_price: Optional[float]
    actual_price: Optional[float]
    price_at_forecast: float
    time_horizon_hours: int
    accuracy: Optional[bool] = None
    return_pct: Optional[float] = None
    

@dataclass
class ValidationMetrics:
    """Validation metrics for a set of forecasts"""
    total_predictions: int
    correct_predictions: int
    accuracy_percentage: float
    precision_by_confidence: Dict[str, float]
    recall_by_confidence: Dict[str, float]
    average_return: float
    sharpe_ratio: float
    max_drawdown: float
    winning_trades: int
    losing_trades: int
    average_winning_return: float
    average_losing_return: float


class CryptoValidator:
    """Main validation framework for crypto forecasting"""
    
    def __init__(self, results_dir: str = "validation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.coingecko = CoinGeckoTool()
        
        # Popular cryptocurrencies for testing
        self.test_coins = [
            "bitcoin", "ethereum", "solana", "cardano", "polygon",
            "chainlink", "avalanche-2", "polkadot", "uniswap", "litecoin"
        ]
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for validation"""
        log_file = self.results_dir / "validation.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def run_live_validation(self, duration_hours: int = 24, interval_hours: int = 1, 
                                 coins: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run live validation by making forecasts and checking them after the specified interval
        
        Args:
            duration_hours: Total duration to run validation
            interval_hours: How often to make new forecasts
            coins: List of coins to test (default: bitcoin only)
        """
        if coins is None:
            coins = ["bitcoin"]
        
        self.logger.info(f"Starting live validation for {duration_hours} hours with {interval_hours}h intervals")
        self.logger.info(f"Testing coins: {coins}")
        
        results = []
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        while datetime.now() < end_time:
            for coin in coins:
                try:
                    # Make forecast
                    forecast_result = await self._make_forecast(coin, f"{interval_hours} hours")
                    if forecast_result:
                        results.append(forecast_result)
                        self.logger.info(f"Forecast made for {coin}: {forecast_result.forecast_direction.value}")
                        
                        # Schedule validation check
                        asyncio.create_task(
                            self._validate_forecast_after_delay(forecast_result, interval_hours)
                        )
                    
                except Exception as e:
                    self.logger.error(f"Error making forecast for {coin}: {e}")
            
            # Wait for next interval
            await asyncio.sleep(interval_hours * 3600)
        
        # Save results
        self._save_live_results(results, start_time, end_time)
        return self._calculate_metrics(results)
    
    async def run_backtesting_validation(self, days_back: int = 30, 
                                       coins: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run backtesting validation using historical data
        
        Args:
            days_back: How many days of historical data to test
            coins: List of coins to test
        """
        if coins is None:
            coins = self.test_coins[:5]  # Test top 5 coins by default
        
        self.logger.info(f"Starting backtesting validation for {days_back} days")
        self.logger.info(f"Testing coins: {coins}")
        
        all_results = []
        
        for coin in coins:
            coin_results = await self._backtest_coin(coin, days_back)
            all_results.extend(coin_results)
        
        # Calculate and save metrics
        metrics = self._calculate_metrics(all_results)
        self._save_backtest_results(all_results, metrics, days_back)
        
        return metrics
    
    async def _backtest_coin(self, coin: str, days_back: int) -> List[ForecastResult]:
        """Backtest a single coin"""
        self.logger.info(f"Backtesting {coin}")
        
        # Get historical data using the correct CoinGeckoTool interface
        try:
            # Use the _run method with proper query format
            query = f"{coin} ohlcv {days_back + 1} days"
            historical_data_json = self.coingecko._run(query=query)
            
            if not historical_data_json:
                self.logger.error(f"No historical data response for {coin}")
                return []
            
            # Parse the JSON response
            import json
            historical_data_dict = json.loads(historical_data_json)
            
            # Check for errors in the response
            if "error" in historical_data_dict:
                self.logger.error(f"Error in historical data for {coin}: {historical_data_dict['error']}")
                return []
            
            # Extract OHLCV data from the response
            ohlcv_data = None
            if 'ohlcv_data' in historical_data_dict:
                ohlcv_data = historical_data_dict['ohlcv_data']
            elif isinstance(historical_data_dict, list):
                ohlcv_data = historical_data_dict
            else:
                self.logger.error(f"Cannot find ohlcv_data in response for {coin}")
                return []
            
            if not ohlcv_data:
                self.logger.error(f"Empty OHLCV data for {coin}")
                return []
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data)
            
            # Ensure we have the required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns for {coin}: {missing_columns}")
                return []
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {coin}: {e}")
            return []
        
        results = []
        
        # Test every 6 hours for the period (simplified backtesting)
        step_size = max(1, len(df) // (days_back * 4))  # Aim for ~4 forecasts per day
        
        for i in range(0, len(df) - step_size, step_size):
            try:
                forecast_time = df.iloc[i]['timestamp']
                forecast_price = df.iloc[i]['close']  # Use close price as the forecast baseline
                
                # Look ahead for actual price (simulate 1-hour forecast)
                if i + step_size < len(df):
                    actual_price = df.iloc[i + step_size]['close']
                    
                    # Simulate forecast using simple technical analysis
                    forecast_direction = self._simulate_forecast_direction(df.iloc[:i+1])
                    
                    # Calculate accuracy
                    price_change = actual_price - forecast_price
                    was_correct = (
                        (forecast_direction == ForecastDirection.UP and price_change > 0) or
                        (forecast_direction == ForecastDirection.DOWN and price_change < 0) or
                        (forecast_direction == ForecastDirection.NEUTRAL and abs(price_change/forecast_price) < 0.01)
                    )
                    
                    result = ForecastResult(
                        crypto=coin,
                        timestamp=forecast_time,
                        forecast_direction=forecast_direction,
                        confidence=ConfidenceLevel.MEDIUM,
                        predicted_price=None,  # Not predicting exact price in this simulation
                        actual_price=actual_price,
                        price_at_forecast=forecast_price,
                        time_horizon_hours=1,  # Simplified to 1-hour forecasts
                        accuracy=was_correct,
                        return_pct=(price_change / forecast_price) * 100 if forecast_price > 0 else 0
                    )
                    
                    results.append(result)
                    
            except Exception as e:
                self.logger.error(f"Error processing forecast for {coin} at index {i}: {e}")
                continue
        
        self.logger.info(f"Generated {len(results)} backtest forecasts for {coin}")
        return results
    
    def _simulate_forecast_direction(self, historical_df: pd.DataFrame) -> ForecastDirection:
        """Simulate forecast direction using simple technical analysis"""
        if len(historical_df) < 5:
            return ForecastDirection.NEUTRAL
        
        try:
            # Simple moving average strategy
            recent_prices = historical_df['close'].tail(5)
            avg_price = recent_prices.mean()
            current_price = recent_prices.iloc[-1]
            
            # Simple momentum check
            if current_price > avg_price * 1.01:  # 1% above average
                return ForecastDirection.UP
            elif current_price < avg_price * 0.99:  # 1% below average
                return ForecastDirection.DOWN
            else:
                return ForecastDirection.NEUTRAL
                
        except Exception:
            return ForecastDirection.NEUTRAL
    
    async def _make_forecast(self, coin: str, horizon: str) -> Optional[ForecastResult]:
        """Make a forecast using the main forecasting tool"""
        try:
            # Run the main forecasting tool from the parent directory
            parent_dir = Path(__file__).parent.parent
            cmd = ["uv", "run", "--active", "main.py", "forecast", coin, "--horizon", horizon, "--yes"]
            
            self.logger.info(f"Running forecast command: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=parent_dir  # Run from parent directory where main.py is located
            )
            
            if process.returncode != 0:
                self.logger.error(f"Forecast failed for {coin} (return code: {process.returncode})")
                self.logger.error(f"STDERR: {process.stderr}")
                if process.stdout:
                    self.logger.error(f"STDOUT: {process.stdout}")
                return None
            
            # Parse the forecast result from the output or results file
            forecast_data = self._parse_forecast_result(coin)
            
            if forecast_data:
                # Get current price
                current_price = self._get_current_price(coin)
                
                return ForecastResult(
                    crypto=coin,
                    timestamp=datetime.now(),
                    forecast_direction=ForecastDirection(forecast_data.get('direction', 'NEUTRAL')),
                    confidence=ConfidenceLevel(forecast_data.get('confidence', 'MEDIUM')),
                    predicted_price=forecast_data.get('predicted_price'),
                    actual_price=None,  # Will be filled later
                    price_at_forecast=current_price,
                    time_horizon_hours=int(horizon.split()[0])
                )
            
        except Exception as e:
            self.logger.error(f"Error making forecast for {coin}: {e}")
            return None
    
    def _parse_forecast_result(self, coin: str) -> Optional[Dict[str, Any]]:
        """Parse forecast result from the results directory"""
        try:
            # Look for the most recent results file in the parent directory
            parent_dir = Path(__file__).parent.parent
            results_dir = parent_dir / "results"
            if not results_dir.exists():
                return None
            
            # Find the most recent forecast for this coin
            coin_dirs = [d for d in results_dir.iterdir() if d.is_dir() and coin in d.name.lower()]
            if not coin_dirs:
                return None
            
            latest_dir = max(coin_dirs, key=lambda x: x.stat().st_mtime)
            forecast_file = latest_dir / "forecast_results.json"
            
            if forecast_file.exists():
                with open(forecast_file, 'r') as f:
                    data = json.load(f)
                    return data.get('forecast', {})
            
        except Exception as e:
            self.logger.error(f"Error parsing forecast result for {coin}: {e}")
        
        return None
    
    async def _validate_forecast_after_delay(self, forecast: ForecastResult, delay_hours: int):
        """Validate a forecast after the specified delay"""
        await asyncio.sleep(delay_hours * 3600)
        
        try:
            # Get current price
            actual_price = self._get_current_price(forecast.crypto)
            forecast.actual_price = actual_price
            
            # Calculate return
            forecast.return_pct = ((actual_price - forecast.price_at_forecast) / forecast.price_at_forecast) * 100
            
            # Determine accuracy
            if forecast.forecast_direction == ForecastDirection.UP:
                forecast.accuracy = actual_price > forecast.price_at_forecast
            elif forecast.forecast_direction == ForecastDirection.DOWN:
                forecast.accuracy = actual_price < forecast.price_at_forecast
            else:  # NEUTRAL
                forecast.accuracy = abs(forecast.return_pct) < 1.0
            
            self.logger.info(
                f"Validated {forecast.crypto}: "
                f"Predicted {forecast.forecast_direction.value}, "
                f"Return: {forecast.return_pct:.2f}%, "
                f"Correct: {forecast.accuracy}"
            )
            
        except Exception as e:
            self.logger.error(f"Error validating forecast for {forecast.crypto}: {e}")
    
    def _calculate_metrics(self, results: List[ForecastResult]) -> ValidationMetrics:
        """Calculate comprehensive validation metrics"""
        if not results:
            return ValidationMetrics(0, 0, 0, {}, {}, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Filter out results without accuracy data
        valid_results = [r for r in results if r.accuracy is not None]
        
        if not valid_results:
            return ValidationMetrics(0, 0, 0, {}, {}, 0, 0, 0, 0, 0, 0, 0)
        
        total_predictions = len(valid_results)
        correct_predictions = sum(1 for r in valid_results if r.accuracy)
        accuracy_percentage = (correct_predictions / total_predictions) * 100
        
        # Calculate metrics by confidence level
        precision_by_confidence = {}
        recall_by_confidence = {}
        
        for confidence in ConfidenceLevel:
            conf_results = [r for r in valid_results if r.confidence == confidence]
            if conf_results:
                correct = sum(1 for r in conf_results if r.accuracy)
                precision_by_confidence[confidence.value] = (correct / len(conf_results)) * 100
            else:
                precision_by_confidence[confidence.value] = 0
        
        # Financial metrics
        returns = [r.return_pct for r in valid_results if r.return_pct is not None]
        average_return = np.mean(returns) if returns else 0
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = (np.mean(returns) / np.std(returns)) if returns and np.std(returns) > 0 else 0
        
        # Max drawdown
        cumulative_returns = np.cumsum(returns) if returns else [0]
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak)
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Win/loss metrics
        winning_trades = sum(1 for r in returns if r > 0)
        losing_trades = sum(1 for r in returns if r < 0)
        winning_returns = [r for r in returns if r > 0]
        losing_returns = [r for r in returns if r < 0]
        
        average_winning_return = np.mean(winning_returns) if winning_returns else 0
        average_losing_return = np.mean(losing_returns) if losing_returns else 0
        
        return ValidationMetrics(
            total_predictions=total_predictions,
            correct_predictions=correct_predictions,
            accuracy_percentage=accuracy_percentage,
            precision_by_confidence=precision_by_confidence,
            recall_by_confidence=recall_by_confidence,
            average_return=average_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            average_winning_return=average_winning_return,
            average_losing_return=average_losing_return
        )
    
    def _save_live_results(self, results: List[ForecastResult], start_time: datetime, end_time: datetime):
        """Save live validation results"""
        filename = f"live_validation_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.results_dir / filename
        
        data = {
            "validation_type": "live",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_hours": (end_time - start_time).total_seconds() / 3600,
            "results": [self._forecast_result_to_dict(r) for r in results],
            "metrics": self._metrics_to_dict(self._calculate_metrics(results))
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Live validation results saved to {filepath}")
    
    def _save_backtest_results(self, results: List[ForecastResult], metrics: ValidationMetrics, days_back: int):
        """Save backtesting results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_validation_{days_back}days_{timestamp}.json"
        filepath = self.results_dir / filename
        
        data = {
            "validation_type": "backtest",
            "days_back": days_back,
            "timestamp": timestamp,
            "results": [self._forecast_result_to_dict(r) for r in results],
            "metrics": self._metrics_to_dict(metrics)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Backtest validation results saved to {filepath}")
    
    def _forecast_result_to_dict(self, result: ForecastResult) -> Dict[str, Any]:
        """Convert ForecastResult to dictionary"""
        return {
            "crypto": result.crypto,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None,
            "forecast_direction": result.forecast_direction.value,
            "confidence": result.confidence.value,
            "predicted_price": result.predicted_price,
            "actual_price": result.actual_price,
            "price_at_forecast": result.price_at_forecast,
            "time_horizon_hours": result.time_horizon_hours,
            "accuracy": result.accuracy,
            "return_pct": result.return_pct
        }
    
    def _metrics_to_dict(self, metrics: ValidationMetrics) -> Dict[str, Any]:
        """Convert ValidationMetrics to dictionary"""
        return {
            "total_predictions": metrics.total_predictions,
            "correct_predictions": metrics.correct_predictions,
            "accuracy_percentage": metrics.accuracy_percentage,
            "precision_by_confidence": metrics.precision_by_confidence,
            "recall_by_confidence": metrics.recall_by_confidence,
            "average_return": metrics.average_return,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "winning_trades": metrics.winning_trades,
            "losing_trades": metrics.losing_trades,
            "average_winning_return": metrics.average_winning_return,
            "average_losing_return": metrics.average_losing_return
        }

    def _get_current_price(self, coin: str) -> float:
        """Get current price for a coin using the CoinGeckoTool interface"""
        try:
            query = f"{coin} current price"
            price_data_json = self.coingecko._run(query=query)
            
            if not price_data_json:
                self.logger.error(f"No price data response for {coin}")
                return 0.0
            
            # Parse the JSON response
            import json
            price_data = json.loads(price_data_json)
            
            # Check for errors
            if "error" in price_data:
                self.logger.error(f"Error getting price for {coin}: {price_data['error']}")
                return 0.0
            
            # Extract current price
            current_price = price_data.get('current_price', 0.0)
            if current_price == 0.0:
                self.logger.warning(f"Zero price returned for {coin}")
            
            return current_price
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {coin}: {e}")
            return 0.0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CryptoAgentForecaster Validation Framework")
    parser.add_argument("--mode", choices=["live", "backtest"], required=True,
                       help="Validation mode")
    parser.add_argument("--duration", type=int, default=24,
                       help="Duration in hours for live validation")
    parser.add_argument("--interval", type=int, default=1,
                       help="Interval in hours for live validation")
    parser.add_argument("--days-back", type=int, default=30,
                       help="Days of historical data for backtesting")
    parser.add_argument("--coins", nargs="+", default=["bitcoin"],
                       help="Coins to test")
    
    args = parser.parse_args()
    
    validator = CryptoValidator()
    
    async def main():
        if args.mode == "live":
            await validator.run_live_validation(
                duration_hours=args.duration,
                interval_hours=args.interval,
                coins=args.coins
            )
        elif args.mode == "backtest":
            await validator.run_backtesting_validation(
                days_back=args.days_back,
                coins=args.coins
            )
    
    asyncio.run(main()) 