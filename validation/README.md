# 🔮 Crypto Agent Forecaster Validation Framework

A comprehensive, scientific validation toolkit for testing the accuracy and performance of your crypto forecasting AI agents. This framework provides automated backtesting, live validation, advanced analytics, and VPS deployment capabilities.

## 🌟 Features

### 📊 Validation Types
- **Live Validation**: Real-time forecasting with actual market data
- **Backtesting**: Historical data simulation for rapid testing
- **Multi-Coin Analysis**: Test across multiple cryptocurrencies simultaneously
- **Continuous Monitoring**: 24/7 automated validation on VPS

### 📈 Scientific Metrics
- **Accuracy Analysis**: Overall and per-direction accuracy
- **Financial Performance**: Returns, Sharpe ratio, max drawdown
- **Risk Assessment**: Value at Risk (VaR), volatility analysis
- **Confidence Scoring**: Performance by confidence levels
- **Time-Based Analysis**: Performance by hour and day of week

### 📋 Professional Reporting
- **HTML Reports**: Comprehensive visual reports with charts
- **CSV Exports**: Raw data for further analysis
- **Email Notifications**: Automated report delivery
- **Real-time Dashboards**: Live performance monitoring

### 🚀 VPS Deployment
- **Automated Setup**: One-command deployment to VPS
- **System Monitoring**: Resource usage and health checks
- **Error Recovery**: Automatic restart on failures
- **Email Alerts**: Instant notifications of issues

## 🛠️ Installation

### Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- Access to your crypto forecasting tool
- API keys for CoinGecko (optional for Pro features)
- Email account for notifications (optional)

### Quick Setup

1. **Install uv (if not already installed)**
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

2. **Install the validation framework**
```bash
cd validation/
uv pip install -e .
```

3. **Run first test**
```bash
uv run python cli.py quick-test
```

## 🚀 Quick Start

### 1. Quick Test (Recommended First Step)
```bash
uv run python cli.py quick-test
```
Runs a 6-hour Bitcoin validation to test the system.

### 2. Live Validation
```bash
# 24-hour validation with multiple coins
uv run python cli.py live --duration 24 --coins bitcoin ethereum solana

# Quick 6-hour test
uv run python cli.py live -d 6 -c bitcoin
```

### 3. Backtesting
```bash
# 30-day backtest with top 5 coins
uv run python cli.py backtest --days 30

# Quick 7-day test with specific coins
uv run python cli.py backtest -d 7 -c bitcoin ethereum
```

### 4. Generate Reports
```bash
# Generate comprehensive HTML report
uv run python cli.py report

# Generate report without opening browser
uv run python cli.py report --no-open
```

### 5. Check Status
```bash
uv run python cli.py status
```

## 🖥️ VPS Deployment

### Automated Setup
```bash
# Install dependencies and create system service
sudo uv run python cli.py deploy --install-deps --create-service

# Start continuous validation
uv run python cli.py deploy
```

### Manual Setup
1. **Create Configuration**
```bash
uv run python cli.py deploy  # Creates vps_config.ini
```

2. **Edit Configuration**
```ini
[validation]
interval_hours = 1
test_coins = bitcoin,ethereum,solana,cardano,polygon

[email]
enabled = true
username = your_email@gmail.com
password = your_app_password
recipients = recipient@email.com

[resources]
max_memory_mb = 2048
max_cpu_percent = 80.0
```

3. **Deploy**
```bash
uv run python cli.py deploy --config vps_config.ini
```

## 📊 Understanding the Results

### Accuracy Metrics
- **Overall Accuracy**: Percentage of correct predictions
- **Direction Accuracy**: Accuracy by UP/DOWN/NEUTRAL predictions
- **Confidence Accuracy**: Performance by HIGH/MEDIUM/LOW confidence

### Financial Metrics
- **Average Return**: Mean return per prediction
- **Sharpe Ratio**: Risk-adjusted returns (>1.0 is excellent)
- **Max Drawdown**: Largest loss from peak
- **Win Rate**: Percentage of profitable predictions

### Performance Benchmarks
- **>60% Accuracy**: Excellent performance
- **50-60% Accuracy**: Good performance
- **<50% Accuracy**: Needs improvement
- **Sharpe Ratio >1.0**: Strong risk-adjusted returns

## 🔧 Advanced Usage

### Custom Validation Scripts

```python
from validation.validator import CryptoValidator
from validation.analytics import ValidationAnalytics

# Custom validation
validator = CryptoValidator()

# Run backtesting
metrics = await validator.run_backtesting_validation(
    days_back=60,
    coins=["bitcoin", "ethereum", "solana"]
)

# Generate analytics
analytics = ValidationAnalytics()
report = analytics.generate_comprehensive_report()
```

### Programmatic Access

```python
# Load validation results
import json
from pathlib import Path

results_dir = Path("validation_results")
for result_file in results_dir.glob("*.json"):
    with open(result_file) as f:
        data = json.load(f)
        print(f"Accuracy: {data['metrics']['accuracy_percentage']:.1f}%")
```

### Development Setup

```bash
# Install with development dependencies
cd validation/
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Format code
uv run black .
uv run ruff check --fix .
```

## 📁 Directory Structure

```
validation/
├── validator.py          # Main validation engine
├── analytics.py          # Analytics and reporting
├── vps_deployment.py     # VPS automation
├── cli.py               # Command-line interface
├── README.md            # This file
├── pyproject.toml       # Dependencies and configuration
└── examples/            # Example configurations

validation_results/
├── live_validation_*.json      # Live validation results
├── backtest_validation_*.json  # Backtesting results
├── reports/                    # Generated reports
│   ├── validation_report_*.html
│   ├── validation_metrics_*.csv
│   └── *.png                   # Charts
└── validation.log              # Logs

vps_deployment/
├── validation_results/    # VPS validation results
├── logs/                 # VPS logs
└── vps_config.ini       # VPS configuration
```

## 🔍 Troubleshooting

### Common Issues

1. **"No validation results found"**
   - Run `uv run python cli.py quick-test` first
   - Check that your main forecasting tool is working
   - Verify API keys are configured

2. **Forecast tool not found**
   - Ensure `main.py` exists in the project root
   - Check that the forecast command works: `python main.py forecast bitcoin`

3. **Email notifications not working**
   - Use app passwords, not regular passwords
   - Enable 2FA and generate app password
   - Check SMTP settings in configuration

4. **VPS deployment issues**
   - Run with `sudo` for system service creation
   - Check firewall settings
   - Ensure Python path is correct

5. **uv installation issues**
   - Ensure uv is in your PATH: `which uv`
   - Try reinstalling: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Check uv version: `uv --version`

### Debug Mode
```bash
# Run with verbose logging
uv run python cli.py live --verbose

# Check system logs
journalctl -u crypto-validator.service -f
```

## 📈 Performance Optimization

### For Better Accuracy
1. **Increase data sources**: Add more technical indicators
2. **Tune parameters**: Adjust confidence thresholds
3. **Focus on best assets**: Identify top-performing cryptocurrencies
4. **Time analysis**: Optimize for best-performing hours/days

### For Faster Execution
1. **Reduce coins**: Test fewer cryptocurrencies initially
2. **Shorter horizons**: Use shorter backtesting periods
3. **Optimize intervals**: Adjust validation frequency
4. **Resource limits**: Set appropriate CPU/memory limits

## 🚀 Using uv for Performance

The validation framework uses [uv](https://github.com/astral-sh/uv) for fast and reliable package management:

- **10-100x faster** than pip for package installation
- **Reproducible** installs with lockfiles
- **Cross-platform** support
- **Zero configuration** required

### uv Commands
```bash
# Install dependencies
uv pip install -e .

# Run scripts
uv run python cli.py [command]

# Sync environment (for development)
uv pip sync

# Add new dependency
uv add package-name
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `uv pip install -e ".[dev]"`
4. Add tests for new functionality
5. Run tests: `uv run pytest`
6. Format code: `uv run black . && uv run ruff check --fix .`
7. Update documentation
8. Submit a pull request

## 📜 License

This validation framework is part of the Crypto Agent Forecaster project.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `validation_results/validation.log`
3. Create an issue with error details and configuration

---

**Happy Validating! 🔮📊** 