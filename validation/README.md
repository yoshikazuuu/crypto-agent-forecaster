# 🔮 Crypto Agent Forecaster - Validation Framework

A comprehensive validation system to test the accuracy and performance of cryptocurrency forecasting models. This framework provides scientific validation, backtesting, live testing, and automated monitoring capabilities.

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🎯 Core Commands](#-core-commands)
- [🖥️ Background Execution](#️-background-execution)
- [📊 Monitoring & Management](#-monitoring--management)
- [📈 Test Types](#-test-types)
- [🔧 Advanced Usage](#-advanced-usage)
- [📁 File Structure](#-file-structure)
- [🆘 Troubleshooting](#-troubleshooting)

## 🚀 Quick Start

```bash
# Navigate to validation directory
cd validation

# Quick 7-day backtest (2 coins, instant results)
python cli.py backtest --days 7 -c bitcoin -c ethereum

# Comprehensive 30-day backtest (all 10 coins)
python cli.py full-backtest --days 30

# Background execution with monitoring
./run_background.sh nohup full-backtest
./monitor.sh follow
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- Virtual environment activated
- CoinGecko API access
- Screen (for background sessions)

### Setup
```bash
# Ensure you're in the validation directory
cd crypto-agent-forecaster/validation

# Install dependencies (if not already done)
pip install -r ../requirements.txt

# Make scripts executable
chmod +x run_background.sh monitor.sh

# Test installation
python cli.py --help
```

## 🎯 Core Commands

### Main CLI Interface (`python cli.py`)

#### 📊 Backtesting Commands
```bash
# Quick backtest (customizable days and coins)
python cli.py backtest --days 7 -c bitcoin -c ethereum
python cli.py backtest --days 30 -c solana -c cardano -c polygon

# Comprehensive backtest (all 10 supported coins)
python cli.py full-backtest --days 30
python cli.py full-backtest --days 7

# Supported cryptocurrencies:
# bitcoin, ethereum, solana, cardano, polygon, 
# chainlink, avalanche-2, polkadot, uniswap, litecoin
```

#### 🔴 Live Validation Commands
```bash
# Quick live test (6 hours, Bitcoin only)
python cli.py quick-test

# Comprehensive live test (6 hours, all 10 coins)
python cli.py full-test

# Custom live validation
python cli.py live --duration 12 --interval 2 -c bitcoin -c ethereum
```

#### 📋 Status and Reporting
```bash
# View recent validation results
python cli.py status

# Generate comprehensive HTML report
python cli.py report

# Generate report to specific directory
python cli.py report --output ./reports --no-open
```

#### 🚀 VPS Deployment
```bash
# Deploy to VPS with continuous validation
python cli.py deploy --install-deps --create-service
```

### Command Reference Table

| Command | Description | Duration | Coins | Output |
|---------|-------------|----------|-------|---------|
| `backtest` | Custom backtesting | Configurable | Configurable | Instant |
| `full-backtest` | Comprehensive backtest | 30 days (default) | All 10 coins | Instant |
| `quick-test` | Quick live validation | 6 hours | Bitcoin | Real-time |
| `full-test` | Comprehensive live test | 6 hours | All 10 coins | Real-time |
| `live` | Custom live validation | Configurable | Configurable | Real-time |
| `status` | Show recent results | - | - | Instant |
| `report` | Generate HTML report | - | - | Instant |

## 🖥️ Background Execution

### Background Runner (`./run_background.sh`)

#### Screen Sessions (Recommended for Interactive)
```bash
# Start background test in screen session
./run_background.sh screen quick-backtest
./run_background.sh screen full-backtest
./run_background.sh screen full-test

# Manage screen sessions
screen -ls                           # List all sessions
screen -r crypto-validation-[ID]     # Attach to session
# Press Ctrl+A, D to detach (keeps running)
screen -X -S [session-name] quit     # Kill session
```

#### Nohup Background (Recommended for Unattended)
```bash
# Start background test with logging
./run_background.sh nohup quick-backtest
./run_background.sh nohup full-backtest
./run_background.sh nohup full-test

# Custom command
./run_background.sh nohup custom
# Then enter: backtest --days 60 -c bitcoin -c ethereum -c solana
```

#### Background Options Comparison

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| `screen` | Interactive monitoring | Easy attach/detach, visual | Requires screen knowledge |
| `nohup` | Unattended operation | Simple, automatic logging | No interaction |
| `service` | Production deployment | Automatic restart, system integration | Complex setup |

## 📊 Monitoring & Management

### Monitor Script (`./monitor.sh`)

#### Quick Status Check
```bash
# Full status overview (default)
./monitor.sh

# Check specific components
./monitor.sh processes    # Running validation processes
./monitor.sh logs        # Recent log files
./monitor.sh results     # Recent validation results
```

#### Real-time Monitoring
```bash
# Follow latest log in real-time
./monitor.sh follow

# Follow specific log file
tail -f validation_[test-type]_[timestamp].log

# Monitor process status
ps aux | grep "python cli.py"
```

#### Process Management
```bash
# Emergency stop all validation processes
./monitor.sh kill-all

# Kill specific process
kill [PID]

# Check if process is still running
ps aux | grep [PID]
```

### Log File Patterns
```
validation_quick-backtest_20250531_045506.log
validation_full-backtest_20250531_045557.log
validation_full-test_20250531_120000.log
validation_custom_20250531_130000.log
```

## 📈 Test Types

### 1. Quick Backtest
- **Duration**: 7 days historical data
- **Coins**: Bitcoin + Ethereum (2 coins)
- **Execution Time**: ~5 seconds
- **Use Case**: Quick validation, testing changes

```bash
python cli.py backtest --days 7 -c bitcoin -c ethereum
./run_background.sh nohup quick-backtest
```

### 2. Full Backtest
- **Duration**: 30 days historical data (configurable)
- **Coins**: All 10 supported cryptocurrencies
- **Execution Time**: ~10-15 seconds
- **Use Case**: Comprehensive analysis, research

```bash
python cli.py full-backtest --days 30
./run_background.sh nohup full-backtest
```

### 3. Quick Live Test
- **Duration**: 6 hours real-time
- **Coins**: Bitcoin only
- **Execution Time**: 6 hours
- **Use Case**: Quick live validation

```bash
python cli.py quick-test
./run_background.sh screen quick-test  # Recommended for live tests
```

### 4. Full Live Test
- **Duration**: 6 hours real-time
- **Coins**: All 10 supported cryptocurrencies
- **Execution Time**: 6 hours
- **Use Case**: Comprehensive live validation

```bash
python cli.py full-test
./run_background.sh screen full-test  # Recommended for live tests
```

### 5. Custom Validation
- **Duration**: Configurable
- **Coins**: Configurable
- **Execution Time**: Variable
- **Use Case**: Specific research needs

```bash
# Custom backtest
python cli.py backtest --days 60 -c bitcoin -c ethereum -c solana

# Custom live test
python cli.py live --duration 24 --interval 1 -c cardano -c polygon
```

## 🔧 Advanced Usage

### Performance Metrics Explained

| Metric | Description | Good Performance |
|--------|-------------|------------------|
| **Overall Accuracy** | % of correct predictions | > 55% (60%+ excellent) |
| **Total Predictions** | Number of forecasts tested | Varies by test type |
| **Average Return** | Mean return per prediction | Close to 0% (market neutral) |
| **Sharpe Ratio** | Risk-adjusted return | > 0.5 (1.0+ excellent) |
| **Win Rate** | % of profitable predictions | > 50% |

### Supported Cryptocurrencies

1. **bitcoin** - Bitcoin (BTC)
2. **ethereum** - Ethereum (ETH)
3. **solana** - Solana (SOL)
4. **cardano** - Cardano (ADA)
5. **polygon** - Polygon (MATIC)
6. **chainlink** - Chainlink (LINK)
7. **avalanche-2** - Avalanche (AVAX)
8. **polkadot** - Polkadot (DOT)
9. **uniswap** - Uniswap (UNI)
10. **litecoin** - Litecoin (LTC)

### Configuration Options

```bash
# Backtest options
--days [number]           # Historical days to test (default: 30)
-c [coin] -c [coin]      # Cryptocurrencies to test (multiple -c flags)
--verbose                # Verbose output

# Live validation options
--duration [hours]       # Total test duration (default: 24)
--interval [hours]       # Forecast interval (default: 1)
-c [coin] -c [coin]      # Cryptocurrencies to test

# Report options
--output [directory]     # Output directory for reports
--no-open               # Don't open report in browser
```

### Environment Variables

```bash
# Optional: Set CoinGecko API key for higher rate limits
export COINGECKO_API_KEY="your-api-key-here"

# Optional: Adjust API rate limiting
export API_RATE_LIMIT_DELAY="1.5"
```

## 📁 File Structure

```
validation/
├── 📄 README.md                    # This documentation
├── 🐍 cli.py                       # Main CLI interface
├── 🐍 validator.py                 # Core validation engine
├── 🐍 analytics.py                 # Advanced analytics and reporting
├── 🐍 vps_deployment.py           # VPS deployment automation
├── 🔧 run_background.sh           # Background execution runner
├── 📊 monitor.sh                  # Monitoring and management
├── 📁 validation_results/          # JSON result files
├── 📄 validation_*.log             # Log files (generated)
└── 📁 __pycache__/                # Python cache (ignore)
```

### Generated Files

- **`validation_results/*.json`** - Detailed validation results
- **`validation_*.log`** - Execution logs from background runs
- **`reports/*.html`** - Generated HTML reports (if using report command)

## 🆘 Troubleshooting

### Common Issues

#### 1. "No validation processes currently running"
```bash
# Check if validation completed quickly (backtests are fast)
./monitor.sh results

# For live tests, check if they're still running
./monitor.sh processes
```

#### 2. "No historical data for [coin]"
```bash
# Check internet connection
ping api.coingecko.com

# Verify coin name (use CoinGecko IDs)
python cli.py backtest --days 7 -c bitcoin  # Not "BTC"
```

#### 3. "Screen session terminated immediately"
```bash
# Check for errors in the command
screen -ls  # Should show no sessions if completed quickly

# Use nohup instead for better error logging
./run_background.sh nohup quick-backtest
```

#### 4. "Permission denied"
```bash
# Make scripts executable
chmod +x run_background.sh monitor.sh

# Check file permissions
ls -la *.sh
```

#### 5. "Module not found" errors
```bash
# Ensure you're in the validation directory
cd crypto-agent-forecaster/validation

# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall dependencies if needed
pip install -r ../requirements.txt
```

### Performance Issues

#### Slow API responses
```bash
# Set API key for higher rate limits
export COINGECKO_API_KEY="your-key"

# Reduce number of coins for faster testing
python cli.py backtest --days 7 -c bitcoin -c ethereum
```

#### High memory usage
```bash
# Use shorter test periods
python cli.py backtest --days 7  # Instead of 30+ days

# Test fewer coins at once
python cli.py backtest --days 30 -c bitcoin -c ethereum
```

### Getting Help

#### Built-in Help
```bash
# Main CLI help
python cli.py --help

# Command-specific help
python cli.py backtest --help
python cli.py live --help

# Background runner help
./run_background.sh help

# Monitor help
./monitor.sh help
```

#### Debug Mode
```bash
# Run with verbose output
python cli.py backtest --days 7 -c bitcoin --verbose

# Check detailed logs
tail -f validation_results/validation.log
```

#### Example Troubleshooting Session
```bash
# 1. Check current status
./monitor.sh

# 2. Look at recent logs
./monitor.sh logs

# 3. Check for running processes
./monitor.sh processes

# 4. If stuck, clean restart
./monitor.sh kill-all
python cli.py backtest --days 7 -c bitcoin  # Simple test

# 5. Check results
python cli.py status
```

---

## 🎯 Quick Reference Card

### Essential Commands
```bash
# Testing
python cli.py backtest --days 7 -c bitcoin -c ethereum    # Quick test
python cli.py full-backtest --days 30                     # Comprehensive test

# Background execution
./run_background.sh nohup full-backtest                   # Background + logs
./run_background.sh screen full-test                      # Interactive background

# Monitoring
./monitor.sh                                              # Full status
./monitor.sh follow                                       # Live log following
./monitor.sh kill-all                                     # Emergency stop

# Results
python cli.py status                                      # Recent results
python cli.py report                                      # HTML report
```

### Best Practices
1. **Start small**: Use `quick-backtest` first to verify setup
2. **Use nohup**: For unattended long-running tests
3. **Monitor logs**: Use `./monitor.sh follow` for live tests
4. **Check results**: Use `python cli.py status` after completion
5. **Clean up**: Use `./monitor.sh kill-all` if needed

### Performance Expectations
- **Quick Backtest**: ~5 seconds, 75-85% accuracy typical
- **Full Backtest**: ~15 seconds, 65-85% accuracy typical
- **Live Tests**: 6 hours real-time, varies by market conditions

---

**🔮 Happy Forecasting! May your predictions be ever accurate! 🚀** 