# 🔮 Crypto Agent Forecaster Validation - Quick Overview

## 📁 File Structure & Functions

### 🎯 Main Scripts
- **`cli.py`** - Main command-line interface for all validation functions
- **`validator.py`** - Core validation engine with backtesting and live testing
- **`analytics.py`** - Advanced analytics and HTML report generation
- **`vps_deployment.py`** - VPS deployment and continuous monitoring

### 🔧 Helper Scripts  
- **`run_background.sh`** - Background execution with screen/nohup
- **`monitor.sh`** - Process monitoring and log management
- **`help.sh`** - Quick help and command reference
- **`setup.sh`** - Installation and environment setup

### 📚 Documentation
- **`README.md`** - Comprehensive documentation (13KB)
- **`OVERVIEW.md`** - This quick reference
- **`example_usage.py`** - Python API examples

## ⚡ Quick Start Commands

```bash
# 1. Quick test (5 seconds)
python cli.py backtest --days 7 -c bitcoin -c ethereum

# 2. Background comprehensive test
./run_background.sh nohup full-backtest

# 3. Monitor progress
./monitor.sh follow

# 4. Check results
python cli.py status
```

## 🎯 All Available Commands

### CLI Commands (`python cli.py`)
| Command | Function | Speed | Use Case |
|---------|----------|-------|----------|
| `backtest` | Historical data testing | Instant | Custom analysis |
| `full-backtest` | All 10 coins, 30 days | ~15s | Comprehensive research |
| `quick-test` | Live Bitcoin test | 6 hours | Quick validation |
| `full-test` | Live all coins test | 6 hours | Full live validation |
| `live` | Custom live testing | Variable | Specific research |
| `status` | Show recent results | Instant | Results review |
| `report` | Generate HTML reports | Instant | Professional reporting |
| `deploy` | VPS deployment | Variable | Production setup |

### Background Scripts
| Script | Function | Best For |
|--------|----------|----------|
| `./run_background.sh screen [test]` | Interactive background | Live monitoring |
| `./run_background.sh nohup [test]` | Unattended background | Long-term testing |
| `./monitor.sh` | Full status check | System overview |
| `./monitor.sh follow` | Live log following | Real-time monitoring |
| `./monitor.sh kill-all` | Emergency stop | Process cleanup |

### Help & Support
| Command | Function |
|---------|----------|
| `./help.sh` | Quick commands |
| `./help.sh examples` | Usage examples |
| `./help.sh troubleshoot` | Problem solving |
| `./help.sh coins` | Supported cryptocurrencies |
| `python cli.py --help` | Detailed CLI help |

## 🎲 Test Types Summary

### 1. **Quick Backtest** (5 seconds)
```bash
python cli.py backtest --days 7 -c bitcoin -c ethereum
```
- **Purpose**: Fast verification, development testing
- **Results**: Usually 75-85% accuracy

### 2. **Full Backtest** (15 seconds)  
```bash
python cli.py full-backtest --days 30
./run_background.sh nohup full-backtest
```
- **Purpose**: Comprehensive analysis, research
- **Results**: Usually 65-85% accuracy, 830 predictions

### 3. **Live Tests** (6 hours)
```bash
./run_background.sh screen full-test
```
- **Purpose**: Real-world validation
- **Results**: Market-dependent, real-time updates

## 💰 Supported Cryptocurrencies (10 total)

```
bitcoin, ethereum, solana, cardano, polygon,
chainlink, avalanche-2, polkadot, uniswap, litecoin
```

## 📊 Performance Benchmarks

- **Excellent**: >60% accuracy, Sharpe ratio >1.0
- **Good**: 55-60% accuracy, Sharpe ratio >0.5  
- **Needs work**: <55% accuracy

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "No processes running" | Backtests finish fast, check `./monitor.sh results` |
| "Permission denied" | Run `chmod +x *.sh` |
| "No historical data" | Use CoinGecko IDs: `bitcoin` not `BTC` |
| "Module not found" | Ensure in `/validation` directory |

## 🎯 Recommended Workflow

1. **First Time**: `python cli.py backtest --days 7 -c bitcoin -c ethereum`
2. **Development**: `./run_background.sh nohup quick-backtest`  
3. **Research**: `./run_background.sh nohup full-backtest`
4. **Production**: `./run_background.sh screen full-test`
5. **Monitoring**: `./monitor.sh follow`
6. **Results**: `python cli.py status` → `python cli.py report`

---
**🔮 For detailed documentation, see `README.md` or run `./help.sh`** 