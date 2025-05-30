#!/bin/bash
# 🆘 Quick Help for Crypto Agent Forecaster Validation Framework

echo "🆘 Crypto Agent Forecaster - Quick Help"
echo "======================================="

show_quick_commands() {
    echo ""
    echo "⚡ Most Common Commands:"
    echo "----------------------"
    echo "  python cli.py backtest --days 7 -c bitcoin -c ethereum    # Quick test"
    echo "  python cli.py full-backtest --days 30                     # Full test"
    echo "  ./run_background.sh nohup full-backtest                   # Background"
    echo "  ./monitor.sh                                              # Check status"
    echo "  ./monitor.sh follow                                       # Watch logs"
    echo ""
}

show_all_commands() {
    echo ""
    echo "📋 All Available Commands:"
    echo "-------------------------"
    echo ""
    echo "🎯 Core Testing:"
    echo "  python cli.py backtest --days 7 -c bitcoin -c ethereum"
    echo "  python cli.py full-backtest --days 30"
    echo "  python cli.py quick-test"
    echo "  python cli.py full-test"
    echo "  python cli.py live --duration 12 --interval 2 -c bitcoin"
    echo ""
    echo "🖥️ Background Execution:"
    echo "  ./run_background.sh screen quick-backtest"
    echo "  ./run_background.sh nohup full-backtest"
    echo "  ./run_background.sh screen full-test"
    echo ""
    echo "📊 Monitoring:"
    echo "  ./monitor.sh                    # Full status"
    echo "  ./monitor.sh processes          # Running processes"
    echo "  ./monitor.sh logs              # Recent log files"
    echo "  ./monitor.sh results           # Recent results"
    echo "  ./monitor.sh follow            # Follow latest log"
    echo "  ./monitor.sh kill-all          # Stop all processes"
    echo ""
    echo "📈 Results & Reports:"
    echo "  python cli.py status           # Recent validation results"
    echo "  python cli.py report           # Generate HTML report"
    echo ""
}

show_examples() {
    echo ""
    echo "🎯 Common Usage Examples:"
    echo "------------------------"
    echo ""
    echo "1. 🚀 Quick Start (First time):"
    echo "   python cli.py backtest --days 7 -c bitcoin -c ethereum"
    echo ""
    echo "2. 📊 Comprehensive Test:"
    echo "   ./run_background.sh nohup full-backtest"
    echo "   ./monitor.sh follow"
    echo ""
    echo "3. 🔴 Live 6-Hour Test:"
    echo "   ./run_background.sh screen full-test"
    echo "   # Press Ctrl+A, D to detach"
    echo "   screen -r crypto-validation-[TAB]  # To reattach later"
    echo ""
    echo "4. 🔍 Custom Research:"
    echo "   python cli.py backtest --days 60 -c bitcoin -c ethereum -c solana"
    echo ""
    echo "5. 📋 Check Results:"
    echo "   ./monitor.sh results"
    echo "   python cli.py status"
    echo "   python cli.py report"
    echo ""
}

show_troubleshooting() {
    echo ""
    echo "🆘 Quick Troubleshooting:"
    echo "------------------------"
    echo ""
    echo "❌ Problem: 'No validation processes running'"
    echo "   → Backtests finish quickly! Check: ./monitor.sh results"
    echo ""
    echo "❌ Problem: 'Permission denied'"
    echo "   → Fix: chmod +x run_background.sh monitor.sh help.sh"
    echo ""
    echo "❌ Problem: 'No historical data for [coin]'"
    echo "   → Use CoinGecko IDs: bitcoin (not BTC), ethereum (not ETH)"
    echo ""
    echo "❌ Problem: 'Module not found'"
    echo "   → Ensure you're in validation/ directory"
    echo "   → Check: pip install -r ../requirements.txt"
    echo ""
    echo "💡 Emergency Reset:"
    echo "   ./monitor.sh kill-all"
    echo "   python cli.py backtest --days 7 -c bitcoin"
    echo ""
}

show_performance() {
    echo ""
    echo "📊 Performance Expectations:"
    echo "---------------------------"
    echo ""
    echo "⚡ Quick Backtest (7 days, 2 coins):"
    echo "   • Duration: ~5 seconds"
    echo "   • Typical Accuracy: 75-85%"
    echo "   • Use Case: Testing, verification"
    echo ""
    echo "🚀 Full Backtest (30 days, 10 coins):"
    echo "   • Duration: ~15 seconds"
    echo "   • Typical Accuracy: 65-85%"
    echo "   • Use Case: Comprehensive analysis"
    echo ""
    echo "🔴 Live Tests (6 hours real-time):"
    echo "   • Duration: 6 hours"
    echo "   • Accuracy: Varies by market conditions"
    echo "   • Use Case: Real-world validation"
    echo ""
    echo "🎯 Good Performance Indicators:"
    echo "   • Accuracy > 55% (60%+ excellent)"
    echo "   • Sharpe Ratio > 0.5 (1.0+ excellent)"
    echo "   • Win Rate > 50%"
    echo ""
}

show_coins() {
    echo ""
    echo "💰 Supported Cryptocurrencies:"
    echo "------------------------------"
    echo ""
    echo "Use these exact names with -c flag:"
    echo ""
    echo "  1. bitcoin        - Bitcoin (BTC)"
    echo "  2. ethereum       - Ethereum (ETH)"
    echo "  3. solana         - Solana (SOL)"
    echo "  4. cardano        - Cardano (ADA)"
    echo "  5. polygon        - Polygon (MATIC)"
    echo "  6. chainlink      - Chainlink (LINK)"
    echo "  7. avalanche-2    - Avalanche (AVAX)"
    echo "  8. polkadot      - Polkadot (DOT)"
    echo "  9. uniswap        - Uniswap (UNI)"
    echo " 10. litecoin       - Litecoin (LTC)"
    echo ""
    echo "Example usage:"
    echo "  python cli.py backtest --days 7 -c bitcoin -c ethereum -c solana"
    echo ""
}

show_usage() {
    echo ""
    echo "📖 Help Topics:"
    echo "--------------"
    echo "  ./help.sh                    # Quick commands (default)"
    echo "  ./help.sh all               # All available commands"
    echo "  ./help.sh examples          # Usage examples"
    echo "  ./help.sh troubleshoot      # Common problems"
    echo "  ./help.sh performance       # Performance expectations"
    echo "  ./help.sh coins             # Supported cryptocurrencies"
    echo "  ./help.sh readme            # Open full README"
    echo ""
    echo "💡 For comprehensive documentation:"
    echo "  cat README.md | less"
    echo "  open README.md              # On macOS"
    echo ""
}

# Main execution
case "${1:-quick}" in
    "quick"|"")
        show_quick_commands
        ;;
    "all"|"commands")
        show_all_commands
        ;;
    "examples"|"example")
        show_examples
        ;;
    "troubleshoot"|"trouble"|"help"|"debug")
        show_troubleshooting
        ;;
    "performance"|"perf"|"metrics")
        show_performance
        ;;
    "coins"|"crypto"|"currencies")
        show_coins
        ;;
    "readme"|"doc"|"docs")
        if command -v less >/dev/null 2>&1; then
            cat README.md | less
        else
            cat README.md
        fi
        ;;
    "usage"|"--help"|"-h")
        show_usage
        ;;
    *)
        echo "❌ Unknown help topic: $1"
        show_usage
        exit 1
        ;;
esac 