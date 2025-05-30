#!/bin/bash
# 🚀 Background Validation Runner for Crypto Agent Forecaster
# 
# This script provides multiple ways to run validation tests in the background

echo "🚀 Crypto Agent Forecaster - Background Validation Runner"
echo "========================================================="

# Function to show usage
show_usage() {
    echo ""
    echo "📋 Available Background Options:"
    echo ""
    echo "1. 🖥️  Screen Session (Recommended)"
    echo "   ./run_background.sh screen [test-type]"
    echo ""
    echo "2. 📝 Background with Logs"
    echo "   ./run_background.sh nohup [test-type]"
    echo ""
    echo "3. 🔄 Systemd Service (Production)"
    echo "   ./run_background.sh service [test-type]"
    echo ""
    echo "📊 Available Test Types:"
    echo "   • quick-backtest    - 7-day backtest (Bitcoin + Ethereum)"
    echo "   • full-backtest     - 30-day backtest (all 10 coins)"
    echo "   • full-test         - 6-hour live test (all 10 coins)"
    echo "   • custom            - Custom command"
    echo ""
    echo "🎯 Examples:"
    echo "   ./run_background.sh screen full-test"
    echo "   ./run_background.sh nohup full-backtest"
    echo "   ./run_background.sh screen quick-backtest"
    echo ""
}

# Function to run in screen
run_in_screen() {
    local test_type=$1
    local session_name="crypto-validation-$(date +%Y%m%d-%H%M%S)"
    
    echo "🖥️  Starting Screen Session: $session_name"
    echo ""
    
    case $test_type in
        "quick-backtest")
            echo "🚀 Running Quick Backtest (7 days, Bitcoin + Ethereum)"
            screen -dmS "$session_name" bash -c "cd $(pwd) && python cli.py backtest --days 7 -c bitcoin -c ethereum; echo 'Test completed. Press any key to exit.'; read"
            ;;
        "full-backtest")
            echo "🚀 Running Full Backtest (30 days, all 10 coins)"
            screen -dmS "$session_name" bash -c "cd $(pwd) && python cli.py full-backtest --days 30; echo 'Test completed. Press any key to exit.'; read"
            ;;
        "full-test")
            echo "🚀 Running Full Live Test (6 hours, all 10 coins)"
            screen -dmS "$session_name" bash -c "cd $(pwd) && python cli.py full-test; echo 'Test completed. Press any key to exit.'; read"
            ;;
        "custom")
            echo "🚀 Enter your custom command:"
            read -p "Command: python cli.py " custom_cmd
            screen -dmS "$session_name" bash -c "cd $(pwd) && python cli.py $custom_cmd; echo 'Test completed. Press any key to exit.'; read"
            ;;
        *)
            echo "❌ Unknown test type: $test_type"
            show_usage
            exit 1
            ;;
    esac
    
    echo "✅ Screen session '$session_name' started!"
    echo ""
    echo "📋 Management Commands:"
    echo "   screen -r $session_name    # Attach to session"
    echo "   screen -ls                 # List all sessions"
    echo "   screen -X -S $session_name quit  # Kill session"
    echo ""
    echo "💡 Tip: Use Ctrl+A, D to detach without stopping the test"
    echo ""
    
    # Wait a moment then show session list
    sleep 2
    echo "📊 Current Screen Sessions:"
    screen -ls
}

# Function to run with nohup
run_with_nohup() {
    local test_type=$1
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="validation_${test_type}_${timestamp}.log"
    
    echo "📝 Running with nohup (background + logs)"
    echo "📄 Log file: $log_file"
    echo ""
    
    case $test_type in
        "quick-backtest")
            echo "🚀 Starting Quick Backtest..."
            nohup python cli.py backtest --days 7 -c bitcoin -c ethereum > "$log_file" 2>&1 &
            ;;
        "full-backtest")
            echo "🚀 Starting Full Backtest..."
            nohup python cli.py full-backtest --days 30 > "$log_file" 2>&1 &
            ;;
        "full-test")
            echo "🚀 Starting Full Live Test..."
            nohup python cli.py full-test > "$log_file" 2>&1 &
            ;;
        "custom")
            echo "🚀 Enter your custom command:"
            read -p "Command: python cli.py " custom_cmd
            nohup python cli.py $custom_cmd > "$log_file" 2>&1 &
            ;;
        *)
            echo "❌ Unknown test type: $test_type"
            show_usage
            exit 1
            ;;
    esac
    
    local pid=$!
    echo "✅ Process started with PID: $pid"
    echo ""
    echo "📋 Management Commands:"
    echo "   tail -f $log_file          # Follow logs"
    echo "   ps aux | grep $pid         # Check process status"
    echo "   kill $pid                  # Stop the test"
    echo ""
    echo "💡 The test is now running in the background!"
}

# Function to create systemd service
create_service() {
    local test_type=$1
    
    echo "🔄 Creating Systemd Service (Production Mode)"
    echo ""
    echo "⚠️  This requires sudo privileges"
    echo ""
    
    # This would create a proper systemd service
    echo "📋 Service creation not implemented in this demo"
    echo "💡 For production, consider using the VPS deployment:"
    echo "   python cli.py deploy --install-deps --create-service"
}

# Main execution
case "${1:-help}" in
    "screen")
        run_in_screen "${2:-full-backtest}"
        ;;
    "nohup")
        run_with_nohup "${2:-full-backtest}"
        ;;
    "service")
        create_service "${2:-full-backtest}"
        ;;
    "help"|"--help"|"-h"|"")
        show_usage
        ;;
    *)
        echo "❌ Unknown option: $1"
        show_usage
        exit 1
        ;;
esac 