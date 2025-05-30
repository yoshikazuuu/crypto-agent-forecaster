#!/bin/bash
# 📊 Background Validation Monitor
# 
# Monitor running validation tests and view results

echo "📊 Crypto Agent Forecaster - Background Monitor"
echo "==============================================="

# Function to show running processes
show_processes() {
    echo ""
    echo "🔄 Running Validation Processes:"
    echo "--------------------------------"
    
    # Check for Python validation processes
    local validation_processes=$(ps aux | grep "python cli.py" | grep -v grep)
    
    if [[ -n "$validation_processes" ]]; then
        echo "$validation_processes"
    else
        echo "❌ No validation processes currently running"
    fi
    
    echo ""
    
    # Check for screen sessions
    echo "🖥️  Active Screen Sessions:"
    echo "---------------------------"
    screen -ls 2>/dev/null || echo "❌ No screen sessions found"
    
    echo ""
}

# Function to show recent log files
show_logs() {
    echo "📝 Recent Log Files:"
    echo "-------------------"
    
    # Find validation log files
    local log_files=$(ls -t validation_*.log 2>/dev/null | head -5)
    
    if [[ -n "$log_files" ]]; then
        for log_file in $log_files; do
            local size=$(wc -l < "$log_file" 2>/dev/null || echo "0")
            local modified=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$log_file" 2>/dev/null || echo "Unknown")
            echo "  📄 $log_file ($size lines, modified: $modified)"
        done
        
        echo ""
        echo "💡 Commands:"
        local latest_log=$(ls -t validation_*.log 2>/dev/null | head -1)
        if [[ -n "$latest_log" ]]; then
            echo "   tail -f $latest_log    # Follow latest log"
        fi
        echo "   tail -f [log-file]     # Follow specific log"
    else
        echo "❌ No validation log files found"
    fi
    
    echo ""
}

# Function to show validation results
show_results() {
    echo "📊 Recent Validation Results:"
    echo "-----------------------------"
    
    # Find recent result files
    local result_files=$(ls -t validation_results/*.json 2>/dev/null | head -3)
    
    if [[ -n "$result_files" ]]; then
        for result_file in $result_files; do
            local basename=$(basename "$result_file")
            local modified=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$result_file" 2>/dev/null || echo "Unknown")
            
            # Try to extract metrics from the file
            local total_predictions=$(grep '"total_predictions"' "$result_file" 2>/dev/null | sed 's/.*: *\([0-9]*\).*/\1/' | head -1)
            local accuracy=$(grep '"accuracy_percentage"' "$result_file" 2>/dev/null | sed 's/.*: *\([0-9.]*\).*/\1/' | head -1)
            
            if [[ -n "$total_predictions" && -n "$accuracy" ]]; then
                echo "  📈 $basename"
                echo "     └─ Predictions: $total_predictions, Accuracy: ${accuracy}%, Modified: $modified"
            else
                echo "  📄 $basename (Modified: $modified)"
            fi
        done
        
        echo ""
        echo "💡 Commands:"
        echo "   python cli.py status   # Detailed status"
        echo "   python cli.py report   # Generate HTML report"
    else
        echo "❌ No validation results found"
    fi
    
    echo ""
}

# Function to show usage
show_usage() {
    echo ""
    echo "📋 Available Commands:"
    echo "---------------------"
    echo "  ./monitor.sh processes   # Show running processes"
    echo "  ./monitor.sh logs        # Show recent log files"
    echo "  ./monitor.sh results     # Show recent results"
    echo "  ./monitor.sh status      # Full status (default)"
    echo "  ./monitor.sh follow      # Follow latest log"
    echo "  ./monitor.sh kill-all    # Kill all validation processes"
    echo ""
}

# Function to follow latest log
follow_latest_log() {
    local latest_log=$(ls -t validation_*.log 2>/dev/null | head -1)
    
    if [[ -n "$latest_log" ]]; then
        echo "📝 Following latest log: $latest_log"
        echo "   (Press Ctrl+C to stop)"
        echo ""
        tail -f "$latest_log"
    else
        echo "❌ No log files found to follow"
        echo ""
        echo "💡 Start a background validation first:"
        echo "   ./run_background.sh nohup quick-backtest"
    fi
}

# Function to kill all validation processes
kill_all_processes() {
    echo "🛑 Killing all validation processes..."
    echo ""
    
    # Kill Python validation processes
    local pids=$(ps aux | grep "python cli.py" | grep -v grep | awk '{print $2}')
    
    if [[ -n "$pids" ]]; then
        for pid in $pids; do
            echo "  🔴 Killing process $pid"
            kill "$pid" 2>/dev/null || echo "    ⚠️  Failed to kill $pid"
        done
    else
        echo "❌ No validation processes found"
    fi
    
    # Kill screen sessions with crypto-validation
    local screen_sessions=$(screen -ls 2>/dev/null | grep crypto-validation | awk '{print $1}')
    
    if [[ -n "$screen_sessions" ]]; then
        for session in $screen_sessions; do
            echo "  🔴 Killing screen session $session"
            screen -X -S "$session" quit 2>/dev/null || echo "    ⚠️  Failed to kill $session"
        done
    fi
    
    echo ""
    echo "✅ Cleanup completed"
}

# Main execution
case "${1:-status}" in
    "processes"|"proc"|"ps")
        show_processes
        ;;
    "logs"|"log")
        show_logs
        ;;
    "results"|"result")
        show_results
        ;;
    "status"|"")
        show_processes
        show_logs
        show_results
        ;;
    "follow"|"tail")
        follow_latest_log
        ;;
    "kill-all"|"kill"|"stop")
        kill_all_processes
        ;;
    "help"|"--help"|"-h")
        show_usage
        ;;
    *)
        echo "❌ Unknown command: $1"
        show_usage
        exit 1
        ;;
esac 