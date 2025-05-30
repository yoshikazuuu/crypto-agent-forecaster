#!/bin/bash
# Setup script for Crypto Agent Forecaster Validation Framework

echo "🔮 Setting up Crypto Agent Forecaster Validation Framework"
echo "=========================================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Make sure you're in the validation/ directory."
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    
    # Detect OS and install uv
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        # Windows
        echo "Please install uv manually from: https://github.com/astral-sh/uv"
        echo "Or use: pip install uv"
        exit 1
    else
        echo "❌ Unsupported OS. Please install uv manually: https://github.com/astral-sh/uv"
        exit 1
    fi
    
    # Source the shell to get uv in PATH
    source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "✅ uv is already installed ($(uv --version))"
fi

# Install the validation framework
echo "📦 Installing validation framework dependencies..."

# Try installing in editable mode
if uv pip install -e .; then
    echo "✅ Validation framework installed successfully!"
else
    echo "⚠️ Editable install failed, trying regular install..."
    if uv pip install .; then
        echo "✅ Validation framework installed successfully!"
    else
        echo "❌ Installation failed. Trying alternative approach..."
        
        # Fallback: install dependencies directly
        echo "📦 Installing dependencies directly..."
        uv pip install pandas numpy matplotlib seaborn scipy plotly psutil schedule typer rich python-dateutil requests crewai google-generativeai python-dotenv ta Pillow mplfinance
        
        if [ $? -eq 0 ]; then
            echo "✅ Dependencies installed successfully!"
            echo "⚠️ Note: You'll need to run commands from this directory"
        else
            echo "❌ Failed to install dependencies"
            exit 1
        fi
    fi
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "🚀 Quick start commands:"
echo "• Test the system:       uv run python cli.py quick-test"
echo "• 6h test (all coins):   uv run python cli.py full-test"
echo "• Instant test (all):    uv run python cli.py full-backtest --days 30"
echo "• Live validation:       uv run python cli.py live --duration 6 --coins bitcoin"
echo "• Backtesting:           uv run python cli.py backtest --days 30"
echo "• Generate reports:      uv run python cli.py report"
echo "• Show help:             uv run python cli.py --help"
echo ""
echo "📖 For full documentation, see README.md"

echo ""
echo "🖥️ For VPS deployment:"
echo "  sudo uv run python cli.py deploy --install-deps --create-service" 
 