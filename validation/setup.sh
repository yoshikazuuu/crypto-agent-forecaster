#!/bin/bash
# Setup script for Crypto Agent Forecaster Validation Framework

set -e

echo "🔮 Setting up Crypto Agent Forecaster Validation Framework"
echo "=" * 60

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    
    # Detect OS and install uv accordingly
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        # Linux or macOS
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        # Windows (Git Bash/MSYS2)
        echo "Please install uv manually on Windows:"
        echo "powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""
        exit 1
    else
        echo "Unsupported OS. Please install uv manually:"
        echo "https://github.com/astral-sh/uv#installation"
        exit 1
    fi
    
    echo "✅ uv installed successfully"
else
    echo "✅ uv is already installed ($(uv --version))"
fi

# Install the validation framework
echo "📦 Installing validation framework dependencies..."
uv pip install -e .

echo "✅ Installation complete!"
echo ""
echo "🚀 Quick start:"
echo "  uv run python cli.py quick-test"
echo ""
echo "📚 More commands:"
echo "  uv run python cli.py --help"
echo "  uv run python cli.py live --duration 24 --coins bitcoin ethereum"
echo "  uv run python cli.py backtest --days 30"
echo "  uv run python cli.py report"
echo ""
echo "🖥️ For VPS deployment:"
echo "  sudo uv run python cli.py deploy --install-deps --create-service" 
 