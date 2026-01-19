#!/bin/bash
# Setup script for Chess Tutor

set -e

echo "=================================================="
echo "  Chess Tutor - Setup Script"
echo "=================================================="
echo

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "✅ Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✅ Dependencies installed"

# Check for Stockfish
echo
echo "Checking for Stockfish..."
if command -v stockfish &> /dev/null; then
    STOCKFISH_PATH=$(which stockfish)
    echo "✅ Stockfish found: $STOCKFISH_PATH"
else
    echo "⚠️  Stockfish not found in PATH"
    echo
    echo "Please install Stockfish:"
    echo "  - Linux (Debian/Ubuntu): sudo apt install stockfish"
    echo "  - Linux (Fedora): sudo dnf install stockfish"
    echo "  - macOS: brew install stockfish"
    echo "  - Windows: Download from https://stockfishchess.org/download/"
    echo
    echo "Or set STOCKFISH_PATH in your .env file"
fi

# Setup .env file
echo
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created"
    echo
    echo "⚠️  IMPORTANT: Edit .env to configure:"
    echo "    OPENAI_API_KEY=your-key-here"
    echo
    echo "    Or for vLLora gateway:"
    echo "    USE_LOCAL_GATEWAY=true"
    echo "    LLM_BASE_URL=http://localhost:9090/v1"
else
    echo "✅ .env file already exists"
fi

# Summary
echo
echo "=================================================="
echo "  Setup Complete!"
echo "=================================================="
echo
echo "To run the chess tutor:"
echo
echo "  1. Configure .env (OpenAI API key or vLLora gateway)"
echo "  2. Activate the virtual environment:"
echo "       source venv/bin/activate"
echo "  3. Run the demo:"
echo "       python examples/demo.py"
echo
echo "For vLLora tracing, set USE_LOCAL_GATEWAY=true in .env"
echo
