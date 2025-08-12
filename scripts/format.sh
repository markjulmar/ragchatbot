#!/bin/bash
# Code formatting script

echo "🎨 Formatting code..."
echo

# Format with Black
echo "📝 Formatting code with Black..."
uv run black backend/ main.py
echo

# Sort imports with isort
echo "📦 Sorting imports with isort..."
uv run isort backend/ main.py
echo

echo "✨ Code formatting complete!"