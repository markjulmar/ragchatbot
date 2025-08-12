#!/bin/bash
# Linting script

echo "🔍 Running linters..."
echo

# Linting with flake8
echo "🔍 Linting with flake8..."
uv run flake8 backend/ main.py
echo

# Type checking with mypy
echo "🔬 Type checking with mypy..."
uv run mypy backend/ main.py
echo

echo "✅ Linting complete!"