#!/bin/bash
# Testing script

echo "🧪 Running tests..."
echo

# Run pytest with verbose output and coverage
uv run pytest backend/tests/ -v --tb=short

echo "✅ Testing complete!"