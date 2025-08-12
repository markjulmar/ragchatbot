#!/bin/bash
# Code quality check script

set -e  # Exit on any error

echo "🔍 Running code quality checks..."
echo

# Black formatting check
echo "📝 Checking code formatting with Black..."
if uv run black backend/ main.py --check --diff; then
    echo "✅ All files are properly formatted"
else
    echo "❌ Some files need formatting. Run 'uv run black backend/ main.py' to fix."
    exit 1
fi
echo

# Import sorting check
echo "📦 Checking import sorting with isort..."
if uv run isort backend/ main.py --check-only --diff; then
    echo "✅ All imports are properly sorted"
else
    echo "❌ Some imports need sorting. Run 'uv run isort backend/ main.py' to fix."
    exit 1
fi
echo

# Linting with flake8
echo "🔍 Linting with flake8..."
if uv run flake8 backend/ main.py; then
    echo "✅ No linting issues found"
else
    echo "❌ Linting issues found. Please fix the issues above."
    exit 1
fi
echo

# Type checking with mypy
echo "🔬 Type checking with mypy..."
if uv run mypy backend/ main.py; then
    echo "✅ No type errors found"
else
    echo "❌ Type errors found. Please fix the issues above."
    exit 1
fi
echo

# Run tests
echo "🧪 Running tests..."
if uv run pytest backend/tests/ -v; then
    echo "✅ All tests passed"
else
    echo "❌ Some tests failed. Please fix the failing tests."
    exit 1
fi
echo

echo "🎉 All quality checks passed!"