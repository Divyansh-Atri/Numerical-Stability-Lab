#!/bin/bash

echo "=========================================="
echo "Numerical Stability Project Verification"
echo "=========================================="
echo ""

echo "Checking project structure..."
echo ""

# Check directories
for dir in notebooks utils tests plots; do
    if [ -d "$dir" ]; then
        echo "✓ $dir/ directory exists"
    else
        echo "✗ $dir/ directory missing"
    fi
done

echo ""
echo "Checking core files..."
echo ""

# Check files
for file in README.md requirements.txt PROJECT_SUMMARY.md QUICKSTART.md .gitignore; do
    if [ -f "$file" ]; then
        echo "✓ $file exists"
    else
        echo "✗ $file missing"
    fi
done

echo ""
echo "Checking notebooks..."
echo ""

# Check notebooks
for i in {01..09}; do
    nb_count=$(ls notebooks/${i}_*.ipynb 2>/dev/null | wc -l)
    if [ $nb_count -eq 1 ]; then
        nb_name=$(ls notebooks/${i}_*.ipynb)
        echo "✓ Notebook $i: $(basename $nb_name)"
    else
        echo "✗ Notebook $i missing"
    fi
done

echo ""
echo "Checking utility modules..."
echo ""

for module in __init__.py error_metrics.py floating_point_tools.py linear_algebra_utils.py; do
    if [ -f "utils/$module" ]; then
        echo "✓ utils/$module"
    else
        echo "✗ utils/$module missing"
    fi
done

echo ""
echo "Checking test notebook..."
echo ""

if [ -f "tests/sanity_checks.ipynb" ]; then
    echo "✓ tests/sanity_checks.ipynb"
else
    echo "✗ tests/sanity_checks.ipynb missing"
fi

echo ""
echo "=========================================="
echo "Project structure verification complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Launch Jupyter: jupyter notebook"
echo "3. Start with notebooks/01_floating_point_basics.ipynb"
echo ""
