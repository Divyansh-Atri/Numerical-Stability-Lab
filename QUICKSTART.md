# Quick Start Guide

## Installation

```bash
# Navigate to project directory
cd numerical-lab

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## Running the Notebooks

### Core Sequence (Required)

Execute these notebooks in order:

1. **01_floating_point_basics.ipynb** (5 min)
   - IEEE-754 representation
   - Machine epsilon
   - ULP analysis

2. **02_summation_and_cancellation.ipynb** (5 min)
   - Kahan summation
   - Catastrophic cancellation
   - Error growth experiments

3. **03_polynomial_evaluation.ipynb** (3 min)
   - Horner's method
   - Wilkinson polynomial

4. **04_linear_systems_stability.ipynb** (5 min)
   - Gaussian elimination with pivoting
   - Normal equations vs QR
   - Hilbert matrix

5. **05_backward_error_analysis.ipynb** (4 min)
   - Forward vs backward error
   - Backward stability
   - Conditioning vs stability

6. **06_summary_and_guidelines.ipynb** (3 min)
   - Key findings
   - Practical recommendations
   - Decision trees

**Total time: ~25 minutes**

### Advanced Topics (Optional)

7. **07_mixed_precision_experiments.ipynb** (3 min)
   - float32 vs float64
   - Iterative refinement

8. **08_floating_point_nonassociativity.ipynb** (2 min)
   - Non-associativity demonstrations
   - Reproducibility issues

9. **09_iterative_solvers_stability.ipynb** (3 min)
   - Jacobi and Conjugate Gradient
   - Convergence analysis

**Additional time: ~8 minutes**

### Validation

Run **tests/sanity_checks.ipynb** to verify all implementations work correctly.

## Tips

- **Execute cells sequentially**: Use Shift+Enter
- **Restart & Run All**: Kernel → Restart & Run All
- **Plots**: Automatically saved to `plots/` directory
- **Errors**: If you encounter import errors, ensure you're in the correct directory

## What to Expect

Each notebook will:
1. Explain the theory clearly
2. Implement algorithms from scratch
3. Run controlled experiments
4. Generate publication-quality plots
5. Interpret results

## Key Outputs

- **~20 plots** showing error growth, stability comparisons, etc.
- **Empirical verification** of theoretical error bounds
- **Practical guidelines** for algorithm selection

## Common Issues

### Import Errors
```python
# Make sure you're running from the notebooks/ directory
import sys
sys.path.append("..")  # This line should be in each notebook
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Plots Not Showing
```python
# Add this to notebook cells if needed
%matplotlib inline
```

## Next Steps

After running all notebooks:

1. Review the **README.md** for comprehensive documentation
2. Check **PROJECT_SUMMARY.md** for detailed project overview
3. Explore the **utils/** directory to see algorithm implementations
4. Modify experiments to test your own hypotheses

## Questions?

- Check the README.md for detailed explanations
- Review the code comments in utils/ modules
- Each notebook has markdown cells explaining the theory

Enjoy exploring numerical stability!
