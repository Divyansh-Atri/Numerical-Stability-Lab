# Plot Gallery

This directory contains all publication-quality plots generated from the numerical stability experiments.

## Generated Plots (17 total)

### Notebook 01: Floating-Point Basics
1. **01_rounding_error_distribution.png** (69.8 KB)
   - Distribution of rounding errors when converting float64 → float32 → float64
   - Shows histogram on linear and log scales

2. **01_ulp_spacing.png** (62.6 KB)
   - ULP (units in last place) spacing vs magnitude
   - Demonstrates non-uniform distribution of floating-point numbers

### Notebook 02: Summation and Cancellation
3. **02_summation_error_growth.png** (184.1 KB)
   - Error growth comparison: Naive vs Kahan vs Pairwise summation
   - Shows improvement factor over naive summation

4. **02_catastrophic_cancellation.png** (94.8 KB)
   - Demonstrates catastrophic cancellation in (1 - cos(x))/x²
   - Compares unstable and stable formulations

5. **02_quadratic_formula.png** (51.9 KB)
   - Stability of quadratic formula for different parameter values
   - Shows error growth for unstable vs stable variants

6. **02_log1p_expm1.png** (112.2 KB)
   - Precision loss in log(1+x) and exp(x)-1 for small x
   - Demonstrates need for specialized functions

### Notebook 03: Polynomial Evaluation
7. **03_wilkinson.png** (106.1 KB)
   - Wilkinson polynomial evaluation and error
   - Classic example of ill-conditioning

### Notebook 04: Linear Systems Stability
8. **04_hilbert_conditioning.png** (108.5 KB)
   - Hilbert matrix condition numbers vs size
   - Error growth with conditioning

9. **04_least_squares_comparison.png** (78.6 KB)
   - Normal equations vs QR factorization
   - Shows condition number squaring effect

### Notebook 05: Backward Error Analysis
10. **05_forward_vs_backward.png** (125.8 KB)
    - Forward vs backward error comparison
    - Verification of error bound: forward ≤ κ × backward

### Notebook 06: Summary and Guidelines
11. **06_summary_visualization.png** (199.6 KB)
    - Comprehensive 4-panel summary:
      - Error growth rates
      - Conditioning effects
      - Algorithm stability ratings
      - Problem conditioning regions

### Notebook 07: Mixed Precision (Advanced)
12. **07_mixed_precision_summation.png** (73.8 KB)
    - float32 vs float64 error accumulation
    - Shows ~10⁹ difference in precision

13. **07_iterative_refinement.png** (55.8 KB)
    - Convergence of mixed-precision iterative refinement
    - Demonstrates recovery of full precision

### Notebook 09: Iterative Solvers (Advanced)
14. **09_jacobi_convergence.png** (52.9 KB)
    - Jacobi iteration convergence on diagonally dominant matrix

15. **09_cg_convergence.png** (56.8 KB)
    - Conjugate Gradient convergence on SPD matrix

### Additional Comprehensive Plots
16. **algorithm_comparison.png** (73.3 KB)
    - Bar chart comparing accuracy, speed, and stability
    - Covers all major algorithms in the project

17. **complete_summary.png** (195.4 KB)
    - Final comprehensive summary with 5 panels:
      - Common error sources (pie chart)
      - Precision levels comparison
      - Condition number safety zones
      - Error growth rates
      - Recommended algorithms by task

## Total Size
All plots: ~1.8 MB

## Usage

These plots are referenced in the Jupyter notebooks and can be used for:
- Presentations
- Reports
- Documentation
- Educational materials

All plots are saved at 150 DPI for high quality while maintaining reasonable file sizes.

## Regenerating Plots

To regenerate all plots:
```bash
python3 generate_all_plots.py
```

This will overwrite existing plots with fresh versions based on the current code.
