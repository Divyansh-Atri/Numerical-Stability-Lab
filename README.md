# Numerical Stability of Floating-Point Algorithms: Theory and Experiments

A rigorous, notebook-based study of how floating-point arithmetic affects numerical algorithms, with emphasis on the distinction between **conditioning** and **stability**, and on how algorithmic choices amplify or control rounding errors.

## Project Structure

```
numerical-lab/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── notebooks/                         # Jupyter notebooks (execute in order)
│   ├── 01_floating_point_basics.ipynb
│   ├── 02_summation_and_cancellation.ipynb
│   ├── 03_polynomial_evaluation.ipynb
│   ├── 04_linear_systems_stability.ipynb
│   ├── 05_backward_error_analysis.ipynb
│   ├── 06_summary_and_guidelines.ipynb
│   ├── 07_mixed_precision_experiments.ipynb      # Advanced
│   ├── 08_floating_point_nonassociativity.ipynb  # Advanced
│   └── 09_iterative_solvers_stability.ipynb      # Advanced
├── utils/                             # Shared utility modules
│   ├── __init__.py
│   ├── error_metrics.py              # Error measurement functions
│   ├── floating_point_tools.py       # Summation, polynomial evaluation
│   └── linear_algebra_utils.py       # Linear solvers, factorizations
├── tests/                             # Sanity checks
│   └── sanity_checks.ipynb
└── plots/                             # Generated figures
```

## Notebook Execution Order

The notebooks are designed to be executed sequentially:

### Core Sequence

1. **01_floating_point_basics.ipynb**
   - IEEE-754 representation
   - Machine epsilon and unit roundoff
   - Rounding modes and error bounds
   - ULP (units in last place) analysis

2. **02_summation_and_cancellation.ipynb**
   - Naive vs Kahan summation
   - Pairwise summation
   - Catastrophic cancellation examples
   - Error growth analysis

3. **03_polynomial_evaluation.ipynb**
   - Naive power-based evaluation
   - Horner's method
   - Stability comparison
   - Wilkinson's polynomial example

4. **04_linear_systems_stability.ipynb**
   - Gaussian elimination with/without pivoting
   - Normal equations vs QR for least squares
   - Explicit inversion vs direct solve
   - Ill-conditioned systems (Hilbert matrix)

5. **05_backward_error_analysis.ipynb**
   - Forward vs backward error
   - Backward stability definition
   - Empirical backward error measurement
   - Conditioning vs stability distinction

6. **06_summary_and_guidelines.ipynb**
   - Key empirical findings
   - Practical guidelines for algorithm selection
   - Summary of stability principles

### Advanced Topics (Optional Difficulty Boost)

7. **07_mixed_precision_experiments.ipynb**
   - float32 vs float64 error accumulation
   - Mixed-precision iterative refinement
   - Precision-dependent error growth

8. **08_floating_point_nonassociativity.ipynb**
   - Associativity violation stress tests
   - Order-dependent summation errors
   - Reproducibility challenges

9. **09_iterative_solvers_stability.ipynb**
   - Jacobi and Gauss-Seidel convergence
   - Conjugate gradient stability
   - Preconditioning effects

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone or navigate to the project directory
cd numerical-lab

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Running the Notebooks

1. Start Jupyter: `jupyter notebook`
2. Navigate to the `notebooks/` directory
3. Open notebooks in order (01 through 06 for core, 07-09 for advanced)
4. Execute cells sequentially using **Shift+Enter**
5. Each notebook supports **"Restart & Run All"** for reproducibility

## Assumptions and Scope

### Floating-Point Arithmetic

- All experiments use **IEEE-754 double precision** (binary64) unless otherwise specified
- Machine epsilon: ε ≈ 2.22 × 10⁻¹⁶
- Assumes round-to-nearest-even rounding mode (default)
- No use of extended precision or symbolic computation in main experiments

### Implementation Constraints

- All core algorithms implemented **from scratch** in Python
- NumPy used only for basic array operations and reference solutions
- SciPy used for validation and comparison
- Deterministic experiments with fixed random seeds

### Limitations

- Focuses on dense linear algebra; sparse methods not covered in depth
- Does not address parallel or distributed numerical computation
- Hardware-specific optimizations (SIMD, FMA) not explicitly studied
- Assumes standard IEEE-754 compliance (may vary across platforms)

## Key Empirical Observations

### Summation

- **Naive summation**: Error grows as O(nε) for n summands
- **Kahan summation**: Reduces error to O(ε) + O(nε²), dramatic improvement
- **Pairwise summation**: Error grows as O(log n · ε), good balance of simplicity and accuracy

### Polynomial Evaluation

- **Naive method**: Unstable due to repeated power computation and cancellation
- **Horner's method**: Stable and efficient, minimizes rounding error
- Error difference can be orders of magnitude for high-degree polynomials

### Linear Systems

- **Gaussian elimination without pivoting**: Can fail catastrophically on well-conditioned systems
- **Partial pivoting**: Essential for stability, minimal performance cost
- **Normal equations**: Squares condition number, loses accuracy for κ(A) > 10⁸
- **QR factorization**: Preserves condition number, recommended for ill-conditioned least squares
- **Explicit inversion**: Always inferior to direct solve, both in cost and accuracy

### Conditioning vs Stability

- **Conditioning**: Property of the problem (data sensitivity)
- **Stability**: Property of the algorithm (error propagation)
- A stable algorithm applied to an ill-conditioned problem still produces large errors
- An unstable algorithm can fail even on well-conditioned problems

### Backward Error

- **Backward stable algorithms**: Produce exact solution to nearby problem
- Backward error ≈ O(ε) indicates backward stability
- Forward error ≤ (condition number) × (backward error)
- Backward stability is the gold standard for numerical algorithms

## Practical Implications

### Algorithm Selection Guidelines

1. **Summation**: Use Kahan or pairwise for critical applications
2. **Polynomial evaluation**: Always use Horner's method
3. **Linear systems**: Always use pivoting; prefer LU or QR over normal equations
4. **Least squares**: Use QR for ill-conditioned problems (κ > 10⁶)
5. **Never explicitly invert matrices** unless the inverse itself is needed

### Error Analysis Workflow

1. Identify the **condition number** of your problem
2. Choose a **backward stable** algorithm when possible
3. Expect forward error ≈ (condition number) × (machine epsilon)
4. If observed error exceeds this bound, suspect algorithmic instability

### When to Worry

- Condition numbers > 10¹⁰ (near machine precision limit)
- Subtracting nearly equal quantities (catastrophic cancellation)
- Forming A^T A explicitly (squares condition number)
- Algorithms without pivoting or equilibration

## Testing and Validation

Run the sanity check notebook to verify the implementation:

```bash
jupyter notebook tests/sanity_checks.ipynb
```

This notebook validates:
- Utility function correctness
- Algorithm implementations against NumPy/SciPy
- Numerical stability properties
- Reproducibility of experiments

## References and Further Reading

### Foundational Texts

- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.
- Trefethen, L. N., & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.
- Goldberg, D. (1991). "What every computer scientist should know about floating-point arithmetic." *ACM Computing Surveys*, 23(1), 5-48.

### IEEE-754 Standard

- IEEE Standard 754-2008 for Floating-Point Arithmetic

### Specific Algorithms

- Kahan, W. (1965). "Further remarks on reducing truncation errors." *Communications of the ACM*, 8(1), 40.
- Wilkinson, J. H. (1963). *Rounding Errors in Algebraic Processes*. Prentice-Hall.

## Author

**Divyansh Atri**

This project demonstrates deep understanding of numerical stability, floating-point arithmetic, and algorithmic robustness through rigorous theory and carefully designed experiments.

## License

This project is provided for educational and research purposes.
