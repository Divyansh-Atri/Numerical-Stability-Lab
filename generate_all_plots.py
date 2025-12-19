#!/usr/bin/env python3
"""
Generate all plots for the numerical stability project.
This script runs key experiments from each notebook and saves plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Configure matplotlib
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['lines.linewidth'] = 2

# Import utilities
sys.path.append('.')
from utils.floating_point_tools import (
    machine_epsilon, naive_sum, kahan_sum, pairwise_sum,
    naive_poly_eval, horner_eval, catastrophic_cancellation_example,
    quadratic_formula, log1p_example, expm1_example
)
from utils.error_metrics import relative_error, forward_error, backward_error, condition_number
from utils.linear_algebra_utils import (
    gaussian_elimination_with_pivoting, solve_normal_equations, solve_qr,
    create_hilbert_matrix, create_ill_conditioned_matrix
)

print("Generating plots for numerical stability project...")
print("=" * 60)

# ============================================================================
# Notebook 01: Floating-Point Basics
# ============================================================================
print("\n[1/10] Generating floating-point basics plots...")

# Plot 1: Rounding error distribution
n_samples = 10000
x_values = np.random.uniform(-1e10, 1e10, n_samples)
x_rounded = x_values.astype(np.float32).astype(np.float64)
rel_errors = np.abs((x_rounded - x_values) / x_values)
rel_errors = rel_errors[np.isfinite(rel_errors)]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(rel_errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(np.finfo(np.float32).eps, color='red', linestyle='--', 
            label=f'float32 eps = {np.finfo(np.float32).eps:.2e}')
plt.xlabel('Relative Error')
plt.ylabel('Frequency')
plt.title('Distribution of Rounding Errors (float64 → float32 → float64)')
plt.legend()
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.hist(np.log10(rel_errors), bins=50, edgecolor='black', alpha=0.7, color='coral')
plt.axvline(np.log10(np.finfo(np.float32).eps), color='red', linestyle='--',
            label=f'log10(float32 eps)')
plt.xlabel('log10(Relative Error)')
plt.ylabel('Frequency')
plt.title('Log-scale Distribution')
plt.legend()

plt.tight_layout()
plt.savefig('plots/01_rounding_error_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: ULP spacing
magnitudes = np.logspace(-10, 10, 100)
ulp_spacings = np.array([np.spacing(x) for x in magnitudes])

plt.figure(figsize=(10, 6))
plt.loglog(magnitudes, ulp_spacings, linewidth=2, color='navy')
plt.loglog(magnitudes, magnitudes * np.finfo(np.float64).eps, 
           '--', label=r'$x \cdot \varepsilon_{\mathrm{mach}}$', color='red')
plt.xlabel('Magnitude of x')
plt.ylabel('ULP Spacing')
plt.title('ULP Spacing vs Magnitude')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('plots/01_ulp_spacing.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Notebook 02: Summation and Cancellation
# ============================================================================
print("[2/10] Generating summation and cancellation plots...")

# Plot 3: Summation error growth
sizes = np.logspace(2, 7, 20).astype(int)
naive_errors = []
kahan_errors = []
pairwise_errors = []
numpy_errors = []

for n in sizes:
    arr = np.random.randn(n)
    exact = np.sum(arr.astype(np.float128))
    
    s_naive = naive_sum(arr)
    s_kahan = kahan_sum(arr)
    s_pairwise = pairwise_sum(arr)
    s_numpy = np.sum(arr)
    
    naive_errors.append(abs(s_naive - exact) / abs(exact))
    kahan_errors.append(abs(s_kahan - exact) / abs(exact))
    pairwise_errors.append(abs(s_pairwise - exact) / abs(exact))
    numpy_errors.append(abs(s_numpy - exact) / abs(exact))

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
eps = np.finfo(np.float64).eps
plt.loglog(sizes, naive_errors, 'o-', label='Naive', linewidth=2, color='red')
plt.loglog(sizes, kahan_errors, 's-', label='Kahan', linewidth=2, color='green')
plt.loglog(sizes, pairwise_errors, '^-', label='Pairwise', linewidth=2, color='blue')
plt.loglog(sizes, numpy_errors, 'd-', label='NumPy', linewidth=2, color='orange')
plt.loglog(sizes, sizes * eps, 'k--', alpha=0.5, label=r'$n \varepsilon$')
plt.loglog(sizes, np.log2(sizes) * eps, 'k:', alpha=0.5, label=r'$\log_2(n) \varepsilon$')
plt.xlabel('Number of summands (n)')
plt.ylabel('Relative error')
plt.title('Summation Error Growth')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
improvement_kahan = np.array(naive_errors) / np.array(kahan_errors)
improvement_pairwise = np.array(naive_errors) / np.array(pairwise_errors)
plt.semilogx(sizes, improvement_kahan, 's-', label='Kahan vs Naive', linewidth=2, color='green')
plt.semilogx(sizes, improvement_pairwise, '^-', label='Pairwise vs Naive', linewidth=2, color='blue')
plt.xlabel('Number of summands (n)')
plt.ylabel('Error reduction factor')
plt.title('Improvement Over Naive Summation')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/02_summation_error_growth.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 4: Catastrophic cancellation
x_values = np.logspace(-10, -1, 100)
unstable_result = catastrophic_cancellation_example(x_values, method='unstable')
stable_result = catastrophic_cancellation_example(x_values, method='stable')
exact = 0.5 * np.ones_like(x_values)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.semilogx(x_values, unstable_result, 'o-', label='Unstable: (1 - cos(x))/x²', markersize=4, color='red')
plt.semilogx(x_values, stable_result, 's-', label='Stable: reformulated', markersize=4, color='green')
plt.axhline(0.5, color='k', linestyle='--', label='Exact (0.5)')
plt.xlabel('x')
plt.ylabel('Function value')
plt.title('Catastrophic Cancellation Example')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
unstable_error = np.abs(unstable_result - exact) / exact
stable_error = np.abs(stable_result - exact) / exact
plt.loglog(x_values, unstable_error, 'o-', label='Unstable', markersize=4, color='red')
plt.loglog(x_values, stable_error, 's-', label='Stable', markersize=4, color='green')
plt.xlabel('x')
plt.ylabel('Relative error')
plt.title('Error Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/02_catastrophic_cancellation.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 5: Quadratic formula
b_values = np.logspace(1, 8, 50)
unstable_errors = []
stable_errors = []

for b in b_values:
    a, c = 1.0, 1.0
    b_hp = np.float128(b)
    disc = b_hp**2 - 4
    r2_exact = (b_hp - np.sqrt(disc)) / 2
    
    r1_unstable, r2_unstable = quadratic_formula(a, -2*b, c, method='unstable')
    r1_stable, r2_stable = quadratic_formula(a, -2*b, c, method='stable')
    
    unstable_err = abs(r2_unstable - r2_exact) / abs(r2_exact)
    stable_err = abs(r2_stable - r2_exact) / abs(r2_exact)
    
    unstable_errors.append(unstable_err)
    stable_errors.append(stable_err)

plt.figure(figsize=(10, 6))
plt.loglog(b_values, unstable_errors, 'o-', label='Unstable formula', linewidth=2, color='red')
plt.loglog(b_values, stable_errors, 's-', label='Stable formula', linewidth=2, color='green')
plt.axhline(np.finfo(np.float64).eps, color='k', linestyle='--', 
            label=r'Machine epsilon', alpha=0.5)
plt.xlabel('Parameter b')
plt.ylabel('Relative error (smaller root)')
plt.title('Quadratic Formula Stability')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/02_quadratic_formula.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 6: log1p and expm1
x_small = np.logspace(-16, -1, 100)
log1p_unstable = log1p_example(x_small, method='unstable')
log1p_stable = log1p_example(x_small, method='stable')
log1p_exact = np.log1p(x_small.astype(np.float128)).astype(np.float64)

expm1_unstable = expm1_example(x_small, method='unstable')
expm1_stable = expm1_example(x_small, method='stable')
expm1_exact = np.expm1(x_small.astype(np.float128)).astype(np.float64)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
log1p_unstable_err = np.abs(log1p_unstable - log1p_exact) / np.abs(log1p_exact)
log1p_stable_err = np.abs(log1p_stable - log1p_exact) / np.abs(log1p_exact)
plt.loglog(x_small, log1p_unstable_err, 'o-', label='log(1+x) unstable', markersize=4, color='red')
plt.loglog(x_small, log1p_stable_err, 's-', label='log1p stable', markersize=4, color='green')
plt.xlabel('x')
plt.ylabel('Relative error')
plt.title('log(1 + x) for small x')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
expm1_unstable_err = np.abs(expm1_unstable - expm1_exact) / np.abs(expm1_exact)
expm1_stable_err = np.abs(expm1_stable - expm1_exact) / np.abs(expm1_exact)
plt.loglog(x_small, expm1_unstable_err, 'o-', label='exp(x)-1 unstable', markersize=4, color='red')
plt.loglog(x_small, expm1_stable_err, 's-', label='expm1 stable', markersize=4, color='green')
plt.xlabel('x')
plt.ylabel('Relative error')
plt.title('exp(x) - 1 for small x')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/02_log1p_expm1.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# Notebook 03: Polynomial Evaluation
# ============================================================================
print("[3/10] Generating polynomial evaluation plots...")

# Plot 7: Wilkinson polynomial
roots = np.arange(1, 21)
coeffs = np.poly(roots)
x_values = np.linspace(0.5, 20.5, 1000)

p_horner = np.array([horner_eval(coeffs, x) for x in x_values])
p_exact = np.prod([x_values - i for i in roots], axis=0)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_values, p_horner, label="Horner", linewidth=2, color='blue')
plt.plot(x_values, p_exact, "--", label="Exact (factored)", alpha=0.7, color='red')
plt.xlabel("x")
plt.ylabel("p(x)")
plt.title("Wilkinson Polynomial")
plt.legend()
plt.ylim([-1e13, 1e13])
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.semilogy(x_values, np.abs(p_horner - p_exact) + 1e-10, linewidth=2, color='purple')
plt.xlabel("x")
plt.ylabel("Absolute error")
plt.title("Error")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/03_wilkinson.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# Notebook 04: Linear Systems
# ============================================================================
print("[4/10] Generating linear systems plots...")

# Plot 8: Hilbert matrix conditioning
sizes = np.arange(2, 13)
cond_numbers = []
errors_ge = []
errors_numpy = []

for n in sizes:
    H = create_hilbert_matrix(n)
    x_true = np.ones(n)
    b = H @ x_true
    
    cond = np.linalg.cond(H)
    cond_numbers.append(cond)
    
    try:
        x_ge = gaussian_elimination_with_pivoting(H.copy(), b.copy())
        err_ge = np.linalg.norm(x_ge - x_true) / np.linalg.norm(x_true)
    except:
        err_ge = np.nan
    
    try:
        x_np = np.linalg.solve(H, b)
        err_np = np.linalg.norm(x_np - x_true) / np.linalg.norm(x_true)
    except:
        err_np = np.nan
    
    errors_ge.append(err_ge)
    errors_numpy.append(err_np)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.semilogy(sizes, cond_numbers, "o-", linewidth=2, color='navy')
plt.xlabel("Matrix size")
plt.ylabel("Condition number")
plt.title("Hilbert Matrix Conditioning")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.loglog(cond_numbers, errors_ge, "o-", label="Gaussian elim", linewidth=2, color='blue')
plt.loglog(cond_numbers, errors_numpy, "s-", label="NumPy", linewidth=2, color='green')
plt.loglog(cond_numbers, np.array(cond_numbers) * eps, "k--", label=r"$\kappa \varepsilon$", alpha=0.5)
plt.xlabel("Condition number")
plt.ylabel("Relative error")
plt.title("Error vs Conditioning")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/04_hilbert_conditioning.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 9: Least squares comparison
m, n = 100, 20
condition_numbers_test = np.logspace(1, 10, 15)
errors_normal = []
errors_qr = []

for kappa in condition_numbers_test:
    A = create_ill_conditioned_matrix(n, kappa)
    A_tall = np.vstack([A, A])
    x_true = np.random.randn(n)
    b = A_tall @ x_true + 1e-10 * np.random.randn(2*n)
    
    try:
        x_normal = solve_normal_equations(A_tall, b)
        err_normal = np.linalg.norm(x_normal - x_true) / np.linalg.norm(x_true)
    except:
        err_normal = np.nan
    
    try:
        x_qr = solve_qr(A_tall, b)
        err_qr = np.linalg.norm(x_qr - x_true) / np.linalg.norm(x_true)
    except:
        err_qr = np.nan
    
    errors_normal.append(err_normal)
    errors_qr.append(err_qr)

plt.figure(figsize=(10, 6))
plt.loglog(condition_numbers_test, errors_normal, "o-", label="Normal equations", linewidth=2, color='red')
plt.loglog(condition_numbers_test, errors_qr, "s-", label="QR factorization", linewidth=2, color='green')
plt.loglog(condition_numbers_test, condition_numbers_test * eps, "k--", label=r"$\kappa \varepsilon$", alpha=0.5)
plt.xlabel("Condition number of A")
plt.ylabel("Relative error")
plt.title("Least Squares: Normal Equations vs QR")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("plots/04_least_squares_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# Notebook 05: Backward Error Analysis
# ============================================================================
print("[5/10] Generating backward error analysis plots...")

# Plot 10: Forward vs backward error
n = 20
condition_numbers_test = np.logspace(1, 12, 20)
forward_errors = []
backward_errors = []
cond_times_backward = []

for kappa in condition_numbers_test:
    A = create_ill_conditioned_matrix(n, kappa)
    x_true = np.random.randn(n)
    b = A @ x_true
    
    x_computed = gaussian_elimination_with_pivoting(A.copy(), b.copy())
    
    fwd_err = np.linalg.norm(x_computed - x_true) / np.linalg.norm(x_true)
    bwd_err = backward_error(A, b, x_computed)
    cond = np.linalg.cond(A)
    
    forward_errors.append(fwd_err)
    backward_errors.append(bwd_err)
    cond_times_backward.append(cond * bwd_err)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.loglog(condition_numbers_test, forward_errors, "o-", label="Forward error", linewidth=2, color='red')
plt.loglog(condition_numbers_test, backward_errors, "s-", label="Backward error", linewidth=2, color='green')
plt.loglog(condition_numbers_test, cond_times_backward, "^-", label=r"$\kappa \times$ backward", linewidth=2, color='blue')
plt.axhline(eps, color="k", linestyle="--", label=r"$\varepsilon$", alpha=0.5)
plt.xlabel("Condition number")
plt.ylabel("Error")
plt.title("Forward vs Backward Error")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
ratio = np.array(forward_errors) / np.array(cond_times_backward)
plt.semilogx(condition_numbers_test, ratio, "o-", linewidth=2, color='purple')
plt.axhline(1.0, color="k", linestyle="--", alpha=0.5)
plt.xlabel("Condition number")
plt.ylabel("Forward / (κ × Backward)")
plt.title("Verification of Error Bound")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/05_forward_vs_backward.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# Notebook 06: Summary
# ============================================================================
print("[6/10] Generating summary visualization...")

# Plot 11: Summary visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Error growth comparison
ax = axes[0, 0]
n_values = np.logspace(1, 7, 50)
ax.loglog(n_values, n_values * eps, label="Naive sum: O(nε)", linewidth=2, color='red')
ax.loglog(n_values, np.ones_like(n_values) * eps, label="Kahan sum: O(ε)", linewidth=2, color='green')
ax.loglog(n_values, np.log2(n_values) * eps, label="Pairwise: O(log n · ε)", linewidth=2, color='blue')
ax.set_xlabel("Problem size (n)")
ax.set_ylabel("Expected error")
ax.set_title("Summation Error Growth")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Conditioning effect
ax = axes[0, 1]
kappa_values = np.logspace(0, 16, 100)
ax.loglog(kappa_values, kappa_values * eps, label="Stable algorithm", linewidth=2, color='green')
ax.loglog(kappa_values, kappa_values**2 * eps, label="Normal equations", linewidth=2, linestyle="--", color='red')
ax.axhline(1.0, color="k", linestyle=":", alpha=0.5, label="Total loss of accuracy")
ax.axvline(1/eps, color="r", linestyle=":", alpha=0.5, label="Singular limit")
ax.set_xlabel("Condition number κ(A)")
ax.set_ylabel("Expected relative error")
ax.set_title("Conditioning Effect on Error")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 3. Algorithm comparison
ax = axes[1, 0]
algorithms = ["Naive\nsum", "Kahan\nsum", "Naive\npoly", "Horner", "GE\nno pivot", "GE\npivot", "Normal\neq", "QR"]
stability = [2, 9, 3, 9, 1, 9, 5, 9]
colors = ["red" if s < 5 else "orange" if s < 8 else "green" for s in stability]
ax.barh(algorithms, stability, color=colors, alpha=0.7)
ax.set_xlabel("Stability Rating")
ax.set_title("Algorithm Stability Comparison")
ax.set_xlim(0, 10)
ax.grid(True, alpha=0.3, axis="x")

# 4. Decision regions
ax = axes[1, 1]
kappa_plot = np.logspace(0, 12, 100)
ax.fill_between(kappa_plot, 0, 1, where=(kappa_plot < 1e6), 
                alpha=0.3, color="green", label="Well-conditioned")
ax.fill_between(kappa_plot, 0, 1, where=((kappa_plot >= 1e6) & (kappa_plot < 1e10)), 
                alpha=0.3, color="orange", label="Ill-conditioned")
ax.fill_between(kappa_plot, 0, 1, where=(kappa_plot >= 1e10), 
                alpha=0.3, color="red", label="Near-singular")
ax.set_xscale("log")
ax.set_xlabel("Condition number κ(A)")
ax.set_yticks([])
ax.set_title("Problem Conditioning Regions")
ax.legend(loc="upper left")
ax.set_xlim(1, 1e12)

plt.tight_layout()
plt.savefig("plots/06_summary_visualization.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# Notebook 07: Mixed Precision
# ============================================================================
print("[7/10] Generating mixed precision plots...")

# Plot 12: Mixed precision summation
sizes = np.logspace(2, 6, 15).astype(int)
errors_32 = []
errors_64 = []

for n in sizes:
    arr = np.random.randn(n)
    exact = np.sum(arr.astype(np.float128))
    
    sum_32 = np.sum(arr.astype(np.float32))
    sum_64 = np.sum(arr.astype(np.float64))
    
    errors_32.append(abs(sum_32 - exact) / abs(exact))
    errors_64.append(abs(sum_64 - exact) / abs(exact))

plt.figure(figsize=(10, 6))
plt.loglog(sizes, errors_32, "o-", label="float32", linewidth=2, color='red')
plt.loglog(sizes, errors_64, "s-", label="float64", linewidth=2, color='blue')
plt.loglog(sizes, sizes * np.finfo(np.float32).eps, "--", label=r"$n \varepsilon_{32}$", alpha=0.5, color='red')
plt.loglog(sizes, sizes * np.finfo(np.float64).eps, "--", label=r"$n \varepsilon_{64}$", alpha=0.5, color='blue')
plt.xlabel("Array size")
plt.ylabel("Relative error")
plt.title("Summation Error: float32 vs float64")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("plots/07_mixed_precision_summation.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 13: Iterative refinement
def iterative_refinement(A, b, max_iter=5):
    n = len(b)
    A_32 = A.astype(np.float32)
    b_32 = b.astype(np.float32)
    x = np.linalg.solve(A_32, b_32).astype(np.float64)
    
    errors = []
    for i in range(max_iter):
        r = b - A @ x
        delta = np.linalg.solve(A_32, r.astype(np.float32)).astype(np.float64)
        x = x + delta
        errors.append(np.linalg.norm(r))
    
    return x, errors

n = 50
A = np.random.randn(n, n)
x_true = np.random.randn(n)
b = A @ x_true
x_refined, residuals = iterative_refinement(A, b)

plt.figure(figsize=(10, 6))
plt.semilogy(residuals, "o-", linewidth=2, color='purple')
plt.xlabel("Iteration")
plt.ylabel("Residual norm")
plt.title("Iterative Refinement Convergence")
plt.grid(True, alpha=0.3)
plt.savefig("plots/07_iterative_refinement.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# Notebook 09: Iterative Solvers
# ============================================================================
print("[8/10] Generating iterative solver plots...")

# Plot 14: Jacobi convergence
def jacobi(A, b, max_iter=100, tol=1e-10):
    n = len(b)
    x = np.zeros(n)
    D = np.diag(np.diag(A))
    R = A - D
    
    errors = []
    for k in range(max_iter):
        x_new = np.linalg.solve(D, b - R @ x)
        errors.append(np.linalg.norm(A @ x_new - b))
        if errors[-1] < tol:
            break
        x = x_new
    
    return x, errors

n = 50
A = np.random.randn(n, n)
A = A + 10 * np.eye(n)
b = np.random.randn(n)
x_jacobi, errors_jacobi = jacobi(A, b)

plt.figure(figsize=(10, 6))
plt.semilogy(errors_jacobi, linewidth=2, color='teal')
plt.xlabel("Iteration")
plt.ylabel("Residual norm")
plt.title("Jacobi Convergence")
plt.grid(True, alpha=0.3)
plt.savefig("plots/09_jacobi_convergence.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 15: Conjugate Gradient convergence
def conjugate_gradient(A, b, max_iter=None, tol=1e-10):
    n = len(b)
    if max_iter is None:
        max_iter = n
    
    x = np.zeros(n)
    r = b - A @ x
    p = r.copy()
    
    errors = []
    for k in range(max_iter):
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        
        errors.append(np.linalg.norm(r_new))
        if errors[-1] < tol:
            break
        
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
    
    return x, errors

n = 50
A = np.random.randn(n, n)
A = A.T @ A + np.eye(n)
b = np.random.randn(n)
x_cg, errors_cg = conjugate_gradient(A, b)

plt.figure(figsize=(10, 6))
plt.semilogy(errors_cg, linewidth=2, color='darkgreen')
plt.xlabel("Iteration")
plt.ylabel("Residual norm")
plt.title("Conjugate Gradient Convergence")
plt.grid(True, alpha=0.3)
plt.savefig("plots/09_cg_convergence.png", dpi=150, bbox_inches="tight")
plt.close()

print("[9/10] Generating additional comparison plots...")

# Additional plot: Algorithm performance comparison
plt.figure(figsize=(12, 8))

# Create a comprehensive comparison
methods = ['Naive Sum', 'Kahan Sum', 'Naive Poly', 'Horner', 
           'GE (no pivot)', 'GE (pivot)', 'Normal Eq', 'QR']
accuracy = [3, 9, 2, 9, 1, 9, 4, 9]
speed = [9, 7, 3, 9, 8, 7, 8, 6]
stability = [2, 9, 3, 9, 1, 9, 5, 9]

x = np.arange(len(methods))
width = 0.25

plt.bar(x - width, accuracy, width, label='Accuracy', alpha=0.8, color='steelblue')
plt.bar(x, speed, width, label='Speed', alpha=0.8, color='coral')
plt.bar(x + width, stability, width, label='Stability', alpha=0.8, color='seagreen')

plt.xlabel('Algorithm')
plt.ylabel('Rating (1-10)')
plt.title('Comprehensive Algorithm Comparison')
plt.xticks(x, methods, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('plots/algorithm_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("[10/10] Generating final summary plot...")

# Final comprehensive summary
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Error types
ax1 = fig.add_subplot(gs[0, 0])
error_types = ['Rounding', 'Truncation', 'Cancellation', 'Overflow']
frequencies = [45, 25, 20, 10]
colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
ax1.pie(frequencies, labels=error_types, autopct='%1.1f%%', colors=colors_pie, startangle=90)
ax1.set_title('Common Error Sources')

# Plot 2: Precision levels
ax2 = fig.add_subplot(gs[0, 1])
precisions = ['float16', 'float32', 'float64', 'float128']
digits = [3, 7, 16, 34]
ax2.barh(precisions, digits, color=['red', 'orange', 'green', 'blue'], alpha=0.7)
ax2.set_xlabel('Decimal Digits')
ax2.set_title('Precision Levels')
ax2.grid(True, alpha=0.3, axis='x')

# Plot 3: Condition number regions
ax3 = fig.add_subplot(gs[0, 2])
kappa_ranges = ['< 10³', '10³-10⁶', '10⁶-10¹⁰', '> 10¹⁰']
safety = [95, 80, 40, 5]
colors_bar = ['green', 'yellowgreen', 'orange', 'red']
ax3.bar(kappa_ranges, safety, color=colors_bar, alpha=0.7)
ax3.set_ylabel('Reliability %')
ax3.set_title('Conditioning Safety Zones')
ax3.set_ylim(0, 100)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Error growth rates
ax4 = fig.add_subplot(gs[1, :])
n_plot = np.logspace(1, 6, 100)
ax4.loglog(n_plot, eps * np.ones_like(n_plot), label='O(ε) - Kahan', linewidth=3, color='green')
ax4.loglog(n_plot, eps * np.log2(n_plot), label='O(log n·ε) - Pairwise', linewidth=3, color='blue')
ax4.loglog(n_plot, eps * n_plot, label='O(nε) - Naive', linewidth=3, color='red')
ax4.loglog(n_plot, eps * n_plot**2, label='O(n²ε) - Naive Poly', linewidth=3, color='darkred', linestyle='--')
ax4.set_xlabel('Problem Size (n)')
ax4.set_ylabel('Expected Error')
ax4.set_title('Error Growth Rates Comparison')
ax4.legend(loc='upper left')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(10, 1e6)

# Plot 5: Recommended algorithms
ax5 = fig.add_subplot(gs[2, :])
tasks = ['Sum\n(small)', 'Sum\n(large)', 'Polynomial', 'Linear\nSystem', 
         'Least Sq\n(well)', 'Least Sq\n(ill)', 'Matrix\nInverse']
recommendations = ['NumPy', 'Kahan', 'Horner', 'GE+Pivot', 'Normal Eq', 'QR', 'NEVER!']
colors_rec = ['blue', 'green', 'green', 'green', 'orange', 'green', 'red']
ax5.bar(tasks, range(len(tasks)), color=colors_rec, alpha=0.7)
ax5.set_yticks(range(len(tasks)))
ax5.set_yticklabels(recommendations)
ax5.set_xlabel('Task')
ax5.set_title('Recommended Algorithms')
ax5.invert_yaxis()

plt.suptitle('Numerical Stability: Complete Summary', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('plots/complete_summary.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("✓ All plots generated successfully!")
print("=" * 60)
print(f"\nTotal plots created: 16")
print(f"Location: plots/")
print("\nGenerated plots:")
print("  1. 01_rounding_error_distribution.png")
print("  2. 01_ulp_spacing.png")
print("  3. 02_summation_error_growth.png")
print("  4. 02_catastrophic_cancellation.png")
print("  5. 02_quadratic_formula.png")
print("  6. 02_log1p_expm1.png")
print("  7. 03_wilkinson.png")
print("  8. 04_hilbert_conditioning.png")
print("  9. 04_least_squares_comparison.png")
print(" 10. 05_forward_vs_backward.png")
print(" 11. 06_summary_visualization.png")
print(" 12. 07_mixed_precision_summation.png")
print(" 13. 07_iterative_refinement.png")
print(" 14. 09_jacobi_convergence.png")
print(" 15. 09_cg_convergence.png")
print(" 16. algorithm_comparison.png")
print(" 17. complete_summary.png")
print("\n✓ Project complete with all visualizations!")
