"""
Floating-point arithmetic tools and algorithms.

This module implements fundamental floating-point algorithms
demonstrating numerical stability concepts, including:
- Summation algorithms (naive, Kahan, pairwise)
- Polynomial evaluation (naive, Horner)
- Catastrophic cancellation examples
"""

import numpy as np


def machine_epsilon(dtype=np.float64):
    """
    Compute machine epsilon for a given floating-point type.
    
    Machine epsilon is the smallest number ε such that 1 + ε > 1
    in floating-point arithmetic.
    
    Parameters
    ----------
    dtype : numpy dtype, optional
        The floating-point type (default: np.float64)
        
    Returns
    -------
    float
        Machine epsilon for the given type
    """
    eps = dtype(1.0)
    while dtype(1.0) + dtype(eps / 2.0) > dtype(1.0):
        eps = dtype(eps / 2.0)
    return eps


def ulp_distance(x, y, dtype=np.float64):
    """
    Compute the distance between two floats in units of last place (ULP).
    
    Parameters
    ----------
    x : float
        First value
    y : float
        Second value
    dtype : numpy dtype, optional
        The floating-point type (default: np.float64)
        
    Returns
    -------
    float
        ULP distance between x and y
    """
    x = dtype(x)
    y = dtype(y)
    
    if x == y:
        return 0.0
    
    # Get the spacing at the larger magnitude
    spacing = np.spacing(max(abs(x), abs(y)))
    
    return abs(x - y) / spacing


def naive_sum(arr):
    """
    Naive summation algorithm (left-to-right accumulation).
    
    This is the standard summation that accumulates errors
    as the sum progresses.
    
    Parameters
    ----------
    arr : array_like
        Array of numbers to sum
        
    Returns
    -------
    float
        Sum of array elements
    """
    arr = np.asarray(arr)
    total = 0.0
    for x in arr:
        total = total + x
    return total


def kahan_sum(arr):
    """
    Kahan (compensated) summation algorithm.
    
    This algorithm reduces the numerical error in the sum of a
    sequence of finite-precision floating-point numbers by
    maintaining a running compensation for lost low-order bits.
    
    Reference:
        Kahan, W. (1965). "Further remarks on reducing truncation errors"
    
    Parameters
    ----------
    arr : array_like
        Array of numbers to sum
        
    Returns
    -------
    float
        Compensated sum of array elements
    """
    arr = np.asarray(arr)
    
    total = 0.0
    compensation = 0.0  # Running compensation for lost low-order bits
    
    for x in arr:
        y = x - compensation      # Subtract the compensation
        t = total + y             # Add to running total
        compensation = (t - total) - y  # Recover the low-order bits
        total = t
    
    return total


def pairwise_sum(arr):
    """
    Pairwise summation algorithm (recursive).
    
    This algorithm reduces rounding error by summing pairs of numbers
    recursively, achieving O(log n) error growth instead of O(n).
    
    Parameters
    ----------
    arr : array_like
        Array of numbers to sum
        
    Returns
    -------
    float
        Pairwise sum of array elements
    """
    arr = np.asarray(arr)
    n = len(arr)
    
    if n == 0:
        return 0.0
    elif n == 1:
        return arr[0]
    elif n == 2:
        return arr[0] + arr[1]
    else:
        mid = n // 2
        return pairwise_sum(arr[:mid]) + pairwise_sum(arr[mid:])


def naive_poly_eval(coeffs, x):
    """
    Naive polynomial evaluation using powers of x.
    
    For polynomial p(x) = a₀ + a₁x + a₂x² + ... + aₙxⁿ,
    compute each term separately and sum.
    
    This is numerically unstable due to:
    1. Repeated multiplication to compute powers
    2. Potential for catastrophic cancellation in summation
    
    Parameters
    ----------
    coeffs : array_like
        Polynomial coefficients [a₀, a₁, ..., aₙ]
    x : float
        Point at which to evaluate
        
    Returns
    -------
    float
        p(x)
    """
    coeffs = np.asarray(coeffs)
    result = 0.0
    
    for i, a in enumerate(coeffs):
        result += a * (x ** i)
    
    return result


def horner_eval(coeffs, x):
    """
    Horner's method for polynomial evaluation.
    
    For polynomial p(x) = a₀ + a₁x + a₂x² + ... + aₙxⁿ,
    rewrite as: p(x) = a₀ + x(a₁ + x(a₂ + ... + x(aₙ)))
    
    This is numerically stable and efficient:
    - Only n multiplications and n additions
    - Minimizes rounding error accumulation
    
    Parameters
    ----------
    coeffs : array_like
        Polynomial coefficients [a₀, a₁, ..., aₙ]
    x : float
        Point at which to evaluate
        
    Returns
    -------
    float
        p(x)
    """
    coeffs = np.asarray(coeffs)
    
    # Start with highest degree coefficient
    result = coeffs[-1]
    
    # Work backwards through coefficients
    for a in coeffs[-2::-1]:
        result = result * x + a
    
    return result


def catastrophic_cancellation_example(x, method='unstable'):
    """
    Demonstrate catastrophic cancellation in computing (1 - cos(x)) / x².
    
    For small x, direct computation loses precision due to cancellation.
    The stable form uses the identity: (1 - cos(x)) / x² = (sin(x/2) / (x/2))² / 2
    
    Parameters
    ----------
    x : float or array_like
        Input value(s)
    method : str, optional
        'unstable': direct computation (1 - cos(x)) / x²
        'stable': using sin(x/2) identity
        
    Returns
    -------
    float or ndarray
        Computed value
    """
    x = np.asarray(x)
    
    if method == 'unstable':
        # Direct computation - unstable for small x
        return (1.0 - np.cos(x)) / (x * x)
    
    elif method == 'stable':
        # Stable reformulation using half-angle
        half_x = x / 2.0
        sin_half = np.sin(half_x)
        return (sin_half / half_x) ** 2 / 2.0
    
    else:
        raise ValueError(f"Unknown method: {method}")


def quadratic_formula(a, b, c, method='unstable'):
    """
    Solve quadratic equation ax² + bx + c = 0.
    
    The standard formula can suffer from catastrophic cancellation
    when b² >> 4ac. A stable variant avoids this.
    
    Parameters
    ----------
    a, b, c : float
        Quadratic coefficients
    method : str, optional
        'unstable': standard formula
        'stable': numerically stable variant
        
    Returns
    -------
    tuple of float
        The two roots (x₁, x₂)
    """
    if method == 'unstable':
        # Standard formula - can be unstable
        discriminant = b**2 - 4*a*c
        sqrt_disc = np.sqrt(discriminant)
        
        x1 = (-b + sqrt_disc) / (2*a)
        x2 = (-b - sqrt_disc) / (2*a)
        
        return (x1, x2)
    
    elif method == 'stable':
        # Stable variant avoiding cancellation
        discriminant = b**2 - 4*a*c
        sqrt_disc = np.sqrt(discriminant)
        
        # Compute the root with same sign as b first
        if b >= 0:
            x1 = (-b - sqrt_disc) / (2*a)
        else:
            x1 = (-b + sqrt_disc) / (2*a)
        
        # Use Vieta's formula for the second root
        x2 = c / (a * x1)
        
        return (x1, x2)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def log1p_example(x, method='unstable'):
    """
    Compute log(1 + x) for small x.
    
    Direct computation log(1 + x) loses precision for small x
    due to rounding in the addition. The stable method uses
    the built-in log1p function.
    
    Parameters
    ----------
    x : float or array_like
        Input value(s)
    method : str, optional
        'unstable': direct computation log(1 + x)
        'stable': using np.log1p
        
    Returns
    -------
    float or ndarray
        Computed value
    """
    x = np.asarray(x)
    
    if method == 'unstable':
        return np.log(1.0 + x)
    elif method == 'stable':
        return np.log1p(x)
    else:
        raise ValueError(f"Unknown method: {method}")


def expm1_example(x, method='unstable'):
    """
    Compute exp(x) - 1 for small x.
    
    Direct computation loses precision for small x.
    The stable method uses the built-in expm1 function.
    
    Parameters
    ----------
    x : float or array_like
        Input value(s)
    method : str, optional
        'unstable': direct computation exp(x) - 1
        'stable': using np.expm1
        
    Returns
    -------
    float or ndarray
        Computed value
    """
    x = np.asarray(x)
    
    if method == 'unstable':
        return np.exp(x) - 1.0
    elif method == 'stable':
        return np.expm1(x)
    else:
        raise ValueError(f"Unknown method: {method}")
