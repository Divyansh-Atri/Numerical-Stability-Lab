"""
Error metrics for numerical stability analysis.

This module provides functions to compute various error measures
used in numerical analysis, including forward error, backward error,
and condition numbers.
"""

import numpy as np


def relative_error(computed, exact):
    """
    Compute the relative error between computed and exact values.
    
    Parameters
    ----------
    computed : float or array_like
        The computed (approximate) value
    exact : float or array_like
        The exact (true) value
        
    Returns
    -------
    float or ndarray
        Relative error: |computed - exact| / |exact|
        Returns inf if exact is zero
    """
    computed = np.asarray(computed)
    exact = np.asarray(exact)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err = np.abs(computed - exact) / np.abs(exact)
    
    return rel_err


def forward_error(computed, exact, norm='fro'):
    """
    Compute the forward error in a given norm.
    
    For vectors/matrices, uses the specified norm.
    For scalars, uses absolute error.
    
    Parameters
    ----------
    computed : array_like
        The computed result
    exact : array_like
        The exact result
    norm : str or int, optional
        The norm to use (default: 'fro' for Frobenius)
        
    Returns
    -------
    float
        Forward error: ||computed - exact||
    """
    computed = np.asarray(computed)
    exact = np.asarray(exact)
    
    if computed.ndim == 0:
        return np.abs(computed - exact)
    
    return np.linalg.norm(computed - exact, ord=norm)


def backward_error(A, b, x, norm='fro'):
    """
    Compute the backward error for a linear system Ax = b.
    
    The backward error is the smallest perturbation to the data
    (A, b) such that x is the exact solution to the perturbed problem.
    
    For the system Ax = b, the normwise backward error is:
        ||b - Ax|| / (||A|| ||x|| + ||b||)
    
    Parameters
    ----------
    A : ndarray
        Coefficient matrix
    b : ndarray
        Right-hand side vector
    x : ndarray
        Computed solution
    norm : str or int, optional
        The norm to use (default: 'fro')
        
    Returns
    -------
    float
        Normwise backward error
    """
    A = np.asarray(A)
    b = np.asarray(b)
    x = np.asarray(x)
    
    residual = b - A @ x
    
    norm_r = np.linalg.norm(residual, ord=norm if norm != 'fro' else 2)
    norm_A = np.linalg.norm(A, ord=norm)
    norm_x = np.linalg.norm(x, ord=2 if norm == 'fro' else norm)
    norm_b = np.linalg.norm(b, ord=2 if norm == 'fro' else norm)
    
    denominator = norm_A * norm_x + norm_b
    
    if denominator == 0:
        return np.inf
    
    return norm_r / denominator


def condition_number(A, norm='fro'):
    """
    Compute the condition number of a matrix.
    
    The condition number measures how sensitive the solution of Ax = b
    is to perturbations in A and b. A large condition number indicates
    an ill-conditioned problem.
    
    Parameters
    ----------
    A : ndarray
        Input matrix
    norm : str or int, optional
        The norm to use (default: 'fro' for Frobenius)
        Options: 'fro', 2, 1, -1, inf, -inf
        
    Returns
    -------
    float
        Condition number: ||A|| ||A^{-1}||
    """
    A = np.asarray(A)
    
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Condition number requires a square matrix")
    
    # Use numpy's built-in condition number for efficiency
    if norm == 'fro':
        return np.linalg.cond(A, p='fro')
    else:
        return np.linalg.cond(A, p=norm)


def residual_norm(A, b, x, norm=2):
    """
    Compute the residual norm ||b - Ax||.
    
    Parameters
    ----------
    A : ndarray
        Coefficient matrix
    b : ndarray
        Right-hand side vector
    x : ndarray
        Computed solution
    norm : int or str, optional
        The norm to use (default: 2)
        
    Returns
    -------
    float
        Residual norm
    """
    A = np.asarray(A)
    b = np.asarray(b)
    x = np.asarray(x)
    
    residual = b - A @ x
    return np.linalg.norm(residual, ord=norm)


def componentwise_backward_error(A, b, x):
    """
    Compute the componentwise backward error for Ax = b.
    
    This is a tighter bound than the normwise backward error,
    defined as:
        min {ε : (A + ΔA)x = b + Δb, |ΔA| ≤ ε|A|, |Δb| ≤ ε|b|}
    
    Approximated by: ||r|| / (||A|| ||x|| + ||b||)_∞
    where r = b - Ax is the residual.
    
    Parameters
    ----------
    A : ndarray
        Coefficient matrix
    b : ndarray
        Right-hand side vector
    x : ndarray
        Computed solution
        
    Returns
    -------
    float
        Componentwise backward error
    """
    A = np.asarray(A)
    b = np.asarray(b)
    x = np.asarray(x)
    
    residual = b - A @ x
    
    # Componentwise bound
    bound = np.abs(A) @ np.abs(x) + np.abs(b)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        comp_err = np.abs(residual) / bound
        comp_err = np.where(bound == 0, np.inf, comp_err)
    
    return np.max(comp_err)


def loss_of_precision(exact, computed):
    """
    Estimate the number of significant digits lost in a computation.
    
    Parameters
    ----------
    exact : float or array_like
        The exact value
    computed : float or array_like
        The computed value
        
    Returns
    -------
    float
        Approximate number of decimal digits lost
    """
    rel_err = relative_error(computed, exact)
    
    # Handle special cases
    if np.any(rel_err == 0):
        return 0.0
    if np.any(np.isinf(rel_err)):
        return np.inf
    
    # Convert to scalar if needed
    if isinstance(rel_err, np.ndarray):
        rel_err = np.max(rel_err)
    
    return -np.log10(rel_err)
