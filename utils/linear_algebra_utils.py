"""
Linear algebra utilities for numerical stability experiments.

This module implements fundamental linear algebra algorithms
from scratch to study their numerical stability properties:
- Gaussian elimination (with and without pivoting)
- Least squares solvers (normal equations vs QR)
- Matrix factorizations
"""

import numpy as np


def gaussian_elimination(A, b, pivot=False):
    """
    Solve Ax = b using Gaussian elimination.
    
    This is the basic elimination algorithm without pivoting,
    which can be numerically unstable.
    
    Parameters
    ----------
    A : ndarray, shape (n, n)
        Coefficient matrix
    b : ndarray, shape (n,)
        Right-hand side vector
    pivot : bool, optional
        If True, use partial pivoting (default: False)
        
    Returns
    -------
    ndarray, shape (n,)
        Solution vector x
        
    Raises
    ------
    ValueError
        If matrix is singular or nearly singular
    """
    n = len(b)
    A = A.astype(np.float64, copy=True)
    b = b.astype(np.float64, copy=True)
    
    # Forward elimination
    for k in range(n - 1):
        if pivot:
            # Partial pivoting: find row with largest pivot
            max_row = k + np.argmax(np.abs(A[k:, k]))
            if max_row != k:
                # Swap rows
                A[[k, max_row]] = A[[max_row, k]]
                b[[k, max_row]] = b[[max_row, k]]
        
        # Check for zero pivot
        if np.abs(A[k, k]) < 1e-14:
            raise ValueError(f"Zero or near-zero pivot at position {k}")
        
        # Eliminate column k below diagonal
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if np.abs(A[i, i]) < 1e-14:
            raise ValueError(f"Zero or near-zero diagonal at position {i}")
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    
    return x


def gaussian_elimination_with_pivoting(A, b):
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.
    
    Partial pivoting improves numerical stability by choosing
    the largest available pivot at each step.
    
    Parameters
    ----------
    A : ndarray, shape (n, n)
        Coefficient matrix
    b : ndarray, shape (n,)
        Right-hand side vector
        
    Returns
    -------
    ndarray, shape (n,)
        Solution vector x
    """
    return gaussian_elimination(A, b, pivot=True)


def lu_factorization(A, pivot=False):
    """
    Compute LU factorization of A.
    
    Decomposes A into A = LU (or PA = LU with pivoting)
    where L is lower triangular and U is upper triangular.
    
    Parameters
    ----------
    A : ndarray, shape (n, n)
        Input matrix
    pivot : bool, optional
        If True, use partial pivoting (default: False)
        
    Returns
    -------
    L : ndarray, shape (n, n)
        Lower triangular matrix
    U : ndarray, shape (n, n)
        Upper triangular matrix
    P : ndarray, shape (n, n), optional
        Permutation matrix (only if pivot=True)
    """
    n = A.shape[0]
    A = A.astype(np.float64, copy=True)
    L = np.eye(n)
    P = np.eye(n)
    
    for k in range(n - 1):
        if pivot:
            # Partial pivoting
            max_row = k + np.argmax(np.abs(A[k:, k]))
            if max_row != k:
                # Swap rows in A, L, and P
                A[[k, max_row]] = A[[max_row, k]]
                P[[k, max_row]] = P[[max_row, k]]
                if k > 0:
                    L[[k, max_row], :k] = L[[max_row, k], :k]
        
        # Compute multipliers and update
        for i in range(k + 1, n):
            L[i, k] = A[i, k] / A[k, k]
            A[i, k:] -= L[i, k] * A[k, k:]
    
    U = A
    
    if pivot:
        return L, U, P
    else:
        return L, U


def solve_normal_equations(A, b):
    """
    Solve least squares problem min ||Ax - b|| using normal equations.
    
    Forms and solves the normal equations: A^T A x = A^T b
    
    This method is simple but numerically unstable:
    - Squares the condition number: κ(A^T A) = κ(A)²
    - Can lose accuracy for ill-conditioned problems
    
    Parameters
    ----------
    A : ndarray, shape (m, n)
        Coefficient matrix (m >= n)
    b : ndarray, shape (m,)
        Right-hand side vector
        
    Returns
    -------
    ndarray, shape (n,)
        Least squares solution x
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    # Form normal equations
    ATA = A.T @ A
    ATb = A.T @ b
    
    # Solve using Gaussian elimination with pivoting
    x = gaussian_elimination_with_pivoting(ATA, ATb)
    
    return x


def qr_factorization_gram_schmidt(A, modified=True):
    """
    Compute QR factorization using Gram-Schmidt orthogonalization.
    
    Parameters
    ----------
    A : ndarray, shape (m, n)
        Input matrix
    modified : bool, optional
        If True, use modified Gram-Schmidt (more stable)
        If False, use classical Gram-Schmidt
        
    Returns
    -------
    Q : ndarray, shape (m, n)
        Orthogonal matrix
    R : ndarray, shape (n, n)
        Upper triangular matrix
    """
    m, n = A.shape
    A = A.astype(np.float64, copy=True)
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    if modified:
        # Modified Gram-Schmidt (more stable)
        for j in range(n):
            Q[:, j] = A[:, j]
            
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], Q[:, j])
                Q[:, j] -= R[i, j] * Q[:, i]
            
            R[j, j] = np.linalg.norm(Q[:, j])
            if R[j, j] > 1e-14:
                Q[:, j] /= R[j, j]
            else:
                raise ValueError(f"Matrix is rank deficient at column {j}")
    
    else:
        # Classical Gram-Schmidt (less stable)
        for j in range(n):
            Q[:, j] = A[:, j]
            
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], A[:, j])
                Q[:, j] -= R[i, j] * Q[:, i]
            
            R[j, j] = np.linalg.norm(Q[:, j])
            if R[j, j] > 1e-14:
                Q[:, j] /= R[j, j]
            else:
                raise ValueError(f"Matrix is rank deficient at column {j}")
    
    return Q, R


def solve_qr(A, b, method='modified'):
    """
    Solve least squares problem min ||Ax - b|| using QR factorization.
    
    This is more numerically stable than normal equations:
    - Preserves the condition number: κ(R) ≈ κ(A)
    - Recommended for ill-conditioned problems
    
    Parameters
    ----------
    A : ndarray, shape (m, n)
        Coefficient matrix (m >= n)
    b : ndarray, shape (m,)
        Right-hand side vector
    method : str, optional
        'modified': use modified Gram-Schmidt (default)
        'classical': use classical Gram-Schmidt
        
    Returns
    -------
    ndarray, shape (n,)
        Least squares solution x
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    # Compute QR factorization
    Q, R = qr_factorization_gram_schmidt(A, modified=(method == 'modified'))
    
    # Solve R x = Q^T b
    QTb = Q.T @ b
    x = np.zeros(A.shape[1])
    
    # Back substitution
    for i in range(A.shape[1] - 1, -1, -1):
        x[i] = (QTb[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
    
    return x


def matrix_inverse_solve(A, b):
    """
    Solve Ax = b by explicitly computing A^{-1} and forming x = A^{-1} b.
    
    This is numerically unstable and inefficient:
    - More expensive than direct solution
    - Amplifies rounding errors
    - Should be avoided in practice
    
    Included here for educational comparison.
    
    Parameters
    ----------
    A : ndarray, shape (n, n)
        Coefficient matrix
    b : ndarray, shape (n,)
        Right-hand side vector
        
    Returns
    -------
    ndarray, shape (n,)
        Solution vector x
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    # Compute inverse using Gaussian elimination
    n = A.shape[0]
    A_aug = np.hstack([A, np.eye(n)])
    
    # Forward elimination with pivoting
    for k in range(n):
        # Partial pivoting
        max_row = k + np.argmax(np.abs(A_aug[k:, k]))
        if max_row != k:
            A_aug[[k, max_row]] = A_aug[[max_row, k]]
        
        # Eliminate
        pivot = A_aug[k, k]
        A_aug[k, :] /= pivot
        
        for i in range(n):
            if i != k:
                factor = A_aug[i, k]
                A_aug[i, :] -= factor * A_aug[k, :]
    
    # Extract inverse
    A_inv = A_aug[:, n:]
    
    # Multiply by b
    x = A_inv @ b
    
    return x


def create_ill_conditioned_matrix(n, condition_number):
    """
    Create an ill-conditioned matrix with specified condition number.
    
    Uses SVD construction: A = U Σ V^T where Σ has controlled singular values.
    
    Parameters
    ----------
    n : int
        Matrix size
    condition_number : float
        Desired condition number
        
    Returns
    -------
    ndarray, shape (n, n)
        Ill-conditioned matrix
    """
    # Random orthogonal matrices
    U, _ = np.linalg.qr(np.random.randn(n, n))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Construct singular values with desired condition number
    sigma = np.logspace(0, -np.log10(condition_number), n)
    
    # Form matrix
    A = U @ np.diag(sigma) @ V.T
    
    return A


def create_hilbert_matrix(n):
    """
    Create the Hilbert matrix of size n.
    
    The Hilbert matrix H[i,j] = 1/(i+j+1) is a classic example
    of an ill-conditioned matrix. Its condition number grows
    exponentially with n.
    
    Parameters
    ----------
    n : int
        Matrix size
        
    Returns
    -------
    ndarray, shape (n, n)
        Hilbert matrix
    """
    i, j = np.ogrid[0:n, 0:n]
    H = 1.0 / (i + j + 1)
    return H
