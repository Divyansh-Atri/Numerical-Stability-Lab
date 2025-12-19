"""
Utility modules for numerical stability experiments.
"""

from .error_metrics import *
from .floating_point_tools import *
from .linear_algebra_utils import *

__all__ = [
    'relative_error',
    'forward_error',
    'backward_error',
    'condition_number',
    'machine_epsilon',
    'ulp_distance',
    'kahan_sum',
    'naive_sum',
    'horner_eval',
    'naive_poly_eval',
    'gaussian_elimination',
    'gaussian_elimination_with_pivoting',
    'solve_normal_equations',
    'solve_qr',
    'matrix_inverse_solve',
]
