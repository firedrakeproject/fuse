"""Utility functions for the fuse package, including symbolic and matrix operations."""

import numpy as np
import sympy as sp
import math


def fold_reduce(func_list, *prev):
    """Apply a list of functions from right to left (right-to-left function comprehension).

    Parameters
    ----------
    func_list : list of callable
        The list of functions to apply sequentially.
    *prev : tuple
        The starting value(s) passed to the first function invocation.

    Returns
    -------
    any
        The final result after applying all functions.
    """
    for func in reversed(func_list):
        prev = func(*prev)
    return prev


def sympy_to_numpy(array, symbols, values):
    """Evaluate symbols at values, then convert to a NumPy array if all have been replaced.

    Parameters
    ----------
    array : sympy.Matrix or sympy.Array
        The SymPy expression array to substitute into.
    symbols : list of sympy.Symbol
        The list of symbols contained within the SymPy expressions.
    values : list
        The values to replace the symbols with.

    Returns
    -------
    numpy.ndarray or sympy.Matrix
        The evaluated array converted to NumPy float64 if all symbols are replaced,
        otherwise the substituted SymPy matrix/array object.
    """
    substituted = array.subs({symbols[i]: values[i] for i in range(len(values))})

    if len(array.atoms(sp.Symbol)) == len(values) and all(not isinstance(v, sp.Expr) for v in values):
        nparray = np.array(substituted).astype(np.float64)

        if len(nparray.shape) > 1:
            return nparray.squeeze()

        if len(nparray.shape) == 0:
            return nparray.item()
    else:
        nparray = substituted

    return nparray


def tabulate_sympy(expr, pts):
    """Evaluate a SymPy expression at multiple points.

    Parameters
    ----------
    expr : sympy.Expr
        SymPy matrix expression in x, y, z for components of R^d.
    pts : list of array-like
        The coordinates of n points in R^d.

    Returns
    -------
    numpy.ndarray
        The evaluation of the expression at the given points.
    """
    res = np.array(pts)
    i = 0
    syms = ["x", "y", "z"]
    for pt in pts:
        if not hasattr(pt, "__iter__"):
            pt = (pt,)
        subbed = expr.evalf(subs={syms[i]: pt[i] for i in range(len(pt))})
        subbed = np.array(subbed).astype(np.float64)
        res[i] = subbed[0]
        i += 1
    final = res.squeeze()
    return final


def max_deg_sp_mat(sp_mat):
    """Compute the maximum polynomial degree among all components of a SymPy matrix.

    Parameters
    ----------
    sp_mat : iterable of sympy.Expr
        The SymPy matrix or collection of expressions.

    Returns
    -------
    int
        The maximum degree among all polynomial components.
    """
    degs = []
    for comp in sp_mat:
        # only compute degree if component is a polynomial
        if sp.sympify(comp).as_poly():
            degs += [sp.sympify(comp).as_poly().degree()]
    return max(degs)


def numpy_to_str_tuple(arr, scale=1):
    """Convert a NumPy array to a comma-separated string representation enclosed in parentheses.

    Parameters
    ----------
    arr : array-like
        The array of numerical values to convert.
    scale : float, default 1
        A multiplier applied to each element before conversion.

    Returns
    -------
    str
        The formatted string tuple representation (e.g., '(1.0,2.0)').
    """
    str_as = []
    for a in arr:
        str_a = str(scale*a)
        str_as += [str_a]
    return f'({",".join(str_as)})'


# def orientation_value(identity_arg, perm_arg):
#     # copy arrays as they are modified in place
#     identity = identity_arg.copy()
#     perm = perm_arg.copy()

#     val = 0
#     for i in range(len(identity)):
#         loc = perm.index(identity[i])
#         perm.remove(identity[i])
#         val += loc * math.factorial(len(identity) - i - 1)
#     return val

def orientation_value(identity_arg, perm_arg):
    """Compute the orientation value/index corresponding to a permutation relative to an identity array.

    Parameters
    ----------
    identity_arg : list or array-like
        The reference identity array/sequence.
    perm_arg : list or array-like
        The permuted sequence to compute the orientation value for.

    Returns
    -------
    int
        The calculated orientation value.
    """
    # copy arrays as they are modified in place
    identity = identity_arg.copy()
    perm = perm_arg.copy()

    val = 0
    for i in range(len(perm)):
        loc = identity.index(perm[i])
        identity.remove(perm[i])
        val += loc * math.factorial(len(perm) - i - 1)
    return val
