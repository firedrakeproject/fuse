import numpy as np
import sympy as sp


def fold_reduce(func_list, *prev):
    """
    Right to left function comprehension

    :param: func_list: list of functions
    :param: prev: starting value(s)
    """
    for func in reversed(func_list):
        prev = func(*prev)
    return prev


def sympy_to_numpy(array, symbols, values):
    """
    Convert a sympy array to a numpy array.

    :param: array: sympy array
    :param: symbols: array of symbols contained in the sympy exprs
    :param: values: array of values to replace the symbols with.

    Due to how sympy handles arrays, we need to squeeze if resulting array
    is greater than 1 dimension to remove extra dimensions
    """
    substituted = array.subs({symbols[i]: values[i] for i in range(len(values))})
    nparray = np.array(substituted).astype(np.float64)

    if len(nparray.shape) > 1:
        return nparray.squeeze()

    if len(nparray.shape) == 0:
        return nparray.item()

    return nparray


def tabulate_sympy(expr, pts):
    # expr: sp matrix expression in x,y,z for components of R^d
    # pts: n values in R^d
    # returns: evaluation of expr at pts
    res = np.zeros((pts.shape[0],) + (expr.shape[-1],))
    i = 0
    syms = ["x", "y", "z"]
    for pt in pts:
        if not hasattr(pt, "__iter__"):
            pt = (pt,)
        subbed = expr.evalf(subs={syms[i]: pt[i] for i in range(len(pt))})
        subbed = np.array(subbed).astype(np.float64)
        res[i] = subbed[0]
        i += 1
    # final = res.squeeze()
    return res


def max_deg_sp_expr(sp_expr):
    degs = []
    if isinstance(sp_expr, sp.Matrix):
        for comp in sp_expr:
            # only compute degree if component is a polynomial
            if sp.sympify(comp).as_poly():
                degs += [sp.sympify(comp).as_poly().degree()]
    else:
        if sp.sympify(sp_expr).as_poly():
            degs += [sp.sympify(sp_expr).as_poly().degree()]

    return max(degs)
