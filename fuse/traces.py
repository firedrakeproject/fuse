import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from fuse.utils import sympy_to_numpy, numpy_to_str_tuple


class Trace():

    def __init__(self, cell):
        self.domain = cell

    def __call__(self, trace_entity):
        raise NotImplementedError("Trace uninstanitated")

    def plot(self, ax, coord, trace_entity, **kwargs):
        raise NotImplementedError("Trace uninstanitated")

    def tabulate(self, Qwts, trace_entity):
        raise NotImplementedError("Tabulation uninstantiated")

    def _to_dict(self):
        return {"trace": str(self)}

    def dict_id(self):
        return "Trace"

    def _from_dict(obj_dict):
        # might want to actually save these as functions or something for ambiguity?
        tr_id = obj_dict["trace"]
        if tr_id == "Grad":
            return TrGrad
        elif tr_id == "Hess":
            return TrHess
        raise ValueError("Trace not found")


class TrGrad(Trace):

    def __init__(self, cell):
        super(TrGrad, self).__init__(cell)

    def __call__(self, v, trace_entity):
        # Compute grad v and then dot with tangent rotated according to the group member
        raise NotImplementedError("Gradient immersions are under development")
        g = None
        tangent = np.array(g(np.array(self.domain.basis_vectors())[0]))

        def apply(*x):
            X = sp.DeferredVector('x')
            dX = tuple([X[i] for i in range(self.domain.dim())])
            compute_v = v(*dX, sym=True)
            grad_v = sp.Matrix([sp.diff(compute_v, dX[i]) for i in range(len(dX))])
            eval_grad_v = sympy_to_numpy(grad_v, dX, v.attach_func(*x))
            result = np.dot(tangent, np.array(eval_grad_v))

            if not hasattr(result, "__iter__"):
                return (result,)
            return tuple(result)
        return apply

    def plot(self, ax, coord, trace_entity, **kwargs):
        circle1 = plt.Circle(coord, 0.075, fill=False, **kwargs)
        ax.add_patch(circle1)

    def to_tikz(self, coord, trace_entity, scale, color="black"):
        return f"\\draw[{color}] {numpy_to_str_tuple(coord, scale)} circle (4pt) node[anchor = south] {{}};"

    def __repr__(self):
        return "Grad"


class TrHess(Trace):

    def __init__(self, cell):
        super(TrHess, self).__init__(cell)

    def __call__(self, v, trace_entity):
        raise NotImplementedError("Hessian trace needs reviewing")
        g = None
        b0, b1 = self.domain.basis_vectors()
        tangent0 = np.array(g(b0))
        tangent1 = np.array(g(b1))

        def apply(*x):
            X = sp.DeferredVector('x')

            dX = tuple([X[i] for i in range(self.domain.dim())])
            hess_v = sp.Matrix([[sp.diff(v(*dX, sym=True), dX[i], dX[j]) for i in range(len(dX))] for j in range(len(dX))])
            eval_hess_v = sympy_to_numpy(hess_v, dX, v.attach_func(*x))
            result = np.dot(np.matmul(tangent0, np.array(eval_hess_v)), tangent1)
            if not hasattr(result, "__iter__"):
                return (result,)
            return tuple(result)
        return apply

    def plot(self, ax, coord, trace_entity, **kwargs):
        circle1 = plt.Circle(coord, 0.15, fill=False, **kwargs)
        ax.add_patch(circle1)

    def to_tikz(self, coord, trace_entity, scale, color="black"):
        return f"\\draw[{color}] {numpy_to_str_tuple(coord, scale)} circle (6pt) node[anchor = south] {{}};"

    def __repr__(self):
        return "Hess"
