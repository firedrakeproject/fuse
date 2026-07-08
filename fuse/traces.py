import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import reduce
from fuse.utils import sympy_to_numpy, numpy_to_str_tuple


def _resolve_direction(spec, domain, trace_entity):
    """Resolve a direction spec - a fixed ambient vector, or one of the
    keywords "tangent"/"normal" - to a concrete vector, given the facet
    (trace_entity) immersed within domain. "tangent" mirrors
    TrHCurl.tabulate; "normal" mirrors TrHDiv.tabulate."""
    if not isinstance(spec, str):
        return np.asarray(spec, dtype=float)
    sd = domain.get_spatial_dimension()
    basis = np.array(domain.basis_vectors(entity=trace_entity))
    if spec == "tangent":
        if trace_entity.dimension != 1:
            raise ValueError('"tangent" direction requires a 1D (edge) entity')
        return basis[0]
    if spec == "normal":
        if trace_entity.dimension != sd - 1:
            raise ValueError('"normal" direction is only defined on facets (codimension 1 entities)')
        if sd == 2:
            return np.matmul(basis, np.array([[0, -1], [1, 0]]))[0]
        if sd == 3:
            return np.cross(basis[0], basis[1])
        raise ValueError("normal direction not implemented in dimension > 3")
    raise ValueError(f"Unknown direction keyword {spec!r}")


def _directional_deriv_terms(directions, domain, trace_entity):
    """Expand the order-k mixed directional derivative d/dv_1 ... d/dv_k into
    FIAT-style [(coeff, alpha)] terms, by taking the outer product of the k
    resolved direction vectors and accumulating entries that land on the same
    multi-index - generalizing FIAT's own PointSecondDerivative
    (FIAT/functional.py, which does exactly this for k=2 via numpy.outer and
    a defaultdict keyed by alpha) to arbitrary k."""
    sd = domain.get_spatial_dimension()
    vectors = [_resolve_direction(s, domain, trace_entity) for s in directions]
    tensor = reduce(np.multiply.outer, vectors)
    tau = defaultdict(float)
    for index in np.ndindex(tensor.shape):
        alpha = [0] * sd
        for i in index:
            alpha[i] += 1
        tau[tuple(alpha)] += tensor[index]
    return [(coeff, alpha) for alpha, coeff in tau.items()]


class Trace():

    def __init__(self, cell=None, alpha=None, directions=None):
        self.domain = cell
        self.alpha = alpha
        self.directions = directions

    def add_cell(self, cell):
        return type(self)(cell=cell, alpha=self.alpha, directions=self.directions)

    def __call__(self, trace_entity):
        raise NotImplementedError("Trace uninstanitated")

    def plot(self, ax, coord, trace_entity, **kwargs):
        raise NotImplementedError("Trace uninstanitated")

    def tabulate(self, Qwts, trace_entity):
        raise NotImplementedError("Tabulation uninstantiated")

    def tabulate_derivs(self, Qwts, trace_entity):
        if self.alpha is not None and self.directions is not None:
            raise ValueError("Specify either alpha or directions, not both")
        if self.directions is not None:
            return _directional_deriv_terms(self.directions, self.domain, trace_entity)
        if self.alpha is None:
            return None
        return [(1.0, self.alpha)]

    def _to_dict(self):
        return {"trace": str(self)}

    def dict_id(self):
        return "Trace"

    def _from_dict(obj_dict):
        # might want to actually save these as functions or something for ambiguity?
        tr_id = obj_dict["trace"]
        if tr_id == "H1":
            return TrH1
        elif tr_id == "HDiv":
            return TrHDiv
        elif tr_id == "HCurl":
            return TrHCurl
        elif tr_id == "Grad":
            return TrGrad
        elif tr_id == "Hess":
            return TrHess
        raise ValueError("Trace not found")


class TrH1(Trace):

    def __call__(self, v, trace_entity):
        return v

    def plot(self, ax, coord, trace_entity, **kwargs):
        ax.scatter(*coord, **kwargs)

    def to_tikz(self, coord, trace_entity, scale, color="black"):
        return f"\\filldraw[{color}] {numpy_to_str_tuple(coord, scale)} circle (2pt) node[anchor = south] {{}};"

    def tabulate(self, Qpts, trace_entity):
        return np.ones(len(Qpts))
        # return Qwts

    def manipulate_basis(self, basis):
        return np.array([1])

    def __repr__(self):
        return "H1"


class TrHDiv(Trace):

    def __call__(self, v, trace_entity):
        def apply(*x):
            result = np.dot(self.tabulate(None, trace_entity), np.array(v(*x)).squeeze())
            if isinstance(result, np.float64):
                # todo: might always be a float
                return (result,)
            return tuple(result)
        return apply

    def plot(self, ax, coord, trace_entity, **kwargs):
        # plot dofs of the type associated with this space
        vec = self.tabulate([], trace_entity).squeeze()
        ax.quiver(*coord, *vec, **kwargs)

    def tabulate(self, Qwts, trace_entity):
        # entityBasis = np.array(trace_entity.basis_vectors())
        cellEntityBasis = np.array(self.domain.basis_vectors(entity=trace_entity))
        # basis = np.matmul(entityBasis, cellEntityBasis)
        basis = cellEntityBasis
        if trace_entity.dimension == 1:
            result = np.matmul(basis, np.array([[0, -1], [1, 0]]))
        elif trace_entity.dimension == 2:
            result = np.cross(basis[0], basis[1])
        else:
            raise ValueError("Immersion of HDiv edges not defined in 3D")
        return result

    def manipulate_basis(self, basis):
        if basis.shape[-1] == 1:
            return basis
        elif basis.shape == (1, 2):
            result = np.matmul(basis, np.array([[0, -1], [1, 0]]))
        elif basis.shape == (2, 2):
            # Two dim cross product - pad with zeros and take z component of result
            zeros_row = np.zeros((basis.shape[0], 1), dtype=basis.dtype)
            basis = np.hstack([basis, zeros_row])
            result = np.cross(basis[0], basis[1])[2]
        elif basis.shape == (2, 3):
            result = np.cross(basis[0], basis[1])
        else:
            raise ValueError("Immersion of HDiv edges not defined in 3D")
        return result

    def to_tikz(self, coord, trace_entity, scale, color="black"):
        vec = self.tabulate([], trace_entity).squeeze()
        end_point = [coord[i] + 0.25*vec[i] for i in range(len(coord))]
        arw = "-{Stealth[length=3mm, width=2mm]}"
        return f"\\draw[thick, {color}, {arw}] {numpy_to_str_tuple(coord, scale)} -- {numpy_to_str_tuple(end_point, scale)};"

    def __repr__(self):
        return "HDiv"


class TrHCurl(Trace):

    def __call__(self, v, trace_entity):
        def apply(*x):
            result = np.dot(self.tabulate(None, trace_entity), np.array(v(*x)).squeeze())
            if isinstance(result, np.float64):
                return (result,)
            return tuple(result)
        return apply

    def tabulate(self, Qwts, trace_entity):
        # tangent = trace_entity.basis_vectors()
        subEntityBasis = np.array(self.domain.basis_vectors(entity=trace_entity))
        # result = np.matmul(tangent, subEntityBasis)
        return subEntityBasis
        # return result

    def manipulate_basis(self, basis):
        return basis[0]

    def plot(self, ax, coord, trace_entity, **kwargs):
        vec = self.tabulate([], trace_entity).squeeze()
        ax.quiver(*coord, *vec, **kwargs)

    def to_tikz(self, coord, trace_entity, scale, color="black"):
        vec = self.tabulate([], trace_entity).squeeze()
        end_point = [coord[i] + 0.25*vec[i] for i in range(len(coord))]
        arw = "-{Stealth[length=3mm, width=2mm]}"
        return f"\\draw[thick, {color}, {arw}] {numpy_to_str_tuple(coord, scale)} -- {numpy_to_str_tuple(end_point, scale)};"

    def __repr__(self):
        return "HCurl"


class TrGrad(Trace):

    def __call__(self, v, trace_entity):
        # Compute grad v and then dot with tangent rotated according to the group member
        # raise NotImplementedError("Gradient immersions are under development")
        def apply(*x):
            result = np.dot(self.tabulate(None, trace_entity), np.array(v(*x)).squeeze())
            if isinstance(result, np.float64):
                return (result,)
            return tuple(result)
        return apply

    def convert_to_fiat(self, qpts, pts, wts):
        shp = (self.domain.get_spatial_dimension(),)
        alphas = []
        for i in range(pts.shape[0]):
            new = np.zeros(shp, dtype=int)
            new[i] = 1
            alphas += [tuple(new)]
        deriv_dicts = []
        for alpha in alphas:
            deriv_dicts += [{tuple(p): [(1.0, tuple(alpha), tuple())] for p in pts.T}]

        # self.alpha = tuple(alpha)
        # self.order = sum(self.alpha)
        return [({}, d) for d in deriv_dicts]

    def plot(self, ax, coord, trace_entity, g, **kwargs):
        circle1 = plt.Circle(coord, 0.075, fill=False, **kwargs)
        ax.add_patch(circle1)

    def tabulate(self, Qpts, trace_entity):
        return np.array([])

    def to_tikz(self, coord, trace_entity, scale, color="black"):
        return f"\\draw[{color}] {numpy_to_str_tuple(coord, scale)} circle (4pt) node[anchor = south] {{}};"

    def __repr__(self):
        return "Grad"


class TrHess(Trace):

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

    def tabulate(self, Qpts, trace_entity):
        return np.array([])

    def plot(self, ax, coord, trace_entity, **kwargs):
        circle1 = plt.Circle(coord, 0.15, fill=False, **kwargs)
        ax.add_patch(circle1)

    def to_tikz(self, coord, trace_entity, scale, color="black"):
        return f"\\draw[{color}] {numpy_to_str_tuple(coord, scale)} circle (6pt) node[anchor = south] {{}};"

    def __repr__(self):
        return "Hess"
