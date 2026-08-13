from FIAT.polynomial_set import ONPolynomialSet
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import cell_to_simplex
from FIAT import expansions, polynomial_set, reference_element
from itertools import chain
from fuse.utils import tabulate_sympy, max_deg_sp_mat
import sympy as sp
import numpy as np
from functools import total_ordering


def normalise_shape(shape):
    """Map declared value shape to a tuple of ints.
    """
    if isinstance(shape, (int, np.integer)):
        shape = () if shape == 0 else (shape,)
    return tuple(int(extent) for extent in shape)


def weighted_shape(weight, space):
    """The value shape contributed by one weighted space of a combination.
    """
    if isinstance(weight, sp.Matrix):
        if space.shape:
            raise ValueError(
                f"Cannot weight a space of value shape {space.shape} by the matrix {weight}: only scalar spaces have matrix valued weights.")
        return (len(weight),)
    return space.shape


@total_ordering
class PolynomialSpace(object):
    """
    maxdegree: the highest polynomial degree present in the space.

    mindegree: the exclusive lower degree bound; the space keeps degrees in
    (mindegree, maxdegree]. A complete space (constants included) has mindegree = -1.

    shape: the value shape of the space, as a tuple or integer.
    The default empty tuple results in a scalar valued space.
    """

    def __init__(self, maxdegree, mindegree=-1, shape=()):
        self.maxdegree = maxdegree
        self.mindegree = mindegree
        self.shape = normalise_shape(shape)

    def complete(self):
        return self.mindegree < 0

    def degree(self):
        return self.maxdegree

    def to_ON_polynomial_set(self, ref_el, k=None):
        # how does super/sub degrees work here
        if not isinstance(ref_el, reference_element.Cell):
            ref_el = ref_el.to_fiat()
        ref_el = cell_to_simplex(ref_el)
        shape = self.shape

        base_ON = ONPolynomialSet(ref_el, self.maxdegree, shape, scale="orthonormal")
        dimPmin = expansions.polynomial_dimension(ref_el, self.mindegree)
        if dimPmin == 0:
            return base_ON
        dimPmax = expansions.polynomial_dimension(ref_el, self.maxdegree)
        if shape:
            num_components = int(np.prod(shape))
            indices = list(chain(*(range(i * dimPmin, i * dimPmax) for i in range(num_components))))
        else:
            indices = list(range(dimPmin, dimPmax))
        restricted_ON = base_ON.take(indices)
        return restricted_ON

    def __repr__(self):
        res = ""
        if self.complete():
            res += "P" + str(self.maxdegree)
        else:
            res = "P" + "(min " + str(self.mindegree) + " max " + str(self.maxdegree) + ")"
        if self.shape:
            res += "^" + "x".join(str(extent) for extent in self.shape)
        return res

    def __mul__(self, x):
        """
        When multiplying a Polynomial Space by a sympy object, you need to multiply with
        the sympy object on the right. This is due to Sympy's implementation of __mul__ not
        passing to this handler as it should.
        """
        if isinstance(x, sp.Symbol):
            return ConstructedPolynomialSpace([x], [self])
        elif isinstance(x, sp.Matrix):
            return ConstructedPolynomialSpace([x], [self])
        else:
            raise TypeError(f'Cannot multiply a PolySpace with {type(x)}')

    __rmul__ = __mul__

    def __add__(self, x):
        return ConstructedPolynomialSpace([1, 1], [self, x])

    def __eq__(self, other):
        """these comparison operators are not quite right - better to use a combining function
        for tensor products TODO"""
        assert isinstance(other, PolynomialSpace)
        max_bool = self.maxdegree == other.maxdegree
        min_bool = self.mindegree == other.mindegree
        shape = self.shape == other.shape
        return max_bool and min_bool and shape

    def __lt__(self, other):
        """these comparison operators are not quite right - better to use a combining function
        for tensor products TODO"""
        assert isinstance(other, PolynomialSpace)
        if self.maxdegree > other.maxdegree:
            return True
        elif self.maxdegree == other.maxdegree:
            return self.mindegree <= other.mindegree
        return False

    def __hash__(self):
        """Hash."""
        return hash((self.shape, self.mindegree, self.maxdegree))

    def restrict(self, mindegree, maxdegree):
        return PolynomialSpace(maxdegree, mindegree=mindegree, shape=self.shape)

    def to_vector(self, shape):
        return PolynomialSpace(self.maxdegree, self.mindegree, shape=shape)

    def _to_dict(self):
        return {"shape": self.shape, "min": self.mindegree, "max": self.maxdegree}

    def dict_id(self):
        return "PolynomialSpace"

    def _from_dict(obj_dict):
        shape = obj_dict["shape"] if "shape" in obj_dict else obj_dict["set_shape"]
        return PolynomialSpace(obj_dict["max"], obj_dict["min"], shape)


class ConstructedPolynomialSpace(PolynomialSpace):
    """
    Sub degree is inherited from the largest of the component spaces,
    super degree is unknown.

    weights can either be 1 or a polynomial in x, where x in R^d
    """
    def __init__(self, weights, spaces):

        self.weights = weights
        self.spaces = spaces

        weight_degrees = [0 if not (isinstance(w, sp.Expr) or isinstance(w, sp.Matrix)) else max_deg_sp_mat(w) for w in self.weights]

        maxdegree = max([space.maxdegree + w_deg for space, w_deg in zip(spaces, weight_degrees)])
        mindegree = min([space.mindegree + w_deg for space, w_deg in zip(spaces, weight_degrees)])

        # A combination is scalar only if every part is. Where more than one part
        # carries a shape they must agree, as the sum lives in a single space.
        shapes = set(s for s in map(weighted_shape, self.weights, self.spaces) if s)
        if len(shapes) > 1:
            raise ValueError(
                "Cannot combine polynomial spaces of differing value shapes: "
                f"{sorted(shapes)}.")
        shape = shapes.pop() if shapes else ()

        super(ConstructedPolynomialSpace, self).__init__(maxdegree, mindegree, shape=shape)

    def __repr__(self):
        return "+".join([str(w) + "*" + str(x) for (w, x) in zip(self.weights, self.spaces)])

    def to_ON_polynomial_set(self, ref_el):
        if not isinstance(ref_el, reference_element.Cell):
            ref_el = ref_el.to_fiat()
        k = max([s.maxdegree for s in self.spaces])
        space_poly_sets = [s.to_ON_polynomial_set(ref_el) for s in self.spaces]
        sd = ref_el.get_spatial_dimension()
        ref_el = cell_to_simplex(ref_el)

        if all([w == 1 for w in self.weights]):
            weighted_sets = space_poly_sets

        # otherwise have to work on this through tabulation

        Q = create_quadrature(ref_el, 2 * (k + 1))
        Qpts, Qwts = Q.get_points(), Q.get_weights()
        weighted_sets = []

        for (space, w) in zip(space_poly_sets, self.weights):
            if not (isinstance(w, sp.Expr) or isinstance(w, sp.Matrix)):
                weighted_sets.append(space)
            else:
                w_deg = max_deg_sp_mat(w)
                Pkpw = ONPolynomialSet(ref_el, space.degree + w_deg, scale="orthonormal")
                vec_Pkpw = ONPolynomialSet(ref_el, space.degree + w_deg, self.shape, scale="orthonormal")

                space_at_Qpts = space.tabulate(Qpts)[(0,) * sd]
                Pkpw_at_Qpts = Pkpw.tabulate(Qpts)[(0,) * sd]

                tabulated_expr = tabulate_sympy(w, Qpts).T
                if tabulated_expr.shape[0] != int(np.prod(self.shape)):
                    raise ValueError(f"Weight {w} has {tabulated_expr.shape[0]} components but the space has value shape {self.shape}.")
                scaled_at_Qpts = space_at_Qpts[:, None, :] * tabulated_expr[None, :, :]
                PkHw_coeffs = np.dot(np.multiply(scaled_at_Qpts, Qwts), Pkpw_at_Qpts.T)
                weighted_sets.append(polynomial_set.PolynomialSet(ref_el,
                                                                  space.degree + w_deg,
                                                                  space.degree + w_deg,
                                                                  vec_Pkpw.get_expansion_set(),
                                                                  PkHw_coeffs))
        combined_sets = weighted_sets[0]
        for i in range(1, len(weighted_sets)):
            combined_sets = polynomial_set.polynomial_set_union_normalized(combined_sets, weighted_sets[i])
        return combined_sets

    def __mul__(self, x):
        return ConstructedPolynomialSpace([x*w for w in self.weights],
                                          self.spaces)
    __rmul__ = __mul__

    def __add__(self, x):
        w = self.weights.copy()
        w.extend([1])
        s = self.spaces.copy()
        s.extend([x])
        return ConstructedPolynomialSpace(w, s)

    def _to_dict(self):
        super_dict = super(ConstructedPolynomialSpace, self)._to_dict()
        super_dict["spaces"] = self.spaces
        super_dict["weights"] = self.weights
        return super_dict

    def dict_id(self):
        return "ConstructedPolynomialSpace"

    def _from_dict(obj_dict):
        return ConstructedPolynomialSpace(obj_dict["weights"], obj_dict["spaces"])


P0 = PolynomialSpace(0)
P1 = PolynomialSpace(1)
P2 = PolynomialSpace(2)
P3 = PolynomialSpace(3)
P4 = PolynomialSpace(4)
