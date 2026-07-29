from FIAT.polynomial_set import ONPolynomialSet
from FIAT.expansions import morton_index2, morton_index3
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import cell_to_simplex
from FIAT import expansions, polynomial_set, reference_element
from itertools import chain
from fuse.utils import tabulate_sympy, max_deg_sp_expr
import sympy as sp
import numpy as np
from functools import total_ordering

morton_index = {2: morton_index2, 3: morton_index3}


def normalise_shape(shape):
    """Canonicalise a declared value shape to a tuple of positive ints.

    Scalar is the empty tuple.
    """
    if shape is True:
        raise ValueError("shape=True is no longer supported: the value shape must be given explicitly.")
    if isinstance(shape, (int, np.integer)):
        shape = () if shape == 0 else (shape,)
    else:
        try:
            shape = tuple(shape)
        except TypeError:
            raise ValueError(f"Value shape {shape!r} must be an integer or a sequence of integers.")
    for extent in shape:
        if not isinstance(extent, (int, np.integer)) or extent < 1:
            raise ValueError(f"Value shape {shape} must contain positive integers only.")
    return tuple(int(extent) for extent in shape)


def weighted_shape(weight, space):
    """The value shape contributed by one weighted space of a combination.

    A matrix weight is what gives a scalar space its components, so it sets the
    shape; any other weight leaves the space's own shape alone.
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
    contains: the degree of the maximum degree Lagrange space that is spanned by this element. If this
    element's polynomial space does not include the constant function, this function should
    return -1.

    maxdegree: the degree of the minimum degree Lagrange space that spans this element.If this
    element contains basis functions that are not in any Lagrange space, this property should
    be None.

    mindegree: the degree of the polynomial in the space with the lowest degree.

    shape: the value shape of the space, as a tuple. The empty tuple is scalar valued. This is
    the shape of the value, not of the cell, so it is independent of the spatial dimension.

    Note that on a simplex cells, the polynomial space of Lagrange space is a complete polynomial
    space, but on other cells this is not true. For example, on quadrilateral cells, the degree 1
    Lagrange space includes the degree 2 polynomial xy.
    """

    def __init__(self, maxdegree, contains=None, mindegree=0, shape=()):
        self.maxdegree = maxdegree
        self.mindegree = mindegree

        if not contains and mindegree == 0:
            self.contains = maxdegree
        elif not contains and mindegree >= 0:
            self.contains = -1
        else:
            self.contains = contains

        self.shape = normalise_shape(shape)

    def complete(self):
        return self.mindegree == self.maxdegree

    def degree(self):
        return self.maxdegree

    def to_ON_polynomial_set(self, ref_el, k=None):
        if not isinstance(ref_el, reference_element.Cell):
            ref_el = ref_el.to_fiat()
        ref_el = cell_to_simplex(ref_el)
        base_ON = ONPolynomialSet(ref_el, self.maxdegree, self.shape, scale="orthonormal")
        indices = None

        if self.mindegree > 0:
            dimPmin = expansions.polynomial_dimension(ref_el, self.mindegree)
            dimPmax = expansions.polynomial_dimension(ref_el, self.maxdegree)
            if self.shape:
                num_components = int(np.prod(self.shape))
                indices = list(chain(*(range(i * dimPmin, i * dimPmax) for i in range(num_components))))
            else:
                indices = list(range(dimPmin, dimPmax))

        if self.contains != self.maxdegree and self.contains != -1:
            indices = [morton_index[sd](p, q) for p in range(self.contains + 1) for q in range(self.contains + 1)]

        if indices is None:
            return base_ON

        restricted_ON = base_ON.take(indices)
        return restricted_ON

    def __repr__(self):
        res = ""
        if self.complete():
            res += "P" + str(self.maxdegree)
        elif self.mindegree > 0:
            res = "P" + "(min " + str(self.mindegree) + " max " + str(self.maxdegree) + ")"
        else:
            res += "Psub" + str(self.contains) + "sup" + str(self.maxdegree)
        if self.shape:
            res += "^" + "x".join(str(extent) for extent in self.shape)
        return res

    def __mul__(self, x):
        """
        When multiplying a Polynomial Space by a sympy object, you need to multiply with
        the sympy object on the right. This is due to Sympy's implementation of __mul__ not
        passing to this handler as it should.
        """
        if isinstance(x, sp.Symbol) or isinstance(x, sp.Expr) or isinstance(x, sp.Matrix):
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
        contains = self.contains == other.contains
        shape = self.shape == other.shape
        return max_bool and min_bool and contains and shape

    def __lt__(self, other):
        """these comparison operators are not quite right - better to use a combining function
        for tensor products TODO"""
        assert isinstance(other, PolynomialSpace)
        if self.maxdegree > other.maxdegree:
            return True
        elif self.maxdegree == other.maxdegree:
            return self.contains >= other.contains
        return False

    def __hash__(self):
        """Hash."""
        return hash((self.shape, self.mindegree, self.contains, self.maxdegree))

    def restrict(self, mindegree, maxdegree):
        return PolynomialSpace(maxdegree, contains=-1, mindegree=mindegree, shape=self.shape)

    def to_vector(self, shape):
        return PolynomialSpace(self.maxdegree, self.contains, self.mindegree, shape=shape)

    def _to_dict(self):
        return {"shape": self.shape, "min": self.mindegree, "contains": self.contains, "max": self.maxdegree}

    def dict_id(self):
        return "PolynomialSpace"

    def _from_dict(obj_dict):
        return PolynomialSpace(obj_dict["max"], obj_dict["contains"], obj_dict["min"], obj_dict["shape"])


class ConstructedPolynomialSpace(PolynomialSpace):
    """
    Sub degree is inherited from the largest of the component spaces,
    super degree is unknown.

    weights can either be 1 or a polynomial in x, where x in R^d
    """
    def __init__(self, weights, spaces):

        self.weights = weights
        self.spaces = spaces

        weight_degrees = [0 if not (isinstance(w, sp.Expr) or isinstance(w, sp.Matrix)) else max_deg_sp_expr(w) for w in self.weights]

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

        super(ConstructedPolynomialSpace, self).__init__(maxdegree, -1, mindegree, shape=shape)

    def __repr__(self):
        return "+".join([str(w) + "*" + str(x) for (w, x) in zip(self.weights, self.spaces)])

    def to_ON_polynomial_set(self, ref_el):
        if not isinstance(ref_el, reference_element.Cell):
            ref_el = ref_el.to_fiat()
        k = max([s.maxdegree for s in self.spaces])
        sd = ref_el.get_spatial_dimension()
        ref_el = cell_to_simplex(ref_el)

        # otherwise have to work on this through tabulation

        weighted_sets = []

        for (s, w) in zip(self.spaces, self.weights):
            space = s.to_ON_polynomial_set(ref_el)
            if not (isinstance(w, sp.Expr) or isinstance(w, sp.Matrix)):
                weighted_sets.append(space)
            else:
                if isinstance(w, sp.Expr):
                    w = sp.Matrix([[w]])
                    vec = False
                else:
                    vec = True
                w_deg = max_deg_sp_expr(w)
                Q = create_quadrature(ref_el, 2 * (k + w_deg + 1))
                Qpts, Qwts = Q.get_points(), Q.get_weights()
                Pkpw = ONPolynomialSet(ref_el, space.degree + w_deg, s.shape, scale="orthonormal")

                space_at_Qpts = space.tabulate(Qpts)[(0,) * sd]
                Pkpw_at_Qpts = Pkpw.tabulate(Qpts)[(0,) * sd]

                tabulated_expr = tabulate_sympy(w, Qpts).T

                if tabulated_expr.shape[0] != int(np.prod(self.shape)):
                    raise ValueError(f"Weight {w} has {tabulated_expr.shape[0]} components but the space has value shape {self.shape}.")

                scaled_at_Qpts = space_at_Qpts[:, None, :] * tabulated_expr[None, :, :]
                if not (vec and len(s.shape) > 0):
                    scaled_at_Qpts = scaled_at_Qpts.squeeze()
                PkHw_coeffs = np.dot(np.multiply(scaled_at_Qpts, Qwts), Pkpw_at_Qpts.T)
                if len(PkHw_coeffs.shape) == 1:
                    PkHw_coeffs = PkHw_coeffs.reshape(1, -1)
                weighted_sets.append(polynomial_set.PolynomialSet(ref_el,
                                                                  space.degree + w_deg,
                                                                  space.degree + w_deg,
                                                                  Pkpw.get_expansion_set(),
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

    def to_vector(self):
        return ConstructedPolynomialSpace(self.weights, [space.to_vector() for space in self.spaces])

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

Q1 = PolynomialSpace(1, 2)
Q2 = PolynomialSpace(2, 3)
Q3 = PolynomialSpace(3, 4)
Q4 = PolynomialSpace(4, 5)
