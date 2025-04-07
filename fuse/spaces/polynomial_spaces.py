from FIAT.polynomial_set import ONPolynomialSet
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import cell_to_simplex
from FIAT import expansions, polynomial_set, reference_element
from itertools import chain
from fuse.utils import tabulate_sympy, max_deg_sp_expr
import sympy as sp
import numpy as np
from functools import total_ordering


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

    Note that on a simplex cells, the polynomial space of Lagrange space is a complete polynomial
    space, but on other cells this is not true. For example, on quadrilateral cells, the degree 1
    Lagrange space includes the degree 2 polynomial xy.
    """

    def __init__(self, maxdegree, contains=None, mindegree=0, set_shape=False):
        self.maxdegree = maxdegree
        self.mindegree = mindegree

        if not contains and mindegree == 0:
            self.contains = maxdegree
        elif not contains and mindegree >= 0:
            self.contains = -1
        else:
            self.contains = contains

        self.set_shape = set_shape

    def complete(self):
        return self.mindegree == self.maxdegree

    def degree(self):
        return self.maxdegree

    def to_ON_polynomial_set(self, ref_el, k=None):
        # how does super/sub degrees work here
        if not isinstance(ref_el, reference_element.Cell):
            ref_el = ref_el.to_fiat()
        sd = ref_el.get_spatial_dimension()
        ref_el = cell_to_simplex(ref_el)
        if self.set_shape:
            shape = (sd,)
        else:
            shape = tuple()
        base_ON = ONPolynomialSet(ref_el, self.maxdegree, shape, scale="orthonormal")

        # if self.contains:
        #     expansion_set = expansions.ExpansionSet(ref_el)
        #     num_exp_functions = expansion_set.get_num_members(self.maxdegree)
        #     dimPmin = expansions.polynomial_dimension(ref_el, self.mindegree)
        #     dimPmax = expansions.polynomial_dimension(ref_el, self.maxdegree)
        #     indices = list(chain(*(range(i * dimPmin, i * dimPmax) for i in range(sd))))
        #     from math import comb
        #     print(list(chain(*[range(0, 0), range(2, 3)])))
        # base_ON.expansion_set._tabulate_on_cell(3, np.array([[0,0]]))
        # if self.set_shape:
        #     indices = list(chain(*(range(i * dimPmin, i * dimPmax) for i in range(sd))))
        # else:
        #     indices = list(range(dimPmin, dimPmax))
        # restricted_ON = base_ON.take(indices)
        # return restricted_ON

        # breakpoint()

        if self.mindegree > 0:
            dimPmin = expansions.polynomial_dimension(ref_el, self.mindegree)
            dimPmax = expansions.polynomial_dimension(ref_el, self.maxdegree)
            if self.set_shape:
                indices = list(chain(*(range(i * dimPmin, i * dimPmax) for i in range(sd))))
            else:
                indices = list(range(dimPmin, dimPmax))
            restricted_ON = base_ON.take(indices)
            return restricted_ON

        return base_ON

    def __repr__(self):
        res = ""
        if self.complete():
            res += "P" + str(self.maxdegree)
        elif self.mindegree > 0:
            res = "P" + "(min " + str(self.mindegree) + " max " + str(self.maxdegree) + ")"
        else:
            res += "Psub" + str(self.contains) + "sup" + str(self.maxdegree)
        if self.set_shape:
            res += "^d"
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
        contains = self.contains == other.contains
        shape = self.set_shape == other.set_shape
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
        return hash((self.set_shape, self.mindegree, self.contains, self.maxdegree))

    def restrict(self, mindegree, maxdegree):
        return PolynomialSpace(maxdegree, contains=-1, mindegree=mindegree, set_shape=self.set_shape)

    def _to_dict(self):
        return {"set_shape": self.set_shape, "min": self.mindegree, "contains": self.contains, "max": self.maxdegree}

    def dict_id(self):
        return "PolynomialSpace"

    def _from_dict(obj_dict):
        return PolynomialSpace(obj_dict["max"], obj_dict["contains"], obj_dict["min"], obj_dict["set_shape"])


class ConstructedPolynomialSpace(PolynomialSpace):
    """
    Sub degree is inherited from the largest of the component spaces,
    super degree is unknown.

    weights can either be 1 or a polynomial in x, where x in R^d
    """
    def __init__(self, weights, spaces):

        self.weights = weights
        self.spaces = spaces

        maxdegree = max([space.maxdegree for space in spaces])
        mindegree = min([space.mindegree for space in spaces])
        vec = any([s.set_shape for s in spaces])

        super(ConstructedPolynomialSpace, self).__init__(maxdegree, -1, mindegree, set_shape=vec)

    def __repr__(self):
        return "+".join([str(w) + "*" + str(x) for (w, x) in zip(self.weights, self.spaces)])

    def to_ON_polynomial_set(self, ref_el):
        if not isinstance(ref_el, reference_element.Cell):
            ref_el = ref_el.to_fiat()
        k = max([s.maxdegree for s in self.spaces])
        sd = ref_el.get_spatial_dimension()
        ref_el = cell_to_simplex(ref_el)

        # otherwise have to work on this through tabulation

        Q = create_quadrature(ref_el, 2 * (k + 1))
        Qpts, Qwts = Q.get_points(), Q.get_weights()
        weighted_sets = []

        for (s, w) in zip(self.spaces, self.weights):
            space = s.to_ON_polynomial_set(ref_el)
            if s.set_shape:
                shape = (sd,)
            else:
                shape = tuple()
            if not (isinstance(w, sp.Expr) or isinstance(w, sp.Matrix)):
                weighted_sets.append(space)
            else:
                if isinstance(w, sp.Expr):
                    w = sp.Matrix([[w]])
                    vec = False
                else:
                    vec = True
                w_deg = max_deg_sp_expr(w)
                Pkpw = ONPolynomialSet(ref_el, space.degree + w_deg, shape, scale="orthonormal")
                # vec_Pkpw = ONPolynomialSet(ref_el, space.degree + w_deg, (sd,), scale="orthonormal")

                space_at_Qpts = space.tabulate(Qpts)[(0,) * sd]
                Pkpw_at_Qpts = Pkpw.tabulate(Qpts)[(0,) * sd]

                tabulated_expr = tabulate_sympy(w, Qpts).T
                if s.set_shape or vec:
                    scaled_at_Qpts = space_at_Qpts[:, None, :] * tabulated_expr[None, :, :]
                else:
                    # breakpoint()
                    scaled_at_Qpts = space_at_Qpts[:, None, :] * tabulated_expr[None, :, :]
                    scaled_at_Qpts = scaled_at_Qpts.squeeze()
                PkHw_coeffs = np.dot(np.multiply(scaled_at_Qpts, Qwts), Pkpw_at_Qpts.T)
                # breakpoint()
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
        return ConstructedPolynomialSpace(self.weights.extend([1]),
                                          self.spaces.extend(x))

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
