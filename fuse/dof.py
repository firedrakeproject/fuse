from FIAT.functional import Functional
from fuse.utils import sympy_to_numpy
from fuse.traces import TrH1, TrHCurl, TrHDiv
from fuse.actions import as_action, H1Action, HCurlAction, HDivAction
import copy
import warnings
import numpy as np
import sympy as sp


class Pairing():
    """
    The pairing combines a kernel and an input function.
    In many cases, this takes the form of an inner product.
    """

    def __init__(self):
        self.entity = None

    def _to_dict(self):
        o_dict = {"entity": self.entity}
        return o_dict


class DeltaPairing(Pairing):
    """
    The delta pairing allows evaluation at point

    :param: kernel: Normally a PointKernel
    """

    def __init__(self):
        super(DeltaPairing, self).__init__()

    def __call__(self, kernel, v, cell):
        assert isinstance(kernel, PointKernel)
        return v(*kernel.pt)

    def tabulate(self):
        return 1

    def add_entity(self, entity):
        res = DeltaPairing()
        res.entity = entity
        return res

    def __repr__(self):
        return self.describe()

    def describe(self, orientation=None):
        return "{fn}({kernel})"

    def dict_id(self):
        return "Delta"

    def _from_dict(obj_dict):
        new_obj = DeltaPairing()
        new_obj.add_entity(obj_dict["entity"])
        return new_obj


class L2Pairing(Pairing):
    """ The L2 inner product as a pairing """

    def __init__(self):
        super(L2Pairing, self).__init__()

    def __call__(self, kernel, v, cell):
        Qpts, Qwts = self.entity.quadrature(5)
        return sum([wt*np.dot(kernel(*pt), v(*pt)) for pt, wt in zip(Qpts, Qwts)])

    def add_entity(self, entity):
        res = L2Pairing()
        res.entity = entity
        return res

    def __repr__(self):
        return self.describe()

    def describe(self, orientation=None):
        entity = str(self.entity) if orientation is None else "{}({})".format(str(orientation), str(self.entity))
        return "integral_{}({{kernel}} * {{fn}}) dx) ".format(entity)

    def dict_id(self):
        return "L2Inner"

    def _from_dict(obj_dict):
        new_obj = L2Pairing()
        new_obj.add_entity(obj_dict["entity"])
        return new_obj


class BaseKernel():
    """A DOF kernel.

    A kernel must be associated with an ``action``, an instance of
    :class:`fuse.actions.KernelAction`, that describes the transformation of
    the kernel value under symmetry.

    This class also accepts the shorthand H1, HCurl, HDiv or L2 for a form action.
    Scalar kernels default to H1.
    """

    def __init__(self, action=None):
        self.attachment = False
        self.orientation = None
        self._action = as_action(action)
        if self._action is None and self.rank == 1:
            warnings.warn(f"Vector valued kernel {self} has no action: assuming HCurl. "
                          "Pass action=HCurl (or another action) explicitly.", DeprecationWarning, stacklevel=3)

    @property
    def rank(self):
        """0 for scalar valued kernels, 1 for vector valued ones."""
        return 0

    @property
    def declared_action(self):
        return self._action

    @property
    def action(self):
        if self._action is not None:
            return self._action
        return HCurlAction if self.rank == 1 else H1Action

    def with_action(self, action):
        """A copy of this kernel with ``action``."""
        res = copy.copy(self)
        res._action = as_action(action)
        return res

    def action_for_trace(self, trace):
        """
        This function verifies the given trace matches the action.

        To be removed when we no longer support specifying the trace type.

        A kernel that declares no action takes
        it from the trace (deprecated, see :func:`fuse.triples.immerse`).
        """
        expected = {TrHCurl: HCurlAction, TrHDiv: HDivAction}.get(type(trace))
        if expected is None:
            return self
        if self.declared_action is None:
            return self.with_action(expected)
        if not expected.matches(self.declared_action):
            raise ValueError(f"Kernel {self} has action {self.declared_action} but is immersed with {trace}")
        return self

    def act(self, g):
        """ Transform the kernel by the symmetry ``g`` of the associated entity"""
        res = copy.copy(self)
        res.orientation = g if self.orientation is None else g.compose(self.orientation)
        return res

    def permute(self, g):
        return self.act(g)

    def _check_serialisable(self):
        """Serialisation does not keep a kernel's orientation yet, so refuse
        to encode an acted on kernel rather than silently lose it."""
        if self.orientation is not None:
            raise NotImplementedError(f"Kernel {self} has been acted on by {self.orientation}, and serialising "
                                      "its orientation is not supported yet. Construct the kernel explicitly instead.")

    def action_matrix(self, entity):
        """
        Represents the inverse transform of the entity's basis under a symmetry.

        This is the inverse transpose of the linear part of the kernel's symmetry, in the
        coordinates of ``entity``."""
        if entity.get_spatial_dimension() == 0:
            return np.eye(0)
        bvs = np.array(entity.basis_vectors())
        oriented = entity.__dict__.setdefault("_oriented_basis_vectors", {})
        key = None if self.orientation is None else self.orientation.perm
        if key not in oriented:
            oriented[key] = np.array(entity.orient(self.orientation).basis_vectors())
        return np.matmul(np.linalg.inv(oriented[key]), bvs)

    def __repr__(self):
        return "BaseKernel"

    def __call__(self, *args):
        raise NotImplementedError("This method should be implemented by the subclass")


class PointKernel(BaseKernel):

    def __init__(self, x, action=None):
        if not isinstance(x, tuple):
            x = (x,)
        self.pt = x
        super(PointKernel, self).__init__(action)

    def __repr__(self):
        x = list(map(str, list(self.pt)))
        return ','.join(x)

    def degree(self, interpolant_degree):
        return interpolant_degree

    def act(self, g):
        res = copy.copy(self)
        res.pt = g(self.pt)
        return res

    def __call__(self, *args):
        return self.pt

    def evaluate(self, Qpts, Qwts, basis_change, action, dim, value_shape):
        return np.array([self.pt for _ in Qpts]).astype(np.float64), np.ones_like(Qwts), [[tuple()] for pt in Qpts]

    def _to_dict(self):
        self._check_serialisable()
        o_dict = {"pt": self.pt, "action": self._action}
        return o_dict

    def dict_id(self):
        return "PointKernel"

    def _from_dict(obj_dict):
        return PointKernel(tuple(obj_dict["pt"]), action=obj_dict.get("action"))


class VectorKernel(BaseKernel):

    def __init__(self, x, g=None, action=None):
        self.pt = x
        self.g = g
        super(VectorKernel, self).__init__(action)

    @property
    def rank(self):
        return 0 if np.ndim(self.pt) == 0 else 1

    def __repr__(self):
        if isinstance(self.pt, tuple):
            x = list(map(str, list(self.pt)))
        else:
            x = [str(self.pt)]
        return ','.join(x)

    def degree(self, interpolant_degree):
        return interpolant_degree

    def __call__(self, *args):
        return self.pt

    def evaluate(self, Qpts, Qwts, basis_change, action, dim, value_shape):
        comps = list(np.ndindex(value_shape)) if len(value_shape) else [tuple()]
        value = action.act(basis_change, np.array(self.pt, dtype=np.float64))
        return Qpts, np.array([wt*value for wt in Qwts]).astype(np.float64), [comps for pt in Qpts]

    def _to_dict(self):
        self._check_serialisable()
        o_dict = {"pt": self.pt, "action": self._action}
        return o_dict

    def dict_id(self):
        return "VectorKernel"

    def _from_dict(obj_dict):
        return VectorKernel(tuple(obj_dict["pt"]), action=obj_dict.get("action"))


def evaluate_polynomial(kernel, args):
    """The value of a polynomial kernel at values ``args``.

    The polynomial's symbols are replaced with the values of args.

    Constructs a lambda function version of the polynomial on first run,
    to avoid repeated sympy substitutions.

    If the values in args are symbolic, falls back to sympy.
    """
    args = args[:len(kernel.syms)]
    if "_numeric" not in kernel.__dict__:
        exprs = kernel.fn.as_expr() if kernel.shape == 0 else [f.as_expr() for f in kernel.fn]
        free = set().union(*[sp.sympify(e).free_symbols for e in np.atleast_1d(exprs)])
        kernel._numeric = sp.lambdify(kernel.syms, exprs, "numpy") if free <= set(kernel.syms) else None
    if kernel._numeric is None or any(isinstance(a, sp.Basic) for a in args):
        if kernel.shape == 0:
            return sympy_to_numpy(kernel.fn, kernel.syms, args)
        return [sympy_to_numpy(f, kernel.syms, args) for f in kernel.fn]
    return np.array(kernel._numeric(*args), dtype=np.float64)


class BarycentricPolynomialKernel(BaseKernel):

    def __init__(self, fn, g=None, symbols=[], action=None):
        if hasattr(fn, "__iter__") or isinstance(fn, sp.Matrix):
            # if len(symbols) != 0 and any(not sp.sympify(fn[i]).as_poly() for i in range(len(fn))):
            #     raise ValueError("Function components must be able to be interpreted as a sympy polynomial")
            self.fn = [sp.Poly(fn[i], symbols) for i in range(len(fn))]
            self.shape = len(fn)
        else:
            # if len(symbols) != 0 and not sp.sympify(fn).as_poly():
            #     raise ValueError("Function must be able to be interpreted as a sympy polynomial")
            # self.fn = sp.sympify(fn)
            self.fn = sp.Poly(fn, symbols)
            self.shape = 0
        self.g = g
        self.syms = symbols
        super(BarycentricPolynomialKernel, self).__init__(action)

    @property
    def rank(self):
        return 0 if self.shape == 0 else 1

    def __repr__(self):
        return str(self.fn)

    def degree(self, interpolant_degree):
        if self.shape != 0:
            return max([self.fn[i].as_poly().total_degree() for i in range(self.shape)]) + interpolant_degree
        if len(self.fn.free_symbols) == 0:  # this should probably be removed
            return interpolant_degree
        return self.fn.as_poly().total_degree() + interpolant_degree

    def __call__(self, *args):
        return evaluate_polynomial(self, args)

    def evaluate(self, Qpts, bary_pts, Qwts, basis_change, action, dim, value_shape):
        if len(value_shape) == 0:
            comps = [[tuple()] for pt in Qpts]
        else:
            comps = [list(np.ndindex(value_shape)) for pt in Qpts]
        wts = [wt*action.act(basis_change, self(*pt)) for pt, wt in zip(bary_pts, Qwts)]
        return Qpts, np.array(wts).astype(np.float64), comps

    def _to_dict(self):
        self._check_serialisable()
        o_dict = {"fn": self.fn, "syms": self.syms, "action": self._action}
        return o_dict

    def dict_id(self):
        return "BarycentricPolynomialKernel"

    def _from_dict(obj_dict):
        return BarycentricPolynomialKernel(obj_dict["fn"], symbols=obj_dict["syms"], action=obj_dict.get("action"))


class PolynomialKernel(BaseKernel):

    def __init__(self, fn, g=None, symbols=[], action=None):
        if hasattr(fn, "__iter__"):
            if len(symbols) != 0 and any(not sp.sympify(fn[i]).as_poly() for i in range(len(fn))):
                raise ValueError("Function components must be able to be interpreted as a sympy polynomial")
            self.fn = [sp.sympify(fn[i]).as_poly() for i in range(len(fn))]
            self.shape = len(fn)
        else:
            self.fn = sp.sympify(fn)
            self.shape = 0
        self.g = g
        self.syms = symbols
        super(PolynomialKernel, self).__init__(action)

    @property
    def rank(self):
        return 0 if self.shape == 0 else 1

    def __repr__(self):
        return str(self.fn)

    def degree(self, interpolant_degree):
        if self.shape != 0:
            return max([self.fn[i].as_poly().total_degree() for i in range(self.shape)]) + interpolant_degree
        if len(self.fn.free_symbols) == 0:  # this should probably be removed
            return interpolant_degree
        return self.fn.as_poly().total_degree() + interpolant_degree

    def __call__(self, *args):
        return evaluate_polynomial(self, args)

    def evaluate(self, Qpts, Qwts, basis_change, action, dim, value_shape):
        if len(value_shape) == 0:
            comps = [[tuple()] for pt in Qpts]
        else:
            comps = [list(np.ndindex(value_shape)) for pt in Qpts]
        wts = [wt*action.act(basis_change, self(*np.matmul(basis_change.T, pt))) for pt, wt in zip(Qpts, Qwts)]
        return Qpts, np.array(wts).astype(np.float64), comps

    def _to_dict(self):
        self._check_serialisable()
        o_dict = {"fn": self.fn, "action": self._action}
        return o_dict

    def dict_id(self):
        return "PolynomialKernel"

    def _from_dict(obj_dict):
        return PolynomialKernel(obj_dict["fn"], action=obj_dict.get("action"))


class ComponentKernel(BaseKernel):
    """Selects a component of the value of the function the DOF acts on.

    Used bare, this is a moment against a single component.
    A kernel passed as ``base_kernel`` is restricted to the given component,
    e.g. a scalar functional can be lifted to act on one component of a
    vector valued function.

    Only ``PointKernel`` and scalar ``PolynomialKernel`` may be wrapped.
    """

    def __init__(self, comp, base_kernel=None):
        self.comp = tuple(comp)
        # The permitted types are dictated by the isinstance dispatches in
        # DOF.to_quadrature, which cannot see through this wrapper.
        if base_kernel is not None:
            if isinstance(base_kernel, PointKernel):
                pass
            elif isinstance(base_kernel, PolynomialKernel) and base_kernel.shape == 0:
                pass
            else:
                raise NotImplementedError(
                    f"Cannot wrap {type(base_kernel).__name__} in a ComponentKernel. ")
        self.base_kernel = base_kernel
        super(ComponentKernel, self).__init__()

    def __repr__(self):
        if self.base_kernel is None:
            return f"[{self.comp}]"
        return f"{self.base_kernel}[{self.comp}]"

    def degree(self, interpolant_degree):
        if self.base_kernel is None:
            return interpolant_degree
        return self.base_kernel.degree(interpolant_degree)

    @property
    def rank(self):
        return 0 if self.base_kernel is None else self.base_kernel.rank

    def act(self, g):
        if self.base_kernel is None:
            return self
        # The component index is unchanged by a cell symmetry, as each
        # component transforms as a scalar under the identity pullback.
        return ComponentKernel(self.comp, self.base_kernel.act(g))

    def action_matrix(self, entity):
        if self.base_kernel is None:
            return super(ComponentKernel, self).action_matrix(entity)
        return self.base_kernel.action_matrix(entity)

    @property
    def declared_action(self):
        return self._action if self.base_kernel is None else self.base_kernel.declared_action

    @property
    def action(self):
        return super(ComponentKernel, self).action if self.base_kernel is None else self.base_kernel.action

    def with_action(self, action):
        if self.base_kernel is None:
            return super(ComponentKernel, self).with_action(action)
        return ComponentKernel(self.comp, self.base_kernel.with_action(action))

    def __call__(self, *args):
        if self.base_kernel is None:
            return tuple(args[i] if i in self.comp else 0 for i in range(len(args)))
        return self.base_kernel(*args)

    def _shift(self, comp):
        """Select ``self.comp`` of a scalar base, or offset a shaped base by it."""
        if len(comp) == 0:
            return self.comp
        if len(comp) != len(self.comp):
            raise ValueError(f"Cannot offset a component of rank {len(comp)} by one of rank {len(self.comp)}.")
        return tuple(a + b for a, b in zip(self.comp, comp))

    def evaluate(self, Qpts, Qwts, basis_change, action, dim, value_shape):
        if self.base_kernel is None:
            return Qpts, Qwts, [[self.comp] for pt in Qpts]
        pts, wts, comps = self.base_kernel.evaluate(Qpts, Qwts, basis_change,
                                                    action, dim, tuple())
        return pts, wts, [[self._shift(c) for c in cs] for cs in comps]

    def _to_dict(self):
        o_dict = {"comp": self.comp, "base_kernel": self.base_kernel}
        return o_dict

    def dict_id(self):
        return "ComponentKernel"

    def _from_dict(obj_dict):
        return ComponentKernel(tuple(obj_dict["comp"]), obj_dict.get("base_kernel"))


class DOF():

    def __init__(self, pairing, kernel, entity=None, attachment=None, target_space=None, g=None, immersed=False, generation=None, sub_id=None, cell=None, entity_o=False):
        self.pairing = pairing
        self.kernel = kernel
        self.immersed = immersed
        self.cell_defined_on = entity
        self.attachment = attachment
        # an explicit trace for the derivative immersions (TrGrad, TrHess);
        # every other DOF is traced through its kernel's action
        self.target_space = target_space
        self.g = g
        self.id = None
        self.sub_id = sub_id
        self.cell = cell
        self.entity_o = entity_o

        if generation is None:
            self.generation = {}
        else:
            self.generation = generation
        if entity is not None:
            self.pairing = self.pairing.add_entity(entity)

    def __call__(self, g, entity_o=False):
        new_generation = self.generation.copy()
        return DOF(self.pairing, self.kernel.act(g), self.cell_defined_on, self.attachment, self.target_space, g, self.immersed, new_generation, self.sub_id, self.cell, entity_o)

    def lifted_action(self):
        """The kernel's action, resolved on the entity the DOF is defined on and,
        for an immersed DOF, lifted into the cell."""
        action = self.kernel.action.resolve(self.cell_defined_on, self.kernel.rank)
        return action.lifted() if self.immersed else action

    def eval(self, fn, pullback=True):
        return self.pairing(self.kernel, fn, self.cell)

    def tabulate(self, Qpts):
        return self.kernel.tabulate(Qpts)

    def add_context(self, dof_gen, cell, space, g, overall_id=None, generator_id=None):
        # For some of these, we only want to store the first instance of each
        self.generation[cell.dim()] = dof_gen
        self.cell = cell
        if self.cell_defined_on is None:
            self.trace_entity = cell
            self.cell_defined_on = cell
            self.pairing = self.pairing.add_entity(cell)
        if self.id is None and overall_id is not None:
            self.id = overall_id
        if self.sub_id is None and generator_id is not None:
            self.sub_id = generator_id

    def convert_to_fiat(self, ref_el, interpolant_degree, value_shape=tuple()):
        # TODO deriv dict needs implementing (currently {})
        return Functional(ref_el, value_shape, self.to_quadrature(interpolant_degree, value_shape), {}, str(self))

    def to_quadrature(self, arg_degree, value_shape):
        entity = self.cell_defined_on
        Qpts, Qwts = entity.quadrature(self.kernel.degree(arg_degree))
        Qwts = Qwts.reshape(Qwts.shape + (1,))
        basis_change = self.kernel.action_matrix(entity)
        action = self.kernel.action.resolve(entity, self.kernel.rank)

        if isinstance(self.kernel, BarycentricPolynomialKernel):
            pts = [np.matmul(basis_change.T, pt) for pt in Qpts]
            bary_pts = entity.cartesian_to_barycentric(pts)
            pts, wts, comps = self.kernel.evaluate(Qpts, bary_pts, Qwts, basis_change, action, self.cell.dimension, value_shape)
        else:
            pts, wts, comps = self.kernel.evaluate(Qpts, Qwts, basis_change, action, self.cell.dimension, value_shape)

        if self.immersed:
            pts = np.array([self.cell.attachment(self.cell.id, entity.id)(*pt) for pt in pts])
            if self.target_space is not None:
                # derivative traces are not yet kernel actions
                wts = np.outer(wts, self.target_space.tabulate(pts, entity))
            elif not isinstance(self.kernel, (PointKernel, ComponentKernel)):
                L = action.lift_matrix(self.cell, entity, self.kernel.rank)
                if L is not None:
                    wts = np.matmul(np.reshape(wts, (len(wts), -1)), L.T)
        # pt dict is { pt: [(weight, component)]}
        if len(comps) > 0 and len(wts) > 0 and len(wts[0]) != len(comps[0]):
            raise ValueError(f"{self} produced {len(wts[0])} weights for {len(comps[0])} components. The kernel and the value shape disagree.")
        pt_dict = {tuple(pt): [(w, c) for w, c in zip(wt, cp)] for pt, wt, cp in zip(pts, wts, comps)}
        return pt_dict

    def __repr__(self, fn="v"):
        return self.pairing.describe(self.kernel.orientation).format(fn=fn, kernel=self.kernel)

    def immerse(self, entity, attachment, target_space, g, triple):
        """This DOF immersed on ``entity``. ``target_space`` is an explicit
        trace (deprecated for TrH1, TrHCurl and TrHDiv, which are checked
        against the kernel's action), or None to trace through the action."""
        new_generation = self.generation.copy()
        kernel = self.kernel
        if isinstance(target_space, (TrH1, TrHCurl, TrHDiv)):
            kernel = kernel.action_for_trace(target_space)
            target_space = None
        return ImmersedDOF(self.pairing, kernel, entity, attachment, target_space, g, triple, new_generation, self.sub_id, self.cell)

    def lift_direction(self):
        """The direction an immersed DOF's kernel is lifted along: the tangent
        of a flat form on an edge or the normal of a hodge form, or None."""
        if not self.immersed or isinstance(self.kernel, (PointKernel, ComponentKernel)):
            return None
        entity, rank = self.cell_defined_on, self.kernel.rank
        L = self.kernel.action.resolve(entity, rank).lift_matrix(self.cell, entity, rank)
        if L is None or L.shape[1] != 1:
            return None
        return L[:, 0]

    def with_kernel(self, kernel):
        """A copy of this DOF acting through a different kernel.

        The generation dict is copied but its generators are shared, so the
        copy is grouped with the original by ``_entity_associations``.
        """
        return DOF(self.pairing, kernel, self.cell_defined_on, self.attachment, self.target_space, self.g, self.immersed, self.generation.copy(), self.sub_id, self.cell, self.entity_o)

    def _to_dict(self):
        """ almost certainly needs more things"""
        o_dict = {"pairing": self.pairing, "kernel": self.kernel}
        return o_dict

    def dict_id(self):
        return "DOF"

    def _from_dict(obj_dict):
        return DOF(obj_dict["pairing"], obj_dict["kernel"])


class ImmersedDOF(DOF):
    # probably need to add a convert to fiat method here to capture derivatives from immersion
    def __init__(self, pairing, kernel, entity=None, attachment=None, target_space=None, g=None, triple=None, generation=None, sub_id=None, cell=None, entity_o=False):
        self.immersed = True
        self.triple = triple
        super(ImmersedDOF, self).__init__(pairing, kernel, entity=entity, attachment=attachment, target_space=target_space, g=g, immersed=True, generation=generation, sub_id=sub_id, cell=cell, entity_o=entity_o)

    def eval(self, fn, pullback=True):
        attached_fn = fn.attach(self.attachment)

        if pullback:
            attached_fn = self.trace(attached_fn)

        return self.pairing(self.kernel, attached_fn, self.cell)

    def trace(self, fn):
        """The trace of ``fn`` on the entity, the dual of the kernel's lift."""
        if self.target_space is not None:
            # derivative traces are not yet kernel actions
            return self.target_space(fn, self.cell_defined_on)
        if isinstance(self.kernel, (PointKernel, ComponentKernel)):
            return fn
        entity, rank = self.cell_defined_on, self.kernel.rank
        action = self.kernel.action.resolve(entity, rank)
        if action.lift_matrix(self.cell, entity, rank) is None:
            return fn

        def restricted(*x):
            return tuple(action.trace(self.cell, entity, rank, np.array(fn(*x), dtype=float).squeeze()))
        return restricted

    def tabulate(self, Qpts):
        # modify this to take reference space q pts
        res, _ = self.kernel.tabulate(Qpts, self.attachment)
        if self.target_space is None:
            return res
        return self.target_space.tabulate(Qpts, self.pairing.entity)*res

    def __call__(self, g, entity_o=False):
        index_trace = self.cell.d_entities_ids(self.cell_defined_on.dim()).index(self.cell_defined_on.id)
        permuted_e, permuted_g = self.cell.permute_entities(g, self.cell_defined_on.dim())[index_trace]
        new_cell_defined_on = self.cell.get_node(permuted_e)

        new_attach = lambda *x: g(self.attachment(*x))
        return ImmersedDOF(self.pairing, self.kernel.act(permuted_g), new_cell_defined_on,
                           new_attach, self.target_space, g, self.triple, self.generation, self.sub_id, self.cell, entity_o)

    def __repr__(self):
        trace = self.target_space
        if trace is None:
            trace = self.kernel.action.resolve(self.cell_defined_on, self.kernel.rank)
        fn = "tr_{1}_{0}(v)".format(str(self.cell_defined_on), str(trace))
        return super(ImmersedDOF, self).__repr__(fn)

    def immerse(self, entity, attachment, trace, g):
        raise RuntimeError("Error: Immersing twice not supported")

    def with_kernel(self, kernel):
        return ImmersedDOF(self.pairing, kernel, self.cell_defined_on, self.attachment, self.target_space, self.g, self.triple, self.generation.copy(), self.sub_id, self.cell, self.entity_o)


class FuseFunction():

    def __init__(self, eq, attach_func=None, symbols=None):
        self.eq = eq
        self.attach_func = attach_func
        self.symbols = symbols

    def __call__(self, *x, sym=False):
        if self.symbols:
            if self.attach_func and not sym:
                res = self.eq.xreplace({symb: val for (symb, val) in zip(self.symbols, self.attach_func(*x))})
            else:
                res = self.eq.xreplace({symb: val for (symb, val) in zip(self.symbols, x)})
            if res.free_symbols == set():
                array = np.array(res).astype(np.float64)
                return array
            else:
                return res
        if self.attach_func and not sym:
            return self.eq(*self.attach_func(*x))
        else:
            # TODO remove this as will already be symbolic
            return self.eq(*x)

    def attach(self, attachment):
        if not self.attach_func:
            return FuseFunction(self.eq, attach_func=attachment, symbols=self.symbols)
        else:
            old_attach = self.attach_func
            if self.symbols:
                return FuseFunction(self.eq,
                                    attach_func=attachment(old_attach(*self.symbols)),
                                    symbols=self.symbols)
            else:
                return FuseFunction(self.eq,
                                    attach_func=lambda *x: attachment(old_attach(*x)))

    def __repr__(self):
        if self.attach_func:
            return "v(G(x))"
        else:
            return "v(x)"

    def _to_dict(self):
        return {"eq": self.eq}

    def dict_id(self):
        return "Function"
