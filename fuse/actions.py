"""How the value of a DOF kernel transforms.

A kernel has a support (a point, or a function on an entity), which its class
moves, and a value, which its *action* transforms. The action is chosen with
the kernel and can be any :class:`KernelAction`:

* :class:`FormAction` is the de Rham family: the value is a k-form, read as a
  scalar or a vector through a proxy.
* Other actions (for example a symmetric tensor action) are added by
  implementing the same interface.

The spaces ``H1``, ``HCurl``, ``HDiv`` and ``L2`` are accepted wherever an
action is, as shorthands for form actions (see :func:`as_action`).
"""
import numpy as np

from fuse.spaces.interpolation_spaces import InterpolationSpace
from fuse.traces import TrH1, TrHCurl, TrHDiv


class KernelAction():
    """Interface for the action of geometric maps on a kernel's value.

    ``A`` is the action matrix of a symmetry: the inverse transpose of its
    linear part, in the coordinates of the entity the kernel is defined on.
    """

    def resolve(self, entity, rank):
        """This action on a kernel of ``rank`` defined on ``entity``, with
        anything that depends on the entity filled in."""
        return self

    def act(self, A, value):
        """The value transformed by a symmetry with action matrix ``A``."""
        raise NotImplementedError

    def lift_matrix(self, cell, entity, rank):
        """Matrix taking the value on ``entity`` to its value in ``cell`` when
        the DOF is immersed, or None when the value is unchanged."""
        raise NotImplementedError

    def restrict(self, cell, entity, rank, value):
        """The trace: a value in ``cell`` restricted to ``entity``.

        It is the dual of the lift, ``<lift(k), v> = <k, restrict(v)>``.
        """
        L = self.lift_matrix(cell, entity, rank)
        if L is None:
            return value
        return np.matmul(np.transpose(L), np.ravel(value))

    def lifted(self):
        """The action on the value once it is lifted into the cell."""
        raise NotImplementedError

    def trace(self, cell, entity, rank):
        """The trace object (used for plotting and the orientation matrices)."""
        raise NotImplementedError(f"{self} has no trace")

    def matches(self, other):
        """Whether this action is the one ``other`` describes."""
        return self == other

    def dict_id(self):
        return type(self).__name__

    def _to_dict(self):
        return {}


class FormAction(KernelAction):
    """The value is a k-form, identified with a scalar or vector by ``proxy``.

    ``proxy`` is one of

    * ``"scalar"``: a 0-form, or a top form (a density);
    * ``"flat"``: a vector ``v`` read as the form ``v♭``, or on an edge the
      scalar coefficient of its 1-form;
    * ``"hodge"``: a vector ``v`` read as ``★v♭``, or on an entity of dimension
      n the scalar coefficient of its n-form.

    On the entity the kernel is defined on, the action depends only on ``k``.
    The proxy matters when the DOF is immersed: a flat form lifts along the
    entity (a tangent), a hodge form across it (a normal), which needs
    ``k = d - 1`` for the cell it lands in.

    ``k=None`` is filled in from the entity: the top degree for a scalar, one
    less for a vector.
    """

    PROXIES = ("scalar", "flat", "hodge")

    def __init__(self, k=None, proxy="scalar"):
        if proxy not in self.PROXIES:
            raise ValueError(f"proxy must be one of {self.PROXIES}, not {proxy!r}")
        self.k = k
        self.proxy = proxy

    def __repr__(self):
        return f"FormAction(k={self.k}, proxy={self.proxy!r})"

    def __eq__(self, other):
        return isinstance(other, FormAction) and (self.k, self.proxy) == (other.k, other.proxy)

    def __hash__(self):
        return hash((FormAction, self.k, self.proxy))

    def _to_dict(self):
        return {"k": self.k, "proxy": self.proxy}

    def _from_dict(obj_dict):
        return FormAction(obj_dict["k"], obj_dict["proxy"])

    def matches(self, other):
        return isinstance(other, FormAction) and self.proxy == other.proxy and \
            (self.k is None or other.k is None or self.k == other.k)

    def resolve(self, entity, rank):
        n = entity.dimension
        k = self.k
        if k is None:
            k = n if rank == 0 else n - 1
        if rank == 0:
            valid = {"scalar": k in (0, n), "flat": k == n == 1, "hodge": k == n}[self.proxy]
        elif rank == 1:
            valid = (self.proxy == "flat" and k == 1) or (self.proxy == "hodge" and k == n - 1)
        else:
            valid = False
        if not valid:
            raise ValueError(f"{self} cannot be the action of a rank {rank} kernel on a {n}-dimensional entity")
        return FormAction(k, self.proxy)

    def lifted(self):
        # the form degree and proxy are kept; only the ambient dimension changes
        return self

    def trace(self, cell, entity, rank):
        if self.k == 0 or self.proxy == "scalar":
            return TrH1(cell)
        if self.proxy == "hodge":
            return TrHDiv(cell)
        # a tangential vector on a face has no trace class of its own
        return TrHCurl(cell) if entity.dimension == 1 else TrH1(cell)

    def act(self, A, value):
        if self.k == 0:
            return value
        if np.ndim(value) == 0:
            # top form: scales by the determinant
            return np.linalg.det(A) * value
        if self.proxy == "flat":
            return np.matmul(A, value)
        return np.linalg.det(A) * np.matmul(np.linalg.inv(A).T, value)

    def lift_matrix(self, cell, entity, rank):
        if self.k == 0 or self.proxy == "scalar":
            return None
        basis = np.array(cell.basis_vectors(entity=entity))
        if self.proxy == "flat":
            if rank == 0:
                return basis.T
            entity_basis = np.array(entity.basis_vectors()).T
            return np.matmul(basis.T, np.linalg.inv(entity_basis))
        if self.k != cell.dimension - 1:
            raise ValueError(f"A {self} kernel on a {entity.dimension}-dimensional entity "
                             f"cannot be immersed in a {cell.dimension}-dimensional cell")
        if rank != 0:
            raise NotImplementedError("Immersing a vector valued hodge kernel is not supported")
        if cell.dimension == 2:
            return np.matmul(basis, np.array([[0, -1], [1, 0]])).T
        if cell.dimension == 3:
            return np.cross(basis[0], basis[1]).reshape(3, 1)
        raise NotImplementedError("Immersing a hodge kernel is only implemented in 2D and 3D")


class SymmetricTensorAction(KernelAction):
    """The value is a symmetric matrix ``V``, transformed as ``A V Aᵀ``.

    Not used by any element yet: it shows a non de Rham action plugging in.
    """

    def __repr__(self):
        return "SymmetricTensorAction()"

    def __eq__(self, other):
        return isinstance(other, SymmetricTensorAction)

    def __hash__(self):
        return hash(SymmetricTensorAction)

    def _from_dict(obj_dict):
        return SymmetricTensorAction()

    def act(self, A, value):
        return np.matmul(np.matmul(A, value), np.transpose(A))

    def lift_matrix(self, cell, entity, rank):
        raise NotImplementedError("Immersing a symmetric tensor kernel is not supported yet")


H1Action = FormAction(0, "scalar")
L2Action = FormAction(None, "scalar")
HCurlAction = FormAction(1, "flat")
HDivAction = FormAction(None, "hodge")

_BY_SPACE_NAME = {"H_1": H1Action, "C_0": H1Action, "L_2": L2Action,
                  "HCurl": HCurlAction, "HDiv": HDivAction}


def as_action(action):
    """A kernel action from an action, or from one of the spaces H1, HCurl,
    HDiv or L2 as a shorthand for the matching form action."""
    if action is None or isinstance(action, KernelAction):
        return action
    if isinstance(action, InterpolationSpace) and action.name in _BY_SPACE_NAME:
        return _BY_SPACE_NAME[action.name]
    raise ValueError(f"{action} is not a kernel action: use a KernelAction, or H1, HCurl, HDiv or L2")
