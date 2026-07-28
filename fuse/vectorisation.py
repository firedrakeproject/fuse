from fuse.dof import ComponentKernel
from fuse.triples import ElementTriple
from fuse.tensor_products import TensorProductTriple
from fuse.spaces.polynomial_spaces import normalise_shape
import numpy as np


class VectorTriple(ElementTriple):
    """A vector valued element formed from a scalar valued one.

    The DOFs of the base element are repeated once per component, each
    restricted to that component, and the polynomial space gains a shape.

    ``dim`` is the value shape, and defaults to the spatial dimension of the
    cell. It need not be the spatial dimension: an element carrying one scalar
    per chemical species, energy group or Fourier mode has as many components as
    the model has unknowns. A shape of rank two gives a full tensor valued
    element, without any symmetry constraint.
    """

    def __init__(self, base, dim=None, perm=True):
        if str(base.spaces[1]) in ("HDiv", "HCurl"):
            raise ValueError(
                f"Cannot vectorise an element in {base.spaces[1]}. The components do not transform independently under the pullback.")
        if base.spaces[0].shape:
            raise ValueError("Cannot vectorise an element that is already vector valued.")
        if isinstance(base, TensorProductTriple) or base.flat:
            raise ValueError("Cannot vectorise a tensor product element.")

        if dim is None:
            dim = base.cell.get_spatial_dimension()
        shape = normalise_shape(dim)
        if not shape:
            raise ValueError("Cannot vectorise to a scalar: dim must have a component.")

        # ensure base is fully set up
        base.to_ufl()
        self.base = base
        self.shape = shape
        # the number of clones of each base DOF, one per component
        self.N = int(np.prod(shape))
        self.cell = base.cell
        self.spaces = (base.spaces[0].to_vector(shape), base.spaces[1], base.spaces[2])
        self.DOFGenerator = base.DOFGenerator
        self.flat = False

        self.ref_el = None

        self.dofs = None
        self.dofs = self.generate()
        self.perm = perm

    def generate(self):
        """Clone each DOF once per component, tagging it with a ComponentKernel.

        Builds ``(new_dofs, comp_map)``, where ``comp_map`` sends each new DOF id
        to the ``(original dof id, flat component position)`` it came from.

        Components are enumerated in ``np.ndindex`` order, which is the order
        FIAT's ``ONPolynomialSet`` lays out the members of a shaped space.
        Ordering is component innermost, matching the convention of
        ``finat.TensorFiniteElement`` with ``shape_innermost``.
        Each clone keeps the entity of the DOF it came from, so the entity grouping of the original
        element carries over unchanged.
        """
        if self.dofs is None:
            new_dofs = []
            comp_map = {}
            for dof in self.base.generate():
                for flat, c in enumerate(np.ndindex(self.shape)):
                    new_dof = dof.with_kernel(ComponentKernel(c, base_kernel=dof.kernel))
                    new_dof.id = len(new_dofs)
                    comp_map[new_dof.id] = (dof.id, flat)
                    new_dofs.append(new_dof)
            self.dofs, self.comp_map = new_dofs, comp_map
        return self.dofs

    def num_dofs(self):
        return self.N * self.base.num_dofs()

    def sub_elements(self):
        return [self.base]

    def _check_dof_ordering(self):
        """Confirm the component innermost numbering the matrix lift relies on.

        setup_ids_and_nodes visits entities in sorted order and keeps the order of
        self.dofs within each entity, which places component c of base DOF i at
        N*i + c. The lift below is only correct while that holds.
        """
        for vec_id, (base_id, flat) in self.comp_map.items():
            expected = self.N * self.base.dof_id_to_fiat_id[base_id] + flat
            if self.dof_id_to_fiat_id[vec_id] != expected:
                raise ValueError(
                    f"DOF {vec_id} is numbered {self.dof_id_to_fiat_id[vec_id]} rather than "
                    f"{expected}, so the orientation matrices cannot be built by a Kronecker "
                    "product with the identity.")

    def setup_matrices(self):
        # Each component of a vector triple transforms as a scalar under the
        # identity pullback, and an orientation acts on every component alike, so the
        # matrices of the base element lift by a Kronecker product with the identity.
        self._check_dof_ordering()

        identity = np.eye(self.N)
        matrices = {dim: {e_id: {val: np.kron(mat, identity) for val, mat in by_val.items()}
                          for e_id, by_val in by_entity.items()}
                    for dim, by_entity in self.base.matrices.items()}
        reversed_matrices = self.reverse_dof_perms(matrices)

        self.pure_perm = False
        self.entity_perms = None
        self.apply_matrices = True
        return matrices, reversed_matrices

    def __repr__(self):
        return "Vector(%s, %s)" % (repr(self.base), "x".join(map(str, self.shape)))

    def _to_dict(self):
        o_dict = {"base": self.base, "dim": self.shape}
        return o_dict

    def dict_id(self):
        return "VectorTriple"

    def _from_dict(o_dict):
        return VectorTriple(o_dict["base"], o_dict.get("dim"))
