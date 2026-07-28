from fuse.dof import ComponentKernel
from fuse.triples import ElementTriple
from fuse.tensor_products import TensorProductTriple
import numpy as np


class VectorTriple(ElementTriple):
    """A vector valued element formed from a scalar valued one.

    The DOFs of the base element are repeated once per component, each
    restricted to that component, and the polynomial space gains a shape. The
    number of components is the spatial dimension of the cell.
    """

    def __init__(self, base, perm=True):
        if str(base.spaces[1]) in ("HDiv", "HCurl"):
            raise ValueError(
                f"Cannot vectorise an element in {base.spaces[1]}. Its pullback mixes "
                "components, so the components do not transform independently.")
        if base.spaces[0].set_shape:
            raise ValueError("Cannot vectorise an element that is already vector valued.")
        if isinstance(base, TensorProductTriple) or base.flat:
            raise ValueError("Cannot vectorise a tensor product element.")

        # ensure base is fully set up
        base.to_ufl()
        self.base = base
        self.N = base.cell.get_spatial_dimension()
        self.cell = base.cell
        self.spaces = (base.spaces[0].to_vector(), base.spaces[1], base.spaces[2])
        self.DOFGenerator = base.DOFGenerator
        self.flat = False

        self.ref_el = None

        self.dofs = None
        self.dofs = self.generate()
        self.perm = perm

    def generate(self):
        """Clone each DOF once per component, tagging it with a ComponentKernel.

        Builds ``(new_dofs, comp_map)``, where ``comp_map`` sends each new DOF id
        to the ``(original dof id, component)`` it came from.

        Ordering is component innermost, matching the convention of
        ``finat.TensorFiniteElement`` with ``shape_innermost``.
        Each clone keeps the entity of the DOF it came from, so the entity grouping of the original
        element carries over unchanged.
        """
        if self.dofs is None:
            new_dofs = []
            comp_map = {}
            for dof in self.base.generate():
                for c in range(self.N):
                    new_dof = dof.with_kernel(ComponentKernel((c,), base_kernel=dof.kernel))
                    new_dof.id = len(new_dofs)
                    comp_map[new_dof.id] = (dof.id, c)
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
        for vec_id, (base_id, comp) in self.comp_map.items():
            expected = self.N * self.base.dof_id_to_fiat_id[base_id] + comp
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
        return "Vector(%s)" % repr(self.base)
