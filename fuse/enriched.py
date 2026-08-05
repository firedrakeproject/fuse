import numpy as np
from fuse.tensor_products import TensorProductTriple
import finat.ufl


class EnrichedElement(TensorProductTriple):
    """
    Non-nodal representation of an enriched element.

    In general, FUSE element triples should be represented nodally,
    however this may not be possible for all constructions.

    In particular, we need to preserve tensor product structure.
    """

    def __init__(self, A, B, flat=False, symmetric=None, matrices=True):
        valid_types = (TensorProductTriple, EnrichedElement)
        if not isinstance(A, valid_types) or not isinstance(B, valid_types):
            raise ValueError("EnrichedElement should only be used for Tensor product elements. Use + between triples for enrichment.")
        self.A = A
        self.B = B
        self.spaces = (A.spaces[0] + B.spaces[0], A.spaces[1], max([A.spaces[2], B.spaces[2]]))

        self.DOFGenerator = [A.DOFGenerator, B.DOFGenerator]
        if A.cell.flat != B.cell.flat:
            raise ValueError("Tensor products must both be flat or both not flat for enrichment.")
        self.cell = A.cell
        self.flat = flat or self.cell.flat
        if hasattr(A, "unflat_cell"):
            self.unflat_cell = A.unflat_cell
        if getattr(A, "trace", None) is not getattr(B, "trace", None):
            raise ValueError("Cannot enrich elements with different traces.")
        self.trace = getattr(A, "trace", None)
        # See TensorProductTriple.__init__ for the meaning of ``symmetric``.
        self.requested_symmetric = symmetric
        self.symmetric = True if symmetric is None else symmetric
        self.apply_matrices = matrices
        if self.apply_matrices:
            self.setup_matrices()

        self.pure_perm = not matrices

    @property
    def sub_elements(self):
        return [self.A, self.B]

    def get_value_shape(self):
        if str(self.spaces[1]) in ("HDiv", "HCurl"):
            return (self.cell.get_spatial_dimension(),)
        return super().get_value_shape()

    def __repr__(self):
        return "Enriched(%s, %s)" % (repr(self.A), repr(self.B))

    def __add__(self, other):
        assert self.spaces[0].shape == other.spaces[0].shape
        assert str(self.spaces[1]) == str(other.spaces[1])
        return EnrichedElement(self, other, flat=self.flat and other.flat,
                               matrices=self.apply_matrices or other.apply_matrices)

    def setup_matrices(self):
        if self.flat and not self.symmetric:
            raise NotImplementedError("Matrices for flattened cells that are not symmetric not supported")
        self.A.to_ufl()
        self.B.to_ufl()
        dofs = self.generate()
        dof_keys, key_to_index = self._axis_key_maps(dofs)
        # Reset closure failures
        self._closure_failures = set()
        oriented_mats_by_entity, flat_by_entity = self._initialise_entity_dicts(dofs, tensor=(not self.flat))
        if self.flat:
            cell = self.A.unflat_cell
        else:
            cell = self.cell
        top = cell.to_fiat().get_topology()
        seen_total_dims = set()
        for dim in top.keys():
            total_dim = sum(dim) if self.flat else dim
            if total_dim in seen_total_dims:
                continue
            seen_total_dims.add(total_dim)
            ents = self.entity_dofs[total_dim].keys()
            # comp_os = cell.component_orientations()
            for e_idx, e in enumerate(ents):
                ent_dofs = self.entity_dofs[total_dim][e]
                if len(ent_dofs) >= 1:
                    sub_mat = oriented_mats_by_entity[total_dim][e_idx]
                    a_mat = self.A.generation_order_matrices()[total_dim][e_idx]
                    a_ent_ids = self.A.entity_dofs[total_dim][e]
                    b_mat = self.B.generation_order_matrices()[total_dim][e_idx]
                    b_ent_ids = self.B.entity_dofs[total_dim][e]

                    for o in a_mat.keys():
                        a_sub_mat = a_mat[o][np.ix_(a_ent_ids, a_ent_ids)]
                        b_sub_mat = b_mat[o][np.ix_(b_ent_ids, b_ent_ids)]
                        combined_sub_mat = np.block([[a_sub_mat, np.zeros((a_sub_mat.shape[0], b_sub_mat.shape[1]))],
                                                    [np.zeros((b_sub_mat.shape[0], a_sub_mat.shape[1])), b_sub_mat]])
                        sub_mat[o][np.ix_(ent_dofs, ent_dofs)] = np.matmul(sub_mat[o][np.ix_(ent_dofs, ent_dofs)], combined_sub_mat)
                    if self.flat:
                        entity = self.cell.d_entities(total_dim)[e]
                        self._fill_axis_permutations(entity, dim, ent_dofs, sub_mat, dof_keys, key_to_index)

        self.matrices = oriented_mats_by_entity
        self.reversed_matrices = self.reverse_dof_perms(self.matrices)
        if self.flat:
            self._snapshot_generation_order()
            self._regroup_matrices()

        self._resolve_symmetry()

    def generate(self):
        a_dofs = self.A.generate()
        b_dofs = self.B.generate()
        numAdofs = len(a_dofs)
        self.entity_dofs = {}
        for dim in self.A.entity_dofs.keys():
            self.entity_dofs[dim] = {}
            for ent in self.A.entity_dofs[dim]:
                self.entity_dofs[dim][ent] = self.A.entity_dofs[dim][ent] + [b_dof + numAdofs for b_dof in self.B.entity_dofs[dim][ent]]
        self.dofs = a_dofs + b_dofs
        return self.dofs

    def to_ufl(self):
        ufl_sub_elements = [e.to_ufl() for e in self.sub_elements]
        return finat.ufl.EnrichedElement(*ufl_sub_elements, triple=self)

    def flatten(self, symmetric=None):
        if symmetric is None:
            symmetric = self.requested_symmetric
        return EnrichedElement(self.A.flatten(), self.B.flatten(), flat=True, symmetric=symmetric, matrices=self.apply_matrices)

    def unflatten(self):
        return EnrichedElement(self.A.unflatten(), self.B.unflatten(), flat=False, symmetric=self.requested_symmetric, matrices=self.apply_matrices)
