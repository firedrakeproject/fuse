import numpy as np
from fuse.triples import ElementTriple
import finat.ufl


class EnrichedElement(ElementTriple):
    """
    Non-nodal representation of an enriched element.

    In general, FUSE element triples should be represented nodally,
    however this may not be possible for all constructions.

    In particular, we need to preserve tensor product structure.
    """

    def __init__(self, A, B, flat=False, symmetric=True, matrices=True):
        from fuse.tensor_products import TensorProductTriple
        if not isinstance(A, TensorProductTriple) or not isinstance(B, TensorProductTriple):
            raise ValueError("EnrichedElement should only be used for Tensor product elements. Use + between triples for enrichment.")
        self.A = A
        self.B = B
        self.spaces = (A.spaces[0] + B.spaces[0], A.spaces[1], max([A.spaces[2], B.spaces[2]]))

        self.DOFGenerator = [A.DOFGenerator, B.DOFGenerator]
        if A.cell.flat != B.cell.flat:
            raise ValueError("Tensor products must both be flat or both not flat for enrichment.")
        self.cell = A.cell
        self.symmetric = symmetric
        self.apply_matrices = matrices
        if self.apply_matrices:
            self.setup_matrices()

        self.pure_perm = not matrices

    @property
    def sub_elements(self):
        return [self.A, self.B]

    def __repr__(self):
        return "Enriched(%s, %s)" % (repr(self.A), repr(self.B))

    def setup_matrices(self):
        if self.cell.flat and not self.symmetric:
            raise NotImplementedError("Matrices for flattened cells that are not symmetric not supported")
        self.A.to_ufl()
        self.B.to_ufl()
        oriented_mats_by_entity, flat_by_entity = self._initialise_entity_dicts(self.generate(), tensor=(not self.cell.flat))
        if self.cell.flat:
            cell = self.A.unflat_cell
        else:
            cell = self.cell
        top = cell.to_fiat().get_topology()
        for dim in top.keys():
            total_dim = sum(dim) if self.cell.flat else dim
            ents = self.entity_dofs[total_dim].keys()
            # comp_os = cell.component_orientations()
            for e_idx, e in enumerate(ents):
                ent_dofs = self.entity_dofs[total_dim][e]
                if len(ent_dofs) >= 1:
                    sub_mat = oriented_mats_by_entity[total_dim][e_idx]
                    a_mat = self.A.matrices[total_dim][e_idx]
                    a_ent_ids = self.A.entity_dofs[total_dim][e]
                    b_mat = self.B.matrices[total_dim][e_idx]
                    b_ent_ids = self.B.entity_dofs[total_dim][e]

                    for o in a_mat.keys():
                        a_sub_mat = a_mat[o][np.ix_(a_ent_ids, a_ent_ids)]
                        b_sub_mat = b_mat[o][np.ix_(b_ent_ids, b_ent_ids)]
                        combined_sub_mat = np.block([[a_sub_mat, np.zeros((a_sub_mat.shape[0], b_sub_mat.shape[1]))],
                                                    [np.zeros((b_sub_mat.shape[0], a_sub_mat.shape[1])), b_sub_mat]])
                        sub_mat[o][np.ix_(ent_dofs, ent_dofs)] = np.matmul(sub_mat[o][np.ix_(ent_dofs, ent_dofs)], combined_sub_mat)

        self.matrices = oriented_mats_by_entity
        self.reversed_matrices = self.reverse_dof_perms(self.matrices)

    def generate(self):
        a_dofs = self.A.generate()
        b_dofs = self.B.generate()
        numAdofs = len(a_dofs)
        self.entity_dofs = {}
        for dim in self.A.entity_dofs.keys():
            self.entity_dofs[dim] = {}
            for ent in self.A.entity_dofs[dim]:
                self.entity_dofs[dim][ent] = self.A.entity_dofs[dim][ent] + [b_dof + numAdofs for b_dof in self.B.entity_dofs[dim][ent]]
        return a_dofs + b_dofs

    def to_ufl(self):
        ufl_sub_elements = [e.to_ufl() for e in self.sub_elements]
        return finat.ufl.EnrichedElement(*ufl_sub_elements, triple=self)

    def flatten(self):
        return EnrichedElement(self.A.flatten(), self.B.flatten(), flat=True, symmetric=self.symmetric, matrices=self.matrices)

    def unflatten(self):
        return EnrichedElement(self.A.unflatten(), self.B.unflatten(), flat=False, symmetric=self.symmetric, matrices=self.matrices)
