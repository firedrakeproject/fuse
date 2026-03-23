from fuse.triples import ElementTriple
from fuse.cells import TensorProductPoint
from finat.ufl import TensorProductElement, FuseElement


def tensor_product(A, B):
    if not (isinstance(A, ElementTriple) and isinstance(B, ElementTriple)):
        raise ValueError("Both components of Tensor Product need to be a Fuse Triple.")
    return TensorProductTriple(A, B)


class TensorProductTriple(ElementTriple):

    def __init__(self, A, B, flat=False):
        self.A = A
        self.B = B
        self.spaces = []
        for (a, b) in zip(self.A.spaces, self.B.spaces):
            self.spaces.append(a if a >= b else b)

        self.DOFGenerator = [A.DOFGenerator, B.DOFGenerator]
        self.cell = TensorProductPoint(A.cell, B.cell)
        self.flat = flat
        self.apply_matrices = False
        self.setup_matrices()
        if self.flat:
            self.unflat_cell = self.cell
            self.cell = self.cell.flatten()

    def sub_elements(self):
        return [self.A, self.B]

    def __repr__(self):
        return "TensorProd(%s, %s)" % (repr(self.A), repr(self.B))

    def setup_matrices(self):
        if self.A.cell.dimension > 1 or self.B.cell.dimension > 1:
            raise NotImplementedError("Combining of matrices not implemented in 3D")
        if self.cell.flat:
            raise NotImplementedError("Matrices for flattened cells not yet implemented")
        self.A.to_ufl()
        self.B.to_ufl()
        oriented_mats_by_entity, flat_by_entity = self._initialise_entity_dicts(self.generate())
        if self.flat:
            top = self.unflat_cell.to_fiat().get_topology()
        else:
            top = self.cell.to_fiat().get_topology()
        for dim in top.keys():
            a_ents = self.A.cell.get_topology()[dim[0]].keys()
            b_ents = self.A.cell.get_topology()[dim[1]].keys()
            ents = [(a, b) for a in a_ents for b in b_ents]
            for e, (a, b) in enumerate(ents):
                ent_dofs = self.entity_dofs[dim][(a, b)]
                if len(ent_dofs) > 1:
                    sub_mat = oriented_mats_by_entity[sum(dim)][e]
                    a_mat = self.A.matrices[dim[0]][a]
                    b_mat = self.B.matrices[dim[1]][b]
                    # need to make groups for tensor product cell that are
                    # different for flat or not
                    breakpoint()
        from collections import defaultdict
        from FIAT.reference_element import tuple_sum

        breakpoint()
        return super().setup_matrices()

    def generate(self):
        a_dofs = self.A.generate()
        b_dofs = self.B.generate()
        a_ent_assocs, _, _ = self.A._entity_associations(a_dofs, overall=False)
        b_ent_assocs, _, _ = self.B._entity_associations(b_dofs, overall=False)
        if self.flat:
            top = self.unflat_cell.to_fiat().get_topology()
        else:
            top = self.cell.to_fiat().get_topology()
        self.entity_dofs = {}
        dofs = []
        counter = 0
        for dim in top.keys():
            ents_A = a_ent_assocs[dim[0]].keys()
            ents_B = b_ent_assocs[dim[1]].keys()
            self.entity_dofs[dim] = {(a_e, b_e): tuple() for a_e in ents_A for b_e in ents_B}
            for a_e, b_e in self.entity_dofs[dim].keys():
                a_dofs = [d for dofs in a_ent_assocs[dim[0]][a_e].values() for d in dofs]
                b_dofs = [d for dofs in b_ent_assocs[dim[1]][b_e].values() for d in dofs]
                new_dofs = [(a, b) for a in a_dofs for b in b_dofs]
                dofs += new_dofs
                self.entity_dofs[dim][(a_e, b_e)] = [i + counter for i in range(len(new_dofs))]
                counter += len(new_dofs)
        self.dofs = dofs
        return dofs

    def to_ufl(self):
        if self.flat:
            return FuseElement(self, self.cell.to_ufl())
        ufl_sub_elements = [e.to_ufl() for e in self.sub_elements()]
        return TensorProductElement(*ufl_sub_elements, cell=self.cell.to_ufl())

    def flatten(self):
        return TensorProductTriple(self.A, self.B, flat=True)

    def unflatten(self):
        return TensorProductTriple(self.A, self.B, flat=False)
