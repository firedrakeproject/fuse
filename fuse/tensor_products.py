from fuse.triples import ElementTriple, compute_form_degree
from fuse.traces import TrHCurl, TrHDiv
from fuse.cells import TensorProductPoint
import numpy as np
from finat.ufl import TensorProductElement, FuseElement, HDivElement, HCurlElement


def tensor_product(A, B, matrices=True):
    if not (isinstance(A, ElementTriple) and isinstance(B, ElementTriple)):
        raise ValueError("Both components of Tensor Product need to be a Fuse Triple.")
    return TensorProductTriple(A, B, matrices=matrices)


def symmetric_tensor_product(A, B):
    if not (isinstance(A, ElementTriple) and isinstance(B, ElementTriple)):
        raise ValueError("Both components of Tensor Product need to be a Fuse Triple.")
    return TensorProductTriple(A, B, symmetric=True)


def hcurl_transform(tensor_element):
    gem_transformer, mat_transformer = select_fuse_hcurl_transformer(tensor_element)
    tensor_element.add_mat_transformer(mat_transformer, TrHCurl)
    return gem_transformer


class TensorProductTriple(ElementTriple):

    def __init__(self, A, B, flat=False, symmetric=True, matrices=True):
        self.A = A
        self.B = B
        self.spaces = []
        for (a, b) in zip(self.A.spaces, self.B.spaces):
            self.spaces.append(a if a >= b else b)

        self.DOFGenerator = [A.DOFGenerator, B.DOFGenerator]
        self.cell = TensorProductPoint(A.cell, B.cell)
        self.symmetric = symmetric
        self.flat = flat
        if self.flat:
            self.unflat_cell = self.cell
            self.cell = self.cell.flatten()
        self.mat_transformer = None
        self.apply_matrices = matrices
        if self.apply_matrices:
            self.setup_matrices()

    @property
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
            b_ents = self.B.cell.get_topology()[dim[1]].keys()
            ents = [(a, b) for a in a_ents for b in b_ents]
            comp_os = self.cell.component_orientations()
            for e, (a, b) in enumerate(ents):
                ent_dofs = self.entity_dofs[dim][(a, b)]
                if len(ent_dofs) >= 1:
                    sub_mat = oriented_mats_by_entity[dim][e]
                    a_mat = self.A.matrices[dim[0]][a]
                    a_ent_ids = self.A.entity_ids[dim[0]][a]
                    b_mat = self.B.matrices[dim[1]][b]
                    b_ent_ids = self.B.entity_ids[dim[1]][b]

                    os = [(o_a, o_b) for o_a in a_mat.keys() for o_b in b_mat.keys()]
                    for o in os:
                        a_sub_mat = a_mat[o[0]][np.ix_(a_ent_ids, a_ent_ids)]
                        b_sub_mat = b_mat[o[1]][np.ix_(b_ent_ids, b_ent_ids)]
                        if self.mat_transformer is not None:
                            o_classes = (self.A.cell.group.get_member_by_val(o[0]), self.B.cell.group.get_member_by_val(o[1]))
                            combined_sub_mat = self.mat_transformer(a_sub_mat, b_sub_mat, o_classes)
                        else:
                            combined_sub_mat = np.kron(a_sub_mat, b_sub_mat)
                        new_o = comp_os[dim][o]
                        sub_mat[new_o][np.ix_(ent_dofs, ent_dofs)] = np.matmul(sub_mat[new_o][np.ix_(ent_dofs, ent_dofs)], combined_sub_mat)
        # from collections import defaultdict
        # from FIAT.reference_element import tuple_sum
        self.matrices = oriented_mats_by_entity
        self.reversed_matrices = self.reverse_dof_perms(self.matrices)

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
        ufl_sub_elements = [e.to_ufl() for e in self.sub_elements]
        return TensorProductElement(*ufl_sub_elements, cell=self.cell.to_ufl(), triple=self)

    def flatten(self):
        return TensorProductTriple(self.A, self.B, flat=True, symmetric=self.symmetric, matrices=self.apply_matrices)

    def unflatten(self):
        return TensorProductTriple(self.A, self.B, flat=False, symmetric=self.symmetric, matrices=self.apply_matrices)


def compute_matrix_transform(trace, cell, o):
    bvs = np.array(cell.basis_vectors())
    new_bvs = np.array(cell.orient(~o).basis_vectors())
    basis_change = np.matmul(new_bvs, np.linalg.inv(bvs))
    # if len(ent_dofs_ids) == basis_change.shape[0]:
    #     sub_mat = basis_change
    # elif len(dof_gen_class[dim].g2.members()) == 2 and len(ent_dofs_ids) == 1:
    #     # equivalently g1 trivial
    #     sub_mat = trace.manipulate_basis(basis_change)
    # else:
    # case where value change is a restriction of the full transformation of the basis
    value_change = trace(cell).manipulate_basis(basis_change)
    # sub_mat = np.kron((~o).matrix_form(), value_change)
    return value_change


class HDiv(TensorProductTriple):

    def __init__(self, tensor_element):
        self.gem_transformer, self.mat_transformer = self.select_fuse_hdiv_transformer(tensor_element)
        self.trace = TrHDiv
        super(HDiv, self).__init__(tensor_element.A, tensor_element.B, tensor_element.flat, tensor_element.symmetric, tensor_element.matrices)
        self.spaces[1] = TrHDiv

    def to_ufl(self):
        return HDivElement(super(HDiv, self).to_ufl(), self.gem_transformer)

    def select_fuse_hdiv_transformer(self, element):
        # Assume: something x interval
        import gem
        assert len(element.sub_elements) == 2
        assert element.sub_elements[1].cell.get_shape() == 1
        # Globally consistent edge orientations of the reference
        # quadrilateral: rightward horizontally, upward vertically.
        # Their rotation by 90 degrees anticlockwise is interpreted as the
        # positive direction for normal vectors.
        ks = tuple(compute_form_degree(fe.cell, fe.spaces) for fe in element.sub_elements)
        transform = lambda cell, o: compute_matrix_transform(element.trace, cell, o)
        if ks == (0, 1):
            # Make the scalar value the right hand rule normal on the
            # y-aligned edges.
            cell = element.sub_elements[1].cell
            bv = cell.basis_vectors()[0]
            return lambda v: [gem.Product(gem.Literal(bv[0]), v), gem.Zero()], lambda m_a, m_b, o: np.kron(transform(cell, o[1]) @ m_a, m_b)
        elif ks == (1, 0):
            # Make the scalar value the upward-pointing normal on the
            # x-aligned edges.
            cell = element.sub_elements[0].cell
            bv = cell.basis_vectors()[0][0]
            return lambda v: [gem.Zero(), gem.Product(gem.Literal(bv), v)], lambda m_a, m_b, o: np.kron(m_a, transform(cell, o[0]) @ m_b)
        # elif ks == (2, 0):
        #     # Same for 3D, so z-plane.
        #     return lambda v: [gem.Zero(), gem.Zero(), v]
        # elif ks == (1, 1):
        #     if element.mapping == "contravariant piola":
        #         # Pad the 2-vector normal on the "base" cell into a
        #         # 3-vector, maintaining direction.
        #         return lambda v: [gem.Indexed(v, (0,)),
        #                           gem.Indexed(v, (1,)),
        #                           gem.Zero()]
        #     elif element.mapping == "covariant piola":
        #         # Rotate the 2-vector tangential component on the "base"
        #         # cell 90 degrees anticlockwise into a 3-vector and pad.
        #         return lambda v: [gem.Indexed(v, (1,)),
        #                           gem.Product(gem.Literal(-1), gem.Indexed(v, (0,))),
        #                           gem.Zero()]
        #     else:
        #         assert False, "Unexpected original mapping!"
        else:
            raise NotImplementedError("Unexpected original mapping!")
            assert False, "Unexpected form degree combination!"

    def select_fuse_hcurl_transformer(element):
        import gem
        # Assume: something x interval
        assert len(element.sub_elements) == 2
        assert element.sub_elements[1].cell.get_shape() == 1

        # Globally consistent edge orientations of the reference
        # quadrilateral: rightward horizontally, upward vertically.
        # Tangential vectors interpret these as the positive direction.
        dim = element.cell.get_spatial_dimension()
        ks = tuple(compute_form_degree(fe.cell, fe.spaces) for fe in element.sub_elements)
        if all(str(fe.spaces[1]) == "H1" or str(fe.spaces[1]) == "L2" for fe in element.sub_elements):  # affine mapping
            if ks == (1, 0):
                # Can only be 2D.  Make the scalar value the
                # tangential following the cell edge direction on the x-aligned edges.
                bv = element.sub_elements[0].cell.basis_vectors()[0][0]
                return lambda v: [gem.Product(gem.Literal(bv), v), gem.Zero()]
            elif ks == (0, 1):
                # Can be any spatial dimension.  Make the scalar value the
                # tangential following the cell edge direction .
                bv = element.sub_elements[1].cell.basis_vectors()[0][0]
                return lambda v: [gem.Zero()] * (dim - 1) + [gem.Product(gem.Literal(bv), v)]
            else:
                assert False
        # elif any(str(fe.spaces[1]) == "HCurl" for fe in element.sub_elements):  # Covariant Piola mapping
        #     # Second factor must be continuous interval.  Just padding.
        #     return lambda v: [gem.Indexed(v, (0,)),
        #                       gem.Indexed(v, (1,)),
        #                       gem.Zero()]
        # elif any(str(fe.spaces[1]) == "HDiv" for fe in element.sub_elements):  # Contravariant Piola mapping
        #     # Second factor must be continuous interval.  Rotate the
        #     # 2-vector tangential component on the "base" cell 90 degrees
        #     # clockwise into a 3-vector and pad.
        #     return lambda v: [gem.Product(gem.Literal(-1), gem.Indexed(v, (1,))),
        #                       gem.Indexed(v, (0,)),
        #                       gem.Zero()]
        else:
            raise NotImplementedError("Unexpected original mapping!")
            assert False, "Unexpected original mapping!"
