from fuse.triples import ElementTriple, compute_form_degree
from fuse.traces import TrHCurl, TrHDiv
from fuse.cells import TensorProductPoint
from fuse.enriched import EnrichedElement
import numpy as np
from finat.ufl import TensorProductElement, FuseElement, HDivElement, HCurlElement
from itertools import product
from functools import reduce
from collections import defaultdict

def tensor_product(*factors, matrices=True):
    if not all(isinstance(f, ElementTriple) for f in factors):
        raise ValueError("All components of Tensor Product need to be a Fuse Triple.")
    return TensorProductTriple(*factors, matrices=matrices)


def symmetric_tensor_product(*factors, matrices=True):
    if not all(isinstance(f, ElementTriple) for f in factors):
        raise ValueError("All components of Tensor Product need to be a Fuse Triple.")
    return TensorProductTriple(*factors, matrices=matrices, symmetric=True)


def flatten_dictionary(tensor_dict):
    counters = {}
    flat_dict = {}
    for dim in tensor_dict.keys():
        total_dim = sum(dim)
        if total_dim not in counters.keys():
            counters[total_dim] = 0
            flat_dict[total_dim] = {}
        for i in range(len(tensor_dict[dim].keys())):
            flat_dict[total_dim][i + counters[total_dim]] = tensor_dict[dim][i]
        counters[total_dim] += len(tensor_dict[dim].keys())
    return flat_dict


class TensorProductTriple(ElementTriple):

    def __init__(self, *factors, flat=False, symmetric=True, matrices=True):
        self.factors = factors
        self.spaces = []
        for i in range(len(self.factors[0].spaces)):
            self.spaces.append(max(f.spaces[i] for f in self.factors))

        self.DOFGenerator = [f.DOFGenerator for f in self.factors]
        self.cell = TensorProductPoint(*[f.cell for f in factors])
        self.symmetric = symmetric
        self.flat = flat
        if self.flat:
            self.unflat_cell = self.cell
            self.cell = self.cell.flatten()
        self.dofs = self.generate()

        self.mat_transformer = None
        self.apply_matrices = matrices
        if self.apply_matrices:
            self.setup_matrices()

        self.pure_perm = not matrices

    @property
    def sub_elements(self):
        return self.factors

    def __repr__(self):
        return f"TensorProd({','.join(['{}' for f in self.factors])})".format(*(repr(f) for f in self.factors))

    def _entity_associations(self, dofs, overall=True):
        return self.entity_assocs, None, None

    def setup_matrices(self):
        if self.cell.flat and not self.symmetric:
            raise NotImplementedError("Matrices for flattened cells that are not symmetric not supported")
        for f in self.factors:
            f.to_ufl()
        oriented_mats_by_entity, flat_by_entity = self._initialise_entity_dicts(self.generate(), tensor=True)
        if self.flat:
            cell = self.unflat_cell
        else:
            cell = self.cell
        top = cell.to_fiat().get_topology()
        if len(self.factors) == 2:
            for dim in top.keys():
                total_dim = sum(dim) if self.flat else dim
                f_ents = [f.cell.get_topology()[d].keys() for f, d in zip(self.factors, dim)]
                ents = list(product(*(f_ents)))
                comp_os = cell.component_orientations()
                for e, sub_ents in enumerate(ents):
                    ent_dofs = self.entity_dofs[total_dim][self.ent_mapping[dim][sub_ents]]
                    if len(ent_dofs) >= 1:
                        sub_mat = oriented_mats_by_entity[dim][e]
                        mats = [f.matrices[d][ent] for f, d, ent in zip(self.factors, dim, sub_ents)]
                        ent_ids = [f.entity_ids[d][ent] for f, d, ent in zip(self.factors, dim, sub_ents)]
                        os = list(product(*([mat.keys() for mat in mats])))
                        for o in os:
                            sub_mats = [mat[o_f][np.ix_(ent_id, ent_id)] for mat, o_f, ent_id in zip(mats, o, ent_ids)]
                            if self.mat_transformer is not None:
                                o_classes = (f.cell.group.get_member_by_val(o_f) for f, o_f in zip(self.factors, o))
                                combined_sub_mat = self.mat_transformer(*sub_mats, o_classes)
                            else:
                                combined_sub_mat = reduce(lambda acc, x: np.kron(acc, x), sub_mats)
                            new_o = comp_os[dim][o]
                            if new_o in sub_mat.keys():
                                sub_mat[new_o][np.ix_(ent_dofs, ent_dofs)] = np.matmul(sub_mat[new_o][np.ix_(ent_dofs, ent_dofs)], combined_sub_mat)
                            # sub_mat[new_o][np.ix_(ent_dofs, ent_dofs)] = np.eye(np.matmul(sub_mat[new_o][np.ix_(ent_dofs, ent_dofs)], combined_sub_mat).shape[0])

        if self.cell.flat:
            oriented_mats_by_entity = flatten_dictionary(oriented_mats_by_entity)

        self.matrices = oriented_mats_by_entity
        self.reversed_matrices = self.reverse_dof_perms(self.matrices)

    def generate(self):
        dofs = [f.generate() for f in self.factors]
        ent_assocs = [f._entity_associations(dofs_f, overall=False)[0] for f, dofs_f in zip(self.factors, dofs)]
        if self.flat:
            top = self.unflat_cell.to_fiat().get_topology()
        else:
            top = self.cell.to_fiat().get_topology()
        self.entity_dofs = defaultdict(dict)
        self.ent_mapping = {}
        self.entity_assocs = defaultdict(dict)
        self.dof_ids = {}
        dofs = []
        ent_counter = defaultdict(lambda: 0)
        dof_counter = 0
        for dim in top.keys():
            total_dim = sum(dim) if self.flat else dim
            ents = [ent_assoc[d].keys() for ent_assoc, d in zip(ent_assocs, dim)]
            # if total_dim not in self.entity_dofs.keys():
            #     self.entity_dofs[total_dim] = {}
            #     self.entity_assocs[total_dim] = {}
            self.ent_mapping[dim] = {}
            ent_list = []
            for i, ent in enumerate(list(product(*ents))):
                self.ent_mapping[dim][ent] = i + ent_counter[total_dim] if self.flat else ent
                self.entity_dofs[total_dim][self.ent_mapping[dim][ent]] = []
                ent_list += [ent]
            for es in ent_list:
                e_dofs = [[d for dofs in ent_assoc[d][e].values() for d in dofs] for ent_assoc, d, e in zip(ent_assocs, dim, es)]
                new_dofs = list(product(*e_dofs))
                dofs += new_dofs
                dof_gens = "(" + "*".join([",".join(list(ent_assoc[d][e].keys())) for ent_assoc, d, e in zip(ent_assocs, dim, es)]) + ")"
                self.entity_assocs[total_dim][self.ent_mapping[dim][es]] = {dof_gens: new_dofs}
                self.entity_dofs[total_dim][self.ent_mapping[dim][es]] += [i + dof_counter for i in range(len(new_dofs))]
                for d in new_dofs:
                    self.dof_ids[d] = dof_counter
                    dof_counter += 1
                ent_counter[total_dim] += 1

        return dofs

    def to_ufl(self):
        ufl_sub_elements = [e.to_ufl() for e in self.sub_elements]
        if self.flat:
            return FuseElement(self, self.cell.to_ufl())
        return TensorProductElement(*ufl_sub_elements, cell=self.cell.to_ufl(), triple=self)

    def __add__(self, other):
        # assert self.cell == other.cell
        assert self.spaces[0].set_shape == other.spaces[0].set_shape
        assert str(self.spaces[1]) == str(other.spaces[1])

        return EnrichedElement(self, other, symmetric=self.symmetric and other.symmetric, matrices=self.apply_matrices or other.apply_matrices)

    def flatten(self):
        return TensorProductTriple(*self.factors, flat=True, symmetric=self.symmetric, matrices=self.apply_matrices)

    def unflatten(self):
        return TensorProductTriple(*self.factors, flat=False, symmetric=self.symmetric, matrices=self.apply_matrices)


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
        self.base_element = tensor_element
        self.gem_transformer, self.mat_transformer = self.select_fuse_hdiv_transformer(tensor_element)
        self.trace = TrHDiv
        super(HDiv, self).__init__(*tensor_element.factors, flat=tensor_element.flat, symmetric=tensor_element.symmetric, matrices=tensor_element.matrices)
        # self.spaces = (self.spaces[0], TrHDiv, self.spaces[2])

    def to_ufl(self):
        return HDivElement(super(HDiv, self).to_ufl(), transform=self.gem_transformer)
    
    def repr(self):
        return "HDiv(" + super(HDiv, self).repr() + ")"

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
            bv = cell.basis_vectors()[0][0]
            original = lambda v: [gem.Product(gem.Literal(-1), v), gem.Zero()]
            new = lambda v: [gem.Product(gem.Literal(bv), v), gem.Zero()], lambda m_a, m_b, o: np.kron(transform(cell, o[1]) @ m_a, m_b)
            return original, lambda m_a, m_b, o: np.kron(transform(cell, o[1]) @ m_a, m_b)
        elif ks == (1, 0):
            # Make the scalar value the upward-pointing normal on the
            # x-aligned edges.
            cell = element.sub_elements[0].cell
            bv = cell.basis_vectors()[0][0]
            return lambda v: [gem.Zero(), v], lambda m_a, m_b, o: np.kron(m_a, transform(cell, o[0]) @ m_b)
            # return lambda v: [gem.Zero(), gem.Product(gem.Literal(bv), v)], lambda m_a, m_b, o: np.kron(m_a, transform(cell, o[0]) @ m_b)
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

    def flatten(self):
        return HDiv(self.base_element.flatten())

    def unflatten(self):
        return HDiv(self.base_element.unflatten())


class HCurl(TensorProductTriple):

    def __init__(self, tensor_element):
        self.gem_transformer, self.mat_transformer = self.select_fuse_hcurl_transformer(tensor_element)
        self.trace = TrHCurl
        super(HDiv, self).__init__(*tensor_element.factors, tensor_element.flat, tensor_element.symmetric, tensor_element.matrices)
        self.spaces = (self.spaces[0], TrHCurl, self.spaces[2])

    def to_ufl(self):
        return HCurlElement(super(HCurl, self).to_ufl(), self.gem_transformer)

    def repr(self):
        return "HCurl(" + super(HCurl, self).repr() + ")"

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

    def flatten(self):
        return HCurl(self.base_element.flatten())

    def unflatten(self):
        return HCurl(self.base_element.unflatten())
