from fuse.triples import ElementTriple
from fuse.traces import TrHCurl, TrHDiv
from fuse.spaces.element_sobolev_spaces import CellHDiv, CellHCurl
from fuse.cells import TensorProductPoint
import numpy as np
from finat.ufl import TensorProductElement, FuseElement, HDivElement, HCurlElement
from itertools import product, permutations
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


def leaf_dof_keys(elem, out=None):
    """Map each of ``elem``'s generated DOFs to a tuple of per-axis leaf DOFs.

    A tensor product DOF is a tuple with one component per factor, but a
    factor may itself be a product, so a component can be a nested tuple.
    Descending to the one-dimensional leaves gives every DOF a flat key with
    one entry per spatial axis, which is what an axis permutation acts on.
    """
    if out is None:
        out = {}
    from fuse.enriched import EnrichedElement
    if isinstance(elem, EnrichedElement):
        # Checked before TensorProductTriple, which it subclasses.
        leaf_dof_keys(elem.A, out)
        leaf_dof_keys(elem.B, out)
    elif isinstance(elem, TensorProductTriple):
        sub_keys = [leaf_dof_keys(f) for f in elem.factors]
        for dof in elem.generate():
            out[dof] = sum((sub_keys[i][comp] for i, comp in enumerate(dof)), ())
    else:
        for dof in elem.generate():
            out[dof] = (dof,)
    return out


class TensorProductTriple(ElementTriple):

    # Axis permutations the DOF set turned out not to be closed under.
    # Populated by ``_fill_face_axis_swaps``; stays empty when matrices are
    # not built at all.
    _closure_failures = frozenset()

    def __init__(self, *factors, flat=False, symmetric=None, matrices=True):
        if len(factors) < 2:
            raise ValueError("Cannot create a tensor product with fewer than 2 factors")
        self.factors = factors
        self.spaces = []
        for i in range(len(self.factors[0].spaces)):
            self.spaces.append(max(f.spaces[i] for f in self.factors))

        self.DOFGenerator = [f.DOFGenerator for f in self.factors]
        self.cell = TensorProductPoint(*[f.cell for f in factors])
        # ``symmetric=None`` means derive it from whether the DOF set is
        # actually closed under axis permutation; True additionally asserts
        # that it is, False opts out of building the axis-swap orientations.
        self.requested_symmetric = symmetric
        self.symmetric = True if symmetric is None else symmetric
        self.flat = flat
        if self.flat:
            self.unflat_cell = self.cell
            self.cell = self.cell.flatten()
        self.dofs = self.generate()

        # Subclasses (HDiv, HCurl) set self.mat_transformer before calling
        # this constructor; only default it here if they haven't.
        self.mat_transformer = getattr(self, "mat_transformer", None)
        self.apply_matrices = matrices
        if self.apply_matrices:
            self.setup_matrices()

        self.pure_perm = not matrices

    @property
    def sub_elements(self):
        return self.factors

    @property
    def form_degree(self):
        # Using lowest dimension dof to define form degree, tensor product dims are additive
        return min(sum(comp.cell_defined_on.dim() for comp in dof) for dof in self.generate())

    def __repr__(self):
        return f"TensorProd({','.join(['{}' for f in self.factors])})".format(*(repr(f) for f in self.factors))

    def _entity_associations(self, dofs, overall=True):
        return self.entity_assocs, None, None

    def setup_matrices(self):
        if self.flat and not self.symmetric:
            raise NotImplementedError("Matrices for flattened cells that are not symmetric not supported")
        for f in self.factors:
            f.to_ufl()
        dofs = self.generate()
        dof_keys, key_to_index = self._axis_key_maps(dofs)
        oriented_mats_by_entity, flat_by_entity = self._initialise_entity_dicts(dofs, tensor=True)
        if self.flat:
            cell = self.unflat_cell
        else:
            cell = self.cell
        top = cell.to_fiat().get_topology()
        for dim in top.keys():
            total_dim = sum(dim) if self.flat else dim
            f_ents = [f.cell.get_topology()[d].keys() for f, d in zip(self.factors, dim)]
            ents = list(product(*(f_ents)))
            comp_os = cell.component_orientations()
            for e, sub_ents in enumerate(ents):
                ent_dofs = self.entity_dofs[total_dim][self.ent_mapping[dim][sub_ents]]
                if len(ent_dofs) >= 1:
                    sub_mat = oriented_mats_by_entity[dim][e]
                    mats = [f.generation_order_matrices()[d][ent] for f, d, ent in zip(self.factors, dim, sub_ents)]
                    ent_ids = [f.entity_dofs[d][ent] for f, d, ent in zip(self.factors, dim, sub_ents)]
                    os = list(product(*([mat.keys() for mat in mats])))
                    for o in os:
                        sub_mats = [mat[o_f][np.ix_(ent_id, ent_id)] for mat, o_f, ent_id in zip(mats, o, ent_ids)]
                        if self.mat_transformer is not None:
                            o_classes = [f.cell.group.get_member_by_val(o_f) for f, o_f in zip(self.factors, o)]
                            combined_sub_mat = self.mat_transformer(*sub_mats, o_classes)
                        else:
                            combined_sub_mat = reduce(lambda acc, x: np.kron(acc, x), sub_mats)
                        new_o = comp_os[dim][o]
                        if new_o in sub_mat.keys():
                            sub_mat[new_o][np.ix_(ent_dofs, ent_dofs)] = np.matmul(sub_mat[new_o][np.ix_(ent_dofs, ent_dofs)], combined_sub_mat)
                        # sub_mat[new_o][np.ix_(ent_dofs, ent_dofs)] = np.eye(np.matmul(sub_mat[new_o][np.ix_(ent_dofs, ent_dofs)], combined_sub_mat).shape[0])
                    if self.flat:
                        self._fill_face_axis_swaps(dim, ent_dofs, sub_mat, dof_keys, key_to_index)

        if self.flat:
            oriented_mats_by_entity = flatten_dictionary(oriented_mats_by_entity)

        self.matrices = oriented_mats_by_entity
        self.reversed_matrices = self.reverse_dof_perms(self.matrices)

        if self.flat:
            self._snapshot_generation_order()
            self._regroup_matrices()

        self._resolve_symmetry()

    def _resolve_symmetry(self):
        """Settle ``self.symmetric`` against the closure the fill observed.

        A flat element is symmetric exactly when every entity's DOFs are
        closed under permutation of that entity's axes, which is what
        ``_fill_face_axis_swaps`` needs in order to produce the axis-swap
        orientations at all.
        """
        closed = not self._closure_failures
        if self.requested_symmetric is None:
            self.symmetric = closed
        elif self.requested_symmetric and not closed:
            raise NotImplementedError(
                "%r was declared symmetric but its DOFs are not closed under "
                "axis permutation %r" % (self, sorted(self._closure_failures)))

    def _axis_permutation_sign(self, tau):
        """Value change an H(div) DOF picks up from permuting axes.

        Reordering an entity's axes by an odd permutation reverses its
        orientation, so the normal an H(div) DOF integrates against flips.
        Nothing downstream supplies this: Firedrake's assembly selects one of
        these matrices and multiplies by it once (``FuseMatrixApplyBuilder``
        in ``firedrake/pack.py``), so the sign has to be carried here.

        The reflection part of an orientation already carries its own sign
        through the matrix being composed with, leaving only the
        permutation's parity. Tangential (H(curl)) and scalar DOFs are
        unaffected by the reversal itself.
        """
        if str(self.spaces[1]) != "HDiv":
            return 1
        inversions = sum(1 for i in range(len(tau))
                         for j in range(i + 1, len(tau)) if tau[i] > tau[j])
        return -1 if inversions % 2 else 1

    def _axis_key_maps(self, dofs):
        """Per-axis leaf keys for ``dofs``, indexed both ways.

        Returns ``(dof_keys, key_to_index)`` where ``dof_keys`` maps a global
        DOF index to its leaf key and ``key_to_index`` inverts that. Resets
        the record of axis permutations the DOF set is not closed under.
        """
        self._closure_failures = set()
        leaves = leaf_dof_keys(self)
        dof_keys = {}
        key_to_index = {}
        for i, dof in enumerate(dofs):
            key = leaves.get(dof)
            if key is None:
                continue
            dof_keys[i] = key
            key_to_index[key] = i
        return dof_keys, key_to_index

    def _snapshot_generation_order(self):
        """Keep a copy of the matrices indexed in generation DOF order.

        ``_regroup_matrices`` rewrites ``self.matrices`` into the
        dimension-grouped order Firedrake consumes, but ``self.entity_dofs``
        stays in generation order. Parent elements pair the two when they
        read a factor's blocks, so they need the un-regrouped copy.
        """
        self._gen_order_matrices = {dim: {e: {o: mat.copy() for o, mat in os.items()}
                                          for e, os in ents.items()}
                                    for dim, ents in self.matrices.items()}

    def _regroup_matrices(self):
        """Re-express the orientation matrices in closure DOF order.

        FUSE generates tensor-product and enriched DOFs in an interleaved
        order. Firedrake packs each cell's closure DOFs by entity dimension
        and, within a dimension, by entity number, and applies these matrices
        in that order.

        Ordering by dimension alone is not enough. An entity's matrix is the
        identity apart from a block sitting at that entity's own DOFs, so if
        the entities within a dimension come out in the wrong order the block
        lands on a different entity's DOFs -- the right transformation applied
        to the wrong DOF. That stays invisible while every orientation is the
        identity and only bites once an entity is actually reversed.
        """
        grouped = [dof
                   for total_dim in sorted(self.entity_dofs)
                   for ent in sorted(self.entity_dofs[total_dim])
                   for dof in self.entity_dofs[total_dim][ent]]
        n = len(grouped)
        if grouped == list(range(n)):
            return
        ix = np.ix_(grouped, grouped)
        for mats in (self.matrices, self.reversed_matrices):
            for ents in mats.values():
                for os in ents.values():
                    for k in list(os.keys()):
                        os[k] = os[k][ix].copy()

    def _fill_face_axis_swaps(self, dim, ent_dofs, sub_mat, dof_keys, key_to_index):
        """Populate the axis-permuting orientations of an entity.

        The per-entity loop in ``setup_matrices`` fills only the reflection
        subgroup (extrinsic orientation ``eo == 0``, canonical keys
        ``0..2**d - 1``) because it enumerates products of the factors' own
        orientations, which cannot permute axes. Enriched elements are worse
        still: they combine their summands block-diagonally, so they cannot
        even express a permutation that maps one summand's DOFs onto
        another's.

        The remaining members compose those reflections with a pure DOF
        permutation. An axis permutation ``tau`` sends the DOF whose per-axis
        leaf key is ``k`` to the DOF with key ``tau(k)``, so looking that key
        up gives the permutation directly, for any number of axes and across
        summand blocks. The canonical key ``2**d * eo + io`` (see
        ``fuse.utils.canonical_tensor_orientation_key``) is then
        ``M[io] @ P_tau``.

        Skipped when the entity's DOFs are not closed under ``tau``; the
        caller records that as a failure of symmetry.
        """
        ed = sum(dim) if isinstance(dim, tuple) else dim
        if ed < 2 or len(ent_dofs) == 0:
            # A point or an interval has no axes to permute.
            return
        keys = [dof_keys.get(d) for d in ent_dofs]
        if any(k is None for k in keys):
            self._closure_failures.add((ed, None))
            return
        # Which leaf axes this entity actually extends along. Taking these
        # from the DOFs rather than from `dim` is what lets one code path
        # serve hex cells, hex faces, and factors that are themselves
        # flattened quads.
        active = {tuple(j for j, c in enumerate(k) if c.cell_defined_on.dim() == 1) for k in keys}
        if len(active) != 1 or len(next(iter(active))) != ed:
            # The entity's DOFs disagree about which axes it extends along,
            # so there is no well-defined action to build.
            self._closure_failures.add((ed, None))
            return
        act = active.pop()
        local = {d: i for i, d in enumerate(ent_dofs)}
        grid = np.ix_(ent_dofs, ent_dofs)
        for eo, tau in enumerate(sorted(permutations(range(ed)))):
            if eo == 0:
                continue
            perm = []
            for k in keys:
                new_key = list(k)
                for i in range(ed):
                    new_key[act[i]] = k[act[tau.index(i)]]
                target = key_to_index.get(tuple(new_key))
                if target is None or target not in local:
                    perm = None
                    break
                perm.append(local[target])
            if perm is None:
                self._closure_failures.add((ed, eo))
                continue
            P = self._axis_permutation_sign(tau) * np.eye(len(ent_dofs))[perm]
            for io in range(2 ** ed):
                swap_key = 2 ** ed * eo + io
                if io in sub_mat and swap_key in sub_mat:
                    sub_mat[swap_key][grid] = np.matmul(sub_mat[io][grid], P)

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
        assert self.spaces[0].shape == other.spaces[0].shape
        assert str(self.spaces[1]) == str(other.spaces[1])
        from fuse.enriched import EnrichedElement
        return EnrichedElement(self, other, flat=self.flat and other.flat,
                               matrices=self.apply_matrices or other.apply_matrices)

    def flatten(self):
        return TensorProductTriple(*self.factors, flat=True, symmetric=self.requested_symmetric, matrices=self.apply_matrices)

    def unflatten(self):
        return TensorProductTriple(*self.factors, flat=False, symmetric=self.requested_symmetric, matrices=self.apply_matrices)


def compute_matrix_transform(trace, cell, o):
    dim = cell.get_spatial_dimension()
    bvs = np.array(cell.basis_vectors())
    new_bvs = np.array(cell.orient(~o).basis_vectors())
    if bvs.shape[0] != dim:
        # basis_vectors() gives one vector per non-reference vertex, which
        # only forms a square (invertible) basis for simplices (vertex
        # count == dim + 1). For non-simplex cells (e.g. a quadrilateral
        # face of a hex), the vectors from the reference vertex to its two
        # adjacent vertices (the first `dim` entries) already form a valid
        # basis; later entries are redundant (e.g. diagonals).
        bvs = bvs[:dim]
        new_bvs = new_bvs[:dim]
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
        super(HDiv, self).__init__(*tensor_element.factors, flat=tensor_element.flat, symmetric=tensor_element.requested_symmetric, matrices=tensor_element.apply_matrices)
        self.spaces = (self.spaces[0], CellHDiv(self.cell), self.spaces[2])

    def to_ufl(self):
        return HDivElement(super(HDiv, self).to_ufl(), transform=self.gem_transformer)

    def repr(self):
        return "HDiv(" + super(HDiv, self).repr() + ")"

    def select_fuse_hdiv_transformer(self, element):
        # Assume: something x interval
        import gem
        assert len(element.sub_elements) == 2
        assert element.sub_elements[1].cell.get_shape() == 1
        ks = tuple(fe.form_degree for fe in element.sub_elements)
        dims = tuple(fe.cell.get_spatial_dimension() for fe in element.sub_elements)
        transform = lambda cell, o: compute_matrix_transform(self.trace, cell, o)
        if ks == (0, 1) and dims == (1, 1):
            # Both factors are 1D intervals (2D quad case).  Make the
            # scalar value the right hand rule normal on the y-aligned
            # edges.
            cell = element.sub_elements[1].cell
            bv = cell.basis_vectors()[0][0]
            mats = lambda m_a, m_b, o: np.kron(transform(cell, o[1]) * m_a, m_b)
            return lambda v: [gem.Product(gem.Literal(bv), v), gem.Zero()], mats
        elif ks == (1, 0) and dims == (1, 1):
            # Both factors are 1D intervals (2D quad case).  Make the
            # scalar value the upward-pointing normal on the x-aligned
            # edges.
            cell = element.sub_elements[0].cell
            bv = cell.basis_vectors()[0][0]
            return lambda v: [gem.Zero(), gem.Product(gem.Literal(bv), v)], lambda m_a, m_b, o: np.kron(m_a, transform(cell, o[0]) * m_b)
        elif ks == (2, 0) and dims == (2, 1):
            # First factor is a plain (unwrapped) scalar DG element on a
            # 2D base cell, second is a CG interval
            cell = element.sub_elements[0].cell
            mats = lambda m_a, m_b, o: np.kron(m_a, transform(cell, o[0]) * m_b)
            return lambda v: [gem.Zero(), gem.Zero(), v], mats
        elif ks == (1, 1) and dims == (2, 1) and str(element.sub_elements[0].spaces[1]) == "HDiv":
            # First factor is an already H(div)-wrapped 2D element (the
            # in-plane RT part), second is a DG interval: the horizontal
            # (x, y) components of a 3D H(div) field.
            cell = element.sub_elements[1].cell
            mats = lambda m_a, m_b, o: np.kron(m_a, transform(cell, o[1]) * m_b)
            return lambda v: [gem.Indexed(v, (0,)), gem.Indexed(v, (1,)), gem.Zero()], mats
        elif ks == (1, 1) and dims == (2, 1) and str(element.sub_elements[0].spaces[1]) == "HCurl":
            # First factor is an already H(curl)-wrapped 2D element,
            # second is a DG interval: rotate the tangential 2-vector 90
            # degrees anticlockwise into a 3-vector and pad.
            cell = element.sub_elements[1].cell
            mats = lambda m_a, m_b, o: np.kron(m_a, transform(cell, o[1]) * m_b)
            return lambda v: [gem.Indexed(v, (1,)), gem.Product(gem.Literal(-1), gem.Indexed(v, (0,))), gem.Zero()], mats
        else:
            raise NotImplementedError("Unexpected original mapping!")
            assert False, "Unexpected form degree combination!"

    def flatten(self):
        return HDiv(self.base_element.flatten())

    def unflatten(self):
        return HDiv(self.base_element.unflatten())


class HCurl(TensorProductTriple):

    def __init__(self, tensor_element):
        self.base_element = tensor_element
        self.gem_transformer, self.mat_transformer = self.select_fuse_hcurl_transformer(tensor_element)
        self.trace = TrHCurl
        super(HCurl, self).__init__(*tensor_element.factors, flat=tensor_element.flat, symmetric=tensor_element.requested_symmetric, matrices=tensor_element.apply_matrices)
        self.spaces = (self.spaces[0], CellHCurl(self.cell), self.spaces[2])

    def to_ufl(self):
        return HCurlElement(super(HCurl, self).to_ufl(), self.gem_transformer)

    def repr(self):
        return "HCurl(" + super(HCurl, self).repr() + ")"

    def select_fuse_hcurl_transformer(self, element):
        import gem
        # Assume: something x interval
        assert len(element.sub_elements) == 2
        assert element.sub_elements[1].cell.get_shape() == 1

        dim = element.cell.get_spatial_dimension()
        ks = tuple(fe.form_degree for fe in element.sub_elements)
        dims = tuple(fe.cell.get_spatial_dimension() for fe in element.sub_elements)
        transform = lambda cell, o: compute_matrix_transform(self.trace, cell, o)
        if all(str(fe.spaces[1]) == "H1" or str(fe.spaces[1]) == "L2" for fe in element.sub_elements) and dims == (1, 1):  # affine mapping, both factors 1D intervals (2D quad case)
            if ks == (1, 0):
                # Can only be 2D.  Make the scalar value the
                # tangential following the cell edge direction on the x-aligned edges.
                cell = element.sub_elements[0].cell
                bv = element.sub_elements[0].cell.basis_vectors()[0][0]
                mats = lambda m_a, m_b, o: np.kron(transform(cell, o[0]) * m_a, m_b)
                return lambda v: [gem.Product(gem.Literal(bv), v), gem.Zero()], mats
            elif ks == (0, 1):
                # Can be any spatial dimension.  Make the scalar value the
                # tangential following the cell edge direction .
                cell = element.sub_elements[1].cell
                bv = element.sub_elements[1].cell.basis_vectors()[0][0]
                mats = lambda m_a, m_b, o: np.kron(m_a, transform(cell, o[1]) * m_b)
                return lambda v: [gem.Zero()] * (dim - 1) + [gem.Product(gem.Literal(bv), v)], mats
            else:
                assert False
        elif ks == (1, 0) and dims == (2, 1) and str(element.sub_elements[0].spaces[1]) == "HCurl":
            # First factor is an already H(curl)-wrapped 2D element (an
            # in-plane tangential edge component), second is a CG interval
            mats = lambda m_a, m_b, o: np.kron(m_a, m_b)
            return lambda v: [gem.Indexed(v, (0,)), gem.Indexed(v, (1,)), gem.Zero()], mats
        elif ks == (0, 1) and dims == (2, 1) and str(element.sub_elements[0].spaces[1]) == "H1":
            # First factor is a plain (unwrapped) bilinear (Q1) scalar
            # element on a 2D base cell, second is a DG interval
            cell = element.sub_elements[1].cell
            mats = lambda m_a, m_b, o: np.kron(m_a, transform(cell, o[1]) * m_b)
            return lambda v: [gem.Zero(), gem.Zero(), v], mats
        else:
            raise NotImplementedError("Unexpected original mapping!")
            assert False, "Unexpected original mapping!"

    def flatten(self):
        return HCurl(self.base_element.flatten())

    def unflatten(self):
        return HCurl(self.base_element.unflatten())
