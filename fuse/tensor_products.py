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


def one_dim_signature(elem):
    """What makes two one-dimensional elements interchangeable as axes.

    Axes may be swapped only between factors describing the same space. That
    cannot be decided by identity: an axis is often built by calling the same
    constructor twice, giving equal elements on distinct cells. Compare what
    determines the space instead -- its polynomial and Sobolev spaces, and how
    its DOFs are distributed over entities. The last of these is what
    separates, say, CG2 from DG2, whose spaces print the same but which place
    their DOFs differently.
    """
    entity_assoc = elem._entity_associations(elem.generate(), overall=False)[0]
    return (str(elem.spaces[0]), str(elem.spaces[1]), str(elem.spaces[2]),
            tuple(sorted((d, tuple(len(v) for v in ents.values()))
                         for d, ents in entity_assoc.items())))


def one_dim_dof_keys(elem, out=None):
    """Map each of ``elem``'s generated DOFs to a per-axis key.

    A tensor product DOF is a tuple with one component per factor, but a
    factor may itself be a product, so a component can be a nested tuple.
    Descending to the one-dimensional leaves gives every DOF a flat key with
    one entry per spatial axis, which is what an axis permutation acts on.

    Each axis contributes ``(signature, position, entity_dimension)`` rather
    than the DOF itself, so that axes built separately but describing the same
    space match -- while axes describing different spaces still do not,
    leaving such a product correctly asymmetric. The dimension travels along
    so callers can still tell which axes an entity extends along.
    """
    if out is None:
        out = {}
    from fuse.enriched import EnrichedElement
    if isinstance(elem, EnrichedElement):
        # Checked before TensorProductTriple, which it subclasses.
        one_dim_dof_keys(elem.A, out)
        one_dim_dof_keys(elem.B, out)
    elif isinstance(elem, TensorProductTriple):
        sub_keys = [one_dim_dof_keys(f) for f in elem.factors]
        for dof in elem.generate():
            out[dof] = sum((sub_keys[i][comp] for i, comp in enumerate(dof)), ())
    else:
        # for dof in elem.generate():
        #     out[dof] = (dof,)
        signature = one_dim_signature(elem)
        for position, dof in enumerate(elem.generate()):
            out[dof] = ((signature, position, dof.cell_defined_on.dim()),)
    return out


class TensorProductTriple(ElementTriple):

    # Axis permutations the DOF set is not closed under, recorded by
    # _fill_axis_permutations and read by _resolve_symmetry.
    _closure_failures = frozenset()

    def __init__(self, *factors, flat=False, symmetric=None, matrices=True):
        if len(factors) < 2:
            raise ValueError("Cannot create a tensor product with fewer than 2 factors")
        self.factors = factors
        (poly_a, wi_a, pullback_a) = A.spaces
        (poly_b, wi_b, pullback_b) = B.spaces
        if pullback_a != pullback_b:
            raise ValueError("Tensor product factors must share the same pullback.")
        self.spaces = [poly_a if poly_a >= poly_b else poly_b,
                       wi_a if wi_a >= wi_b else wi_b,
                       pullback_a]

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

        self.mat_transformer = getattr(self, "mat_transformer", None)
        self.trace = getattr(self, "trace", None)
        self.apply_matrices = matrices
        if self.apply_matrices:
            self.setup_matrices()

        self.pure_perm = not matrices

    @property
    def sub_elements(self):
        return self.factors

    @property
    def form_degree(self):
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
        # Reset closure_failures as we are constructing matrices
        self._closure_failures = set()

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
                        entity = self.cell.d_entities(total_dim)[self.ent_mapping[dim][sub_ents]]
                        self._fill_axis_permutations(entity, dim, ent_dofs, sub_mat, dof_keys, key_to_index)

        if self.flat:
            oriented_mats_by_entity = flatten_dictionary(oriented_mats_by_entity)

        self.matrices = oriented_mats_by_entity
        self.reversed_matrices = self.reverse_dof_perms(self.matrices)

        if self.flat:
            self._snapshot_generation_order()
            self._regroup_matrices()

        self._resolve_symmetry()

    def _resolve_symmetry(self):
        """Reconcile ``self.symmetric`` with observed matrix construction.
        """
        closed = not self._closure_failures
        if self.requested_symmetric is None:
            self.symmetric = closed
        elif self.requested_symmetric and not closed:
            raise NotImplementedError(
                "%r was declared symmetric but its DOFs are not closed under "
                "axis permutation %r" % (self, sorted(self._closure_failures)))

    def _orientation_value_change(self, entity, key):
        """Change of value an orientation induces on this element's DOFs.

        Reorienting an entity moves its basis vectors, and how a DOF responds
        depends on what it measures: an H(div) DOF integrates against a
        normal, an H(curl) DOF against a tangent, a scalar DOF against
        nothing. The trace knows that, so ask it rather than special-casing
        the family here.

        There are exactly three ways to have no change of value, and each is a
        positive statement rather than a failure to work one out:

        - the element is scalar valued, so no direction moves;
        - the trace measures nothing on an entity of this dimension (H(div)
          has no normal on a cell interior);
        - the direction has been carried onto a *different* DOF's rather than
          rescaled, so the transport permutation being built alongside already
          accounts for it. An H(curl) axis swap exchanges the two tangential
          components, and transport finds the exchanged DOF. H(div) is the
          opposite case: every DOF on a facet shares its normal, so the
          permutation has nowhere to put the flip and it surfaces here.
        """
        if self.trace is None:
            return 1
        member = entity.group.get_member_by_val(key)
        trace = self.trace(self.cell)

        def spanning(ent):
            # The vectors spanning the entity, in the containing cell's frame.
            return np.array(self.cell.basis_vectors(entity=ent), dtype=float)[:ent.dimension]

        # Guard the trace call alone: the shapes it declines are the second
        # case above. Failures in basis_vectors are bugs and must propagate.
        try:
            ref = np.asarray(trace.manipulate_basis(spanning(entity)), dtype=float).ravel()
            new = np.asarray(trace.manipulate_basis(spanning(entity.orient(~member))), dtype=float).ravel()
        except ValueError:
            return 1

        norm = ref.dot(ref)
        if norm == 0:
            raise ValueError("%r measures a degenerate direction on %r" % (trace, entity))
        ratio = new.dot(ref) / norm
        if not np.isclose(abs(ratio), 1.0):
            return 1
        # Only the sign carries information; snap to integer so the matrices stay exact
        # signed permutations under inversion and reindexing.
        return 1 if ratio > 0 else -1

    def _axis_key_maps(self, dofs):
        """Per-axis keys for ``dofs`` on one dimensional entities, indexed both ways.

        Returns ``(dof_keys, key_to_index)`` where ``dof_keys`` maps a
        DOF index to its one_dim key and ``key_to_index`` inverts that.
        """
        leaves = one_dim_dof_keys(self)
        dof_keys = {i: leaves[dof] for i, dof in enumerate(dofs)}
        key_to_index = {key: i for i, key in dof_keys.items()}
        return dof_keys, key_to_index

    def _snapshot_generation_order(self):
        """Keep a copy of the matrices indexed in generation DOF order.
        """
        self._gen_order_matrices = {dim: {e: {o: mat.copy() for o, mat in os.items()}
                                          for e, os in ents.items()}
                                    for dim, ents in self.matrices.items()}

    def _regroup_matrices(self):
        """Re-express the orientation matrices in closure DOF order.

        FUSE generates tensor-product and enriched DOFs in an interleaved
        order.
        Firedrake packs each cell's closure DOFs by entity dimension
        and, within a dimension, by entity number, and applies these matrices
        in that order.
        """
        grouped = [dof
                   for total_dim in sorted(self.entity_dofs)
                   for ent in sorted(self.entity_dofs[total_dim])
                   for dof in self.entity_dofs[total_dim][ent]]
        n = len(grouped)
        # Kept so callers can map a generated DOF to the row it ends up in,
        # rather than reproducing this ordering and drifting out of step.
        self.closure_order = grouped
        if grouped == list(range(n)):
            # already in order
            return
        ix = np.ix_(grouped, grouped)
        for mats in (self.matrices, self.reversed_matrices):
            for ents in mats.values():
                for os in ents.values():
                    for k in list(os.keys()):
                        os[k] = os[k][ix].copy()

    def _fill_axis_permutations(self, entity, dim, ent_dofs, sub_mat, dof_keys, key_to_index):
        """Populate the axis-permuting orientations of an entity.

        Orientations are keyed ``2**d * eo + io``, with ``eo`` the extrinsic
        orientation (which axis permutation) and ``io`` the intrinsic one
        (which axes are reflected) -- see
        ``fuse.utils.canonical_tensor_orientation_key``. The per-entity loop in
        ``setup_matrices`` reaches only the reflections, ``eo == 0``, because it
        enumerates products of the factors' own orientations and those cannot
        permute axes. This fills the rest, as

            ``M[2**d * eo + io] = M[io] @ (value_change * P_axis_perm)``

        ``P_axis_perm`` comes from the per-axis keys: permuting the key of a DOF
        names its image, so looking that key up gives the permutation directly,
        for any number of axes and across enriched summands. ``value_change``
        comes from the element's trace, and is owed only for the permutation
        itself since ``M[io]`` already carries the reflection's.

        An entity whose DOFs are not closed under some permutation is recorded
        in ``_closure_failures``, leaving those orientations as the identity;
        ``_resolve_symmetry`` reads that to decide whether the element is
        symmetric at all.
        """
        entity_dim = sum(dim) if isinstance(dim, tuple) else dim
        if entity_dim < 2 or not ent_dofs:
            # Points and intervals have no non-trivial axis permutations.
            return

        # These are the DOF keys in the order this entity's block uses.
        keys = [dof_keys[dof] for dof in ent_dofs]

        # Which of a key's axes this entity extends along, read from the DOFs
        # rather than from `dim` so that one code path serves hex cells, hex
        # faces, and factors that are themselves flattened quads. Key entries
        # are (signature, position, dimension); an entity extends along the
        # one dimensional ones.
        spanned_axes_set = {tuple(i for i, axis in enumerate(key) if axis[2] == 1) for key in keys}
        if len(spanned_axes_set) != 1:
            # The entity's DOFs disagree about which axes it extends along,
            # so there is no well-defined action to build.
            self._closure_failures.add((entity_dim, None))
            return

        spanned_axes = spanned_axes_set.pop()
        if len(spanned_axes) != entity_dim:
            # The keys describe a different-dimensional entity than the one
            # we are trying to permute.
            self._closure_failures.add((entity_dim, None))
            return

        local_index = {dof: i for i, dof in enumerate(ent_dofs)}
        grid = np.ix_(ent_dofs, ent_dofs)

        # ``eo == 0`` is the identity permutation, already filled by the
        # orientation loop above.
        for eo, axis_perm in enumerate(sorted(permutations(range(entity_dim)))):
            if eo == 0:
                continue

            transported = self._transport(keys, spanned_axes, axis_perm, key_to_index, local_index)
            if transported is None:
                self._closure_failures.add((entity_dim, eo))
                continue

            # Only the permutation's own value change is owed here; sub_mat[io]
            # already carries the reflection's. Hence the pure permutation key
            # 2**entity_dim * eo, NOT swap_key.
            value_change = self._orientation_value_change(entity, 2 ** entity_dim * eo)
            permutation_matrix = value_change * np.eye(len(ent_dofs))[transported]
            for io in range(2 ** entity_dim):
                swap_key = 2 ** entity_dim * eo + io
                if io in sub_mat and swap_key in sub_mat:
                    sub_mat[swap_key][grid] = np.matmul(sub_mat[io][grid], permutation_matrix)

    @staticmethod
    def _transport(keys, spanned_axes, axis_perm, key_to_index, local):
        """Return the DOF permutation induced by ``axis_perm``.

        ``None`` means the entity's DOFs are not closed under the permutation:
        either the permuted key does not exist at all, or it exists on a
        different entity and this local block cannot represent it.
        """
        permuted_indices = []
        for key in keys:
            # Start from the DOF's per-axis key and move only the axes this
            # entity actually spans.
            permuted_key = list(key)
            for source_pos, source_axis in enumerate(spanned_axes):
                target_axis = spanned_axes[axis_perm[source_pos]]
                permuted_key[target_axis] = key[source_axis]

            target_dof = key_to_index.get(tuple(permuted_key))
            if target_dof not in local:
                return None
            permuted_indices.append(local[target_dof])

        return permuted_indices

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
