"""Check that hypercube orientation matrices are genuine group representations.

An entity's orientation matrices describe how its DOFs transform under the
entity's symmetry group, so they must satisfy ``M[g] @ M[h] == M[g*h]``. That
is a self-contained criterion -- it needs no reference implementation -- and it
catches both a missing axis-permutation orientation (left as the identity) and
one built from the wrong DOF permutation.

These are lightweight unit tests: no Firedrake, only ``fuse`` and FIAT.
"""
import itertools
import numpy as np
import pytest
from FIAT.reference_element import UFCInterval
from FIAT.orientation_utils import (make_entity_permutations_simplex,
                                    make_entity_permutations_tensorproduct)

from fuse.element_construction import (periodic_table, construct_interval_cgN,
                                       construct_interval_dgN_integral)
from fuse.tensor_products import tensor_product, symmetric_tensor_product


# Quad builds are cheap; every (k, deg) is worth covering.
QUAD_PARAMS = [(k, deg) for k in range(4) for deg in (1, 2, 3)]

# Hex builds take well over a minute each, so only the cases that carry
# signal are listed. Entities with at most one DOF give identity matrices,
# which satisfy the criterion vacuously -- that rules out hex CG1/CG2 and
# hex RT1/ND1. The first degree with a multi-DOF entity is 3 for the scalar
# families and 2 for the vector ones.
HEX_PARAMS = [(0, 3), (3, 2), (1, 2), (2, 2)]


def homomorphism_failures(elem, dim, ent_id):
    """Count ``(g, h)`` pairs where ``M[g] @ M[h] != M[g*h]``."""
    entity = elem.cell.d_entities(dim)[ent_id]
    mats = elem.matrices[dim][ent_id]
    members = entity.group.members()
    bad, total = 0, 0
    for g in members:
        for h in members:
            keys = (g.numeric_rep(), h.numeric_rep(), (g * h).numeric_rep())
            if any(k not in mats for k in keys):
                continue
            total += 1
            if not np.allclose(mats[keys[0]] @ mats[keys[1]], mats[keys[2]]):
                bad += 1
    return bad, total


def assert_is_representation(elem):
    dim = elem.cell.get_spatial_dimension()
    checked = 0
    for d in range(dim + 1):
        for ent_id, dofs in elem.entity_dofs[d].items():
            if len(dofs) == 0:
                continue
            bad, total = homomorphism_failures(elem, d, ent_id)
            assert bad == 0, "dim %d entity %d: %d/%d products wrong" % (d, ent_id, bad, total)
            checked += total
    assert checked > 0


def regrouped_positions(elem):
    """Where each generated DOF ends up in the reindexed matrices.

    Read from the element rather than recomputed here: these tests check the
    matrices against ``entity_dofs``, and a second copy of the ordering rule
    would silently disagree the moment the real one changed.
    """
    n = sum(len(dofs) for ents in elem.entity_dofs.values() for dofs in ents.values())
    grouped = getattr(elem, "closure_order", list(range(n)))
    return {gen: pos for pos, gen in enumerate(grouped)}


@pytest.mark.parametrize("k,deg", QUAD_PARAMS)
def test_quad_orientation_matrices_are_representations(k, deg):
    assert_is_representation(periodic_table(2, 2, k, deg))


@pytest.mark.slow
@pytest.mark.parametrize("k,deg", HEX_PARAMS)
def test_hex_orientation_matrices_are_representations(k, deg):
    assert_is_representation(periodic_table(2, 3, k, deg))


def assert_entity_blocks(elem):
    """An entity's matrices may only mix that entity's own DOFs.

    Anything else means a block was written at the wrong offset -- which is
    what happens when matrices reindexed into Firedrake's closure order are
    paired with ``entity_dofs``, still in generation order.
    """
    positions = regrouped_positions(elem)
    for dim, ents in elem.entity_dofs.items():
        if dim == 0:
            continue
        for ent_id, dofs in ents.items():
            if len(dofs) == 0:
                continue
            own = [positions[d] for d in dofs]
            outside = [i for i in range(len(positions)) if i not in own]
            for key, mat in elem.matrices[dim][ent_id].items():
                block = mat[np.ix_(outside, outside)]
                assert np.allclose(block, np.eye(len(outside))), \
                    "dim %d entity %d orientation %d touches other entities' DOFs" % (dim, ent_id, key)


@pytest.mark.parametrize("k,deg", QUAD_PARAMS)
def test_quad_matrices_respect_entity_blocks(k, deg):
    assert_entity_blocks(periodic_table(2, 2, k, deg))


@pytest.mark.slow
@pytest.mark.parametrize("k,deg", HEX_PARAMS)
def test_hex_matrices_respect_entity_blocks(k, deg):
    assert_entity_blocks(periodic_table(2, 3, k, deg))


@pytest.mark.parametrize("d,deg", [
    (2, 2), (2, 3), (3, 3),
    pytest.param(2, 4, marks=pytest.mark.xfail(
        strict=True,
        reason="pre-existing: the interval element reflects 3+ interior nodes "
               "by pairing them ([1,0,2]) rather than reversing ([2,1,0]), so "
               "every product built from it inherits the wrong reflection")),
])
def test_cell_interior_matches_fiat_tensorproduct(d, deg):
    """Pin the interior block against FIAT's own tensor-product permutations.

    The homomorphism criterion only checks self-consistency, so it cannot
    detect a convention that is uniformly wrong. FIAT's
    ``make_entity_permutations_tensorproduct`` is an independent source for
    exactly the same object: how the interior nodes of an interval product
    are permuted by each orientation.
    """
    elem = periodic_table(2, d, 0, deg)
    positions = regrouped_positions(elem)
    interior = [positions[i] for i in elem.entity_dofs[d][0]]
    assert len(interior) == (deg - 1) ** d

    o_p_maps = [make_entity_permutations_simplex(1, deg - 1)] * d
    tuple_perm_map = make_entity_permutations_tensorproduct(
        [UFCInterval()] * d, [deg - 1] * d, o_p_maps)

    mats = elem.matrices[d][0]
    for tup, perm in tuple_perm_map.items():
        eo, flips = tup[0], tup[1:]
        key = (2 ** d) * eo + sum(b * 2 ** (d - 1 - i) for i, b in enumerate(flips))
        assert key in mats
        expected = np.eye(len(perm))[list(perm)]
        block = mats[key][np.ix_(interior, interior)]
        assert np.allclose(block, expected), \
            "orientation %d (eo=%d, flips=%s) disagrees with FIAT" % (key, eo, flips)


def test_symmetry_is_derived_not_assumed():
    """A product of unequal factors is not closed under the axis swap."""
    cg = construct_interval_cgN(2)
    dg = construct_interval_dgN_integral(1)

    asymmetric = tensor_product(cg, dg).flatten()
    assert asymmetric.symmetric is False

    assert tensor_product(cg, cg).flatten().symmetric is True


def test_declared_symmetry_is_checked():
    cg = construct_interval_cgN(2)
    dg = construct_interval_dgN_integral(1)
    with pytest.raises(NotImplementedError):
        symmetric_tensor_product(cg, dg).flatten()


def test_asymmetric_element_rejected_by_fiat():
    """Flattening stays permissive; handing the result to FIAT does not.

    The hex H(div)/H(curl) constructions legitimately build non-symmetric
    flat pieces to use as factors, so the rejection belongs at the boundary
    where every orientation must actually be supplied.
    """
    cg = construct_interval_cgN(2)
    dg = construct_interval_dgN_integral(1)
    elem = tensor_product(cg, dg).flatten()
    with pytest.raises(NotImplementedError):
        elem.to_fiat()


@pytest.mark.parametrize("k,deg", QUAD_PARAMS)
def test_quad_constructors_are_symmetric(k, deg):
    assert periodic_table(2, 2, k, deg).symmetric is True


def test_quad_axis_swap_crosses_enriched_components():
    """RT's interior axis swap must map one component onto the other.

    Each summand of RT_k on a quad carries interior DOFs along a single
    axis, so the swap sends every DOF of one summand to a DOF of the other.
    A within-summand permutation (such as treating the block as a square
    grid) cannot do that.
    """
    elem = periodic_table(2, 2, 2, 2)
    positions = regrouped_positions(elem)
    interior = [positions[i] for i in elem.entity_dofs[2][0]]
    assert len(interior) == 4

    # Key 4 is the pure axis swap: eo == 1, no reflections. Swapping two axes
    # is an odd permutation, so it reverses orientation and an H(div) DOF
    # changes sign on top of being moved.
    swap = elem.matrices[2][0][4][np.ix_(interior, interior)]
    half = len(interior) // 2
    expected = -np.block([[np.zeros((half, half)), np.eye(half)],
                          [np.eye(half), np.zeros((half, half))]])
    assert np.allclose(swap, expected)


def test_leaf_key_transport_covers_every_axis_permutation():
    """Every axis permutation of a hex cell block resolves to a real DOF."""
    elem = periodic_table(2, 2, 0, 3)
    assert not elem._closure_failures
    # All 2**d * d! orientations are present on the cell entity.
    d = elem.cell.get_spatial_dimension()
    expected = 2 ** d * len(list(itertools.permutations(range(d))))
    assert len(elem.matrices[d][0]) == expected
