"""Checks of the orientation data, using FUSE's own machinery.

Firedrake transforms the DOFs on an entity shared by two cells with
``matrices[dim][entity][key]``. The two cells agree when these matrices are
the action of the cell symmetries on the DOFs, and the keys name the
symmetries Firedrake means by them. So it is enough to check that

* ``mapping(g)`` is the action of ``g`` on the DOFs, and is a representation;
* each entity's matrices form a representation of its symmetry group, start
  from the identity, and are inverted by ``reversed_matrices``;
* ``orientation_key_to_member`` follows Firedrake's key formula.

No Firedrake import: this module is a smoke module.
"""
import itertools
from math import factorial

import numpy as np
import pytest
from sympy.combinatorics import Permutation

from fuse import polygon, make_tetrahedron
from fuse.cells import ufc_triangle, ufc_tetrahedron
from fuse.element_construction import periodic_table

PARAMS = ([(col, dim, 3, 0) for dim in (2, 3) for col in (0, 1)]
          + [(col, 2, k, deg) for deg in range(1, 7) for k in range(4) for col in (0, 1)]
          + [(col, 3, k, deg) for deg in range(1, 5) for k in range(4) for col in (0, 1)])


def _id(p):
    return "c{}-d{}-k{}-deg{}".format(*p)


_elements = {}


def element(p):
    if p not in _elements:
        el = periodic_table(*p)
        el.to_fiat()
        _elements[p] = el
    return _elements[p]


@pytest.mark.parametrize("p", [(0, 2, 1, 2), (1, 2, 2, 3), (0, 3, 1, 3)], ids=_id)
def test_mapping_is_the_action_on_the_dofs(p):
    """M_g from the generators equals the action of g on every DOF: the image
    of DOF p, evaluated on a basis, is column p of M_g applied to the DOFs."""
    el = periodic_table(*p)
    dofs = el.generate()
    V, cell = el.spaces[0], el.cell
    F = V.functional_matrix(cell, dofs)
    for g in cell.group.members():
        assert np.allclose(el.mapping(g).T @ F, V.functional_matrix(cell, [d(g) for d in dofs]), atol=1e-8)


@pytest.mark.parametrize("p", [(0, 2, 1, 3), (1, 2, 2, 3), (0, 3, 1, 2), (1, 3, 2, 2)], ids=_id)
def test_mapping_is_a_representation(p):
    el = periodic_table(*p)
    el.generate()
    members = el.cell.group.members()
    M = {g.perm: el.mapping(g) for g in members}
    assert np.allclose(M[el.cell.group.identity.perm], np.eye(len(el.generate())))
    for g in members:
        for h in members:
            assert np.allclose(M[h.compose(g).perm], M[h.perm] @ M[g.perm], atol=1e-8)


@pytest.mark.parametrize("p", PARAMS, ids=_id)
def test_orientation_matrices(p):
    """On each entity: key 0 is the identity, every matrix is inverted by its
    reversed matrix, and the matrices compose as the entity's symmetries do."""
    el = element(p)
    n = len(el.generate())
    min_ids = el.cell.get_starter_ids()
    for dim in el.matrices:
        for entity in el.cell.d_entities(dim):
            e_id = entity.id - min_ids[dim]
            mats, inverses = el.matrices[dim][e_id], el.reversed_matrices[dim][e_id]
            assert np.allclose(mats[0], np.eye(n))
            for key in mats:
                assert np.allclose(inverses[key] @ mats[key], np.eye(n), atol=1e-8)
            key_of = {entity.orientation_key_to_member(k).perm: k for k in mats}
            for k1, k2 in itertools.product(mats, repeat=2):
                h1, h2 = entity.orientation_key_to_member(k1), entity.orientation_key_to_member(k2)
                # matrices act on rows, so they compose in the opposite order to the symmetries
                assert np.allclose(mats[key_of[h1.compose(h2).perm]], mats[k2] @ mats[k1], atol=1e-8), (dim, e_id, k1, k2)


# ------------------------------------------------------------------ Firedrake's orientation keys

def firedrake_key(fiat_cone, plex_cone):
    """Firedrake's ``_compute_orientation_simplex``: the Lehmer code of the
    FIAT cone located within the DMPlex cone."""
    cone = list(plex_cone)
    inds = []
    for q in fiat_cone:
        k = cone.index(q)
        inds.append(k)
        cone.pop(k)
    n = len(inds)
    return sum(factorial(n - 1 - i) * inds[i] for i in range(n))


def key_reorientations(ref_el, dim, e_id):
    """{Firedrake key k: the reorientation it stands for} for an entity of the
    FIAT cell ``ref_el``, as a permutation of the entity's local vertex
    positions, worked out from the key formula alone."""
    top, conn = ref_el.get_topology(), ref_el.connectivity
    loc = list(top[dim][e_id])
    nv = len(loc)
    cone = [(v,) for v in loc] if dim == 1 else [tuple(top[dim - 1][s]) for s in conn[(dim, dim - 1)][e_id]]

    def global_cone(s):
        return [frozenset(s[loc.index(v)] for v in c) for c in cone]

    by_key = {}
    plex_cone = global_cone(tuple(range(nv)))
    for s in itertools.permutations(range(nv)):
        by_key.setdefault(firedrake_key(global_cone(s), plex_cone), []).append(s)
    s0 = by_key[0][0]
    return {k: tuple((~Permutation([ss[0].index(s0[i]) for i in range(nv)])).array_form) for k, ss in by_key.items()}


def test_firedrake_key_docstring_example():
    # the example in firedrake's _compute_orientation_simplex
    assert firedrake_key([32, 33], [33, 32]) == 1
    assert firedrake_key([32, 33], [32, 33]) == 0


@pytest.mark.parametrize("make_cell", [polygon, ufc_triangle, make_tetrahedron, ufc_tetrahedron],
                         ids=["triangle", "ufc_triangle", "tetrahedron", "ufc_tetrahedron"])
def test_orientation_keys_follow_firedrake(make_cell):
    """Each entity converts Firedrake's orientation keys to the symmetries
    Firedrake's key formula stands for on the FIAT cell."""
    cell = make_cell(3) if make_cell is polygon else make_cell()
    ref_el = cell.to_fiat()
    for dim in range(1, cell.dimension):
        for e_id, entity in enumerate(cell.d_entities(dim)):
            for k, h in key_reorientations(ref_el, dim, e_id).items():
                assert tuple(entity.orientation_key_to_member(k).perm.array_form) == h
