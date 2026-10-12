"""Tests for subgroups, transversals, entity stabilisers and actions on spans."""
import numpy as np
import pytest

from fuse import polygon, make_tetrahedron, Point, C3, PermutationSetRepresentation
from fuse.groups import tet_edges, tet_faces


def linear_part(g):
    """Linear part Dg of the affine map of g (x' = [x, 1] @ M)."""
    M = np.asarray(g.transform_matrix, dtype=float)
    d = M.shape[1]
    return M[:d, :].T


def rotations_and_reflections(G):
    members = G.members()
    return ([g for g in members if g.perm.order() == 3],
            [g for g in members if g.perm.order() == 2])


@pytest.fixture
def tri():
    return polygon(3)


@pytest.fixture
def tet():
    return make_tetrahedron()


def test_compose_is_function_composition(tet):
    G = tet.group
    x = (0.1, 0.2, 0.3)
    for g in G.members():
        for h in G.members():
            assert np.allclose(g.compose(h)(x), g(h(x)))


@pytest.mark.parametrize("cell_fn,orders", [
    (lambda: Point(1, [Point(0), Point(0)], vertex_num=2), {0: 1, 1: 2}),
    (lambda: polygon(3), {0: 2, 1: 2, 2: 6}),
    (make_tetrahedron, {0: 6, 1: 4, 2: 6, 3: 24}),
])
def test_entity_stabiliser(cell_fn, orders):
    """A vertex of a triangle is fixed by the identity and one reflection; an
    edge of a tetrahedron by four symmetries; and so on."""
    cell = cell_fn()
    for dim, order in orders.items():
        for e in cell.d_entities(dim):
            H = cell.entity_stabiliser(e)
            assert H.size() == order
            assert cell.group.is_subgroup(H)


def test_reflection_reverses_rotations(tri):
    """In the symmetry group of the triangle, take H = {e, s} for a reflection s,
    and the rotations C3 as the transversal. Reflecting after a rotation is the
    same as rotating the other way and then reflecting: s o a = a^-1 o s. So the
    decomposition of s o a has a' = a^-1 and stabiliser part s."""
    G = tri.group
    A = C3.add_cell(tri)
    _, reflections = rotations_and_reflections(G)
    for s in reflections:
        H = [G.identity, s]
        assert G.is_left_transversal(A, H)
        for a in A.members():
            ap, h = A.decompose(s.compose(a), H)
            assert ap.perm == (~a).perm
            assert h.perm == s.perm


def test_rotation_transversal_has_constant_stabiliser_part(tri):
    """With the rotations as transversal, the stabiliser part of g o a depends
    only on g, not on a. This is what lets an orientation matrix be written as
    a permutation of blocks tensored with one fixed block."""
    G = tri.group
    A = C3.add_cell(tri)
    _, reflections = rotations_and_reflections(G)
    H = [G.identity, reflections[0]]
    for g in G.members():
        assert len({A.decompose(g.compose(a), H)[1].perm for a in A.members()}) == 1


def test_non_subgroup_transversal_has_varying_stabiliser_part(tri):
    """A transversal does not have to be a subgroup. {e, r o s, r o r} meets each
    coset of H = {e, s} once, but for four of the six symmetries g the
    stabiliser part of g o a changes with a, so no single block describes them."""
    G = tri.group
    rotations, reflections = rotations_and_reflections(G)
    s, r, e = reflections[0], rotations[0], G.identity
    H = [e, s]
    A = PermutationSetRepresentation([e.perm, r.compose(s).perm, r.compose(r).perm], cell=tri)
    assert G.is_left_transversal(A, H)
    assert not G.is_subgroup(A)
    varying = [g for g in G.members() if len({A.decompose(g.compose(a), H)[1].perm for a in A.members()}) > 1]
    assert len(varying) == 4


def test_not_a_transversal(tri):
    G = tri.group
    rotations, reflections = rotations_and_reflections(G)
    A = C3.add_cell(tri)
    assert not G.is_left_transversal(A, A)          # the rotations are not a transversal of themselves
    assert not G.is_subgroup([G.identity, rotations[0]])
    with pytest.raises(ValueError):
        A.decompose(reflections[0].compose(A.members()[1]), [G.identity, rotations[0]])


@pytest.mark.parametrize("dim,transversal", [(1, tet_edges), (2, tet_faces)])
def test_left_transversal_iff_entities_enumerated(tet, dim, transversal):
    """A set A is a left transversal of the stabiliser of an entity E exactly
    when a -> a(E) visits every entity of that dimension once."""
    G = tet.group
    A = transversal.add_cell(tet)
    ents = tet.d_entities(dim)

    def vertex_set(points):
        return frozenset(tuple(np.round(np.array(p, dtype=float).ravel(), 8)) for p in points)

    def coords(e):
        return [tet.get_node(v, return_coords=True) for v in e.ordered_vertices()]

    sets = [vertex_set(coords(e)) for e in ents]
    for e in ents:
        images = [vertex_set(a(tuple(np.array(x, dtype=float).ravel())) for x in coords(e)) for a in A.members()]
        enumerates = sorted(sets.index(i) for i in images) == list(range(len(ents)))
        assert G.is_left_transversal(A, tet.entity_stabiliser(e)) == enumerates


def test_span_of_one_vector(tri):
    """A vector v along an edge of the triangle, acted on by s -> Ds v. Only the
    identity and the reflection that reverses v keep its span, and that
    reflection acts on it by -1."""
    G = tri.group
    v = np.array(tri.basis_vectors()[0], dtype=float)

    def act(s):
        return (linear_part(s) @ v)[None, :]

    H = G.span_stabiliser(act)
    assert H.size() == 2
    assert G.is_subgroup(H)
    rho = H.span_representation(act)
    s = [h for h in H.members() if not h.perm.is_Identity][0]
    assert np.allclose(rho[tuple(s.perm.array_form)], [[-1.0]])
    assert np.allclose(linear_part(s) @ v, -v)


def test_span_of_two_vectors(tri):
    """Two independent vectors in the plane of the triangle span R^2, so every
    symmetry keeps the span, and the action is the 2D representation of the
    triangle's symmetries.
    This has trace 2 at the identity, 0 for reflections and -1
    for rotations, which is checked in this test
    The test also ensures that composition is respected."""
    G = tri.group
    B = np.array(tri.basis_vectors(), dtype=float)

    def act(s):
        return (linear_part(s) @ B.T).T

    H = G.span_stabiliser(act)
    assert H.size() == 6
    rho = H.span_representation(act)
    rotations, reflections = rotations_and_reflections(G)

    def key(g):
        return tuple(g.perm.array_form)

    assert np.isclose(np.trace(rho[key(G.identity)]), 2)
    assert all(np.isclose(np.trace(rho[key(s)]), 0) for s in reflections)
    assert all(np.isclose(np.trace(rho[key(r)]), -1) for r in rotations)
    for g in G.members():
        for h in G.members():
            assert np.allclose(rho[key(g.compose(h))], rho[key(g)] @ rho[key(h)])
