"""Tests for kernel actions: how a symmetry or an immersion transforms a kernel's value."""
import numpy as np
import pytest
import sympy as sp

from fuse import (polygon, make_tetrahedron, PointKernel, PolynomialKernel, VectorKernel, DOF, L2Pairing,
                  DOFGenerator, ElementTriple, immerse, PolynomialSpace, S1, C3, C0, H1, L2, HCurl, HDiv,
                  TrHCurl, TrHDiv, FormAction, SymmetricTensorAction)
from fuse.actions import as_action

x, y = sp.symbols("x y")


@pytest.fixture
def tri():
    return polygon(3)


@pytest.fixture
def true_action(monkeypatch):
    """The form action with the orientation sign on the kernel's own entity."""
    monkeypatch.setattr(FormAction, "legacy_orientation_sign", False)


def members(entity):
    return entity.group.members()


def action_matrix(entity, g):
    return PolynomialKernel(1).act(g).action_matrix(entity)


def reflections(entity):
    return [g for g in members(entity) if np.linalg.det(action_matrix(entity, g)) < 0]


# ------------------------------------------------------------------ choosing an action

def test_shorthands():
    assert as_action(H1) == FormAction(0, "scalar")
    assert as_action(L2) == FormAction(None, "scalar")
    assert as_action(HCurl) == FormAction(1, "flat")
    assert as_action(HDiv) == FormAction(None, "hodge")
    action = SymmetricTensorAction()
    assert as_action(action) is action
    assert as_action(None) is None
    with pytest.raises(ValueError):
        as_action("HDiv")
    with pytest.raises(ValueError):
        FormAction(1, "sharp")


def test_defaults():
    """Scalar kernels default to H1. A vector kernel must declare its action:
    without one it is assumed HCurl, with a deprecation warning."""
    assert PolynomialKernel(x).action == as_action(H1)
    assert PointKernel((0.0,)).action == as_action(H1)
    assert PolynomialKernel((x, y), action=HDiv).action == as_action(HDiv)
    with pytest.warns(DeprecationWarning):
        k = VectorKernel((1, 0))
    assert k.action == as_action(HCurl)


@pytest.mark.parametrize("action,dim,rank,k", [
    (HDiv, 1, 0, 1),     # a flux density on an edge
    (HDiv, 2, 0, 2),     # a flux density on a face
    (HDiv, 2, 1, 1),     # a vector proxy of a 1-form in 2D
    (HDiv, 3, 1, 2),     # a vector proxy of a 2-form in 3D
    (HCurl, 1, 0, 1),    # a tangential density on an edge
    (HCurl, 3, 1, 1),
    (L2, 2, 0, 2),
    (H1, 2, 0, 0),
])
def test_resolve_fills_in_k(tri, action, dim, rank, k):
    entity = {1: tri.edges()[0], 2: tri, 3: make_tetrahedron()}[dim]
    assert as_action(action).resolve(entity, rank).k == k


@pytest.mark.parametrize("action,rank", [(HCurl, 0), (H1, 1), (L2, 1)])
def test_resolve_rejects(tri, action, rank):
    """A scalar on a triangle is not a 1-form, and a vector is not a 0-form."""
    with pytest.raises(ValueError):
        as_action(action).resolve(tri, rank)


# ------------------------------------------------------------------ action of symmetries

@pytest.mark.parametrize("action,sign", [(H1, 1), (HCurl, -1), (HDiv, -1), (L2, -1)])
def test_interval_reflection(tri, true_action, action, sign):
    """On an interval, a moment of a 0-form is kept by the reflection; a moment
    of a 1-form (whether read flat or hodge, which agree on an interval) changes
    sign."""
    edge = tri.edges()[0]
    A = action_matrix(edge, reflections(edge)[0])
    a = as_action(action).resolve(edge, 0)
    assert np.isclose(a.act(A, 1.0), sign)


def test_legacy_orientation_sign(tri):
    """While the legacy sign is in place, a 1-form moment on an interval is
    kept by the reflection, as a 0-form would be."""
    edge = tri.edges()[0]
    A = action_matrix(edge, reflections(edge)[0])
    assert np.isclose(as_action(HCurl).resolve(edge, 0).act(A, 1.0), 1)


def test_flat_and_hodge_differ_by_orientation(tri, true_action):
    """On the triangle, the same vector read flat and hodge transforms the same
    way under rotations and with opposite signs under reflections."""
    v = np.array([0.3, -0.7])
    flat = as_action(HCurl).resolve(tri, 1)
    hodge = as_action(HDiv).resolve(tri, 1)
    for g in members(tri):
        A = action_matrix(tri, g)
        assert np.allclose(hodge.act(A, v), np.sign(np.linalg.det(A)) * flat.act(A, v))


@pytest.mark.parametrize("action", [HCurl, HDiv])
def test_form_action_composes(tri, true_action, action):
    a = as_action(action).resolve(tri, 1)
    v = np.array([0.3, -0.7])
    for g in members(tri):
        for h in members(tri):
            Ag, Ah = action_matrix(tri, g), action_matrix(tri, h)
            assert np.allclose(a.act(Ah @ Ag, v), a.act(Ah, a.act(Ag, v)))


def test_symmetric_tensor_action(tri):
    """A different action plugs into the same interface: a symmetric matrix
    transforms by A V A^T, which stays symmetric and composes."""
    a = SymmetricTensorAction()
    V = np.array([[2.0, 0.5], [0.5, 1.0]])
    for g in members(tri):
        for h in members(tri):
            Ag, Ah = action_matrix(tri, g), action_matrix(tri, h)
            W = a.act(Ag, V)
            assert np.allclose(W, W.T)
            assert np.allclose(a.act(Ah @ Ag, V), a.act(Ah, W))


# ------------------------------------------------------------------ lifts

def test_edge_lifts(tri):
    """A scalar on an edge lifts to its tangent when flat and to its normal
    when hodge."""
    edge = tri.edges()[0]
    t = as_action(HCurl).resolve(edge, 0).lift_matrix(tri, edge, 0).ravel()
    n = as_action(HDiv).resolve(edge, 0).lift_matrix(tri, edge, 0).ravel()
    assert np.isclose(np.dot(t, n), 0)
    assert np.isclose(np.linalg.norm(t), np.linalg.norm(n))
    assert as_action(H1).resolve(edge, 0).lift_matrix(tri, edge, 0) is None


def test_hodge_needs_codimension_one():
    """A flux density on an edge has no normal in a tetrahedron."""
    tet = make_tetrahedron()
    edge = tet.edges()[0]
    with pytest.raises(ValueError):
        as_action(HDiv).resolve(edge, 0).lift_matrix(tet, edge, 0)


# ------------------------------------------------------------------ immersion

def edge_triple(tri, action=None):
    edge = tri.edges()[0]
    xs = [DOF(L2Pairing(), PolynomialKernel(1, action=action))]
    return ElementTriple(edge, (PolynomialSpace(0), C0, None), DOFGenerator(xs, S1, S1))


def test_trace_must_match_action(tri):
    xs = [immerse(tri, edge_triple(tri, HDiv), TrHCurl)]
    with pytest.raises(ValueError):
        ElementTriple(tri, (PolynomialSpace(1, shape=2), C0, None), DOFGenerator(xs, C3, S1)).generate()


def test_trace_without_action_is_deprecated(tri):
    with pytest.warns(DeprecationWarning):
        immerse(tri, edge_triple(tri), TrHDiv)


@pytest.mark.parametrize("action,trace", [(HCurl, TrHCurl), (HDiv, TrHDiv)])
def test_declared_matches_trace(tri, action, trace):
    """Declaring the action gives the same DOFs as taking it from the trace."""
    space = PolynomialSpace(1, shape=2)

    def nodes(triple):
        xs = [immerse(tri, triple, trace)]
        dofs = ElementTriple(tri, (space, C0, None), DOFGenerator(xs, C3, S1)).generate()
        return [dof.to_quadrature(1, (2,)) for dof in dofs]

    with pytest.warns(DeprecationWarning):
        inferred = nodes(edge_triple(tri))
    for a, b in zip(nodes(edge_triple(tri, action)), inferred):
        assert a.keys() == b.keys()
        assert all(np.allclose([w for w, _ in a[p]], [w for w, _ in b[p]]) for p in a)


# ------------------------------------------------------------------ composition of kernel symmetries

def test_point_kernel_action_composes(tri):
    """Acting on a point kernel by g and then by h moves the point by h o g."""
    k = PointKernel((0.1, 0.2))
    for g in members(tri):
        for h in members(tri):
            assert np.allclose(k.act(g).act(h).pt, h.compose(g)(k.pt))


def test_kernel_action_composes(tri):
    """The same law for kernels that carry an orientation: acting by g and then
    by h is acting by h o g, and the action matrices multiply in that order."""
    k = PolynomialKernel((x, y), action=HCurl)
    for g in members(tri):
        for h in members(tri):
            kgh = k.act(g).act(h)
            assert kgh.orientation.perm == h.compose(g).perm
            assert np.allclose(kgh.action_matrix(tri),
                               k.act(h).action_matrix(tri) @ k.act(g).action_matrix(tri))
