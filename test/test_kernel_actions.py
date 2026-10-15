"""Tests for kernel actions: how a symmetry or an immersion transforms a kernel's value."""
import itertools

import numpy as np
import pytest
import sympy as sp

from fuse import (polygon, make_tetrahedron, PointKernel, PolynomialKernel, VectorKernel, BarycentricPolynomialKernel,
                  DOF, L2Pairing, DOFGenerator, ElementTriple, immerse, PolynomialSpace, S1, C3, C0, H1, L2, HCurl, HDiv,
                  TrH1, TrHCurl, TrHDiv, FormAction, SymmetricTensorAction)
from fuse.actions import as_action

x, y = sp.symbols("x y")


@pytest.fixture
def tri():
    return polygon(3)


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
def test_interval_reflection(tri, action, sign):
    """On an interval, a moment of a 0-form is kept by the reflection; a moment
    of a 1-form (whether read flat or hodge, which agree on an interval) changes
    sign."""
    edge = tri.edges()[0]
    A = action_matrix(edge, reflections(edge)[0])
    a = as_action(action).resolve(edge, 0)
    assert np.isclose(a.act(A, 1.0), sign)


def test_flat_and_hodge_differ_by_orientation(tri):
    """On the triangle, the same vector read flat and hodge transforms the same
    way under rotations and with opposite signs under reflections."""
    v = np.array([0.3, -0.7])
    flat = as_action(HCurl).resolve(tri, 1)
    hodge = as_action(HDiv).resolve(tri, 1)
    for g in members(tri):
        A = action_matrix(tri, g)
        assert np.allclose(hodge.act(A, v), np.sign(np.linalg.det(A)) * flat.act(A, v))


@pytest.mark.parametrize("action", [HCurl, HDiv])
def test_form_action_composes(tri, action):
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
    return ElementTriple(edge, (PolynomialSpace(0), C0, None), DOFGenerator(xs, S1))


def test_trace_must_match_action(tri):
    with pytest.raises(ValueError):
        immerse(tri, edge_triple(tri, HDiv), TrHCurl)


def test_trace_without_action_is_deprecated(tri):
    with pytest.warns(DeprecationWarning):
        immerse(tri, edge_triple(tri), TrHDiv)


@pytest.mark.parametrize("action,trace", [(HCurl, TrHCurl), (HDiv, TrHDiv)])
def test_declared_matches_trace(tri, action, trace):
    """Declaring the action gives the same DOFs as taking it from the trace."""
    space = PolynomialSpace(1, shape=2)

    def nodes(triple):
        xs = [immerse(tri, triple, trace)]
        dofs = ElementTriple(tri, (space, C0, None), DOFGenerator(xs, C3)).generate()
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


@pytest.mark.parametrize("value,action", [((1.0, 2.0), HCurl), ((1.0, 2.0), HDiv), (1.0, L2)])
def test_vector_kernel_acts_like_a_constant_polynomial(tri, value, action):
    """A constant vector (or scalar) kernel transforms exactly as the same
    constant written as a polynomial kernel."""
    Qpts, Qwts = tri.quadrature(1)
    bary = tri.cartesian_to_barycentric(Qpts)
    shape = (2,) if np.ndim(value) else ()
    s = sp.symbols("s_0 s_1 s_2")
    for g in members(tri):
        kv = VectorKernel(value, action=action).act(g)
        kp = BarycentricPolynomialKernel(value, symbols=s, action=action).act(g)
        A = kv.action_matrix(tri)
        act = kv.action.resolve(tri, kv.rank)
        _, wv, cv = kv.evaluate(Qpts, Qwts, A, act, 2, shape)
        _, wp, cp = kp.evaluate(Qpts, bary, Qwts, A, act, 2, shape)
        assert np.allclose(wv, wp) and cv == cp


# ------------------------------------------------------------------ traces are dual to lifts

@pytest.mark.parametrize("cell_fn", [lambda: polygon(3), make_tetrahedron], ids=["triangle", "tetrahedron"])
def test_lift_is_dual_to_trace(cell_fn):
    """<lift(k), v> in the cell equals <k, tr(v)> on the entity, and the trace
    is the field itself for H1, its tangential component on an edge for HCurl
    and its normal component on a facet for HDiv."""
    cell = cell_fn()
    rng = np.random.default_rng(0)
    v = rng.standard_normal(cell.dimension)
    kappa = rng.standard_normal()
    for action, dim in [(H1, 0), (HCurl, 1), (HDiv, cell.dimension - 1)]:
        for entity in cell.d_entities(dim):
            a = as_action(action).resolve(entity, 0)
            restricted = a.trace(cell, entity, 0, v)
            if action is H1:
                expected = v
            else:
                basis = np.array(cell.basis_vectors(entity=entity))
                if action is HCurl:
                    direction = basis[0]
                elif cell.dimension == 2:
                    direction = basis[0] @ np.array([[0, -1], [1, 0]])
                else:
                    direction = np.cross(basis[0], basis[1])
                expected = [direction @ v]
            assert np.allclose(np.ravel(restricted), np.ravel(expected))
            L = a.lift_matrix(cell, entity, 0)
            if L is not None:
                assert np.isclose(np.dot(np.ravel(L) * kappa, v), kappa * np.ravel(restricted)[0])


def test_flat_vector_on_face_restricts_to_tangential_part():
    """A flat vector on a face of a tetrahedron lifts into the plane of the face,
    so its trace forgets the normal component of a field."""
    tet = make_tetrahedron()
    face = tet.d_entities(2)[0]
    a = as_action(HCurl).resolve(face, 1)
    L = a.lift_matrix(tet, face, 1)
    n = np.cross(*np.array(tet.basis_vectors(entity=face)))
    assert np.allclose(n @ L, 0)
    assert np.allclose(a.trace(tet, face, 1, n), 0)


# ------------------------------------------------------------------ the trace is read from the kernels

def immersed_edge_dofs(tri, triple):
    space = PolynomialSpace(1, shape=2)
    return ElementTriple(tri, (space, C0, None), DOFGenerator([immerse(tri, triple)], C3)).generate()


def test_immersed_dofs_trace_through_their_kernels(tri):
    """Without a trace argument nothing is stored on the immersion: each DOF
    lifts along the tangent (HCurl) or the normal (HDiv) of its edge."""
    assert immerse(tri, edge_triple(tri, HCurl)).trace is None
    for action, normal in [(HCurl, False), (HDiv, True)]:
        for dof in immersed_edge_dofs(tri, edge_triple(tri, action)):
            tangent = np.array(tri.basis_vectors(entity=dof.cell_defined_on))[0]
            assert np.isclose(abs(np.dot(dof.lift_direction(), tangent)), 0 if normal else 1)


def test_immerse_with_redundant_trace(tri):
    with pytest.warns(DeprecationWarning):
        immerse(tri, edge_triple(tri, HDiv), TrHDiv)
    with pytest.raises(ValueError):
        immerse(tri, edge_triple(tri, HDiv), TrH1)


def test_mixed_actions_are_immersed_together(tri):
    """One immersed triple can hold kernels with different actions: each DOF
    is traced through its own kernel."""
    edge = tri.edges()[0]
    xs = [DOF(L2Pairing(), PolynomialKernel(1, action=HCurl)), DOF(L2Pairing(), PolynomialKernel(1, action=HDiv))]
    triple = ElementTriple(edge, (PolynomialSpace(1), L2, None), DOFGenerator(xs, S1))
    for dof in immersed_edge_dofs(tri, triple):
        tangent = np.array(tri.basis_vectors(entity=dof.cell_defined_on))[0]
        assert np.isclose(abs(np.dot(dof.lift_direction(), tangent)), 1 if dof.kernel.action == as_action(HCurl) else 0)


# ------------------------------------------------------------------ immersion respects the entity's symmetry

def affine_map(cell, g):
    """The affine map of g, x -> g(x), and its linear part."""
    M = np.array(g.transform_matrix, dtype=float)    # x' = [x, 1] @ M
    d = cell.dimension
    return (lambda x: np.r_[np.array(x, dtype=float), 1] @ M), M[:d, :].T


def apply_functional(quadrature, v):
    return sum(w * np.atleast_1d(v(pt))[c[0] if c else 0] for pt, wcs in quadrature.items() for w, c in wcs)


def random_field(rng, d, shape):
    """A random polynomial of degree 2, integrated exactly by the quadrature."""
    monomials = [e for e in itertools.product(range(3), repeat=d) if sum(e) <= 2]
    C = rng.standard_normal((len(monomials),) + shape)
    return lambda x: sum(c * np.prod([xi**ei for xi, ei in zip(x, e)]) for c, e in zip(C, monomials))


@pytest.mark.parametrize("p", [(0, 2, 0, 2), (0, 2, 1, 1), (0, 2, 1, 2), (0, 2, 2, 1), (0, 2, 2, 2),
                               (1, 2, 1, 2), (1, 2, 2, 1), (0, 3, 0, 3), (0, 3, 1, 1), (0, 3, 2, 1),
                               (1, 3, 1, 1), (0, 3, 1, 2), (0, 3, 2, 2), (1, 3, 2, 1), (1, 3, 2, 2)],
                         ids=lambda p: "c{}-d{}-k{}-deg{}".format(*p))
def test_immersed_dof_moves_with_the_cell(p):
    """Moving an immersed DOF by a symmetry g of the cell (which moves it to
    another entity, acted on by the induced symmetry of the entity) gives the
    functional v -> l(g*v): its points are moved by g and its values by the
    action of g on the lifted kernel."""
    from fuse.element_construction import periodic_table
    from fuse.dof import ImmersedDOF
    rng = np.random.default_rng(0)
    elem = periodic_table(*p)
    cell = elem.cell
    d = cell.dimension
    shape = () if p[2] in (0, 3) else (d,)
    for dof in elem.generate():
        if not isinstance(dof, ImmersedDOF):
            continue
        lifted = dof.kernel.action.resolve(dof.cell_defined_on, dof.kernel.rank).lifted()
        quadrature = dof.to_quadrature(3, shape)
        for g in members(cell):
            T, DT = affine_map(cell, g)
            A = np.linalg.inv(DT).T
            v = random_field(rng, d, shape)
            expected = 0.0
            for pt, wcs in quadrature.items():
                if shape:
                    w = np.zeros(d)
                    for wv, c in wcs:
                        w[c[0]] = wv
                    expected += np.dot(lifted.act(A, w), v(T(pt)))
                else:
                    expected += sum(wv for wv, _ in wcs) * v(T(pt))
            assert np.isclose(apply_functional(dof(g).to_quadrature(3, shape), v), expected, atol=1e-8), (dof, g)


# ------------------------------------------------------------------ serialisation

@pytest.mark.parametrize("kernel", [PolynomialKernel(1, action=HDiv), PolynomialKernel((x, y), action=HCurl),
                                    PointKernel((0.5,)), PolynomialKernel(x, action=SymmetricTensorAction())])
def test_kernel_action_serialises(kernel):
    from fuse.serialisation import ElementSerialiser
    converter = ElementSerialiser()
    decoded = converter.decode(converter.encode(kernel))
    assert decoded.declared_action == kernel.declared_action


@pytest.mark.parametrize("cell_fn", [lambda: polygon(3), make_tetrahedron], ids=["triangle", "tetrahedron"])
def test_induced_entity_symmetry(cell_fn):
    """A symmetry g of the cell sends an entity E to E' and induces a symmetry
    h of the reference entity with g o psi_E = psi_E' o h, where psi are the
    attachments of the entities in the cell."""
    cell = cell_fn()
    rng = np.random.default_rng(0)
    for dim in range(1, cell.dimension):
        for i, E in enumerate(cell.d_entities(dim)):
            psi_E = cell.attachment(cell.id, E.id)
            for g in members(cell):
                E_image, h = cell.permute_entities(g, dim)[i]
                psi_image = cell.attachment(cell.id, E_image)
                x = tuple(0.2 * rng.standard_normal(dim))
                assert np.allclose(np.array(g(psi_E(*x)), dtype=float), np.array(psi_image(*h(x)), dtype=float))
