from fuse import *
from fuse.element_construction import (construct_tri_cgN, construct_tri_ndN, construct_tri_rtN,
                                       construct_tet_cgN, construct_tet_ndN, construct_tet_rtN,
                                       construct_dgNminus)
from fuse.dof import ImmersedDOF
from fuse.serialisation import ElementSerialiser
from fuse.tensor_products import TensorProductTriple
from FIAT.lagrange import Lagrange
from FIAT.quadrature_schemes import create_quadrature
from recursivenodes import recursive_nodes
import numpy as np
import sympy as sp
import pytest


def interval_dgN(deg):
    """A DG element on an interval, built as CG's edge sub-element is."""
    edge = polygon(3).edges()[0]
    pts = recursive_nodes(1, deg, domain="equilateral")[1:-1].flatten()
    dofs = [DOFGenerator([DOF(DeltaPairing(), PointKernel((p,)))], S2, S1)
            for p in pts[:len(pts) // 2]]
    return ElementTriple(edge, (PolynomialSpace(deg), CellL2, C0), dofs)


def interval_cg1():
    cell = polygon(3)
    edge = cell.edges()[0]
    vert = cell.vertices()[0]
    dg0 = ElementTriple(vert, (P0, CellL2, C0),
                        DOFGenerator([DOF(DeltaPairing(), PointKernel(()))], S1, S1))
    return ElementTriple(edge, (P1, CellH1, C0), [DOFGenerator([immerse(edge, dg0, TrH1)], S2, S1)])


# The DOFs of a k-form are integrals over k-dimensional entities, so the form
# degree is the dimension of the lowest dimensional entity carrying a DOF.
@pytest.mark.parametrize("name,builder,expected", [
    ("tri cg2", lambda: construct_tri_cgN(2), 0),
    ("tri nd1", lambda: construct_tri_ndN(1), 1),
    ("tri rt1", lambda: construct_tri_rtN(1), 1),
    ("tri dg1", lambda: construct_dgNminus(2)(1), 2),
    ("tet cg2", lambda: construct_tet_cgN(2), 0),
    ("tet nd1", lambda: construct_tet_ndN(1), 1),
    ("tet rt1", lambda: construct_tet_rtN(1), 2),
    ("tet dg1", lambda: construct_dgNminus(3)(1), 3),
    ("interval cg1", interval_cg1, 0),
    ("interval dg3", lambda: interval_dgN(3), 1),
])
def test_form_degree_by_entity_dim(name, builder, expected):
    assert builder().form_degree == expected


@pytest.mark.parametrize("deg", [1, 2, 3])
def test_form_degree_unchanged_for_scalar_cg(deg):
    assert construct_tri_cgN(deg).form_degree == 0


@pytest.mark.parametrize("deg", [1, 2])
def test_to_fiat_reports_form_degree(deg):
    assert construct_tri_cgN(deg).to_fiat().get_formdegree() == 0


def test_component_kernel_wraps_point_kernel():
    """Wrapping preserves the base's points and weights, and only rewrites components."""
    cell = polygon(3)
    base = DOF(DeltaPairing(), PointKernel((0.25, 0.25)), entity=cell, cell=cell)
    wrapped = DOF(DeltaPairing(), ComponentKernel((1,), PointKernel((0.25, 0.25))),
                  entity=cell, cell=cell)

    base_dict = base.to_quadrature(1, ())
    wrapped_dict = wrapped.to_quadrature(1, (2,))

    assert list(base_dict.keys()) == list(wrapped_dict.keys())
    for pt in base_dict:
        assert [w for w, _ in base_dict[pt]] == [w for w, _ in wrapped_dict[pt]]
        assert [c for _, c in base_dict[pt]] == [()]
        assert [c for _, c in wrapped_dict[pt]] == [(1,)]


def test_component_kernel_bare_behaviour_preserved():
    bare = ComponentKernel((1,))
    assert bare.permute(None) is bare
    assert bare.degree(3) == 3
    assert repr(bare) == "[(1,)]"


def test_component_kernel_rejects_unsupported_bases():
    """to_quadrature dispatches on kernel type and cannot see through the wrapper."""
    x = sp.Symbol("x")
    for bad in [BarycentricPolynomialKernel(x, symbols=[x]),
                VectorKernel(1),
                ComponentKernel((0,), PointKernel((0.5,)))]:
        with pytest.raises(NotImplementedError):
            ComponentKernel((0,), bad)


def test_component_kernel_comp_is_tuple():
    """A list component would silently trigger fancy indexing in FIAT's to_riesz."""
    assert isinstance(ComponentKernel([0]).comp, tuple)
    assert isinstance(ComponentKernel._from_dict({"comp": [1], "base_kernel": None}).comp, tuple)


@pytest.mark.parametrize("deg", [1, 3])
def test_with_kernel_preserves_attributes(deg):
    """CG3 covers immersed vertex, immersed edge and non-immersed interior DOFs."""
    for dof in construct_tri_cgN(deg).generate():
        copy = dof.with_kernel(PointKernel((0.5,)))

        assert type(copy) is type(dof)
        for attr in ["cell_defined_on", "attachment", "g", "immersed", "sub_id", "cell", "entity_o"]:
            assert getattr(copy, attr) is getattr(dof, attr)
        if isinstance(dof, ImmersedDOF):
            assert copy.triple is dof.triple
        # add_entity is reapplied by DOF.__init__, so check it round trips
        assert copy.pairing.entity is dof.pairing.entity
        assert copy.pairing.orientation is dof.pairing.orientation
        # a non immersed DOF is always given a fresh TrH1, so compare by value
        assert type(copy.target_space) is type(dof.target_space)


def test_with_kernel_replaces_only_the_kernel():
    dof = construct_tri_cgN(1).generate()[0]
    original = dof.kernel
    new_kernel = PointKernel((0.5,))
    copy = dof.with_kernel(new_kernel)

    assert copy.kernel is new_kernel
    assert dof.kernel is original
    assert copy is not dof
    assert copy.generation is not dof.generation
    assert copy.generation == dof.generation


def test_component_dofs_ordering():
    base_triple = construct_tri_cgN(2)
    base = base_triple.generate()
    vector_triple = VectorTriple(base_triple)
    dofs = vector_triple.generate()
    comp_map = vector_triple.comp_map

    assert len(dofs) == 2 * len(base)
    assert [d.id for d in dofs] == list(range(2 * len(base)))
    for i, dof in enumerate(dofs):
        parent, comp = base[i // 2], i % 2
        assert comp_map[dof.id] == (parent.id, comp)
        assert dof.cell_defined_on is parent.cell_defined_on
        assert dof.kernel.comp == (comp,)
        assert dof.kernel.base_kernel is parent.kernel


@pytest.mark.parametrize("deg", [1, 2, 3])
def test_component_dofs_quadrature_matches_scalar(deg):
    """Component DOFs keep the scalar points and weights, and only rewrite the component."""
    base_triple = construct_tri_cgN(deg)
    base = base_triple.generate()
    vector_triple = VectorTriple(base_triple)
    dofs = vector_triple.generate()
    comp_map = vector_triple.comp_map

    for dof in dofs:
        parent_id, comp = comp_map[dof.id]
        parent = next(d for d in base if d.id == parent_id)
        scalar = parent.to_quadrature(4, ())
        vector = dof.to_quadrature(4, (2,))

        assert list(scalar.keys()) == list(vector.keys())
        for pt in scalar:
            assert [w for w, _ in scalar[pt]] == [w for w, _ in vector[pt]]
            assert [c for _, c in scalar[pt]] == [()]
            assert [c for _, c in vector[pt]] == [(comp,)]


@pytest.mark.parametrize("builder,match", [
    (lambda: construct_tri_rtN(1), "HDiv"),
    (lambda: construct_tri_ndN(1), "HCurl"),
])
def test_vector_triple_rejects_piola_mapped(builder, match):
    with pytest.raises(ValueError, match=match):
        VectorTriple(builder())


def test_vector_triple_rejects_vector_valued():
    with pytest.raises(ValueError, match="already vector valued"):
        VectorTriple(VectorTriple(construct_tri_cgN(1)))


@pytest.mark.parametrize("builder,sd", [(construct_tri_cgN, 2), (construct_tet_cgN, 3)])
@pytest.mark.parametrize("deg", [1, 2])
def test_vector_triple_shape_degree_and_count(builder, sd, deg):
    base = builder(deg)
    vec = VectorTriple(base)

    assert vec.N == sd
    assert vec.get_value_shape() == (sd,)
    # inherited, and 0 because the clones keep their parent's entity
    assert vec.form_degree == base.form_degree == 0
    assert vec.num_dofs() == sd * base.num_dofs()
    assert len(vec.generate()) == sd * len(base.generate())
    assert vec.spaces[0].set_shape
    assert not base.spaces[0].set_shape


@pytest.mark.parametrize("builder", [construct_tri_cgN, construct_tet_cgN])
@pytest.mark.parametrize("deg", [1, 2])
def test_vector_triple_entity_ids_scale(builder, deg):
    """Pins the DOF ordering contract that the orientation matrix lift relies on."""
    base = builder(deg)
    vec = VectorTriple(base)
    vec.to_ufl()
    N = vec.N

    assert len(vec.nodes) == vec.poly_set.get_num_members()
    for dim in vec.entity_ids:
        for entity in vec.entity_ids[dim]:
            assert len(vec.entity_ids[dim][entity]) == N * len(base.entity_ids[dim][entity])

    for vec_id, (base_id, comp) in vec.comp_map.items():
        assert vec.dof_id_to_fiat_id[vec_id] == N * base.dof_id_to_fiat_id[base_id] + comp


def block_lagrange(ref_el, deg, N, pts):
    """Vector Lagrange laid out to match: row N*i+c is scalar basis fn i in component c."""
    sd = ref_el.get_spatial_dimension()
    scalar = Lagrange(ref_el, deg).tabulate(0, pts)[(0,) * sd]
    basis = np.zeros((N * scalar.shape[0], N, len(pts)))
    for i in range(scalar.shape[0]):
        for c in range(N):
            basis[N * i + c, c, :] = scalar[i, :]
    return basis


def tabulate_vector_cg(builder, deg):
    elem = VectorTriple(builder(deg)).to_fiat()
    sd = elem.ref_el.get_spatial_dimension()
    pts = create_quadrature(elem.ref_el, 2 * deg + 2).get_points()
    return elem, pts, elem.tabulate(0, pts)[(0,) * sd]


VECTOR_CG = [(construct_tri_cgN, 2, 1), (construct_tri_cgN, 2, 2),
             (construct_tri_cgN, 2, 3), (construct_tet_cgN, 3, 1),
             (construct_tet_cgN, 3, 2)]


@pytest.mark.parametrize("builder,sd,deg", VECTOR_CG)
def test_vector_cg_spans_block_lagrange(builder, sd, deg):
    """The two bases must span the same space, up to an invertible change of basis."""
    elem, pts, mine = tabulate_vector_cg(builder, deg)
    ref = block_lagrange(elem.ref_el, deg, sd, pts)

    flat_mine = mine.reshape(mine.shape[0], -1).T
    flat_ref = ref.reshape(ref.shape[0], -1).T
    change, _, _, _ = np.linalg.lstsq(flat_ref, flat_mine, rcond=None)

    assert np.allclose(flat_mine, flat_ref @ change)
    assert np.allclose(flat_ref, flat_mine @ np.linalg.inv(change))


@pytest.mark.parametrize("builder,sd,deg", VECTOR_CG)
def test_vector_cg_component_block(builder, sd, deg):
    """Basis function N*i+c lives entirely in component c, pinning component innermost.

    Insensitive to the order of the DOFs within an entity, which FUSE and FIAT do not
    share at degree 3, so unlike an exact comparison this holds at every degree.
    """
    _, _, mine = tabulate_vector_cg(builder, deg)

    for i in range(mine.shape[0] // sd):
        for c in range(sd):
            for other in range(sd):
                if other != c:
                    assert np.allclose(mine[sd * i + c, other, :], 0)


@pytest.mark.parametrize("builder,sd,deg", VECTOR_CG)
def test_vector_cg_component_slices_span_scalar(builder, sd, deg):
    """Fixed component slices reproduce the scalar space, whatever order they arrive in."""
    elem, pts, mine = tabulate_vector_cg(builder, deg)
    scalar = Lagrange(elem.ref_el, deg).tabulate(0, pts)[(0,) * elem.ref_el.get_spatial_dimension()]

    for c in range(sd):
        component = np.array([mine[sd * i + c, c, :] for i in range(mine.shape[0] // sd)])
        change, _, _, _ = np.linalg.lstsq(scalar.T, component.T, rcond=None)
        assert np.allclose(component.T, scalar.T @ change)


@pytest.mark.parametrize("builder,sd,deg", VECTOR_CG)
def test_vector_cg_fiat_metadata(builder, sd, deg):
    base = builder(deg).to_fiat()
    elem = VectorTriple(builder(deg)).to_fiat()

    assert elem.value_shape() == (sd,)
    assert elem.get_formdegree() == 0
    assert elem.space_dimension() == sd * base.space_dimension()


@pytest.mark.parametrize("builder,sd,deg", VECTOR_CG)
def test_vector_cg_matrices_are_kron(builder, sd, deg):
    base = builder(deg)
    vec = VectorTriple(base)
    vec.to_ufl()
    size = sd * base.num_dofs()

    for dim in vec.matrices:
        for entity in vec.matrices[dim]:
            for orientation, matrix in vec.matrices[dim][entity].items():
                expected = np.kron(base.matrices[dim][entity][orientation], np.eye(sd))
                assert np.allclose(matrix, expected)
                reverse = vec.reversed_matrices[dim][entity][orientation]
                assert np.allclose(matrix @ reverse, np.eye(size))


def test_vector_cg_dof_ordering_is_asserted():
    base = construct_tri_cgN(2)
    vec = VectorTriple(base)
    vec.ref_el = vec.cell.to_fiat()
    vec.poly_set = vec.spaces[0].to_ON_polynomial_set(vec.ref_el)
    vec.entity_ids, vec.nodes = vec.setup_ids_and_nodes()

    first = next(iter(vec.comp_map))
    vec.comp_map[first] = (vec.comp_map[first][0], vec.comp_map[first][1] + 1)
    with pytest.raises(ValueError, match="Kronecker"):
        vec.setup_matrices()


@pytest.mark.parametrize("deg", [1, 2, 3])
def test_vector_triple_round_trip(deg):
    """Serialisation must reconstruct an element that converts identically."""
    original = VectorTriple(construct_tri_cgN(deg))
    converter = ElementSerialiser()
    decoded = converter.decode(converter.encode(original))

    assert isinstance(decoded, VectorTriple)
    assert decoded.N == original.N
    assert decoded.num_dofs() == original.num_dofs()
    assert len(decoded.generate()) == len(original.generate())
    assert decoded.get_value_shape() == original.get_value_shape()
    assert decoded.form_degree == original.form_degree

    was, now = original.to_fiat(), decoded.to_fiat()
    pts = create_quadrature(was.ref_el, 2 * deg + 2).get_points()
    assert np.allclose(was.tabulate(0, pts)[(0, 0)], now.tabulate(0, pts)[(0, 0)])
    assert was.entity_dofs() == now.entity_dofs()
    assert was.get_formdegree() == now.get_formdegree()


@pytest.mark.parametrize("deg", [1, 2])
def test_vector_triple_round_trip_matrices(deg):
    """Orientation matrices could differ while tabulation still matched."""
    original = VectorTriple(construct_tri_cgN(deg))
    original.to_ufl()
    converter = ElementSerialiser()
    decoded = converter.decode(converter.encode(original))
    decoded.to_ufl()

    for dim in original.matrices:
        for entity in original.matrices[dim]:
            for orientation, matrix in original.matrices[dim][entity].items():
                assert np.allclose(matrix, decoded.matrices[dim][entity][orientation])
                assert np.allclose(original.reversed_matrices[dim][entity][orientation],
                                   decoded.reversed_matrices[dim][entity][orientation])


def test_vector_triple_registered_in_serialiser():
    """Unregistered ids are returned as their reference string rather than decoded."""
    assert ElementSerialiser().obj_types["VectorTriple"] is VectorTriple


def test_component_kernel_round_trip():
    """VectorTriple stores only its base, so this path needs its own cover."""
    converter = ElementSerialiser()
    bare = converter.decode(converter.encode(ComponentKernel((1,))))
    assert bare.comp == (1,)
    assert isinstance(bare.comp, tuple)
    assert bare.base_kernel is None

    converter = ElementSerialiser()
    wrapped = converter.decode(converter.encode(ComponentKernel((1,), PointKernel((0.25, 0.5)))))
    assert wrapped.comp == (1,)
    assert isinstance(wrapped.comp, tuple)
    assert isinstance(wrapped.base_kernel, PointKernel)
    assert wrapped.base_kernel.pt == (0.25, 0.5)


@pytest.mark.parametrize("flat", [False, True])
def test_vector_triple_rejects_tensor_product(flat):
    """Fails with a clear message rather than an AttributeError from generate()."""
    cell = polygon(3)
    edge = cell.edges()[0]
    vert = cell.vertices()[0]
    dg0 = ElementTriple(vert, (P0, CellL2, C0),
                        DOFGenerator([DOF(DeltaPairing(), PointKernel(()))], S1, S1))
    interval = ElementTriple(edge, (P1, CellH1, C0),
                             [DOFGenerator([immerse(edge, dg0, TrH1)], S2, S1)])
    tp = TensorProductTriple(interval, interval)
    if flat:
        tp = tp.flatten()

    with pytest.raises(ValueError, match="tensor product"):
        VectorTriple(tp)


@pytest.mark.parametrize("deg", [1, 2, 3])
def test_vector_cg_matches_firedrake_vector_cg(deg):
    """End to end check that exercises Firedrake's fuse_orientations.

    Imported inside the test so the rest of this module stays Firedrake free.
    """
    from firedrake import (UnitSquareMesh, FunctionSpace, VectorFunctionSpace, Function,
                           SpatialCoordinate, as_vector, assemble, dot, dx, project)

    mesh = UnitSquareMesh(4, 4, use_fuse=True)
    V = FunctionSpace(mesh, VectorTriple(construct_tri_cgN(deg)).to_ufl())
    W = VectorFunctionSpace(mesh, "CG", deg)
    assert V.dim() == W.dim()

    x, y = SpatialCoordinate(mesh)
    expr = as_vector([x**deg + 2 * y, 3 * x - y**deg])

    # an expression of this degree lies in the space, so interpolation is exact
    interpolated = Function(V).interpolate(expr)
    residual = interpolated - expr
    assert assemble(dot(residual, residual) * dx) < 1e-20

    # and projecting agrees with Firedrake's own vector CG, which spans the same space
    difference = project(expr, V) - project(expr, W)
    assert assemble(dot(difference, difference) * dx) < 1e-20
