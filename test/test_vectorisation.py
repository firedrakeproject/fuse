from fuse import *
from fuse.element_construction import (construct_tri_cgN, construct_tri_ndN, construct_tri_rtN,
                                       construct_tet_cgN, construct_tet_ndN, construct_tet_rtN,
                                       construct_dgNminus)
from fuse.dof import ImmersedDOF
from fuse.serialisation import ElementSerialiser
from fuse.tensor_products import TensorProductTriple
from finat.ufl import CellBackend
from FIAT.lagrange import Lagrange
from FIAT.quadrature_schemes import create_quadrature
from recursivenodes import recursive_nodes
import numpy as np
import sympy as sp
import pytest


def interval_dgN(deg):
    edge = polygon(3).edges()[0]
    pts = recursive_nodes(1, deg, domain="equilateral")[1:-1].flatten()
    dofs = [DOFGenerator([DOF(DeltaPairing(), PointKernel((p,)))], S2, S1)
            for p in pts[:len(pts) // 2]]
    return ElementTriple(edge, (PolynomialSpace(deg), C0, Fid), dofs)


def interval_cg1():
    cell = polygon(3)
    edge = cell.edges()[0]
    vert = cell.vertices()[0]
    dg0 = ElementTriple(vert, (P0, C0, Fid),
                        DOFGenerator([DOF(DeltaPairing(), PointKernel(()))], S1, S1))
    return ElementTriple(edge, (P1, C0, Fid), [DOFGenerator([immerse(edge, dg0, TrH1)], S2, S1)])


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
    (lambda: construct_tri_rtN(1), "contravariant Piola"),
    (lambda: construct_tri_ndN(1), "covariant Piola"),
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
    assert vec.spaces[0].shape
    assert not base.spaces[0].shape


@pytest.mark.parametrize("builder", [construct_tri_cgN, construct_tet_cgN])
@pytest.mark.parametrize("deg", [1, 2])
def test_vector_triple_entity_ids_scale(builder, deg):
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
    sd = ref_el.get_spatial_dimension()
    scalar = Lagrange(ref_el, deg).tabulate(0, pts)[(0,) * sd]
    basis = np.zeros((N * scalar.shape[0], N, len(pts)))
    for i in range(scalar.shape[0]):
        for c in range(N):
            basis[N * i + c, c, :] = scalar[i, :]
    return basis


def tabulate_vector_cg(builder, deg, dim=None):
    elem = VectorTriple(builder(deg), dim).to_fiat()
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
    """Serialisation must reconstruct an element identically."""
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
    assert ElementSerialiser().obj_types["VectorTriple"] is VectorTriple


def test_component_kernel_round_trip():
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
    cell = polygon(3)
    edge = cell.edges()[0]
    vert = cell.vertices()[0]
    dg0 = ElementTriple(vert, (P0, C0, Fid),
                        DOFGenerator([DOF(DeltaPairing(), PointKernel(()))], S1, S1))
    interval = ElementTriple(edge, (P1, C0, Fid),
                             [DOFGenerator([immerse(edge, dg0, TrH1)], S2, S1)])
    tp = TensorProductTriple(interval, interval)
    if flat:
        tp = tp.flatten()

    with pytest.raises(ValueError, match="tensor product"):
        VectorTriple(tp)


@pytest.mark.parametrize("deg", [1, 2, 3])
def test_vector_cg_matches_firedrake_vector_cg(deg):
    """End to end check that exercises Firedrake's fuse_orientations.

    """
    from firedrake import (UnitSquareMesh, FunctionSpace, VectorFunctionSpace, Function,
                           SpatialCoordinate, as_vector, assemble, dot, dx, project)

    mesh = UnitSquareMesh(4, 4, cell_backend=CellBackend.FUSE)
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


@pytest.mark.parametrize("dim", [1, 4])
def test_vector_cg_matches_firedrake_at_non_gdim(dim):
    """A value dimension unrelated to the mesh, against Firedrake's own dim= form.

    """
    from firedrake import (UnitSquareMesh, FunctionSpace, VectorFunctionSpace, Function,
                           SpatialCoordinate, as_vector, assemble, dot, dx, project)

    deg = 2
    mesh = UnitSquareMesh(4, 4, cell_backend=CellBackend.FUSE)
    V = FunctionSpace(mesh, VectorTriple(construct_tri_cgN(deg), dim).to_ufl())
    W = VectorFunctionSpace(mesh, "CG", deg, dim=dim)
    assert V.dim() == W.dim()

    x, y = SpatialCoordinate(mesh)
    expr = as_vector([(i + 1) * x**deg - i * y for i in range(dim)])

    interpolated = Function(V).interpolate(expr)
    residual = interpolated - expr
    assert assemble(dot(residual, residual) * dx) < 1e-20

    difference = project(expr, V) - project(expr, W)
    assert assemble(dot(difference, difference) * dx) < 1e-20


@pytest.mark.parametrize("given,expected", [
    (False, ()), (0, ()), ((), ()), ([], ()),
    (1, (1,)), (4, (4,)), ((4,), (4,)), ([3], (3,)),
    ((2, 2), (2, 2)), ([2, 3], (2, 3)),
])
def test_shape_normalisation(given, expected):
    assert PolynomialSpace(2, shape=given).shape == expected


@pytest.mark.parametrize("given", [True, -1, 0.5, (2, 0), (2, -1), ("2",), None])
def test_shape_rejects_invalid(given):
    """True is rejected explicitly: the shape can no longer be inferred from the cell."""
    with pytest.raises(ValueError):
        PolynomialSpace(2, shape=given)


@pytest.mark.parametrize("shape", [(), (4,), (2, 2)])
def test_shape_round_trips(shape):
    """A tuple decodes from JSON as a list, so the space must renormalise it."""
    space = PolynomialSpace(2, shape=shape)
    serialiser = ElementSerialiser()
    decoded = serialiser.decode(serialiser.encode(space))

    assert decoded.shape == shape
    assert decoded == space
    assert hash(decoded) == hash(space)


@pytest.mark.parametrize("shape,members", [((4,), 4), ((2, 2), 4), ((3,), 3)])
def test_polynomial_set_takes_any_shape(shape, members):
    cell = polygon(3)
    scalar = PolynomialSpace(2).to_ON_polynomial_set(cell)
    shaped = PolynomialSpace(2, shape=shape).to_ON_polynomial_set(cell)

    assert shaped.get_shape() == shape
    assert shaped.get_num_members() == members * scalar.get_num_members()


def test_constructed_space_rejects_disagreeing_shapes():
    with pytest.raises(ValueError, match="differing value shapes"):
        PolynomialSpace(1, shape=2) + PolynomialSpace(1, shape=3)


def test_constructed_space_rejects_mismatched_weight_width():
    """The weight is what gives a scalar space its components, so its width must agree.

    Caught when the combination is built rather than when it is tabulated, because
    the weight width is what the combined shape is derived from.
    """
    x, y = sp.Symbol("x"), sp.Symbol("y")
    with pytest.raises(ValueError, match=r"differing value shapes: \[\(2,\), \(3,\)\]"):
        PolynomialSpace(1, shape=3) + PolynomialSpace(1)*sp.Matrix([[x, y]])


def test_constructed_space_rejects_non_row_weight():
    """tabulate_sympy only reads a weight's first row, so a matrix weight must be one.

    The combined shape is counted from every entry, so the two disagree here and the
    tabulation-time guard is what catches it.
    """
    x, y = sp.Symbol("x"), sp.Symbol("y")
    space = PolynomialSpace(1) * sp.Matrix([[x, y], [y, x]])
    with pytest.raises(ValueError, match="components but the space"):
        space.to_ON_polynomial_set(polygon(3))


def test_constructed_space_rejects_weighting_a_shaped_space():
    x, y = sp.Symbol("x"), sp.Symbol("y")
    with pytest.raises(ValueError, match="only scalar spaces"):
        PolynomialSpace(1, shape=2) * sp.Matrix([[x, y]])


# dim is independent of the cell, so the same value is used on tri and tet
NON_GDIM = [(construct_tri_cgN, 4), (construct_tri_cgN, 1), (construct_tet_cgN, 4),
            (construct_tri_cgN, (2, 2)), (construct_tet_cgN, (2, 2))]


@pytest.mark.parametrize("builder,dim", NON_GDIM)
@pytest.mark.parametrize("deg", [1, 2])
def test_vector_triple_honours_dim(builder, dim, deg):
    base = builder(deg)
    vec = VectorTriple(base, dim)
    shape = (dim,) if isinstance(dim, int) else dim
    N = int(np.prod(shape))

    assert vec.shape == shape
    assert vec.get_value_shape() == shape
    assert vec.N == N
    assert vec.num_dofs() == N * base.num_dofs()
    # components of a 0-form are still 0-forms, whatever the cell
    assert vec.form_degree == base.form_degree == 0

    elem = vec.to_fiat()
    assert elem.value_shape() == shape
    assert elem.space_dimension() == N * base.to_fiat().space_dimension()


def test_vector_triple_rejects_scalar_dim():
    with pytest.raises(ValueError, match="must have a component"):
        VectorTriple(construct_tri_cgN(1), 0)


@pytest.mark.parametrize("builder,dim", NON_GDIM)
@pytest.mark.parametrize("deg", [1, 2])
def test_non_gdim_component_block(builder, dim, deg):
    """Each basis function is supported in exactly one component.
    """
    _, _, mine = tabulate_vector_cg(builder, deg, dim)
    shape = (dim,) if isinstance(dim, int) else dim
    components = list(np.ndindex(shape))
    N = len(components)

    for i in range(mine.shape[0]):
        for j, other in enumerate(components):
            if j != i % N:
                assert np.allclose(mine[(i,) + other], 0)
        assert not np.allclose(mine[(i,) + components[i % N]], 0)


@pytest.mark.parametrize("builder,dim", NON_GDIM)
@pytest.mark.parametrize("deg", [1, 2])
def test_non_gdim_component_slices_span_scalar(builder, dim, deg):
    """Fixed component slices produce the scalar space."""
    elem, pts, mine = tabulate_vector_cg(builder, deg, dim)
    sd = elem.ref_el.get_spatial_dimension()
    scalar = Lagrange(elem.ref_el, deg).tabulate(0, pts)[(0,) * sd]
    shape = (dim,) if isinstance(dim, int) else dim
    components = list(np.ndindex(shape))
    N = len(components)

    for c, comp in enumerate(components):
        block = np.array([mine[(N * i + c,) + comp] for i in range(mine.shape[0] // N)])
        change, _, _, _ = np.linalg.lstsq(scalar.T, block.T, rcond=None)
        assert np.allclose(block.T, scalar.T @ change)


@pytest.mark.parametrize("builder,dim", NON_GDIM)
def test_non_gdim_matrices_are_kron(builder, dim):
    base = builder(2)
    base.to_ufl()
    vec = VectorTriple(base, dim)
    vec.to_ufl()

    for d, by_entity in vec.matrices.items():
        for e_id, by_val in by_entity.items():
            for val, mat in by_val.items():
                assert np.allclose(mat, np.kron(base.matrices[d][e_id][val], np.eye(vec.N)))
                assert np.allclose(vec.reversed_matrices[d][e_id][val] @ mat,
                                   np.eye(mat.shape[0]))


@pytest.mark.parametrize("builder,dim", NON_GDIM)
def test_non_gdim_round_trip(builder, dim):
    """VectorTriple stores only its base, so dim needs its own serialisation cover."""
    vec = VectorTriple(builder(1), dim)
    serialiser = ElementSerialiser()
    decoded = serialiser.decode(serialiser.encode(vec))

    assert isinstance(decoded, VectorTriple)
    assert decoded.shape == vec.shape
    assert decoded.get_value_shape() == vec.get_value_shape()

    pts = create_quadrature(vec.to_fiat().ref_el, 4).get_points()
    sd = vec.to_fiat().ref_el.get_spatial_dimension()
    assert np.allclose(decoded.to_fiat().tabulate(0, pts)[(0,) * sd],
                       vec.to_fiat().tabulate(0, pts)[(0,) * sd])
    assert decoded.entity_ids == vec.entity_ids
