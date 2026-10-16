from fuse import *
from firedrake import *
from finat.ufl import CellBackend
from fuse.serialisation import ElementSerialiser
from fuse.actions import HDivAction
from test_convert_to_fiat import create_cg1
from test_orientations import interpolate_vs_project, get_expression
import pytest
import numpy as np

vert = Point(0)
edge = Point(1, [Point(0), Point(0)], vertex_num=2)
tri = polygon(3)


def test_dg_examples():
    converter = ElementSerialiser()
    encoded = converter.encode(vert)
    decoded = converter.decode(encoded)
    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(decoded, (P0, C0), DOFGenerator(xs, S1))

    # [test_serialise 0]
    converter = ElementSerialiser()
    encoded = converter.encode(dg0)
    decoded = converter.decode(encoded)
    # [test_serialise 1]

    for dof in decoded.generate():
        assert dof.eval(lambda: 1) == 1

    xs = [DOF(DeltaPairing(), PointKernel((-1,)))]
    dg1 = ElementTriple(edge, (P1, C0), DOFGenerator(xs, S2))

    converter = ElementSerialiser()
    encoded = converter.encode(dg1)
    decoded = converter.decode(encoded)

    for dof in decoded.generate():
        assert np.allclose(abs(dof.eval(lambda x: x)), 1)


def test_repeated_objs():
    repeated_edge = Point(1, [vert, vert], vertex_num=2)
    converter = ElementSerialiser()
    encoded = converter.encode(repeated_edge)
    decoded = converter.decode(encoded)
    print(encoded)
    print(decoded)


def test_cg_examples():
    cells = [vert, edge, tri]

    for cell in cells:
        triple = create_cg1(cell)

        dofs = [d.eval(FuseFunction(lambda *x: x)) for d in triple.generate()]
        converter = ElementSerialiser()
        encoded = converter.encode(triple)
        decoded = converter.decode(encoded)
        for d in decoded.generate():
            dof_val = d.eval(FuseFunction(lambda *x: x))
            assert any([np.allclose(dof_val, dof_val2) for dof_val2 in dofs])


def test_tetrahedron_example():
    triple = periodic_table(0, 3, 3, 2)

    dofs = [d.eval(FuseFunction(lambda *x: x)) for d in triple.generate()]
    converter = ElementSerialiser()
    encoded = converter.encode(triple)
    decoded = converter.decode(encoded)
    for d in decoded.generate():
        dof_val = d.eval(FuseFunction(lambda *x: x))
        assert any([np.allclose(dof_val, dof_val2) for dof_val2 in dofs])


def round_trip(obj):
    converter = ElementSerialiser()
    return converter.decode(converter.encode(obj))


# Known serialiser gaps, to be fixed with the serialiser (strict, so these flag once fixed):
# * the serialiser shares objects that compare equal, and PolynomialSpace equality ignores
#   the weights, so the three weighted parts of the tetrahedron ND space decode as one;
# * groups are rebuilt from their members, which reorders them, so DOFs are generated in another order.
_serialiser_gaps = {(0, 3, 1, 1): "weighted spaces compare equal",
                    (0, 3, 1, 2): "weighted spaces compare equal",
                    (1, 3, 3, 2): "decoded groups reorder their members"}


def _round_trip_param(p):
    if p in _serialiser_gaps:
        return pytest.param(*p, marks=pytest.mark.xfail(reason=_serialiser_gaps[p], strict=True))
    return p


round_trip_params = [_round_trip_param(p) for p in
                     [(col, dim, 3, 0) for dim in (2, 3) for col in (0, 1)]
                     + [(col, 2, k, deg) for deg in range(1, 7) for k in range(4) for col in (0, 1)]
                     + [(col, 3, k, deg) for deg in range(1, 3) for k in range(4) for col in (0, 1)]]


@pytest.mark.parametrize("col,dim,k,deg", round_trip_params)
def test_round_trip_keeps_element(col, dim, k, deg):
    "The decoded element has the same DOFs and orientation matrices as the original."
    elem = periodic_table(col, dim, k, deg)
    decoded = round_trip(elem)
    space = elem.spaces[0]
    assert np.allclose(space.functional_matrix(elem.cell, elem.generate()),
                       space.functional_matrix(elem.cell, decoded.generate()))
    elem.to_fiat()
    decoded.to_fiat()
    for d in elem.matrices:
        for e in elem.matrices[d]:
            for v in elem.matrices[d][e]:
                assert np.allclose(elem.matrices[d][e][v], decoded.matrices[d][e][v]), (d, e, v)


def test_acted_on_kernels_are_not_serialised():
    "Serialisation does not keep a kernel's orientation yet, so encoding one raises rather than losing it."
    g = tri.group.members()[1]
    kernel = VectorKernel((1, 0), action=HDivAction).act(g)
    with pytest.raises(NotImplementedError):
        round_trip(kernel)


@pytest.mark.parametrize("name,params", [("cg1_triangle", (0, 2, 0, 1)), ("nd1_triangle", (0, 2, 1, 1)),
                                         ("rt1_tetrahedron", (0, 3, 2, 1))])
def test_dicts_with_a_pullback_still_decode(name, params):
    "Elements serialised before the pullback was removed decode to the element built today."
    import pathlib
    path = pathlib.Path(__file__).parent / "data" / "serialised_with_pullback" / f"{name}.json"
    decoded = ElementSerialiser().decode(path.read_text())
    elem = periodic_table(*params)
    assert decoded.pullback() == elem.pullback()
    space = elem.spaces[0]
    assert np.allclose(space.functional_matrix(elem.cell, elem.generate()),
                       space.functional_matrix(elem.cell, decoded.generate()))


cg_params = [(0, 0, deg, deg + 0.75) for deg in list(range(1, 3))] + [(1, 0, deg, deg + 0.75) for deg in list(range(1, 3))]
nd_params = [(0, 1, deg, deg - 0.2) for deg in list(range(1, 3))]
rt_params = [(0, 2, deg, deg - 0.2) for deg in list(range(1, 3))]
dg_params = [(0, 3, deg, deg + 0.75) for deg in list(range(0, 3))] + [(1, 3, deg, deg + 0.75) for deg in list(range(0, 3))]
nd2_params = [(1, 1, deg, deg + 0.75) for deg in list(range(1, 3))]
bdm_params = [(1, 2, deg, deg + 0.75) for deg in list(range(1, 3))]


@pytest.mark.parametrize("col,k,deg,conv_rate", cg_params + nd_params + rt_params + dg_params + nd2_params + bdm_params)
def test_post_serialisation_convergence(col, k, deg, conv_rate):
    "Tests that appropriate convergence is still achieved after serialisation."
    elem = periodic_table(col, 2, k, deg)
    converter = ElementSerialiser()
    encoded = converter.encode(elem)
    elem_decoded = converter.decode(encoded)
    scale_range = range(3, 6)
    diff_inte = [0 for i in scale_range]
    for n in scale_range:
        mesh = UnitSquareMesh(2**n, 2**n, cell_backend=CellBackend.FUSE)

        V = FunctionSpace(mesh, elem_decoded.to_ufl())
        x, y = SpatialCoordinate(mesh)
        expr = cos(x*pi*2)*sin(y*pi*2)
        shape = elem.get_value_shape()
        if len(shape) > 0:
            expr = as_vector([expr] * shape[0])
        _, exact = get_expression(V)
        _, diff_inte[n-min(scale_range)] = interpolate_vs_project(V, expr, exact)

    print("interpolation l2 error norms:", diff_inte)
    diff_inte = np.array(diff_inte)
    conv = np.log2(diff_inte[:-1] / diff_inte[1:])
    print("convergence order:", conv)
    assert all([c > conv_rate for c in conv])
