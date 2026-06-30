from fuse import *
from firedrake import *
from fuse.serialisation import ElementSerialiser
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
    dg0 = ElementTriple(decoded, (P0, CellL2, C0), DOFGenerator(xs, S1, S1))

    # [test_serialise 0]
    converter = ElementSerialiser()
    encoded = converter.encode(dg0)
    decoded = converter.decode(encoded)
    # [test_serialise 1]

    for dof in decoded.generate():
        assert dof.eval(lambda: 1) == 1

    xs = [DOF(DeltaPairing(), PointKernel((-1,)))]
    dg1 = ElementTriple(edge, (P1, CellL2, C0), DOFGenerator(xs, S2, S1))

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
        mesh = UnitSquareMesh(2**n, 2**n, use_fuse=True)

        V = FunctionSpace(mesh, elem_decoded.to_ufl())
        x, y = SpatialCoordinate(mesh)
        expr = cos(x*pi*2)*sin(y*pi*2)
        if len(elem.get_value_shape()) > 0:
            expr = as_vector([expr, expr])
        _, exact = get_expression(V)
        _, diff_inte[n-min(scale_range)] = interpolate_vs_project(V, expr, exact)

    print("interpolation l2 error norms:", diff_inte)
    diff_inte = np.array(diff_inte)
    conv = np.log2(diff_inte[:-1] / diff_inte[1:])
    print("convergence order:", conv)
    assert all([c > conv_rate for c in conv])
