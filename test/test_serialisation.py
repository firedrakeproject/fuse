from fuse import *
from fuse.serialisation import ElementSerialiser
from test_convert_to_fiat import create_cg1
# from test_2d_examples_docs import construct_nd
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

        dofs = [d.eval(MyTestFunction(lambda *x: x)) for d in triple.generate()]
        converter = ElementSerialiser()
        encoded = converter.encode(triple)
        decoded = converter.decode(encoded)
        for d in decoded.generate():
            dof_val = d.eval(MyTestFunction(lambda *x: x))
            assert any([np.allclose(dof_val, dof_val2) for dof_val2 in dofs])


# def test_ned():
#     cell = polygon(3)
#     triple = construct_nd(cell)
#     converter = ElementSerialiser()
#     encoded = converter.encode(triple)

#     decoded = converter.decode(encoded)
#     for d in decoded.generate():
#         dof_val = d.eval(phi_2)
#         assert any([np.allclose(dof_val, dof_val2) for dof_val2 in dofs])
