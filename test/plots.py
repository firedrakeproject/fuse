from fuse import *
from test_2d_examples_docs import construct_cg3
# .. plot:: ../../test/plots.py


def construct_dg1():
    # [test_dg1_int 0]
    edge = Point(1, [Point(0, group=S1), Point(0, group=S1)], vertex_num=2, group=S2)
    xs = [DOF(DeltaPairing(), PointKernel((-1,)))]
    dg1 = ElementTriple(edge, (P1, CellL2, C0), DOFGenerator(xs, S2, S1))
    # [test_dg1_int 1]
    return dg1


def plot_dg1():
    # tri = polygon(3)
    # cg1 = create_cg1(tri)
    cg3 = construct_cg3()
    # dg1 = construct_dg1()
    cg3.plot()


plot_dg1()
