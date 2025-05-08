from fuse import *
from test_convert_to_fiat import create_cg1, create_dg1, create_cg2
from test_2d_examples_docs import construct_cg3
import pytest
import numpy as np
import sympy as sp

vert = Point(0)
edge = Point(1, [Point(0), Point(0)], vertex_num=2)
tri = polygon(3)


@pytest.mark.parametrize("cell", [edge])
def test_cg_perms(cell):
    cg1 = create_cg1(cell)
    cg1.to_fiat()


@pytest.mark.parametrize("cell", [edge])
def test_cg2_perms(cell):
    cg2 = create_cg2(cell)
    cg2.to_fiat()


def test_cg3_perms():
    cg3 = construct_cg3()
    cg3.to_fiat()


@pytest.mark.parametrize("cell", [edge])
def test_dg_perms(cell):
    dg1 = create_dg1(cell)
    dg1.to_fiat()


@pytest.mark.parametrize("cell", [Point(1, [Point(0), Point(0)], vertex_num=2), polygon(3)])
def test_basic_perms(cell):
    cell_group = cell.group
    bvs = cell.basis_vectors()
    M = np.array(bvs).T
    for g in cell_group.members():
        trans_bvs = np.array([g(bvs[i]) for i in range(len(bvs))]).T
        print(g)
        print(np.linalg.solve(M, trans_bvs))


# @pytest.mark.parametrize("cell", [pytest.param(edge, marks=pytest.mark.xfail(reason='Dense Permutations needed'))])

@pytest.mark.parametrize("cell", [tri])
def test_nd_perms(cell):
    deg = 1
    edge = cell.edges(get_class=True)[0]
    x = sp.Symbol("x")
    y = sp.Symbol("y")

    xs = [DOF(L2Pairing(), PolynomialKernel(edge.basis_vectors()[0]))]
    dofs = DOFGenerator(xs, S1, S2)
    int_ned = ElementTriple(edge, (P1, CellHCurl, C0), dofs)

    xs = [immerse(cell, int_ned, TrHCurl)]
    tri_dofs = DOFGenerator(xs, C3, S3)

    M = sp.Matrix([[y, -x]])
    vec_Pk = PolynomialSpace(deg - 1, set_shape=True)
    Pk = PolynomialSpace(deg - 1)
    nd = vec_Pk + (Pk.restrict(deg - 2, deg - 1))*M

    ned = ElementTriple(cell, (nd, CellHCurl, C0), [tri_dofs])
    ned_fiat = ned.to_fiat()
    for i, mat in ned.matrices[2][0].items():
        print(i)
        print(mat)


@pytest.mark.xfail(reason='Conversion of non simplex ref els to fiat needed')
def test_square():
    square = polygon(4)
    print("testsq", square.group.size())
    edge = square.d_entities(1, get_class=True)[0]
# edge.basis_vectors()[0]
    xs = [DOF(L2Pairing(), PointKernel(-0.5))]
    dg0_int = ElementTriple(edge, (P0, CellL2, C0),
                            DOFGenerator(xs, S2, S2))

    e_xs = [immerse(square, dg0_int, TrHDiv)]
    e_dofs = DOFGenerator(e_xs, sq_edges, S1)

    # i_xs = [lambda g: DOF(DeltaPairing(), PointKernel(g((0, 0))))]
    # i_dofs = DOFGenerator(i_xs, S1, S1)

    sq = ElementTriple(square, (P3, CellH1, C0), [e_dofs])
    for dof in sq.generate():
        print(dof)
    sq.to_fiat()
