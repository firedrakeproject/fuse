from fuse import *
import pytest
from sympy.combinatorics import Permutation
import sympy as sp
import numpy as np
np.set_printoptions(linewidth=90, precision=4, suppress=True)


def test_dg1():
    tetra = make_tetrahedron()

    xs = [DOF(DeltaPairing(), PointKernel(tuple(tetra.vertices(return_coords=True)[0])))]
    dg1 = ElementTriple(tetra, (P1, CellL2, "C0"),
                        DOFGenerator(xs, Z4, S1))

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    z = sp.Symbol("z")
    test_func = FuseFunction(x + y + z, symbols=(x, y, z))

    dof_vals = [x+y+z for (x, y, z) in tetra.vertices(return_coords=True)]

    for dof in dg1.generate():
        assert any(np.isclose(val, dof.eval(test_func)) for val in dof_vals)


def construct_tet_cg3():
    # [make_tet 0]
    tetra = make_tetrahedron()
    # [make_tet 1]
    vert = tetra.vertices()[0]
    edge = tetra.edges()[0]
    face = tetra.d_entities(2)[0]
    # [make_tet 2]

    # [test_tet_cg3 0]
    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, CellL2, "C0"),
                        DOFGenerator(xs, S1, S1))

    xs = [DOF(DeltaPairing(), PointKernel((-1/3,)))]
    dg1_int = ElementTriple(edge, (P0, CellL2, "C0"),
                            DOFGenerator(xs, S2, S1))

    xs = [DOF(DeltaPairing(), PointKernel((0, 0)))]
    dg0_face = ElementTriple(face, (P0, CellL2, "C0"),
                             DOFGenerator(xs, S1, S1))

    v_xs = [immerse(tetra, dg0, TrH1)]
    cgverts = DOFGenerator(v_xs, Z4, S1)

    e_xs = [immerse(tetra, dg1_int, TrH1)]
    cgedges = DOFGenerator(e_xs, tet_edges, S1)

    f_xs = [immerse(tetra, dg0_face, TrH1)]
    cgfaces = DOFGenerator(f_xs, tet_faces, S1)

    cg3 = ElementTriple(tetra, (P1, CellH1, "C0"),
                        [cgverts, cgedges, cgfaces])
    # [test_tet_cg3 1]

    return cg3


def plot_tet_cg3():
    cg3 = construct_tet_cg3()
    cg3.plot()


def test_tet_cg3():
    cg3 = construct_tet_cg3()

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    z = sp.Symbol("z")
    test_func = FuseFunction(sp.Matrix([10*x, 3*y/np.sqrt(3), z*4]), symbols=(x, y, z))
    cg3.plot(filename="tet_cg3.png")
    print(cg3.to_tikz())
    for dof in cg3.generate():
        print(dof)
        dof.eval(test_func)


def construct_tet_rt(cell=None):
    if cell is None:
        cell = make_tetrahedron()
    face = cell.d_entities(2, get_class=True)[0]
    deg = 1
    # [test_tet_rt 0]
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    z = sp.Symbol("z")
    M = sp.Matrix([[x, y, z]])

    vec_Pd = PolynomialSpace(deg - 1, set_shape=True)
    Pd = PolynomialSpace(deg - 1)
    rt_space = vec_Pd + (Pd.restrict(deg - 2, deg - 1))*M

    xs = [DOF(L2Pairing(), VectorKernel((1,)))]
    dofs = DOFGenerator(xs, S1, S3)
    face_vec = ElementTriple(face, (rt_space, CellHDiv, "C0"), dofs)

    im_xs = [immerse(cell, face_vec, TrHDiv)]
    face = DOFGenerator(im_xs, tet_faces, S4)

    rt1 = ElementTriple(cell, (rt_space, CellHDiv, "C0"),
                        [face])
    # [test_tet_rt 1]
    return rt1


def construct_tet_ned(cell=None):
    deg = 1
    tri = make_tetrahedron()
    edge = tri.edges()[0]

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    z = sp.Symbol("z")

    M1 = sp.Matrix([[0, z, -y]])
    M2 = sp.Matrix([[z, 0, -x]])
    M3 = sp.Matrix([[y, -x, 0]])

    vec_Pd = PolynomialSpace(deg - 1, set_shape=True)
    Pd = PolynomialSpace(deg - 1)
    nd_space = vec_Pd + (Pd.restrict(deg - 2, deg - 1))*M1 + (Pd.restrict(deg - 2, deg - 1))*M2 + (Pd.restrict(deg - 2, deg - 1))*M3

    # [test_tet_ned 0]
    xs = [DOF(L2Pairing(), VectorKernel((1,)))]
    dofs = DOFGenerator(xs, S1, S2)

    edges = ElementTriple(edge, (vec_Pd, CellHCurl, L2), dofs)
    xs = [immerse(tri, edges, TrHCurl)]
    tet_edges = PermutationSetRepresentation([Permutation([0, 1, 2, 3]), Permutation([1, 2, 3, 0]),
                                              Permutation([2, 3, 0, 1]), Permutation([1, 3, 0, 2]),
                                              Permutation([2, 0, 1, 3]), Permutation([3, 0, 1, 2])])
    edge_dofs = DOFGenerator(xs, tet_edges, S1)
    # [test_tet_ned 1]

    return ElementTriple(tri, (nd_space, CellHCurl, L2), [edge_dofs])


# def construct_tet_nd2(tet=None):
#     if tet is None:
#         tet = polygon(3)
#     deg = 2
#     edge = tri.edges()[0]
#     x = sp.Symbol("x")
#     y = sp.Symbol("y")

#     xs = [DOF(L2Pairing(), PolynomialKernel((1/2)*(x + 1), symbols=(x,)))]

#     dofs = DOFGenerator(xs, S2, S2)
#     int_ned1 = ElementTriple(edge, (PolynomialSpace(1, set_shape=True), CellHCurl, C0), dofs)
#     v_2 = np.array(tri.get_node(tri.ordered_vertices()[2], return_coords=True))
#     v_1 = np.array(tri.get_node(tri.ordered_vertices()[1], return_coords=True))
#     xs = [DOF(L2Pairing(), VectorKernel((v_2 - v_1)/2))]

#     center_dofs = DOFGenerator(xs, S2, S3)
#     xs = [immerse(tri, int_ned1, TrHCurl)]
#     tet_edges = PermutationSetRepresentation([Permutation([0, 1, 2, 3]), Permutation([1, 2, 3, 0]),
#                                               Permutation([2, 3, 0, 1]), Permutation([1, 3, 0, 2]),
#                                               Permutation([2, 0, 1, 3]), Permutation([3, 0, 1, 2])])
#     edge_dofs = DOFGenerator(xs, tet_edges, S1)

#     vec_Pk = PolynomialSpace(deg - 1, set_shape=True)
#     Pk = PolynomialSpace(deg - 1)
#     M = sp.Matrix([[y, -x]])
#     nd = vec_Pk + (Pk.restrict(deg-2, deg-1))*M

#     ned = ElementTriple(tri, (nd, CellHCurl, C0), [edge_dofs, center_dofs])
#     return ned

def plot_tet_rt():
    rt = construct_tet_rt()
    rt.plot()


def test_tet_rt():
    rt1 = construct_tet_rt()
    ls = rt1.generate()
    # TODO make this a proper test
    for dof in ls:
        print(dof)
    rt1.to_fiat()


def test_tet_nd():
    nd1 = construct_tet_ned()
    ls = nd1.generate()
    # TODO make this a proper test
    for dof in ls:
        # dof_dict = dof.to_quadrature(1)
        # print(np.array(list(dof_dict.keys())[0]), list(dof_dict.values()))
        print(dof)
    # plot_tet_ned()
    nd1.to_fiat()


def plot_tet_ned():
    ned = construct_tet_ned()
    ned.plot(filename="tet_nd2_fiat.png")
