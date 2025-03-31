from fuse import *
import sympy as sp
import numpy as np


def test_dg1():
    tetra = make_tetrahedron()
    

    xs = [DOF(DeltaPairing(), PointKernel(tuple(tetra.vertices(return_coords=True)[0])))]
    dg1 = ElementTriple(tetra, (P1, CellL2, "C0"),
                        DOFGenerator(xs, Z4, S1))

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    z = sp.Symbol("z")
    test_func = MyTestFunction(x + y + z, symbols=(x, y, z))

    dof_vals = [x+y+z for (x, y, z) in tetra.vertices(return_coords=True)]

    for dof in dg1.generate():
        assert any(np.isclose(val, dof.eval(test_func)) for val in dof_vals)


def test_tet_cg3():
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

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    z = sp.Symbol("z")
    test_func = MyTestFunction(sp.Matrix([10*x, 3*y/np.sqrt(3), z*4]), symbols=(x, y, z))

    for dof in cg3.generate():
        print(dof)
        dof.eval(test_func)


def construct_tet_rt(cell):
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

    xs = [DOF(L2Pairing(), PolynomialKernel(1))]
    dofs = DOFGenerator(xs, S1, S3)
    face_vec = ElementTriple(face, (rt_space, CellHDiv, "C0"), dofs)

    im_xs = [immerse(cell, face_vec, TrHDiv)]
    face = DOFGenerator(im_xs, tet_faces, S4)

    rt1 = ElementTriple(cell, (rt_space, CellHDiv, "C0"),
                        [face])
    # [test_tet_rt 1]
    return rt1


def test_tet_rt():
    tetra = make_tetrahedron()
    rt1 = construct_tet_rt(tetra)
    ls = rt1.generate()
    # TODO make this a proper test
    for dof in ls:
        print(dof)


def test_tet_ned():
    tetra = make_tetrahedron()
    edge = tetra.edges()[0]
    # [test_tet_ned 0] 
    xs = [DOF(L2Pairing(), PolynomialKernel(1))]
    dofs = DOFGenerator(xs, S1, S2)
    int_ned = ElementTriple(edge, (P1, CellHCurl, "C0"), dofs)
    ls = int_ned.generate()

    im_xs = [immerse(tetra, int_ned, TrHCurl)]
    edge = DOFGenerator(im_xs, tet_edges, Z4)

    ned = ElementTriple(tetra, (P1, CellHCurl, "C0"),
                        [edge])
    # [test_tet_ned 1]

    ls = ned.generate()
    # TODO make this a proper test
    for dof in ls:
        print(dof)
