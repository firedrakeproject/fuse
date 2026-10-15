from fuse import *
from sympy.combinatorics import Permutation
import sympy as sp
import numpy as np
np.set_printoptions(linewidth=90, precision=4, suppress=True)


def test_dg1():
    tetra = make_tetrahedron()

    xs = [DOF(DeltaPairing(), PointKernel(tuple(tetra.vertices(return_coords=True)[0])))]
    dg1 = ElementTriple(tetra, (P1, "C0", Fid),
                        DOFGenerator(xs, Z4))

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
    dg0 = ElementTriple(vert, (P0, "C0", Fid),
                        DOFGenerator(xs, S1))

    xs = [DOF(DeltaPairing(), PointKernel((-1/3,)))]
    dg1_int = ElementTriple(edge, (P1, "C0", Fid),
                            DOFGenerator(xs, S2))

    xs = [DOF(DeltaPairing(), PointKernel((0, 0)))]
    dg0_face = ElementTriple(face, (P0, "C0", Fid),
                             DOFGenerator(xs, S1))

    v_xs = [immerse(tetra, dg0)]
    cgverts = DOFGenerator(v_xs, Z4)

    e_xs = [immerse(tetra, dg1_int)]
    cgedges = DOFGenerator(e_xs, tet_edges)

    f_xs = [immerse(tetra, dg0_face)]
    cgfaces = DOFGenerator(f_xs, tet_faces)

    cg3 = ElementTriple(tetra, (P3, "C0", Fid),
                        [cgverts, cgedges, cgfaces])
    # [test_tet_cg3 1]

    return cg3


def construct_tet_cg4(cell=None, perm=True):
    tetra = make_tetrahedron()
    vert = tetra.vertices()[0]
    edge = tetra.edges()[0]
    face = tetra.d_entities(2)[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, "C0", Fid),
                        DOFGenerator(xs, S1), perm)

    xs = [DOF(DeltaPairing(), PointKernel((-np.sqrt(3/7),)))]
    center = [DOF(DeltaPairing(), PointKernel((0,)))]
    dg2_int = ElementTriple(edge, (P2, "C0", Fid),
                            [DOFGenerator(xs, S2), DOFGenerator(center, S1)], perm)

    # xs = [DOF(DeltaPairing(), PointKernel((-1/np.sqrt(5), -0.26)))]
    # xs = [DOF(DeltaPairing(), PointKernel((-0.3919 * 0.8516, -0.226 * 0.8516)))]
    # xs = [DOF(DeltaPairing(), PointKernel((-0.333534405313151, -0.1925661786915493)))]
    xs = [DOF(DeltaPairing(), PointKernel((0.333534405313151, -0.1925661786915493)))]
    # xs = [DOF(DeltaPairing(), PointKernel((0, 2*np.sqrt(3)/9)))]
    dg1_face = ElementTriple(face, (P1, C0, Fid),
                             DOFGenerator(xs, C3), perm)

    xs = [DOF(DeltaPairing(), PointKernel((0, 0, 0)))]
    int_dof = DOFGenerator(xs, S1)

    v_xs = [immerse(tetra, dg0)]
    cgverts = DOFGenerator(v_xs, Z4)

    e_xs = [immerse(tetra, dg2_int)]
    cgedges = DOFGenerator(e_xs, tet_edges)

    f_xs = [immerse(tetra, dg1_face)]
    cgfaces = DOFGenerator(f_xs, tet_faces)
    P4 = PolynomialSpace(4)

    cg4 = ElementTriple(tetra, (P4, "C0", Fid),
                        [cgverts, cgedges, cgfaces, int_dof], perm)

    return cg4


def construct_tet_cg6(cell=None, perm=True):
    deg = 6
    tetra = make_tetrahedron()
    vert = tetra.vertices()[0]
    edge = tetra.edges()[0]
    face = tetra.d_entities(2)[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, "C0", Fid),
                        DOFGenerator(xs, S1), perm)

    xs = [DOF(DeltaPairing(), PointKernel((-0.8302238962785669,)))]
    xs1 = [DOF(DeltaPairing(), PointKernel((-0.4688487934707142),))]
    xs2 = [DOF(DeltaPairing(), PointKernel((0),))]
    dg2_int = ElementTriple(edge, (PolynomialSpace(4), "C0", Fid),
                            [DOFGenerator(xs, S2), DOFGenerator(xs1, S2), DOFGenerator(xs2, S1)], perm)

    # xs = [DOF(DeltaPairing(), PointKernel((-1/np.sqrt(5), -0.26)))]
    # xs = [DOF(DeltaPairing(), PointKernel((-0.3919 * 0.8516, -0.226 * 0.8516)))]

    xs = [DOF(DeltaPairing(), PointKernel((0.656043230571497, -0.37876673577048625)))]
    # xs = [DOF(DeltaPairing(), PointKernel((0, 0.7575334715409713)))]
    xs1 = [DOF(DeltaPairing(), PointKernel((-0.23578311807038468, -0.36380923030819223)))]
    # xs1 = [DOF(DeltaPairing(), PointKernel((0.23578311807038468, -0.36380923030819223)))]
    # xs1 = [DOF(DeltaPairing(), PointKernel((-0.3926918344886, -0.0202164966524)))]
    xs2 = [DOF(DeltaPairing(), PointKernel((0, 0)))]
    # v_0 = np.array(face.get_node(face.ordered_vertices()[0], return_coords=True))
    # v_1 = np.array(face.get_node(face.ordered_vertices()[1], return_coords=True))
    # xs = [DOF(DeltaPairing(), PointKernel(tuple((2*v_0 + v_1)/2)))]
    dg1_face1 = ElementTriple(face, (P1, C0, Fid), DOFGenerator(xs, C3), perm)
    dg1_face2 = ElementTriple(face, (P2, C0, Fid), DOFGenerator(xs1, S3), perm)
    dg1_face3 = ElementTriple(face, (P0, C0, Fid), DOFGenerator(xs2, S1), perm)

    xs = [DOF(DeltaPairing(), PointKernel((0.3130289641134407, 0.31302896411344056, 0.3130289641134407))),]
    # a point towards the midpoint of the first edge, generated like the edges
    xs1 = [DOF(DeltaPairing(), PointKernel((0.0, 0.0, 0.2987804259073821))),]
    # grp = PermutationSetRepresentation([Permutation([0, 1, 2, 3]), Permutation([0, 2, 3, 1]), Permutation([1, 0, 3, 2]), Permutation([0, 3, 1, 2])])
    int_dof = DOFGenerator(xs, Z4)
    int_dof2 = DOFGenerator(xs1, tet_edges)

    v_xs = [immerse(tetra, dg0)]
    cgverts = DOFGenerator(v_xs, Z4)

    e_xs = [immerse(tetra, dg2_int)]
    cgedges = DOFGenerator(e_xs, tet_edges)

    f_xs = [immerse(tetra, dg1_face1), immerse(tetra, dg1_face2), immerse(tetra, dg1_face3)]
    cgfaces1 = DOFGenerator(f_xs, tet_faces)
    P = PolynomialSpace(deg)

    cg = ElementTriple(tetra, (P, "C0", Fid), [cgverts, cgedges, cgfaces1, int_dof, int_dof2], perm)
    assert len(cg.generate()) == (deg + 1)*(deg + 2)*(deg + 3)/6
    cg.to_fiat()
    return cg


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


def test_tet_cg4():
    cg4 = construct_tet_cg4()
    cg4.generate()
    # cg4.plot(filename="tet_cg4.png")
    for dof in cg4.generate():
        print(dof)
    cg4.to_fiat()


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

    vec_Pd = PolynomialSpace(deg - 1, shape=3)
    Pd = PolynomialSpace(deg - 1)
    rt_space = vec_Pd + (Pd.restrict(deg - 2, deg - 1))*M

    xs = [DOF(L2Pairing(), VectorKernel(1, action=HDiv))]
    dofs = DOFGenerator(xs, S1)
    face_vec = ElementTriple(face, (P0, L2, Fdiv), dofs)

    im_xs = [immerse(cell, face_vec)]
    face = DOFGenerator(im_xs, tet_faces)

    rt1 = ElementTriple(cell, (rt_space, "C0", Fdiv),
                        [face])
    # [test_tet_rt 1]
    return rt1


def construct_tet_rt2(cell=None, perm=None):
    if cell is None:
        cell = make_tetrahedron()
    face = cell.d_entities(2, get_class=True)[0]
    deg = 2
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    z = sp.Symbol("z")
    M = sp.Matrix([[x, y, z]])

    vec_Pd = PolynomialSpace(deg - 1, shape=3)
    Pd = PolynomialSpace(deg - 1)
    rt_space = vec_Pd + (Pd.restrict(deg - 2, deg - 1))*M

    xs = [DOF(L2Pairing(), PolynomialKernel(1/3 - (1/2)*x + y/(2*np.sqrt(3)), symbols=(x, y), action=HDiv))]
    dofs = DOFGenerator(xs, C3)
    face_vec = ElementTriple(face, (P1, L2, Fdiv), dofs)

    im_xs = [immerse(cell, face_vec)]
    faces = DOFGenerator(im_xs, tet_faces)

    v_0 = np.array(cell.get_node(cell.ordered_vertices()[0], return_coords=True))
    v_1 = np.array(cell.get_node(cell.ordered_vertices()[1], return_coords=True))
    v_2 = np.array(cell.get_node(cell.ordered_vertices()[2], return_coords=True))
    v_3 = np.array(cell.get_node(cell.ordered_vertices()[3], return_coords=True))
    xs = [DOF(L2Pairing(), VectorKernel((v_2 - v_0)/2, action=HDiv)),
          DOF(L2Pairing(), VectorKernel((v_2 - v_1)/2, action=HDiv)),
          DOF(L2Pairing(), VectorKernel((v_2 - v_3)/2, action=HDiv))]
    interior = DOFGenerator(xs, S1)

    rt2 = ElementTriple(cell, (rt_space, "C0", Fdiv),
                        [faces, interior])
    return rt2


def construct_tet_rt3(cell=None, perm=None):
    if cell is None:
        cell = make_tetrahedron()
    face = cell.d_entities(2, get_class=True)[0]
    deg = 3
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    z = sp.Symbol("z")
    M = sp.Matrix([[x, y, z]])

    vec_Pd = PolynomialSpace(deg - 1, shape=3)
    Pd = PolynomialSpace(deg - 1)
    rt_space = vec_Pd + (Pd.restrict(deg - 2, deg - 1))*M

    s_0 = sp.Symbol("s_0")
    s_1 = sp.Symbol("s_1")
    s_2 = sp.Symbol("s_2")
    symbols = (s_0, s_1, s_2)
    vertex_basis = s_2*(2*s_2 - 1)
    edge_basis = 4*s_1*s_0
    xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(vertex_basis, symbols=symbols, action=HDiv))]
    xs1 = [DOF(L2Pairing(), BarycentricPolynomialKernel(edge_basis, symbols=symbols, action=HDiv))]
    dofs = [DOFGenerator(xs, C3), DOFGenerator(xs1, C3)]
    face_vec = ElementTriple(face, (P2, L2, Fdiv), dofs)

    im_xs = [immerse(cell, face_vec)]
    faces = DOFGenerator(im_xs, tet_faces)

    v_0 = np.array(cell.get_node(cell.ordered_vertices()[0], return_coords=True))
    v_1 = np.array(cell.get_node(cell.ordered_vertices()[1], return_coords=True))
    v_2 = np.array(cell.get_node(cell.ordered_vertices()[2], return_coords=True))
    v_3 = np.array(cell.get_node(cell.ordered_vertices()[3], return_coords=True))
    xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(s_0*(v_2 - v_0)/2, symbols=(s_0, s_1), action=HDiv)),
          DOF(L2Pairing(), BarycentricPolynomialKernel(s_0*(v_2 - v_1)/2, symbols=(s_0, s_1), action=HDiv)),
          DOF(L2Pairing(), BarycentricPolynomialKernel(s_0*(v_2 - v_3)/2, symbols=(s_0, s_1), action=HDiv))]
    interior = DOFGenerator(xs, C4)

    rt2 = ElementTriple(cell, (rt_space, "C0", Fdiv),
                        [faces, interior])
    return rt2


def test_rt3():
    rt3 = construct_tet_rt3()
    rt3.to_fiat()


def construct_tet_bdm_old(cell=None, perm=None):
    if cell is None:
        cell = make_tetrahedron()
    face = cell.d_entities(2, get_class=True)[0]
    deg = 1
    x = sp.Symbol("x")
    y = sp.Symbol("y")

    space = PolynomialSpace(deg, shape=3)

    xs = [DOF(L2Pairing(), PolynomialKernel(1/3 - (1/2)*x + y/(2*np.sqrt(3)), symbols=(x, y), action=HDiv))]
    dofs = DOFGenerator(xs, C3)
    face_vec = ElementTriple(face, (PolynomialSpace(deg), L2, Fdiv), dofs)
    im_xs = [immerse(cell, face_vec)]
    faces = DOFGenerator(im_xs, tet_faces)

    bdm = ElementTriple(cell, (space, "C0", Fdiv), [faces])
    return bdm


def construct_tet_bdm(cell=None, perm=None):
    if cell is None:
        cell = make_tetrahedron()
    face = cell.d_entities(2, get_class=True)[0]
    deg = 1

    space = PolynomialSpace(deg, shape=3)

    s_0 = sp.Symbol("s_0")
    s_1 = sp.Symbol("s_1")
    s_2 = sp.Symbol("s_2")
    xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(1.0*s_2, symbols=(s_0, s_1, s_2), action=HDiv))]
    dofs = DOFGenerator(xs, C3)
    face_vec = ElementTriple(face, (PolynomialSpace(deg), L2, Fdiv), dofs)
    im_xs = [immerse(cell, face_vec)]
    faces = DOFGenerator(im_xs, tet_faces)

    bdm = ElementTriple(cell, (space, C0, Fdiv), [faces])
    return bdm


def construct_tet_bdm2(cell=None, perm=None):
    if cell is None:
        cell = make_tetrahedron()
    face = cell.d_entities(2, get_class=True)[0]
    deg = 2

    space = PolynomialSpace(deg, shape=3)

    s_0 = sp.Symbol("s_0")
    s_1 = sp.Symbol("s_1")
    s_2 = sp.Symbol("s_2")
    symbols = (s_0, s_1, s_2)
    vertex_basis = s_2*(2*s_2 - 1)
    edge_basis = 4*s_1*s_0
    xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(vertex_basis, symbols=symbols, action=HDiv))]
    xs1 = [DOF(L2Pairing(), BarycentricPolynomialKernel(edge_basis, symbols=symbols, action=HDiv))]
    dofs = [DOFGenerator(xs, C3), DOFGenerator(xs1, C3)]
    face_vec = ElementTriple(face, (PolynomialSpace(deg), L2, Fdiv), dofs)
    im_xs = [immerse(cell, face_vec)]
    faces = DOFGenerator(im_xs, tet_faces)

    s_0 = sp.Symbol("s_0")
    s_1 = sp.Symbol("s_1")
    s_2 = sp.Symbol("s_2")
    s_3 = sp.Symbol("s_3")
    symbols = (s_0, s_1, s_2, s_3)
    # a kernel with the symmetries of the first edge, generated like the edges
    xs = [DOF(L2Pairing(), BarycentricPolynomialKernel([s_0 + s_3, -s_0 - s_3, s_3 - s_0], symbols=symbols, action=HDiv))]
    interior = DOFGenerator(xs, tet_edges)

    bdm2 = ElementTriple(cell, (space, "C0", Fdiv), [faces, interior])
    return bdm2


def construct_tet_ned(cell=None):
    deg = 1
    tet = make_tetrahedron()
    edge = tet.edges()[0]

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    z = sp.Symbol("z")

    M1 = sp.Matrix([[0, z, -y]])
    M2 = sp.Matrix([[z, 0, -x]])
    M3 = sp.Matrix([[y, -x, 0]])

    vec_Pd = PolynomialSpace(deg - 1, shape=3)
    Pd = PolynomialSpace(deg - 1)
    nd_space = vec_Pd + (Pd.restrict(deg - 2, deg - 1))*M1 + (Pd.restrict(deg - 2, deg - 1))*M2 + (Pd.restrict(deg - 2, deg - 1))*M3

    # [test_tet_ned 0]
    xs = [DOF(L2Pairing(), VectorKernel(1, action=HCurl))]
    dofs = DOFGenerator(xs, S1)

    edges = ElementTriple(edge, (P0, L2, Fcurl), dofs)
    xs = [immerse(tet, edges)]
    tet_edges = PermutationSetRepresentation([Permutation([0, 1, 2, 3]), Permutation([3, 2, 0, 1]),
                                              Permutation([1, 0, 3, 2]), Permutation([1, 2, 3, 0]),
                                              Permutation([2, 0, 1, 3]), Permutation([2, 3, 1, 0])])
    edge_dofs = DOFGenerator(xs, tet_edges)
    # [test_tet_ned 1]

    return ElementTriple(tet, (nd_space, L2, Fcurl), [edge_dofs])


def construct_tet_ned_2nd_kind_old(tet=None, perm=None):
    if tet is None:
        tet = make_tetrahedron()
    deg = 1
    edge = tet.edges()[0]
    x = sp.Symbol("x")

    vec_Pd = PolynomialSpace(deg, shape=3)
    nd_space = vec_Pd

    xs = [DOF(L2Pairing(), PolynomialKernel((1/2)*(x + 1), symbols=(x,), action=HCurl))]
    dofs = DOFGenerator(xs, S2)
    int_ned1 = ElementTriple(edge, (P1, L2, Fcurl), dofs)

    xs = [immerse(tet, int_ned1)]
    edge_dofs = DOFGenerator(xs, tet_edges)

    ned = ElementTriple(tet, (nd_space, C0, Fcurl), [edge_dofs])
    return ned


def construct_tet_ned_2nd_kind(tet=None, perm=None):
    if tet is None:
        tet = make_tetrahedron()
    deg = 1
    edge = tet.edges()[0]

    vec_Pd = PolynomialSpace(deg, shape=3)
    nd_space = vec_Pd

    s_0 = sp.Symbol("s_0")
    xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(1 - s_0, symbols=(s_0,), action=HCurl))]
    dofs = DOFGenerator(xs, S2)
    int_ned1 = ElementTriple(edge, (P1, L2, Fcurl), dofs)

    xs = [immerse(tet, int_ned1)]
    edge_dofs = DOFGenerator(xs, tet_edges)

    ned = ElementTriple(tet, (nd_space, C0, Fcurl), [edge_dofs])
    return ned


def construct_tet_ned2(tet=None, perm=None):
    if tet is None:
        tet = make_tetrahedron()
    deg = 2
    edge = tet.edges()[0]
    face = tet.d_entities(2, get_class=True)[0]
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    z = sp.Symbol("z")
    M1 = sp.Matrix([[0, z, -y]])
    M2 = sp.Matrix([[z, 0, -x]])
    M3 = sp.Matrix([[y, -x, 0]])

    vec_Pd = PolynomialSpace(deg - 1, shape=3)
    Pd = PolynomialSpace(deg - 1)
    nd_space = vec_Pd + (Pd.restrict(deg - 2, deg - 1))*M1 + (Pd.restrict(deg - 2, deg - 1))*M2 + (Pd.restrict(deg - 2, deg - 1))*M3

    xs = [DOF(L2Pairing(), PolynomialKernel((1/2)*(x + 1), symbols=(x,), action=HCurl))]
    dofs = DOFGenerator(xs, S2)
    int_ned1 = ElementTriple(edge, (P1, L2, Fcurl), dofs)

    v_0 = np.array(face.get_node(face.ordered_vertices()[0], return_coords=True))
    # v_1 = np.array(face.get_node(face.ordered_vertices()[1], return_coords=True))
    v_2 = np.array(face.get_node(face.ordered_vertices()[2], return_coords=True))
    # xs = [DOF(L2Pairing(), VectorKernel((v_2 - v_0)/2))]
    xs = [DOF(L2Pairing(), BarycentricPolynomialKernel((v_2 - v_0)/2, symbols=(x,), action=HCurl))]
    # breakpoint()
    # xs = [DOF(L2Pairing(), VectorKernel((v_2 - v_1)/2))]
    # xs = [DOF(L2Pairing(), VectorKernel((v_1 - v_0)/2)), DOF(L2Pairing(), VectorKernel((v_2 - v_0)/2)),]
    # the two images span the constant vectors, which every symmetry preserves,
    # so they form one generator
    xs = [x(s) for x in xs for s in face.group._match(S2.add_cell(face))]
    center_dofs = DOFGenerator(xs, S1)
    face_vec = ElementTriple(face, (PolynomialSpace(0, shape=2), L2, Fcurl), center_dofs)
    im_xs = [immerse(tet, face_vec)]
    face_dofs = DOFGenerator(im_xs, tet_faces)

    xs = [immerse(tet, int_ned1)]
    tet_edges = PermutationSetRepresentation([Permutation([0, 1, 2, 3]), Permutation([3, 2, 0, 1]),
                                              Permutation([1, 0, 3, 2]), Permutation([1, 2, 3, 0]),
                                              Permutation([2, 0, 1, 3]), Permutation([2, 3, 1, 0])])
    edge_dofs = DOFGenerator(xs, tet_edges)

    ned = ElementTriple(tet, (nd_space, C0, Fcurl), [edge_dofs, face_dofs])
    return ned


def construct_tet_ned_2nd_kind_2(tet=None, both=False):
    if tet is None:
        tet = make_tetrahedron()
    deg = 2
    edge = tet.edges()[0]
    face = tet.d_entities(2, get_class=True)[0]
    vec_Pd = PolynomialSpace(deg, shape=3)
    nd_space = vec_Pd

    s_0 = sp.Symbol("s_0")
    xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(s_0*(2*s_0 - 1), symbols=(s_0,), action=HCurl))]
    centre = [DOF(L2Pairing(), BarycentricPolynomialKernel(4*s_0*(1 - s_0), symbols=(s_0,), action=HCurl))]
    dofs = [DOFGenerator(xs, S2), DOFGenerator(centre, S1)]
    int_ned1 = ElementTriple(edge, (P2, L2, Fcurl), dofs)
    edge_dofs = DOFGenerator([immerse(tet, int_ned1)], tet_edges)

    s_1 = sp.Symbol("s_1")
    s_2 = sp.Symbol("s_2")
    xs = [DOF(L2Pairing(), BarycentricPolynomialKernel([(np.sqrt(3)/6)*(s_2 - s_1 - 1), (s_0 / 2)], symbols=(s_0, s_1, s_2), action=HCurl))]
    # moments against the Raviart-Thomas functions of degree 1 on the face
    xy = sp.Matrix([sp.symbols("x y")])
    face_space = PolynomialSpace(0, shape=2) + PolynomialSpace(0).restrict(-1, 0)*xy
    face_vec = ElementTriple(face, (face_space, L2, Fcurl), DOFGenerator(xs, C3))

    face_dofs = DOFGenerator([immerse(tet, face_vec)], tet_faces)

    ned = ElementTriple(tet, (nd_space, C0, Fcurl), [edge_dofs, face_dofs])
    return ned


def construct_tet_ned_2nd_kind_3(tet=None):
    if tet is None:
        tet = make_tetrahedron()
    deg = 3
    edge = tet.edges()[0]
    face = tet.d_entities(2, get_class=True)[0]
    vec_Pd = PolynomialSpace(deg, shape=3)
    nd_space = vec_Pd

    s_0 = sp.Symbol("s_0")
    s_1 = sp.Symbol("s_1")
    xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(s_1**3, symbols=(s_0, s_1), action=HCurl))]
    centre = [DOF(L2Pairing(), BarycentricPolynomialKernel(3.0*s_0*s_1**2, symbols=(s_0, s_1), action=HCurl))]
    dofs = [DOFGenerator(xs, S2), DOFGenerator(centre, S2)]
    int_ned1 = ElementTriple(edge, (P3, L2, Fcurl), dofs)
    edge_dofs = DOFGenerator([immerse(tet, int_ned1)], tet_edges)

    s_1 = sp.Symbol("s_1")
    s_2 = sp.Symbol("s_2")
    xs = [DOF(L2Pairing(), BarycentricPolynomialKernel([(np.sqrt(3)/6)*s_0*s_1 + (np.sqrt(3)/3)*s_1**2, -0.5*s_0*s_1], symbols=(s_0, s_1, s_2), action=HCurl))]
    xs2 = [DOF(L2Pairing(), BarycentricPolynomialKernel([-0.5*s_0*s_1*s_2, (np.sqrt(3)/2)*s_0*s_1*s_2], symbols=(s_0, s_1, s_2), action=HCurl))]
    # moments against the Raviart-Thomas functions of degree 2 on the face
    xy = sp.Matrix([sp.symbols("x y")])
    face_space = PolynomialSpace(1, shape=2) + PolynomialSpace(1).restrict(0, 1)*xy
    # the two images of the bubble moment span an invariant space, so they form one generator
    xs2 = [x(s) for x in xs2 for s in face.group._match(S2.add_cell(face))]
    face_vec = ElementTriple(face, (face_space, L2, Fcurl), [DOFGenerator(xs, S3), DOFGenerator(xs2, S1)])

    face_dofs = DOFGenerator([immerse(tet, face_vec)], tet_faces)
    # moments against the lowest order Raviart-Thomas field of the face (0, 3, 1),
    # l_0 grad(l_3) x grad(l_1) + l_3 grad(l_1) x grad(l_0) + l_1 grad(l_0) x grad(l_3)
    # in the barycentric coordinates l_i of this tetrahedron, and its images on the other faces
    s_3 = sp.Symbol("s_3")
    phi_0 = [-s_0/4 - s_1/4, -s_1/4 - s_3/4, -s_0/4 - s_3/4]
    xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(phi_0, symbols=(s_0, s_1, s_2, s_3), action=HCurl))]
    interiors = [DOFGenerator(xs, tet_faces)]

    ned = ElementTriple(tet, (nd_space, C0, Fcurl), [edge_dofs, face_dofs] + interiors)
    return ned


def construct_tet_ned_2nd_kind_2_non_bary(tet=None, both=False):
    if tet is None:
        tet = make_tetrahedron()
    deg = 2
    edge = tet.edges()[0]
    face = tet.d_entities(2, get_class=True)[0]
    vec_Pd = PolynomialSpace(deg, shape=3)
    nd_space = vec_Pd

    x = sp.Symbol("x")
    xs = [DOF(L2Pairing(), PolynomialKernel((x/2)*(x + 1), symbols=(x,), action=HCurl))]
    centre = [DOF(L2Pairing(), PolynomialKernel((1 - x**2), symbols=(x,), action=HCurl))]
    dofs = [DOFGenerator(xs, S2), DOFGenerator(centre, S1)]
    int_ned1 = ElementTriple(edge, (P2, L2, Fcurl), dofs)
    edge_dofs = DOFGenerator([immerse(tet, int_ned1)], tet_edges)

    y = sp.Symbol("y")
    rt_vecs = [[(np.sqrt(3)/6)*x, -1/3 + (np.sqrt(3)/6)*y],
               [(-np.sqrt(3)/6) + (np.sqrt(3)/6)*x, 1/6 + (np.sqrt(3)/6)*y],
               [(np.sqrt(3)/6) + (np.sqrt(3)/6)*x, 1/6 + (np.sqrt(3)/6)*y]]
    xs = [DOF(L2Pairing(), PolynomialKernel(rt_vecs[1], symbols=(x, y), action=HCurl))]
    # moments against the Raviart-Thomas functions of degree 1 on the face
    xy = sp.Matrix([sp.symbols("x y")])
    face_space = PolynomialSpace(0, shape=2) + PolynomialSpace(0).restrict(-1, 0)*xy
    face_vec = ElementTriple(face, (face_space, L2, Fcurl), DOFGenerator(xs, C3))

    face_dofs = DOFGenerator([immerse(tet, face_vec)], tet_faces)

    ned = ElementTriple(tet, (nd_space, C0, Fcurl), [edge_dofs, face_dofs])
    return ned


def construct_tet_ned3_old(tet=None, both=False):
    if tet is None:
        tet = make_tetrahedron()
    deg = 3
    edge = tet.edges()[0]
    face = tet.d_entities(2, get_class=True)[0]
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    z = sp.Symbol("z")
    M1 = sp.Matrix([[0, z, -y]])
    M2 = sp.Matrix([[z, 0, -x]])
    M3 = sp.Matrix([[y, -x, 0]])

    vec_Pd = PolynomialSpace(deg - 1, shape=3)
    Pd = PolynomialSpace(deg - 1)
    nd_space = vec_Pd + (Pd.restrict(deg - 2, deg - 1))*M1 + (Pd.restrict(deg - 2, deg - 1))*M2 + (Pd.restrict(deg - 2, deg - 1))*M3

    s_0 = sp.Symbol("s_0")
    s_1 = sp.Symbol("s_1")
    s_2 = sp.Symbol("s_2")
    xs = [DOF(L2Pairing(), BarycentricPolynomialKernel(s_1, symbols=(s_0, s_1), action=HCurl))]
    centre = [DOF(L2Pairing(), BarycentricPolynomialKernel(2*s_0*s_1, symbols=(s_0, s_1), action=HCurl))]
    dofs = [DOFGenerator(xs, S2), DOFGenerator(centre, S1)]
    int_ned1 = ElementTriple(edge, (P2, L2, Fcurl), dofs)
    edge_dofs = DOFGenerator([immerse(tet, int_ned1)], tet_edges)

    v_0 = np.array(face.get_node(face.ordered_vertices()[0], return_coords=True))
    v_1 = np.array(face.get_node(face.ordered_vertices()[1], return_coords=True))
    # v_2 = np.array(face.get_node(face.ordered_vertices()[2], return_coords=True))

    # xs1 = [DOF(L2Pairing(), BarycentricPolynomialKernel((v_2 - v_0)*s_1/2, symbols=(s_0, s_1, s_2)))]
    # xs1 = [DOF(L2Pairing(), BarycentricPolynomialKernel((v_2 - v_1)*(s_1)/2, symbols=(s_0, s_1, s_2)))]ß
    xs1 = [DOF(L2Pairing(), BarycentricPolynomialKernel((v_1 - v_0)*(s_0)/2, symbols=(s_0, s_1, s_2), action=HCurl))]
    # xs1 = [DOF(L2Pairing(), BarycentricPolynomialKernel((v_2 - v_0)*(s_2)/2, symbols=(s_0, s_1, s_2)))]
    # face_vec1 = ElementTriple(face, (P1, C0, Fcurl), DOFGenerator(xs1, C3, S3))
    # face_vec2 = ElementTriple(face, (P1, C0, Fcurl), DOFGenerator(xs2, C3, S3))

    # face_dofs = DOFGenerator([immerse(tet, face_vec1), immerse(tet, face_vec2)], tet_faces, S1)
    # xs1 = [DOF(L2Pairing(), BarycentricPolynomialKernel((v_1 - v_0)*s_0/2, symbols=(s_0, s_1, s_2)))]

    face_vec1 = ElementTriple(face, (PolynomialSpace(1, shape=2), L2, Fid), DOFGenerator(xs1, S3))
    face_dofs = DOFGenerator([immerse(tet, face_vec1)], tet_faces)

    xs = [DOF(L2Pairing(), VectorKernel([1, 0, 0], action=HCurl)),
          DOF(L2Pairing(), VectorKernel([0, 1, 0], action=HCurl)),
          DOF(L2Pairing(), VectorKernel([0, 0, 1], action=HCurl))]
    int_dofs = DOFGenerator(xs, S1)

    ned = ElementTriple(tet, (nd_space, C0, Fcurl), [edge_dofs, face_dofs, int_dofs])
    assert len(ned.generate()) == (1/2)*deg*(deg + 2)*(deg + 3)
    ned.to_fiat()
    return ned


def test_plot_tet_ned2():
    nd = construct_tet_ned2()
    # nd.plot(filename="new.png")
    nd.to_fiat()


def plot_tet_rt():
    rt = construct_tet_rt()
    rt.plot()


def test_tet_rt2():
    rt2 = construct_tet_rt2()
    ls = rt2.generate()
    # TODO make this a proper test
    for dof in ls:
        print(dof.to_quadrature(1, (3,)))
    rt2.to_fiat()


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
        # dof_dict = dof.to_quadrature(1, (3,))
        # print(np.array(list(dof_dict.keys())[0]), list(dof_dict.values()))
        print(dof)
    # plot_tet_ned()
    nd1.to_fiat()


def construct_V(elem):
    nodes = elem.nodes
    ref_el = elem.ref_el
    entity_ids = elem.entity_ids
    poly_set = elem.poly_set
    from FIAT.dual_set import DualSet

    dual = DualSet(nodes, ref_el, entity_ids)

    old_coeffs = poly_set.get_coeffs()
    dualmat = dual.to_riesz(poly_set)

    shp = dualmat.shape
    A = dualmat.reshape((shp[0], -1))
    B = old_coeffs.reshape((shp[0], -1))
    V = np.dot(A, np.transpose(B))
    return V


def inte_mom(pt_dict, fn):
    pts = list(pt_dict.keys())
    wts = np.array([foo[0][0] for foo in list(pt_dict.values())])
    result = np.dot(wts, [fn(p) for p in pts])
    return result


def plot_tet_ned():
    ned = construct_tet_ned()
    ned.plot(filename="tet_nd2_fiat.png")
