from fuse import *
import numpy as np
import sympy as sp
from sympy.combinatorics import Permutation
from test_2d_examples_docs import construct_cg1
# import matplotlib.pyplot as plt


triangle_nums = [sum(range(i)) for i in range(2, 20)]


def triangle_coords(n, verts=[(-1, -np.sqrt(3)/3), (0, 2*np.sqrt(3)/3), (1, -np.sqrt(3)/3)], scale=0.8):
    assert n in triangle_nums
    if n == 1:
        return [(0, 0)]
    on_edge = triangle_nums.index(n) - 1
    sub_verts = [(scale*x, scale*y) for (x, y) in verts]
    if on_edge > 0:
        ratio = 1 / (on_edge + 1)
        for i in range(on_edge):
            sub_verts += [(sub_verts[0][0] + ratio*(i+1)*(sub_verts[1][0] - sub_verts[0][0]), sub_verts[0][1] + ratio*(i+1)*(sub_verts[1][1] - sub_verts[0][1]))]
            sub_verts += [(sub_verts[0][0] + ratio*(i+1)*(sub_verts[2][0] - sub_verts[0][0]), sub_verts[0][1] + ratio*(i+1)*(sub_verts[2][1] - sub_verts[0][1]))]
            sub_verts += [(sub_verts[2][0] + ratio*(i+1)*(sub_verts[1][0] - sub_verts[2][0]), sub_verts[2][1] + ratio*(i+1)*(sub_verts[1][1] - sub_verts[2][1]))]
        if (n - len(sub_verts)) > 0:
            int_scale = (sub_verts[0][0] + ratio*3*(sub_verts[1][0] - sub_verts[0][0])) / sub_verts[0][0]
            interior = triangle_coords((n - len(sub_verts)), sub_verts[0:3], int_scale)
            sub_verts += interior
    assert len(sub_verts) == n
    return sub_verts


def convert_to_generation(coords, verts=[(-1, -np.sqrt(3)/3), (0, 2*np.sqrt(3)/3), (1, -np.sqrt(3)/3)]):
    coords_S1 = []
    coords_C3 = []
    coords_S3 = []
    n = len(coords)
    center = (sum([x for (x, y) in verts])/len(verts), sum([y for (x, y) in verts])/len(verts))
    if center in coords:
        coords_S1 += [center]
        coords.remove(center)
    divide_1 = ((verts[0][0] + verts[1][0])/2, (verts[0][1] + verts[1][1])/2)
    divide_2 = ((verts[0][0] + verts[2][0])/2, (verts[0][1] + verts[2][1])/2)
    for coord in coords:
        if check_multiple(coord, verts[0]) or (check_multiple(coord, divide_1) and check_below_line(divide_2, (0, 0), coord) <= 0) or (check_multiple(coord, divide_2) and check_below_line(divide_1, (0, 0), coord) <= 0):
            coords_C3 += [coord]
        elif check_below_line(verts[0], (0, 0), coord) == -1 and check_below_line(divide_2, (0, 0), coord) == -1:
            coords_S3 += [coord]
    assert n == len(coords_S1) + len(coords_S3)*6 + len(coords_C3)*3
    return coords_S1, coords_C3, coords_S3


def check_below_line(seg_1, seg_2, coord):
    if seg_1[0] - seg_2[0] == 0:
        if coord[0] == seg_1[0]:
            return 0
        elif coord[0] > seg_1[0]:
            return 1
        else:
            return -1

    if seg_1[1] - seg_2[1] == 0:
        if coord[1] == seg_1[1]:
            return 0
        elif coord[1] > seg_1[1]:
            return 1
        else:
            return -1

    m = (seg_1[1] - seg_2[1])/(seg_1[0] - seg_2[0])
    c = seg_1[1] - seg_1[0]*m
    eq = lambda x: m*x + c

    if eq(coord[0]) == coord[1]:
        return 0
    elif eq(coord[0]) < coord[1]:
        return 1
    else:
        return -1


def check_multiple(coord_1, coord_2):
    return check_below_line(coord_2, (0, 0), coord_1) == 0


def CR_n(cell, deg):
    if deg % 2 == 0:
        raise ValueError(f"Crouzeix-Raviart with Gauss points is only unisolvent for odd degree, not {deg}")
    points = np.polynomial.legendre.leggauss(deg)[0]
    Pk = PolynomialSpace(deg)
    sym_points = [DOF(DeltaPairing(), PointKernel((pt,))) for pt in points[:len(points)//2]]
    if 0 in points:
        edge_dg0 = ElementTriple(cell.edges(get_class=True)[0], (Pk, C0), [DOFGenerator(sym_points, S2),
                                                                           DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1)])
    else:
        edge_dg0 = ElementTriple(cell.edges(get_class=True)[0], (Pk, C0), [DOFGenerator(sym_points, S2)])
    edge_xs = [immerse(cell, edge_dg0)]

    interior_coords = triangle_coords(triangle_nums[deg - 3])
    s1, c3, s3 = convert_to_generation(interior_coords)

    s1 = [DOF(DeltaPairing(), PointKernel(c)) for c in s1]
    c3 = [DOF(DeltaPairing(), PointKernel(c)) for c in c3]
    s3 = [DOF(DeltaPairing(), PointKernel(c)) for c in s3]

    # each C3 point lies on a reflection axis, so it is fixed by that reflection and
    # needs its own generator: two such points together are fixed only by the identity
    c3_gens = [DOFGenerator([c], C3) for c in c3]
    return ElementTriple(cell, (Pk, C0), [DOFGenerator(edge_xs, C3), DOFGenerator(s1, S1)] + c3_gens + [DOFGenerator(s3, S3)])


# coords = triangle_coords(21)
# edge_coords = [(-1, -np.sqrt(3)/3), (0, 2*np.sqrt(3)/3), (1, -np.sqrt(3)/3)]

# ax = plt.gca()
# ax.scatter([e[0] for e in coords], [e[1] for e in coords], color="grey")
# ax.scatter([e[0] for e in edge_coords], [e[1] for e in edge_coords], color="grey")
# ax.figure.savefig("triangle.png")

# s1, c3, s3 = convert_to_generation(coords)
# gen_coords = s1+c3+s3
# ax = plt.gca()
# ax.scatter([e[0] for e in s1], [e[1] for e in s1], color="black")
# ax.scatter([e[0] for e in c3], [e[1] for e in c3], color="green")
# ax.scatter([e[0] for e in s3], [e[1] for e in s3], color="blue")
# # ax.scatter([e[0] for e in edge_coords], [e[1] for e in edge_coords], color="blue")
# # ax.figure.savefig("triangle_gen.png")
# crn = CR_n(polygon(3), 5)
# print(len(crn.generate()))
# for d in crn.generate():
#     print(d)


# Element builders shared across the test suite. They need no Firedrake,
# so tests using them can run where Firedrake isn't installed.


def create_dg0(cell):
    xs = [DOF(DeltaPairing(), PointKernel(cell.vertices(return_coords=True)[0]))]
    Pk = PolynomialSpace(0)
    dg = ElementTriple(cell, (Pk, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices()))))
    return dg


def create_dg1(cell):
    if cell.dim() == 1:
        x, w = np.polynomial.legendre.leggauss(2)
        xs = [DOF(DeltaPairing(), PointKernel((x[0],)))]
    else:
        xs = [DOF(DeltaPairing(), PointKernel(cell.vertices(return_coords=True)[0]))]
    Pk = PolynomialSpace(1)
    dg = ElementTriple(cell, (Pk, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices()))))
    return dg


def create_dg2(cell):
    x, w = np.polynomial.legendre.leggauss(3)
    xs = [DOF(DeltaPairing(), PointKernel((x[0], )))]
    center = [DOF(DeltaPairing(), PointKernel((0,)))]

    Pk = PolynomialSpace(2)
    dg = ElementTriple(cell, (Pk, C0), [DOFGenerator(xs, get_cyc_group(len(cell.vertices()))),
                                        DOFGenerator(center, S1)])
    return dg


def create_dg1_uneven(cell):
    xs = [DOF(DeltaPairing(), PointKernel(-0.75,))]
    center = [DOF(DeltaPairing(), PointKernel((0.25,)))]
    Pk = PolynomialSpace(1)
    dg = ElementTriple(cell, (Pk, C0), [DOFGenerator(xs, S1),
                                        DOFGenerator(center, S1)])
    return dg


def create_dg1_tet(cell):
    xs = [DOF(DeltaPairing(), PointKernel(tuple(cell.vertices(return_coords=True)[0])))]
    dg1 = ElementTriple(cell, (P1, C0), DOFGenerator(xs, Z4))

    return dg1


def create_cr(cell):
    Pk = PolynomialSpace(1)
    edge_dg0 = ElementTriple(cell.edges(get_class=True)[0], (Pk, C0), DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1))
    edge_xs = [immerse(cell, edge_dg0)]

    return ElementTriple(cell, (Pk, C0), [DOFGenerator(edge_xs, C3)])


def create_cr3(cell):
    Pk = PolynomialSpace(3)
    edge_dg0 = ElementTriple(cell.edges(get_class=True)[0], (Pk, C0), [DOFGenerator([DOF(DeltaPairing(), PointKernel((-np.sqrt(3/5),)))], S2),
                                                                       DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1)])
    edge_xs = [immerse(cell, edge_dg0)]
    center = [DOF(DeltaPairing(), PointKernel((0, 0)))]

    return ElementTriple(cell, (Pk, C0), [DOFGenerator(edge_xs, C3), DOFGenerator(center, S1)])


def create_fortin_soulie(cell):
    Pk = PolynomialSpace(2)
    edge_2 = ElementTriple(cell.edges(get_class=True)[0], (Pk, C0), [DOFGenerator([DOF(DeltaPairing(), PointKernel((-1/3,)))], S2)])
    edge_1 = ElementTriple(cell.edges(get_class=True)[0], (Pk, C0), [DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1)])
    edge_2xs = [immerse(cell, edge_2)]
    edge_1xs = [immerse(cell, edge_1, node=1)]

    group_2 = PermutationSetRepresentation([Permutation([2, 0, 1]), Permutation([0, 1, 2])])
    return ElementTriple(cell, (Pk, C0), [DOFGenerator(edge_2xs, group_2), DOFGenerator(edge_1xs, S1)])


def create_cf(cell):
    Pk = PolynomialSpace(3)
    edge_dg0 = ElementTriple(cell.edges(get_class=True)[0], (Pk, C0),
                             [DOFGenerator([DOF(DeltaPairing(), PointKernel((-1/2,)))], S2),
                              DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1)])
    edge_xs = [immerse(cell, edge_dg0)]
    center = [DOF(DeltaPairing(), PointKernel((0, 0)))]

    return ElementTriple(cell, (Pk, C0), [DOFGenerator(edge_xs, C3), DOFGenerator(center, S1)])


def create_cg1(cell):
    deg = 1
    vert_dg = create_dg0(cell.vertices()[0])
    xs = [immerse(cell, vert_dg)]

    Pk = PolynomialSpace(deg)
    cg = ElementTriple(cell, (Pk, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices()))))
    return cg


def create_cg1_quad():
    deg = 1
    cell = polygon(4)
    # cell = constructCellComplex("quadrilateral").cell_complex

    vert_dg = create_dg0(cell.vertices()[0])
    xs = [immerse(cell, vert_dg)]

    Pk = PolynomialSpace(deg, deg + 1)
    cg = ElementTriple(cell, (Pk, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices()))))

    return cg


def create_cg1_quad_tensor():
    A = construct_cg1()
    B = construct_cg1()
    elem = tensor_product(A, B).flatten()
    return elem


def create_cg1_flipped(cell):
    deg = 1
    vert_dg = create_dg0(cell.vertices()[0])
    xs = [immerse(cell, vert_dg, node=1)]

    Pk = PolynomialSpace(deg)
    cg = ElementTriple(cell, (Pk, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices()))))

    for dof in cg.generate():
        print(dof)
    return cg


def create_cg2(cell=None):
    deg = 2
    if cell is None:
        cell = Point(1, [Point(0), Point(0)], vertex_num=2)
    if cell.dim() > 1:
        raise NotImplementedError("This method is for cg2 on edges, please use create_cg2_tri for triangles")
    vert_dg = create_dg0(cell.vertices()[0])
    xs = [immerse(cell, vert_dg)]
    center = [DOF(DeltaPairing(), PointKernel((0,)))]

    Pk = PolynomialSpace(deg)
    cg = ElementTriple(cell, (Pk, C0), [DOFGenerator(xs, get_cyc_group(len(cell.vertices()))),
                                        DOFGenerator(center, S1)])
    return cg


def create_cg2_tri(cell=None):
    deg = 2
    if cell is None:
        cell = polygon(3)
    Pk = PolynomialSpace(deg)

    vert_dg0 = create_dg0(cell.vertices()[0])
    xs = [immerse(cell, vert_dg0)]

    edge_dg0 = ElementTriple(cell.edges(get_class=True)[0], (PolynomialSpace(0), C0), DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1))
    edge_xs = [immerse(cell, edge_dg0)]

    cg = ElementTriple(cell, (Pk, C0), [DOFGenerator(xs, get_cyc_group(len(cell.vertices()))),
                                        DOFGenerator(edge_xs, C3)])
    return cg


def create_cg1_tet(cell):

    vert = cell.vertices()[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, "C0"),
                        DOFGenerator(xs, S1))

    v_xs = [immerse(cell, dg0)]
    cgverts = DOFGenerator(v_xs, Z4)

    cg1 = ElementTriple(cell, (P1, "C0"),
                        [cgverts])

    return cg1


def create_cg2_tet(cell):

    vert = cell.vertices()[0]
    edge = cell.edges()[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, "C0"),
                        DOFGenerator(xs, S1))

    xs = [DOF(DeltaPairing(), PointKernel((0,)))]
    dg1_int = ElementTriple(edge, (P1, "C0"),
                            DOFGenerator(xs, S1))

    v_xs = [immerse(cell, dg0)]
    cgverts = DOFGenerator(v_xs, Z4)

    e_xs = [immerse(cell, dg1_int)]
    cgedges = DOFGenerator(e_xs, tet_edges)

    cg2 = ElementTriple(cell, (P2, "C0"),
                        [cgverts, cgedges])

    return cg2


def create_cg3_tet(cell=None, perm=True):
    if cell is None:
        cell = make_tetrahedron()

    vert = cell.vertices()[0]
    edge = cell.edges()[0]
    face = cell.d_entities(2)[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, "C0"),
                        DOFGenerator(xs, S1))

    xs = [DOF(DeltaPairing(), PointKernel((-1/np.sqrt(5),)))]
    dg1_int = ElementTriple(edge, (P1, "C0"),
                            DOFGenerator(xs, S2))

    xs = [DOF(DeltaPairing(), PointKernel((0, 0)))]
    dg0_face = ElementTriple(face, (P0, "C0"),
                             DOFGenerator(xs, S1))

    v_xs = [immerse(cell, dg0)]
    cgverts = DOFGenerator(v_xs, Z4)

    e_xs = [immerse(cell, dg1_int)]
    cgedges = DOFGenerator(e_xs, tet_edges)

    f_xs = [immerse(cell, dg0_face)]
    cgfaces = DOFGenerator(f_xs, tet_faces)

    cg3 = ElementTriple(cell, (P3, "C0"),
                        [cgverts, cgedges, cgfaces], perm)

    return cg3


def construct_nd2(tri=None):
    if tri is None:
        tri = polygon(3)
    deg = 2
    edge = tri.edges()[0]
    x = sp.Symbol("x")
    y = sp.Symbol("y")

    xs = [DOF(L2Pairing(), PolynomialKernel((1/2)*(x + 1), symbols=(x,), action=HCurl))]

    dofs = DOFGenerator(xs, S2)
    int_ned1 = ElementTriple(edge, (P1, L2), dofs)
    v_2 = np.array(tri.get_node(tri.ordered_vertices()[2], return_coords=True))
    v_1 = np.array(tri.get_node(tri.ordered_vertices()[1], return_coords=True))
    xs = [DOF(L2Pairing(), VectorKernel((v_2 - v_1)/2, action=HCurl))]
    # the two images span the constant vectors, which every symmetry preserves,
    # so they form one generator
    xs = [x(s) for x in xs for s in tri.group._match(S2.add_cell(tri))]
    center_dofs = DOFGenerator(xs, S1)
    xs = [immerse(tri, int_ned1)]
    tri_dofs = DOFGenerator(xs, C3)

    vec_Pk = PolynomialSpace(deg - 1, shape=2)
    Pk = PolynomialSpace(deg - 1)
    M = sp.Matrix([[y, -x]])
    nd = vec_Pk + (Pk.restrict(deg-2, deg-1))*M

    ned = ElementTriple(tri, (nd, C0), [tri_dofs, center_dofs])
    return ned


def construct_rt2(tri=None):
    if tri is None:
        tri = polygon(3)
    edge = tri.d_entities(1, get_class=True)[0]
    deg = 2
    x = sp.Symbol("x")
    y = sp.Symbol("y")

    xs = [DOF(L2Pairing(), PolynomialKernel((1/2)*(1 + x), symbols=(x,), action=HDiv))]
    dofs = DOFGenerator(xs, S2)
    int_rt2 = ElementTriple(edge, (P1, L2), dofs)

    xs = [immerse(tri, int_rt2)]
    tri_dofs = DOFGenerator(xs, C3)

    v_0 = np.array(tri.get_node(tri.ordered_vertices()[0], return_coords=True))
    v_2 = np.array(tri.get_node(tri.ordered_vertices()[2], return_coords=True))
    v_1 = np.array(tri.get_node(tri.ordered_vertices()[1], return_coords=True))
    i_xs = [DOF(L2Pairing(), VectorKernel((v_2 - v_1)/2, action=HDiv)), DOF(L2Pairing(), VectorKernel((v_2 - v_0)/2, action=HDiv))]
    i_dofs = DOFGenerator(i_xs, S1)

    vec_Pk = PolynomialSpace(deg - 1, shape=2)
    Pk = PolynomialSpace(deg - 1)
    M = sp.Matrix([[x, y]])
    rt_space = vec_Pk + (Pk.restrict(deg-2, deg-1))*M
    rt2 = ElementTriple(tri, (rt_space, C0), [tri_dofs, i_dofs])
    return rt2
