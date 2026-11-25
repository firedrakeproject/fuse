import pytest
from fuse import *
import numpy as np
from test_convert_to_fiat import create_dg1, create_cg2, create_cg2_tri
from test_2d_examples_docs import construct_cg3


def create_cg1(cell):
    deg = 1
    vert_dg = create_dg1(cell.vertices()[0])
    xs = [immerse(cell, vert_dg, TrH1)]

    Pk = PolynomialSpace(deg)
    cg = ElementTriple(cell, (Pk, CellL2, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1))
    return cg


@pytest.mark.skipbasix
@pytest.mark.parametrize("elem_gen,deg", [(create_cg1, 1),
                                          (create_cg2, 2)])
def test_basix_conversion_interval(elem_gen, deg):
    import basix
    cell = Point(1, [Point(0), Point(0)], vertex_num=2)
    cg = elem_gen(cell)
    element = cg.to_basix()

    points = np.array([[0.0], [1.0], [0.5], [-1.0]])
    fuse_to_basix = element.tabulate(0, points)
    lagrange = basix.create_element(
        basix.ElementFamily.P, basix.CellType.interval, deg, basix.LagrangeVariant.equispaced
    )
    basix_pts = lagrange.tabulate(0, points)
    assert np.allclose(fuse_to_basix, basix_pts)


@pytest.mark.skipbasix
@pytest.mark.parametrize("elem_gen,deg", [(create_cg1, 1),
                                          (create_cg2_tri, 2),
                                          pytest.param(construct_cg3, 3, marks=pytest.mark.xfail(reason='M needs work'))])
def test_basix_conversion(elem_gen, deg):
    import basix
    cell = polygon(3)
    cg = elem_gen(cell)
    element = cg.to_basix()

    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    fuse_to_basix = element.tabulate(0, points)

    lagrange = basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, deg, basix.LagrangeVariant.equispaced
    )
    basix_pts = lagrange.tabulate(0, points)
    assert np.allclose(fuse_to_basix, basix_pts)


@pytest.mark.skipbasix
def test_cells():
    import basix
    from basix import CellType
    from fuse_basix.basix_interface import ufc_triangle, transform_points

    print(basix.cell.geometry(CellType.triangle))
    print(basix.cell.topology(CellType.triangle))

    tri_fuse = polygon(3)
    tri_basix = ufc_triangle()

    print("RIGHT", tri_basix)
    # print([tri_basix.get_node(i, return_coords=True) for i in tri_basix.ordered_vertices()])
    print(tri_basix.get_topology())
    print(tri_basix.basis_vectors())
    # tri_basix.plot(filename="basix2.png")

    print("FUSE", tri_fuse)
    # print([tri_fuse.get_node(i, return_coords=True) for i in tri_fuse.ordered_vertices()])
    print(tri_fuse.get_topology())
    print(tri_fuse.basis_vectors())
    # tri_fuse.plot(filename="fuse.png")
    print(tri_fuse.vertices(return_coords=True))
    print(transform_points(tri_fuse, tri_basix, tri_fuse.vertices(return_coords=True)))

# def test_del():
#     M = [[], [], [], []]
#     for _ in range(4):
#         M[0].append(np.array([[[[1.0]]]]))
#     M[2].append(np.array([[[[1.0]]]]))

#     # There are no DOFs associates with the edges for this element, so we add an empty
#     # matrix for each edge.

#     for _ in range(4):
#         M[1].append(np.zeros((0, 1, 0, 1)))
#     print(M)

#     x = [[], [], [], []]
#     x[0].append(np.array([[0.0, 0.0]]))
#     x[0].append(np.array([[1.0, 0.0]]))
#     x[0].append(np.array([[0.0, 1.0]]))

#     for _ in range(3):
#         x[1].append(np.zeros((0, 2)))
#     x[2].append(np.zeros((0, 2)))

#     M = [[], [], [], []]
#     for _ in range(3):
#         M[0].append(np.array([[[1.0]]]))

#     for _ in range(3):
#         M[1].append(np.zeros((0, 1, 0)))
#     M[2].append(np.zeros((0, 1, 0, 1)))
#     print(M)
#     # cell = polygon(3)
#     # cg = create_cg1(cell)
#     # x, M = cg.to_basix()
#     # print(M)
