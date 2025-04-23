from pytest import *
from fuse import *
from fuse.groups import perm_matrix_to_perm_array
from sympy.combinatorics import Permutation
from test_convert_to_fiat import create_dg1
import numpy as np


def test_numerical_orientations():
    vert = Point(0)
    print(vert.group.compute_num_reps())
    edge = Point(1, [Point(0), Point(0)], vertex_num=2)
    # group = S2.add_cell(edge)
    print(edge.group.compute_num_reps())

    cell = polygon(3)
    # group = S3.add_cell(cell)
    print(cell.group.compute_num_reps())
    print(cell.group.compute_num_reps(base_val=5))
    mems = cell.group.members()
    for m in mems:
        print(m.compute_perm())
        print(m.transform_matrix)


def test_permsets():

    cell = polygon(3)
    p = Permutation([0, 2, 1])
    print(p.size)
    print(p.is_Identity)
    c3 = C3.add_cell(cell)

    deg = 1
    vert_dg = create_dg1(cell.vertices(get_class=True)[0])
    xs = [immerse(cell, vert_dg, TrH1)]

    Pk = PolynomialSpace(deg, deg)
    cg = ElementTriple(cell, (Pk, CellL2, C0), DOFGenerator(xs, c3, S1))

    dofs = cg.generate()

    for d in dofs:
        print(d)

    tri = cell
    edge = tri.edges(get_class=True)[0]
    vert = tri.vertices(get_class=True)[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, CellL2, C0), DOFGenerator(xs, S1, S1))

    v_xs = [immerse(tri, dg0, TrH1)]
    v_dofs = DOFGenerator(v_xs, c3, S1)

    xs = [DOF(DeltaPairing(), PointKernel((-1/3)))]
    dg0_int = ElementTriple(edge, (P1, CellH1, C0), DOFGenerator(xs, S2, S1))
    print([d.generation for d in dg0_int.generate()])

    e_xs = [immerse(tri, dg0_int, TrH1)]
    e_dofs = DOFGenerator(e_xs, c3, S1)

    i_xs = [lambda g: DOF(DeltaPairing(), PointKernel(g((0, 0))))]
    i_dofs = DOFGenerator(i_xs, S1, S1)

    cg3 = ElementTriple(tri, (P3, CellH1, C0), [v_dofs, e_dofs, i_dofs])

    for d in cg3.generate():
        print(d)


def test_conj():
    cell = polygon(3)
    g = cell.group.members()[4]
    print("g", g)

    # print(cell.group.conjugacy_class(g))
    import numpy as np
    for g in cell.group.members():
        print("g", g)
        print("transformed", g((-1/2, -np.sqrt(3)/3)))
        # for row in g.transform_matrix:
        #     print(row)

    # print("tri_c3")
    # g1 = tri_C3.add_cell(cell)
    # for g in g1.members():
    #     print("g", g)
    #     print("g", g.perm.cycle_structure)
    #     # for row in g.matrix_form():
    #     #     print(row)
    # print("others")
    # for g in cell.group.members():
    #     if g not in g1.members():
    #         print("g", g)
    #         print("g", g.perm.cycle_structure)
    #         # for row in g.matrix_form():
    #         #     print(row)
    mems = cell.group.members()
    for m in mems:
        print(m.numeric_rep())


# def test_group_equality():
#     cell = polygon(3)

#     s1 = S1.add_cell(cell)
#     s1_new = S1.add_cell(cell)

#     assert s1 == s1_new

    # cell2 = Point(1, [Point(0), Point(0)], vertex_num=2)

    # s1 = S2.add_cell(cell)
    # s1_new = S2.add_cell(cell2)

    # assert not s1 == s1_new

def test_perm_mat_conversion():
    cell = polygon(3)
    cS3 = S3.add_cell(cell)

    for g in cS3.members():
        mat_form = g.matrix_form()
        array_form = perm_matrix_to_perm_array(mat_form)
        assert np.allclose(g.perm.array_form, array_form)


def test_numeric_reps():
    cell = polygon(4)
    rot4 = get_cyc_group(4).add_cell(cell)

    assert sorted([m.numeric_rep() for m in rot4.members()]) == list(range(len(rot4.members())))
