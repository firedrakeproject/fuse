from fuse import *
from test_convert_to_fiat import create_cg1, create_dg1, construct_cg3, construct_rt, construct_nd
from test_orientations import construct_nd2

import sympy as sp


def test_permute_dg1():
    cell = Point(1, [Point(0), Point(0)], vertex_num=2)

    dg1 = create_dg1(cell)

    for dof in dg1.generate():
        print(dof)

    for g in dg1.cell.group.members():
        print("g", g)
        for dof in dg1.generate():
            print(dof, "->", dof(g))


def test_permute_cg1():
    cell = Point(1, [Point(0), Point(0)], vertex_num=2)

    cg1 = create_cg1(cell)

    for dof in cg1.generate():
        print(dof)

    for g in cg1.cell.group.members():
        print("g", g)
        for dof in cg1.generate():
            print(dof, "->", dof(g))


def test_permute_cg3():
    cell = polygon(3)

    cg3 = construct_cg3(cell)

    for dof in cg3.generate():
        print(dof)

    for g in cg3.cell.group.members():
        print(g)
        for dof in cg3.generate():
            print(dof, "->", dof(g))


def test_permute_rt():
    cell = polygon(3)

    rt = construct_rt(cell)
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    func = FuseFunction(sp.Matrix([x, -1/3 + 2*y]), symbols=(x, y))

    for dof in rt.generate():
        print(dof)

    for g in rt.cell.group.members():
        print(g.numeric_rep())
        for dof in rt.generate():
            print(dof, "->", dof(g), "eval, ", dof(g).eval(func))


def test_permute_nd():
    cell = polygon(3)

    nd = construct_nd(cell)
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    func = FuseFunction(sp.Matrix([x, -1/3 + 2*y]), symbols=(x, y))

    # for dof in nd.generate():
    #     print(dof)

    for g in nd.cell.group.members():
        print("g:", g, g.numeric_rep())
        for dof in nd.generate():
            print(dof(g).convert_to_fiat(cell.to_fiat(), 0, (2,)).pt_dict)
            print(dof, "->", dof(g), "eval, ", dof(g).eval(func))


def test_permute_nd2():
    cell = polygon(3)

    nd = construct_nd2(cell)
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    func = FuseFunction(sp.Matrix([x, -1/3 + 2*y]), symbols=(x, y))

    # for dof in nd.generate():
    #     print(dof)

    for g in nd.cell.group.members():
        print("g:", g.numeric_rep())
        if g.numeric_rep() == 2:
            for i, dof in enumerate(nd.generate()):
                if 0 < i < 2:
                    print(dof(g).convert_to_fiat(cell.to_fiat(), 2, (2,)).pt_dict)
                    print(dof, "->", dof(g), "eval, ", dof(g).eval(func))


def test_permute_nodes():
    cell = polygon(3)
    cg1 = create_cg1(cell)
    degree = cg1.spaces[0].degree()
    nodes = [d.convert_to_fiat(cell.to_fiat(), degree) for d in cg1.generate()]
    print([n.pt_dict for n in nodes])
    for g in cg1.cell.group.members():
        print(g, [d(g).convert_to_fiat(cell.to_fiat(), degree).pt_dict for d in cg1.generate()])


# def test_edge_parametrisation():
#     tri = polygon(3)
#     for i in tri.d_entities(1):
#         print(tri.generate_facet_parameterisation(i.id))
#     from fuse.dof import ParameterisationKernel
#
#     deg = 2
#     edge = tri.edges()[0]
#     x = sp.Symbol("x")
#     y = sp.Symbol("y")
#
#     xs = [DOF(L2Pairing(), ParameterisationKernel())]
#
#     dofs = DOFGenerator(xs, S2, S2)
#     int_ned1 = ElementTriple(edge, (P1, CellHCurl, C0), dofs)
#
#     xs = [DOF(L2Pairing(), ComponentKernel((0,))),
#           DOF(L2Pairing(), ComponentKernel((1,)))]
#     center_dofs = DOFGenerator(xs, S1, S3)
#     xs = [immerse(tri, int_ned1, TrHCurl)]
#     tri_dofs = DOFGenerator(xs, C3, S1)
#
#     vec_Pk = PolynomialSpace(deg - 1, shape=2)
#     Pk = PolynomialSpace(deg - 1)
#     M = sp.Matrix([[y, -x]])
#     nd = vec_Pk + (Pk.restrict(deg-2, deg-1))*M
#
#     ned = ElementTriple(tri, (nd, CellHCurl, C0), [tri_dofs, center_dofs])
#     for n in ned.generate():
#         print(n)
#
#     from test_orientations import construct_nd2_for_fiat
#
#     ned_fiat = construct_nd2_for_fiat(tri)
#
#     print("fiat")
#     for n in ned_fiat.generate():
#         print(n)


def test_generate_quadrature():
    cell = polygon(3)
    degree = 2
    # elem = create_dg1(cell)
    # elem = create_cg1(cell)
    # elem = construct_nd(cell)
    elem = construct_nd2(cell)
    # elem = construct_nd2_for_fiat(cell)
    from FIAT.nedelec import Nedelec
    fiat_elem = Nedelec(cell.to_fiat(), degree)
    # from FIAT.lagrange import Lagrange
    # fiat_elem = Lagrange(cell.to_fiat(), degree)
    degree = elem.spaces[0].degree()
    print(degree)
    for d in fiat_elem.dual_basis():
        print("fiat", d.pt_dict)
    print()
    for d in elem.generate():
        print("fuse", d.to_quadrature(degree, value_shape=(2,)))

    elem.to_fiat()
