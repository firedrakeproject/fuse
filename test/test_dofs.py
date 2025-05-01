from fuse import *
from test_convert_to_fiat import create_cg1, create_dg1, construct_cg3, construct_rt, construct_nd, create_hermite
import sympy as sp
import numpy as np


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
    # func = FuseFunction(sp.Matrix([x, -1/3 + 2*y]), symbols=(x, y))

    # phi_0 = FuseFunction(sp.Matrix([-0.333333333333333*y - 0.192450089729875, 0.333333333333333*x + 0.333333333333333]), symbols=(x, y))
    # phi_1 = FuseFunction(sp.Matrix([0.333333333333333*y + 0.192450089729875, 0.333333333333333 - 0.333333333333333*x]), symbols=(x, y))

    # # original dofs
    phi_2 = FuseFunction(sp.Matrix([1/3 - (np.sqrt(3)/6)*y, (np.sqrt(3)/6)*x]), symbols=(x, y))
    phi_0 = FuseFunction(sp.Matrix([-1/6 - (np.sqrt(3)/6)*y, (-np.sqrt(3)/6) + (np.sqrt(3)/6)*x]), symbols=(x, y))
    phi_1 = FuseFunction(sp.Matrix([-1/6 - (np.sqrt(3)/6)*y,
                                    (np.sqrt(3)/6) + (np.sqrt(3)/6)*x]), symbols=(x, y))

    for g in nd.cell.group.members():
        print(g)
        for dof in nd.generate():
            print(dof, "->", dof(g), "eval p2 ", dof(g).eval(phi_2), "eval p0 ", dof(g).eval(phi_0), "eval p1 ", dof(g).eval(phi_1))

    # reflected dofs
    phi_2 = FuseFunction(sp.Matrix([0.288675134594813*y - 0.333333333333333, -0.288675134594813*x]), symbols=(x, y))
    phi_0 = FuseFunction(sp.Matrix([0.288675134594813*y + 0.166666666666667, -0.288675134594813*x - 0.288675134594813]), symbols=(x, y))
    phi_1 = FuseFunction(sp.Matrix([0.288675134594813*y + 0.166666666666667, 0.288675134594813 - 0.288675134594813*x]), symbols=(x, y))
    reflect = nd.cell.group.get_member([0, 1, 2])
    print(nd.cell.permute_entities(reflect, 1))
    reflect = nd.cell.group.get_member([2, 0, 1])
    print(nd.cell.permute_entities(reflect, 1))
    # print(reflect)
    print(nd.cell.get_topology())
    # nd.cell.plot(filename="test_perms.png")
    for g in nd.cell.group.members():
        print(g)
        for dof in nd.generate():
            print(dof, "->", dof(g), "eval p2 ", dof(g).eval(phi_2), "eval p0 ", dof(g).eval(phi_0), "eval p1 ", dof(g).eval(phi_1))
    #     # print(dof.convert_to_fiat(cell.to_fiat(), 1)(lambda x: np.array([1/3 - (np.sqrt(3)/6)*x[1], (np.sqrt(3)/6)*x[0]])))

    # for g in nd.cell.group.members():
    #     print(g)
    #     print(nd.cell.permute_entities(g, 0))
    #     print(nd.cell.permute_entities(g, 1))


def test_convert_dofs():
    cell = polygon(3)

    cg3 = create_hermite(cell)

    # for dof in cg3.generate():
    #     print(dof)
    #     # print("old")
    #     # # old = dof.convert_to_fiat(cell.to_fiat(), 5).pt_dict
    #     # print(old)
    #     # print("new")
    #     new = dof.convert_to_fiat(cell.to_fiat(), 5)[0].pt_dict
    #     print(new)
    #     new = dof.convert_to_fiat(cell.to_fiat(), 5)[0].deriv_dict
    #     print(new)
    # print(len(cg3.generate()))
    fuse_elem = cg3.to_fiat()
    for n in fuse_elem.dual.nodes:
        print(n.pt_dict)
        print(n.deriv_dict)

    from FIAT.hermite import CubicHermite
    fiat_elem = CubicHermite(cell.to_fiat(), 3)

    print(len([n.pt_dict for n in fiat_elem.dual.nodes]))
    for n in fiat_elem.dual.nodes:
        print(n.pt_dict)
        print(n.deriv_dict)