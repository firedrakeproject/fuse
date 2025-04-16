from fuse import *
import sympy as sp
import numpy as np

np.set_printoptions(legacy="1.25")


def construct_dg0():
    # [test_dg0 0]
    vert = Point(0)
    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, CellL2, C0), DOFGenerator(xs, S1, S1))
    # [test_dg0 1]
    return dg0


def construct_dg0_int():
    edge = Point(1, [Point(0), Point(0)], vertex_num=2)
    xs = [DOF(DeltaPairing(), PointKernel((0,)))]
    dg1 = ElementTriple(edge, (P0, CellL2, C0), DOFGenerator(xs, S1, S1))
    return dg1


def construct_dg1():
    # [test_dg1_int 0]
    edge = Point(1, [Point(0), Point(0)], vertex_num=2)
    xs = [DOF(DeltaPairing(), PointKernel((-1,)))]
    dg1 = ElementTriple(edge, (P1, CellL2, C0), DOFGenerator(xs, S2, S1))
    # [test_dg1_int 1]
    return dg1


def plot_dg1():
    dg1 = construct_dg1()
    dg1.plot()


def construct_dg1_tri():
    # [test_dg1_tri 0]
    tri = polygon(3)
    xs = [DOF(DeltaPairing(), PointKernel((-1, -np.sqrt(3)/3)))]
    dg1 = ElementTriple(tri, (P1, CellL2, C0), DOFGenerator(xs, S3/S2, S1))
    # [test_dg1_tri 1]
    return dg1


def plot_dg1_tri():
    dg1 = construct_dg1_tri()
    dg1.plot()


def test_dg_examples():
    dg0 = construct_dg0()
    test_func = MyTestFunction(lambda: 3)

    for dof in dg0.generate():
        assert np.allclose(dof.eval(test_func), 3)

    dg1 = construct_dg1()
    x = sp.Symbol("x")
    test_func = MyTestFunction(2*x, symbols=(x,))

    dof_vals = [-2, 2]

    for dof in dg1.generate():
        # Avoiding assuming order of generation
        assert any(np.isclose(val, dof.eval(test_func)) for val in dof_vals)

    dg1 = construct_dg1_tri()

    dof_vals = [-11, 2, 9]

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    test_func = MyTestFunction(10*x + 3*y/np.sqrt(3), symbols=(x, y))

    for dof in dg1.generate():
        assert any(np.isclose(val, dof.eval(test_func)) for val in dof_vals)


def construct_cg1():
    # [test_cg1 0]
    edge = Point(1, [Point(0), Point(0)], vertex_num=2)
    vert = edge.vertices()[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, CellL2, C0), DOFGenerator(xs, S1, S1))

    xs = [immerse(edge, dg0, TrH1)]
    cg1 = ElementTriple(edge, (P1, CellH1, C0),
                        DOFGenerator(xs, S2, S1))
    # [test_cg1 1]

    return cg1


def plot_cg1():
    cg1 = construct_cg1()
    cg1.plot()


def construct_cg3(tri=None):
    # [test_cg3 0]
    tri = polygon(3)
    edge = tri.edges()[0]
    vert = tri.vertices()[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, CellL2, C0), DOFGenerator(xs, S1, S1))

    v_xs = [immerse(tri, dg0, TrH1)]
    v_dofs = DOFGenerator(v_xs, C3, S1)

    xs = [DOF(DeltaPairing(), PointKernel((-1/3)))]
    dg0_int = ElementTriple(edge, (P1, CellH1, C0), DOFGenerator(xs, S2, S1))
    print([d.generation for d in dg0_int.generate()])

    e_xs = [immerse(tri, dg0_int, TrH1)]
    e_dofs = DOFGenerator(e_xs, C3, S1)

    i_xs = [DOF(DeltaPairing(), PointKernel((0, 0)))]
    i_dofs = DOFGenerator(i_xs, S1, S1)

    cg3 = ElementTriple(tri, (P3, CellH1, C0), [v_dofs, e_dofs, i_dofs])
    # [test_cg3 1]
    return cg3


def test_plot_cg3():
    cg3 = construct_cg3()
    print(cg3.to_tikz())
    # breakpoint()


def test_cg_examples():

    cg1 = construct_cg1()

    x = sp.Symbol("x")
    test_func = MyTestFunction(2*x, symbols=(x,))

    val_set = set([-2, 2])

    for dof in cg1.generate():
        # Avoiding assuming order of generation
        assert any(np.isclose(val, dof.eval(test_func)) for val in val_set)

    cg3 = construct_cg3()

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    test_func = MyTestFunction(sp.Matrix([10*x, 3*y/np.sqrt(3)]), symbols=(x, y))

    dof_vals = np.array([[-10, -1], [0, 2], [10, -1],
                         [-10/3, 1], [-20/3, 0], [10/3, 1],
                         [20/3, 0], [-10/3, -1], [10/3, -1],
                         [0, 0]])

    for dof in cg3.generate():
        print(dof)
        print(dof.sub_id)
        assert any([np.allclose(val, dof.eval(test_func).flatten()) for val in dof_vals])


def construct_nd(tri=None):
    if tri is None:
        tri = polygon(3)
    deg = 1
    edge = tri.edges()[0]
    x = sp.Symbol("x")
    y = sp.Symbol("y")

    # xs = [DOF(L2Pairing(), PointKernel(edge.basis_vectors()[0]))]
    xs = [DOF(L2Pairing(), PolynomialKernel((1,)))]

    dofs = DOFGenerator(xs, S1, S2)
    int_ned = ElementTriple(edge, (P1, CellHCurl, C0), dofs)

    xs = [immerse(tri, int_ned, TrHCurl)]
    tri_dofs = DOFGenerator(xs, C3, S3)

    M = sp.Matrix([[y, -x]])
    vec_Pk = PolynomialSpace(deg - 1, set_shape=True)
    Pk = PolynomialSpace(deg - 1)
    nd = vec_Pk + (Pk.restrict(deg - 2, deg - 1))*M

    ned = ElementTriple(tri, (nd, CellHCurl, C0), [tri_dofs])
    return ned


def test_nd_example():
    tri = polygon(3)

    ned = construct_nd(tri)

    x = sp.Symbol("x")
    y = sp.Symbol("y")

    phi_2 = MyTestFunction(sp.Matrix([1/3 - (np.sqrt(3)/6)*y, (np.sqrt(3)/6)*x]), symbols=(x, y))
    phi_0 = MyTestFunction(sp.Matrix([-1/6 - (np.sqrt(3)/6)*y, (-np.sqrt(3)/6) + (np.sqrt(3)/6)*x]), symbols=(x, y))
    phi_1 = MyTestFunction(sp.Matrix([-1/6 - (np.sqrt(3)/6)*y,
                                     (np.sqrt(3)/6) + (np.sqrt(3)/6)*x]), symbols=(x, y))
    basis_funcs = [phi_0, phi_1, phi_2]

    for dof in ned.generate():
        assert [np.allclose(1, dof.eval(basis_func).flatten()) for basis_func in basis_funcs].count(True) == 1
        assert [np.allclose(0, dof.eval(basis_func).flatten()) for basis_func in basis_funcs].count(True) == 2


def construct_rt(tri=None):
    if tri is None:
        tri = polygon(3)
    deg = 1
    edge = tri.edges()[0]

    x = sp.Symbol("x")
    y = sp.Symbol("y")

    M = sp.Matrix([[x, y]])
    vec_Pd = PolynomialSpace(deg - 1, set_shape=True)
    Pd = PolynomialSpace(deg - 1)
    rt_space = vec_Pd + (Pd.restrict(deg - 2, deg - 1))*M

    xs = [DOF(L2Pairing(), PolynomialKernel(1))]
    dofs = DOFGenerator(xs, S1, S2)

    int_rt = ElementTriple(edge, (rt_space, CellHDiv, C0), dofs)

    xs = [immerse(tri, int_rt, TrHDiv)]
    tri_dofs = DOFGenerator(xs, C3, S3)

    rt = ElementTriple(tri, (rt_space, CellHDiv, C0), [tri_dofs])
    return rt


def test_rt_example():
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    phi_2 = MyTestFunction(sp.Matrix([(np.sqrt(3)/6)*x,
                                      -1/3 + (np.sqrt(3)/6)*y]), symbols=(x, y))
    phi_0 = MyTestFunction(sp.Matrix([(-np.sqrt(3)/6) + (np.sqrt(3)/6)*x,
                                      1/6 + (np.sqrt(3)/6)*y]), symbols=(x, y))
    phi_1 = MyTestFunction(sp.Matrix([(np.sqrt(3)/6) + (np.sqrt(3)/6)*x,
                                      1/6 + (np.sqrt(3)/6)*y]), symbols=(x, y))

    rt = construct_rt()

    basis_funcs = [phi_0, phi_1, phi_2]

    for dof in rt.generate():
        assert [np.allclose(1, dof.eval(basis_func).flatten()) for basis_func in basis_funcs].count(True) == 1
        assert [np.allclose(0, dof.eval(basis_func).flatten()) for basis_func in basis_funcs].count(True) == 2


def construct_hermite():
    tri = polygon(3)
    vert = tri.vertices()[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, CellL2, C0), DOFGenerator(xs, S1, S1))

    v_xs = [immerse(tri, dg0, TrH1)]
    v_dofs = DOFGenerator(v_xs, S3/S2, S1)

    v_derv_xs = [immerse(tri, dg0, TrGrad)]
    v_derv_dofs = DOFGenerator(v_derv_xs, S3/S2, S1)

    v_derv2_xs = [immerse(tri, dg0, TrHess)]
    v_derv2_dofs = DOFGenerator(v_derv2_xs, S3/S2, S1)

    i_xs = [DOF(DeltaPairing(), PointKernel((0, 0)))]
    i_dofs = DOFGenerator(i_xs, S1, S1)

    her = ElementTriple(tri, (P3, CellH2, C0),
                        [v_dofs, v_derv_dofs, v_derv2_dofs, i_dofs])
    return her


def test_hermite_example():
    her = construct_hermite()

    # TODO improve this test
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    phi_0 = MyTestFunction(x**2 + 3*y**3 + 4*x*y, symbols=(x, y))
    ls = her.generate()
    print("num dofs ", her.num_dofs())
    for dof in ls:
        print(dof)
        print("dof eval", dof.eval(phi_0))


def test_square_cg():
    square = polygon(4)

    vert = square.d_entities(0)[0]
    edge = square.d_entities(1)[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, CellL2, C0),
                        DOFGenerator(xs, S1, S1))

    xs = [DOF(DeltaPairing(), PointKernel((0,)))]
    dg0_int = ElementTriple(edge, (P0, CellL2, C0),
                            DOFGenerator(xs, S1, S1))

    v_xs = [immerse(square, dg0, TrH1)]
    v_dofs = DOFGenerator(v_xs, C4, S1)

    e_xs = [immerse(square, dg0_int, TrH1)]
    e_dofs = DOFGenerator(e_xs, sq_edges, S1)

    i_xs = [lambda g: DOF(DeltaPairing(), PointKernel(g((0, 0))))]
    i_dofs = DOFGenerator(i_xs, S1, S1)

    cg3 = ElementTriple(square, (P3, CellH1, C0),
                        [v_dofs, e_dofs, i_dofs])
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    test_func = MyTestFunction(x**2 + y**2, symbols=(x, y))

    dof_vals = np.array([0, 1, 2])
    for dof in cg3.generate():
        assert any([np.allclose(val, dof.eval(test_func).flatten()) for val in dof_vals])


def test_rt_second_order():
    tri = polygon(3)
    edge = tri.d_entities(1, get_class=True)[0]
    x = sp.Symbol("x")
    y = sp.Symbol("y")

    xs = [DOF(L2Pairing(), PolynomialKernel((1/2)*(1 + x), (x,)))]
    dofs = DOFGenerator(xs, S2, S2)
    int_rt2 = ElementTriple(edge, (P1, CellHDiv, C0), dofs)

    xs = [immerse(tri, int_rt2, TrHDiv)]
    tri_dofs = DOFGenerator(xs, C3, S3)

    i_xs = [lambda g: DOF(L2Pairing(), PointKernel(g((1, 0)))),
            lambda g: DOF(L2Pairing(), PointKernel(g((0, 1))))]
    i_dofs = DOFGenerator(i_xs, S1, S3)

    vecP3 = PolynomialSpace(3, set_shape=True)
    rt2 = ElementTriple(tri, (vecP3, CellHDiv, C0), [tri_dofs, i_dofs])

    phi = MyTestFunction(sp.Matrix([(np.sqrt(3)/6) + (np.sqrt(3)/6)*x,
                                    1/6 + (np.sqrt(3)/6)*y]), symbols=(x, y))

    for dof in rt2.generate():
        dof.eval(phi)
    rt2.plot()
