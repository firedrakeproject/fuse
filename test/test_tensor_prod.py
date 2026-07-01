import pytest
import numpy as np
from fuse import *
from firedrake import *
from test_2d_examples_docs import construct_cg1, construct_dg1, construct_dg0_integral, construct_dg1_integral
from test_convert_to_fiat import create_cg2, create_dg0, helmholtz_solve as helmholtz_solve2
from fuse.tensor_products import HDiv as HDiv_fuse
# from test_convert_to_fiat import create_cg1


def create_cg3_interval(cell=None):
    if cell is None:
        cell = line()
    deg = 3
    if cell.dim() > 1:
        raise NotImplementedError("This method is for cg3 on edges, please use construct_cg3 for triangles")
    vert_dg = create_dg0(cell.vertices()[0])
    xs = [immerse(cell, vert_dg, TrH1)]
    interior = [DOF(DeltaPairing(), PointKernel((-1/np.sqrt(5), )))]

    Pk = PolynomialSpace(deg)
    cg = ElementTriple(cell, (Pk, CellL2, C0), [DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1),
                                                DOFGenerator(interior, S2, S1)])
    return cg

def rt1_quad():
    cg1 = construct_cg1()
    dg0 = construct_dg0_integral()
    return HDiv_fuse(tensor_product(cg1, dg0).flatten()) + HDiv_fuse(tensor_product(dg0, cg1).flatten())

def helmholtz_solve(mesh, V):
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = inner(f, v) * dx
    u = Function(V)
    solve(a == L, u)
    f.interpolate(cos(x*pi*2)*cos(y*pi*2))
    print("res", u.dat.data)
    print("true", f.dat.data)
    return sqrt(assemble(dot(u - f, u - f) * dx))


def mass_solve(U):
    f = Function(U)
    f.assign(1)

    out = Function(U)
    u = TrialFunction(U)
    v = TestFunction(U)
    a = inner(u, v)*dx
    L = inner(f, v)*dx
    assemble(L)
    solve(a == L, out)
    assert np.allclose(out.dat.data, f.dat.data, rtol=1e-5)
    return out.dat.data


@pytest.mark.parametrize("generator1, generator2, code1, code2, deg1, deg2",
                         [(construct_cg1, construct_cg1, "CG", "CG", 1, 1),
                          (construct_dg1, construct_dg1, "DG", "DG", 1, 1),
                          (construct_dg1, construct_cg1, "DG", "CG", 1, 1),
                          (construct_dg1_integral, construct_cg1, "DG", "CG", 1, 1)])
def test_ext_mesh(generator1, generator2, code1, code2, deg1, deg2):
    m = UnitIntervalMesh(2, use_fuse=True)
    mesh = ExtrudedMesh(m, 2)

    # manual method of creating tensor product elements
    horiz_elt = FiniteElement(code1, as_cell("interval"), deg1)
    vert_elt = FiniteElement(code2, as_cell("interval"), deg2)
    elt = TensorProductElement(horiz_elt, vert_elt)
    U = FunctionSpace(mesh, elt)
    res1 = mass_solve(U)

    # fuseonic way of creating tensor product elements
    A = generator1()
    B = generator2()
    elem = tensor_product(A, B)

    U = FunctionSpace(mesh, elem.to_ufl())
    res2 = mass_solve(U)

    assert np.allclose(res1, res2)


@pytest.mark.parametrize(["elem_gen", "elem_code", "deg", "conv_rate"], [(construct_cg1, "CG", 1, 1.8),
                                                                         (create_cg2, "CG", 2, 3.8),
                                                                         (create_cg3_interval, "CG", 3, 4.8)])
def test_helmholtz(elem_gen, elem_code, deg, conv_rate):
    vals = range(3, 6)
    res = []
    for r in vals:
        m = UnitIntervalMesh(2**r, use_fuse=True)
        mesh = ExtrudedMesh(m, 2**r)

        A = elem_gen()
        B = elem_gen()
        elem = tensor_product(A, B)

        U = FunctionSpace(mesh, elem.to_ufl())
        res += [helmholtz_solve(mesh, U)]
    print("l2 error norms:", res)
    res = np.array(res)
    conv = np.log2(res[:-1] / res[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > conv_rate).all()

def project_expr(mesh, U, expr):
    x = SpatialCoordinate(mesh)
    f = assemble(project(expr(x), U))
    out = Function(U)
    u = TrialFunction(U)
    v = TestFunction(U)
    a = inner(u, v)*dx
    L = inner(f, v)*dx
    solve(a == L, out)
    res = sqrt(assemble(dot(out - func, out - func) * dx))
    return res

@pytest.mark.parametrize(["elem_gen", "elem_code", "deg", "conv_rate"], [(rt1_quad, "RT", 1, 0.8)])
def test_project_vec_quad(elem_gen, elem_code, deg, conv_rate):
    vals = range(3, 6)
    function = lambda x, i: cos((3/4)*pi*x[i])
    expr = lambda x: as_vector([function(x, 0), function(x, 1)])
    res = []
    res_ufc = []
    for r in vals:
        mesh_fuse = UnitSquareMesh(2**r, 2**r, use_fuse=True)
        U = FunctionSpace(mesh_fuse, elem_gen().to_ufl())
        res += [project_expr(mesh_fuse, U, expr)]

    print("l2 error norms:", res)
    res = np.array(res)
    conv = np.log2(res[:-1] / res[1:])
    print("convergence order:", conv)

    assert (np.array(conv) > conv_rate).all()


@pytest.mark.parametrize(["elem_gen", "elem_code", "deg", "conv_rate"], [(construct_cg1, "CG", 1, 1.8),
                                                                         (create_cg2, "CG", 2, 3.8),
                                                                         (create_cg3_interval, "CG", 3, 4.8)])
def test_helmholtz_3d(elem_gen, elem_code, deg, conv_rate):
    vals = range(2, 4)
    res = []
    for r in vals:
        # m = UnitIntervalMesh(2**r, use_fuse=True)
        # m2 = ExtrudedMesh(m, 2**r)
        # mesh = ExtrudedMesh(m2, 2**r)

        # A = elem_gen()
        # B = elem_gen()
        # C = elem_gen()
        # elem = tensor_product(A, B, C)

        # U = FunctionSpace(mesh, elem.to_ufl())
        # res += [helmholtz_solve(mesh, U)]

        m = UnitIntervalMesh(2**r)
        m2 = ExtrudedMesh(m, 2**r)
        mesh_ufc = ExtrudedMesh(m2, 2**r)
        U = FunctionSpace(mesh_ufc, elem_code, deg)
        res += [helmholtz_solve(mesh_ufc, U)]
    print("l2 error norms:", res)
    res = np.array(res)
    conv = np.log2(res[:-1] / res[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > conv_rate).all()


def test_on_quad_mesh():
    quadrilateral = True
    r = 3
    m = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral, use_fuse=True)
    A = construct_cg1()
    B = construct_cg1()
    elem = tensor_product(A, B)
    elem = elem.flatten()
    U = FunctionSpace(m, elem.to_ufl())
    mass_solve(U)

    U = FunctionSpace(m, "CG", 1)
    mass_solve(U)


def test_cg3():
    r = 1
    mesh = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=True)
    res_fuse = []
    A = create_cg3_interval()
    B = create_cg3_interval()
    # elem = symmetric_tensor_product(A, B, matrices=False).flatten()
    # U = FunctionSpace(mesh, elem.to_ufl())
    # res_fuse += [helmholtz_solve(mesh, U)]
    elem = symmetric_tensor_product(A, B).flatten()
    U = FunctionSpace(mesh, elem.to_ufl())
    res_fuse += [helmholtz_solve(mesh, U)]
    assert all(np.array(res_fuse) < 0.003)


@pytest.mark.parametrize(["elem_gen", "elem_code", "deg", "conv_rate"], [(construct_cg1, "CG", 1, 1.8),
                                                                         (create_cg2, "CG", 2, 3.8),
                                                                         (create_cg3_interval, "CG", 3, 4.8)])
def test_quad_mesh_helmholtz(elem_gen, elem_code, deg, conv_rate):
    quadrilateral = True
    vals = range(3, 6)
    res_fuse = []
    res_fire = []
    for r in vals:
        mesh_fuse = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral, use_fuse=True)
        A = elem_gen()
        B = elem_gen()
        elem = symmetric_tensor_product(A, B).flatten()
        U = FunctionSpace(mesh_fuse, elem.to_ufl())
        res_fuse += [helmholtz_solve(mesh_fuse, U)]

        mesh_ufc = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral)
        U = FunctionSpace(mesh_ufc, elem_code, deg)
        res_fire += [helmholtz_solve(mesh_ufc, U)]
    print("Fuse l2 error norms:", res_fuse)
    res = np.array(res_fuse)
    conv = np.log2(res[:-1] / res[1:])
    print("Fuse convergence order:", conv)
    assert (np.array(conv) > conv_rate).all()

    print("FIAT l2 error norms:", res_fire)
    res = np.array(res_fire)
    conv = np.log2(res[:-1] / res[1:])
    print("Fiat convergence order:", conv)
    assert (np.array(conv) > conv_rate).all()


@pytest.mark.parametrize(["elem_gen", "elem_code", "deg", "conv_rate"], [(construct_cg1, "CG", 1, 1.7),
                                                                         (create_cg2, "CG", 2, 3.8),
                                                                         (create_cg3_interval, "CG", 3, 4.8)])
def test_ext_mesh_helmholtz_3d(elem_gen, elem_code, deg, conv_rate):
    vals = range(2, 4)
    res_fuse = []
    res_fire = []
    for r in vals:
        mesh_fuse = ExtrudedMesh(UnitSquareMesh(2 ** r, 2 ** r, use_fuse=True), 2**r)
        A = elem_gen()
        B = elem_gen()
        C = elem_gen()
        elem = symmetric_tensor_product(A, B, C, matrices=False)
        U = FunctionSpace(mesh_fuse, elem.to_ufl())
        res_fuse += [helmholtz_solve2(U, mesh_fuse)]

        mesh_ufc = ExtrudedMesh(UnitSquareMesh(2 ** r, 2 ** r), 2**r)
        U = FunctionSpace(mesh_ufc, elem_code, deg)
        res_fire += [helmholtz_solve2(U, mesh_ufc)]
    print("Fuse l2 error norms:", res_fuse)
    res = np.array(res_fuse)
    conv = np.log2(res[:-1] / res[1:])
    print("Fuse convergence order:", conv)

    print("FIAT l2 error norms:", res_fire)
    res = np.array(res_fire)
    conv = np.log2(res[:-1] / res[1:])
    print("Fiat convergence order:", conv)
    assert (np.array(conv) > conv_rate).all()
    assert (np.array(conv) > conv_rate).all()


@pytest.mark.parametrize(["elem_gen", "elem_code", "deg", "conv_rate"], [(construct_cg1, "CG", 1, 1.7),
                                                                         (create_cg2, "CG", 2, 3.8),
                                                                         (create_cg3_interval, "CG", 3, 4.8)])
def test_quad_mesh_helmholtz_3d(elem_gen, elem_code, deg, conv_rate):
    vals = range(2, 4)
    res_fuse = []
    res_fire = []
    for r in vals:
        mesh_fuse = UnitCubeMesh(2 ** r, 2 ** r, 2 ** r, hexahedral=True, use_fuse=True)
        A = elem_gen()
        B = elem_gen()
        C = elem_gen()
        elem = symmetric_tensor_product(A, B, C).flatten()
        U = FunctionSpace(mesh_fuse, elem.to_ufl())
        res_fuse += [helmholtz_solve2(U, mesh_fuse)]

        mesh_ufc = UnitCubeMesh(2 ** r, 2 ** r, 2 ** r, hexahedral=True)
        U = FunctionSpace(mesh_ufc, elem_code, deg)
        res_fire += [helmholtz_solve2(U, mesh_ufc)]
    print("Fuse l2 error norms:", res_fuse)
    res = np.array(res_fuse)
    conv = np.log2(res[:-1] / res[1:])
    print("Fuse convergence order:", conv)

    print("FIAT l2 error norms:", res_fire)
    res = np.array(res_fire)
    conv = np.log2(res[:-1] / res[1:])
    print("Fiat convergence order:", conv)
    assert (np.array(conv) > conv_rate).all()
    assert (np.array(conv) > conv_rate).all()


@pytest.mark.parametrize(["A", "B", "res"], [(Point(0), line(), False),
                                             (line(), line(), True),
                                             (polygon(3), line(), False),])
def test_flattening(A, B, res):
    tensor_cell = TensorProductPoint(A, B)
    if not res:
        with pytest.raises(AssertionError):
            tensor_cell.flatten()
    else:
        cell = tensor_cell.flatten()
        cell.construct_fuse_rep()


@pytest.mark.parametrize(["A", "B", "C"], [(line(), line(), line()),])
def test_creation(A, B, C):
    tensor_cell_2d = TensorProductPoint(A, B)
    tensor_cell_2d.to_ufl()
    tensor_cell_2d.to_fiat()
    flat_tensor_cell_2d = tensor_cell_2d.flatten()
    print(flat_tensor_cell_2d)
    tensor_cell_3d = TensorProductPoint(A, B, C)
    tensor_cell_3d.to_ufl()
    tensor_cell_3d.to_fiat()
    flat_tensor_cell_3d = tensor_cell_3d.flatten()
    print(flat_tensor_cell_3d)


def test_cg1_dg0():
    A = construct_cg1()
    B = construct_dg1_integral()
    ab = tensor_product(A, B).flatten()
    ba = tensor_product(B, A).flatten()
    combined = ab + ba
    combined.symmetric = True
    mesh1 = UnitSquareMesh(2, 2, quadrilateral=True, use_fuse=True)
    V = FunctionSpace(mesh1, combined.to_ufl())
    ab = tensor_product(A, B)
    ba = tensor_product(B, A)
    combined = ab + ba
    m = UnitIntervalMesh(2, use_fuse=True)
    mesh2 = ExtrudedMesh(m, 2)
    V2 = FunctionSpace(mesh2, combined.to_ufl())
    # CG_1 = FiniteElement("CG", "interval", 1)
    # DG_1 = FiniteElement("DG", "interval", 1)
    # dgcg = TensorProductElement(DG_1, CG_1)
    # cgdg = TensorProductElement(CG_1, DG_1)
    # combined = dgcg + cgdg
    # V = FunctionSpace(mesh, combined)
    Vs = [V2, V]
    meshes = [mesh2, mesh1]
    for V, mesh in zip(Vs, meshes):
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Function(V)
        x, y = SpatialCoordinate(mesh)
        f.project((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
        a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
        L = inner(f, v) * dx
        u = Function(V)
        solve(a == L, u)
        f.project(cos(x*pi*2)*cos(y*pi*2))
        print("res", u.dat.data)
        print("true", f.dat.data)
        res = sqrt(assemble(dot(u - f, u - f) * dx))
        print(res)
    breakpoint()
    # from finat.element_factory import convert
    # non_sym, _ = convert(non_sym.to_ufl(), shift_axes=0)
    # non_sym2, _ = convert(non_sym2.to_ufl(), shift_axes=0)
    # from FIAT.reference_element import flatten_entities
    # print()
    # print(non_sym.entity_dofs())
    # print(non_sym2.entity_dofs())
    # print(flatten_entities(non_sym.entity_dofs()))
    # print(flatten_entities(non_sym2.entity_dofs()))
    print(non_sym)
    A = construct_dg1_integral()
    B = construct_dg1_integral()
    non_sym1 = tensor_product(A, B)
    print(non_sym1)
    breakpoint()


def test_trace_galerkin_projection():
    mesh = UnitSquareMesh(10, 10, quadrilateral=True)

    x, y = SpatialCoordinate(mesh)
    A = construct_cg1()
    B = construct_dg1_integral()
    elem = tensor_product(A, B).flatten()
    elem2 = tensor_product(B, A).flatten()

    # Define the Trace Space
    T = FunctionSpace(mesh, elem.to_ufl() + elem2.to_ufl())

    # Define trial and test functions
    lambdar = TrialFunction(T)
    gammar = TestFunction(T)

    # Define right hand side function

    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    f.interpolate(cos(x*pi*2)*cos(y*pi*2))

    # Construct bilinear form
    a = inner(lambdar, gammar) * ds + inner(lambdar('+'), gammar('+')) * dS

    # Construct linear form
    l = inner(f, gammar) * ds + inner(f('+'), gammar('+')) * dS

    # Compute the solution
    t = Function(T)
    solve(a == l, t, solver_parameters={'ksp_rtol': 1e-14})

    # Compute error in trace norm
    trace_error = sqrt(assemble(FacetArea(mesh)*inner((t - f)('+'), (t - f)('+')) * dS))

    assert trace_error < 1e-13


def test_hdiv():
    # from fuse.tensor_products import HDiv
    np.set_printoptions(linewidth=90, precision=4, suppress=True)
    m = UnitIntervalMesh(2)
    mesh = ExtrudedMesh(m, 2)
    CG_1 = FiniteElement("CG", "interval", 1)
    DG_0 = FiniteElement("DG", "interval", 0)
    cg1 = construct_cg1()
    dg0 = construct_dg0_integral()
    p1p0 = HDiv_fuse(tensor_product(cg1, dg0).flatten()) + HDiv_fuse(tensor_product(dg0, cg1).flatten())
    P1P0 = TensorProductElement(CG_1, DG_0)
    RT_horiz = HDivElement(P1P0)
    P0P1 = TensorProductElement(DG_0, CG_1)

    RT_vert = HDivElement(P0P1)
    elt = RT_horiz + RT_vert
    elt2 = p1p0.to_ufl()
    mesh2 = UnitSquareMesh(2, 2, quadrilateral=True, use_fuse=True)
    # A = construct_cg1()
    # B = construct_dg0_integral()
    # non_sym1 = tensor_product(A, B)
    # # .flatten()
    # non_sym2 = tensor_product(B, A)
    # combined = HDiv(non_sym1) + HDiv(non_sym2)
    # combined = combined
    # combined.symmetric = True
    # elt = combined.flatten().to_ufl()
    V = FunctionSpace(mesh, elt)
    V2 = FunctionSpace(mesh2, elt2)
    for V, mesh in zip([V, V2], (mesh, mesh2)):
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Function(V)
        x, y = SpatialCoordinate(mesh)
        # f_vec = as_vector(((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2), (1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2)))
        f_vec = as_vector((2, 3))
        f = project(f_vec, V)
        a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
        L = inner(f, v) * dx
        u = Function(V)
        solve(a == L, u)
        print(u.dat.data)
    breakpoint()


def test_transforms():
    edge = Point(1, [Point(0), Point(0)], vertex_num=2)
    rev_edge = edge.orient(edge.group.members()[1])
    from fuse.tensor_products import HDiv, HCurl
    cg1 = construct_cg1()
    rev_cg1 = construct_cg1(rev_edge)
    dg0 = construct_dg0_integral()
    rev_dg0 = construct_dg0_integral(rev_edge)
    import gem
    v = gem.Literal(5)
    print("HCurl")
    print(HCurl(tensor_product(dg0, cg1))(v))
    print(HCurl(tensor_product(rev_dg0, cg1))(v))
    print(HCurl(tensor_product(cg1, dg0))(v))
    print(HCurl(tensor_product(cg1, rev_dg0))(v))
    print(HCurl(tensor_product(dg0, rev_cg1))(v))
    print(HCurl(tensor_product(rev_cg1, dg0))(v))
    print("HDiv")
    print(HDiv(tensor_product(dg0, cg1))(v))
    print(HDiv(tensor_product(rev_dg0, cg1))(v))
    print(HDiv(tensor_product(cg1, dg0))(v))
    print(HDiv(tensor_product(cg1, rev_dg0))(v))
    print(HDiv(tensor_product(dg0, rev_cg1))(v))
    print(HDiv(tensor_product(rev_cg1, dg0))(v))


def test_sum_fac():
    # In 2d we have O(N_q^2N_i^4) -> O(p^6)
    # Sum factorisation gains 1 factor so we expect O(p^5)
    # For CG3 p = 3 so it should be 3x faster
    mesh = ExtrudedMesh(UnitIntervalMesh(10), 10)
    A = create_cg3_interval()
    B = create_cg3_interval()
    elem = tensor_product(A, B)
    mesh2 = UnitSquareMesh(10, 10, quadrilateral=True)
    C = create_cg3_interval()
    D = create_cg3_interval()
    elem2 = symmetric_tensor_product(C, D).flatten()
    V = FunctionSpace(mesh, elem.to_ufl())
    V1 = FunctionSpace(mesh, "CG", 3)
    V2 = FunctionSpace(mesh2, elem2.to_ufl())
    V3 = FunctionSpace(mesh2, "CG", 3)
    Vs = [V, V1, V2, V3]
    for V in Vs:
        print(V)
        u = TrialFunction(V)
        v = TestFunction(V)
        a = dot(grad(u), grad(v))*dx  # Laplace operator
        from tsfc import compile_form
        kernel_vanilla, = compile_form(a, parameters={"mode": "vanilla"})
        print("Local assembly FLOPs with vanilla mode is {0:.3g}".format(kernel_vanilla.flop_count))
        kernel_spectral, = compile_form(a)
        print("Local assembly FLOPs with spectral mode is {0:.3g}".format(kernel_spectral.flop_count))
        assert (kernel_vanilla.flop_count / kernel_spectral.flop_count) > 3


# @pytest.mark.xfail(reason="3D tensor products not implemented")
def test_sum_fac_3d():
    # In 2d we have O(N_q^3N_i^6) -> O(p^9)
    # Sum factorisation gains 2 factors so we expect O(p^7)
    # For CG3 p = 3 so it should be 9x faster - seems that it is faster than this in regular firedrake
    mesh = ExtrudedMesh(UnitSquareMesh(10, 10, use_fuse=True), 10)
    A = create_cg3_interval()
    B = create_cg3_interval()
    C = create_cg3_interval()
    elem = tensor_product(A, B, C)
    mesh2 = UnitCubeMesh(10, 10, 10, hexahedral=True, use_fuse=True)
    elem2 = symmetric_tensor_product(A, B, C).flatten()
    V = FunctionSpace(mesh, elem.to_ufl())
    V1 = FunctionSpace(mesh, "CG", 3)
    V2 = FunctionSpace(mesh2, elem2.to_ufl())
    V3 = FunctionSpace(mesh2, "CG", 3)
    Vs = [V, V1, V2, V3]
    for V in Vs:
        print(V)
        u = TrialFunction(V)
        v = TestFunction(V)
        a = dot(grad(u), grad(v))*dx  # Laplace operator
        from tsfc import compile_form
        kernel_vanilla, = compile_form(a, parameters={"mode": "vanilla"})
        print("Local assembly FLOPs with vanilla mode is {0:.3g}".format(kernel_vanilla.flop_count))
        kernel_spectral, = compile_form(a)
        print("Local assembly FLOPs with spectral mode is {0:.3g}".format(kernel_spectral.flop_count))
        assert (kernel_vanilla.flop_count / kernel_spectral.flop_count) > 3
