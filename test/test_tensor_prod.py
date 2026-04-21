import pytest
import numpy as np
from fuse import *
from firedrake import *
from test_2d_examples_docs import construct_cg1, construct_dg1, construct_dg0_integral, construct_dg1_integral
from test_convert_to_fiat import create_cg2
# from test_convert_to_fiat import create_cg1


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
    m = UnitIntervalMesh(2)
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


def test_helmholtz():
    vals = range(3, 6)
    res = []
    for r in vals:
        m = UnitIntervalMesh(2**r)
        mesh = ExtrudedMesh(m, 2**r)

        A = construct_cg1()
        B = construct_cg1()
        elem = tensor_product(A, B)

        U = FunctionSpace(mesh, elem.to_ufl())
        res += [helmholtz_solve(mesh, U)]
    print("l2 error norms:", res)
    res = np.array(res)
    conv = np.log2(res[:-1] / res[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > 1.8).all()


def test_on_quad_mesh():
    quadrilateral = True
    r = 3
    m = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral)
    A = construct_cg1()
    B = construct_cg1()
    elem = tensor_product(A, B)
    elem = elem.flatten()
    U = FunctionSpace(m, elem.to_ufl())
    mass_solve(U)

    U = FunctionSpace(m, "CG", 1)
    mass_solve(U)


@pytest.mark.parametrize(["elem_gen", "elem_code", "deg", "conv_rate"], [(construct_cg1, "CG", 1, 1.8),
                                                                         (create_cg2, "CG", 2, 3.8)])
def test_quad_mesh_helmholtz(elem_gen, elem_code, deg, conv_rate):
    quadrilateral = True
    vals = range(3, 6)
    res_fuse = []
    for r in vals:
        mesh = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral)

        A = elem_gen()
        B = elem_gen()
        elem = tensor_product(A, B).flatten()
        U = FunctionSpace(mesh, elem.to_ufl())
        res_fuse += [helmholtz_solve(mesh, U)]

    print("Fuse l2 error norms:", res_fuse)
    res = np.array(res_fuse)
    conv = np.log2(res[:-1] / res[1:])
    print("Fuse convergence order:", conv)
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


def test_cg1_dg0():
    A = construct_cg1()
    B = construct_dg1_integral()
    non_sym = tensor_product(A, B)
    # non_sym2 = tensor_product(B, A).flatten()
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
    np.set_printoptions(linewidth=90, precision=4, suppress=True)
    m = UnitIntervalMesh(2)
    mesh = ExtrudedMesh(m, 2)
    CG_1 = FiniteElement("CG", "interval", 1)
    DG_0 = FiniteElement("DG", "interval", 0)
    cg1 = construct_cg1()
    dg0 = construct_dg0_integral()
    p1p0 = tensor_product(cg1, dg0).to_ufl()
    P1P0 = TensorProductElement(CG_1, DG_0)
    RT_horiz = HDivElement(p1p0)
    # RT_horiz = HDivElement(P1P0)
    p0p1 = tensor_product(dg0, cg1).to_ufl()
    P0P1 = TensorProductElement(DG_0, CG_1)
    RT_vert = HDivElement(p0p1)
    # RT_vert = HDivElement(P0P1)
    elt = RT_horiz + RT_vert
    # mesh = UnitSquareMesh(1, 1, quadrilateral=True)
    V = FunctionSpace(mesh, elt)
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
    breakpoint()
    # f.interpolate(cos(x*pi*2)*cos(y*pi*2))
    # V.finat_element.basis_evaluation(1, [(0, 0)])
    # tabulation = V.finat_element.fiat_equivalent.tabulate(0, [(0, 0), (1, 0)])
    # for ent, arr in tabulation.items():
    #     print(ent)
    #     for comp in arr:
    #         print(comp[0], comp[1])
