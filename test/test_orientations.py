import pytest
from firedrake import *
from fuse import *
import numpy as np
import sympy as sp
from test_2d_examples_docs import construct_cg3, construct_nd, construct_rt, construct_nd_2nd_kind, construct_bdm, construct_bdm2
from test_convert_to_fiat import create_cg1, create_cg2_tri, create_dg1
import os

np.set_printoptions(linewidth=90, precision=4, suppress=True)


def construct_nd2(tri=None):
    if tri is None:
        tri = polygon(3)
    deg = 2
    edge = tri.edges()[0]
    x = sp.Symbol("x")
    y = sp.Symbol("y")

    xs = [DOF(L2Pairing(), PolynomialKernel((1/2)*(x + 1), symbols=(x,)))]

    dofs = DOFGenerator(xs, S2, S2)
    int_ned1 = ElementTriple(edge, (PolynomialSpace(1, set_shape=True), CellHCurl, C0), dofs)
    xs = [immerse(tri, int_ned1, TrHCurl)]
    tri_dofs = DOFGenerator(xs, C3, S1)

    xs = [DOF(L2Pairing(), VectorKernel(tri.basis_vectors()[0]))]
    center_dofs = DOFGenerator(xs, tri.basis_group, S3)

    vec_Pk = PolynomialSpace(deg - 1, set_shape=True)
    Pk = PolynomialSpace(deg - 1)
    M = sp.Matrix([[y, -x]])
    nd = vec_Pk + (Pk.restrict(deg-2, deg-1))*M

    ned = ElementTriple(tri, (nd, CellHCurl, C0), [tri_dofs, center_dofs])
    return ned


def test_nd2():
    elem = construct_nd2()
    elem.to_fiat()


def construct_rt2(tri=None):
    if tri is None:
        tri = polygon(3)
    edge = tri.d_entities(1, get_class=True)[0]
    deg = 2
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    vecP1 = PolynomialSpace(1, set_shape=True)

    xs = [DOF(L2Pairing(), PolynomialKernel((1/2)*(1 + x), symbols=(x,)))]
    dofs = DOFGenerator(xs, S2, S2)
    int_rt2 = ElementTriple(edge, (vecP1, CellHDiv, C0), dofs)

    xs = [immerse(tri, int_rt2, TrHDiv)]
    tri_dofs = DOFGenerator(xs, C3, S1)

    i_xs = [DOF(L2Pairing(), VectorKernel(tri.basis_vectors()[0]))]
    i_dofs = DOFGenerator(i_xs, tri.basis_group, S3)

    vec_Pk = PolynomialSpace(deg - 1, set_shape=True)
    Pk = PolynomialSpace(deg - 1)
    M = sp.Matrix([[x, y]])
    rt_space = vec_Pk + (Pk.restrict(deg-2, deg-1))*M
    rt2 = ElementTriple(tri, (rt_space, CellHDiv, C0), [tri_dofs, i_dofs])
    return rt2


def construct_nd2_for_fiat(tri=None):
    if tri is None:
        tri = polygon(3)
    deg = 2
    edge = tri.edges()[0]
    x = sp.Symbol("x")
    y = sp.Symbol("y")

    xs = [DOF(L2Pairing(), PolynomialKernel(np.sqrt(2))),
          DOF(L2Pairing(), PolynomialKernel((3*x*np.sqrt(2)/np.sqrt(3)), symbols=[x]))]

    dofs = DOFGenerator(xs, S1, S2)
    int_ned1 = ElementTriple(edge, (P1, CellHCurl, C0), dofs)

    xs = [DOF(L2Pairing(), PolynomialKernel(np.sqrt(2))),
          DOF(L2Pairing(), PolynomialKernel((-np.sqrt(2)/np.sqrt(3))*(6*x + 3), symbols=[x]))]
    dofs = DOFGenerator(xs, S1, S2)
    int_ned2 = ElementTriple(tri.edges()[0], (P1, CellHCurl, C0), dofs)

    xs = [DOF(L2Pairing(), PolynomialKernel(np.sqrt(2))),
          DOF(L2Pairing(), PolynomialKernel((-np.sqrt(2)/np.sqrt(3))*(6*x - 3), symbols=[x]))]
    dofs = DOFGenerator(xs, S1, S2)
    int_ned3 = ElementTriple(tri.edges()[0], (P1, CellHCurl, C0), dofs)

    xs = [DOF(L2Pairing(), ComponentKernel((0,))),
          DOF(L2Pairing(), ComponentKernel((1,)))]
    center_dofs = DOFGenerator(xs, S1, S3)
    xs = [immerse(tri, int_ned2, TrHCurl, node=0)]
    xs += [immerse(tri, int_ned1, TrHCurl, node=1)]
    xs += [immerse(tri, int_ned3, TrHCurl, node=2)]
    tri_dofs = DOFGenerator(xs, S1, S1)

    vec_Pk = PolynomialSpace(deg - 1, set_shape=True)
    Pk = PolynomialSpace(deg - 1)
    M = sp.Matrix([[y, -x]])
    nd = vec_Pk + (Pk.restrict(deg-2, deg-1))*M

    ned = ElementTriple(tri, (nd, CellHCurl, C0), [tri_dofs, center_dofs])
    return ned


@pytest.mark.parametrize("elem_gen,elem_code,deg", [(construct_nd, "N1curl", 1),
                                                    (construct_nd2, "N1curl", 2)])
def test_surface_const_nd(elem_gen, elem_code, deg):
    cell = polygon(3)
    elem = elem_gen(cell)
    ones = as_vector((0, 1))

    for n in range(2, 6):
        mesh = UnitSquareMesh(n, n)
        if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
            V = FunctionSpace(mesh, elem.to_ufl())
        else:
            V = FunctionSpace(mesh, elem_code, deg)
        normal = FacetNormal(mesh)
        ones1 = interpolate(ones, V)
        res1 = assemble(dot(ones1, normal) * ds)

        print(f"{n}: {res1}")
        assert np.allclose(res1, 0)


@pytest.mark.parametrize("elem_gen,elem_code,deg", [(construct_rt, "RT", 1),
                                                    (construct_rt2, "RT", 2)])
def test_surface_const_rt(elem_gen, elem_code, deg):
    cell = polygon(3)
    elem = elem_gen(cell)
    ones = as_vector((1, 0))

    for n in range(1, 6):
        mesh = UnitSquareMesh(n, n)
        V = FunctionSpace(mesh, elem.to_ufl())
        if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
            V = FunctionSpace(mesh, elem.to_ufl())
        else:
            V = FunctionSpace(mesh, "RT", deg)
        normal = FacetNormal(mesh)
        ones1 = interpolate(ones, V)
        res1 = assemble(dot(ones1, normal) * ds)

        print(f"{n}: {res1}")
        assert np.allclose(res1, 0)


@pytest.mark.parametrize("elem_gen,elem_code,deg", [(construct_rt, "RT", 1),
                                                    (construct_rt2, "RT", 2)])
def test_surface_vec_rt(elem_gen, elem_code, deg):
    cell = polygon(3)
    rt_elem = elem_gen(cell)

    for n in range(2, 6):
        mesh = UnitSquareMesh(n, n)
        x, y = SpatialCoordinate(mesh)
        normal = FacetNormal(mesh)
        test_vec = as_vector((-y, x))
        if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
            V = FunctionSpace(mesh, rt_elem.to_ufl())
        else:
            V = FunctionSpace(mesh, "RT", 1)
        vec = interpolate(test_vec, V)
        res = assemble(dot(vec, normal) * ds)
        print(f"div {n}: {res}")
        assert np.allclose(0, res)


@pytest.mark.parametrize("elem_gen,elem_code,deg", [(construct_nd, "N1curl", 1),
                                                    (construct_nd2, "N1curl", 2)])
def test_surface_vec_nd(elem_gen, elem_code, deg):
    cell = polygon(3)
    nd_elem = elem_gen(cell)

    for n in range(2, 6):

        mesh = UnitSquareMesh(n, n)
        x, y = SpatialCoordinate(mesh)
        normal = FacetNormal(mesh)
        test_vec = as_vector((-y, x))
        if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
            V = FunctionSpace(mesh, nd_elem.to_ufl())
        else:
            V = FunctionSpace(mesh, elem_code, deg)
        vec = interpolate(test_vec, V)
        res = assemble(dot(vec, normal) * ds)
        print(f"curl {n}: {res}")
        assert np.allclose(0, res)


def get_expression(V):
    mesh = V.mesh()
    dim = mesh.geometric_dimension
    if dim == 2:
        x, y = SpatialCoordinate(mesh)
    elif dim == 3:
        x, y, z = SpatialCoordinate(mesh)

    shape = V.value_shape
    if dim == 2:
        if len(shape) == 0:
            exact = Function(FunctionSpace(mesh, 'CG', 5))
            expression = x
        elif len(shape) == 1:
            exact = Function(VectorFunctionSpace(mesh, 'CG', 5))
            expression = as_vector([x, y])
    elif dim == 3:
        if len(shape) == 0:
            exact = Function(FunctionSpace(mesh, 'CG', 5))
            expression = x + y + z
        elif len(shape) == 1:
            exact = Function(VectorFunctionSpace(mesh, 'CG', 5))
            expression = as_vector([x, y, z])
    return expression, exact


def interpolate_vs_project(V, expression, exact):
    f = assemble(interpolate(expression, V))
    expect = project(expression, V)
    exact.interpolate(expression)
    return sqrt(assemble(inner((expect - exact), (expect - exact)) * dx)), sqrt(assemble(inner((f - exact), (f - exact)) * dx))


@pytest.mark.parametrize("elem_gen,elem_code,deg,conv_rate", [(construct_cg3, "CG", 3, 3.8), ])
def test_convergence(elem_gen, elem_code, deg, conv_rate):
    cell = polygon(3)
    elem = elem_gen(cell)
    scale_range = range(3, 6)
    diff_proj = [0 for i in scale_range]
    diff_proj1 = [0 for i in scale_range]
    diff_inte = [0 for i in scale_range]
    diff_inte1 = [0 for i in scale_range]
    for n in scale_range:
        mesh = UnitSquareMesh(2**n, 2**n)

        V = FunctionSpace(mesh, elem_code, deg)
        x, y = SpatialCoordinate(mesh)
        expr = cos(x*pi*2)*sin(y*pi*2)
        _, exact = get_expression(V)
        diff_proj[n-min(scale_range)], diff_inte[n-min(scale_range)] = interpolate_vs_project(V, expr, exact)
        if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
            V = FunctionSpace(mesh, elem.to_ufl())
            x, y = SpatialCoordinate(mesh)
            expr = cos(x*pi*2)*sin(y*pi*2)
            _, exact = get_expression(V)
            diff_proj1[n-min(scale_range)], diff_inte1[n-min(scale_range)] = interpolate_vs_project(V, expr, exact)

    print("projection l2 error norms:", diff_proj)
    diff_proj = np.array(diff_proj)
    conv1 = np.log2(diff_proj[:-1] / diff_proj[1:])
    print("convergence order:", conv1)
    # assert all([c > conv_rate for c in conv1])
    print("interpolation l2 error norms:", diff_inte)
    diff_inte = np.array(diff_inte)
    conv1 = np.log2(diff_inte[:-1] / diff_inte[1:])
    print("convergence order:", conv1)

    if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
        print("interpolation l2 error norms:", diff_inte1)
        diff_inte1 = np.array(diff_inte1)
        conv2 = np.log2(diff_inte1[:-1] / diff_inte1[1:])
        print("convergence order:", conv2)
        assert all([c > conv_rate for c in conv2])
    assert all([c > conv_rate for c in conv1])


@pytest.mark.parametrize("elem_gen,elem_code,deg,conv_rate", [(construct_nd, "N1curl", 1, 0.8),
                                                              (construct_nd_2nd_kind, "N2curl", 1, 1.8),
                                                              (construct_rt2, "RT", 2, 1.8),
                                                              (construct_bdm, "BDM", 1, 1.8),
                                                              (construct_bdm2, "BDM", 2, 2.8),
                                                              (construct_nd2, "N1curl", 2, 1.8)])
def test_convergence_vector(elem_gen, elem_code, deg, conv_rate):
    cell = polygon(3)
    elem = elem_gen(cell)
    scale_range = range(3, 6)
    diff_proj = [0 for i in scale_range]
    diff_inte = [0 for i in scale_range]
    for n in scale_range:
        mesh = UnitSquareMesh(2**n, 2**n)
        if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
            V = FunctionSpace(mesh, elem.to_ufl())
        else:
            V = FunctionSpace(mesh, elem_code, deg)
        x, y = SpatialCoordinate(mesh)
        expr = cos(x*pi*2)*sin(y*pi*2)
        expr = as_vector([expr, expr])
        _, exact = get_expression(V)
        diff_proj[n-min(scale_range)], diff_inte[n-min(scale_range)] = interpolate_vs_project(V, expr, exact)

    print("projection l2 error norms:", diff_proj)
    diff_proj = np.array(diff_proj)
    conv1 = np.log2(diff_proj[:-1] / diff_proj[1:])
    print("convergence order:", conv1)
    print("interpolation l2 error norms:", diff_inte)
    diff_inte = np.array(diff_inte)
    conv2 = np.log2(diff_inte[:-1] / diff_inte[1:])
    print("convergence order:", conv1)

    assert all([c > conv_rate for c in conv1])
    assert all([c > conv_rate for c in conv2])


@pytest.mark.parametrize("elem_gen,elem_code,deg", [(create_cg2_tri, "CG", 2),
                                                    (create_cg1, "CG", 1),
                                                    (create_dg1, "DG", 1),
                                                    (construct_cg3, "CG", 3),
                                                    (construct_rt2, "RT", 2),
                                                    (construct_nd2, "N1curl", 2)])
def test_interpolation(elem_gen, elem_code, deg):
    cell = polygon(3)
    elem = elem_gen(cell)
    mesh = UnitSquareMesh(1, 1)

    if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
        V = FunctionSpace(mesh, elem.to_ufl())
    else:
        V = FunctionSpace(mesh, elem_code, deg)

    expression, _ = get_expression(V)
    expect = project(expression, V)
    f = assemble(interpolate(expression, V))
    assert np.allclose(f.dat.data, expect.dat.data, rtol=1e-14)

    expect = project(expression, V)
    v = TestFunction(V)
    u = TrialFunction(V)

    a = inner(u, v) * dx
    L = inner(expect, v) * dx

    solution = Function(V)
    solve(a == L, solution)

    assert norm(assemble(expect - solution)) < 1e-14


@pytest.mark.parametrize("elem_gen,elem_gen2,elem_code,deg,deg2",
                         [(create_cg1, create_cg1, "CG", 1, 1),
                          (create_cg2_tri, create_cg2_tri, "CG", 2, 2),
                          (construct_cg3, construct_cg3, "CG", 3, 3),
                          (construct_nd2, construct_nd2, "N1curl", 2, 2),
                          (construct_rt2, construct_rt2, "RT", 2, 2),
                          ])
def test_two_form(elem_gen, elem_gen2, elem_code, deg, deg2):

    cell = polygon(3)
    mesh = UnitSquareMesh(3, 3)

    spaces = []
    if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
        elem = elem_gen(cell)
        elem2 = elem_gen2(cell)
        spaces += [("fuse", FunctionSpace(mesh, elem.to_ufl()), FunctionSpace(mesh, elem2.to_ufl()))]
    else:
        spaces += [("fiat", FunctionSpace(mesh, elem_code, deg), FunctionSpace(mesh, elem_code, deg2))]

    for name, V, V2 in spaces:
        v = TestFunction(V)
        u = TrialFunction(V2)
        exp, _ = get_expression(V)
        f = assemble(interpolate(exp, V2))

        a = inner(v, u) * dx
        L = inner(f, v) * dx

        solution = Function(V2)
        solve(a == L, solution)

        assert norm(assemble(f - solution)) < 1e-14
