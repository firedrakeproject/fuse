import unittest.mock as mock
import pytest
from firedrake import *
from fuse import *
import sympy as sp
from test_convert_to_fiat import create_cg1, construct_nd, construct_rt, create_cg2_tri, construct_cg3, create_dg1
import os


def construct_nd2(tri=None):
    if tri is None:
        tri = polygon(3)
    deg = 2
    edge = tri.edges()[0]
    x = sp.Symbol("x")
    y = sp.Symbol("y")

    xs = [DOF(L2Pairing(), PolynomialKernel((1/2)*(x + 1), symbols=(x,)))]

    dofs = DOFGenerator(xs, S2, S2)
    int_ned1 = ElementTriple(edge, (P1, CellHCurl, C0), dofs)
    v_2 = np.array(tri.get_node(tri.ordered_vertices()[2], return_coords = True))
    v_1 = np.array(tri.get_node(tri.ordered_vertices()[1], return_coords = True))
    xs = [DOF(L2Pairing(), PolynomialKernel((v_2 - v_1)/2))]

    center_dofs = DOFGenerator(xs, S2, S3)
    xs = [immerse(tri, int_ned1, TrHCurl)]
    tri_dofs = DOFGenerator(xs, C3, S1)

    vec_Pk = PolynomialSpace(deg - 1, set_shape=True)
    Pk = PolynomialSpace(deg - 1)
    M = sp.Matrix([[y, -x]])
    nd = vec_Pk + (Pk.restrict(deg-2, deg-1))*M

    ned = ElementTriple(tri, (nd, CellHCurl, C0), [tri_dofs, center_dofs])
    return ned


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
        print(V.cell_node_list)
        normal = FacetNormal(mesh)
        ones1 = interpolate(ones, V)
        res1 = assemble(dot(ones1, normal) * ds)

        print(f"{n}: {res1}")
        assert np.allclose(res1, 0)


def test_surface_const_rt():
    cell = polygon(3)
    elem = construct_rt(cell)
    ones = as_vector((1, 0))

    for n in range(1, 6):
        mesh = UnitSquareMesh(n, n)
        V = FunctionSpace(mesh, elem.to_ufl())
        if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
            V = FunctionSpace(mesh, elem.to_ufl())
        else:
            V = FunctionSpace(mesh, "RT", 1)
        normal = FacetNormal(mesh)
        ones1 = interpolate(ones, V)
        res1 = assemble(dot(ones1, normal) * ds)

        print(f"{n}: {res1}")
        assert np.allclose(res1, 0)


@pytest.mark.parametrize("elem_gen,elem_code,deg", [(construct_nd, "N1curl", 1),
                                                    (construct_nd2, "N1curl", 2)])
def test_surface_vec(elem_gen,elem_code,deg):
    cell = polygon(3)
    rt_elem = construct_rt(cell)
    nd_elem = elem_gen(cell)

    for n in range(1, 6):

        mesh = UnitSquareMesh(n, n)
        x, y = SpatialCoordinate(mesh)
        normal = FacetNormal(mesh)
        test_vec = as_vector((-y, x))
        if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
            V = FunctionSpace(mesh, rt_elem.to_ufl())
            vec1 = interpolate(test_vec, V)
            res1 = assemble(dot(vec1, normal) * ds)
        else:
            V = FunctionSpace(mesh, "RT", 1)
            vec2 = interpolate(test_vec, V)
            res1 = assemble(dot(vec2, normal) * ds)
        print(f"div {n}: {res1}")

        if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
            V = FunctionSpace(mesh, nd_elem.to_ufl())
            print(V.cell_node_list)
            vec1 = interpolate(test_vec, V)
            # V2 = FunctionSpace(mesh, create_dg0(cell).to_ufl())
            # u = TestFunction(V2)
            res2 = assemble(dot(vec1, normal) * ds)
        else:
            V = FunctionSpace(mesh, elem_code, deg)
            print(V.cell_node_list)
            vec1 = interpolate(test_vec, V)
            res2 = assemble(dot(vec1, normal) * ds)
        print(f"curl {n}: {res2}")

        assert np.allclose(0, res1)
        assert np.allclose(0, res2)


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
            expression = x + y
        elif len(shape) == 1:
            exact = Function(VectorFunctionSpace(mesh, 'CG', 5))
            expr = x + y
            expression = as_vector([expr, expr])
    elif dim == 3:
        if len(shape) == 0:
            exact = Function(FunctionSpace(mesh, 'CG', 5))
            expression = x + y + z
        elif len(shape) == 1:
            exact = Function(FunctionSpace(mesh, 'CG', 5))
            expression = as_vector([x, y, z])
    return expression, exact

def interpolate_vs_project(V, expression, exact):
    f = assemble(interpolate(expression, V))
    #expect = project(expression, V)
    exact.interpolate(expression)
    print(exact.dat.data)
    #return sqrt(assemble(inner((expect - exact), (expect - exact)) * dx)), 
    print(f.dat.data)
    return 0, sqrt(assemble(inner((f - exact), (f - exact)) * dx)), 




@pytest.mark.parametrize("elem_gen,elem_code,deg,conv_rate", [(construct_nd2, "N1curl", 2, 1.8)])
def test_convergence(elem_gen,elem_code,deg,conv_rate):
    cell = polygon(3)
    elem = elem_gen(cell)
    scale_range = range(0,1)
    diff_proj = [0 for i in scale_range]
    diff_inte = [0 for i in scale_range]
    for n in scale_range:
        mesh = UnitSquareMesh(2**n, 2**n)
    
        if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
            V = FunctionSpace(mesh, elem.to_ufl())
        else:
            V = FunctionSpace(mesh, elem_code, deg)
        print(V.cell_node_list)
        x,y = SpatialCoordinate(mesh)
        expr = cos(x*pi*2)*sin(y*pi*2)
        expr = as_vector([expr,expr])
        _, exact = get_expression(V)
        diff_proj[n-min(scale_range)], diff_inte[n-min(scale_range)] = interpolate_vs_project(V, expr, exact)
    
    #print("projection l2 error norms:", diff_proj)
    #diff_proj = np.array(diff_proj)
    #conv1 = np.log2(diff_proj[:-1] / diff_proj[1:])
    #print("convergence order:", conv1)
    #assert all([c > conv_rate for c in conv1])
    print("interpolation l2 error norms:", diff_inte)
    diff_inte = np.array(diff_inte)
    conv1 = np.log2(diff_inte[:-1] / diff_inte[1:])
    print("convergence order:", conv1)
    assert all([c > conv_rate for c in conv1])


@pytest.mark.parametrize("elem_gen,elem_code,deg", [(create_cg2_tri, "CG", 2),
                                                    (create_cg1, "CG", 1),
                                                    (create_dg1, "DG", 1),
                                                    (construct_cg3, "CG", 3),
                                                    ])
def test_interpolation(elem_gen, elem_code, deg):
    cell = polygon(3)
    elem = elem_gen(cell)
    mesh = UnitSquareMesh(2, 2)

    if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
        V = FunctionSpace(mesh, elem.to_ufl())
    else:
        V = FunctionSpace(mesh, elem_code, deg)
    expression, _ = get_expression(V)
    expect = project(expression, V)
    #f = assemble(interpolate(expression, V))
    print(V.cell_node_list)
    #print(f.dat.data)
    breakpoint()
    assert np.allclose(f.dat.data, expect.dat.data)

@pytest.mark.xfail(reason="issues with tets")
def test_projection_convergence(elem_gen, elem_code, deg, conv_rate):
    cell = make_tetrahedron()
    elem = elem_gen(cell)
    function = lambda x: cos((3/4)*pi*x[0])

    scale_range = range(1, 6)
    diff = [0 for i in scale_range]
    for i in scale_range:
        mesh = UnitSquareMesh(2 ** i, 2 ** i)
        x, y = SpatialCoordinate(mesh)
        expr = cos(x*pi*2)*sin(y*pi*2)
        expr = as_vector([expr, expr])
        expression = as_vector([x, y])
        exact = Function(VectorFunctionSpace(mesh, 'CG', 5))
        exact.interpolate(expr)
        if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
            V2 = FunctionSpace(mesh, elem.to_ufl())
            res2 = project(function(x), V2)
            diff[i - 1] = res2
        else:
            V = FunctionSpace(mesh, elem_code, deg)
            #res = project(function(x), V)
            res = project(expr, V, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})
            diff[i - 1] = sqrt(assemble(inner((res - exact), (res - exact)) * dx))


        #assert np.allclose(res1, res2)

    print("l2 error norms:", diff)
    diff = np.array(diff)
    conv1 = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv1)

    #print("fuse l2 error norms:", diff)
    #diff = np.array(diff)
    #conv2 = np.log2(diff[:-1] / diff[1:])
    #print("fuse convergence order:", conv2)

    assert (np.array(conv1) > conv_rate).all()
    #assert (np.array(conv2) > conv_rate).all()


@pytest.mark.parametrize("elem_gen,elem_code,deg", [
                                                  #  (create_cg2_tri, "CG", 2),
                                                  #  (create_cg1, "CG", 1),
                                                  #  (create_dg1, "DG", 1),
                                                  #  (construct_cg3, "CG", 3),
                                                    (construct_nd, "N1curl", 1),
                                                    ])
def test_two_form(elem_gen, elem_code, deg):
    cell = polygon(3)
    elem = elem_gen(cell)
    mesh = UnitSquareMesh(2, 2)

    if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
        V = FunctionSpace(mesh, elem.to_ufl())
    else:
        V = FunctionSpace(mesh, elem_code, deg)
    if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
        V2 = FunctionSpace(mesh, construct_nd2(cell).to_ufl())
    else:
        V2 = FunctionSpace(mesh, "N1Curl", 2)
    v = TestFunction(V)
    u = TrialFunction(V2)
    assemble(inner(u, v)*dx)

#def test_create_fiat_nd():
#    cell = polygon(3)
#    nd = construct_nd2(cell)
#    nd_fv = construct_nd2_for_fiat(cell)
#    ref_el = cell.to_fiat()
#    deg = 2
#
#    from FIAT.nedelec import Nedelec
#    fiat_elem = Nedelec(ref_el, deg)
#    print("fiat")
#    for f in fiat_elem.dual_basis():
#        print(f.pt_dict)
#
#    #print("fiat: fuse's version")
#    #for d in nd_fv.generate():
#    #    print(d.convert_to_fiat(ref_el, deg).pt_dict)
#
#    print("fuse")
#    dofs = nd.generate()
#    e = cell.edges()[0]
#    s3 = S3.add_cell(cell)
#    g = s3.get_member([1,2,0])
#    print(dofs[-1].to_quadrature(2))
#    print(dofs[-1](g).to_quadrature(2))
#    for d in nd.generate():
#        print(d.convert_to_fiat(ref_el, deg, (2,)).pt_dict)
#    print(g)
#    for d in nd.generate():
#        print(d(g).convert_to_fiat(ref_el, deg, (2,)).pt_dict)
#    #nodes = [d.convert_to_fiat(ref_el, deg, (2,)) for d in dofs]
#    #new_nodes = [d(g).convert_to_fiat(ref_el, deg, (2,)) if d.cell_defined_on == e else d.convert_to_fiat(ref_el, deg, (2,)) for d in dofs]
#    #for i in range(len(new_nodes)):
#    #    print(f"{nodes[i].pt_dict}")
#        # print(f"{dofs[i]}: {new_nodes[i].pt_dict}")
#        # print(f"{dofs[i]}: {new_nodes[i].pt_dict}")
#        # for g in S2.add_cell(cell).members():
#        #     print(d(g))
#        #     print(f"{g} {d(g).convert_to_fiat(ref_el, deg).pt_dict}")
#
#    nd1= construct_nd(cell)
#    print("nd1")
#    for d in nd1.generate():
#        print(d.convert_to_fiat(ref_el, deg).pt_dict)
#    print(g)
#    for d in nd1.generate():
#        print(d(g).convert_to_fiat(ref_el, deg).pt_dict)
#    nd1.to_fiat()
#    
#    nd.to_fiat()
#    #nd_fv.to_fiat()
#
#
#def test_cg3():
#    tri = polygon(3)
#    elem = construct_cg3(tri)
#    for d in elem.generate():
#        print(d.to_quadrature(1))
#    breakpoint()
#
#
#def test_int_nd():
#    tri = polygon(3)
#    deg = 2
#    edge = tri.edges()[0]
#    x = sp.Symbol("x")
#    y = sp.Symbol("y")
#
#    xs = [DOF(L2Pairing(), PolynomialKernel((1/2)*(x + 1), symbols=(x,)))]
#
#    dofs = DOFGenerator(xs, S2, S2)
#    int_ned1 = ElementTriple(edge, (P1, CellHCurl, C0), dofs)
#    dofs = int_ned1.generate()
#    for d in dofs:
#        print(d.to_quadrature(1))
#    breakpoint()
