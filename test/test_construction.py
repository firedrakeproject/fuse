import pytest
import os
import numpy as np
from test_orientations import interpolate_vs_project, get_expression
from test_convert_to_fiat import project as project_test
from fuse.element_construction import periodic_table
from firedrake import *


@pytest.mark.parametrize("col,k,deg", [(col, 3, 0) for col in [0, 1]] + [(col, k, deg) for deg in range(1, 7) for k in [0, 1, 2, 3] for col in [0, 1]])
def test_construction(col, k, deg):
    elem = periodic_table(col, 2, k, deg)
    elem.to_fiat()


@pytest.mark.parametrize("col,k,deg", [(col, 3, 0) for col in [0, 1]] + [(col, k, deg) for deg in range(1, 7) for k in [0, 1, 2, 3] for col in [0, 1]])
def test_construction3d(col, k, deg):
    elem = periodic_table(col, 3, k, deg)
    elem.to_fiat()


cg_params = [(0, 0, deg, deg + 0.8) for deg in list(range(1, 7))] + [(1, 0, deg, deg + 0.8) for deg in list(range(1, 3))]
nd_params = [(0, 1, deg, deg - 0.2) for deg in list(range(1, 7))]
rt_params = [(0, 2, deg, deg - 0.2) for deg in list(range(1, 7))]
dg_params = [(0, 3, deg, deg + 0.8) for deg in list(range(0, 3))] + [(1, 3, deg, deg + 0.8) for deg in list(range(0, 3))]
nd2_params = [(1, 1, deg, deg + 0.8) for deg in list(range(1, 7))]
bdm_params = [(1, 2, deg, deg + 0.8) for deg in list(range(1, 7))]


@pytest.mark.parametrize("col,k,deg,conv_rate", cg_params + nd_params + rt_params + dg_params + nd2_params + bdm_params)
def test_convergence(col, k, deg, conv_rate):
    assert bool(os.environ.get("FIREDRAKE_USE_FUSE", 0))
    elem = periodic_table(col, 2, k, deg)
    scale_range = range(3, 6)
    diff_proj = [0 for i in scale_range]
    diff_inte = [0 for i in scale_range]
    for n in scale_range:
        mesh = UnitSquareMesh(2**n, 2**n)

        V = FunctionSpace(mesh, elem.to_ufl())
        x, y = SpatialCoordinate(mesh)
        expr = cos(x*pi*2)*sin(y*pi*2)
        if len(elem.get_value_shape()) > 0:
            expr = as_vector([expr, expr])
        _, exact = get_expression(V)
        diff_proj[n-min(scale_range)], diff_inte[n-min(scale_range)] = interpolate_vs_project(V, expr, exact)

    print("projection l2 error norms:", diff_proj)
    diff_proj = np.array(diff_proj)
    conv1 = np.log2(diff_proj[:-1] / diff_proj[1:])
    print("convergence order:", conv1)
    # assert all([c > conv_rate for c in conv1])

    print("interpolation l2 error norms:", diff_inte)
    diff_inte = np.array(diff_inte)
    conv2 = np.log2(diff_inte[:-1] / diff_inte[1:])
    print("convergence order:", conv2)
    assert all([c > conv_rate for c in conv2])


cg_params3d = [(0, 0, deg, deg + 0.75) for deg in list(range(1, 6))]
nd_params3d = [(0, 1, deg, deg - 0.2) for deg in list(range(1, 4))]
rt_params3d = [(0, 2, deg, deg - 0.2) for deg in list(range(1, 4))]
dg_params3d = [(0, 3, deg, deg + 0.75) for deg in list(range(0, 4))] + [(1, 3, deg, deg + 0.8) for deg in list(range(0, 3))]
nd2_params3d = [(1, 1, deg, deg + 0.8) for deg in list(range(1, 5))]
bdm_params3d = [(1, 2, deg, deg + 0.8) for deg in list(range(1, 4))]


@pytest.mark.parametrize("col,k,deg,conv_rate", cg_params3d + nd_params3d + rt_params3d + dg_params3d + nd2_params3d + bdm_params3d)
def test_convergence3d(col, k, deg, conv_rate):
    assert bool(os.environ.get("FIREDRAKE_USE_FUSE", 0))
    elem = periodic_table(col, 3, k, deg)

    scale_range = range(2, 4)
    diff_proj = [0 for i in scale_range]
    # diff_proj2 = [0 for i in scale_range]
    for n in scale_range:
        mesh = UnitCubeMesh(2**n, 2**n, 2**n)

        V = FunctionSpace(mesh, elem.to_ufl())
        x, y, z = SpatialCoordinate(mesh)
        expr = cos(x*pi*2)*sin(y*pi*2)
        if len(elem.get_value_shape()) > 0:
            expr = as_vector([expr, expr, expr])
        diff_proj[n-min(scale_range)] = project_test(V, mesh, expr)
        # V = FunctionSpace(mesh, elem2.to_ufl())
        # x, y, z = SpatialCoordinate(mesh)
        # expr = cos(x*pi*2)*sin(y*pi*2)
        # if len(elem.get_value_shape()) > 0:
        #     expr = as_vector([expr, expr, expr])
        # diff_proj2[n-min(scale_range)] = project_test(V, mesh, expr)

    print("projection l2 error norms:", diff_proj)
    diff_proj = np.array(diff_proj)
    conv1 = np.log2(diff_proj[:-1] / diff_proj[1:])
    print("convergence order:", conv1)
    # print("projection l2 error norms:", diff_proj2)
    # diff_proj = np.array(diff_proj2)
    # conv2 = np.log2(diff_proj[:-1] / diff_proj[1:])
    # print("convergence order:", conv2)
    assert all([c > conv_rate for c in conv1])


@pytest.mark.parametrize("deg",
                         [n for n in range(3, 6)])
def test_polynomial_poisson_solve(deg):
    """Constructs a polynomial of order deg and the manufactured soln of poissons eqn,
    ensures it is solved exactly. """
    # Create mesh and define function space
    m = UnitTetrahedronMesh()
    x = SpatialCoordinate(m)
    elem = periodic_table(0, 3, 0, deg)
    V = FunctionSpace(m, elem.to_ufl())
    # from test_3d_examples_docs import construct_tet_cg4
    # V = FunctionSpace(m, construct_tet_cg4().to_ufl())

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v)) * dx
    u_e = x[0]
    ddu = 1/x[0]
    for i in range(deg - 1):
        u_e = u_e*x[0]
        ddu = ddu*x[0]
    bcs = [DirichletBC(V, u_e, "on_boundary")]
    f = Function(V)
    f.interpolate(-deg*(deg-1)*ddu)
    L = f*v*dx

    # Compute solution
    u_r = Function(V)
    solve(a == L, u_r, bcs=bcs, solver_parameters={'ksp_type': 'cg', 'pc_type': 'lu'})

    true = Function(V)
    true.interpolate(u_e)

    res = sqrt(assemble(inner(u_r - true, u_r - true) * dx))
    print(res)
    assert np.allclose(res, 0)


# def test_plane():
#     from fuse import make_tetrahedron
#     cell = make_tetrahedron()
#     verts = cell.ordered_vertex_coords()
#     res = check_below_plane(verts[1], verts[2], verts[3], (verts[1] + verts[2] + verts[3])/3)
#     print(res)


# def test_check_line():
#     from fuse import polygon
#     cell = polygon(3)
#     verts = np.array(sorted(cell.ordered_vertex_coords()))
#     midpoint = (verts[1] + verts[2])/2
#     midpoint1 = (verts[0] + verts[2])/2
#     assert check_below_line(verts[0], midpoint, (0, 0)) == 0
#     assert check_on_line(verts[0], midpoint, (0, 0))
#     assert check_on_line(verts[1], verts[2], midpoint)
#     assert not check_on_line(verts[1], verts[2], midpoint1)

#     assert check_below_line(verts[0], midpoint, (-0.5, 0)) == -1
#     assert check_below_line(verts[0], midpoint, (0, -0.5)) == 1

#     assert check_below_line(verts[1], midpoint1, verts[0]) == 1

# test_construction3d(1,3, 2)
