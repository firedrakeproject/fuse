import pytest
import numpy as np
import sympy as sp
from fuse import *
from firedrake import *
from sympy.combinatorics import Permutation
from FIAT.quadrature_schemes import create_quadrature
from test_2d_examples_docs import construct_cg1, construct_nd, construct_rt, construct_cg3
from test_3d_examples_docs import (construct_tet_rt, construct_tet_rt2, construct_tet_ned, construct_tet_ned_2nd_kind,
                                   construct_tet_ned_2nd_kind_2, construct_tet_ned_2nd_kind_2_non_bary,
                                   construct_tet_bdm, construct_tet_bdm2, construct_tet_ned2, construct_tet_cg4)
from test_polynomial_space import flatten
from element_examples import CR_n
import os
np.set_printoptions(linewidth=120, precision=4, suppress=True)


def create_dg0(cell):
    xs = [DOF(DeltaPairing(), PointKernel(cell.vertices(return_coords=True)[0]))]
    Pk = PolynomialSpace(0)
    dg = ElementTriple(cell, (Pk, CellL2, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1))
    return dg


def create_dg1(cell):
    if cell.dim() == 1:
        x, w = np.polynomial.legendre.leggauss(2)
        xs = [DOF(DeltaPairing(), PointKernel((x[0],)))]
    else:
        xs = [DOF(DeltaPairing(), PointKernel(cell.vertices(return_coords=True)[0]))]
    Pk = PolynomialSpace(1)
    dg = ElementTriple(cell, (Pk, CellL2, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1))
    return dg


def create_dg2(cell):
    x, w = np.polynomial.legendre.leggauss(3)
    xs = [DOF(DeltaPairing(), PointKernel((x[0], )))]
    center = [DOF(DeltaPairing(), PointKernel((0,)))]

    Pk = PolynomialSpace(2)
    dg = ElementTriple(cell, (Pk, CellL2, C0), [DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1),
                                                DOFGenerator(center, S1, S1)])
    return dg


def create_dg1_uneven(cell):
    xs = [DOF(DeltaPairing(), PointKernel(-0.75,))]
    center = [DOF(DeltaPairing(), PointKernel((0.25,)))]
    Pk = PolynomialSpace(1)
    dg = ElementTriple(cell, (Pk, CellL2, C0), [DOFGenerator(xs, S1, S2),
                                                DOFGenerator(center, S1, S2)])
    return dg


def create_dg1_tet(cell):
    xs = [DOF(DeltaPairing(), PointKernel(tuple(cell.vertices(return_coords=True)[0])))]
    dg1 = ElementTriple(cell, (P1, CellL2, C0), DOFGenerator(xs, Z4, S1))

    return dg1


def create_cr(cell):
    Pk = PolynomialSpace(1)
    edge_dg0 = ElementTriple(cell.edges(get_class=True)[0], (Pk, CellL2, C0), DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1, S1))
    edge_xs = [immerse(cell, edge_dg0, TrH1)]

    return ElementTriple(cell, (Pk, CellL2, C0), [DOFGenerator(edge_xs, C3, S1)])


def create_cr3(cell):
    Pk = PolynomialSpace(3)
    edge_dg0 = ElementTriple(cell.edges(get_class=True)[0], (Pk, CellL2, C0), [DOFGenerator([DOF(DeltaPairing(), PointKernel((-np.sqrt(3/5),)))], S2, S1),
                                                                               DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1, S1)])
    edge_xs = [immerse(cell, edge_dg0, TrH1)]
    center = [DOF(DeltaPairing(), PointKernel((0, 0)))]

    return ElementTriple(cell, (Pk, CellL2, C0), [DOFGenerator(edge_xs, C3, S1), DOFGenerator(center, S1, S1)])


def create_fortin_soulie(cell):
    Pk = PolynomialSpace(2)
    edge_2 = ElementTriple(cell.edges(get_class=True)[0], (Pk, CellL2, C0), [DOFGenerator([DOF(DeltaPairing(), PointKernel((-1/3,)))], S2, S1)])
    edge_1 = ElementTriple(cell.edges(get_class=True)[0], (Pk, CellL2, C0), [DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1, S1)])
    edge_2xs = [immerse(cell, edge_2, TrH1)]
    edge_1xs = [immerse(cell, edge_1, TrH1, node=1)]

    group_2 = PermutationSetRepresentation([Permutation([2, 0, 1]), Permutation([0, 1, 2])])
    return ElementTriple(cell, (Pk, CellL2, C0), [DOFGenerator(edge_2xs, group_2, S1), DOFGenerator(edge_1xs, S1, S1)])


def create_cf(cell):
    Pk = PolynomialSpace(3)
    edge_dg0 = ElementTriple(cell.edges(get_class=True)[0], (Pk, CellL2, C0),
                             [DOFGenerator([DOF(DeltaPairing(), PointKernel((-1/2,)))], S2, S1),
                              DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1, S1)])
    edge_xs = [immerse(cell, edge_dg0, TrH1)]
    center = [DOF(DeltaPairing(), PointKernel((0, 0)))]

    return ElementTriple(cell, (Pk, CellL2, C0), [DOFGenerator(edge_xs, C3, S1), DOFGenerator(center, S1, S1)])


def create_cg1(cell):
    deg = 1
    vert_dg = create_dg0(cell.vertices()[0])
    xs = [immerse(cell, vert_dg, TrH1)]

    Pk = PolynomialSpace(deg)
    cg = ElementTriple(cell, (Pk, CellL2, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1))
    return cg


def create_cg1_quad():
    deg = 1
    cell = polygon(4)
    # cell = constructCellComplex("quadrilateral").cell_complex

    vert_dg = create_dg0(cell.vertices()[0])
    xs = [immerse(cell, vert_dg, TrH1)]

    Pk = PolynomialSpace(deg, deg + 1)
    cg = ElementTriple(cell, (Pk, CellL2, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1))

    return cg


def create_cg1_quad_tensor():
    A = construct_cg1()
    B = construct_cg1()
    elem = tensor_product(A, B).flatten()
    return elem


def create_cg1_flipped(cell):
    deg = 1
    vert_dg = create_dg0(cell.vertices()[0])
    xs = [immerse(cell, vert_dg, TrH1, node=1)]

    Pk = PolynomialSpace(deg)
    cg = ElementTriple(cell, (Pk, CellL2, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1))

    for dof in cg.generate():
        print(dof)
    return cg


def create_cg2(cell):
    deg = 2
    if cell.dim() > 1:
        raise NotImplementedError("This method is for cg2 on edges, please use create_cg2_tri for triangles")
    vert_dg = create_dg0(cell.vertices()[0])
    xs = [immerse(cell, vert_dg, TrH1)]
    center = [DOF(DeltaPairing(), PointKernel((0,)))]

    Pk = PolynomialSpace(deg)
    cg = ElementTriple(cell, (Pk, CellL2, C0), [DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1),
                                                DOFGenerator(center, S1, S1)])
    return cg


def create_cg2_tri(cell):
    deg = 2
    Pk = PolynomialSpace(deg)

    vert_dg0 = create_dg0(cell.vertices()[0])
    xs = [immerse(cell, vert_dg0, TrH1)]

    edge_dg0 = ElementTriple(cell.edges(get_class=True)[0], (PolynomialSpace(0), CellL2, C0), DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1, S1))
    edge_xs = [immerse(cell, edge_dg0, TrH1)]

    cg = ElementTriple(cell, (Pk, CellL2, C0), [DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1),
                                                DOFGenerator(edge_xs, C3, S1)])
    return cg


def create_cg1_tet(cell):

    vert = cell.vertices()[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, CellL2, "C0"),
                        DOFGenerator(xs, S1, S1))

    v_xs = [immerse(cell, dg0, TrH1)]
    cgverts = DOFGenerator(v_xs, Z4, S1)

    cg1 = ElementTriple(cell, (P1, CellH1, "C0"),
                        [cgverts])

    return cg1


def create_cg2_tet(cell):

    vert = cell.vertices()[0]
    edge = cell.edges()[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, CellL2, "C0"),
                        DOFGenerator(xs, S1, S1))

    xs = [DOF(DeltaPairing(), PointKernel((0,)))]
    dg1_int = ElementTriple(edge, (P1, CellL2, "C0"),
                            DOFGenerator(xs, S1, S1))

    v_xs = [immerse(cell, dg0, TrH1)]
    cgverts = DOFGenerator(v_xs, Z4, S1)

    e_xs = [immerse(cell, dg1_int, TrH1)]
    cgedges = DOFGenerator(e_xs, tet_edges, S1)

    cg2 = ElementTriple(cell, (P2, CellH1, "C0"),
                        [cgverts, cgedges])

    return cg2


def create_cg3_tet(cell, perm=True):

    vert = cell.vertices()[0]
    edge = cell.edges()[0]
    face = cell.d_entities(2)[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, CellL2, "C0"),
                        DOFGenerator(xs, S1, S1))

    xs = [DOF(DeltaPairing(), PointKernel((-1/np.sqrt(5),)))]
    dg1_int = ElementTriple(edge, (P1, CellL2, "C0"),
                            DOFGenerator(xs, S2, S1))

    xs = [DOF(DeltaPairing(), PointKernel((0, 0)))]
    dg0_face = ElementTriple(face, (P0, CellL2, "C0"),
                             DOFGenerator(xs, S1, S1))

    v_xs = [immerse(cell, dg0, TrH1)]
    cgverts = DOFGenerator(v_xs, Z4, S1)

    e_xs = [immerse(cell, dg1_int, TrH1)]
    cgedges = DOFGenerator(e_xs, tet_edges, S1)

    f_xs = [immerse(cell, dg0_face, TrH1)]
    cgfaces = DOFGenerator(f_xs, tet_faces, S1)

    cg3 = ElementTriple(cell, (P3, CellH1, "C0"),
                        [cgverts, cgedges, cgfaces], perm)

    return cg3


def test_create_fiat_nd():
    cell = polygon(3)
    nd = construct_nd(cell)
    ref_el = cell.to_fiat()
    sd = ref_el.get_spatial_dimension()
    deg = 1

    from FIAT.nedelec import Nedelec
    fiat_elem = Nedelec(ref_el, deg)
    my_elem = nd.to_fiat()

    Q = create_quadrature(ref_el, 2*(deg+1))
    Qpts, _ = Q.get_points(), Q.get_weights()

    fiat_vals = fiat_elem.tabulate(0, Qpts)
    my_vals = my_elem.tabulate(0, Qpts)

    fiat_vals = flatten(fiat_vals[(0,) * sd])
    my_vals = flatten(my_vals[(0,) * sd])

    (x, res, _, _) = np.linalg.lstsq(fiat_vals.T, my_vals.T)
    x1 = np.linalg.inv(x)
    assert np.allclose(np.linalg.norm(my_vals.T - fiat_vals.T @ x), 0)
    assert np.allclose(np.linalg.norm(fiat_vals.T - my_vals.T @ x1), 0)
    assert np.allclose(res, 0)


def test_create_fiat_rt():
    cell = polygon(3)
    rt = construct_rt(cell)
    ref_el = cell.to_fiat()
    sd = ref_el.get_spatial_dimension()
    deg = 1

    from FIAT.raviart_thomas import RaviartThomas
    fiat_elem = RaviartThomas(ref_el, deg)
    my_elem = rt.to_fiat()

    Q = create_quadrature(ref_el, 2*(deg+1))
    Qpts, _ = Q.get_points(), Q.get_weights()

    fiat_vals = fiat_elem.tabulate(0, Qpts)
    my_vals = my_elem.tabulate(0, Qpts)

    fiat_vals = flatten(fiat_vals[(0,) * sd])
    my_vals = flatten(my_vals[(0,) * sd])

    (x, res, _, _) = np.linalg.lstsq(fiat_vals.T, my_vals.T)
    x1 = np.linalg.inv(x)
    assert np.allclose(np.linalg.norm(my_vals.T - fiat_vals.T @ x), 0)
    assert np.allclose(np.linalg.norm(fiat_vals.T - my_vals.T @ x1), 0)
    assert np.allclose(res, 0)


@pytest.mark.parametrize("elem_gen,elem_code,deg", [(create_cg1, "CG", 1),
                                                    (create_dg1, "DG", 1),
                                                    (create_dg2, "DG", 2),
                                                    (create_cg2, "CG", 2)])
def test_create_fiat_lagrange(elem_gen, elem_code, deg):
    cell = Point(1, [Point(0), Point(0)], vertex_num=2)
    elem = elem_gen(cell)
    ref_el = cell.to_fiat()
    sd = ref_el.get_spatial_dimension()

    from FIAT.lagrange import Lagrange
    fiat_elem = Lagrange(ref_el, deg)

    my_elem = elem.to_fiat()

    Q = create_quadrature(ref_el, 2*(deg+1))
    Qpts, _ = Q.get_points(), Q.get_weights()

    fiat_vals = fiat_elem.tabulate(0, Qpts)
    my_vals = my_elem.tabulate(0, Qpts)

    fiat_vals = flatten(fiat_vals[(0,) * sd])
    my_vals = flatten(my_vals[(0,) * sd])

    (x, res, _, _) = np.linalg.lstsq(fiat_vals.T, my_vals.T)
    x1 = np.linalg.inv(x)
    assert np.allclose(np.linalg.norm(my_vals.T - fiat_vals.T @ x), 0)
    assert np.allclose(np.linalg.norm(fiat_vals.T - my_vals.T @ x1), 0)
    assert np.allclose(res, 0)


@pytest.mark.parametrize("elem_gen, cell", [(create_cg1, Point(1, [Point(0), Point(0)], vertex_num=2)),
                                            (create_dg1, Point(1, [Point(0), Point(0)], vertex_num=2)),
                                            (create_dg2, Point(1, [Point(0), Point(0)], vertex_num=2)),
                                            (create_cg2, Point(1, [Point(0), Point(0)], vertex_num=2)),
                                            (create_cg2_tri, polygon(3)),
                                            (create_cg1, polygon(3)),
                                            (create_dg1, polygon(3)),
                                            (construct_cg3, polygon(3)),
                                            (construct_rt, polygon(3)),
                                            (construct_nd, polygon(3)),
                                            (create_cr, polygon(3)),
                                            (create_cf, polygon(3)),
                                            pytest.param(create_fortin_soulie, polygon(3), marks=pytest.mark.xfail(reason='Entity perms for non symmetric elements')),
                                            (create_dg1_tet, make_tetrahedron()),
                                            (construct_tet_rt, make_tetrahedron()),
                                            (create_cg1_tet, make_tetrahedron()),
                                            (create_cg2_tet, make_tetrahedron()),
                                            ])
def test_entity_perms(elem_gen, cell):
    elem = elem_gen(cell)

    elem.to_fiat()
    dim = cell.get_spatial_dimension()
    for i in elem.matrices[dim][0].keys():
        print(elem.matrices[dim][0][i])


@pytest.mark.parametrize("elem_gen,elem_code,deg", [(create_cg1, "CG", 1),
                                                    (create_dg1, "DG", 1),
                                                    pytest.param(create_dg2, "DG", 2, marks=pytest.mark.xfail(reason='Need to update TSFC in CI')),
                                                    (create_cg2, "CG", 2)
                                                    ])
def test_1d(elem_gen, elem_code, deg):
    cell = Point(1, [Point(0), Point(0)], vertex_num=2)
    elem = elem_gen(cell)
    scale_range = range(3, 6)

    diff = [0 for i in scale_range]
    diff2 = [0 for i in scale_range]
    for i in scale_range:
        mesh = UnitIntervalMesh(2 ** i)

        V = FunctionSpace(mesh, elem_code, deg)
        res1 = helmholtz_solve(V, mesh)
        diff2[i-min(scale_range)] = res1

        V2 = FunctionSpace(mesh, elem.to_ufl())
        res2 = helmholtz_solve(V2, mesh)
        diff[i-min(scale_range)] = res2
        assert np.allclose(res1, res2)

    print("firedrake l2 error norms:", diff2)
    diff2 = np.array(diff2)

    print("fuse l2 error norms:", diff)
    diff = np.array(diff)
    assert all([np.allclose(r1, r2) for r1, r2 in zip(diff, diff2)])


@pytest.mark.parametrize("elem_gen,elem_code,deg,conv_rate", [(create_cg1, "CG", 1, 1.8), (create_cg2_tri, "CG", 2, 2.8), (construct_cg3, "CG", 3, 3.8)])
def test_helmholtz_2d(elem_gen, elem_code, deg, conv_rate):
    cell = polygon(3)
    elem = elem_gen(cell)
    scale_range = range(3, 6)

    diff = [0 for i in scale_range]
    diff2 = [0 for i in scale_range]
    for i in scale_range:
        mesh = UnitSquareMesh(2 ** i, 2 ** i)

        V = FunctionSpace(mesh, elem_code, deg)
        res1 = helmholtz_solve(V, mesh)
        diff2[i-min(scale_range)] = res1

        V2 = FunctionSpace(mesh, elem.to_ufl())
        res2 = helmholtz_solve(V2, mesh)
        diff[i-min(scale_range)] = res2
        assert np.allclose(res1, res2)

    print("firedrake l2 error norms:", diff2)
    diff2 = np.array(diff2)
    conv1 = np.log2(diff2[:-1] / diff2[1:])
    print("firedrake convergence order:", conv1)

    print("fuse l2 error norms:", diff)
    diff = np.array(diff)
    conv2 = np.log2(diff[:-1] / diff[1:])
    print("fuse convergence order:", conv2)

    assert (np.array(conv1) > conv_rate).all()
    assert (np.array(conv2) > conv_rate).all()


@pytest.mark.parametrize("elem_gen,elem_code,deg,conv_rate", [(create_cg1_tet, "CG", 1, 1.5), (create_cg2_tet, "CG", 2, 2.8), (create_cg3_tet, "CG", 3, 3.8)])
def test_helmholtz_3d(elem_gen, elem_code, deg, conv_rate):
    cell = make_tetrahedron()
    elem = elem_gen(cell)
    scale_range = range(2, 4)
    diff = [0 for i in scale_range]
    diff2 = [0 for i in scale_range]
    for i in scale_range:
        mesh = UnitCubeMesh(2 ** i, 2 ** i, 2 ** i)

        V = FunctionSpace(mesh, elem_code, deg)
        res1 = helmholtz_solve(V, mesh)
        diff2[i - min(scale_range)] = res1

        V2 = FunctionSpace(mesh, elem.to_ufl())
        res2 = helmholtz_solve(V2, mesh)
        diff[i - min(scale_range)] = res2
        assert np.allclose(res1, res2)

    print("firedrake l2 error norms:", diff2)
    diff2 = np.array(diff2)
    conv1 = np.log2(diff2[:-1] / diff2[1:])
    print("firedrake convergence order:", conv1)

    print("fuse l2 error norms:", diff)
    diff = np.array(diff)
    conv2 = np.log2(diff[:-1] / diff[1:])
    print("fuse convergence order:", conv2)

    assert (np.array(conv1) > conv_rate).all()
    assert (np.array(conv2) > conv_rate).all()


def helmholtz_solve(V, mesh):
    # Define variational problem
    dim = mesh.ufl_cell().topological_dimension
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    expect = Function(V)
    if dim == 1:
        f.interpolate((1+8*pi*pi)*cos(x[0]*pi*2))
        expect.interpolate(cos(x[0]*pi*2))
    elif dim == 2:
        # f.interpolate(x[0]*10 + x[1])
        f.interpolate((1+8*pi*pi)*cos(x[0]*pi*2)*cos(x[1]*pi*2))
        expect.interpolate(cos(x[0]*pi*2)*cos(x[1]*pi*2))
    elif dim == 3:
        r = 2.0
        f.interpolate((1+12*pi*pi/r/r)*cos(x[0]*pi*2/r)*cos(x[1]*pi*2/r)*cos(x[2]*pi*2/r))
        expect.interpolate(cos(x[0]*pi*2/r)*cos(x[1]*pi*2/r)*cos(x[2]*pi*2/r))
    else:
        raise NotImplementedError(f"Not for dim = {dim}")
    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = inner(f, v) * dx

    # elem = V.finat_element.fiat_equivalent
    # W = VectorFunctionSpace(mesh, V.ufl_element())
    # X = assemble(interpolate(mesh.coordinates, W))
    # print(X.dat.data)
    np.set_printoptions(precision=4, suppress=True)
    print()
    # print(assemble(a).M.values)
    # l_a = assemble(L)
    # print(mesh.entity_orientations)

    # breakpoint()

    # Compute solution
    sol = Function(V)
    solve(a == L, sol, solver_parameters={'ksp_type': 'cg', 'pc_type': 'lu'})

    return sqrt(assemble(inner(sol - expect, sol - expect) * dx))


def poisson_solve(r, elem, parameters={}, quadrilateral=False):
    # Create mesh and define function space
    m = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral)
    x = SpatialCoordinate(m)
    V = FunctionSpace(m, elem)

    # Define variational problem
    u = Function(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx

    bcs = [DirichletBC(V, Constant(0), 3),
           DirichletBC(V, Constant(42), 4)]

    # Compute solution
    solve(a == 0, u, solver_parameters=parameters, bcs=bcs)

    f = Function(V)
    f.interpolate(42*x[1])

    return sqrt(assemble(inner(u - f, u - f) * dx))


@pytest.mark.parametrize(['params', 'elem_gen'],
                         [(p, d)
                          for p in [{}, {'snes_type': 'ksponly', 'ksp_type': 'preonly', 'pc_type': 'lu'}]
                          for d in (create_cg1, create_cg2_tri)])
def test_poisson_analytic(params, elem_gen):
    cell = polygon(3)
    elem = elem_gen(cell)
    assert (poisson_solve(2, elem.to_ufl(), parameters=params) < 1.e-9)


@pytest.mark.parametrize(['elem_gen'],
                         [(create_cg1_quad_tensor,), pytest.param(create_cg1_quad, marks=pytest.mark.xfail(reason='Need to allow generation on tensor product quads'))])
def test_quad(elem_gen):
    elem = elem_gen()
    r = 0
    ufl_elem = elem.to_ufl()
    assert (poisson_solve(r, ufl_elem, parameters={}, quadrilateral=True) < 1.e-9)


def test_non_tensor_quad():
    create_cg1_quad()


def project(U, mesh, func):
    f = assemble(interpolate(func, U))

    out = Function(U)
    u = TrialFunction(U)
    v = TestFunction(U)
    a = inner(u, v)*dx
    L = inner(f, v)*dx
    solve(a == L, out)

    res = sqrt(assemble(dot(out - func, out - func) * dx))
    return res


@pytest.mark.parametrize("elem_gen,elem_code,deg", [(create_cg2_tri, "CG", 2),
                                                    (create_cg1, "CG", 1),
                                                    (create_dg1, "DG", 1),
                                                    (create_cr, "CR", 1),
                                                    (create_cr3, "CR", 1),
                                                    (lambda cell: CR_n(cell, 3), "CR", 1),
                                                    (create_cf, "CR", 1),  # Don't think Crouzeix Falk in in Firedrake
                                                    (construct_cg3, "CG", 3),
                                                    ])
def test_project(elem_gen, elem_code, deg):
    cell = polygon(3)
    elem = elem_gen(cell)
    mesh = UnitTriangleMesh()

    # U = FunctionSpace(mesh, elem_code, deg)
    # assert np.allclose(project(U, mesh, Constant(1)), 0, rtol=1e-5)

    U = FunctionSpace(mesh, elem.to_ufl())
    assert np.allclose(project(U, mesh, Constant(1)), 0, rtol=1e-5)


@pytest.mark.parametrize("elem_gen,elem_code,deg", [(create_dg1_tet, "DG", 1)])
def test_project_3d(elem_gen, elem_code, deg):
    cell = make_tetrahedron()
    elem = elem_gen(cell)

    mesh = UnitCubeMesh(3, 3, 3)

    U = FunctionSpace(mesh, elem_code, deg)
    assert np.allclose(project(U, mesh, Constant(1)), 0, rtol=1e-5)

    U = FunctionSpace(mesh, elem.to_ufl())
    assert np.allclose(project(U, mesh, Constant(1)), 0, rtol=1e-5)


@pytest.mark.parametrize("elem_gen,elem_code,deg,conv_rate", [(create_dg1_tet, "DG", 1, 0.8),
                                                              (create_cg1_tet, "CG", 1, 1.8),
                                                              (create_cg2_tet, "CG", 2, 2.8),
                                                              (create_cg3_tet, "CG", 3, 3.8),
                                                              (construct_tet_rt, "RT", 1, 0.8),
                                                              (construct_tet_rt2, "RT", 2, 1.8),
                                                              (construct_tet_ned, "N1curl", 1, 0.8),
                                                              (construct_tet_ned2, "N1curl", 2, 1.8),
                                                              (construct_tet_ned_2nd_kind, "N2curl", 1, 1.8),
                                                              (construct_tet_ned_2nd_kind_2, "N2curl", 2, 2.8),
                                                              (construct_tet_ned_2nd_kind_2_non_bary, "N2curl", 2, 2.8),
                                                              (construct_tet_bdm, "BDM", 1, 1.8),
                                                              (construct_tet_bdm2, "BDM", 2, 2.8)])
def test_projection_convergence_3d(elem_gen, elem_code, deg, conv_rate):
    cell = make_tetrahedron()
    elem = elem_gen(cell)
    function = lambda x: cos((3/4)*pi*x[0])
    if elem_code != "CG" and elem_code != "DG":
        expr = lambda x: as_vector([function(x), function(x), function(x)])
    else:
        expr = function
    scale_range = range(1, 4)

    diff = [0 for i in scale_range]
    diff2 = [0 for i in scale_range]
    for i in scale_range:
        mesh = UnitCubeMesh(2 ** i, 2 ** i, 2 ** i)
        x = SpatialCoordinate(mesh)
        if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
            V2 = FunctionSpace(mesh, elem.to_ufl())
            V = FunctionSpace(mesh, elem_code, deg)
            res2 = project(V2, mesh, expr(x))
            diff[i - min(scale_range)] = res2
            res1 = project(V, mesh, expr(x))
            diff2[i - min(scale_range)] = res1
        else:
            V = FunctionSpace(mesh, elem_code, deg)
            res1 = project(V, mesh, expr(x))
            diff2[i - min(scale_range)] = res1

    print("firedrake l2 error norms:", diff2)
    diff2 = np.array(diff2)
    conv1 = np.log2(diff2[:-1] / diff2[1:])
    print("firedrake convergence order:", conv1)

    if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
        print("fuse l2 error norms:", diff)
        diff = np.array(diff)
        conv2 = np.log2(diff[:-1] / diff[1:])
        print("fuse convergence order:", conv2)
        assert (np.array(conv2) > conv_rate).all()
    else:
        assert (np.array(conv1) > conv_rate).all()


@pytest.mark.parametrize("elem_gen,elem_code,deg,conv_rate", [(construct_tet_rt, "RT", 1, 0.8),
                                                              (construct_tet_ned, "N1curl", 1, 0.8),
                                                              (construct_tet_rt2, "RT", 2, 1.8),
                                                              (construct_tet_ned2, "N1curl", 2, 1.8)])
def test_const_vec(elem_gen, elem_code, deg, conv_rate):
    cell = make_tetrahedron()
    elem = elem_gen(cell)
    vec = as_vector([1, 1, 1])
    scale_range = range(0, 2)
    for i in scale_range:
        mesh = UnitCubeMesh(2 ** i, 2 ** i, 2 ** i)
        if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
            V2 = FunctionSpace(mesh, elem.to_ufl())
            res2 = assemble(interpolate(vec, V2))
            CG3 = VectorFunctionSpace(mesh, "CG", 3)
            res3 = assemble(interpolate(res2, CG3))
            for i in range(res3.dat.data.shape[0]):
                assert np.allclose(res3.dat.data[i], np.array([1, 1, 1]))
        else:
            V = FunctionSpace(mesh, elem_code, deg)
            res1 = assemble(interpolate(vec, V))
            CG3 = VectorFunctionSpace(mesh, "CG", 3)
            res3 = assemble(interpolate(res1, CG3))
            for i in range(res3.dat.data.shape[0]):
                assert np.allclose(res3.dat.data[i], np.array([1, 1, 1]))


@pytest.mark.parametrize("elem_gen,elem_code,deg", [(construct_tet_ned2, "N1curl", 2),
                                                    (construct_tet_rt2, "RT", 2),
                                                    (construct_tet_bdm2, "BDM", 2),
                                                    (construct_tet_ned_2nd_kind_2, "N2curl", 2),
                                                    (construct_tet_ned_2nd_kind_2_non_bary, "N2curl", 2)])
def test_linear_vec(elem_gen, elem_code, deg):
    cell = make_tetrahedron()
    elem = elem_gen(cell)
    i = 0
    mesh = UnitCubeMesh(2 ** i, 2 ** i, 2 ** i)
    # mesh = UnitTetrahedronMesh()
    x = SpatialCoordinate(mesh)
    candidate_vecs = [
        [1, 0, 0], [0, 0, 0],
        [x[0], 0, 0], [0, x[0], 0], [0, 0, x[0]],
        [x[1], 0, 0], [0, x[1], 0], [0, 0, x[1]],
        [x[2], 0, 0], [0, x[2], 0], [0, 0, x[2]],
        [x[0], x[1], 0], [x[1], x[0], 0], [x[1], 0, x[0]], [x[0], x[1], x[2]]
    ]
    print(mesh.entity_orientations)
    for v in candidate_vecs:
        vec = as_vector(v)
        if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
            V2 = FunctionSpace(mesh, elem.to_ufl())
            res2 = assemble(interpolate(vec, V2))
            CG3 = VectorFunctionSpace(mesh, "CG", 3)
            res3 = assemble(interpolate(res2, CG3))
            res4 = assemble(interpolate(vec, CG3))
            if not np.allclose(res3.dat.data, res4.dat.data):
                print(vec)
        else:
            V = FunctionSpace(mesh, elem_code, deg)
            res1 = assemble(interpolate(vec, V))
            CG3 = VectorFunctionSpace(mesh, "CG", 3)
            res3 = assemble(interpolate(res1, CG3))
            res4 = assemble(interpolate(vec, CG3))
            assert np.allclose(res3.dat.data, res4.dat.data)
    assert False


def test_ned_2nd_kind_edges():
    elem = construct_tet_ned_2nd_kind_2()
    from firedrake.utility_meshes import OneTetMesh
    mesh = OneTetMesh()
    V2 = FunctionSpace(mesh, elem.to_ufl())
    coords = mesh.coordinates.dat.data
    o = coords[1]
    t_0 = as_vector(coords[2] - o)
    t_1 = as_vector(coords[3] - o)
    t_2 = as_vector(coords[0] - o)
    t_3 = as_vector(coords[2] - coords[3])
    t_4 = as_vector(coords[0] - coords[3])
    t_5 = as_vector(coords[2] - coords[0])
    x = SpatialCoordinate(mesh)
    vs = [t_0, t_1, t_2, t_3, t_4, t_5]
    for vec in vs:
        res = assemble(interpolate(vec, V2)).dat.data
        assert sum([np.allclose(res[i], 0) for i in range(len(res))]) == 3


def test_ned_2nd_kind_faces():
    elem = construct_tet_ned_2nd_kind_2()
    from firedrake.utility_meshes import OneTetMesh
    mesh = OneTetMesh()
    V2 = FunctionSpace(mesh, elem.to_ufl())
    coords = mesh.coordinates.dat.data
    o = coords[1]
    t_0 = as_vector(coords[2] - o)
    t_1 = as_vector(coords[3] - o)
    t_2 = as_vector(coords[0] - o)
    t_3 = as_vector(coords[2] - coords[3])
    t_4 = as_vector(coords[0] - coords[3])
    x = SpatialCoordinate(mesh)
    vs = [cross(t_0, t_1), cross(t_1, t_2), cross(t_2, t_0), cross(t_3, t_4)]
    for vec in vs:
        res = assemble(interpolate(vec, V2)).dat.data
        assert sum([np.allclose(res[i], 0) for i in range(len(res))]) == 12

def test_face_basis():
    print()
    elem_nb, face = construct_tet_ned_2nd_kind_2()
    elem_nb, face_nb = construct_tet_ned_2nd_kind_2_non_bary()
    rt = construct_rt()
    nd = construct_nd()
    l1 = lambda x: (1/3) - x[0]/2 - x[1]/2*np.sqrt(3)
    l2 = lambda x: (1/3) - x[1]/np.sqrt(3)
    l3 = lambda x: (1/3) + x[0]/2 - x[1]/2*np.sqrt(3)
    tangents = [face.cell.basis_vectors(entity=face.cell.d_entities(1)[i]) for i in range(3)]
    dl1, dl2, dl3 = np.array([-1/2, -1/2*np.sqrt(3)]), np.array([0, -1/np.sqrt(3)]), np.array([1/2, -1/2*np.sqrt(3)])
    vecs = [lambda x: l1(x)*dl2 - l2(x)*dl1, lambda x: l3(x)*dl1 - l1(x)*dl3, lambda x: l2(x)*dl3 - l3(x)*dl2]

    nd_vecs = [lambda x: [1/3 - (np.sqrt(3)/6)*x[1], (np.sqrt(3)/6)*x[0]],
               lambda x: [-1/6 - (np.sqrt(3)/6)*x[1], (-np.sqrt(3)/6) + (np.sqrt(3)/6)*x[0]],
               lambda x: [-1/6 - (np.sqrt(3)/6)*x[1], (np.sqrt(3)/6) + (np.sqrt(3)/6)*x[0]]]

    rt_vecs = [lambda x: [(np.sqrt(3)/6)*x[0], -1/3 + (np.sqrt(3)/6)*x[1]],
               lambda x: [(-np.sqrt(3)/6) + (np.sqrt(3)/6)*x[0], 1/6 + (np.sqrt(3)/6)*x[1]],
               lambda x: [(np.sqrt(3)/6) + (np.sqrt(3)/6)*x[0], 1/6 + (np.sqrt(3)/6)*x[1]]]
    # vecs = [lambda x: bary(x)*-np.matmul(tangent, np.array([[0, -1], [1, 0]])).squeeze() for bary, tangent in zip([l1,l2,l3], tangents)]
    dofs = face.generate()
    res = np.zeros((3, 3))
    for j, v in enumerate(vecs):
        for i in range(len(dofs)):
                res[j][i] = evaluate_pt_dict(dofs[i].to_quadrature(2, (2,)), v)
    print(res)
    dofs = face_nb.generate()
    res = np.zeros((3, 3))
    for j, v in enumerate(vecs):
        for i in range(len(dofs)):
                res[j][i] = evaluate_pt_dict(dofs[i].to_quadrature(2, (2,)), v)
    print(res)

    dofs = nd.generate()
    res = np.zeros((3, 3))
    for j, v in enumerate(vecs):
        for i in range(len(nd.generate())):
                res[j][i] = evaluate_pt_dict(dofs[i].to_quadrature(2, (2,)), v)
    print(res)

    # mesh = UnitTriangleMesh()
    # V = FunctionSpace(mesh, "RT", 1)
    # dual = V.finat_element.fiat_equivalent.dual
    # res_node = np.zeros_like(res)
    # for j, v in enumerate(rt_vecs):
    #     for i in range(len(dual.nodes)):
    #         res_node[j][i] = evaluate_pt_dict(dual.nodes[i].pt_dict, v)
    # print(res_node)
    breakpoint()

def test_ned_2nd_kind_basis_funcs_gen():
    elem, _ = construct_tet_ned_2nd_kind_2()
    elem2, _ = construct_tet_ned_2nd_kind_2_non_bary()
    cell = elem.cell
    x, y, z = sp.Symbol("x"), sp.Symbol("y"), sp.Symbol("z")
    symbols = [x, y, z]
    v_0 = cell.ordered_vertex_coords()[0]
    bvs = np.array(cell.basis_vectors(norm=False))
    res = np.matmul(np.linalg.inv(bvs.T), np.array((x, y, z) - v_0))
    bary = (1 - sum(res),) + tuple(res[i] for i in range(len(res)))
    db = []
    for b in bary:
        db += [np.array((sp.diff(b, x), sp.diff(b, y), sp.diff(b, z))).astype(np.float64)]
    vecs = [sp.Matrix(l1*dl2 - l2*dl1) for l1, dl1 in zip(bary, db) for l2, dl2 in zip(bary, db) if not np.allclose(dl1, dl2) ]
    from firedrake.utility_meshes import OneTetMesh
    mesh = OneTetMesh()
    V3 = FunctionSpace(mesh, elem2.to_ufl())
    V2 = FunctionSpace(mesh, elem.to_ufl())
    V = FunctionSpace(mesh, "N2curl", 2)
    x_m = SpatialCoordinate(mesh)
    total_fuse_zeros = 0
    for v in vecs:
        print(v)
        vec = as_tensor(sp.lambdify(symbols, v)(x_m[0], x_m[1], x_m[2])[:, 0])
        res3 = assemble(interpolate(vec, V3)).dat.data
        res2 = assemble(interpolate(vec, V2)).dat.data
        res = assemble(interpolate(vec, V)).dat.data

        print("FUSE NB", res3[V3.cell_node_list[0][18:]])
        print("FUSE   ", res2[V2.cell_node_list[0][18:]])
        print("FIAT   ", res[V.cell_node_list[0][18:]])
        print("FUSE NB", sum([np.allclose(res3[i], 0) for i in list(V3.cell_node_list[0][18:])]))
        print("FUSE   ", sum([np.allclose(res2[i], 0) for i in list(V2.cell_node_list[0][18:])]))
        print("FIAT   ", sum([np.allclose(res[i], 0) for i in list(V.cell_node_list[0][18:])]))
        total_fuse_zeros += sum([np.allclose(res2[i], 0) for i in list(V2.cell_node_list[0][18:])])
    assert total_fuse_zeros == 8*len(vecs)

def test_ned_2nd_kind_basis_funcs():
    elem, _ = construct_tet_ned_2nd_kind_2()
    elem2, _ = construct_tet_ned_2nd_kind_2_non_bary()
    from firedrake.utility_meshes import OneTetMesh
    mesh = OneTetMesh()
    V3 = FunctionSpace(mesh, elem2.to_ufl())
    V2 = FunctionSpace(mesh, elem.to_ufl())
    V = FunctionSpace(mesh, "N2curl", 2)
    coords = mesh.coordinates.dat.data
    o = coords[1]
    t_0 = coords[2] - o
    t_1 = coords[3] - o
    t_2 = coords[0] - o
    t_3 = coords[2] - coords[3]
    t_4 = coords[0] - coords[3]
    x = SpatialCoordinate(mesh)
    Js = [np.column_stack([t_0, t_1])]
        #   np.column_stack([t_1, t_2]),
        #   np.column_stack([t_2, t_0]),
        #   np.column_stack([t_3, t_4])]
    vecs = [lambda x: [x[2]/4 + np.sqrt(2)/8, x[2]/4 + np.sqrt(2)/8, -(x[0]+x[1])/4],
            lambda x: [-x[1]/4 - np.sqrt(2)/8, (x[0]+x[2])/4, -x[1]/4 - np.sqrt(2)/8],
            lambda x: [(x[2]-x[1])/4, x[0]/4 - np.sqrt(2)/8, -x[0]/4 + np.sqrt(2)/8]]
    print()
    total_fuse_zeros = 0
    for v in vecs:
        vec = as_tensor(v(x))
    #     JtJ1J = np.linalg.inv(J.T @ J) @ J.T
    #     tri = UnitTriangleMesh()
    #     RT = FunctionSpace(tri, "RT", 1).finat_element.fiat_equivalent.get_nodal_basis()

    #     breakpoint()
    #     vec = RT.tabulate(x)
        
        # vec = dot(vec, as_tensor(J.T))
        # vec = dot(dot(as_tensor(JtJ1J), (x - as_vector(o))), as_tensor(J.T))
        res3 = assemble(interpolate(vec, V3)).dat.data
        res2 = assemble(interpolate(vec, V2)).dat.data
        res = assemble(interpolate(vec, V)).dat.data
        # breakpoint()
        # def vec(x):
        #     JtJ1J = np.linalg.inv(J.T @ J) @ J.T
        #     return (JtJ1J @ (x - o)) @ J.T
        # plot_vector_field(coords, vec)
        # dual = V.finat_element.fiat_equivalent.dual
        # dual2 = V2.finat_element.fiat_equivalent.dual
        # res_node2 = np.zeros_like(res2)
        # res_node = np.zeros_like(res)
        # for i in range(len(dual.nodes)):
        #     res_node[i] = evaluate_pt_dict(dual.nodes[i].pt_dict, v)
        #     res_node2[i] = evaluate_pt_dict(dual2.nodes[i].pt_dict, v)
        print("FUSE NB", res3[V3.cell_node_list[0][21:24]])
        print("FUSE", res2[V2.cell_node_list[0][21:24]])
        print("FIAT", res[V.cell_node_list[0][21:24]])
        # print("FUSE N", res_node2[21:24])
        # print("FIAT N", res_node[21:24])
        # print(res2)
        print("FUSE NB", sum([np.allclose(res3[i], 0) for i in list(V3.cell_node_list[0][21:24])]))
        print("FUSE", sum([np.allclose(res2[i], 0) for i in list(V2.cell_node_list[0][21:24])]))
        print("FIAT", sum([np.allclose(res[i], 0) for i in list(V.cell_node_list[0][21:24])]))
        total_fuse_zeros += sum([np.allclose(res2[i], 0) for i in list(V2.cell_node_list[0][21:24])])
    assert total_fuse_zeros == 3
    breakpoint()


def plot_vector_field(coords, fn=FileNotFoundError):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x, y, z = np.meshgrid(np.arange(min(coords[:, 0]), max(coords[:, 0]), 0.3),
                          np.arange(min(coords[:, 1]), max(coords[:, 1]), 0.3),
                          np.arange(min(coords[:, 2]), max(coords[:, 2]), 0.3))
    for i in range(len(coords)):
        ax.plot([coords[i][0], coords[(i + 1) % len(coords)][0]],
                [coords[i][1], coords[(i + 1) % len(coords)][1]],
                zs=[coords[i][2], coords[(i + 1) % len(coords)][2]])
        ax.plot([coords[i][0], coords[(i + 2) % len(coords)][0]],
                [coords[i][1], coords[(i + 2) % len(coords)][1]],
                zs=[coords[i][2], coords[(i + 2) % len(coords)][2]])
        ax.plot([coords[i][0], coords[(i + 3) % len(coords)][0]],
                [coords[i][1], coords[(i + 3) % len(coords)][1]],
                zs=[coords[i][2], coords[(i + 3) % len(coords)][2]])
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    uvw = np.array([fn([x[i], y[i], z[i]]) for i in range(len(x))])
    x = x.reshape((5, 5, 5))
    y = y.reshape((5, 5, 5))
    z = z.reshape((5, 5, 5))
    u = uvw[:, 0].reshape((5, 5, 5))
    v = uvw[:, 1].reshape((5, 5, 5))
    w = uvw[:, 2].reshape((5, 5, 5))
    # u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    # v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    # w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
    #     np.sin(np.pi * z))

    ax.quiver(x, y, z, u, v, w, length=0.1)
    plt.savefig("fn.png")
    # plt.show()l


def evaluate_pt_dict(pt_dict, fn):
    pts = list(pt_dict.keys())
    wts = np.array([[foo[i][0] for i in range(len(foo))] for foo in list(pt_dict.values())])
    result = np.tensordot(wts, [fn(p) for p in pts])
    return result


@pytest.mark.parametrize("elem_gen,elem_code,deg", [(construct_tet_rt, "RT", 1),
                                                    (construct_tet_rt2, "RT", 2),
                                                    (construct_tet_ned, "N1curl", 1),
                                                    (construct_tet_ned2, "N1curl", 2),
                                                    (construct_tet_bdm2, "BDM", 2),
                                                    (construct_tet_ned_2nd_kind_2, "N2curl", 2),
                                                    (construct_tet_ned_2nd_kind_2_non_bary, "N2curl", 2),
                                                    (construct_tet_cg4, "CG", 4),
                                                    ])
def test_vec_two_tet(elem_gen, elem_code, deg):
    cell = make_tetrahedron()
    elem, _ = elem_gen(cell)

    def vec(mesh):
        coords = mesh.coordinates.dat.data
        o = coords[1]
        t_0 = coords[2] - coords[1]
        t_1 = coords[3] - coords[1]
        t_2 = coords[0] - coords[1]
        x = SpatialCoordinate(mesh)
        if elem_code == "CG":
            return x[1]
        # return as_vector(2*t_0 + t_1)
        # return as_vector((1/2)*(t_0 + t_1))
        J = np.column_stack([t_0, t_1])
        JtJ1J = np.linalg.inv(J.T @ J) @ J.T
        res = dot(dot(as_tensor(JtJ1J), (x - as_vector(o))), as_tensor(J.T))
        # def f(x):
        #     return (JtJ1J @ (x - o))
        # @ J.T
        # breakpoint()
        # return res
        # return as_vector(np.linalg.norm(x - o - t_0)*np.linalg.norm(x - o - t_1)*(1/2)*(t_0 + t_1))
        # return as_vector(np.cross(t_2, t_1))
        return as_vector([x[0], 0, 0])
    

    from firedrake.utility_meshes import TwoTetMesh, OneTetMesh
    group = [sp.combinatorics.Permutation([0, 1, 2, 3]),
             sp.combinatorics.Permutation([0, 2, 3, 1]),
             sp.combinatorics.Permutation([0, 3, 1, 2]),
             sp.combinatorics.Permutation([0, 1, 3, 2]),
             sp.combinatorics.Permutation([0, 3, 2, 1]),
             sp.combinatorics.Permutation([0, 2, 1, 3])
             ]
    error_gs = []
    error_row_lists = []
    for g in group:
        # mesh = OneTetMesh()
        mesh = TwoTetMesh(perm=g)
        print(mesh.entity_orientations)
        if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
            V2 = FunctionSpace(mesh, elem.to_ufl())
            # print(elem.matrices[2][0][mesh.entity_orientations[1][10]][np.ix_([18, 19, 20], [18, 19, 20])])
            res2 = assemble(interpolate(vec(mesh), V2))
            # print(res2.dat.data[9:12])
            # print(res2.dat.data[0:9])
            # print(res2.dat.data[30:39])
            if elem_code == "CG":
                CG3 = FunctionSpace(mesh, create_cg3_tet(cell).to_ufl())
            else:
                CG3 = VectorFunctionSpace(mesh, create_cg3_tet(cell).to_ufl())
            res3 = assemble(interpolate(res2, CG3))
            res4 = assemble(interpolate(vec(mesh), CG3))
            print(res2.dat.data)
            # breakpoint()
            error_rows = []
            for i in range(res3.dat.data.shape[0]):
                if not np.allclose(res3.dat.data[i], res4.dat.data[i]):
                    print("error")
                    error_gs += [g]
                    error_rows += [i]
            error_row_lists += [error_rows]
        else:
            V = FunctionSpace(mesh, elem_code, deg)
            res1 = assemble(interpolate(vec(mesh), V))
            CG3 = VectorFunctionSpace(mesh, "CG", 3)
            res3 = assemble(interpolate(res1, CG3))
            # for i in range(res3.dat.data.shape[0]):
            #     assert np.allclose(res3.dat.data[i], np.array([1, 1, 1]))
    assert len(error_gs) == 0


@pytest.mark.parametrize("elem_gen,elem_code,deg,max_err", [(create_cg3_tet, "CG", 3, 1e-13),
                                                            (construct_tet_cg4, "CG", 4, 1e-13),
                                                            (construct_tet_rt2, "RT", 2, 1e-13),
                                                            (construct_tet_bdm2, "BDM", 2, 1e-13),
                                                            (construct_tet_ned_2nd_kind_2, "N2curl", 2, 1e-12),
                                                            (construct_tet_ned2, "N1curl", 2, 1e-13)])
def test_const_two_tet(elem_gen, elem_code, deg, max_err):
    cell = make_tetrahedron()
    # elem_perms = elem_gen(cell, perm=True)
    elem_mats = elem_gen(cell)
    # ufl_elem_perms = elem_perms.to_ufl()
    ufl_elem_mats = elem_mats.to_ufl()

    def expr(mesh):
        x = SpatialCoordinate(mesh)
        if elem_code != "CG":
            return as_vector([x[0], 2*x[1], 3*x[2]])
        return x[0]
    errors = []
    from firedrake.utility_meshes import TwoTetMesh
    group = [sp.combinatorics.Permutation([0, 1, 2, 3]),
             sp.combinatorics.Permutation([0, 2, 3, 1]),
             sp.combinatorics.Permutation([0, 3, 1, 2]),
             sp.combinatorics.Permutation([0, 1, 3, 2]),
             sp.combinatorics.Permutation([0, 3, 2, 1]),
             sp.combinatorics.Permutation([0, 2, 1, 3])]

    for g in group:
        mesh = TwoTetMesh(perm=g)
        print(g)
        print(mesh.entity_orientations)
        if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
            # if elem_code != "RT" and elem_code != "N1curl":
            #     V = FunctionSpace(mesh, ufl_elem_perms)

            #     res = project(V, mesh, expr(mesh))
            #     print("perms", res)
            #     # assert res < max_err
            #     errors += [res]

            V2 = FunctionSpace(mesh, ufl_elem_mats)
            res = project(V2, mesh, expr(mesh))
            print(res)
            errors += [res]
            # assert res < max_err
        else:
            V = FunctionSpace(mesh, elem_code, deg)
            res = project(V, mesh, expr(mesh))
            print(res)
            # assert res < max_err
    assert all([res < max_err for res in errors])


@pytest.mark.parametrize("elem_gen,elem_code,deg",
                         [(construct_tet_cg4, "CG", 4), (construct_tet_rt2, "RT", 2), (construct_tet_ned2, "N1curl", 2), (construct_tet_bdm2, "BDM", 2),
                          ])
def test_3d_two_form(elem_gen, elem_code, deg):

    cell = make_tetrahedron()
    mesh = UnitTetrahedronMesh()
    x = SpatialCoordinate(mesh)

    spaces = []
    if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
        elem = elem_gen(cell)
        elem2 = elem_gen(cell)
        spaces += [("fuse", FunctionSpace(mesh, elem.to_ufl()), FunctionSpace(mesh, elem2.to_ufl()))]
    else:
        spaces += [("fiat", FunctionSpace(mesh, elem_code, deg), FunctionSpace(mesh, elem_code, deg))]

    for name, V, V2 in spaces:
        v = TestFunction(V)
        u = TrialFunction(V2)
        if elem_code == "CG":
            exp = cos((3/4)*pi*x[0])
        else:
            exp = as_vector([cos((3/4)*pi*x[0]), cos((3/4)*pi*x[0]), cos((3/4)*pi*x[0])])
            # exp = as_vector([x[0], x[1], x[2]])
        f = assemble(interpolate(exp, V2))

        # print("A")
        a = assemble(inner(u, v) * dx)
        # print(a.M.values[np.ix_(V.cell_node_list[0][12:16], V.cell_node_list[0][12:16])])
        # np.matmul(L.M.values[np.ix_([0, 1],[0, 1])], transform.T)
        print("L")
        L = assemble(inner(f, v) * dx)
        print(L.dat.data)
        # L.dat.data_rw[0:2] = np.matmul(L.dat.data[0:2], transform.T)

        solution = Function(V2)
        solve(a == L, solution)

        assert norm(assemble(f - solution)) < 1e-14


# TODO this is not a real test
def test_scaling_mesh():
    mesh1 = RectangleMesh(2, 1, 1, 1)
    mesh2 = RectangleMesh(2, 1, 0.5, 1)
    vec = as_vector([1, 1])
    if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):

        elem = construct_rt(polygon(3))
        V1 = FunctionSpace(mesh1, elem.to_ufl())
        V2 = FunctionSpace(mesh2, elem.to_ufl())
    else:
        V1 = FunctionSpace(mesh1, "RT", 1)
        V2 = FunctionSpace(mesh2, "RT", 1)

    res1 = assemble(interpolate(vec, V1))
    print(res1.dat.data)
    res2 = assemble(interpolate(vec, V2))
    print(res2.dat.data)


def test_quartic_poisson_solve():
    # Create mesh and define function space
    r = 0
    m = UnitCubeMesh(2 ** r, 2 ** r, 2 ** r)
    x = SpatialCoordinate(m)
    elem = construct_tet_cg4(make_tetrahedron())
    V = FunctionSpace(m, elem.to_ufl())

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v)) * dx
    u_e = x[0]*x[0]*x[0]*x[0] + 2*x[0]*x[1]*x[1] + 3*x[2]*x[2]*x[2]*x[2] + 6

    bcs = [DirichletBC(V, u_e, "on_boundary")]
    f = Function(V)
    f.interpolate(-12*x[0]*x[0] - 4*x[0] - 36*x[2]*x[2])
    L = f*v*dx

    # Compute solution
    u_r = Function(V)
    solve(a == L, u_r, bcs=bcs, solver_parameters={'ksp_type': 'cg', 'pc_type': 'lu'})

    true = Function(V)
    true.interpolate(u_e)

    res = sqrt(assemble(inner(u_r - true, u_r - true) * dx))
    print(res)
    assert np.allclose(res, 0)
