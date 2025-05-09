import pytest
import numpy as np
from fuse import *
from firedrake import *
from sympy.combinatorics import Permutation
from FIAT.quadrature_schemes import create_quadrature
from test_2d_examples_docs import construct_cg1, construct_nd, construct_rt, construct_cg3
from test_3d_examples_docs import construct_tet_rt
from test_polynomial_space import flatten
from element_examples import CR_n


def create_dg1(cell):
    xs = [DOF(DeltaPairing(), PointKernel(cell.vertices(return_coords=True)[0]))]
    Pk = PolynomialSpace(1)
    dg = ElementTriple(cell, (Pk, CellL2, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1))
    return dg


def create_dg2(cell):
    xs = [DOF(DeltaPairing(), PointKernel(cell.vertices(return_coords=True)[0]))]
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
    vert_dg = create_dg1(cell.vertices()[0])
    xs = [immerse(cell, vert_dg, TrH1)]

    Pk = PolynomialSpace(deg)
    cg = ElementTriple(cell, (Pk, CellL2, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1))
    return cg


def create_cg1_quad():
    deg = 1
    # cell = polygon(4)
    cell = constructCellComplex("quadrilateral").cell_complex

    vert_dg = create_dg1(cell.vertices()[0])
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
    vert_dg = create_dg1(cell.vertices()[0])
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
    vert_dg = create_dg1(cell.vertices()[0])
    xs = [immerse(cell, vert_dg, TrH1)]
    center = [DOF(DeltaPairing(), PointKernel((0,)))]

    Pk = PolynomialSpace(deg)
    cg = ElementTriple(cell, (Pk, CellL2, C0), [DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1),
                                                DOFGenerator(center, S1, S1)])
    return cg


def create_cg2_tri(cell):
    deg = 2
    Pk = PolynomialSpace(deg)

    vert_dg0 = create_dg1(cell.vertices()[0])
    xs = [immerse(cell, vert_dg0, TrH1)]

    edge_dg0 = ElementTriple(cell.edges(get_class=True)[0], (Pk, CellL2, C0), DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))], S1, S1))
    edge_xs = [immerse(cell, edge_dg0, TrH1)]

    cg = ElementTriple(cell, (Pk, CellL2, C0), [DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1),
                                                DOFGenerator(edge_xs, C3, S1)])
    return cg


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
                                            (construct_nd, polygon(3)),
                                            (create_cr, polygon(3)),
                                            (create_cf, polygon(3)),
                                            pytest.param(create_fortin_soulie, polygon(3), marks=pytest.mark.xfail(reason='Entity perms for non symmetric elements')),
                                            (create_dg1_tet, make_tetrahedron()),
                                            (construct_tet_rt, make_tetrahedron())
                                            ])
def test_entity_perms(elem_gen, cell):
    elem = elem_gen(cell)

    print(elem.to_fiat())
    dim = cell.get_spatial_dimension()
    
    print(elem.matrices[dim][0].keys())
    for i in elem.matrices[dim][0].keys():
        print(elem.matrices[dim][0][i])


@pytest.mark.parametrize("elem_gen,elem_code,deg", [(create_cg1, "CG", 1),
                                                    (create_dg1, "DG", 1),
                                                    (create_dg2, "DG", 2),
                                                    (create_cg2, "CG", 2)
                                                    ])
def test_2d(elem_gen, elem_code, deg):
    cell = Point(1, [Point(0), Point(0)], vertex_num=2)
    elem = elem_gen(cell)

    mesh = UnitIntervalMesh(5)
    V1 = FunctionSpace(mesh, elem_code, deg)
    u = TrialFunction(V1)
    v = TestFunction(V1)
    f = Function(V1)
    x, = SpatialCoordinate(mesh)
    f.interpolate(cos(x*pi*2))
    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = inner(f, v) * dx
    u1 = Function(V1)
    solve(a == L, u1)

    V2 = FunctionSpace(mesh, elem.to_ufl())
    u = TrialFunction(V2)
    v = TestFunction(V2)
    f = Function(V2)
    x, = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*cos(x*pi*2))
    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = inner(f, v) * dx
    u2 = Function(V2)
    solve(a == L, u2)

    res = sqrt(assemble(dot(u1 - u1, u1 - u2) * dx))
    assert np.allclose(res, 0)


@pytest.mark.parametrize("elem_gen,elem_code,deg,conv_rate", [(create_cg1, "CG", 1, 1.8), (create_cg2_tri, "CG", 2, 2.8), (construct_cg3, "CG", 3, 3.8)])
def test_helmholtz(elem_gen, elem_code, deg, conv_rate):
    cell = polygon(3)
    elem = elem_gen(cell)
    scale_range = range(3, 6)

    diff = [0 for i in scale_range]
    diff2 = [0 for i in scale_range]
    for i in scale_range:
        mesh = UnitSquareMesh(2 ** i, 2 ** i)

        V = FunctionSpace(mesh, elem_code, deg)
        res1 = helmholtz_solve(mesh, V)
        diff2[i-3] = res1

        V2 = FunctionSpace(mesh, elem.to_ufl())
        res2 = helmholtz_solve(mesh, V2)
        diff[i-3] = res2
        # assert np.allclose(res1, res2)

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


def helmholtz_solve(mesh, V):
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = inner(f, v) * dx
    u = Function(V)

    # l_a = assemble(L)
    # elem = V.finat_element.fiat_equivalent
    # W = VectorFunctionSpace(mesh, V.ufl_element())
    # X = assemble(interpolate(mesh.coordinates, W))

    solve(a == L, u)
    f.interpolate(cos(x*pi*2)*cos(y*pi*2))
    return sqrt(assemble(dot(u - f, u - f) * dx))


def run_test(r, elem, parameters={}, quadrilateral=False):
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
    assert (run_test(2, elem.to_ufl(), parameters=params) < 1.e-9)


@pytest.mark.parametrize(['elem_gen'],
                         [(create_cg1_quad_tensor,), pytest.param(create_cg1_quad, marks=pytest.mark.xfail(reason='Need to allow generation on tensor product quads'))])
def test_quad(elem_gen):
    elem = elem_gen()
    r = 0
    ufl_elem = elem.to_ufl()
    assert (run_test(r, ufl_elem, parameters={}, quadrilateral=True) < 1.e-9)


def test_non_tensor_quad():
    create_cg1_quad()


@pytest.mark.parametrize("elem_gen,elem_code,deg", [(create_cg2_tri, "CG", 2),
                                                    (create_cg1, "CG", 1),
                                                    (create_dg1, "DG", 1),
                                                    (create_cr, "CR", 1),
                                                    (create_cr3, "CR", 1),
                                                    (lambda cell: CR_n(cell, 3), "CR", 1),
                                                    (create_cf, "CR", 1),  # Don't think Crouzeix Falk in in Firedrake
                                                    (construct_cg3, "CG", 3),
                                                    pytest.param(construct_nd, "N1curl", 1, marks=pytest.mark.xfail(reason='Dense Matrices needed')),
                                                    pytest.param(construct_rt, "RT", 1, marks=pytest.mark.xfail(reason='Dense Matrices needed'))])
def test_project(elem_gen, elem_code, deg):
    cell = polygon(3)
    elem = elem_gen(cell)
    mesh = UnitTriangleMesh()

    U = FunctionSpace(mesh, elem_code, deg)

    f = Function(U)
    f.assign(1)

    out = Function(U)
    u = TrialFunction(U)
    v = TestFunction(U)
    a = inner(u, v)*dx
    L = inner(f, v)*dx
    solve(a == L, out)

    assert np.allclose(out.dat.data, f.dat.data, rtol=1e-5)

    U = FunctionSpace(mesh, elem.to_ufl())

    f = Function(U)
    f.assign(1)

    out = Function(U)
    u = TrialFunction(U)
    v = TestFunction(U)
    a = inner(u, v)*dx
    L = inner(f, v)*dx
    solve(a == L, out)

    assert np.allclose(out.dat.data, f.dat.data, rtol=1e-5)


@pytest.mark.parametrize("elem_gen,elem_code,deg", [pytest.param(create_dg1_tet, "DG", 1, marks=pytest.mark.xfail(reason='Issue with fiat vs plexcone - 3D'))])
def test_project_3d(elem_gen, elem_code, deg):
    cell = make_tetrahedron()
    elem = elem_gen(cell)

    mesh = UnitCubeMesh(3, 3, 3)

    U = FunctionSpace(mesh, elem_code, deg)

    f = Function(U)
    f.assign(1)

    out = Function(U)
    u = TrialFunction(U)
    v = TestFunction(U)
    a = inner(u, v)*dx
    L = inner(f, v)*dx
    solve(a == L, out)

    assert np.allclose(out.dat.data, f.dat.data, rtol=1e-5)

    U = FunctionSpace(mesh, elem.to_ufl())

    f = Function(U)
    f.assign(1)

    out = Function(U)
    u = TrialFunction(U)
    v = TestFunction(U)
    a = inner(u, v)*dx
    L = inner(f, v)*dx
    solve(a == L, out)

    assert np.allclose(out.dat.data, f.dat.data, rtol=1e-5)


def test_investigate_dpc():
    mesh = UnitSquareMesh(2, 2, quadrilateral=True)

    U = FunctionSpace(mesh, "DPC", 1)
    print(U)
