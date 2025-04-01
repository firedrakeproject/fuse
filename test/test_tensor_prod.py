import pytest
from fuse import *
from firedrake import *
from test_2d_examples_docs import construct_cg1, construct_dg1
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


@pytest.mark.parametrize("generator, code, deg", [(construct_cg1, "CG", 1), (construct_dg1, "DG", 1)])
def test_tensor_product_ext_mesh(generator, code, deg):
    m = UnitIntervalMesh(2)
    mesh = ExtrudedMesh(m, 2)

    # manual method of creating tensor product elements
    horiz_elt = FiniteElement(code, as_cell("interval"), deg)
    vert_elt = FiniteElement(code, as_cell("interval"), deg)
    elt = TensorProductElement(horiz_elt, vert_elt)
    U = FunctionSpace(mesh, elt)
    mass_solve(U)

    # fuseonic way of creating tensor product elements
    A = generator()
    B = generator()
    elem = tensor_product(A, B)

    U = FunctionSpace(mesh, elem.to_ufl())
    mass_solve(U)


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


def test_quad_mesh_helmholtz():
    quadrilateral = True
    vals = range(3, 6)
    res_fuse = []
    res_fire = []
    for r in vals:
        mesh = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral)

        A = construct_cg1()
        B = construct_cg1()
        elem = tensor_product(A, B).flatten()
        U = FunctionSpace(mesh, elem.to_ufl())
        res_fuse += [helmholtz_solve(mesh, U)]

        U = FunctionSpace(mesh, "CG", 1)
        res_fire += [helmholtz_solve(mesh, U)]
    print("l2 error norms:", res_fuse)
    res = np.array(res_fuse)
    conv = np.log2(res[:-1] / res[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > 1.8).all()

    print("l2 error norms:", res_fire)
    res = np.array(res_fire)
    conv = np.log2(res[:-1] / res[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > 1.8).all()


@pytest.mark.parametrize(["A", "B", "res"], [(Point(0), line(), False),
                                             (line(), line(), True),
                                             (polygon(3), line(), False),])
def test_flattening(A, B, res):
    tensor_cell = TensorProductPoint(A, B)
    if not res:
        with pytest.raises(AssertionError):
            tensor_cell.flatten()
    else:
        tensor_cell.flatten()
