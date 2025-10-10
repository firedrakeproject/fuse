import unittest.mock as mock
from firedrake import *
from fuse import *
import sympy as sp
from test_convert_to_fiat import create_cg1, helmholtz_solve, construct_nd

def dummy_dof_perms(cls, *args, **kwargs):
    # return -1s of right shape here
    oriented_mats_by_entity, flat_by_entity = cls._initialise_entity_dicts(cls.generate())
    for key1, val1 in oriented_mats_by_entity.items():
        for key2, val2 in oriented_mats_by_entity[key1].items():
            for key3, val3 in oriented_mats_by_entity[key1][key2].items():
                oriented_mats_by_entity[key1][key2][key3] = 1 * np.identity(val3.shape[0])
                oriented_mats_by_entity[key1][key2][key3][0] = key1*100 + key2*10 + key3 
    return oriented_mats_by_entity, False, None

def test_orientation_application_mocked():
    deg = 1
    with mock.patch.object(ElementTriple, 'make_dof_perms', new=dummy_dof_perms):
        cell = polygon(3)
        elem = create_cg1(cell)
        mesh = UnitSquareMesh(1, 1)

        V = FunctionSpace(mesh, elem.to_ufl())
        u = TestFunction(V)
        
        f1 = Function(V)
        x, y = SpatialCoordinate(mesh) 
        f1.interpolate(x)
        #res1 = assemble(u * dx)
        #print(res1.dat.data)

        V = FunctionSpace(mesh, "CG", 1)
        u = TestFunction(V)
        f = Function(V) 
        x, y = SpatialCoordinate(mesh) 
        f.interpolate(x)
        print(f"f1, {f1.dat.data}")
        print(f"f, {f.dat.data}")
        #res2 = assemble(u * dx)
        #print(res2.dat.data)
        breakpoint()
        assert np.allclose(res1.dat.data, res2.dat.data)

def test_orientation_application():
    deg = 1
    cell = polygon(3)
    elem = construct_nd(cell)
    mesh = UnitSquareMesh(4, 4)
    ones = as_vector((1,1))

    V = FunctionSpace(mesh, elem.to_ufl())
    u = TestFunction(V)
    res1 = assemble(dot(u, ones) * dx)
    print(res1.dat.data)

    V = FunctionSpace(mesh, "N1curl", 2)
    u = TestFunction(V)
    res2 = assemble(dot(u, ones) * dx)
    print(res2.dat.data)
    assert np.allclose(res1.dat.data, res2.dat.data)



def test_interpolation_const_fire():
    print()
    mesh = UnitSquareMesh(1, 1)
    ones = as_vector((1,0))
    V = FunctionSpace(mesh, "N1curl", 1)
    u = TestFunction(V)
    res2= assemble(interpolate(ones, V))
    print(res2.dat.data)
    print([n.pt_dict for n in V.finat_element.fiat_equivalent.dual_basis()])

def test_interpolation_const_fuse():
    cell = polygon(3)
    elem = construct_nd(cell)
    print()
    #with mock.patch.object(ElementTriple, 'make_dof_perms', new=dummy_dof_perms):
    mesh = UnitSquareMesh(1, 1)
    ones = as_vector((1,0))
    V = FunctionSpace(mesh, elem.to_ufl())
    u = TestFunction(V)
    res1= assemble(interpolate(ones, V))
    print(res1.dat.data)
    print([n.pt_dict for n in V.finat_element.fiat_equivalent.dual_basis()])
    breakpoint()


def test_surface_const_fuse():
    cell = polygon(3)
    elem = construct_nd(cell)
    ones = as_vector((1,0))

    for n in range(1, 6):
        mesh = UnitSquareMesh(n, n)
        V = FunctionSpace(mesh, elem.to_ufl())
        n = FacetNormal(mesh)
        ones1 = interpolate(ones, V)
        res1= assemble(dot(ones1, n) * ds)
        
        print(f"{n}: {res1}")
        assert np.allclose(res1, 0)

def test_surface_const_fire():
    cell = polygon(3)
    elem = construct_nd(cell)
    ones = as_vector((1,0))

    for n in range(1, 6):
        mesh = UnitSquareMesh(n, n)
        
        V = FunctionSpace(mesh, "N1curl", 1)
        n = FacetNormal(mesh)
        ones2 = interpolate(ones, V)
        res1= assemble(dot(ones2, n) * ds)
        
        print(f"{n}: {res1}")
        assert np.allclose(res1, 0)


def test_surface_y():
    import os
    import subprocess
    cell = polygon(3)
    elem = construct_nd(cell)
    ones = as_vector((1,0))

    for n in range(2, 6):

        mesh = UnitSquareMesh(n, n)
        x, y = SpatialCoordinate(mesh)
        V = FunctionSpace(mesh, elem.to_ufl())
        normal = FacetNormal(mesh)
        vec1 = interpolate(as_vector((y, 0)), V)
        res1= assemble(dot(vec1, normal) * ds)

        #V = FunctionSpace(mesh, "N1curl", 1)
        #normal = FacetNormal(mesh)
        #vec2 = interpolate(as_vector((y, 0)), V)
        #res1 = assemble(dot(vec2, normal) * ds)
        print(f"{n}: {res1}")
        
        #print(f"{n}: 1: {res1} 2: {res2}")
        assert np.allclose(1/n, res2)


def test_interpolate_vs_project(V):
    mesh = V.mesh()
    dim = mesh.geometric_dimension()
    if dim == 2:
        x, y = SpatialCoordinate(mesh)
    elif dim == 3:
        x, y, z = SpatialCoordinate(mesh)

    shape = V.value_shape
    if dim == 2:
        if len(shape) == 0:
            expression = x + y
        elif len(shape) == 1:
            expression = as_vector([x, y])
        elif len(shape) == 2:
            expression = as_tensor(([x, y], [x, y]))
    elif dim == 3:
        if len(shape) == 0:
            expression = x + y + z
        elif len(shape) == 1:
            expression = as_vector([x, y, z])
        elif len(shape) == 2:
            expression = as_tensor(([x, y, z], [x, y, z], [x, y, z]))

    f = assemble(interpolate(expression, V))
    expect = project(expression, V)
    print(f.dat.data)
    print(expect.dat.data)
    assert np.allclose(f.dat.data, expect.dat.data, atol=1e-06)

def construct_nd2(tri=None):
    if tri is None:
        tri = polygon(3)
    deg = 2
    edge = tri.edges()[0]
    x = sp.Symbol("x")
    y = sp.Symbol("y")

    #xs = [DOF(L2Pairing(), PointKernel((-np.sqrt(2),)))]
    xs = [DOF(L2Pairing(), PolynomialKernel((1/2)*(x+1), symbols=[x])),
          DOF(L2Pairing(), PolynomialKernel((1/2)*(1-x), symbols=[x]))]


    dofs = DOFGenerator(xs, S1, S2)
    int_ned = ElementTriple(edge, (P1, CellHCurl, C0), dofs)

    xs = [DOF(L2Pairing(), ComponentKernel((0,))),
          DOF(L2Pairing(), ComponentKernel((1,)))]
    center_dofs = DOFGenerator(xs, S1, S3)
    xs = [immerse(tri, int_ned, TrHCurl)]
    tri_dofs = DOFGenerator(xs, C3, S1)

    vec_Pk = PolynomialSpace(deg - 1, set_shape=True)
    Pk = PolynomialSpace(deg - 1)
    M = sp.Matrix([[y, -x]])
    nd = vec_Pk + (Pk.restrict(deg-2, deg-1))*M 


    ned = ElementTriple(tri, (nd, CellHCurl, C0), [tri_dofs, center_dofs])
    return ned


def test_n1():
    cell = polygon(3)
    #elem = construct_nd2(cell)
    mesh = UnitSquareMesh(1,1)

    #print("fuse")
    #V = FunctionSpace(mesh, elem.to_ufl())
    #test_interpolate_vs_project(V)

    print("firedrake")
    V = FunctionSpace(mesh, "N1curl", 2)
    test_interpolate_vs_project(V)

def test_create_fiat_nd():
    cell = polygon(3)
    nd = construct_nd2(cell)
    ref_el = cell.to_fiat()
    sd = ref_el.get_spatial_dimension()
    deg = 2

    from FIAT.nedelec import Nedelec
    fiat_elem = Nedelec(ref_el, deg)
    print("fiat")
    for f in fiat_elem.dual_basis():
        print(f.pt_dict)
    
    print("fuse")
    for d in nd.generate():
        print(d.convert_to_fiat(ref_el, deg).pt_dict)
        print(d)
    breakpoint()
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

