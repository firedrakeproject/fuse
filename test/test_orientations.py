import unittest.mock as mock
import pytest
from firedrake import *
from fuse import *
import sympy as sp
from test_convert_to_fiat import create_cg1, helmholtz_solve, construct_nd, construct_rt, create_cg2_tri, construct_cg3, create_dg0, create_dg1
import os

#with mock.patch.object(ElementTriple, 'make_dof_perms', new=dummy_dof_perms):
def dummy_dof_perms(cls, *args, **kwargs):
    # return -1s of right shape here
    oriented_mats_by_entity, flat_by_entity = cls._initialise_entity_dicts(cls.generate())
    for key1, val1 in oriented_mats_by_entity.items():
        for key2, val2 in oriented_mats_by_entity[key1].items():
            for key3, val3 in oriented_mats_by_entity[key1][key2].items():
                if key1 == 2:
                    oriented_mats_by_entity[key1][key2][key3] = 1 * np.identity(val3.shape[0])
                    #oriented_mats_by_entity[key1][key2][key3][0] = key1*100 + key2*10 + key3 
                    if key3 == 5:
                        
                        oriented_mats_by_entity[key1][key2][key3] = 100 * np.identity(val3.shape[0])
    return oriented_mats_by_entity, False, None

def construct_nd2(tri=None):
    if tri is None:
        tri = polygon(3)
    deg = 2
    edge = tri.edges()[0]
    x = sp.Symbol("x")
    y = sp.Symbol("y")

    xs = [DOF(L2Pairing(), ParameterisationKernel())]


    dofs = DOFGenerator(xs, S2, S2)
    int_ned1 = ElementTriple(edge, (P1, CellHCurl, C0), dofs)


    xs = [DOF(L2Pairing(), ComponentKernel((0,))),
          DOF(L2Pairing(), ComponentKernel((1,)))]
    center_dofs = DOFGenerator(xs, S1, S3)
    xs  = [immerse(tri, int_ned1, TrHCurl)]
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
    xs  = [immerse(tri, int_ned2, TrHCurl, node=0)]
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
                                                    (construct_nd2_for_fiat, "N1curl", 2),
                                                    (construct_nd2, "N1curl", 2)])
def test_surface_const_nd(elem_gen, elem_code, deg):
    cell = polygon(3)
    elem = elem_gen(cell)
    ones = as_vector((0,1))

    for n in range(1, 2):
        mesh = UnitSquareMesh(n, n)
        if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
            V = FunctionSpace(mesh, elem.to_ufl())
        else:
            V = FunctionSpace(mesh, elem_code, deg)
        print(V.cell_node_list)
        normal = FacetNormal(mesh)
        ones1 = interpolate(ones, V)
        res1= assemble(dot(ones1, normal) * ds)
        
        print(f"{n}: {res1}")
        assert np.allclose(res1, 0)


def test_surface_const_rt():
    cell = polygon(3)
    elem = construct_rt(cell)
    ones = as_vector((1,0))

    for n in range(1, 6):
        mesh = UnitSquareMesh(n, n)
        V = FunctionSpace(mesh, elem.to_ufl())
        if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
            V = FunctionSpace(mesh, elem.to_ufl())
        else:
            V = FunctionSpace(mesh, "RT", 1)
        normal = FacetNormal(mesh)
        ones1 = interpolate(ones, V)
        res1= assemble(dot(ones1, normal) * ds)
        
        print(f"{n}: {res1}")
        assert np.allclose(res1, 0)


def test_surface_vec():
    cell = polygon(3)
    rt_elem = construct_rt(cell)
    nd_elem = construct_nd(cell)

    for n in range(1, 6):

        mesh = UnitSquareMesh(n, n)
        x, y = SpatialCoordinate(mesh)
        normal = FacetNormal(mesh)
        test_vec = as_vector((-y,x))
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
            #V2 = FunctionSpace(mesh, create_dg0(cell).to_ufl())
            #u = TestFunction(V2)
            res2 = assemble(dot(vec1, normal) * ds)
        else:
            V = FunctionSpace(mesh, "N1curl", 1)
            print(V.cell_node_list)
            vec1 = interpolate(test_vec, V)
            res2 = assemble(dot(vec1, normal) * ds)
        print(f"curl {n}: {res2}")

        #assert np.allclose(0, res1)
        assert np.allclose(0, res2)


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


def test_degree2_interpolation():
    cell = polygon(3)
    elem = construct_nd2_for_fiat(cell)
    mesh = UnitSquareMesh(1,1)

    if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
        V = FunctionSpace(mesh, elem.to_ufl())
    else:
        V = FunctionSpace(mesh, "N1curl", 2)
    test_interpolate_vs_project(V)

@pytest.mark.parametrize("elem_gen,elem_code,deg", [(create_cg2_tri, "CG", 2),
                                                    (create_cg1, "CG", 1),
                                                    (create_dg1, "DG", 1),
                                                    (construct_cg3, "CG", 3),
                                                    ])
def test_interpolation(elem_gen, elem_code, deg):
    cell = polygon(3)
    elem = elem_gen(cell)
    mesh = UnitSquareMesh(1,1)

    if bool(os.environ.get("FIREDRAKE_USE_FUSE", 0)):
        V = FunctionSpace(mesh, elem.to_ufl())
    else:
        V = FunctionSpace(mesh, elem_code, deg)
    test_interpolate_vs_project(V)

def test_create_fiat_nd():
    cell = polygon(3)
    nd = construct_nd2(cell)
    nd_fv = construct_nd2_for_fiat(cell)
    ref_el = cell.to_fiat()
    sd = ref_el.get_spatial_dimension()
    deg = 2

    from FIAT.nedelec import Nedelec
    fiat_elem = Nedelec(ref_el, deg)
    print("fiat")
    for f in fiat_elem.dual_basis():
        print(f.pt_dict)

    print("fiat: fuse's version") 
    for d in nd_fv.generate():
        print(d.convert_to_fiat(ref_el, deg).pt_dict)

    print("fuse")
    dofs = nd.generate()
    e = cell.edges()[0]
    g = S3.add_cell(cell).members()[2]
    print(g)
    nodes = [d.convert_to_fiat(ref_el, deg) for d in dofs]
    new_nodes = [d(g).convert_to_fiat(ref_el, deg) if d.trace_entity == e else d.convert_to_fiat(ref_el, deg) for d in dofs]
    for i in range(len(new_nodes)):
        print(f"{dofs[i]}: {nodes[i].pt_dict}")
        print(f"{dofs[i]}: {new_nodes[i].pt_dict}")
        #for g in S2.add_cell(cell).members():
        #    print(d(g))
        #    print(f"{g} {d(g).convert_to_fiat(ref_el, deg).pt_dict}")
        print()
    my_elem = nd.to_fiat()
    #my_elem = nd_fv.to_fiat()
    #breakpoint()



