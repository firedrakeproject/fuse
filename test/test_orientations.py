import unittest.mock as mock
from firedrake import *
from fuse import *
from test_convert_to_fiat import create_cg1, helmholtz_solve, construct_nd

def dummy_dof_perms(cls, *args, **kwargs):
    # return -1s of right shape here
    oriented_mats_by_entity, flat_by_entity = cls._initialise_entity_dicts(cls.generate())
    for key1, val1 in oriented_mats_by_entity.items():
        for key2, val2 in oriented_mats_by_entity[key1].items():
            for key3, val3 in oriented_mats_by_entity[key1][key2].items():
                oriented_mats_by_entity[key1][key2][key3] = -1 * np.identity(val3.shape[0])
    return oriented_mats_by_entity, False, None

def test_orientation_application_mocked():
    deg = 1
    with mock.patch.object(ElementTriple, 'make_dof_perms', new=dummy_dof_perms):
        cell = polygon(3)
        elem = create_cg1(cell)
        mesh = UnitSquareMesh(1, 1)

        V = FunctionSpace(mesh, elem.to_ufl())
        u = TestFunction(V)
        res1 = assemble(u * dx)
        print(res1.dat.data)

        V = FunctionSpace(mesh, "CG", 1)
        u = TestFunction(V)
        res2 = assemble(u * dx)
        print(res2.dat.data)
        assert np.allclose(res1.dat.data, res2.dat.data)

def test_orientation_application():
    deg = 1
    with mock.patch.object(ElementTriple, 'make_dof_perms', new=dummy_dof_perms):
        cell = polygon(3)
        elem = construct_nd(cell)
        mesh = UnitSquareMesh(1, 1)
        ones = as_vector((1,1))

        V = FunctionSpace(mesh, elem.to_ufl())
        u = TestFunction(V)
        x = project(ones, V)
        #res1 = assemble(dot(u, x) * dx)
        #print(res1.dat.data)

        V = FunctionSpace(mesh, "N1curl", 1)
        u = TestFunction(V)
        x = project(ones, V)
        res2 = assemble(dot(u, x) * dx)
        print(res2.dat.data)
        #assert np.allclose(res1.dat.data, res2.dat.data)
