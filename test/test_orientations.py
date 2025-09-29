import unittest.mock as mock
from firedrake import *
from fuse import *
from test_convert_to_fiat import create_cg1, helmholtz_solve

def dummy_dof_perms(cls, *args, **kwargs):
    # return -1s of right shape here
    oriented_mats_by_entity, flat_by_entity = cls._initialise_entity_dicts(cls.generate())
    for key1, val1 in oriented_mats_by_entity.items():
        for key2, val2 in oriented_mats_by_entity[key1].items():
            for key3, val3 in oriented_mats_by_entity[key1][key2].items():
                oriented_mats_by_entity[key1][key2][key3] = -1 * np.identity(val3.shape[0])
    return oriented_mats_by_entity, False, None

def test_orientation_application():
    deg = 1
    with mock.patch.object(ElementTriple, 'make_dof_perms', new=dummy_dof_perms):
        cell = polygon(3)
        elem = create_cg1(cell)
        mesh = UnitSquareMesh(4, 4)

        V = FunctionSpace(mesh, elem.to_ufl())
        res = helmholtz_solve(V, mesh)
