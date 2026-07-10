from firedrake import *
from fuse import *
import numpy as np
from test_2d_examples_docs import construct_cg1, construct_cg3
from test_convert_to_fiat import create_cg2, create_cg2_tri

def test_cross_mesh():
    mesh1 = UnitSquareMesh(10, 10, use_fuse=True)
    mesh2 = UnitSquareMesh(10, 10, quadrilateral=True, use_fuse=True)
    A = create_cg2()
    B = create_cg2()
    V1 = FunctionSpace(mesh1, create_cg2_tri().to_ufl())
    V2 = FunctionSpace(mesh2, tensor_product(A, B).flatten().to_ufl())

    f1 = Function(V1)
    f2 = Function(V2)

    x = SpatialCoordinate(mesh1)
    f1 = f1.interpolate(x[0]**2 + x[1]**2)
    f2 = f2.interpolate(f1)
    assert np.allclose(sqrt(assemble(inner(f1, f1) * dx)), sqrt(assemble(inner(f2, f2) * dx)))