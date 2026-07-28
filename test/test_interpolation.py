from firedrake import *
from firedrake.ufl_expr import extract_unique_domain
from fuse import *
import numpy as np
import pytest
from test_2d_examples_docs import construct_cg3
from test_convert_to_fiat import create_cg2, create_cg2_tri

@pytest.mark.xfail(reason="Needs updated FIAT tensor branch")
def test_cross_mesh_tri_to_quad():
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


def test_cross_mesh_fuse_to_ufc():
    mesh1 = UnitSquareMesh(10, 10, use_fuse=True)
    mesh2 = UnitSquareMesh(10, 10)
    V1 = FunctionSpace(mesh1, create_cg2_tri().to_ufl())
    V2 = FunctionSpace(mesh2, "CG", 2)

    f1 = Function(V1)
    f2 = Function(V2)

    x = SpatialCoordinate(mesh1)
    f1 = f1.interpolate(x[0]**2 + x[1]**2)
    f2 = f2.interpolate(f1)
    assert np.allclose(sqrt(assemble(inner(f1, f1) * dx)), sqrt(assemble(inner(f2, f2) * dx)))


def test_cross_mesh():
    dest_quad = False
    atol = 1e-8
    m_src = UnitSquareMesh(2, 3, use_fuse=True)
    m_dest = UnitSquareMesh(3, 5, quadrilateral=dest_quad)
    coords = np.array(
        [[0.56, 0.6], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9], [0.726, 0.6584]]
    )  # fairly arbitrary
    # add the coordinates of the mesh vertices to test boundaries
    vertices_src = m_src.coordinates.dat.data_ro
    coords = np.concatenate((coords, vertices_src))
    vertices_dest = m_dest.coordinates.dat.data_ro
    coords = np.concatenate((coords, vertices_dest))
    expr_src = product(SpatialCoordinate(m_src))
    expr_dest = product(SpatialCoordinate(m_dest))
    dest_eval = PointEvaluator(m_dest, coords)
    expected = np.prod(coords, axis=-1)

    V_src = FunctionSpace(m_src, construct_cg3().to_ufl())
    V_dest = FunctionSpace(m_dest, "CG", 4)

    # test_expression from test_interpolate_cross_mesh in firedrake
    f_dest = assemble(interpolate(expr_src, V_dest))
    assert extract_unique_domain(f_dest) is m_dest
    got = dest_eval.evaluate(f_dest)
    assert np.allclose(got, expected, atol=atol)
    f_dest_2 = Function(V_dest).interpolate(expr_dest)
    assert np.allclose(f_dest.dat.data_ro, f_dest_2.dat.data_ro, atol=atol)

    # test_function from test_interpolate_cross_mesh in firedrake
    f_dest = Function(V_dest).interpolate(expr_src)
    assert extract_unique_domain(f_dest) is m_dest

    got = dest_eval.evaluate(f_dest)
    assert np.allclose(got, expected, atol=atol)

    f_src = Function(V_src).interpolate(expr_src)
    f_dest = assemble(interpolate(f_src, V_dest))
    assert extract_unique_domain(f_dest) is m_dest
    got = dest_eval.evaluate(f_dest)
    assert np.allclose(got, expected, atol=atol)

    f_dest_2 = Function(V_dest).interpolate(expr_dest)
    assert np.allclose(f_dest.dat.data_ro, f_dest_2.dat.data_ro, atol=atol)

    # test Function.interpolate(...)
    f_dest = Function(V_dest)
    f_dest.interpolate(f_src)
    assert extract_unique_domain(f_dest) is m_dest
    got = dest_eval.evaluate(f_dest)
    assert np.allclose(got, expected, atol=atol)
    assert np.allclose(f_dest.dat.data_ro, f_dest_2.dat.data_ro, atol=atol)
