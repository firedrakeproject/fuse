from fuse import *
from firedrake import *
import numpy as np
import sympy as sp
from test_convert_to_fiat import create_cg2_tri, construct_cg3


def construct_bubble(cell=None):
    if cell is None:
        cell = polygon(3)
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    f = (3*np.sqrt(3)/4)*(y + np.sqrt(3)/3)*(np.sqrt(3)*x + y - 2*np.sqrt(3)/3)*(-np.sqrt(3)*x + y - 2*np.sqrt(3)/3)
    space = PolynomialSpace(3).restrict(0, 0)*f
    xs = [DOF(DeltaPairing(), PointKernel((0, 0)))]
    bubble = ElementTriple(cell, (space, CellL2, L2), DOFGenerator(xs, S1, S1))
    return bubble


def test_bubble():
    mesh = UnitTriangleMesh()
    x = SpatialCoordinate(mesh)

    tri = polygon(3)
    bub = construct_bubble(tri)
    cg2 = create_cg2_tri(tri)
    p2b3 = bub + cg2
    V = FunctionSpace(mesh, p2b3.to_ufl())
    W = FunctionSpace(mesh, construct_cg3().to_ufl())

    bubble_func = 27*x[0]*x[1]*(1-x[0]-x[1])
    u = project(bubble_func, V)
    exact = Function(W)
    exact.interpolate(bubble_func, W)
    # make sure that these are the same
    assert sqrt(assemble((u-exact)*(u-exact)*dx)) < 1e-14
