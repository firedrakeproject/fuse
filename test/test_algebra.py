from fuse import *
from firedrake import *
import sympy as sp
from test_convert_to_fiat import create_cg2_tri, create_cg1, create_cr



def test_bubble():
    mesh = UnitTriangleMesh()
    x = SpatialCoordinate(mesh)
    P2 = FiniteElement("CG", "triangle", 2)
    Bubble = FiniteElement("Bubble", "triangle", 3)
    P2B3 = P2 + Bubble
    V = FunctionSpace(mesh, P2B3)
    W = FunctionSpace(mesh, "CG", 3)
    u = project(27*x[0]*x[1]*(1-x[0]-x[1]), V)
    exact = Function(W)
    exact.interpolate(27*x[0]*x[1]*(1-x[0]-x[1]))
    # make sure that these are the same
    assert sqrt(assemble((u-exact)*(u-exact)*dx)) < 1e-14


def construct_bubble(cell=None):
    if cell is None:
        cell = polygon(3)
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    space = PolynomialSpace(0)*(x*y*(-x-y+1))
    breakpoint()
    xs = [DOF(DeltaPairing(), PointKernel((0, 0)))]
    bubble = ElementTriple(cell, (space, CellL2, L2), DOFGenerator(xs, S1, S1))
    return bubble



def test_temp():
    tri = polygon(3)
    cg1 = create_cg1(tri)
    cr1 = create_cr(tri)
    cg2 = cg1 + cr1
    cg2.to_fiat()
    breakpoint()