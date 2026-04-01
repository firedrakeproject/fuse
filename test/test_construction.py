import pytest
import os
import numpy as np
from test_orientations import interpolate_vs_project, get_expression
from fuse.element_construction import periodic_table
from firedrake import *


@pytest.mark.parametrize("k,deg", [(3, 0)] + [(k, deg) for deg in range(1, 7) for k in [0, 3]])
def test_construction(k, deg):
    elem = periodic_table(0, 2, k, deg)
    elem.to_fiat()


@pytest.mark.parametrize("k,deg,conv_rate", [(0, deg, deg+0.8) for deg in list(range(1, 7))] + [(3, deg, deg + 0.8) for deg in list(range(0, 3))])
def test_convergence(k, deg, conv_rate):
    assert bool(os.environ.get("FIREDRAKE_USE_FUSE", 0))
    elem = periodic_table(0, 2, k, deg)
    scale_range = range(3, 6)
    diff_proj = [0 for i in scale_range]
    diff_inte = [0 for i in scale_range]
    for n in scale_range:
        mesh = UnitSquareMesh(2**n, 2**n)

        V = FunctionSpace(mesh, elem.to_ufl())
        x, y = SpatialCoordinate(mesh)
        expr = cos(x*pi*2)*sin(y*pi*2)
        _, exact = get_expression(V)
        diff_proj[n-min(scale_range)], diff_inte[n-min(scale_range)] = interpolate_vs_project(V, expr, exact)

    print("projection l2 error norms:", diff_proj)
    diff_proj = np.array(diff_proj)
    conv1 = np.log2(diff_proj[:-1] / diff_proj[1:])
    print("convergence order:", conv1)
    # assert all([c > conv_rate for c in conv1])

    print("interpolation l2 error norms:", diff_inte)
    diff_inte = np.array(diff_inte)
    conv2 = np.log2(diff_inte[:-1] / diff_inte[1:])
    print("convergence order:", conv2)
    assert all([c > conv_rate for c in conv2])