import pytest
import numpy as np
from redefining_fe import *
from redefining_fe.cells import CellComplexToFiat
from FIAT.quadrature_schemes import create_quadrature

vert = Point(0)
edge = Point(1, [Point(0), Point(0)], vertex_num=2)
tri = n_sided_polygon(3)


# @pytest.mark.parametrize("cell", [vert, edge, tri])
def create_dg1(cell):
    xs = [DOF(DeltaPairing(), PointKernel(cell.vertices(return_coords=True)[0]))]
    Pk = PolynomialSpace(cell.dim, cell.dim)
    dg = ElementTriple(cell, (Pk, CellL2, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1))
    return dg


@pytest.mark.parametrize("cell", [edge, tri])
def test_create_cg1(cell):
    deg = 1
    vert_dg = create_dg1(cell.vertices(get_class=True)[0])
    xs = [immerse(cell, vert_dg, TrH1)]

    Pk = PolynomialSpace(deg, deg)
    cg = ElementTriple(cell, (Pk, CellL2, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1))

    from FIAT.lagrange import Lagrange
    ref_el = CellComplexToFiat(cell)
    sd = ref_el.get_spatial_dimension()

    fiat_elem = Lagrange(ref_el, deg)
    my_elem = cg.to_fiat_elem()

    Q = create_quadrature(ref_el, 2*(deg+1))
    Qpts, _ = Q.get_points(), Q.get_weights()

    fiat_vals = fiat_elem.tabulate(0, Qpts)
    my_vals = my_elem.tabulate(0, Qpts)

    assert np.allclose(fiat_vals[(0,) * sd], my_vals[(0,) * sd])
