from firedrake import *
from fuse import *
from test_convert_to_fiat import create_dg1
from test_tensor_prod import mass_solve


def her_int():
    deg = 3
    cell = Point(1, [Point(0), Point(0)], vertex_num=2)
    vert_dg = create_dg1(cell.vertices()[0])
    xs = [immerse(cell, vert_dg, TrH1), immerse(cell, vert_dg, TrGrad)]

    Pk = PolynomialSpace(deg)
    # TODO fix faking out permutations
    her = ElementTriple(cell, (Pk, CellL2, C0), DOFGenerator(xs, S2, S2))
    return her


r = 3
her1 = her_int()
her2 = her_int()
bfs = tensor_product(her1, her2).flatten()
mesh = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral)
U = FunctionSpace(mesh, bfs.to_ufl())
mass_solve(U)
