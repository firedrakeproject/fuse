
from fuse.cells import Point, Edge, polygon, line, make_tetrahedron, constructCellComplex, TensorProductPoint
from fuse.groups import S1, S2, S3, D4, Z3, Z4, C3, C4, S4, A4, tri_C3, tet_edges, tet_faces, sq_edges, GroupRepresentation, PermutationSetRepresentation, get_cyc_group, get_sym_group
from fuse.dof import DeltaPairing, DOF, L2Pairing, MyTestFunction, PointKernel, PolynomialKernel
from fuse.triples import ElementTriple, DOFGenerator, immerse
from fuse.traces import TrH1, TrGrad, TrHess, TrHCurl, TrHDiv
from fuse.tensor_products import tensor_product

from fuse.spaces.element_sobolev_spaces import CellH1, CellL2, CellHDiv, CellHCurl, CellH2
from fuse.spaces.polynomial_spaces import P0, P1, P2, P3, Q2, PolynomialSpace
from fuse.spaces.interpolation_spaces import C0, L2, H1, HDiv
