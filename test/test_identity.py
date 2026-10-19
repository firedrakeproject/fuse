"""An element's repr is its identity in UFL and Firedrake caches: equal
elements must print the same, and different elements differently."""
import re

from fuse import (polygon, DOF, L2Pairing, DeltaPairing, PolynomialKernel, PointKernel, DOFGenerator,
                  ElementTriple, immerse, PolynomialSpace, P0, C0, L2, S1, C3, HCurl, HDiv)
from fuse.cells import ufc_triangle
from fuse.groups import tet_edges, tet_edges_ufc
from fuse.element_construction import periodic_table


def edge_moments(action, node=0):
    tri = polygon(3)
    edge = tri.edges()[0]
    triple = ElementTriple(edge, (P0, L2), DOFGenerator([DOF(L2Pairing(), PolynomialKernel(1, action=action))], S1))
    return ElementTriple(tri, (PolynomialSpace(1, shape=2), C0), DOFGenerator([immerse(tri, triple, node=node)], C3))


def test_same_element_built_twice_prints_the_same():
    assert repr(periodic_table(0, 2, 1, 2)) == repr(periodic_table(0, 2, 1, 2))


def test_repr_has_no_entity_ids():
    assert not re.search(r"\b[vefc]\d+\b", repr(periodic_table(0, 3, 1, 1)))


def test_kernel_action_is_part_of_the_identity():
    assert repr(edge_moments(HCurl)) != repr(edge_moments(HDiv))


def test_immersion_node_is_part_of_the_identity():
    assert repr(edge_moments(HCurl, node=0)) != repr(edge_moments(HCurl, node=1))


def test_cell_numbering_is_part_of_the_identity():
    regular, ufc = polygon(3), ufc_triangle()
    assert regular.describe() != ufc.describe()
    assert repr(regular.to_ufl()) != repr(ufc.to_ufl())
    # UFL still compares cells by name
    assert regular.to_ufl().cellname == ufc.to_ufl().cellname


def test_transversals_with_the_same_size_differ():
    assert tet_edges.describe() != tet_edges_ufc.describe()


def test_point_and_moment_dofs_differ():
    tri = polygon(3)
    point = ElementTriple(tri, (P0, C0), DOFGenerator([DOF(DeltaPairing(), PointKernel((0, 0)))], S1))
    moment = ElementTriple(tri, (P0, C0), DOFGenerator([DOF(L2Pairing(), PolynomialKernel(1))], S1))
    assert repr(point) != repr(moment)
