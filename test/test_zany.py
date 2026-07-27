"""The automated zany transformation applied to FUSE-defined elements.

FUSE builds its elements on an equilateral reference triangle, while the
built-in FInAT elements use the UFC right triangle, so the transformation
matrices are not comparable entrywise. The *physical* bases are: both are
nodal for the same physical dual functionals, so they agree up to a
permutation of the dofs.
"""
import FIAT
import finat
import gem
import numpy as np
import pytest

from gem.interpreter import evaluate

from fuse import (C0, CellH2, CellL2, DOF, DOFGenerator, DeltaPairing,
                  ElementTriple, P0, P3, P5, PointKernel, S1, S2, S3, C3,
                  TrGrad, TrH1, TrHess, immerse, polygon)

from finat.fiat_elements import ScalarZanyFuseElement, is_scalar_zany
from finat.physically_mapped import PhysicalGeometry


class AffineMapping(PhysicalGeometry):
    """The affine geometry taking one reference cell onto another.

    The automated transformation only asks for the Jacobian and the cell
    size; the remaining geometry is supplied for completeness, since
    PhysicalGeometry is abstract. Unit cell sizes leave the conditioning
    rescaling inactive, so transformed dofs stay comparable with FIAT's.
    """

    def __init__(self, ref_cell, phys_cell):
        self.ref_cell = ref_cell
        self.phys_cell = phys_cell
        self.A, self.b = FIAT.reference_element.make_affine_mapping(
            ref_cell.vertices, phys_cell.vertices)

    def cell_size(self):
        return np.ones((len(self.ref_cell.vertices),))

    def jacobian_at(self, point):
        return gem.Literal(self.A)

    def detJ_at(self, point):
        return gem.Literal(np.linalg.det(self.A))

    def reference_normals(self):
        sd = self.ref_cell.get_spatial_dimension()
        top = self.ref_cell.get_topology()
        return gem.Literal(np.asarray([self.ref_cell.compute_normal(i)
                                       for i in sorted(top[sd - 1])]))

    def physical_normals(self):
        sd = self.phys_cell.get_spatial_dimension()
        top = self.phys_cell.get_topology()
        return gem.Literal(np.asarray([self.phys_cell.compute_normal(i)
                                       for i in sorted(top[sd - 1])]))

    def physical_tangents(self):
        top = self.phys_cell.get_topology()
        return gem.Literal(
            np.asarray([self.phys_cell.compute_normalized_edge_tangent(i)
                        for i in sorted(top[1])]))

    def physical_edge_lengths(self):
        top = self.phys_cell.get_topology()
        return gem.Literal(
            np.asarray([self.phys_cell.volume_of_subcomplex(1, i)
                        for i in sorted(top[1])]))

    def physical_points(self, ps, entity=None):
        return gem.Literal(np.asarray([self.A @ x + self.b for x in ps.points]))

    def physical_vertices(self):
        return gem.Literal(self.phys_cell.verts)


def _vertex_triple(cell):
    vert = cell.vertices()[0]
    xs = [DOF(DeltaPairing(), PointKernel(()))]
    return ElementTriple(vert, (P0, CellL2, C0), DOFGenerator(xs, S1, S1))


def fuse_hermite():
    tri = polygon(3)
    dg0 = _vertex_triple(tri)
    v_dofs = DOFGenerator([immerse(tri, dg0, TrH1())], S3 / S2, S1)
    v_derv_dofs = DOFGenerator([immerse(tri, dg0, TrGrad(alpha=(1, 0))),
                                immerse(tri, dg0, TrGrad(alpha=(0, 1)))],
                               S3 / S2, S1)
    i_dofs = DOFGenerator([DOF(DeltaPairing(), PointKernel((0, 0)))], S1, S1)
    return ElementTriple(tri, (P3, CellH2, C0), [v_dofs, v_derv_dofs, i_dofs])


def fuse_argyris():
    tri = polygon(3)
    edge = tri.edges()[0]
    dg0 = _vertex_triple(tri)
    v_dofs = DOFGenerator([immerse(tri, dg0, TrH1())], S3 / S2, S1)
    v_derv_dofs = DOFGenerator([immerse(tri, dg0, TrGrad(alpha=(1, 0))),
                                immerse(tri, dg0, TrGrad(alpha=(0, 1)))],
                               S3 / S2, S1)
    v_derv2_dofs = DOFGenerator([immerse(tri, dg0, TrHess(alpha=(2, 0))),
                                 immerse(tri, dg0, TrHess(alpha=(1, 1))),
                                 immerse(tri, dg0, TrHess(alpha=(0, 2)))],
                                S3 / S2, S1)
    dg0_edge = ElementTriple(edge, (P0, CellL2, C0),
                             DOFGenerator([DOF(DeltaPairing(), PointKernel((0,)))],
                                          S1, S2))
    e_dofs = DOFGenerator([immerse(tri, dg0_edge, TrGrad(directions=["normal"]))],
                          C3, S1)
    return ElementTriple(tri, (P5, CellH2, C0),
                         [v_dofs, v_derv_dofs, v_derv2_dofs, e_dofs])


@pytest.fixture
def phys_tri():
    K = FIAT.ufc_simplex(2)
    K.vertices = ((0.0, 0.1), (1.17, -0.09), (0.15, 1.84))
    return K


@pytest.fixture
def mapping_type():
    return AffineMapping


def physical_tabulation(finat_element, mapping_type, phys_cell, phys_points):
    """Tabulate the physically mapped basis at the given physical points."""
    fiat_element = finat_element._element
    ref = fiat_element.get_reference_element()
    mapping = mapping_type(ref, phys_cell)
    M = evaluate([finat_element.basis_transformation(mapping)])[0].arr
    Ainv = np.linalg.inv(mapping.A)
    ref_points = np.array([Ainv @ (x - mapping.b) for x in phys_points])
    return M @ fiat_element.tabulate(0, ref_points)[(0, 0)]


def interior_points(phys_cell):
    bary = np.array([[0.2, 0.3, 0.5], [0.5, 0.25, 0.25], [0.1, 0.6, 0.3],
                     [1 / 3, 1 / 3, 1 / 3], [0.7, 0.2, 0.1]])
    return bary @ np.array(phys_cell.vertices)


def test_is_scalar_zany():
    """Only elements carrying derivative dofs take the zany path."""
    assert is_scalar_zany(fuse_hermite().to_fiat())
    assert is_scalar_zany(fuse_argyris().to_fiat())
    assert not is_scalar_zany(FIAT.Lagrange(FIAT.ufc_simplex(2), 3))
    assert not is_scalar_zany(FIAT.RaviartThomas(FIAT.ufc_simplex(2), 1))


@pytest.mark.parametrize("triple", [fuse_hermite, fuse_argyris])
def test_identity_mapping(triple, mapping_type):
    """Mapping the reference cell to itself leaves the basis unchanged."""
    element = ScalarZanyFuseElement(triple())
    ref = element._element.get_reference_element()
    M = evaluate([element.basis_transformation(mapping_type(ref, ref))])[0].arr
    assert np.allclose(M, np.eye(*M.shape))


def test_hermite_matches_builtin(phys_tri, mapping_type):
    """The FUSE Hermite physical basis matches the built-in FInAT one."""
    points = interior_points(phys_tri)
    fuse = physical_tabulation(ScalarZanyFuseElement(fuse_hermite()),
                               mapping_type, phys_tri, points)
    builtin = physical_tabulation(finat.Hermite(FIAT.ufc_simplex(2)),
                                  mapping_type, phys_tri, points)
    assert np.allclose(fuse, builtin)


def test_argyris_matches_builtin(phys_tri, mapping_type):
    """The FUSE Argyris physical basis matches the built-in FInAT one, up to
    the orientation of the edge normal derivatives.

    FUSE and UFC order the vertices of edge 1 oppositely ((2, 0) against
    (0, 2)), so the normal induced on that edge points the other way and the
    corresponding dof carries the opposite sign. Every other basis function
    agrees exactly.
    """
    points = interior_points(phys_tri)
    fuse = physical_tabulation(ScalarZanyFuseElement(fuse_argyris()),
                               mapping_type, phys_tri, points)
    builtin = physical_tabulation(
        finat.Argyris(FIAT.ufc_simplex(2), variant="point"),
        mapping_type, phys_tri, points)

    ref = FIAT.ufc_simplex(2)
    fuse_topology = polygon(3).get_topology()[1]
    flipped = {edge for edge, verts in fuse_topology.items()
               if verts != ref.get_topology()[1][edge]}
    edge_dofs = finat.Argyris(ref, variant="point")._element.entity_dofs()[1]

    signs = np.ones(builtin.shape[0])
    for edge in flipped:
        signs[edge_dofs[edge]] = -1
    assert np.allclose(fuse, signs[:, None] * builtin)
