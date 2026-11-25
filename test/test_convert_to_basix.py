import pytest
from fuse import *
import numpy as np
from test_convert_to_fiat import create_cg1, create_cg2, create_cg2_tri
from test_2d_examples_docs import construct_cg3


@pytest.mark.skipbasix
@pytest.mark.parametrize("elem_gen,deg", [(create_cg1, 1),
                                          (create_cg2, 2)])
def test_basix_conversion_interval(elem_gen, deg):
    import basix
    cell = Point(1, [Point(0), Point(0)], vertex_num=2)
    cg = elem_gen(cell)
    element = cg.to_basix()

    points = np.array([[0.0], [1.0], [0.5], [-1.0]])
    fuse_to_basix = element.tabulate(0, points)
    lagrange = basix.create_element(
        basix.ElementFamily.P, basix.CellType.interval, deg, basix.LagrangeVariant.equispaced
    )
    basix_pts = lagrange.tabulate(0, points)
    assert np.allclose(fuse_to_basix, basix_pts)


@pytest.mark.skipbasix
@pytest.mark.parametrize("elem_gen,deg", [(create_cg1, 1),
                                          (create_cg2_tri, 2),
                                          (construct_cg3, 3)])
def test_basix_conversion(elem_gen, deg):
    """FUSE's CG elements use GLL points on the edges, and after realignment
    their DOFs are ordered as in Basix's GLL Lagrange element."""
    import basix
    cell = polygon(3)
    cg = elem_gen(cell)
    element = cg.to_basix()

    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.3]])
    fuse_to_basix = element.tabulate(0, points)

    lagrange = basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, deg, basix.LagrangeVariant.gll_warped
    )
    basix_pts = lagrange.tabulate(0, points)
    assert np.allclose(fuse_to_basix, basix_pts)


def _vector_elements():
    from test_2d_examples_docs import construct_nd, construct_rt
    from test_orientations import construct_nd2, construct_rt2
    import basix
    N1E, RT = basix.ElementFamily.N1E, basix.ElementFamily.RT
    return [(construct_nd, N1E, 1), (construct_rt, RT, 1),
            (construct_nd2, N1E, 2), (construct_rt2, RT, 2)]


@pytest.mark.skipbasix
@pytest.mark.parametrize("index", range(4))
def test_vector_conversion_spans_basix_space(index):
    """The converted H(curl)/H(div) element spans the same space on the Basix
    reference cell as Basix's own element. The bases differ by the choice
    of DOFs."""
    import basix
    elem_gen, family, deg = _vector_elements()[index]
    element = elem_gen(polygon(3)).to_basix()
    reference = basix.create_element(family, basix.CellType.triangle, deg)
    assert element.map_type == reference.map_type

    rng = np.random.default_rng(0)
    points = rng.random((40, 2))
    points = points[points.sum(axis=1) < 1]
    a = element.tabulate(0, points)[0].transpose(0, 2, 1).reshape(-1, element.dim)
    b = reference.tabulate(0, points)[0].transpose(0, 2, 1).reshape(-1, reference.dim)
    change, *_ = np.linalg.lstsq(b, a, rcond=None)
    assert np.allclose(b @ change, a)
    assert np.linalg.matrix_rank(change) == element.dim


def _shared_edge_mismatch(element, cell_a, cell_b):
    """Assemble ``element`` on two triangles sharing the global edge {1, 2},
    applying Basix's DOF transformations as DOLFINx does, and return the
    largest jump in the conforming trace of the shared edge's basis functions.

    :param: cell_a, cell_b: global vertex numbers in local order, which fix
        each cell's local numbering and orientation of the shared edge.
    """
    import basix
    from basix import MapType
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    edges = basix.topology(basix.CellType.triangle)[1]

    s = np.linspace(0.1, 0.9, 7)
    x = coords[1] + np.outer(s, coords[2] - coords[1])
    tangent = coords[2] - coords[1]
    normal = np.array([tangent[1], -tangent[0]])

    traces = []
    for glob in (cell_a, cell_b):
        v = coords[glob]
        J = np.c_[v[1] - v[0], v[2] - v[0]]
        X = np.linalg.solve(J, (x - v[0]).T).T
        tab = element.tabulate(0, X)[0]  # (point, dof, value)

        # DOLFINx reflects a local edge whose global vertex numbers decrease
        T = np.eye(element.dim)
        for e, (p, q) in enumerate(edges):
            if glob[p] > glob[q]:
                T = element.base_transformations()[e] @ T
        tab = np.einsum("ij,pjv->piv", T, tab)

        if element.map_type == MapType.covariantPiola:
            tab = np.einsum("ab,pib->pia", np.linalg.inv(J).T, tab) @ tangent
        elif element.map_type == MapType.contravariantPiola:
            tab = np.einsum("ab,pib->pia", J / np.linalg.det(J), tab) @ normal
        else:
            tab = tab[..., 0]

        # the shared edge's DOFs, numbered along the edge from global vertex 1
        local = [i for i, (p, q) in enumerate(edges) if {glob[p], glob[q]} == {1, 2}][0]
        traces.append(tab[:, element.entity_dofs[1][local]])
    return np.max(np.abs(traces[0] - traces[1]))


def _all_cell_pairs():
    from itertools import permutations
    return [(list(a), list(b)) for a in permutations([0, 1, 2]) for b in permutations([3, 2, 1])]


@pytest.mark.skipbasix
@pytest.mark.parametrize("name", ["cg2", "cg3", "nd", "rt", "nd2", "rt2"])
def test_continuity_across_reoriented_edges(name):
    """Converted elements are conforming in Basix/DOLFINx: on two cells sharing
    an edge, for every local numbering of both cells (so every local edge,
    reflected or not), the shared edge's basis functions agree."""
    from test_2d_examples_docs import construct_nd, construct_rt
    from test_orientations import construct_nd2, construct_rt2
    elem_gen = {"cg2": create_cg2_tri, "cg3": construct_cg3, "nd": construct_nd,
                "rt": construct_rt, "nd2": construct_nd2, "rt2": construct_rt2}[name]
    element = elem_gen(polygon(3)).to_basix()
    for cell_a, cell_b in _all_cell_pairs():
        assert _shared_edge_mismatch(element, cell_a, cell_b) < 1e-10, (cell_a, cell_b)


@pytest.mark.skipbasix
def test_continuity_check_detects_missing_realignment(monkeypatch):
    """Without realigning FUSE's edge 1, the continuity check fails, so the
    realignment is what makes cg3 conforming."""
    import fuse_basix.basix_interface as bi
    original = bi.basix_entity_alignment

    def unaligned(triple, cell_type):
        return [[(entity, e_F, entity.group.identity) for entity, e_F, _ in aligned]
                for aligned in original(triple, cell_type)]

    monkeypatch.setattr(bi, "basix_entity_alignment", unaligned)
    element = construct_cg3(polygon(3)).to_basix()
    assert max(_shared_edge_mismatch(element, a, b) for a, b in _all_cell_pairs()) > 1e-2


@pytest.mark.skipbasix
def test_ufc_cells_match_basix_geometry():
    import basix
    from fuse_basix.basix_interface import basix_fuse_cell, transform_points

    for name, cell_type in [("interval", basix.CellType.interval), ("triangle", basix.CellType.triangle)]:
        ufc = basix_fuse_cell(name)
        assert np.allclose(np.array(ufc.vertices(return_coords=True), dtype=float), basix.geometry(cell_type))

    tri_fuse = polygon(3)
    mapped = transform_points(tri_fuse, basix_fuse_cell("triangle"), tri_fuse.vertices(return_coords=True))
    assert np.allclose(np.array(mapped, dtype=float), basix.geometry(basix.CellType.triangle))

# def test_del():
#     M = [[], [], [], []]
#     for _ in range(4):
#         M[0].append(np.array([[[[1.0]]]]))
#     M[2].append(np.array([[[[1.0]]]]))

#     # There are no DOFs associates with the edges for this element, so we add an empty
#     # matrix for each edge.

#     for _ in range(4):
#         M[1].append(np.zeros((0, 1, 0, 1)))
#     print(M)

#     x = [[], [], [], []]
#     x[0].append(np.array([[0.0, 0.0]]))
#     x[0].append(np.array([[1.0, 0.0]]))
#     x[0].append(np.array([[0.0, 1.0]]))

#     for _ in range(3):
#         x[1].append(np.zeros((0, 2)))
#     x[2].append(np.zeros((0, 2)))

#     M = [[], [], [], []]
#     for _ in range(3):
#         M[0].append(np.array([[[1.0]]]))

#     for _ in range(3):
#         M[1].append(np.zeros((0, 1, 0)))
#     M[2].append(np.zeros((0, 1, 0, 1)))
#     print(M)
#     # cell = polygon(3)
#     # cg = create_cg1(cell)
#     # x, M = cg.to_basix()
#     # print(M)
