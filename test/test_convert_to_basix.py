import pytest
from fuse import *
import numpy as np
from element_examples import create_cg1, create_cg2, create_cg2_tri
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
    from element_examples import construct_nd2, construct_rt2
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
    from element_examples import construct_nd2, construct_rt2
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
@pytest.mark.parametrize("make_cell,cell_name", [(lambda: Point(1, [Point(0), Point(0)], vertex_num=2), "interval"),
                                                 (lambda: polygon(3), "triangle"),
                                                 (make_tetrahedron, "tetrahedron")])
def test_vertices_map_to_basix(make_cell, cell_name):
    import basix
    from fuse_basix.basix_interface import transform_to_basix_cell
    cell = make_cell()
    mapped = transform_to_basix_cell(cell, cell.vertices(return_coords=True))
    assert np.allclose(np.array(mapped, dtype=float), basix.geometry(getattr(basix.CellType, cell_name)))


def _tet_elements():
    from element_examples import create_cg1_tet, create_cg2_tet, create_cg3_tet
    from test_3d_examples_docs import (construct_tet_cg4, construct_tet_rt, construct_tet_rt2,
                                       construct_tet_ned, construct_tet_ned2)
    import basix
    P, RT, N1E = basix.ElementFamily.P, basix.ElementFamily.RT, basix.ElementFamily.N1E
    return {"cg1": (create_cg1_tet, P, 1), "cg2": (create_cg2_tet, P, 2), "cg3": (create_cg3_tet, P, 3),
            "cg4": (construct_tet_cg4, P, 4), "rt": (construct_tet_rt, RT, 1), "rt2": (construct_tet_rt2, RT, 2),
            "ned": (construct_tet_ned, N1E, 1), "ned2": (construct_tet_ned2, N1E, 2)}


def _tet_element(name):
    elem_gen = _tet_elements()[name][0]
    try:
        return elem_gen(make_tetrahedron())
    except TypeError:  # some constructors build their own cell
        return elem_gen()


@pytest.mark.skipbasix
@pytest.mark.parametrize("name", ["cg1", "cg2", "cg3", "cg4", "rt", "rt2", "ned", "ned2"])
def test_tet_conversion_spans_basix_space(name):
    import basix
    _, family, deg = _tet_elements()[name]
    element = _tet_element(name).to_basix()
    kwargs = {"lagrange_variant": basix.LagrangeVariant.gll_warped} if family == basix.ElementFamily.P else {}
    reference = basix.create_element(family, basix.CellType.tetrahedron, deg, **kwargs)
    assert element.map_type == reference.map_type

    rng = np.random.default_rng(0)
    points = rng.random((1000, 3))  # about 170 land inside the cell
    points = points[points.sum(axis=1) < 1]
    a = element.tabulate(0, points)[0].transpose(0, 2, 1).reshape(-1, element.dim)
    b = reference.tabulate(0, points)[0].transpose(0, 2, 1).reshape(-1, reference.dim)
    change, *_ = np.linalg.lstsq(b, a, rcond=None)
    assert np.allclose(b @ change, a)
    assert np.linalg.matrix_rank(change) == element.dim


def _dolfinx_cell_info(glob):
    """The cell permutation info DOLFINx gives a tetrahedron whose local
    vertices have the global numbers ``glob``.

    Face f takes bits 3f (reflection) and 3f+1, 3f+2 (rotations), and edge e
    bit 12 + e. A face is rotated until its lowest numbered vertex comes first,
    and reflected if the vertex after that is numbered higher than the one
    before it; an edge is reflected if its global numbers decrease."""
    import basix
    faces, edges = basix.topology(basix.CellType.tetrahedron)[2], basix.topology(basix.CellType.tetrahedron)[1]
    info = 0
    for f, verts in enumerate(faces):
        g = [glob[v] for v in verts]
        rots = int(np.argmin(g))
        reflected = g[(rots + 1) % 3] > g[rots - 1]
        info |= (int(reflected) << (3 * f)) | (rots << (3 * f + 1))
    for e, (p, q) in enumerate(edges):
        if glob[p] > glob[q]:
            info |= 1 << (3 * len(faces) + e)
    return info


def _shared_face_mismatch(element, cell_a, cell_b):
    """Assemble ``element`` on two tetrahedra sharing the global face {1, 2, 3},
    transforming the reference basis as DOLFINx does (by the inverse of the
    DOF transformation Basix builds from the cell info), and return the
    largest jump in the conforming trace of the basis functions of the
    shared face and its edges.

    :param: cell_a, cell_b: global vertex numbers in local order.
    """
    import basix
    from basix import MapType
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    top = basix.topology(basix.CellType.tetrahedron)
    shared = {1, 2, 3}

    bary = np.random.default_rng(1).random((6, 3))
    x = (bary / bary.sum(axis=1, keepdims=True)) @ coords[[1, 2, 3]]
    normal = np.ones(3) / np.sqrt(3)

    traces = []
    for glob in (cell_a, cell_b):
        v = coords[glob]
        J = (v[1:] - v[0]).T
        X = np.linalg.solve(J, (x - v[0]).T).T
        tab = element.tabulate(0, X)[0]  # (point, dof, value)

        data = np.eye(element.dim).reshape(-1)
        element.T_apply(data, element.dim, _dolfinx_cell_info(glob))
        T = data.reshape(element.dim, element.dim)
        tab = np.einsum("ij,pjv->piv", np.linalg.inv(T), tab)

        if element.map_type == MapType.covariantPiola:
            tab = np.cross(np.einsum("ab,pib->pia", np.linalg.inv(J).T, tab), normal)
        elif element.map_type == MapType.contravariantPiola:
            tab = np.einsum("ab,pib->pia", J / np.linalg.det(J), tab) @ normal
        else:
            tab = tab[..., 0]

        # DOFs of the shared edges and face, entities ordered by their global vertices
        dofs = []
        for dim in (1, 2):
            entities = sorted((sorted(glob[w] for w in verts), e) for e, verts in enumerate(top[dim])
                              if {glob[w] for w in verts} <= shared)
            for _, e in entities:
                dofs += list(element.entity_dofs[dim][e])
        traces.append(tab[:, dofs])
    return np.max(np.abs(traces[0] - traces[1]))


def _all_tet_pairs():
    from itertools import permutations
    return [(list(a), list(b)) for a in permutations([0, 1, 2, 3]) for b in permutations([4, 1, 2, 3])]


@pytest.mark.skipbasix
def test_tet_continuity_check_on_basix_elements():
    """The shared-face check itself: Basix's own elements pass it."""
    import basix
    from basix import ElementFamily, CellType
    for element in [basix.create_element(ElementFamily.P, CellType.tetrahedron, 4,
                                         lagrange_variant=basix.LagrangeVariant.gll_warped),
                    basix.create_element(ElementFamily.N1E, CellType.tetrahedron, 2),
                    basix.create_element(ElementFamily.RT, CellType.tetrahedron, 2)]:
        for cell_a, cell_b in _all_tet_pairs()[::5]:
            assert _shared_face_mismatch(element, cell_a, cell_b) < 1e-10, (cell_a, cell_b)


@pytest.mark.skipbasix
@pytest.mark.parametrize("name", ["cg2", "cg3", "cg4", "rt", "rt2", "ned", "ned2"])
def test_continuity_across_reoriented_faces(name):
    """Converted tetrahedral elements are conforming in Basix/DOLFINx: for every
    local numbering of two cells sharing a face, the basis functions of the
    shared face and its edges agree."""
    element = _tet_element(name).to_basix()
    for cell_a, cell_b in _all_tet_pairs():
        assert _shared_face_mismatch(element, cell_a, cell_b) < 1e-10, (cell_a, cell_b)


@pytest.mark.skipbasix
def test_face_realignment_direction(monkeypatch):
    """FUSE and Basix order some tet faces differently by a rotation, so the
    direction of the realigning symmetry matters: its inverse breaks
    continuity for elements with several DOFs on a face."""
    import fuse_basix.basix_interface as bi
    original = bi.basix_entity_alignment

    def inverted(triple, cell_type):
        return [[(entity, e_F, entity.group.get_member(o.perm ** -1)) for entity, e_F, o in aligned]
                for aligned in original(triple, cell_type)]

    monkeypatch.setattr(bi, "basix_entity_alignment", inverted)
    element = _tet_element("cg4").to_basix()
    assert max(_shared_face_mismatch(element, a, b) for a, b in _all_tet_pairs()) > 1e-2

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
