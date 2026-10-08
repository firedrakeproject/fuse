from fuse import *
from firedrake import *
from finat.ufl import CellBackend
from fuse.cells import ufc_triangle, ufc_tetrahedron
import pytest
import numpy as np
import sympy as sp
from FIAT.reference_element import default_simplex, ufc_simplex, Simplex
from test_convert_to_fiat import helmholtz_solve


@pytest.fixture(scope='module', params=[0, 1, 2, 3])
def C(request):
    dim = request.param
    if dim == 0:
        return Point(0)
    elif dim == 1:
        return Point(1, [Point(0), Point(0)], vertex_num=2)
    elif dim == 2:
        return polygon(3)
    elif dim == 3:
        return make_tetrahedron()


def test_vertices(C):
    verts = C.vertices()
    assert len(verts) == C.dimension + 1


def test_basis_vectors(C):
    if C.dimension == 0:
        with pytest.raises(ValueError):
            bv_ids = C.basis_vectors()
        with pytest.raises(ValueError):
            bv_coords = C.basis_vectors(return_coords=True)
    else:
        bv_ids = C.basis_vectors()
        bv_coords = C.basis_vectors(return_coords=True)
        assert len(bv_ids) == len(bv_coords)


# def test_basis_group(C):
#     if C.dimension == 0:
#         assert C.basis_group.size() == 1
#     else:
#         bv_coords = C.basis_vectors(return_coords=True)
#         bv_0 = bv_coords[0]
#         for i, g in enumerate(C.basis_group.members()):
#             assert np.allclose(np.array(bv_coords[i]), np.array(g(bv_0)))
#         if C.dimension == 2:
#             for i, g in enumerate(C.basis_group.members()):
#                 bvs = np.array(C.basis_vectors())
#                 new_bvs = np.array(C.orient(g).basis_vectors())
#                 basis_change = np.matmul(np.linalg.inv(new_bvs), bvs)
#                 assert np.allclose(np.array(bv_coords[i]), np.array(np.matmul(basis_change, bv_0)))


def test_sub_basis_vectors():
    cell = polygon(3)

    edges = cell.edges(get_class=True)
    print(cell.vertices())
    print(cell.vertices(return_coords=True))
    for e in edges:
        print(e)
        print(e.vertices())
        print(cell.basis_vectors(entity=e))


def test_permute_entities():
    # cell = polygon(3)
    cell = make_tetrahedron()
    # cell.plot(filename="test_cell.png")

    # for dof in nd.generate():
    # print(dof, "->", dof(reflect), "eval p2 ", dof(reflect).eval(phi_2), "eval p0 ", dof(reflect).eval(phi_0), "eval p1 ", dof(reflect).eval(phi_1))
    # print(dof.convert_to_fiat(cell.to_fiat(), 1)(lambda x: np.array([1/3 - (np.sqrt(3)/6)*x[1], (np.sqrt(3)/6)*x[0]])))

    print(cell.vertices(return_coords=True))
    print([c.point.connections for c in cell.connections])
    # print([[c.point.get_node(c2.point.id, return_coords=True) for c2 in c.point.connections] for c in cell.connections])
    # cell.plot(filename="test_cell_flipped.png")
    # import matplotlib.pyplot as plt
    for i, g in enumerate(cell.group.members()):
        print(i, g)
        print(cell.permute_entities(g, 0))
        print(cell.permute_entities(g, 1))
        print(cell.permute_entities(g, 2))
    #     oriented = cell.orient(g)
    #     print("Edges", oriented.connections)
    #     fig, ax = plt.subplots()
    #     oriented.plot(ax = ax, filename=f"test_cell{i}.png")
    #     oriented.hasse_diagram(filename=f"test_hasse{i}.png")


def test_oriented_verts():
    edge = Point(1, [Point(0), Point(0)], vertex_num=2)

    for g in edge.group.members():
        oriented = edge.orient(g)
        assert g.permute(edge.ordered_vertices()) == oriented.ordered_vertices()

    tri = polygon(3)
    cyclic_tri = C3.add_cell(tri)

    for g in tri.group.members():
        oriented = tri.orient(g)
        if g in cyclic_tri.members():
            # test that rotating does not change the order of edges
            permuted = oriented.permute_entities(g, 1)
            assert all([o.perm.is_Identity for (e, o) in permuted])
        assert g.permute(tri.ordered_vertices()) == oriented.ordered_vertices()

    sq = polygon(4)
    for g in sq.group.members():
        oriented = sq.orient(g)
        print(oriented.permute_entities(g, 1))
        assert g.permute(sq.ordered_vertices()) == oriented.ordered_vertices()

    tetra = make_tetrahedron()
    for g in tetra.group.members():
        oriented = tetra.orient(g)
        permuted = oriented.permute_entities(g, 0)
        print(g, permuted)
        assert g.permute(tetra.ordered_vertices()) == oriented.ordered_vertices()


def test_compare_cell_to_firedrake():
    tri1 = polygon(3)
    tri2 = default_simplex(2)

    n = 3
    vertices = []
    for i in range(n):
        vertices.append(Point(0))
    edges = []
    for i in range(n):
        edges.append(
            Point(1, [vertices[(i) % n], vertices[(i+1) % n]], vertex_num=2))

    cellS3 = S3.add_cell(tri1)
    for g in cellS3.members():
        print(g.perm.array_form)
        try:
            p = g.perm.array_form

            tri3 = Point(2, [edges[p[0]], edges[p[1]], edges[p[2]]], vertex_num=n)
            print(tri1.orient(g).get_topology())
        except AssertionError:
            print('FAIL')

    # print(tri1.get_topology())
    print(tri2.get_topology())
    tri3 = ufc_triangle()
    print(tri3.get_topology())


@pytest.fixture
def mock_cell_complex(mocker, expect):
    mocker.patch('firedrake.mesh.as_cell', return_value=expect.to_ufl("triangle"))


@pytest.mark.skipif("not config.getoption('--run-cleared')", reason="Only run when --run-cleared is given")
@pytest.mark.usefixtures("mock_cell_complex")
@pytest.mark.parametrize(["expect"], [(ufc_triangle(),), (polygon(3),)])
def test_ref_els(expect):
    scale_range = range(3, 6)
    print(expect)
    diff2 = [0 for i in scale_range]
    for i in scale_range:
        mesh = UnitSquareMesh(2 ** i, 2 ** i, cell_backend=CellBackend.FUSE)

        V = FunctionSpace(mesh, "CG", 3)
        res1 = helmholtz_solve(mesh, V)
        diff2[i-3] = res1

    print("firedrake l2 error norms:", diff2)
    diff2 = np.array(diff2)
    conv1 = np.log2(diff2[:-1] / diff2[1:])
    print("firedrake convergence order:", conv1)
    assert (np.array(conv1) > 3.8).all()


@pytest.mark.xfail(reason="need quadrilateral fiat")
def test_comparison():
    from finat.element_factory import as_fiat_cell
    from FIAT.reference_element import TensorProductCell
    name = "quadrilateral"
    tensor_product = as_fiat_cell(constructCellComplex(name))
    tensor_product1 = as_fiat_cell(constructCellComplex("interval*interval"))

    print(isinstance(tensor_product, TensorProductCell))
    print(type(tensor_product).__bases__)
    print(isinstance(tensor_product1, TensorProductCell))
    print(type(tensor_product1).__bases__)

    # print(tensor_product >= tensor_product)
    print(tensor_product >= tensor_product1)
    # print(tensor_product1 >= tensor_product)
    # print(tensor_product1 >= tensor_product1)


def test_self_equality(C):
    assert C == C


@pytest.mark.parametrize(["A", "B", "res"], [(ufc_triangle(), polygon(3), False),
                                             (line(), line(), True),])
def test_equivalence(A, B, res):
    assert A.equivalent(B) == res


@pytest.mark.parametrize(["cell"], [(ufc_triangle(),), (polygon(3),)])
def test_connectivity(cell):
    cell = cell.to_fiat()
    for dim0 in range(cell.get_spatial_dimension()+1):
        connectivity = cell.get_connectivity()[(dim0, 0)]
        topology = cell.get_topology()[dim0]
        assert len(connectivity) == len(topology)

        assert all(connectivity[i] == t for i, t in topology.items())


def test_tensor_connectivity():
    from test_2d_examples_docs import construct_cg1
    A = construct_cg1()
    B = construct_cg1()
    cell = tensor_product(A, B).cell
    cell = cell.to_fiat()
    for dim0 in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        connectivity = cell.get_connectivity()[(dim0, (0, 0))]
        topology = cell.get_topology()[dim0]
        assert len(connectivity) == len(topology)

        assert all(connectivity[i] == t for i, t in topology.items())


@pytest.mark.parametrize(["cell"], [(ufc_triangle(),), (polygon(3),), (make_tetrahedron(), ), (make_tetrahedron(), )])
def test_new_connectivity(cell):
    cell = cell.to_fiat()
    for dim0 in range(cell.get_dimension() + 1):
        connectivity = cell.get_connectivity()[(dim0, 0)]
        topology = cell.get_topology()[dim0]
        assert len(connectivity) == len(topology)
        for i, t in topology.items():
            print(connectivity[i])
            print(t)
        assert all(connectivity[i] == t for i, t in topology.items())


def test_compare_tris():
    fuse_tri = polygon(3)
    ufc_tri = ufc_triangle()
    fiat_tri = ufc_simplex(2)

    # The three representations use different local vertex numbering
    # conventions, so individual cone entries are not expected to
    # agree. The per-dimension entity counts (and hence the cone
    # offsets) are a numbering-independent invariant that must match.
    fiat_cones, fiat_offsets = make_entity_cone_lists(fiat_tri)
    fuse_cones, fuse_offsets = make_entity_cone_lists(fuse_tri.to_fiat())
    ufc_cones, ufc_offsets = make_entity_cone_lists(ufc_tri.to_fiat())

    assert fuse_offsets == fiat_offsets
    assert ufc_offsets == fiat_offsets
    assert len(fuse_cones) == len(fiat_cones)
    assert len(ufc_cones) == len(fiat_cones)


def test_compare_tets():
    fuse_tet = make_tetrahedron()
    ufc_tet = ufc_tetrahedron()
    fiat_tet = ufc_simplex(3)

    fiat_cones, fiat_offsets = make_entity_cone_lists(fiat_tet)
    fuse_cones, fuse_offsets = make_entity_cone_lists(fuse_tet.to_fiat())
    ufc_cones, ufc_offsets = make_entity_cone_lists(ufc_tet.to_fiat())

    assert fuse_offsets == fiat_offsets
    assert ufc_offsets == fiat_offsets
    assert len(fuse_cones) == len(fiat_cones)
    assert len(ufc_cones) == len(fiat_cones)


@pytest.mark.parametrize(["cell"], [(polygon(3),), (make_tetrahedron(),)])
def test_sub_entities_preserve_local_vertex_order(cell):
    """fuse supplies FIAT with an explicit ``sub_entities`` mapping
    (via ``CellComplexToFiatSimplex``) whose per-entity ordering
    follows that entity's own local vertex tuple. FIAT's default
    (auto-computed) ``sub_entities`` instead sorts sub-entities by
    entity id, discarding that ordering. This checks that fuse's
    mapping is actually preserving order where FIAT's default would
    not."""
    fiat_cell = cell.to_fiat()
    topology = fiat_cell.get_topology()
    auto = Simplex(fiat_cell.get_shape(), fiat_cell.vertices, topology)

    saw_a_reordering_case = False
    for dim in range(1, fiat_cell.get_spatial_dimension() + 1):
        for entity, vertex_tuple in topology[dim].items():
            fuse_order = tuple(e for d, e in fiat_cell.sub_entities[dim][entity] if d == 0)
            auto_order = tuple(e for d, e in auto.sub_entities[dim][entity] if d == 0)

            assert fuse_order == vertex_tuple
            assert auto_order == tuple(sorted(vertex_tuple))

            if vertex_tuple != tuple(sorted(vertex_tuple)):
                saw_a_reordering_case = True

    # Guard against the test vacuously passing because every local
    # vertex tuple already happened to be sorted.
    assert saw_a_reordering_case


@pytest.mark.parametrize(["cell"], [(ufc_triangle(),), (polygon(3),), (make_tetrahedron(),), (ufc_tetrahedron(),)])
def test_sub_entity_counts(cell):
    """For a k-dimensional simplex, the number of sub-entities
    (including itself) is 2**(k+1) - 1: one for every non-empty
    subset of its k+1 vertices."""
    fiat_cell = cell.to_fiat()
    for dim, entities in fiat_cell.sub_entities.items():
        for entity, sub_ents in entities.items():
            assert len(sub_ents) == 2 ** (dim + 1) - 1


def make_entity_cone_lists(fiat_cell):
    _dim = fiat_cell.get_dimension()
    _connectivity = fiat_cell.connectivity
    _list = []
    _offset_list = [0 for _ in _connectivity[(0, 0)]]  # vertices have no cones
    _offset = 0
    _n = 0  # num. of entities up to dimension = _d
    for _d in range(_dim):
        _n1 = len(_offset_list)
        for _conn in _connectivity[(_d + 1, _d)]:
            _list += [_c + _n for _c in _conn]  # These are indices into cell_closure[some_cell]
            _offset_list.append(_offset)
            _offset += len(_conn)
        _n = _n1
    _offset_list.append(_offset)
    return _list, _offset_list


def test_tet_groups():
    for cell in [make_tetrahedron(), ufc_tetrahedron()]:
        group = S4.add_cell(cell)
        for j in [1, 2]:

            sub_group = []
            flip_group = []
            for i in range(len(cell.d_entities(j))):
                face = cell.d_entities(j)[i]
                print(face)
                for g in group.members():
                    res = cell.permute_entities(g, j)[0]
                    if res[1].perm.is_Identity and res[0] == face.id and (j == 2 or g.perm.is_even):
                        print(g, res)
                        sub_group += [g.perm]
                    # elif res[1].perm.array_form == [1, 0, 2] and res[0] == face.id and (j == 2 or g.perm.is_even): [0, 2, 1] [2, 1, 0]
                    elif res[1].perm.array_form == [0, 1, 2] and res[0] == face.id and (j == 2 or g.perm.is_even):
                        print(g, res)
                        flip_group += [g.perm]
                print()
            print([s.array_form for s in sub_group])
            print([s.array_form for s in flip_group])


@pytest.mark.parametrize("attachment", [(sp.Integer(-1), sp.Symbol("x")),
                                        (sp.Symbol("x"), sp.Integer(-1)),
                                        (sp.Integer(-1), sp.Integer(1))])
def test_edge_components_evaluate_numerically(attachment):
    """Components without symbols must evaluate like the ones that have them.

    Attachments mixing constant and symbolic components previously returned a
    mix of floats and sympy objects, which numpy cannot compare.
    """
    edge = Edge(Point(0), attachment=attachment)
    res = edge(0.5)
    expected = tuple(float(c.subs({sp.Symbol("x"): 0.5})) for c in attachment)
    assert res == expected
    assert all(not isinstance(v, sp.Expr) for v in res)
