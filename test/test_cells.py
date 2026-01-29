from fuse import *
from firedrake import *
from fuse.cells import ufc_triangle, ufc_tetrahedron
import pytest
import numpy as np
from FIAT.reference_element import default_simplex, ufc_simplex
from test_convert_to_fiat import helmholtz_solve


@pytest.fixture(scope='module', params=[0, 1, 2])
def C(request):
    dim = request.param
    if dim == 0:
        return Point(0)
    elif dim == 1:
        return Point(1, [Point(0), Point(0)], vertex_num=2)
    elif dim == 2:
        return polygon(3)


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
        mesh = UnitSquareMesh(2 ** i, 2 ** i)

        V = FunctionSpace(mesh, "CG", 3)
        res1 = helmholtz_solve(mesh, V)
        diff2[i-3] = res1

    print("firedrake l2 error norms:", diff2)
    diff2 = np.array(diff2)
    conv1 = np.log2(diff2[:-1] / diff2[1:])
    print("firedrake convergence order:", conv1)
    assert (np.array(conv1) > 3.8).all()


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
    fuse_tet = polygon(3)
    ufc_tet = ufc_triangle()
    fiat_tet = ufc_simplex(2)

    print(fiat_tet.get_topology())
    print(fuse_tet.get_topology())
    print(ufc_tet.get_topology())
    fiat_connectivity = fiat_tet.get_connectivity()
    fuse_connectivity = fuse_tet.to_fiat().get_connectivity()
    ufc_connectivity = ufc_tet.to_fiat().get_connectivity()
    _dim = fiat_tet.get_dimension()
    print("fiat")
    print(make_entity_cone_lists(fiat_tet))
    for dim0 in range(_dim):
        connectivity = fiat_connectivity[(dim0+1, dim0)]
        print(connectivity)
    print("fuse")
    print(make_entity_cone_lists(fuse_tet.to_fiat()))
    for dim0 in range(_dim):
        connectivity = fuse_connectivity[(dim0+1, dim0)]
        print(connectivity)
    print("fuse ufc")
    print(make_entity_cone_lists(ufc_tet.to_fiat()))
    for dim0 in range(_dim):
        connectivity = ufc_connectivity[(dim0+1, dim0)]
        print(connectivity)


def test_compare_tets():
    tet = make_tetrahedron()
    # perm = tet.group.get_member([1, 2, 0, 3])
    fuse_tet = tet
    ufc_tet = ufc_tetrahedron()
    fiat_tet = ufc_simplex(3)
    # breakpoint()
    print(fiat_tet.get_topology())
    print(fuse_tet.get_topology())
    print(ufc_tet.get_topology())
    fiat_connectivity = fiat_tet.get_connectivity()
    fuse_connectivity = fuse_tet.to_fiat().get_connectivity()
    ufc_connectivity = ufc_tet.to_fiat().get_connectivity()
    _dim = fiat_tet.get_dimension()
    print("fiat")
    print(make_entity_cone_lists(fiat_tet))
    for dim0 in range(_dim):
        connectivity = fiat_connectivity[(dim0+1, dim0)]
        print(connectivity)
    print("fuse")
    print(make_entity_cone_lists(fuse_tet.to_fiat()))
    for dim0 in range(_dim):
        connectivity = fuse_connectivity[(dim0+1, dim0)]
        print(connectivity)
    print("fuse ufc")
    print(make_entity_cone_lists(ufc_tet.to_fiat()))
    for dim0 in range(_dim):
        connectivity = ufc_connectivity[(dim0+1, dim0)]
        print(connectivity)


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
