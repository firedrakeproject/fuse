import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import networkx as nx
import fuse.groups as fuse_groups
import copy
import sympy as sp
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from sympy.combinatorics.named_groups import SymmetricGroup
from fuse.utils import sympy_to_numpy, fold_reduce
from FIAT.reference_element import Simplex, TensorProductCell as FiatTensorProductCell, Hypercube
from ufl.cell import Cell, TensorProductCell


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def topo_pos(G):
    """
    Helper function for hasse diagram visualisation
    Offsets the nodes and displays in topological order
    """
    pos_dict = {}
    for i, node_list in enumerate(nx.topological_generations(G)):
        x_offset = len(node_list) / 2
        y_offset = 0
        for j, name in enumerate(node_list):
            pos_dict[name] = (j - x_offset, i - j * y_offset)

    return pos_dict


def normalise(v):
    norm = np.linalg.norm(v)
    return v / norm


def make_arrow(ax, mid, edge, direction=1):
    delta = 0.0001 if direction >= 0 else -0.0001
    x, y = edge(mid)
    dir_x, dir_y = edge(mid + delta)
    ax.arrow(x, y, dir_x-x, dir_y-y, head_width=0.05, head_length=0.1)


def make_arrow_3d(ax, mid, edge, direction=1):
    delta = 0.0001 if direction >= 0 else -0.0001
    x, y, z = edge(mid)
    dir_x, dir_y, dir_z = edge(mid + delta)
    a = Arrow3D([x, dir_x], [y, dir_y], [z, dir_z], mutation_scale=10, arrowstyle="-|>", color="black")
    ax.add_artist(a)


def construct_attach_2d(a, b, c, d):
    """
    Compute polynomial attachment in x based on two points (a,b) and (c,d)

    :param: a,b,c,d: two points (a,b) and (c,d)
    """
    x = sp.Symbol("x")
    return [((c-a)/2)*(x+1) + a, ((d-b)/2)*(x+1) + b]


def construct_attach_3d(res):
    """
    Convert matrix of coefficients into a vector of polynomials in x and y

    :param: res: matrix of coefficients
    """
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    xy = sp.Matrix([1, x, y])
    return (xy.T * res)


def compute_scaled_verts(d, n):
    """
    Construct default cell vertices

    :param: d: dimension of cell
    :param: n: number of vertices
    """
    if d == 2:
        source = np.array([0, 1])
        rot_coords = [source for i in range(0, n)]

        rot_mat = np.array([[np.cos((2*np.pi)/n), -np.sin((2*np.pi)/n)], [np.sin((2*np.pi)/n), np.cos((2*np.pi)/n)]])
        for i in range(1, n):
            rot_coords[i] = np.matmul(rot_mat, rot_coords[i-1])
        xdiff, ydiff = (rot_coords[0][0] - rot_coords[1][0],
                        rot_coords[0][1] - rot_coords[1][1])
        scale = 2 / np.sqrt(xdiff**2 + ydiff**2)
        scaled_coords = np.array([[scale*x, scale*y] for (x, y) in rot_coords])
        return scaled_coords
    elif d == 3:
        if n == 4:
            A = [-1, 1, -1]
            B = [1, -1, -1]
            C = [1, 1, 1]
            D = [-1, -1, 1]
            coords = [A, B, C, D]
            face1 = np.array([A, D, C])
            face2 = np.array([A, B, D])
            face3 = np.array([A, C, B])
            face4 = np.array([B, D, C])
            faces = [face1, face2, face3, face4]
        elif n == 8:
            coords = []
            faces = [[] for i in range(6)]
            for i in [-1, 1]:
                for j in [-1, 1]:
                    for k in [-1, 1]:
                        coords.append([i, j, k])

            for j in [-1, 1]:
                for k in [-1, 1]:
                    faces[0].append([1, j, k])
                    faces[1].append([-1, j, k])
                    faces[2].append([j, 1, k])
                    faces[3].append([j, -1, k])
                    faces[4].append([j, k, 1])
                    faces[5].append([j, k, -1])

        else:
            raise ValueError("Polyhedron with {} vertices not supported".format(n))

        xdiff, ydiff, zdiff = (coords[0][0] - coords[1][0],
                               coords[0][1] - coords[1][1],
                               coords[0][2] - coords[1][2])
        scale = 2 / np.sqrt(xdiff**2 + ydiff**2 + zdiff**2)
        scaled_coords = np.array([[scale*x, scale*y, scale*z] for (x, y, z) in coords])
        scaled_faces = np.array([[[scale*x, scale*y, scale*z] for (x, y, z) in face] for face in faces])

        return scaled_coords, scaled_faces
    else:
        raise ValueError("Dimension {} not supported".format(d))


def line():
    """
    Constructs the default 1D interval
    """
    return Point(1, [Point(0), Point(0)], vertex_num=2)


def polygon(n):
    """
    Constructs the 2D default cell with n sides/vertices

    :param: n: number of vertices
    """
    vertices = []
    for i in range(n):
        vertices.append(Point(0))
    edges = []
    for i in range(n):
        edges.append(
            Point(1, [vertices[(i+1) % n], vertices[(i+2) % n]], vertex_num=2))

    return Point(2, edges, vertex_num=n)


def firedrake_triangle():
    vertices = []
    for i in range(3):
        vertices.append(Point(0))
    edges = []
    edges.append(Point(1, [vertices[1], vertices[2]], vertex_num=2))
    edges.append(Point(1, [vertices[0], vertices[2]], vertex_num=2))
    edges.append(Point(1, [vertices[0], vertices[1]], vertex_num=2))
    tri = Point(2, edges, vertex_num=3, edge_orientations={1: [1, 0]})
    # tri = polygon(3)
    s3 = tri.group
    perm = s3.get_member([2, 0, 1])
    return tri.orient(perm)


def firedrake_quad():
    """
    Constructs the a quad cell that matches the firedrake default.
    """
    vertices = []
    for i in range(4):
        vertices.append(Point(0))
    edges = []
    # edges.append(Point(1, [vertices[0], vertices[1]], vertex_num=2))
    # edges.append(Point(1, [vertices[1], vertices[2]], vertex_num=2))
    # edges.append(Point(1, [vertices[3], vertices[2]], vertex_num=2))
    # edges.append(Point(1, [vertices[0], vertices[3]], vertex_num=2))

    edges.append(Point(1, [vertices[0], vertices[1]], vertex_num=2))
    edges.append(Point(1, [vertices[1], vertices[3]], vertex_num=2))
    edges.append(Point(1, [vertices[2], vertices[3]], vertex_num=2))
    edges.append(Point(1, [vertices[0], vertices[2]], vertex_num=2))

    return Point(2, edges, vertex_num=4, edge_orientations={2: [1, 0], 3: [1, 0]})


def make_tetrahedron():
    vertices = []
    for i in range(4):
        vertices.append(Point(0))
    edges = []
    edges.append(
        Point(1, vertex_num=2, edges=[vertices[0], vertices[1]]))
    edges.append(
        Point(1, vertex_num=2, edges=[vertices[1], vertices[2]]))
    edges.append(
        Point(1, vertex_num=2, edges=[vertices[2], vertices[0]]))
    edges.append(
        Point(1, vertex_num=2, edges=[vertices[3], vertices[0]]))
    edges.append(
        Point(1, vertex_num=2, edges=[vertices[1], vertices[3]]))
    edges.append(
        Point(1, vertex_num=2, edges=[vertices[2], vertices[3]]))

    face1 = Point(2, vertex_num=3, edges=[edges[5], edges[3], edges[2]], edge_orientations={2: [1, 0]})
    face2 = Point(2, vertex_num=3, edges=[edges[3], edges[0], edges[4]])
    face3 = Point(2, vertex_num=3, edges=[edges[2], edges[0], edges[1]])
    face4 = Point(2, vertex_num=3, edges=[edges[1], edges[4], edges[5]], edge_orientations={0: [1, 0], 2: [1, 0]})

    return Point(3, vertex_num=4, edges=[face3, face1, face4, face2])


class Point():
    """
    Cell complex representation of a finite element cell

    :param: d: dimension of the cell
    :param: edges: list of subcells (either as edge or point objects)
    :param: vertex_num: Optional argument, number of vertices
    :param: oriented: adds orientation to the cell
    :param: group: Symmetry group of the cell
    :param: edge_orientations: dictionary of the orientations of the subcells

    """

    id_iter = itertools.count()

    def __init__(self, d, edges=[], vertex_num=None, oriented=False, group=None, edge_orientations=None, cell_id=None):
        if not cell_id:
            cell_id = next(self.id_iter)
        self.id = cell_id
        self.dimension = d

        if edge_orientations is None:
            edge_orientations = {}

        if d == 0:
            assert (edges == [])
        if vertex_num:
            edges = self.compute_attachments(vertex_num, edges, edge_orientations)

        self.oriented = oriented
        self.G = nx.DiGraph()
        self.G.add_node(self.id, point_class=self)
        for edge in edges:
            assert edge.lower_dim() < self.dimension
            self.G.add_edge(self.id, edge.point.id, edge_class=edge)
        self.G = nx.compose_all([self.G]
                                + [edge.point.graph() for edge in edges])
        self.connections = edges

        self.group = group
        if not group:
            self.group = self.compute_cell_group()

        self.group = self.group.add_cell(self)

    def compute_attachments(self, n, points, orientations=None):
        """
        Compute the attachment function between two nodes

        :param: n: number of vertices
        :param: points: List of Point objects
        :param: orientations: (Optional) Orientation associated with the attachment
        """
        if orientations is None:
            orientations = {}

        if self.dimension == 1:
            edges = [Edge(points[0], sp.sympify((-1,))),
                     Edge(points[1], sp.sympify((1,)))]
        if self.dimension == 2:
            coords = compute_scaled_verts(2, n)
            edges = []

            for i in range(n):
                a, b = coords[i]
                c, d = coords[(i + 1) % n]

                if i in orientations.keys():
                    edges.append(Edge(points[i], construct_attach_2d(a, b, c, d), o=points[i].group.get_member(orientations[i])))
                else:
                    edges.append(Edge(points[i], construct_attach_2d(a, b, c, d)))
        if self.dimension == 3:
            coords, faces = compute_scaled_verts(3, n)
            coords_2d = np.c_[np.ones(len(faces[0])), compute_scaled_verts(2, len(faces[0]))]
            res = []
            edges = []

            for i in range(len(faces)):
                res = np.linalg.solve(coords_2d, faces[i])

                res_fn = construct_attach_3d(res)
                # breakpoint()
                assert np.allclose(np.array(res_fn.subs({"x": coords_2d[0][1], "y": coords_2d[0][2]})).astype(np.float64), faces[i][0])
                assert np.allclose(np.array(res_fn.subs({"x": coords_2d[1][1], "y": coords_2d[1][2]})).astype(np.float64), faces[i][1])
                assert np.allclose(np.array(res_fn.subs({"x": coords_2d[2][1], "y": coords_2d[2][2]})).astype(np.float64), faces[i][2])
                if i in orientations.keys():
                    edges.append(Edge(points[i], construct_attach_3d(res), o=points[i].group.get_member(orientations[i])))
                else:
                    edges.append(Edge(points[i], construct_attach_3d(res)))

                # breakpoint()
        return edges

    def compute_cell_group(self):
        """
        Systematically work out the symmetry group of the constructed cell.

        Assumes cell has side length of 2.
        """
        verts = self.ordered_vertices()
        v_coords = [self.get_node(v, return_coords=True) for v in verts]
        n = len(verts)
        max_group = SymmetricGroup(n)
        edges = [edge.ordered_vertices() for edge in self.edges()]
        accepted_perms = max_group.elements.copy()
        if n > 2:
            for element in max_group.elements:
                reordered = element(verts)
                for edge in edges:
                    diff = np.subtract(v_coords[reordered.index(edge[0])], v_coords[reordered.index(edge[1])]).squeeze()
                    edge_len = np.sqrt(np.dot(diff, diff))
                    if not np.allclose(edge_len, 2):
                        accepted_perms.remove(element)
                        break
        return fuse_groups.PermutationSetRepresentation(list(accepted_perms))

    def get_spatial_dimension(self):
        return self.dimension

    def dim(self):
        return self.dimension

    def get_shape(self):
        num_verts = len(self.vertices())
        if num_verts == 1:
            # Point
            return 0
        elif num_verts == 2:
            # Line
            return 1
        elif num_verts == 3:
            # Triangle
            return 2
        elif num_verts == 4:
            if self.dimension == 2:
                # quadrilateral
                return 11
            elif self.dimension == 3:
                # tetrahedron
                return 3
        elif num_verts == 8:
            # hexahedron
            return 111
        else:
            raise TypeError("Shape undefined for {}".format(str(self)))

    def get_topology(self):
        structure = [sorted(generation) for generation in nx.topological_generations(self.graph())]
        structure.reverse()

        min_ids = [min(dimension) for dimension in structure]
        vertices = self.ordered_vertices()
        relabelled_verts = {vertices[i]: i for i in range(len(vertices))}

        self.topology = {}
        self.topology_unrelabelled = {}
        for i in range(len(structure)):
            dimension = structure[i]
            self.topology[i] = {}
            self.topology_unrelabelled[i] = {}
            for node in dimension:
                self.topology[i][node - min_ids[i]] = tuple([relabelled_verts[vert] for vert in self.get_node(node).ordered_vertices()])
                self.topology_unrelabelled[i][node - min_ids[i]] = tuple([vert - min_ids[0] for vert in self.get_node(node).ordered_vertices()])
        return self.topology_unrelabelled

    def get_starter_ids(self):
        structure = [sorted(generation) for generation in nx.topological_generations(self.G)]
        structure.reverse()

        min_ids = [min(dimension) for dimension in structure]
        return min_ids

    def graph_dim(self):
        if self.oriented:
            dim = self.dimension + 1
        else:
            dim = self.dimension
        return dim

    def graph(self):
        if self.oriented:
            temp_G = self.G.copy()
            temp_G.remove_node(-1)
            return temp_G
        return self.G

    def hasse_diagram(self, counter=0, filename=None):
        ax = plt.axes()
        nx.draw_networkx(self.G, pos=topo_pos(self.G),
                         with_labels=True, ax=ax)
        edge_dict = {(u, v): self.G.edges[u, v]["edge_class"].o for (u, v) in self.G.edges()}
        nx.draw_networkx_edge_labels(self.G, pos=topo_pos(self.G), edge_labels=edge_dict, ax=ax)
        if filename:
            ax.figure.savefig(filename)
        else:
            plt.show()

    def ordered_vertices(self, get_class=False):
        # define a points vertex order by combining the order of the sub elements
        # vertex list and removing duplicates
        if self.dimension == 0:
            if get_class:
                return [self]
            return [self.id]
        else:
            # convert to dict to remove duplicates while maintaining order
            full_list = [c.ordered_vertices(get_class) for c in self.connections]
            flatten = itertools.chain.from_iterable(full_list)
            verts = list(dict.fromkeys(flatten))
            if self.oriented:
                # make sure this is necessary
                return self.oriented.permute(verts)
            return verts

    def ordered_vertex_coords(self):
        return [self.get_node(o, return_coords=True) for o in self.ordered_vertices()]

    def d_entities_ids(self, d):
        return self.d_entities(d, get_class=False)

    def d_entities(self, d, get_class=True):
        levels = [sorted(generation)
                  for generation in nx.topological_generations(self.G)]
        if get_class:
            res = [self.G.nodes.data("point_class")[i] for i in levels[self.graph_dim() - d]]
        else:
            res = levels[self.graph_dim() - d]
        return res

    def get_node(self, node, return_coords=False):
        if return_coords:
            top_level_node = self.d_entities_ids(self.graph_dim())[0]
            if self.dimension == 0:
                return ()
            return self.attachment(top_level_node, node)()
        return self.G.nodes.data("point_class")[node]

    def dim_of_node(self, node):
        levels = [sorted(generation)
                  for generation in nx.topological_generations(self.G)]
        for i in range(len(levels)):
            if node in levels[i]:
                return self.graph_dim() - i
        raise ValueError("Node not found in graph")

    def vertices(self, get_class=True, return_coords=False):
        # TODO maybe refactor with get_node
        verts = self.d_entities(0, get_class)
        if return_coords:
            verts = self.d_entities_ids(0)
            top_level_node = self.d_entities_ids(self.graph_dim())[0]
            if self.dimension == 0:
                return [()]
            return [self.attachment(top_level_node, v)() for v in verts]
        return verts

    def edges(self, get_class=True):
        return self.d_entities(1, get_class)

    def permute_entities(self, g, d):
        # TODO something is wrong here for squares it can return [()]
        verts = self.vertices(get_class=False)
        entities = self.d_entities_ids(d)
        reordered = g.permute(verts)

        if d == 0:
            entity_group = self.d_entities(d)[0].group
            return list(zip(reordered, [entity_group.identity for r in reordered]))

        entity_dict = {}
        reordered_entity_dict = {}

        for e in self.d_entities(d):
            entity_dict[e.id] = tuple(e.ordered_vertices())
            reordered_entity_dict[e.id] = tuple([reordered[verts.index(i)] for i in e.ordered_vertices()])

        reordered_entities = [tuple() for e in range(len(entities))]
        min_id = min(entities)
        entity_group = self.d_entities(d)[0].group
        for ent in entities:
            for ent1 in entities:
                if set(entity_dict[ent]) == set(reordered_entity_dict[ent1]):
                    if entity_dict[ent] != reordered_entity_dict[ent1]:
                        o = entity_group.transform_between_perms(entity_dict[ent], reordered_entity_dict[ent1])
                        reordered_entities[ent1 - min_id] = (ent, o)
                    else:
                        reordered_entities[ent1 - min_id] = (ent, entity_group.identity)

        return reordered_entities

    def basis_vectors(self, return_coords=True, entity=None):
        if not entity:
            entity = self
        entity_levels = [sorted(generation) for generation in nx.topological_generations(entity.G)]
        self_levels = [sorted(generation) for generation in nx.topological_generations(self.G)]
        vertices = entity_levels[entity.graph_dim()]
        if self.dimension == 0:
            raise ValueError("Dimension 0 entities cannot have Basis Vectors")
        top_level_node = self_levels[0][0]
        v_0 = vertices[0]
        if return_coords:
            v_0_coords = self.attachment(top_level_node, v_0)()
        basis_vecs = []
        for v in vertices[1:]:
            if return_coords:
                v_coords = self.attachment(top_level_node, v)()
                sub = normalise(np.subtract(v_coords, v_0_coords))
                if not hasattr(sub, "__iter__"):
                    basis_vecs.append((sub,))
                else:
                    basis_vecs.append(tuple(sub))
            else:
                basis_vecs.append((v, v_0))
        return basis_vecs

    def plot(self, show=True, plain=False, ax=None, filename=None):
        """ for now into 2 dimensional space """

        top_level_node = self.d_entities(self.graph_dim(), get_class=False)[0]
        xs = np.linspace(-1, 1, 20)
        if ax is None:
            ax = plt.gca()

        if self.dimension == 1:
            # line plot in 1D case
            nodes = self.d_entities(0, get_class=False)
            points = []
            for node in nodes:
                attach = self.attachment(top_level_node, node)
                points.extend(attach())
            plt.plot(np.array(points), np.zeros_like(points), color="black")

        for i in range(self.dimension - 1, -1, -1):
            nodes = self.d_entities(i, get_class=False)
            vert_coords = []
            for node in nodes:
                attach = self.attachment(top_level_node, node)
                if i == 0:
                    plotted = attach()
                    if len(plotted) < 2:
                        plotted = (plotted[0], 0)
                    vert_coords += [plotted]
                    if not plain:
                        plt.plot(plotted[0], plotted[1], 'bo')
                        plt.annotate(node, (plotted[0], plotted[1]))
                elif i == 1:
                    edgevals = np.array([attach(x) for x in xs])
                    if len(edgevals[0]) < 2:
                        plt.plot(edgevals[:, 0], 0, color="black")
                    else:
                        plt.plot(edgevals[:, 0], edgevals[:, 1], color="black")
                    if not plain:
                        make_arrow(ax, 0, attach)
                else:
                    raise ValueError("General plotting not implemented")
            # if i == 2:
            #     if len(vert_coords) > 2:
            #         hull = ConvexHull(vert_coords)
            #         plt.fill(vert_coords[hull.vertices, 0], vert_coords[hull.vertices, 1], alpha=0.5)
        if show:
            plt.show()
        if filename:
            ax.figure.savefig(filename)
            plt.cla()

    def plot3d(self, show=True, ax=None):
        assert self.dimension == 3
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        xs = np.linspace(-1, 1, 20)

        top_level_node = self.d_entities_ids(self.graph_dim())[0]
        nodes = self.d_entities_ids(0)
        for node in nodes:
            attach = self.attachment(top_level_node, node)
            plotted = attach()
            ax.scatter(plotted[0], plotted[1], plotted[2], color="black")

        nodes = self.d_entities_ids(1)
        for node in nodes:
            attach = self.attachment(top_level_node, node)
            edgevals = np.array([attach(x) for x in xs])
            ax.plot3D(edgevals[:, 0], edgevals[:, 1], edgevals[:, 2], color="black")
            make_arrow_3d(ax, 0, attach)
        if show:
            plt.show()

    def attachment(self, source, dst):
        if source == dst:
            # return x
            return lambda *x: x

        paths = nx.all_simple_edge_paths(self.G, source, dst)
        attachments = [[self.G[s][d]["edge_class"]
                        for (s, d) in path] for path in paths]

        if len(attachments) == 0:
            raise ValueError("No paths from node {} to node {}"
                             .format(source, dst))

        # check all attachments resolve to the same function
        if len(attachments) > 1:
            dst_dim = self.dim_of_node(dst)
            basis = np.eye(dst_dim)
            if dst_dim == 0:
                vals = [fold_reduce(attachment) for attachment in attachments]
                assert all(np.isclose(val, vals[0]).all() for val in vals)
            else:
                for i in range(dst_dim):
                    vals = [fold_reduce(attachment, *tuple(basis[i].tolist()))
                            for attachment in attachments]
                    assert all(np.isclose(val, vals[0]).all() for val in vals)

        return lambda *x: fold_reduce(attachments[0], *x)

    def cell_attachment(self, dst):
        if not isinstance(dst, int):
            raise ValueError
        top_level_node = self.d_entities_ids(self.graph_dim())[0]
        return self.attachment(top_level_node, dst)

    def orient(self, o):
        """ Orientation node is always labelled with -1 """
        oriented_point = copy.deepcopy(self)
        top_level_node = oriented_point.d_entities_ids(
            oriented_point.dimension)[0]
        oriented_point.G.add_node(-1, point_class=None)
        oriented_point.G.add_edge(-1, top_level_node,
                                  edge_class=Edge(None, o=o))
        oriented_point.oriented = o
        return oriented_point

    def __repr__(self):
        entity_name = ["v", "e", "f", "c"]
        return entity_name[self.dimension] + str(self.id)

    def copy(self):
        return copy.deepcopy(self)

    def to_fiat(self, name=None):
        if len(self.vertices()) == self.dimension + 1:
            return CellComplexToFiatSimplex(self, name)
        if len(self.vertices()) == 2 ** self.dimension:
            return CellComplexToFiatHypercube(self, name)
        raise NotImplementedError("Custom shape elements/ First class quads are not yet supported")

    def to_ufl(self, name=None):
        return CellComplexToUFL(self, name)

    def _to_dict(self):
        # think this is probably missing stuff
        o_dict = {"dim": self.dimension,
                  "edges": [c for c in self.connections],
                  "oriented": self.oriented,
                  "id": self.id}
        return o_dict

    def dict_id(self):
        return "Cell"

    def _from_dict(o_dict):
        return Point(o_dict["dim"], o_dict["edges"], oriented=o_dict["oriented"], cell_id=o_dict["id"])

    def __eq__(self, other):
        if self.dimension != other.dimension:
            return False
        if set(self.ordered_vertex_coords()) != set(other.ordered_vertex_coords()):
            return False
        return self.get_topology() == other.get_topology()

    def __hash__(self):
        return hash(self.id)


class Edge():
    """
    Representation of the connections in a cell complex.

    :param: point: the point being connected (lower level)
    :param: attachment: the function describing how the point is attached
    :param: o: orientation function (optional)
    """

    def __init__(self, point, attachment=None, o=None):
        self.attachment = attachment
        self.point = point
        self.o = o

    def __call__(self, *x):
        if self.o:
            x = self.o(x)
        if self.attachment:
            syms = ["x", "y", "z"]
            if hasattr(self.attachment, '__iter__'):
                res = []
                for attach_comp in self.attachment:
                    if len(attach_comp.atoms(sp.Symbol)) == len(x):
                        res.append(sympy_to_numpy(attach_comp, syms, x))
                    else:
                        res.append(attach_comp.subs({syms[i]: x[i] for i in range(len(x))}))
                return tuple(res)
            return sympy_to_numpy(self.attachment, syms, x)
        return x

    def ordered_vertices(self, get_class=False):
        verts = self.point.ordered_vertices(get_class)
        if self.o:
            verts = self.o.permute(verts)
        return verts

    def lower_dim(self):
        return self.point.dim()

    def __repr__(self):
        return str(self.point)

    def _to_dict(self):
        o_dict = {"attachment": self.attachment,
                  "point": self.point,
                  "orientation": self.o}
        return o_dict

    def dict_id(self):
        return "Edge"

    def _from_dict(o_dict):
        return Edge(o_dict["point"], o_dict["attachment"], o_dict["orientation"])


class TensorProductPoint():

    def __init__(self, A, B, flat=False):
        self.A = A
        self.B = B
        self.dimension = self.A.dimension + self.B.dimension
        self.flat = flat

    def get_spatial_dimension(self):
        return self.dimension

    def dimension(self):
        return tuple(self.A.dimension, self.B.dimension)

    def d_entities(self, d, get_class=True):
        return self.A.d_entities(d, get_class) + self.B.d_entities(d, get_class)

    def vertices(self, get_class=True, return_coords=False):
        # TODO maybe refactor with get_node
        verts = self.d_entities(0, get_class)
        if return_coords:
            a_verts = self.A.vertices(return_coords=return_coords)
            b_verts = self.B.vertices(return_coords=return_coords)
            return [a + b for a in a_verts for b in b_verts]
        return verts

    def to_ufl(self, name=None):
        if self.flat:
            return CellComplexToUFL(self, "quadrilateral")
        return TensorProductCell(self.A.to_ufl(), self.B.to_ufl())

    def to_fiat(self, name=None):
        if self.flat:
            return CellComplexToFiatHypercube(self, CellComplexToFiatTensorProduct(self, name))
        return CellComplexToFiatTensorProduct(self, name)

    def flatten(self):
        assert self.A == self.B
        return TensorProductPoint(self.A, self.B, True)


class CellComplexToFiatSimplex(Simplex):
    """
    Convert cell complex to fiat

    :param: cell: a fuse cell complex

    Currently assumes simplex.
    """

    def __init__(self, cell, name=None):
        self.fe_cell = cell
        if name is None:
            name = "FuseCell"
        self.name = name

        # verts = [cell.get_node(v, return_coords=True) for v in cell.ordered_vertices()]
        verts = cell.vertices(return_coords=True)
        topology = cell.get_topology()
        shape = cell.get_shape()
        super(CellComplexToFiatSimplex, self).__init__(shape, verts, topology)

    def cellname(self):
        return self.name

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        return self.fe_cell.d_entities(dimension)[0].to_fiat()

    def get_facet_element(self):
        dimension = self.get_spatial_dimension()
        return self.construct_subelement(dimension - 1)


class CellComplexToFiatTensorProduct(FiatTensorProductCell):
    """
    Convert cell complex to fiat

    :param: cell: a fuse tensor product cell complex
    """
    def __new__(cls, cell, name=None, *args, **kwargs):
        if not isinstance(cell, TensorProductPoint):
            return cell.to_fiat()
        return super(CellComplexToFiatTensorProduct, cls).__new__(cls, *args, **kwargs)

    def __init__(self, cell, name=None):
        self.fe_cell = cell
        self.sub_cells = [cell.A.to_fiat(), cell.B.to_fiat()]
        if name is None:
            name = " * ".join([s.name for s in self.sub_cells])
        self.name = name

        super(CellComplexToFiatTensorProduct, self).__init__(cell.A.to_fiat(), cell.B.to_fiat())

    def cellname(self):
        return self.name

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        return CellComplexToFiatTensorProduct(*[c.construct_subelement(d).fe_cell for c, d in zip(self.cells, dimension)])

    def get_facet_element(self):
        dimension = self.get_spatial_dimension()
        return self.construct_subelement(dimension - 1)

    def flatten(self):
        return CellComplexToFiatHypercube(self.fe_cell, self)


class CellComplexToFiatHypercube(Hypercube):
    """
    Convert cell complex to fiat

    :param: cell: a fuse cell complex

    """

    def __init__(self, cell, product):
        self.fe_cell = cell

        super(CellComplexToFiatHypercube, self).__init__(product.get_spatial_dimension(), product)

    def cellname(self):
        return self.name

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        if dimension == self.get_dimension():
            return self
        # assumes symmetric tensor product
        sub_element = self.product.construct_subelement((dimension,) + (0,)*(len(self.product.cells) - 1))
        if isinstance(sub_element, CellComplexToFiatTensorProduct):
            return sub_element.flatten()
        return sub_element

    def get_facet_element(self):
        dimension = self.get_spatial_dimension()
        return self.construct_subelement(dimension - 1)

    def get_dimension(self):
        return self.get_spatial_dimension()


class CellComplexToUFL(Cell):
    """
    Convert cell complex to UFL

    :param: cell: a fuse cell complex

    Currently just maps to a subset of existing UFL cells
    TODO work out generic way around the naming issue
    """

    def __init__(self, cell, name=None):
        self.cell_complex = cell

        # TODO work out generic way around the naming issue
        if not name:
            num_verts = len(cell.vertices())
            if num_verts == 1:
                # Point
                name = "vertex"
            elif num_verts == 2:
                # Line
                name = "interval"
            elif num_verts == 3:
                # Triangle
                name = "triangle"
            elif num_verts == 4:
                if cell.dimension == 2:
                    # quadrilateral
                    name = "quadrilateral"
                elif cell.dimension == 3:
                    # tetrahedron
                    name = "tetrahedron"
            elif num_verts == 8:
                # hexahedron
                name = "hexahedron"
            else:
                raise TypeError("UFL cell conversion undefined for {}".format(str(cell)))
        super(CellComplexToUFL, self).__init__(name)

    def to_fiat(self):
        return self.cell_complex.to_fiat(name=self.cellname())

    def __repr__(self):
        return super(CellComplexToUFL, self).__repr__()

    def reconstruct(self, **kwargs):
        """Reconstruct this cell, overwriting properties by those in kwargs."""
        cell = self.cell_complex
        for key, value in kwargs.items():
            if key == "cell":
                cell = value
            else:
                raise TypeError(f"reconstruct() got unexpected keyword argument '{key}'")
        return CellComplexToUFL(cell, self._cellname)


def constructCellComplex(name):
    if name == "vertex":
        return Point(0).to_ufl(name)
    elif name == "interval":
        return Point(1, [Point(0), Point(0)], vertex_num=2).to_ufl(name)
    elif name == "triangle":
        return polygon(3).to_ufl(name)
        # return firedrake_triangle().to_ufl(name)
    elif name == "quadrilateral":
        interval = Point(1, [Point(0), Point(0)], vertex_num=2)
        return TensorProductPoint(interval, interval).flatten().to_ufl(name)
        # return firedrake_quad().to_ufl(name)
        # return polygon(4).to_ufl(name)
    elif name == "tetrahedron":
        return make_tetrahedron().to_ufl(name)
    elif name == "hexahedron":
        import warnings
        warnings.warn("Hexahedron unimplemented in Fuse")
        import ufl
        return ufl.Cell(name)
    elif "*" in name:
        components = [constructCellComplex(c.strip()).cell_complex for c in name.split("*")]
        return TensorProductPoint(*components).to_ufl(name)
    else:
        raise TypeError("Cell complex construction undefined for {}".format(str(name)))
