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
from sympy.combinatorics import Permutation
from fuse.utils import sympy_to_numpy, fold_reduce, numpy_to_str_tuple, orientation_value, _SYMBOLS
from FIAT.reference_element import Simplex, TensorProductCell as FiatTensorProductCell, Hypercube
from FIAT.quadrature_schemes import create_quadrature
from ufl.cell import Cell, TensorProductCell
from functools import cache
from collections import defaultdict
from itertools import product


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
        source = np.array([-np.sqrt(3)/2, -1/2])
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
    Constructs the default interval, [-1, 1]
    """
    return Point(1, [Point(0), Point(0)], vertex_num=2)


def polygon(n):
    """
    Constructs the 2D default cell with n sides/vertices

    The square is the product of two intervals, flattened.

    :param: n: number of vertices
    """
    if n == 4:
        return TensorProductPoint(line(), line()).flatten()
    vertices = []
    for i in range(n):
        vertices.append(Point(0))
    edges = []
    for i in range(n):
        edges.append(
            Point(1, [vertices[(i+1) % n], vertices[(i+2) % n]], vertex_num=2))

    return Point(2, edges, vertex_num=n)


def ufc_triangle():
    vertices = []
    for i in range(3):
        vertices.append(Point(0))
    edges = []
    edges.append(Point(1, [vertices[1], vertices[2]], vertex_num=2))
    edges.append(Point(1, [vertices[0], vertices[2]], vertex_num=2))
    edges.append(Point(1, [vertices[0], vertices[1]], vertex_num=2))
    tri = Point(2, edges, vertex_num=3, edge_orientations={1: [1, 0]})

    return tri


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

    tetra = Point(3, vertex_num=4, edges=[face3, face1, face4, face2])

    return tetra
    # .orient(tetra.group.members()[3])


def ufc_tetrahedron():
    vertices = []
    for i in range(4):
        vertices.append(Point(0))
    edges = []
    edges.append(
        Point(1, vertex_num=2, edges=[vertices[2], vertices[3]]))
    edges.append(
        Point(1, vertex_num=2, edges=[vertices[1], vertices[3]]))
    edges.append(
        Point(1, vertex_num=2, edges=[vertices[1], vertices[2]]))
    edges.append(
        Point(1, vertex_num=2, edges=[vertices[0], vertices[3]]))
    edges.append(
        Point(1, vertex_num=2, edges=[vertices[0], vertices[2]]))
    edges.append(
        Point(1, vertex_num=2, edges=[vertices[0], vertices[1]]))

    # face1 = Point(2, vertex_num=3, edges=[edges[0], edges[1], edges[2]])
    # face2 = Point(2, vertex_num=3, edges=[edges[0], edges[3], edges[4]])
    # face3 = Point(2, vertex_num=3, edges=[edges[2], edges[3], edges[5]])
    # face4 = Point(2, vertex_num=3, edges=[edges[2], edges[4], edges[5]])
    face1 = Point(2, vertex_num=3, edges=[edges[0], edges[3], edges[4]], edge_orientations={1: [1, 0]})
    face2 = Point(2, vertex_num=3, edges=[edges[4], edges[5], edges[2]], edge_orientations={0: [1, 0]})
    face3 = Point(2, vertex_num=3, edges=[edges[2], edges[1], edges[0]], edge_orientations={0: [1, 0], 2: [1, 0]})
    face4 = Point(2, vertex_num=3, edges=[edges[3], edges[5], edges[1]], edge_orientations={0: [1, 0]})

    tet = Point(3, vertex_num=4, edges=[face2, face1, face3, face4])

    return tet
    # return Point(3, vertex_num=4, edges=[face3, face1, face4, face2])
    # return Point(3, vertex_num=4, edges=[face1, face4, face3, face4], edge_orientations={3: [2, 1, 0]})


class Point():
    """
    Cell complex representation of a finite element cell

    Attributes
    ----------
    d: :obj:`int`
        dimension of the cell
    edges: List
        list of subcells of type:obj:`Edge` or :obj:`Point`
    vertex_num: :obj:`int` (optional)
        number of vertices
    oriented: GroupMemberRep (optional)
        Adds orientation to the cell
    group: GroupRepresentation (optional)
        Symmetry group of the cell
    edge_orientations: dict
        Dictionary of the orientations of the subcells {subcell_id: orientation}

    """

    id_iter = itertools.count()

    def __init__(self, d, edges=[], vertex_num=None, oriented=False, group=None, edge_orientations=None, cell_id=None):
        if cell_id is None:
            cell_id = next(self.id_iter)
        self.id = cell_id
        self.dimension = d

        if edge_orientations is None:
            edge_orientations = {}

        if d == 0:
            assert (edges == [])
        if vertex_num and vertex_num > 1:
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

    def dimensions(self):
        return [i for i in range(self.dimension + 1)]

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

    def get_topology(self, renumber=False):
        """The topology interface conforms to the requirements for FIAT.

        * Global vertex numbers are the vertices in id order (``vertices()``):
          FIAT's vertex coordinates are listed in that order.
        * Each entity's tuple lists its vertices in its own construction order
          (``ordered_vertices()``), so FIAT's local vertex order of every entity,
          the cell included, is FUSE's construction order. Orientation keys are
          permutations of these local positions.

        A cell whose external numbering differs (for example a hypercube
        numbering) should translate here, not in the rest of FUSE.
        """
        structure = [generation for generation in nx.topological_generations(self.graph())]
        structure.reverse()

        min_ids = [min(dimension) for dimension in structure]
        vertices = self.ordered_vertices()
        relabelled_verts = {vertices[i]: i for i in range(len(vertices))}

        self._topology = {}
        self._topology_unrelabelled = {}
        for i in range(len(structure)):
            dimension = structure[i]
            self._topology[i] = {}
            self._topology_unrelabelled[i] = {}
            for node in dimension:
                self._topology[i][node - min_ids[i]] = tuple([relabelled_verts[vert] for vert in self.get_node(node).ordered_vertices()])
                self._topology_unrelabelled[i][node - min_ids[i]] = tuple([vert - min_ids[0] for vert in self.get_node(node).ordered_vertices()])
            self._topology_unrelabelled[i] = dict(sorted(self._topology_unrelabelled[i].items()))
        if renumber:
            return self._topology
        return self._topology_unrelabelled

    def get_renumbered_topology(self):
        structure = [generation for generation in nx.topological_generations(self.graph())]
        structure.reverse()

        min_ids = [min(dimension) for dimension in structure]
        vertices = self.ordered_vertices()
        relabelled_verts = {vertices[i]: i for i in range(len(vertices))}

        self._topology = {}
        for i in range(len(structure)):
            dimension = structure[i]
            self._topology[i] = {}
            for node in dimension:
                self._topology[i][node - min_ids[i]] = tuple([relabelled_verts[vert] for vert in self.get_node(node).ordered_vertices()])
        return self._topology

    def get_sub_entities(self):
        # Construct sub entity dictionary, preserving vertex order.
        min_ids = self.get_starter_ids()
        sub_entities = {d: {e.id - min_ids[d]: [] for e in self.d_entities(d)} for d in range(self.get_spatial_dimension() + 1)}
        self.sub_entities = self._subentity_traversal(sub_entities, min_ids)
        return self.sub_entities

    def _subentity_traversal(self, sub_ents, min_ids):
        dim = self.get_spatial_dimension()
        self_id = self.id - min_ids[dim]

        if dim > 0:
            for p in self.ordered_vertices():
                p_id = p - min_ids[0]
                if (0, p_id) not in sub_ents[dim][self_id]:
                    sub_ents[dim][self_id] += [(0, p_id)]
                    sub_ents = self.get_node(p)._subentity_traversal(sub_ents, min_ids)
        if dim > 1:
            connections = [(c.point.id, c.point.group.identity) for c in self.connections]
            # if self.oriented:
            #     connections = self.permute_entities(self.oriented, dim - 1)
            # if self.dimension == 2:
            #     connections = [connections[1], connections[0], connections[2]]
            #     print([self.get_node(c[0]).id - min_ids[1] for c in connections])
            #     print([c.point.id - min_ids[1] for c in self.connections])
            for e, o in connections:
                p = self.get_node(e).orient(o)
                p_dim = p.get_spatial_dimension()
                p_id = p.id - min_ids[p_dim]
                if (p_dim, p_id) not in sub_ents[dim][self_id]:
                    sub_ents[dim][self_id] = sub_ents[dim][self_id] + [(p_dim, p_id)]
                    sub_ents = p._subentity_traversal(sub_ents, min_ids)

                    # All subentities of p are subentities of self.
                    for entry in sub_ents[p_dim][p_id]:
                        if entry not in sub_ents[dim][self_id]:
                            sub_ents[dim][self_id] = sub_ents[dim][self_id] + [entry]

        if (dim, self_id) not in sub_ents[dim][self_id]:
            sub_ents[dim][self_id] = sub_ents[dim][self_id] + [(dim, self_id)]

        return sub_ents

    @cache
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
        """Get all the d dimensional entities of the cell complex.

        :param: d: Dimension of required entities
        :param: get_class: (Optional) Returns Point classes

        Default return value is list of id numbers of the entities in the cell complex graph."""
        # if d == 0:
        #     return self.ordered_vertices(get_class)
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
        """
        Get vertices (0 dimensional entities) of the cell complex.

        :param: get_class: (Optional) Returns Point classes
        :param: return_coords: (Optional) returns coordinates of vertices

        Default return value is list of id numbers of the vertices in the cell complex graph.
        """
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
        """
        Get edges (1 dimensional entities) of the cell complex.

        :param: get_class: (Optional) Returns Point classes

        Default return value is list of id numbers of the 1 dimensional entities in the cell complex graph.
        """
        return self.d_entities(1, get_class)

    def permute_entities(self, g, d):
        """Where the symmetry ``g`` sends each ``d``-dimensional entity.

        Returns a list of tuples, each containing an entity id and symmetry group member.
        At index ``i``, ``self.permute_entities(g, d)[i]`` returns the entity id that
        ``d_entities(d)[i]`` is sent to under the permutation ``g``, and the orientation
        it is in after permutation.

        ``g`` is read through ``ordered_vertices``.

        """
        verts = self.ordered_vertices()
        entities = self.d_entities_ids(d)
        reordered = g.permute(verts)
        if d == 0:
            # listed in the order of d_entities, like the other dimensions
            image = dict(zip(verts, reordered))
            entity_group = self.d_entities(d)[0].group
            return [(image[v], entity_group.identity) for v in entities]

        entity_dict = {}
        reordered_entity_dict = {}

        for e in self.d_entities(d):
            entity_dict[e.id] = tuple(e.ordered_vertices())
            reordered_entity_dict[e.id] = tuple([reordered[verts.index(i)] for i in e.ordered_vertices()])

        reordered_entities = [tuple() for e in range(len(entities))]
        entity_group = self.d_entities(d)[0].group
        for ent in entities:
            for ent1 in entities:
                if set(entity_dict[ent]) == set(reordered_entity_dict[ent1]):
                    # local vertex i of ent1 is sent to local vertex o(i) of ent
                    o = [entity_dict[ent].index(v) for v in reordered_entity_dict[ent1]]
                    reordered_entities[entities.index(ent1)] = (ent, entity_group.get_member(Permutation(o)))

        return reordered_entities

    def orientation_key_to_member(self, key):
        """The symmetry of this cell that a mesh orientation key stands for.

        A mesh records how a cell sees one of its entities by the order of
        the entity's facets (its cone).
        The required orientation key is the :meth:`cone_orientation` of the
        facets, computed from the order this entity lists them (its connections), against
        the order the mesh lists them.

        Key 0 is the identity;
        Key k stands for the symmetry s whose images of the facets,
        s(f_0), s(f_1), ..., have cone orientation k against f_0, f_1, ...
        """
        if self.dimension == 0:
            if key != 0:
                raise ValueError(f"A vertex has only the orientation key 0, not {key}")
            return self.group.identity
        keys = self.__dict__.setdefault("_orientation_keys", {})
        if not keys:
            verts = self.ordered_vertices()
            cone = [frozenset(verts.index(v) for v in c.point.ordered_vertices()) for c in self.connections]
            for g in self.group.members():
                s = Permutation(g.perm, size=len(verts)).array_form
                image = [frozenset(s[i] for i in f) for f in cone]
                keys[self.cone_orientation(cone, image)] = g
        return keys[key]

    def cone_orientation(self, cone, image):
        """The orientation of the facets listed as ``image`` against ``cone``:
        the Lehmer code of ``image`` within ``cone``."""
        return orientation_value(cone, image)

    def extend_entity_symmetry(self, entity, o):
        """A symmetry of this cell that maps ``entity`` onto itself and
        induces the symmetry ``o`` on it."""
        d = entity.dimension
        if d == self.dimension:
            return self.group._match([o])[0]
        index = self.d_entities_ids(d).index(entity.id)
        for g in self.group._match(self.entity_stabiliser(entity)):
            if self.permute_entities(g, d)[index][1].perm == o.perm:
                return g
        raise ValueError(f"No symmetry of {self} induces {o} on {entity}")

    def entity_stabiliser(self, entity):
        """The members of this cell's symmetry group that map ``entity`` onto itself.

        ``entity`` is a sub-entity of this cell. Returns a permutation set on
        this cell; it is a subgroup of ``self.group``.
        """
        coords = np.array([np.array(self.get_node(v, return_coords=True), dtype=float).ravel()
                           for v in entity.ordered_vertices()])

        def maps_onto_itself(g):
            images = np.array([np.array(g(tuple(x)), dtype=float).ravel() for x in coords])
            return all(np.min(np.linalg.norm(coords - y, axis=1)) < 1e-10 for y in images)

        stab = [g.perm for g in self.group.members() if maps_onto_itself(g)]
        return fuse_groups.PermutationSetRepresentation(stab, cell=self)

    def basis_vectors(self, return_coords=True, entity=None, order=False, norm=True):
        if not entity:
            entity = self
        self_levels = [generation for generation in nx.topological_generations(self.G)]
        vertices = entity.ordered_vertices()
        if self.dimension == 0:
            # return [[]
            raise ValueError("Dimension 0 entities cannot have Basis Vectors")
        if self.oriented:
            # ordered_vertices() handles the orientation so we want to drop the orientation node
            top_level_node = self_levels[1][0]
        else:
            top_level_node = self_levels[0][0]
        v_0 = vertices[0]
        if return_coords:
            v_0_coords = self.attachment(top_level_node, v_0)()
        basis_vecs = []
        for v in vertices[1:]:
            if return_coords:
                v_coords = self.attachment(top_level_node, v)()
                if norm:
                    sub = normalise(np.subtract(v_coords, v_0_coords))
                else:
                    sub = np.subtract(v_coords, v_0_coords)
                if not hasattr(sub, "__iter__"):
                    basis_vecs.append((sub,))
                else:
                    basis_vecs.append(tuple(sub))
            else:
                basis_vecs.append((v, v_0))
        return basis_vecs

    def to_tikz(self, show=True, scale=3):
        tikz_commands = []
        if show:
            tikz_commands += ['\\begin{tikzpicture}']
        bs = '\\'
        arw = "\\tikz \\draw[-{Stealth[length=3mm, width=2mm]}] (-1pt,0) -- (1pt,0);"
        # only need to draw edges
        d_ents = self.d_entities(1)
        for e in d_ents:
            coords = "--".join([numpy_to_str_tuple(self.get_node(v, return_coords=True), scale=scale) for v in e.ordered_vertices()])
            tikz_commands += [f"{bs}draw[thick] {coords} node[midway,sloped, allow upside down] {{ {arw} }};"]
        if show:
            tikz_commands += ['\\end{tikzpicture}']
            return "\n".join(tikz_commands)
        return tikz_commands

    def generate_facet_parameterisation(self, facet_num):
        raise NotImplementedError("Facet Parameterisation can be expressed using polynomials")
        # facet = self.d_entities(self.dimension - 1)[facet_num]
        facet = self.get_node(facet_num)
        facet_dim = facet.dimension
        if facet_dim != self.dimension - 1:
            raise ValueError(f"Supplied node {facet_num} is not a facet")
        if facet_dim > 1:
            raise NotImplementedError("Facet parameterisation is not implemented for dimensions greater than 1")
        verts = facet.vertices()
        v_coords = np.array([self.get_node(v.id, return_coords=True) for v in verts])
        stacked = np.c_[np.ones((self.dimension,)), v_coords[:, 0].reshape(self.dimension, 1)]
        b = np.array([0, 1])
        coeffs = np.linalg.solve(stacked, b)
        symbol_names = ["x", "y", "z"]
        symbols = [1] + [sp.Symbol(symbol_names[d]) for d in range(facet_dim)]
        res = 0
        for d in range(facet_dim + 1):
            res += coeffs[d] * symbols[d]
        return res, symbols[1:]

    def plot(self, show=True, plain=False, ax=None, filename=None):
        """ for now into 2 dimensional space """
        if self.dimension == 3:
            self.plot3d(show, plain, ax, filename)
            return

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

        if show:
            plt.show()
        if filename:
            ax.figure.savefig(filename)
            plt.cla()

    def plot3d(self, show=True, plain=False, ax=None, filename=None):
        assert self.dimension == 3
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        xs = np.linspace(-1, 1, 20)

        top_level_node = self.d_entities_ids(self.graph_dim())[0]
        nodes = self.d_entities_ids(0)
        min_ids = self.get_starter_ids()
        for node in nodes:
            attach = self.attachment(top_level_node, node)
            plotted = attach()
            ax.scatter(plotted[0], plotted[1], plotted[2], color="black")
            if not plain:
                ax.text(plotted[0], plotted[1], plotted[2], node - min_ids[0], None)

        nodes = self.d_entities_ids(1)
        for node in nodes:
            attach = self.attachment(top_level_node, node)
            edgevals = np.array([attach(x) for x in xs])
            ax.plot3D(edgevals[:, 0], edgevals[:, 1], edgevals[:, 2], color="black")
            make_arrow_3d(ax, 0, attach)
            if not plain:
                plotted = attach(0)
                ax.text(plotted[0], plotted[1], plotted[2], node - min_ids[1], None)

        if show:
            plt.show()
        if filename:
            ax.figure.savefig(filename)
            plt.cla()

    def attachment(self, source, dst):
        if source == dst:
            # return x
            return lambda *x: x

        # Cached edge chain as it is fixed once cell is built.
        cache = self.__dict__.setdefault("_attachment_chains", {})
        if (source, dst) not in cache:
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

            cache[(source, dst)] = attachments[0]

        chain = cache[(source, dst)]
        return lambda *x: fold_reduce(chain, *x)

    def attachment_J(self, source, dst):
        attachment = self.attachment(source, dst)
        symbol_names = ["x", "y", "z"]
        symbols = []
        if self.dim_of_node(dst) == 0:
            return 1
        for i in range(self.dim_of_node(dst)):
            symbols += [sp.Symbol(symbol_names[i])]
        J = sp.Matrix(attachment(*symbols)).jacobian(sp.Matrix(symbols))
        return J

    def quadrature(self, degree):
        cache = self.__dict__.setdefault("_quadrature", {})
        if degree not in cache:
            Q = create_quadrature(self.to_fiat(), degree)
            cache[degree] = (Q.get_points(), Q.get_weights())
        pts, wts = cache[degree]
        return pts.copy(), wts.copy()

    def cartesian_to_barycentric(self, pts):
        verts = np.array(self.ordered_vertex_coords())
        v_0 = self.ordered_vertex_coords()[0]
        bvs = np.array(self.basis_vectors(norm=False))
        bary_coords = []
        for pt in pts:
            res = np.matmul(np.linalg.inv(bvs.T), np.array(pt - v_0))
            assert np.allclose(sum(bvs[i]*res[i] for i in range(len(bvs))), np.array(pt - v_0))
            bary_coords += [(1 - sum(res),) + tuple(res[i] for i in range(len(res)))]
            assert np.allclose(np.array(sum(bary_coords[-1][i]*verts[i] for i in range(len(verts)))), pt)
        return bary_coords

    def cell_attachment(self, dst):
        if not isinstance(dst, int):
            raise ValueError
        top_level_node = self.d_entities_ids(self.graph_dim())[0]
        return self.attachment(top_level_node, dst)

    def orient(self, o):
        """ Orientation node is always labelled with -1 """
        if o is None:
            return self
        if self.oriented:
            o = self.oriented * o
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

    def to_fiat(self, name=None, renumber=False):
        if len(self.vertices()) == self.dimension + 1:
            return CellComplexToFiatSimplex(self, name, renumber)
        if len(self.vertices()) == 2 ** self.dimension:
            raise NotImplementedError("Quadrilaterals and hexahedra must be built as products of intervals, "
                                      "with polygon(4) or TensorProductPoint(...).flatten()")
        raise NotImplementedError("Custom shape elements are not yet supported")

    def to_ufl(self, name=None):
        return CellComplexToUFL(self, name)

    def _to_dict(self):
        # think this is probably missing stuff
        # d, edges=[], vertex_num=None, oriented=False, group=None, edge_orientations=None, cell_id=None):
        o_dict = {"dim": self.dimension,
                  "edges": [c for c in self.connections],
                  "vertex_num": len(self.vertices()),
                  "oriented": self.oriented,
                  "group": self.group,
                  "id": self.id}
        return o_dict

    def dict_id(self):
        return "Cell"

    def _from_dict(o_dict):
        return Point(o_dict["dim"], o_dict["edges"], oriented=o_dict["oriented"], group=o_dict["group"], cell_id=o_dict["id"])


class HypercubePoint(Point):
    """
    A product of intervals: a quadrilateral, a hexahedron, or a face of a hexahedron.

    It is built by :meth:`TensorProductPoint.flatten`.

    :param: axes: the intervals it is the product of, in order
    """

    def __init__(self, d, edges, axes, oriented=False, group=None, cell_id=None):
        self.axes = tuple(axes)
        super().__init__(d, edges, oriented=oriented, group=group, cell_id=cell_id)

    def to_fiat(self, name=None, renumber=False):
        product = CellComplexToFiatTensorProduct(TensorProductPoint(*self.axes), name)
        return CellComplexToFiatHypercube(self, product)

    def d_entities(self, d, get_class=True):
        """As for a Point; the flattened cell itself can also be indexed by a
        tuple of factor dimensions."""
        if isinstance(d, tuple):
            entities = self.entities_by_mask[d]
            return entities if get_class else [e.id for e in entities]
        return super().d_entities(d, get_class)

    def axis_reordering(self, cone, image):
        """How the facets listed as ``image`` reorder the axes of ``cone``.

        Both list the facets in pairs, the two ends of each axis in turn, as
        this cell lists its connections. Returns, for each axis i of
        ``image``, the axis of ``cone`` its ends lie on, and whether they are
        swapped.
        """
        d = self.dimension
        pairs = [list(cone[2 * i:2 * i + 2]) for i in range(d)]
        axes, flips = [], []
        for i in range(d):
            ends = list(image[2 * i:2 * i + 2])
            j = next((j for j, pair in enumerate(pairs) if ends[0] in pair), None)
            if j is None or set(pairs[j]) != set(ends):
                raise ValueError(f"{image} does not reorder the axes of {cone}")
            axes.append(j)
            flips.append(pairs[j] != ends)
        return axes, flips

    def cone_orientation(self, cone, image):
        """The orientation of the facets listed as ``image`` against ``cone``:
        2**d * eo + io, where eo is the Lehmer code of the order the axes are
        taken in, and io has bit 2**(d - 1 - i) set when the ends of axis i
        are swapped."""
        axes, flips = self.axis_reordering(cone, image)
        d = self.dimension
        io = sum(2 ** (d - 1 - i) for i, flip in enumerate(flips) if flip)
        return 2 ** d * orientation_value(list(range(d)), axes) + io

    def _to_dict(self):
        o_dict = super()._to_dict()
        o_dict["axes"] = list(self.axes)
        return o_dict

    def dict_id(self):
        return "HypercubeCell"

    def _from_dict(o_dict):
        return HypercubePoint(o_dict["dim"], o_dict["edges"], o_dict["axes"], oriented=o_dict["oriented"],
                              group=o_dict["group"], cell_id=o_dict["id"])


@cache
def _component_evaluator(expr, nvals):
    """
    Compile one scalar attachment component for numeric evaluation.

    Only components fully determined by the given values are compiled;
    everything else defers to sympy_to_numpy, which decides how partially
    substituted expressions are represented. Cells are rebuilt frequently
    from the same attachment polynomials, so this is cached on the
    expression rather than on the edge holding it.
    """
    if len(expr.atoms(sp.Symbol)) == nvals:
        fn = sp.lambdify(_SYMBOLS[:nvals], expr, "math")
        return lambda *x: float(fn(*x))
    return lambda *x: sympy_to_numpy(expr, _SYMBOLS, x)


@cache
def _matrix_evaluator(expr, nvals):
    """
    Compile a matrix valued attachment for numeric evaluation.
    """
    if len(expr.atoms(sp.Symbol)) != nvals:
        return None
    fn = sp.lambdify(_SYMBOLS[:nvals], expr, "numpy")

    def evaluate(*x):
        res = np.array(fn(*x)).astype(np.float64)
        if len(res.shape) > 1:
            return res.squeeze()
        if len(res.shape) == 0:
            return res.item()
        return res

    return evaluate


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

    def _evaluator(self, nvals):
        """
        Compiled form of the attachment, or None if it must stay symbolic.

        Compilation depends only on the attachment, so it is cached per edge.
        """
        cache = self.__dict__.setdefault("_evaluator_cache", {})
        if nvals not in cache:
            if hasattr(self.attachment, '__iter__'):
                parts = [_component_evaluator(c, nvals) for c in self.attachment]

                def built(*x):
                    return tuple(p(*x) for p in parts)
            else:
                built = _matrix_evaluator(sp.ImmutableMatrix(self.attachment), nvals)
            cache[nvals] = built
        return cache[nvals]

    def __getstate__(self):
        # Compiled evaluators cannot be pickled, and are rebuilt on demand.
        state = self.__dict__.copy()
        state.pop("_evaluator_cache", None)
        return state

    def __call__(self, *x):
        if self.o:
            x = self.o(x)
        if self.attachment:
            if not any(isinstance(v, sp.Expr) for v in x):
                evaluate = self._evaluator(len(x))
                if evaluate is not None:
                    return evaluate(*x)
            syms = ["x", "y", "z"]
            if hasattr(self.attachment, '__iter__'):
                res = []
                for attach_comp in self.attachment:
                    if len(attach_comp.atoms(sp.Symbol)) <= len(x):
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


def _facet_images(entity, g):
    """Where the symmetry ``g`` of ``entity`` sends each of its facets."""
    if entity.dimension == 0:
        return {}
    verts = entity.ordered_vertices()
    s = Permutation(g.perm, size=len(verts)).array_form
    facets = [c.point for c in entity.connections]
    by_vertices = {frozenset(verts.index(v) for v in f.ordered_vertices()): f for f in facets}
    return {f: by_vertices[frozenset(s[verts.index(v)] for v in f.ordered_vertices())] for f in facets}


class TensorProductPoint():
    id_iter = itertools.count()

    def __init__(self, *factors):
        self.id = next(self.id_iter)
        self.A = factors[0]
        self.B = factors[1]
        self.factors = factors
        self.dimension = sum(f.dimension for f in factors)
        self.flat = False
        self.fiat_elem = None
        self.group = self.compute_cell_group()
        self.entities = {}

        for d in self.dimensions()[:-1]:
            self.entities[d] = [TensorProductPoint(*entities) for entities in product(*(f.d_entities(degree, True) for f, degree in zip(factors, d)))]
        self.entities[self.dim()] = [self]

    def ordered_vertices(self):
        return self.entities[0]

    def ordered_vertex_coords(self):
        return [sum(verts, ()) for verts in product(*(f.vertices(return_coords=True) for f in self.factors))]

    def component_orientations(self):
        """For each tuple of entity dimensions, the orientation key of the
        flattened entity that each tuple of the factor entities' keys stands for."""
        flat = self.flatten()
        self.component_os_to_os = {}
        for dim in self.to_fiat().get_topology():
            self.component_os_to_os[dim] = {}
            sub = tuple(f.d_entities(d)[0] for f, d in zip(self.factors, dim))
            entity = flat.entities_by_tuple[sub]
            cone = [child for _, _, child in self._facets(sub, dim)]
            for keys in product(*(range(e.group.size()) for e in sub)):
                images = [_facet_images(e, e.orientation_key_to_member(k)) for e, k in zip(sub, keys)]
                image = [tuple(m.get(c, c) for m, c in zip(images, child)) for child in cone]
                self.component_os_to_os[dim][keys] = entity.cone_orientation(cone, image) if cone else 0
        return self.component_os_to_os

    def compute_cell_group(self):
        """
        Systematically work out the symmetry group of the tensor product cell.
        """
        verts = self.vertices()
        group = list(product(*(f.group.members() for f in self.factors)))
        # group = [(g_a, g_b) for g_a in self.A.group.members() for g_b in self.B.group.members()]
        perms = []
        for gs in group:
            new_verts = list(product(*(g.permute(f.vertices()) for g, f in zip(gs, self.factors))))
            # new_verts = [(v_a, v_b) for v_a in g_a.permute(self.A.vertices()) for v_b in g_b.permute(self.B.vertices())]
            perm = [verts.index(v) for v in new_verts]
            perms += [fuse_groups.Permutation(perm)]

        grp = fuse_groups.PermutationSetRepresentation(perms).add_cell(self)
        return grp

    def get_starter_ids(self):
        # this doesn't actually make sense - remove when confirmed all changes to eliminate min ids from triple is done
        raise NotImplementedError
        a_starts = self.A.get_starter_ids()
        b_starts = self.B.get_starter_ids()
        ids = []
        for a, b in zip(a_starts, b_starts):
            ids += [max(a, b)]
        return ids

    def get_spatial_dimension(self):
        return self.dimension

    def get_sub_entities(self):
        return self.to_fiat().sub_entities

    def dim(self):
        return self.dimensions()[-1]

    def dimensions(self):
        return list(product(*(f.dimensions() for f in self.factors)))

    def d_entities(self, d, get_class=True):
        if isinstance(d, tuple):
            if get_class:
                return self.entities[d]
            return [e.id for e in self.entities[d]]
        raise NotImplementedError("Tensor Product point must be indexed by a tuple of dimensions")

    def vertices(self, get_class=True, return_coords=False):
        # TODO maybe refactor with get_node
        if return_coords:
            # a_verts = self.A.vertices(return_coords=return_coords)
            # b_verts = self.B.vertices(return_coords=return_coords)
            # return [a + b for a in a_verts for b in b_verts]
            return [sum(verts, ()) for verts in product(*(f.vertices(return_coords=True) for f in self.factors))]
        # return [(a, b) for a in self.A.vertices() for b in self.B.vertices()]
        return list(product(*(f.vertices() for f in self.factors)))

    def __repr__(self):
        return "*".join([str(f) for f in self.factors])

    def to_ufl(self, name=None):
        return TensorProductCell(*[f.to_ufl() for f in self.factors])

    def to_fiat(self, name=None):
        if self.fiat_elem is None:
            self.fiat_elem = CellComplexToFiatTensorProduct(self, name)
        return self.fiat_elem

    def flatten(self):
        """The product as a single cell, a :class:`HypercubePoint`.

        Every entity is the product of one entity of each factor.
        """
        if getattr(self, "_flat", None) is not None:
            # built once, so that everything using this product shares one flattened cell
            return self._flat
        if not all(f.dimension == 1 or isinstance(f, HypercubePoint) for f in self.factors):
            raise NotImplementedError("Only products of intervals and of flattened products can be flattened")
        masks_by_dim = defaultdict(list)
        for mask in itertools.product(*(range(f.dimension + 1) for f in self.factors)):
            masks_by_dim[sum(mask)].append(mask)

        points = {}
        by_mask = {}
        for d in sorted(masks_by_dim):
            for mask in masks_by_dim[d]:
                by_mask[mask] = list(itertools.product(*(f.d_entities(k) for f, k in zip(self.factors, mask))))
                for entity in by_mask[mask]:
                    if d == 0:
                        points[entity] = Point(0)
                        continue
                    edges = []
                    for axis, factor_edge, child in self._facets(entity, mask):
                        attachment = self._attachment(axis, factor_edge, mask, tuple(e.dimension for e in child))
                        edges.append(Edge(points[child], attachment, factor_edge.o))
                    if d == 1:
                        points[entity] = Point(1, edges)
                    else:
                        axes = sum((e.axes if isinstance(e, HypercubePoint) else (e,) for e in entity if e.dimension > 0), ())
                        points[entity] = HypercubePoint(d, edges, axes)
        top = points[tuple(self.factors)]
        top.factors = self.factors
        top.entities_by_tuple = points
        top.entities_by_mask = {mask: [points[e] for e in ents] for mask, ents in by_mask.items()}
        self._flat = top
        return top

    @staticmethod
    def _facets(entity, mask):
        """The codimension one facets of a product entity, as
        (axis, the factor's edge to its facet, the facet)."""
        for axis, k in enumerate(mask):
            if k == 0:
                continue
            for factor_edge in entity[axis].connections:
                child = list(entity)
                child[axis] = factor_edge.point
                yield axis, factor_edge, tuple(child)

    @staticmethod
    def _attachment(axis, factor_edge, mask, child_mask):
        """The attachment of a facet into its product entity, as a tuple of
        sympy expressions: the factor's attachment on ``axis``, the identity
        on the other axes.

        For example, for the facet (0, 1) of (1, 1) on axis 0 with the factor
        attachment (-1,), the result is (-1, x)."""
        child_syms = _SYMBOLS[:sum(child_mask)]
        result = tuple()
        offset = 0
        for i, k in enumerate(child_mask):
            if i == axis:
                local = _SYMBOLS[:k]
                subs = {old: new for old, new in zip(local, child_syms[offset:offset + k])}
                result += tuple(sp.sympify(c).subs(subs) for c in factor_edge.attachment)
            else:
                result += tuple(child_syms[offset:offset + k])
            offset += k
        return result


class CellComplexToFiatSimplex(Simplex):
    """
    Convert cell complex to fiat

    :param: cell: a fuse cell complex

    Currently assumes simplex.
    """

    def __init__(self, cell, name=None, renumber=False):
        self.fe_cell = cell
        if name is None:
            name = "FuseCell"
        self.name = name

        # verts = [cell.get_node(v, return_coords=True) for v in cell.ordered_vertices()]
        verts = cell.vertices(return_coords=True)
        topology = cell.get_topology(renumber)
        shape = cell.get_shape()
        sub_ents = cell.get_sub_entities()
        super(CellComplexToFiatSimplex, self).__init__(shape, verts, topology, sub_ents)
        # if len(verts) == 4:
        #     breakpoint()

    def cellname(self):
        return self.name

    def construct_subelement(self, dimension, e_id=0, o=None):
        """Constructs the reference element of a cell
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        :arg e_id: subentity id, default 0, (integer)
        :arg o: orientation of subentity, default None (GroupMemberRep)
        """
        if o:
            return self.fe_cell.d_entities(dimension)[e_id].orient(o).to_fiat(renumber=True)
        else:
            return self.fe_cell.d_entities(dimension)[e_id].to_fiat(renumber=True)

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
        fiat_factors = [f.to_fiat() for f in cell.factors]
        if name is None:
            name = " * ".join([s.name for s in fiat_factors])
        self.name = name
        super(CellComplexToFiatTensorProduct, self).__init__(*fiat_factors)

    def cellname(self):
        return self.name

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        return CellComplexToFiatTensorProduct(TensorProductPoint(*[c.construct_subelement(d).fe_cell for c, d in zip(self.cells, dimension)]))

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
        self.name = product.name
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
        return self.fe_cell.d_entities(dimension)[0].to_fiat()

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
        return self.cell_complex.to_fiat(name=self.cellname)

    def __repr__(self):
        return "FUSE_" + super(CellComplexToUFL, self).__repr__()

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
        return line().to_ufl(name)
    elif name == "triangle":
        return polygon(3).to_ufl(name)
        # return ufc_triangle().to_ufl(name)
    elif name == "quadrilateral":
        return polygon(4).to_ufl(name)
    elif name == "tetrahedron":
        # return ufc_tetrahedron().to_ufl(name)
        return make_tetrahedron().to_ufl(name)
    elif name == "hexahedron":
        return TensorProductPoint(line(), line(), line()).flatten().to_ufl(name)
    elif "*" in name:
        components = [constructCellComplex(c.strip()).cell_complex for c in name.split("*")]
        return TensorProductPoint(*components).to_ufl(name)
    else:
        raise TypeError("Cell complex construction undefined for {}".format(str(name)))


def compare_topologies(base, new):
    """Compute orientations of sub entities against a base topology

       base topology is assumed to follow the FIAT numbering convention,
       ie edges are ordered from lower to higher
       """
    if list(base.keys()) != list(new.keys()):
        raise ValueError("Topologies of different sizes cannot be compared")
    orientations = []
    for dim in sorted(base.keys()):
        for entity in sorted(base[dim].keys()):
            assert entity in new[dim].keys()
            base_array = list(base[dim][entity])
            new_array = list(new[dim][entity])
            if sorted(base_array) == sorted(new_array):
                orientations += [orientation_value(base_array, new_array)]
            else:
                # numbering does not match - renumber
                # base is treated as ordered list
                # new is renumbered by sorting the values
                base_numbering = {base_array[i]: i for i in range(len(base_array))}
                sorted_new = sorted(new_array)
                new_numbering = {sorted_new[i]: i for i in range(len(new_array))}
                base_renumbered = [base_numbering[b] for b in base_array]
                new_renumbered = [new_numbering[n] for n in new_array]
                orientations += [orientation_value(base_renumbered, new_renumbered)]

    return orientations
