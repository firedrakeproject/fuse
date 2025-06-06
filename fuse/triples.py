from fuse.cells import Point, TensorProductPoint
from fuse.spaces.element_sobolev_spaces import ElementSobolevSpace
from fuse.dof import DeltaPairing, L2Pairing, FuseFunction, PointKernel
from fuse.traces import Trace
from fuse.groups import perm_matrix_to_perm_array, perm_list_to_matrix
from fuse.utils import numpy_to_str_tuple
from FIAT.dual_set import DualSet
from FIAT.finite_element import CiarletElement
# from FIAT.reference_element import ufc_cell
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import inspect
from finat.ufl import FuseElement
import warnings
import numpy as np
import scipy


class ElementTriple():
    """
    Class to represent the three core parts of the element

    :param: cell: CellComplex
    :param: spaces: Triple of spaces: (PolynomialSpace, SobolovSpace, InterpolationSpace)
    :param: dof_gen: Generator Triple to generate the degrees of freedom.
    """

    def __init__(self, cell, spaces, dof_gen):
        assert isinstance(cell, Point) or isinstance(cell, TensorProductPoint)
        if isinstance(dof_gen, DOFGenerator):
            dof_gen = [dof_gen]
        for d in dof_gen:
            assert isinstance(d, DOFGenerator)
            d.add_cell(cell)

        self.cell = cell
        cell_spaces = []
        for space in spaces:
            # TODO: Fix this to a more sensible condition when all spaces
            # implemented
            if inspect.isclass(space) and issubclass(space, ElementSobolevSpace):
                cell_spaces.append(space(cell))
            else:
                cell_spaces.append(space)
        self.spaces = tuple(cell_spaces)
        self.DOFGenerator = dof_gen
        self.flat = False

    def __repr__(self):
        return "FuseTriple(%s, %s, (%s, %s, %s), %s)" % (
               repr(self.DOFGenerator), repr(self.cell), repr(self.spaces[0]), repr(self.spaces[1]), repr(self.spaces[2]), "X")

    def generate(self):
        res = []
        id_counter = 0
        for dof_gen in self.DOFGenerator:
            generated = dof_gen.generate(self.cell, self.spaces[1], id_counter)
            res.extend(generated)
            id_counter += len(generated)
        return res

    def __iter__(self):
        yield self.cell
        yield self.spaces
        yield self.DOFGenerator

    def num_dofs(self):
        return sum([dof_gen.num_dofs() for dof_gen in self.DOFGenerator])

    def degree(self):
        # TODO this isn't really correct
        return self.spaces[0].degree()

    def get_dof_info(self, dof, tikz=True):
        colours = {False: {0: "b", 1: "r", 2: "g", 3: "b"},
                   True: {0: "blue", 1: "red", 2: "green", 3: "black"}}
        if dof.trace_entity.dimension == 0:
            center = self.cell.cell_attachment(dof.trace_entity.id)()
        elif dof.trace_entity.dimension == 1:
            center = self.cell.cell_attachment(dof.trace_entity.id)(0)
        elif dof.trace_entity.dimension == 2:
            center = self.cell.cell_attachment(dof.trace_entity.id)(0, 0)
        else:
            center = list(sum(np.array(self.cell.vertices(return_coords=True))))

        return center, colours[tikz][dof.trace_entity.dimension]

    def get_value_shape(self):
        # TODO Shape should be specificed somewhere else probably
        if self.spaces[0].set_shape:
            return (self.cell.get_spatial_dimension(),)
        else:
            return ()

    def to_ufl(self):
        return FuseElement(self)

    def to_fiat(self):
        ref_el = self.cell.to_fiat()
        dofs = self.generate()
        degree = self.spaces[0].degree()
        entity_ids = {}
        entity_perms = {}
        nodes = []
        top = ref_el.get_topology()
        min_ids = self.cell.get_starter_ids()
        poly_set = self.spaces[0].to_ON_polynomial_set(ref_el)

        for dim in sorted(top):
            entity_ids[dim] = {i: [] for i in top[dim]}
            entity_perms[dim] = {}

        entities = [(dim, entity) for dim in sorted(top) for entity in sorted(top[dim])]
        # if sort_entities:
        #     # sort the entities by support vertex ids
        # support = [sorted(top[dim][entity]) for dim, entity in entities]
        # entities = [entity for verts, entity in sorted(zip(support, entities))]
        counter = 0
        for entity in entities:
            dim = entity[0]
            for i in range(len(dofs)):
                if entity[1] == dofs[i].trace_entity.id - min_ids[dim]:
                    entity_ids[dim][dofs[i].trace_entity.id - min_ids[dim]].append(counter)
                    nodes.append(dofs[i].convert_to_fiat(ref_el, degree))
                    counter += 1
        # entity_orientations = compare_topologies(ufc_cell(self.cell.to_ufl().cellname()).get_topology(), self.cell.get_topology()
        # self.matrices_by_entity = self.make_entity_dense_matrices(ref_el, entity_ids, nodes, poly_set)
        mat_perms, entity_perms, pure_perm = self.make_dof_perms(ref_el, entity_ids, nodes, poly_set)
        self.matrices = mat_perms
        form_degree = 1 if self.spaces[0].set_shape else 0

        # TODO: Change this when Dense case in Firedrake
        if entity_perms is not None:
            dual = DualSet(nodes, ref_el, entity_ids, entity_perms)
        else:
            dual = DualSet(nodes, ref_el, entity_ids)
        return CiarletElement(poly_set, dual, degree, form_degree)

    def to_tikz(self, show=True, scale=3):
        """Generates tikz code for the element diagram

        Requires the \\usetikzlibrary{arrows.meta} library
        """
        tikz_commands = []
        if show:
            tikz_commands += ['\\begin{tikzpicture}']
        tikz_commands += self.cell.to_tikz(show=False, scale=scale)

        dofs = self.generate()
        identity = FuseFunction(lambda *x: x)
        for dof in dofs:
            center, color = self.get_dof_info(dof)
            if isinstance(dof.pairing, DeltaPairing):
                coord = dof.eval(identity, pullback=False)
                if isinstance(dof.target_space, Trace):
                    tikz_commands += [dof.target_space.to_tikz(coord, dof.trace_entity, dof.g, scale, color)]
                else:
                    tikz_commands += [f"\\filldraw[{color}] {numpy_to_str_tuple(coord, scale)} circle (2pt) node[anchor = south] {{}};"]
            elif isinstance(dof.pairing, L2Pairing):
                coord = center
                tikz_commands += [dof.target_space.to_tikz(coord, dof.trace_entity, dof.g, scale, color)]
        if show:
            tikz_commands += ['\\end{tikzpicture}']
            return "\n".join(tikz_commands)
        return tikz_commands

    def plot(self, filename="temp.png"):
        """
        Generates Matplotlib code for the element diagrams."""
        dofs = self.generate()
        identity = FuseFunction(lambda *x: x)

        if self.cell.dimension == 0:
            raise ValueError(" Dimension 0 cells cannot be plotted")

        if self.cell.dimension < 3:
            fig = plt.figure()
            ax = plt.gca()
            self.cell.plot(show=False, plain=True, ax=ax)
            for dof in dofs:
                center, color = self.get_dof_info(dof)
                if isinstance(dof.pairing, DeltaPairing) and isinstance(dof.kernel, PointKernel):
                    coord = dof.eval(identity, pullback=False)
                elif isinstance(dof.pairing, L2Pairing):
                    coord = center
                if len(coord) == 1:
                    coord = (coord[0], 0)
                if isinstance(dof.target_space, Trace):
                    dof.target_space.plot(ax, coord, dof.trace_entity, dof.g, color=color)
                else:
                    ax.scatter(*coord, color=color)
                ax.text(*coord, dof.id)
            plt.axis('off')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if filename:
                fig.savefig(filename)
            else:
                plt.show()
        elif self.cell.dimension == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            self.cell.plot3d(show=False, ax=ax)
            for dof in dofs:
                center, color = self.get_dof_info(dof)
                if center is None:
                    center = [0, 0, 0]
                if isinstance(dof.pairing, DeltaPairing):
                    coord = dof.eval(identity, pullback=False)
                    if isinstance(dof.target_space, Trace):
                        dof.target_space.plot(ax, coord, dof.trace_entity, dof.g, color=color)
                    else:
                        ax.scatter(*coord, color=color)
                elif isinstance(dof.pairing, L2Pairing):
                    coord = center
                    dof.target_space.plot(ax, center, dof.trace_entity, dof.g, color=color, length=0.2)
                ax.text(*coord, dof.id)
            plt.axis('off')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if filename:
                fig.savefig(filename)
            else:
                plt.show()
        else:
            raise ValueError("Plotting not supported in this dimension")

    def compute_dense_matrix(self, ref_el, entity_ids, nodes, poly_set):
        dual = DualSet(nodes, ref_el, entity_ids)

        old_coeffs = poly_set.get_coeffs()
        dualmat = dual.to_riesz(poly_set)

        shp = dualmat.shape
        A = dualmat.reshape((shp[0], -1))
        B = old_coeffs.reshape((shp[0], -1))
        V = np.dot(A, np.transpose(B))

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                new_coeffs_flat = scipy.linalg.solve(V, B, transposed=True)
            except (scipy.linalg.LinAlgWarning, scipy.linalg.LinAlgError):
                raise np.linalg.LinAlgError("Singular Vandermonde matrix")
        return A, new_coeffs_flat

    def make_entity_dense_matrices(self, ref_el, entity_ids, nodes, poly_set):
        degree = self.spaces[0].degree()
        min_ids = self.cell.get_starter_ids()
        nodes = [d.convert_to_fiat(ref_el, degree) for d in self.generate()]
        sub_ents = []
        res_dict = {}
        for d in range(0, self.cell.dim() + 1):
            res_dict[d] = {}
            sub_ents += self.cell.d_entities(d)
        for e in sub_ents:
            dim = e.dim()
            e_id = e.id - min_ids[dim]
            res_dict[dim][e_id] = {}
            dof_ids = [d.id for d in self.generate() if d.trace_entity == e]
            res_dict[dim][e_id][0] = np.eye(len(dof_ids))
            original_V, original_basis = self.compute_dense_matrix(ref_el, entity_ids, nodes, poly_set)

            for g in self.cell.group.members():
                permuted_e, permuted_g = self.cell.permute_entities(g, dim)[e_id]
                val = permuted_g.numeric_rep()
                if val not in res_dict[dim][e_id].keys() and permuted_e == e.id:
                    if len(dof_ids) == 0:
                        res_dict[dim][e_id][val] = []
                    elif g.perm.is_Identity:
                        res_dict[dim][e_id][val] = np.eye(len(dof_ids))
                    else:
                        new_nodes = [d(g).convert_to_fiat(ref_el, degree) if d.trace_entity == e else d.convert_to_fiat(ref_el, degree) for d in self.generate()]
                        transformed_V, transformed_basis = self.compute_dense_matrix(ref_el, entity_ids, new_nodes, poly_set)
                        res_dict[dim][e_id][val] = np.matmul(transformed_basis, original_V.T)[np.ix_(dof_ids, dof_ids)]
        return res_dict

    def make_overall_dense_matrices(self, ref_el, entity_ids, nodes, poly_set):
        min_ids = self.cell.get_starter_ids()
        dim = self.cell.dim()
        e = self.cell
        e_id = e.id - min_ids[dim]
        res_dict = {dim: {e_id: {}}}
        degree = self.spaces[0].degree()
        original_V, original_basis = self.compute_dense_matrix(ref_el, entity_ids, nodes, poly_set)
        for g in self.cell.group.members():
            val = g.numeric_rep()
            if g.perm.is_Identity:
                res_dict[dim][e_id][val] = np.eye(len(nodes))
            else:
                new_nodes = [d(g).convert_to_fiat(ref_el, degree) for d in self.generate()]
                transformed_V, transformed_basis = self.compute_dense_matrix(ref_el, entity_ids, new_nodes, poly_set)
                res_dict[dim][e_id][val] = np.matmul(transformed_basis, original_V.T)
        return res_dict

    def _entity_associations(self, dofs):
        min_ids = self.cell.get_starter_ids()
        entity_associations = {dim: {e.id - min_ids[dim]: {} for e in self.cell.d_entities(dim)}
                               for dim in range(self.cell.dim() + 1)}
        cell_dim = self.cell.dim()
        cell_dict = entity_associations[cell_dim][0]
        pure_perm = True
        sub_pure_perm = True

        # construct mapping of entities to the dof generators and the dofs they generate
        for d in dofs:
            sub_dim = d.trace_entity.dim()
            sub_dict = entity_associations[sub_dim][d.trace_entity.id - min_ids[sub_dim]]
            for dim in set([sub_dim, cell_dim]):
                dof_gen = str(d.generation[dim])

                if not len(d.generation[dim].g2.members()) == 1:
                    if dim == cell_dim:
                        pure_perm = False
                    else:
                        sub_pure_perm = False

                if dof_gen in sub_dict.keys() and (dim < cell_dim or not d.immersed):
                    sub_dict[dof_gen] += [d]
                elif dim < cell_dim or not d.immersed:
                    sub_dict[dof_gen] = [d]

                if dof_gen in cell_dict.keys() and dim == cell_dim and d.immersed:
                    cell_dict[dof_gen] += [d]
                elif dim == cell_dim and d.immersed:
                    cell_dict[dof_gen] = [d]
        return entity_associations, pure_perm, sub_pure_perm

    def _initialise_entity_dicts(self, dofs):
        min_ids = self.cell.get_starter_ids()
        dof_id_mat = np.eye(len(dofs))
        oriented_mats_by_entity = {}
        flat_by_entity = {}
        for dim in range(self.cell.dim() + 1):
            oriented_mats_by_entity[dim] = {}
            flat_by_entity[dim] = {}
            ents = self.cell.d_entities(dim)
            for e in ents:
                e_id = e.id - min_ids[dim]
                members = e.group.members()
                oriented_mats_by_entity[dim][e_id] = {}
                flat_by_entity[dim][e_id] = {}
                for g in members:
                    val = g.numeric_rep()
                    oriented_mats_by_entity[dim][e_id][val] = dof_id_mat.copy()
                    flat_by_entity[dim][e_id][val] = []
        return oriented_mats_by_entity, flat_by_entity

    def make_dof_perms(self, ref_el, entity_ids, nodes, poly_set):
        dofs = self.generate()
        min_ids = self.cell.get_starter_ids()
        entity_associations, pure_perm, sub_pure_perm = self._entity_associations(dofs)
        if pure_perm is False:
            # TODO think about where this call goes
            return self.make_overall_dense_matrices(ref_el, entity_ids, nodes, poly_set), None, pure_perm

        oriented_mats_by_entity, flat_by_entity = self._initialise_entity_dicts(dofs)
        # for each entity, look up generation on that entity and permute the
        # dof mapping according to the generation
        for dim in range(self.cell.dim() + 1):
            ents = self.cell.d_entities(dim)
            for e in ents:
                e_id = e.id - min_ids[dim]
                members = e.group.members()
                for g in members:
                    val = g.numeric_rep()
                    total_ent_dof_ids = []
                    for dof_gen in entity_associations[dim][e_id].keys():
                        ent_dofs = entity_associations[dim][e_id][dof_gen]
                        ent_dofs_ids = np.array([ed.id for ed in ent_dofs], dtype=int)
                        # (dof_gen, ent_dofs)
                        total_ent_dof_ids += [ed.id for ed in ent_dofs if ed.id not in total_ent_dof_ids]
                        dof_gen_class = ent_dofs[0].generation
                        if not len(dof_gen_class[dim].g2.members()) == 1:
                            # if DOFs on entity are not perms, get the matrix
                            oriented_mats_by_entity[dim][e_id][val][np.ix_(ent_dofs_ids, ent_dofs_ids)] = self.matrices_by_entity[dim][e_id][val]
                        elif g.perm.is_Identity or (pure_perm and len(ent_dofs_ids) == 1):
                            oriented_mats_by_entity[dim][e_id][val][np.ix_(ent_dofs_ids, ent_dofs_ids)] = np.eye(len(ent_dofs_ids))
                        elif g in dof_gen_class[dim].g1.members() and dim < self.cell.dim():
                            # Permutation of DOF on the entity they are defined on
                            sub_mat = g.matrix_form()
                            oriented_mats_by_entity[dim][e_id][val][np.ix_(ent_dofs_ids, ent_dofs_ids)] = sub_mat.copy()
                        elif len(dof_gen_class.keys()) == 1 and dim == self.cell.dim():
                            # case for dofs defined on the cell and not immersed
                            sub_mat = g.matrix_form()
                            oriented_mats_by_entity[dim][e_id][val][np.ix_(ent_dofs_ids, ent_dofs_ids)] = sub_mat.copy()
                        else:
                            # TODO what if an orientation is not in G1
                            pass

                        if len(dof_gen_class.keys()) == 2 and dim == self.cell.dim():
                            # Handle immersion - can only happen once so number of keys is max 2
                            dimensions = list(dof_gen_class.keys())
                            dimensions.remove(dim)
                            immersed_dim = dimensions[0]
                            identity = [d_ent.id for d_ent in self.cell.d_entities(immersed_dim)]
                            permuted_ents = self.cell.permute_entities(g, immersed_dim)
                            # g_sub_mat = g.matrix_form()
                            g_sub_mat = perm_list_to_matrix(identity, [sub_e for sub_e, _ in permuted_ents])
                            for sub_e, sub_g in permuted_ents:
                                sub_e = self.cell.get_node(sub_e)
                                sub_e_id = sub_e.id - min_ids[sub_e.dim()]
                                sub_ent_ids = []
                                for (k, v) in entity_associations[immersed_dim][sub_e_id].items():
                                    sub_ent_ids += [e.id for e in v]
                                sub_mat = oriented_mats_by_entity[immersed_dim][sub_e_id][sub_g.numeric_rep()][np.ix_(sub_ent_ids, sub_ent_ids)]

                                expanded = np.kron(g_sub_mat, sub_mat)
                                # potentially permute the dof ids instead
                                oriented_mats_by_entity[self.cell.dim()][0][val][np.ix_(ent_dofs_ids, ent_dofs_ids)] = expanded.copy()
                    if pure_perm and sub_pure_perm:
                        flat_by_entity[dim][e_id][val] = perm_matrix_to_perm_array(oriented_mats_by_entity[dim][e_id][val][np.ix_(total_ent_dof_ids, total_ent_dof_ids)])

        # remove immersed DOFs from flat form
        oriented_mats_overall = oriented_mats_by_entity[dim][0]
        for val, mat in oriented_mats_overall.items():
            cell_dofs = entity_ids[dim][0]
            flat_by_entity[dim][e_id][val] = perm_matrix_to_perm_array(mat[np.ix_(cell_dofs, cell_dofs)])

        if pure_perm and sub_pure_perm:
            return oriented_mats_by_entity, flat_by_entity, True
        return oriented_mats_by_entity, None, False

    def _to_dict(self):
        o_dict = {"cell": self.cell, "spaces": self.spaces, "dofs": self.DOFGenerator}
        return o_dict

    def dict_id(self):
        return "Triple"

    def _from_dict(o_dict):
        return ElementTriple(o_dict["cell"], o_dict["spaces"], o_dict["dofs"])


class DOFGenerator():

    def __init__(self, generator_funcs, gen_group, trans_group):
        # assert isinstance(G_1, Group)
        # assert isinstance(G_2, Group)
        self.x = generator_funcs
        self.g1 = gen_group
        self.g2 = trans_group
        self.dof_numbers = None
        self.ls = None

    def __iter__(self):
        yield self.x
        yield self.g1
        yield self.g2

    def add_cell(self, cell):
        self.g1 = self.g1.add_cell(cell)
        self.g2 = self.g2.add_cell(cell)

    def num_dofs(self):
        if self.dof_numbers is None:
            raise ValueError("DOFs not generated yet")
        return self.dof_numbers

    def generate(self, cell, space, id_counter):
        if self.ls is None:
            self.ls = []
            for l_g in self.x:
                i = 0
                for g in self.g1.members():
                    generated = l_g(g)
                    if not isinstance(generated, list):
                        generated = [generated]
                    for dof in generated:
                        dof.add_context(self, cell, space, g, id_counter, i)
                        id_counter += 1
                        i += 1
                    self.ls.extend(generated)
            self.dof_numbers = len(self.ls)
            self.dof_ids = [dof.id for dof in self.ls]
        return self.ls

    def make_entity_ids(self):
        dofs = self.ls
        entity_ids = {}
        min_ids = dofs[0].cell.get_starter_ids()

        top = dofs[0].cell.get_topology()

        for dim in sorted(top):
            entity_ids[dim] = {i: [] for i in top[dim]}

        for i in range(len(dofs)):
            entity = dofs[i].trace_entity
            dim = entity.dim()
            entity_ids[dim][entity.id - min_ids[dim]].append(i)
        return entity_ids

    def __repr__(self):
        repr_str = "DOFGen("
        for x_elem in self.x:
            repr_str += "g(" + str(x_elem) + ")"
        repr_str += str(self.g1) + str(self.g2) + ")"
        return repr_str

    def _to_dict(self):
        o_dict = {"x": self.x, "g1": self.g1, "g2": self.g2}
        return o_dict

    def dict_id(self):
        return "DOFGen"

    def _from_dict(obj_dict):
        return DOFGenerator(obj_dict["x"], obj_dict["g1"], obj_dict["g2"])


class ImmersedDOFs():

    def __init__(self, target_cell, triple, trace, start_node=0):
        self.target_cell = target_cell
        self.triple = triple
        self.C, self.V, self.E = triple
        self.trace = trace(target_cell)
        self.start_node = start_node

    def __call__(self, g):
        target_node, o = self.target_cell.permute_entities(g, self.C.dim())[self.start_node]
        if self.C.dim() > 0 and o != o.group.identity:
            raise ValueError("Not matching orientation - groups incorrect")
        attachment = self.target_cell.cell_attachment(target_node)
        new_dofs = []

        def oriented_attachment(*x):
            return attachment(*o(x))

        for generated_dof in self.triple.generate():
            new_dof = generated_dof.immerse(self.target_cell.get_node(target_node),
                                            oriented_attachment,
                                            self.trace, g, self.triple)
            new_dofs.append(new_dof)
        return new_dofs

    def __repr__(self):
        repr_str = ""
        for dof_gen in self.E:
            repr_str += "Im_" + str(self.trace) + "_" + str(self.target_cell) + "(" + str(dof_gen) + ")"
        return repr_str

    def _to_dict(self):
        o_dict = {"target_cell": self.target_cell, "triple": self.triple, "trace": self.trace}
        return o_dict

    def dict_id(self):
        return "ImmersedDOF"

    def _from_dict(obj_dict):
        return ImmersedDOFs(obj_dict["target_cell"], obj_dict["triple"], obj_dict["trace"])


def immerse(target_cell, triple, target_space, node=0):
    return ImmersedDOFs(target_cell, triple, target_space, node)
