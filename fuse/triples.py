from fuse.cells import Point, TensorProductPoint
from fuse.dof import DeltaPairing, L2Pairing, FuseFunction, PointKernel
from fuse.traces import TrH1, TrHCurl, TrHDiv
from fuse.utils import numpy_to_str_tuple
from fuse.spaces.polynomial_spaces import riesz_matrices
from FIAT.dual_set import DualSet
from FIAT.finite_element import CiarletElement
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from finat.ufl import FuseElement
import warnings
import numpy as np
import scipy
from functools import cache


def is_permutation_matrix(M):
    """Whether ``M`` is an unsigned permutation matrix."""
    R = np.round(M)
    return np.allclose(M, R) and set(np.unique(R)) <= {0, 1} and np.allclose(R.sum(axis=0), 1) and np.allclose(R.sum(axis=1), 1)


def plot_dof(ax, dof, coord, **kwargs):
    """Draw a DOF at ``coord``: an arrow along the direction its kernel is
    lifted (a tangent or a normal), the derivative trace's marker, or a point."""
    if dof.target_space is not None:
        dof.target_space.plot(ax, coord, dof.cell_defined_on, **{k: v for k, v in kwargs.items() if k != "length"})
        return
    direction = dof.lift_direction()
    if direction is None:
        ax.scatter(*coord, color=kwargs.get("color"))
    else:
        ax.quiver(*coord, *direction, **kwargs)


def dof_to_tikz(dof, coord, scale, color="black"):
    """The tikz command drawing a DOF, as :func:`plot_dof` does."""
    if dof.target_space is not None:
        return dof.target_space.to_tikz(coord, dof.cell_defined_on, scale, color)
    direction = dof.lift_direction()
    if direction is None:
        return f"\\filldraw[{color}] {numpy_to_str_tuple(coord, scale)} circle (2pt) node[anchor = south] {{}};"
    end_point = [coord[i] + 0.25*direction[i] for i in range(len(coord))]
    arw = "-{Stealth[length=3mm, width=2mm]}"
    return f"\\draw[thick, {color}, {arw}] {numpy_to_str_tuple(coord, scale)} -- {numpy_to_str_tuple(end_point, scale)};"


class ElementTriple():
    """
    Class to represent the three core parts of the element

    :param: cell: CellComplex
    :param: spaces: Triple of spaces: (PolynomialSpace, InterpolationSpace, Pullback)
    :param: dof_gen: Generator Triple to generate the degrees of freedom.
    """

    def __init__(self, cell, spaces, dof_gen, perm=True):
        assert isinstance(cell, Point) or isinstance(cell, TensorProductPoint)
        if isinstance(dof_gen, DOFGenerator):
            dof_gen = [dof_gen]
        for d in dof_gen:
            assert isinstance(d, DOFGenerator)
            d.add_context(cell, spaces[0])

        self.cell = cell
        self.spaces = tuple(spaces)
        self.DOFGenerator = dof_gen
        self.flat = False

        self.ref_el = None

        self.dofs = None
        self.dofs = self.generate()
        self.perm = perm

    def setup_ids_and_nodes(self):
        dofs = self.generate()
        degree = self.spaces[0].degree() + 1
        value_shape = self.get_value_shape()
        top = self.ref_el.get_topology()
        min_ids = self.cell.get_starter_ids()
        entity_ids = {}
        nodes = []

        for dim in sorted(top):
            entity_ids[dim] = {i: [] for i in top[dim]}

        self.dof_id_to_fiat_id = {}
        entities = [(dim, entity) for dim in sorted(top) for entity in sorted(top[dim])]
        counter = 0
        for entity in entities:
            dim = entity[0]
            for i in range(len(dofs)):
                if entity[1] == dofs[i].cell_defined_on.id - min_ids[dim]:
                    self.dof_id_to_fiat_id[dofs[i].id] = counter
                    entity_ids[dim][dofs[i].cell_defined_on.id - min_ids[dim]].append(counter)
                    nodes.append(dofs[i].convert_to_fiat(self.ref_el, degree, value_shape))
                    counter += 1
        self.nodes = nodes
        # for i in range(4):
        #     entity_ids[2][i] = [entity_ids[2][i][-1]] + entity_ids[2][i][:-1]
        return entity_ids, nodes

    def setup_matrices(self):
        """
        Builds the orientation matrices, the transformations of the DOFs
        under a reorientation of the entity. Structure:
            `` matrices[dimension][entity_id][orientation_key]
        Each matrix only transforms the DOFs on the entity referred to.

        The inverse matrices are also provided. 
        """
        dofs = self.generate()
        n = len(dofs)
        to_fiat = np.zeros((n, n))
        for i, d in enumerate(dofs):
            to_fiat[self.dof_id_to_fiat_id[d.id], i] = 1
        actions = {}

        def action(g):
            # row p: the image of DOF p under g, in FIAT order
            if g.perm not in actions:
                actions[g.perm] = to_fiat @ self.mapping(g).T @ to_fiat.T
            return actions[g.perm]

        min_ids = self.cell.get_starter_ids()
        matrices, reversed_matrices = {}, {}
        for dim in range(self.cell.dim() + 1):
            matrices[dim], reversed_matrices[dim] = {}, {}
            for entity in self.cell.d_entities(dim):
                e_id = entity.id - min_ids[dim]
                rows = self.entity_ids[dim][e_id]
                matrices[dim][e_id], reversed_matrices[dim][e_id] = {}, {}
                for key in range(entity.group.size()):
                    M, R = np.eye(n), np.eye(n)
                    if rows:
                        g = self.cell.extend_entity_symmetry(entity, entity.orientation_key_to_member(key))
                        block = action(g)[np.ix_(rows, rows)]
                        M[np.ix_(rows, rows)] = block
                        R[np.ix_(rows, rows)] = np.linalg.inv(block)
                    matrices[dim][e_id][key] = M
                    reversed_matrices[dim][e_id][key] = R
        self.pure_perm = self.perm and all(is_permutation_matrix(M) for by_entity in matrices.values()
                                           for by_key in by_entity.values() for M in by_key.values())
        self.apply_matrices = True
        self.entity_perms = None
        return matrices, reversed_matrices

    def __repr__(self):
        return "FuseTriple(%s, %s, (%s, %s, %s), %s)" % (
               repr(self.DOFGenerator), repr(self.cell), repr(self.spaces[0]), repr(self.spaces[1]), repr(self.spaces[2]), "X")

    @cache
    def generate(self):
        if self.dofs is None:
            self.dofs = []
            id_counter = 0
            for dof_gen in self.DOFGenerator:
                generated = dof_gen.generate(self.cell, self.spaces[2], id_counter)
                self.dofs.extend(generated)
                id_counter += len(generated)
        return self.dofs

    def mapping(self, g):
        """
        A matrix describing how the symmetry ``g`` of the cell transforms the DOFs
        For an element triple, this is the direct sum of the generators' mappings."""
        self.generate()
        return scipy.linalg.block_diag(*[gen.mapping(g) for gen in self.DOFGenerator])

    def __iter__(self):
        yield self.cell
        yield self.spaces
        yield self.DOFGenerator

    def num_dofs(self):
        return sum([dof_gen.num_dofs() for dof_gen in self.DOFGenerator])

    @property
    def degree(self):
        # TODO this isn't really correct
        return self.spaces[0].degree() + 1

    @property
    def form_degree(self):
        """The degree of differential form this element represents.

        The DOFs of a k-form are integrals over k-dimensional entities, so the
        form degree is the dimension of the lowest dimensional entity carrying
        a DOF.
        """
        return min(dof.cell_defined_on.dim() for dof in self.generate())

    def get_dof_info(self, dof, tikz=True):
        colours = {False: {0: "b", 1: "r", 2: "g", 3: "b"},
                   True: {0: "blue", 1: "red", 2: "green", 3: "black"}}
        if dof.cell_defined_on.dimension == 0:
            center = self.cell.cell_attachment(dof.cell_defined_on.id)()
        elif dof.cell_defined_on.dimension == 1:
            center = self.cell.cell_attachment(dof.cell_defined_on.id)(0)
        elif dof.cell_defined_on.dimension == 2:
            center = self.cell.cell_attachment(dof.cell_defined_on.id)(0, 0)
        else:
            center = list(sum(np.array(self.cell.vertices(return_coords=True))))

        return center, colours[tikz][dof.cell_defined_on.dimension]

    def get_value_shape(self):
        return self.spaces[0].shape

    def to_ufl(self):
        if self.ref_el is None:
            # set up for eventual conversion to FIAT if not already done
            self.ref_el = self.cell.to_fiat()
            self.poly_set = self.spaces[0].to_ON_polynomial_set(self.ref_el)
            self.entity_ids, self.nodes = self.setup_ids_and_nodes()
            self.matrices, self.reversed_matrices = self.setup_matrices()
        return FuseElement(self)

    def to_fiat(self):
        # call this to ensure set up is complete
        self.to_ufl()
        form_degree = self.form_degree
        degree = self.spaces[0].degree()
        # sanity check that the dofs span the space
        original_V, original_basis = self.compute_dense_matrix(self.ref_el, self.entity_ids, self.nodes, self.poly_set)
        if self.pure_perm:
            dual = DualSet(self.nodes, self.ref_el, self.entity_ids, self.entity_perms)
        else:
            dual = DualSet(self.nodes, self.ref_el, self.entity_ids)
        return CiarletElement(self.poly_set, dual, degree, form_degree)

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
            elif isinstance(dof.pairing, L2Pairing):
                coord = center
            tikz_commands += [dof_to_tikz(dof, coord, scale, color)]
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
                plot_dof(ax, dof, coord, color=color)
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
                    plot_dof(ax, dof, coord, color=color)
                elif isinstance(dof.pairing, L2Pairing):
                    coord = center
                    plot_dof(ax, dof, center, color=color, length=0.2)
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
        A, B = riesz_matrices(nodes, poly_set)
        V = np.dot(A, np.transpose(B))

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                new_coeffs_flat = scipy.linalg.solve(V, B, transposed=True)
            except (scipy.linalg.LinAlgWarning, scipy.linalg.LinAlgError):
                raise np.linalg.LinAlgError(
                    "Singular Vandermonde matrix (shape %s, rank %d)"
                    % (V.shape, np.linalg.matrix_rank(V)))
        return A, new_coeffs_flat

    def reverse_dof_perms(self, matrices):
        return {dim: {e: {k: np.linalg.inv(M) for k, M in by_key.items()} for e, by_key in by_entity.items()}
                for dim, by_entity in matrices.items()}

    def _to_dict(self):
        o_dict = {"cell": self.cell, "spaces": self.spaces, "dofs": self.DOFGenerator}
        return o_dict

    def dict_id(self):
        return "Triple"

    def _from_dict(o_dict):
        return ElementTriple(o_dict["cell"], o_dict["spaces"], o_dict["dofs"])


class DOFGenerator():
    """A DOF generation tuple (X, A): the generator DOFs X and the generation
    transversal A, which generates the full set of DOFs by symmetry."""

    def __init__(self, generator_funcs, transversal, *removed):
        if removed:
            raise TypeError("DOFGenerator takes (X, A): the third argument (G2) has been removed. "
                            "How the DOFs transform is now computed from the kernels' actions.")
        self.x = generator_funcs
        self.transversal = transversal
        self.dof_numbers = None
        self.ls = None
        self._generator_dofs = None
        # for each member of the transversal (in the order of transversal.members()),
        # the positions in self.ls of the DOFs it generates
        self._positions_by_member = None
        self.cell = None
        self.space = None
        self._stabiliser = None
        self._representation = None

    def __iter__(self):
        yield self.x
        yield self.transversal

    def add_context(self, cell, space):
        """ Fix the cell and space of the DOF generator. A single generator object can
            only belong to one triple."""
        if self.cell is not None and (self.cell is not cell or self.space is not space):
            raise ValueError(f"{self} already belongs to a triple on {self.cell}: use a new DOFGenerator")
        self.cell = cell
        self.space = space
        self.transversal = self.transversal.add_cell(cell)

    def num_dofs(self):
        if self.dof_numbers is None:
            raise ValueError("DOFs not generated yet")
        return self.dof_numbers

    def generate(self, cell, space, id_counter):
        if self.ls is None:
            self.ls = []
            self._generator_dofs = []
            members = self.transversal.members()
            self._positions_by_member = [[] for _ in members]
            for l_g in self.x:
                i = 0
                for a, g in enumerate(members):
                    generated = l_g(g)
                    if not isinstance(generated, list):
                        generated = [generated]
                    for dof in generated:
                        dof.add_context(self, cell, space, g, id_counter, i)
                        id_counter += 1
                        i += 1
                    self._positions_by_member[a] += range(len(self.ls), len(self.ls) + len(generated))
                    self.ls.extend(generated)
                    if g.perm.is_Identity:
                        self._generator_dofs.extend(generated)
            self.dof_numbers = len(self.ls)
            self.dof_ids = [dof.id for dof in self.ls]
        return self.ls

    def generator_dofs(self):
        """The generator DOFs X, with their immersion expanded through ``generate()`` 
            """
        if self._generator_dofs is None:
            raise ValueError("DOFs not generated yet")
        return self._generator_dofs

    def _act(self):
        """Constructs a function from the symmetry group of the cell 
        to DOFs evaluated on the cell transformed by that symmetry group member."""
        X = self.generator_dofs()
        G = self.cell.group
        cache = self.__dict__.setdefault("_act_cache", {})

        def act(s):
            # act by the cell's own member, so that it composes with the kernels' orientations
            s, = G._match([s])
            if s.perm not in cache:
                cache[s.perm] = self.space.functional_matrix(self.cell, [d(s) for d in X])
            return cache[s.perm]
        return act

    def stabiliser(self):
        """For non immersed DOFs, the stabiliser is H_X: the symmetries of the triple's cell
        that fix the span of this DOF generator tuple.
        For immersed DOFs, the stabiliser is H_E: the symmetries that do not move the entity they
        are immersed on."""
        if self._stabiliser is None:
            entities = {(x.C.dim(), x.start_node) for x in self.x if isinstance(x, ImmersedDOFs)}
            if len(entities) == 1 and all(isinstance(x, ImmersedDOFs) for x in self.x):
                dim, node = entities.pop()
                self._stabiliser = self.cell.entity_stabiliser(self.cell.d_entities(dim)[node])
            else:
                self._stabiliser = self.cell.group.span_stabiliser(self._act())
        return self._stabiliser

    def representation(self):
        """rho_X: the action of the stabiliser H_X on span(X), as {permutation: matrix}."""
        if self._representation is None:
            self._representation = self.stabiliser().span_representation(self._act())
        return self._representation

    def mapping(self, g):
        """M_g: how the symmetry ``g`` of the cell transforms the DOFs this
        generator generates, in generation order. Column p writes the image of
        DOF p under ``g`` in terms of the generated DOFs, so that
        M_{h o g} = M_h M_g.

        A generated DOF is a member ``a`` of the transversal applied to a
        generator DOF. Writing g o a = a' o h, with h in the stabiliser,
        ``g`` sends the block generated by ``a`` to the block generated by
        ``a'``, transformed within it by h.
        """
        if not self.ls:
            # an empty generator has no stabiliser to decompose by
            return np.zeros((0, 0))
        G = self.cell.group
        g, = G._match([g])
        A = G._match(self.transversal)
        H = self.stabiliser()
        index = self._positions_by_member
        M = np.zeros((len(self.ls), len(self.ls)))
        for i, a in enumerate(A):
            a_new, h = self.transversal.decompose(g.compose(a), H)
            j = [b.perm for b in A].index(a_new.perm)
            M[np.ix_(index[j], index[i])] = self._stabiliser_mapping(h)
        return M

    def _stabiliser_mapping(self, h):
        """How a member ``h`` of the stabiliser transforms the generator DOFs.

        For immersed DOFs, ``h`` fixes the entity they are immersed on and
        acts on them through the symmetry it induces on that entity.
        """
        immersed = [isinstance(x, ImmersedDOFs) for x in self.x]
        if not any(immersed):
            return self.representation()[tuple(h.perm.array_form)]
        if not all(immersed):
            raise NotImplementedError(f"{self} mixes immersed and non-immersed generators")
        blocks = []
        for x in self.x:
            entity, o = self.cell.permute_entities(h, x.C.dim())[x.start_node]
            if entity != self.cell.d_entities_ids(x.C.dim())[x.start_node]:
                raise ValueError(f"{h} does not fix the entity {x} is immersed on")
            o, = x.C.group._match([o])
            blocks.append(x.triple.mapping(o))
        return scipy.linalg.block_diag(*blocks)

    def check_transversal(self):
        """This function checks the given generation transversal is
        actually a left transversal of the stabiliser H."""
        if not self.generator_dofs():
            return
        H = self.stabiliser()
        if not self.cell.group.is_left_transversal(self.transversal, H):
            raise ValueError(f"{self.transversal} is not a left transversal of the stabiliser of the generator DOFs "
                             f"of {self} (order {H.size()}), so it does not generate each DOF exactly once")

    def make_entity_ids(self):
        dofs = self.ls
        entity_ids = {}
        min_ids = dofs[0].cell.get_starter_ids()

        top = dofs[0].cell.get_topology()

        for dim in sorted(top):
            entity_ids[dim] = {i: [] for i in top[dim]}

        for i in range(len(dofs)):
            entity = dofs[i].cell_defined_on
            dim = entity.dim()
            entity_ids[dim][entity.id - min_ids[dim]].append(i)
        return entity_ids

    def __repr__(self):
        repr_str = "DOFGen("
        for x_elem in self.x:
            repr_str += "g(" + str(x_elem) + ")"
        repr_str += str(self.transversal) + ")"
        return repr_str

    def _to_dict(self):
        o_dict = {"x": self.x, "transversal": self.transversal}
        return o_dict

    def dict_id(self):
        return "DOFGen"

    def _from_dict(obj_dict):
        # dicts written before G2 was removed store the transversal as "g1"
        return DOFGenerator(obj_dict["x"], obj_dict["transversal"] if "transversal" in obj_dict else obj_dict["g1"])


class ImmersedDOFs():

    def __init__(self, target_cell, triple, trace=None, start_node=0):
        self.target_cell = target_cell
        self.triple = triple
        self.C, self.V, self.E = triple
        # an explicit trace; without one each DOF is traced through its kernel's action
        self.trace = None if trace is None else trace(target_cell)
        self.start_node = start_node

    def __call__(self, g):
        target_node, o = self.target_cell.permute_entities(g, self.C.dim())[self.start_node]
        if self.C.dim() > 0 and o != o.group.identity:
            raise ValueError(f"{g} sends the base entity onto {self.target_cell.get_node(target_node)} but reorients it; "
                             "each member of the transversal must carry the base entity onto its image without reorienting it")
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
        trace = "" if self.trace is None else str(self.trace) + "_"
        for dof_gen in self.E:
            repr_str += "Im_" + trace + str(self.target_cell) + "(" + str(dof_gen) + ")"
        return repr_str

    def _to_dict(self):
        o_dict = {"target_cell": self.target_cell, "triple": self.triple, "trace": self.trace}
        return o_dict

    def dict_id(self):
        return "ImmersedDOF"

    def _from_dict(obj_dict):
        return ImmersedDOFs(obj_dict["target_cell"], obj_dict["triple"], obj_dict["trace"])


def _named_trace(action, entity):
    """The trace class that names how ``action`` lifts from ``entity``, for
    checking the deprecated trace argument of :func:`immerse`."""
    if action.k == 0 or action.proxy == "scalar":
        return TrH1
    if action.proxy == "hodge":
        return TrHDiv
    return TrHCurl if entity.dimension == 1 else TrH1


def immerse(target_cell, triple, target_space=None, node=0):
    """Immerse the DOFs of ``triple`` into ``target_cell``.

    Each DOF is lifted, and so traced, through its own kernel's action.
    ``target_space`` is only needed for the derivative traces (``TrGrad``,
    ``TrHess``). Passing TrH1, TrHCurl or TrHDiv is deprecated: it is checked
    against the kernels, or, for kernels that declare no action, supplies
    their action.
    """
    if target_space is None or not issubclass(target_space, (TrH1, TrHCurl, TrHDiv)):
        return ImmersedDOFs(target_cell, triple, target_space, node)
    trace = target_space(target_cell)
    undeclared = [dof.kernel for dof in triple.generate() if dof.kernel.declared_action is None]
    if undeclared and isinstance(trace, (TrHCurl, TrHDiv)):
        warnings.warn(f"Kernels {undeclared} declare no action and are immersed with {trace}: taking the "
                      "action from the trace. Pass action=HCurl or action=HDiv to the kernels instead.",
                      DeprecationWarning, stacklevel=2)
    elif not undeclared:
        for dof in triple.generate():
            action = dof.kernel.action.resolve(dof.cell_defined_on, dof.kernel.rank)
            if _named_trace(action, dof.cell_defined_on) is not target_space:
                raise ValueError(f"The kernel {dof.kernel} of {triple} has action {action}, which does not "
                                 f"lift with the {trace} trace")
        warnings.warn(f"The trace is read from the kernels' actions: drop the {trace} argument to immerse.",
                      DeprecationWarning, stacklevel=2)
    return ImmersedDOFs(target_cell, triple, target_space, node)
