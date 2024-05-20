# from firedrake import *
from groups.groups import Group, S1, S2, S3
from cell_complex.cells import Point, Edge
import sympy as sp
from spaces.element_sobolev_spaces import ElementSobolevSpace
from dof_lang.dof import DeltaPairing, L2InnerProd, DOF, MyTestFunction
import matplotlib.pyplot as plt


class ElementTriple():

    def __init__(self, cell, spaces, dof_gen):
        assert isinstance(cell, Point)
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
            if hasattr(space, "domain"):
                cell_spaces.append(space(cell))
            else:
                cell_spaces.append(space)
        self.spaces = tuple(cell_spaces)
        self.DOFGenerator = dof_gen

        # assert self.num_dofs() > self.spaces[0].subdegree

    def generate(self):
        res = []
        for dof_gen in self.DOFGenerator:
            res.extend(dof_gen.generate(self.cell))
        return res

    def __iter__(self):
        yield self.cell
        yield self.spaces
        yield self.DOFGenerator

    def num_dofs(self):
        return sum([dof_gen.num_dofs() for dof_gen in self.DOFGenerator])

    def plot(self):
        # point evaluation nodes only
        dofs = self.generate()
        identity = MyTestFunction(lambda *x: x)
        if self.cell.dimension < 3:
            self.cell.plot(show=False, plain=True)
            for dof in dofs:
                print(dof)
                print(dof.trace_entity)
                coord = dof.eval(identity)
                if dof.trace_entity.dimension == 1:
                    color = "r"
                elif dof.trace_entity.dimension == 2:
                    color = "g"
                else:
                    color = "b"
                plt.scatter(coord[0], coord[1], marker="o", color=color)
            plt.show()
        elif self.cell.dimension == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            self.cell.plot3d(show=False, ax=ax)
            for dof in dofs:
                coord = dof.eval(identity)
                if dof.trace_entity.dimension == 1:
                    color = "r"
                elif dof.trace_entity.dimension == 2:
                    color = "g"
                else:
                    color = "b"
                ax.scatter(coord[0], coord[1], coord[2], color=color)
            plt.show()
        else:
            raise ValueError("Plotting not supported in this dimension")


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
        
    def generate(self, cell):
        if self.ls is None:
            self.ls = []
            for g in self.g1.members():
                for l_g in self.x:
                    generated = l_g(g)
                    if not isinstance(generated, list):
                        generated = [generated]
                    for dof in generated:
                        dof.add_entity(cell)
                    self.ls.extend(generated)
            self.dof_numbers = len(self.ls)
            return self.ls
        return self.ls
    
    def __repr__(self):
        repr_str = ""
        for x_elem in self.x:
            repr_str += "g(" + str(x_elem) + ")"
        return repr_str


class ImmersedDOF():

    def __init__(self, target_cell, triple, target_space, start_node=0):
        self.target_cell = target_cell
        self.triple = triple
        self.C, self.V, self.E = triple
        self.target_space = target_space(target_cell)
        self.start_node = start_node

    def __call__(self, g):
        target_node_pair = self.target_cell.permute_entities(g, self.C.dim())[self.start_node]
        print(target_node_pair)
        target_node, o = target_node_pair

        print("Attaching node", target_node)

        attachment = self.target_cell.cell_attachment(target_node)
        new_dofs = []
        for generated_dof in self.triple.generate():
            new_dof = generated_dof.immerse(self.target_cell.get_node(target_node),
                                            lambda x: attachment(o(x)),
                                            self.target_space, g)
            new_dofs.append(new_dof)
        return new_dofs
    
    def __repr__(self):
        repr_str = ""
        for dof_gen in self.E:
            repr_str += "Im_" + str(self.target_space) + "_" + str(self.target_cell) + "(" + str(dof_gen) + ")"
        return repr_str


def immerse(target_cell, triple, target_space, node=0):
    return ImmersedDOF(target_cell, triple, target_space, node)
