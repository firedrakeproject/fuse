import fuse.cells as cells
from fuse.utils import orientation_value
from sympy.combinatorics import PermutationGroup, Permutation
from sympy.combinatorics.named_groups import SymmetricGroup, DihedralGroup, CyclicGroup, AlternatingGroup
from sympy.matrices.expressions import PermutationMatrix
import numpy as np
import sympy as sp


def perm_matrix_to_perm_array(p_mat):
    summed = np.sum(p_mat, axis=0)
    if np.all(summed == np.zeros_like(summed)):
        return list(np.zeros_like(summed))
    assert np.all(summed == np.ones_like(summed))
    res = []
    for row in p_mat:
        indices = list(row).index(1)
        res += [indices]
    return res


class GroupMemberRep(object):

    def __init__(self, perm, M, group):
        self.perm = perm
        self.transform_matrix = M
        self.group = group

    def __call__(self, x):
        if isinstance(x, cells.Point):
            return x.orient(self)
        if isinstance(x, sp.Expr):
            x_ones = sp.r_[sp.array(x), sp.ones(1)]
        else:
            x_ones = np.r_[np.array(x), np.ones(1)]

        sum = np.matmul(x_ones, self.transform_matrix)
        if len(sum.shape) > 1:
            return tuple(map(tuple, sum))
        return tuple(sum)

    def permute(self, lst):
        n = len(lst)
        if n > self.perm.size:
            temp_perm = Permutation(self.perm, size=n)
            return temp_perm(lst)
        return self.perm(lst)

    def compute_perm(self, base_val=None):
        if base_val:
            val_list = [x + base_val for x in self.perm.array_form]
        else:
            val_list = self.perm.array_form
        val = self.numeric_rep()
        return val, val_list

    def numeric_rep(self):
        identity = self.group.identity.perm.array_form
        m_array = self.perm.array_form
        return orientation_value(identity, m_array)

    def __eq__(self, x):
        assert isinstance(x, GroupMemberRep)
        return self.perm == x.perm and self.group.cell.dim() == x.group.cell.dim()

    def __hash__(self):
        return hash((self.perm, self.group))

    def __mul__(self, x):
        assert isinstance(x, GroupMemberRep)
        return self.group.get_member(self.perm * x.perm)

    def __invert__(self):
        return self.group.get_member(~self.perm)

    def __repr__(self):
        string = "g"
        string += str(self.perm.array_form)
        return string

    def matrix_form(self):
        return np.array(PermutationMatrix(self.perm).as_explicit()).astype(np.float64)

    def lin_combination_form(self):
        if self.group.cell.dimension == 0:
            return [1]
        bvs = self.group.cell.basis_vectors()
        M = np.array(bvs).T
        trans_bvs = np.array([self(bvs[i]) for i in range(len(bvs))]).T
        return np.linalg.solve(M, trans_bvs)


class PermutationSetRepresentation():
    """
        A representation of a set of permutations (can be a full group) on a cell.

        Args:
            [permutations]: the list of permutations in the set
            cell (optional): the cell the group is representing the operations on

    """
    def __init__(self, perm_list, cell=None):
        assert len(perm_list) > 0
        self.perm_list = perm_list

        if not any([p.is_Identity for p in self.perm_list]):
            p = self.perm_list[0]
            self.perm_list.append(Permutation([i for i in range(0, p.size)]))

        if cell is not None:
            self.cell = cell
            vertices = cell.vertices(return_coords=True)
            self._members = []
            counter = 0

            for p in self.perm_list:
                if len(vertices) > p.size:
                    temp_perm = Permutation(p, size=len(vertices))
                    reordered = temp_perm(vertices)
                else:
                    reordered = p(vertices)
                A = np.c_[np.array(vertices, dtype=float), np.ones(len(vertices))]
                b = np.array(reordered, dtype=float)

                M, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                p_rep = GroupMemberRep(p, M, self)
                if p.is_Identity:
                    self.identity = p_rep
                self._members.append(p_rep)
                counter += 1

    def add_cell(self, cell):
        return PermutationSetRepresentation(self.perm_list, cell=cell)

    def members(self, perm=False):
        if self.cell is None:
            raise ValueError("Group does not have a domain - members have not been calculated")
        if perm:
            return [m.perm for m in self._members]
        return self._members

    def transform_between_perms(self, perm1, perm2):
        member_perms = self.members(perm=True)
        perm1 = Permutation.from_sequence(perm1)
        perm2 = Permutation.from_sequence(perm2)
        assert perm1 in member_perms
        assert perm2 in member_perms
        return self.get_member(~Permutation(perm1)) * self.get_member(Permutation(perm2))

    def get_member(self, perm):
        if not isinstance(perm, Permutation):
            perm = Permutation.from_sequence(perm)
        for m in self.members():
            if m.perm == perm:
                return m
        raise ValueError("Permutation not a member of group")

    def compute_num_reps(self, base_val=0):
        """ Compute the numerical represention of each member as compared to the identity.
        Where the numerical rep is:

        M.index(id[0]) = a; M.remove(id[0])
        M.index(id[1]) = b; M.remove(id[1])
        M.index(id[2]) = c; M.remove(id[2])

        o = (a * 2!) + (b * 1!) + (c * 0!)
        """
        members = self.members()
        res = {}
        for m in members:
            val, perm = m.compute_perm(base_val)
            res[val] = perm
        return res

    def size(self):
        return len(self.perm_list)

    def __mul__(self, other_group):
        # convert to set to remove duplicates
        return PermutationSetRepresentation(list(set(self.perm_list + other_group.perm_list)))

    def __repr__(self):
        return "GS" + str(self.size())


class GroupRepresentation(PermutationSetRepresentation):
    """
    A representation of a group by its matrix operations.

    Args:
        base_group: the sympy group that is being represented
        cell (optional): the cell the group is representing the operations on

    """

    def __init__(self, base_group, cell=None):
        assert isinstance(base_group, PermutationGroup)
        self.base_group = base_group
        self.generators = []
        if cell is not None:
            self.cell = cell
            vertices = cell.vertices(return_coords=True)

            self._members = []
            counter = 0
            for g in self.base_group.elements:
                if len(vertices) > g.size:
                    temp_perm = Permutation(g, size=len(vertices))
                    reordered = temp_perm(vertices)
                else:
                    reordered = g(vertices)
                A = np.c_[np.array(vertices, dtype=float), np.ones(len(vertices))]
                b = np.array(reordered, dtype=float)

                M, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                p_rep = GroupMemberRep(g, M, self)
                self._members.append(p_rep)
                if g.is_Identity:
                    self.identity = p_rep
                counter += 1

            # this order produces simpler generator lists
            # self.generators.reverse()

            # temp_group_elems = self.base_group.elements

            # temp_group_elems.remove(self.base_group.identity)

            # remaining_members = self.compute_reps(self.base_group.identity,
            #                                       None, temp_group_elems)
            # assert (len(remaining_members) == 0)
        else:
            self.cell = None

    def conjugacy_class(self, g):
        conj_class = set()
        for x in self.members():
            res = ~x * g * x
            conj_class.add(res)
        return conj_class

    def add_cell(self, cell):
        return GroupRepresentation(self.base_group, cell=cell)

    def size(self):
        if hasattr(self, "_members"):
            assert len(self._members) == self.base_group.order()
        return self.base_group.order()

    def members(self, perm=False):
        if self.cell is None:
            raise ValueError("Group does not have a domain - members have not been calculated")
        if perm:
            return [m.perm for m in self._members]
        return self._members

    def transform_between_perms(self, perm1, perm2):
        member_perms = self.members(perm=True)
        perm1 = Permutation.from_sequence(perm1)
        perm2 = Permutation.from_sequence(perm2)
        assert perm1 in member_perms
        assert perm2 in member_perms
        return ~self.get_member(Permutation(perm1)) * self.get_member(Permutation(perm2))

    def get_member(self, perm):
        for m in self.members():
            if m.perm == perm:
                return m
        raise ValueError("Permutation not a member of group")

    # def compute_reps(self, g, path, remaining_members):
    #     # breadth first search to find generator representations of all members
    #     if len(remaining_members) == 0:
    #         return remaining_members

    #     next_candidates = []
    #     for generator in self.generators:
    #         new_perm = g*generator.perm
    #         if new_perm in remaining_members:
    #             if not path:
    #                 new_path = generator.rep
    #                 new_M = generator.transform_matrix
    #                 print(new_M)
    #                 assert (new_perm == generator.perm)
    #                 self._members.append(generator)
    #             else:
    #                 new_path = path.copy()
    #                 new_path.extend(generator.rep)
    #                 print(path)
    #                 new_M = np.matmul(generator.transform_matrix, new_M)
    #                 self._members.append(GroupMemberRep(new_perm,
    #                                                     new_path,
    #                                                     self))
    #             remaining_members.remove(new_perm)
    #             next_candidates.append((new_perm, new_path))

    #     for (new_perm, new_path) in next_candidates:
    #         remaining_members = self.compute_reps(new_perm,
    #                                               new_path,
    #                                               remaining_members)
    #     return remaining_members

    def __mul__(self, other_group):
        return GroupRepresentation(PermutationGroup(self.base_group.generators + other_group.base_group.generators))

    def __truediv__(self, other_frac):
        """ This isn't a mathematically accurate representation of
            what it means to be a quotient group but it works on S3/S2
            Have to compare cyclic forms as groups may not be defined on
            the same number of elements
            Doesn't work on D4/S2 but does on D4/C4 """
        assert isinstance(other_frac, GroupRepresentation)
        self_cyclic_gens = [gen.cyclic_form
                            for gen in self.base_group.generators]
        other_cyclic_gens = [gen.cyclic_form
                             for gen in other_frac.base_group.generators]
        if not all([c2 in self_cyclic_gens for c2 in other_cyclic_gens]):
            raise ValueError("Invalid Quotient - mismatched cycles")
        remaining_perms = [gen for gen in self.base_group.generators
                           if gen.cyclic_form not in other_cyclic_gens]

        if len(remaining_perms) == 0:
            raise ValueError("Invalid Quotient - no group formed")

        return GroupRepresentation(PermutationGroup(remaining_perms))

    def __repr__(self):
        return "GR" + str(self.size())

    # def __eq__(self, other):
    #     # TODO work on idea of group equality
    #     assert isinstance(other, GroupRepresentation)
    #     res = True
    #     for m in self.members():
    #         res = res and m in other.members()
    #     return res

    def _to_dict(self):
        return {"members": [m.perm.array_form for m in self._members]}

    def dict_id(self):
        return "Group"

    def _from_dict(o_dict):
        perm_group = PermutationGroup([Permutation(m) for m in o_dict["members"]])
        # , o_dict["cell"]
        return GroupRepresentation(perm_group)


def get_sym_group(n):
    return GroupRepresentation(SymmetricGroup(n))


def get_cyc_group(n):
    return GroupRepresentation(CyclicGroup(n))


S1 = GroupRepresentation(SymmetricGroup(1))
S2 = GroupRepresentation(SymmetricGroup(2))
S3 = GroupRepresentation(SymmetricGroup(3))
S4 = GroupRepresentation(SymmetricGroup(4))

D4 = GroupRepresentation(DihedralGroup(4))

C3 = GroupRepresentation(CyclicGroup(3))
C4 = GroupRepresentation(CyclicGroup(4))

Z2 = GroupRepresentation(CyclicGroup(2))
Z3 = GroupRepresentation(CyclicGroup(3))
Z4 = GroupRepresentation(CyclicGroup(4))


D2 = GroupRepresentation(DihedralGroup(2))
A4 = GroupRepresentation(AlternatingGroup(4))
A3 = GroupRepresentation(AlternatingGroup(3))

tri_C3 = PermutationSetRepresentation([Permutation([0, 1, 2]), Permutation([2, 0, 1]), Permutation([1, 0, 2])])
# tet_edges = PermutationSetRepresentation([Permutation([0, 1, 2, 3]), Permutation([0, 2, 3, 1]), Permutation([1, 2, 0, 3]),
#                                           Permutation([0, 3, 1, 2]), Permutation([1, 3, 2, 0]), Permutation([2, 3, 0, 1])])
tet_edges_old = PermutationSetRepresentation([Permutation([0, 1, 2, 3]), Permutation([1, 2, 3, 0]), Permutation([2, 3, 0, 1]),
                                              Permutation([1, 3, 0, 2]), Permutation([2, 0, 1, 3]), Permutation([3, 0, 1, 2])])
tet_edges = PermutationSetRepresentation([Permutation([0, 1, 2, 3]), Permutation([1, 2, 3, 0]), Permutation([2, 0, 3, 1]),
                                          Permutation([3, 2, 0, 1]), Permutation([2, 0, 1, 3]), Permutation([0, 3, 1, 2])])
tet_faces_old = PermutationSetRepresentation([Permutation([0, 1, 2, 3]), Permutation([1, 2, 3, 0]), Permutation([1, 3, 2, 0]),
                                              Permutation([3, 0, 2, 1])])
tet_faces = PermutationSetRepresentation([Permutation([0, 1, 2, 3]), Permutation([0, 2, 3, 1]), Permutation([0, 3, 1, 2]),
                                          Permutation([3, 2, 0, 1])])

sq_edges = PermutationSetRepresentation([Permutation([0, 1, 2, 3]), Permutation([1, 2, 3, 0]), Permutation([3, 0, 1, 2]), Permutation([2, 3, 0, 1])])
