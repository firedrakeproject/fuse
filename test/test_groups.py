from pytest import *
from fuse import *
from fuse.groups import perm_matrix_to_perm_array
from sympy.combinatorics import Permutation
from test_convert_to_fiat import create_dg1
from test_3d_examples_docs import (construct_tet_cg4, construct_tet_cg6, construct_tet_rt2,
                                   construct_tet_ned_2nd_kind_2, construct_tet_ned_2nd_kind_3)
from FIAT.orientation_utils import make_entity_permutations_simplex
import numpy as np


def test_numerical_orientations():
    vert = Point(0)
    print(vert.group.compute_num_reps())
    edge = Point(1, [Point(0), Point(0)], vertex_num=2)
    # group = S2.add_cell(edge)
    print(edge.group.compute_num_reps())

    cell = polygon(3)
    # group = S3.add_cell(cell)
    print(cell.group.compute_num_reps())
    print(cell.group.compute_num_reps(base_val=5))
    mems = cell.group.members()
    for m in mems:
        print(m.compute_perm())
        print(m.transform_matrix)


def test_permsets():

    cell = polygon(3)
    p = Permutation([0, 2, 1])
    print(p.size)
    print(p.is_Identity)
    c3 = C3.add_cell(cell)

    deg = 1
    vert_dg = create_dg1(cell.vertices(get_class=True)[0])
    xs = [immerse(cell, vert_dg, TrH1)]

    Pk = PolynomialSpace(deg, deg)
    cg = ElementTriple(cell, (Pk, CellL2, C0), DOFGenerator(xs, c3, S1))

    dofs = cg.generate()

    for d in dofs:
        print(d)

    tri = cell
    edge = tri.edges(get_class=True)[0]
    vert = tri.vertices(get_class=True)[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, CellL2, C0), DOFGenerator(xs, S1, S1))

    v_xs = [immerse(tri, dg0, TrH1)]
    v_dofs = DOFGenerator(v_xs, c3, S1)

    xs = [DOF(DeltaPairing(), PointKernel((-1/3)))]
    dg0_int = ElementTriple(edge, (P1, CellH1, C0), DOFGenerator(xs, S2, S1))
    print([d.generation for d in dg0_int.generate()])

    e_xs = [immerse(tri, dg0_int, TrH1)]
    e_dofs = DOFGenerator(e_xs, c3, S1)

    i_xs = [lambda g: DOF(DeltaPairing(), PointKernel(g((0, 0))))]
    i_dofs = DOFGenerator(i_xs, S1, S1)

    cg3 = ElementTriple(tri, (P3, CellH1, C0), [v_dofs, e_dofs, i_dofs])

    for d in cg3.generate():
        print(d)


def test_conj():
    cell = polygon(3)
    g = cell.group.members()[4]
    print("g", g)

    # print(cell.group.conjugacy_class(g))
    import numpy as np
    for g in cell.group.members():
        print("g", g)
        print("transformed", g((-1/2, -np.sqrt(3)/3)))
        # for row in g.transform_matrix:
        #     print(row)

    # print("others")
    # for g in cell.group.members():
    #     if g not in g1.members():
    #         print("g", g)
    #         print("g", g.perm.cycle_structure)
    #         # for row in g.matrix_form():
    #         #     print(row)
    mems = cell.group.members()
    for m in mems:
        print(m.numeric_rep())


# def test_group_equality():
#     cell = polygon(3)

#     s1 = S1.add_cell(cell)
#     s1_new = S1.add_cell(cell)

#     assert s1 == s1_new

    # cell2 = Point(1, [Point(0), Point(0)], vertex_num=2)

    # s1 = S2.add_cell(cell)
    # s1_new = S2.add_cell(cell2)

    # assert not s1 == s1_new

def test_perm_mat_conversion():
    cell = polygon(3)
    cS3 = S3.add_cell(cell)

    for g in cS3.members():
        mat_form = g.matrix_form()
        array_form = perm_matrix_to_perm_array(mat_form)
        assert np.allclose(g.perm.array_form, array_form)


def test_coset_convention():
    # Cosets are the left cosets gH, and cosets_by_submember recovers the right
    # factor h of x = g*h. matrix_form_subgroup depends on this order, so it is
    # pinned here: reversing the product is not a relabelling.
    cell = polygon(3)
    cS3 = S3.add_cell(cell)
    cC3 = C3.add_cell(cell)

    sub_members = cC3.members()
    assert sub_members[0].perm.is_Identity

    cosets = cS3.cosets(cC3)
    by_submember = cS3.cosets_by_submember(cC3)

    seen = []
    for coset in cosets:
        assert len(coset) == len(sub_members)
        for i, h in enumerate(sub_members):
            assert coset[i] == coset[0] * h
            assert by_submember[coset[i].array_form] == h
        seen += [m.array_form for m in coset]
    assert sorted(seen) == sorted([m.array_form for m in cS3.members()])

    assert {k: v.array_form for k, v in by_submember.items()} == {
        (0, 1, 2): (0, 1, 2), (0, 2, 1): (0, 1, 2),
        (1, 2, 0): (1, 2, 0), (1, 0, 2): (1, 2, 0),
        (2, 0, 1): (2, 0, 1), (2, 1, 0): (2, 0, 1)}


def test_face_matrices_match_fiat():
    # The orientation numbering FUSE produces for a triangular face has to agree
    # with the one FIAT indexes its entity permutations by.
    elem = construct_tet_cg4()
    elem.to_ufl()
    fiat_perms = make_entity_permutations_simplex(2, 2)

    for face, dof_ids in elem.entity_ids[2].items():
        for o, mat in elem.matrices[2][face].items():
            block = np.asarray(mat)[np.ix_(dof_ids, dof_ids)]
            assert perm_matrix_to_perm_array(block) == fiat_perms[o]


SIX_MEMBER_XFAIL = mark.xfail(
    strict=True,
    reason="A free orbit of the full symmetry group legitimately violates this. The six "
           "member branch of matrix_form_subgroup sends orientation 3, not 0, to the "
           "identity permutation, so M(0) is not the identity and the family is not a "
           "homomorphism. A replacement satisfying both was tried (6dacc09) and broke "
           "test_const_vec[N1-3]: facet matching only fixes the family up to a left "
           "factor, and that replacement picked a different, face dependent one.")


def entity_orientation_matrices(elem, dim, entity=0):
    dof_ids = elem.entity_ids[dim][entity]
    return {o: np.asarray(mat)[np.ix_(dof_ids, dof_ids)]
            for o, mat in elem.matrices[dim][entity].items()}


@mark.parametrize("elem_gen", [
    param(construct_tet_cg4, id="CG-4"),
    param(construct_tet_rt2, id="RT-2"),
    param(construct_tet_ned_2nd_kind_2, id="N2curl-2"),
    param(construct_tet_cg6, id="CG-6", marks=SIX_MEMBER_XFAIL),
    param(construct_tet_ned_2nd_kind_3, id="N2curl-3", marks=SIX_MEMBER_XFAIL),
])
def test_orientation_matrix_is_representation(elem_gen):
    # Entities whose DOFs come from a coset orbit carry a representation of their
    # symmetry group: the identity orientation does not move DOFs, and composing two
    # orientations composes their matrices. A free orbit of the full symmetry group
    # does not - see SIX_MEMBER_XFAIL.
    elem = elem_gen()
    elem.to_ufl()

    for dim in range(1, elem.cell.dim()):
        if not elem.entity_ids[dim][0]:
            continue
        mats = entity_orientation_matrices(elem, dim)
        assert np.allclose(mats[0], np.eye(mats[0].shape[0]))

        members = elem.cell.d_entities(dim)[0].group.members()
        for a in members:
            for b in members:
                composed = mats[(a * b).numeric_rep()]
                assert np.allclose(composed, mats[b.numeric_rep()] @ mats[a.numeric_rep()])


def test_s3():
    # Golden pin on the six member branch. These matrices are the ones proven correct by
    # the full suite, so any change to that branch has to reproduce them exactly. Runs in
    # milliseconds, unlike the assembly tests that actually discriminate.
    from fuse.groups import perm_list_to_matrix

    s3 = S3.add_cell(polygon(3))
    for g in s3.members():
        members = [m.numeric_rep() for m in s3.members()]
        permuted_members = [((m)*(~g)).numeric_rep() for m in s3.members()]
        mapping = {4: 4, 3: 0, 0: 3}
        if (~g).numeric_rep() in mapping.keys():
            n = g.group.get_member_by_val(mapping[(~g).numeric_rep()])
            permuted_members = [((m)*(~n)).numeric_rep() for m in s3.members()]
        expected = perm_list_to_matrix(members, permuted_members)

        assert np.allclose(g.matrix_form_subgroup(s3), expected), g
