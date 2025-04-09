from test_2d_examples_docs import construct_cg3, construct_nd, construct_rt, construct_hermite
from test_3d_examples_docs import construct_tet_ned, construct_tet_rt, construct_tet_cg3


def test_2d_elements():
    elems = [construct_cg3, construct_nd, construct_rt, construct_hermite]

    for e in elems:
        triple = e()
        print(triple.to_tikz())


def test_3d_elements():
    elems = [construct_tet_ned, construct_tet_rt, construct_tet_cg3]

    for e in elems:
        triple = e()
        print(triple.to_tikz())
