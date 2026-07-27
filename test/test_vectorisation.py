from fuse import *
from fuse.element_construction import (construct_tri_cgN, construct_tri_ndN, construct_tri_rtN,
                                       construct_tet_cgN, construct_tet_ndN, construct_tet_rtN,
                                       construct_dgNminus)
from recursivenodes import recursive_nodes
import sympy as sp
import pytest


def interval_dgN(deg):
    """A DG element on an interval, built as CG's edge sub-element is."""
    edge = polygon(3).edges()[0]
    pts = recursive_nodes(1, deg, domain="equilateral")[1:-1].flatten()
    dofs = [DOFGenerator([DOF(DeltaPairing(), PointKernel((p,)))], S2, S1)
            for p in pts[:len(pts) // 2]]
    return ElementTriple(edge, (PolynomialSpace(deg), CellL2, C0), dofs)


def interval_cg1():
    cell = polygon(3)
    edge = cell.edges()[0]
    vert = cell.vertices()[0]
    dg0 = ElementTriple(vert, (P0, CellL2, C0),
                        DOFGenerator([DOF(DeltaPairing(), PointKernel(()))], S1, S1))
    return ElementTriple(edge, (P1, CellH1, C0), [DOFGenerator([immerse(edge, dg0, TrH1)], S2, S1)])


# The DOFs of a k-form are integrals over k-dimensional entities, so the form
# degree is the dimension of the lowest dimensional entity carrying a DOF.
@pytest.mark.parametrize("name,builder,expected", [
    ("tri cg2", lambda: construct_tri_cgN(2), 0),
    ("tri nd1", lambda: construct_tri_ndN(1), 1),
    ("tri rt1", lambda: construct_tri_rtN(1), 1),
    ("tri dg1", lambda: construct_dgNminus(2)(1), 2),
    ("tet cg2", lambda: construct_tet_cgN(2), 0),
    ("tet nd1", lambda: construct_tet_ndN(1), 1),
    ("tet rt1", lambda: construct_tet_rtN(1), 2),
    ("tet dg1", lambda: construct_dgNminus(3)(1), 3),
    ("interval cg1", interval_cg1, 0),
    ("interval dg3", lambda: interval_dgN(3), 1),
])
def test_form_degree_by_entity_dim(name, builder, expected):
    assert builder().form_degree == expected


@pytest.mark.parametrize("deg", [1, 2, 3])
def test_form_degree_unchanged_for_scalar_cg(deg):
    assert construct_tri_cgN(deg).form_degree == 0


@pytest.mark.parametrize("deg", [1, 2])
def test_to_fiat_reports_form_degree(deg):
    assert construct_tri_cgN(deg).to_fiat().get_formdegree() == 0


def test_component_kernel_wraps_point_kernel():
    """Wrapping preserves the base's points and weights, and only rewrites components."""
    cell = polygon(3)
    base = DOF(DeltaPairing(), PointKernel((0.25, 0.25)), entity=cell, cell=cell)
    wrapped = DOF(DeltaPairing(), ComponentKernel((1,), PointKernel((0.25, 0.25))),
                  entity=cell, cell=cell)

    base_dict = base.to_quadrature(1, ())
    wrapped_dict = wrapped.to_quadrature(1, (2,))

    assert list(base_dict.keys()) == list(wrapped_dict.keys())
    for pt in base_dict:
        assert [w for w, _ in base_dict[pt]] == [w for w, _ in wrapped_dict[pt]]
        assert [c for _, c in base_dict[pt]] == [()]
        assert [c for _, c in wrapped_dict[pt]] == [(1,)]


def test_component_kernel_bare_behaviour_preserved():
    bare = ComponentKernel((1,))
    assert bare.permute(None) is bare
    assert bare.degree(3) == 3
    assert repr(bare) == "[(1,)]"


def test_component_kernel_rejects_unsupported_bases():
    """to_quadrature dispatches on kernel type and cannot see through the wrapper."""
    x = sp.Symbol("x")
    for bad in [BarycentricPolynomialKernel(x, symbols=[x]),
                VectorKernel(1),
                ComponentKernel((0,), PointKernel((0.5,)))]:
        with pytest.raises(NotImplementedError):
            ComponentKernel((0,), bad)


def test_component_kernel_comp_is_tuple():
    """A list component would silently trigger fancy indexing in FIAT's to_riesz."""
    assert isinstance(ComponentKernel([0]).comp, tuple)
    assert isinstance(ComponentKernel._from_dict({"comp": [1], "base_kernel": None}).comp, tuple)

