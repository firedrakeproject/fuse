"""FUSE elements used in DOLFINx function spaces, through the Basix conversion.

The tests in test_convert_to_basix.py reproduce DOLFINx's orientation rules
by hand. These build real DOLFINx function spaces instead, on meshes whose
cells have their vertices shuffled, so that the shared entities are seen with
every relative orientation.
"""
import pytest
import numpy as np
from fuse import polygon, make_tetrahedron
from element_examples import create_cg1, create_cg2_tri, construct_nd2, construct_rt2
from test_2d_examples_docs import construct_cg3, construct_nd, construct_rt


def _triangle_elements():
    return {"cg1": (create_cg1, "Lagrange", 1), "cg2": (create_cg2_tri, "Lagrange", 2),
            "cg3": (construct_cg3, "Lagrange", 3),
            "nd": (construct_nd, "N1curl", 1), "nd2": (construct_nd2, "N1curl", 2),
            "rt": (construct_rt, "RT", 1), "rt2": (construct_rt2, "RT", 2)}


def _tet_elements():
    from element_examples import create_cg1_tet, create_cg2_tet, create_cg3_tet
    from test_3d_examples_docs import (construct_tet_cg4, construct_tet_rt, construct_tet_rt2,
                                       construct_tet_ned, construct_tet_ned2)
    return {"cg1": (create_cg1_tet, "Lagrange", 1), "cg2": (create_cg2_tet, "Lagrange", 2),
            "cg3": (create_cg3_tet, "Lagrange", 3), "cg4": (construct_tet_cg4, "Lagrange", 4),
            "rt": (construct_tet_rt, "RT", 1), "rt2": (construct_tet_rt2, "RT", 2),
            "ned": (construct_tet_ned, "N1curl", 1), "ned2": (construct_tet_ned2, "N1curl", 2)}


def _fuse_element(cell_name, name):
    """The FUSE element, its family name in DOLFINx and its degree."""
    if cell_name == "triangle":
        gen, family, deg = _triangle_elements()[name]
        return gen(polygon(3)), family, deg
    gen, family, deg = _tet_elements()[name]
    try:
        return gen(make_tetrahedron()), family, deg
    except TypeError:  # some constructors build their own cell
        return gen(), family, deg


def _shuffled_mesh(cell_name, n=3, seed=0):
    """A unit square/cube mesh with each cell's vertices randomly reordered."""
    import basix.ufl
    import ufl
    from mpi4py import MPI
    from dolfinx import mesh
    if cell_name == "triangle":
        msh = mesh.create_unit_square(MPI.COMM_SELF, n, n, diagonal=mesh.DiagonalType.crossed)
    else:
        msh = mesh.create_unit_cube(MPI.COMM_SELF, n, n, n)
    x = msh.geometry.x[:, :msh.geometry.dim]
    cells = np.array(msh.geometry.dofmaps[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    cells = np.array([rng.permutation(c) for c in cells])
    coord_el = ufl.Mesh(basix.ufl.element("Lagrange", cell_name, 1, shape=(x.shape[1],)))
    return mesh.create_mesh(MPI.COMM_SELF, cells, coord_el, x)


def _function_space(msh, triple):
    import basix.ufl
    from dolfinx import fem
    return fem.functionspace(msh, basix.ufl.wrap_element(triple.to_basix()))


def _conformity_jump(u, family):
    """The squared jump, integrated over interior facets, of the part of u
    that the element's Sobolev space makes continuous: the value for H1, the
    tangential part for H(curl), the normal part for H(div)."""
    import ufl
    from dolfinx import fem
    n = ufl.FacetNormal(u.function_space.mesh)
    if family == "Lagrange":
        j = ufl.jump(u)
    elif family == "RT":
        j = ufl.jump(u, n)
    elif u.ufl_shape == (2,):
        j = sum(u(s)[0] * n(s)[1] - u(s)[1] * n(s)[0] for s in "+-")
    else:
        j = sum(ufl.cross(u(s), n(s)) for s in "+-")
    return fem.assemble_scalar(fem.form(ufl.inner(j, j) * ufl.dS))


def _random_function(V, seed=1):
    from dolfinx import fem
    u = fem.Function(V)
    u.x.array[:] = np.random.default_rng(seed).standard_normal(u.x.array.size)
    return u


def _polynomial(family, deg, gdim):
    """A function in the element's space: degree deg for Lagrange, vector
    degree deg - 1 (contained in both N1curl and RT of degree deg) otherwise."""
    if family == "Lagrange":
        return lambda x: 1 + x[0] ** deg + 2 * x[1] - x[gdim - 1] ** deg
    return lambda x: np.array([(i + 1) + (deg > 1) * x[(i + 1) % gdim] for i in range(gdim)])


_cases = ([("triangle", name) for name in _triangle_elements()]
          + [("tetrahedron", name) for name in ["cg1", "cg2", "cg3", "cg4", "rt", "rt2", "ned", "ned2"]])


@pytest.mark.skipdolfinx
@pytest.mark.parametrize("cell_name,name", _cases)
def test_dolfinx_function_space(cell_name, name):
    """A function space built from the converted element has as many global
    DOFs as DOLFINx's own element of that family and degree, and an arbitrary
    function in it is conforming across every interior facet."""
    import basix.ufl
    from dolfinx import fem
    triple, family, deg = _fuse_element(cell_name, name)
    msh = _shuffled_mesh(cell_name)
    V = _function_space(msh, triple)

    reference = fem.functionspace(msh, basix.ufl.element(family, cell_name, deg))
    assert V.dofmap.index_map.size_global == reference.dofmap.index_map.size_global
    assert V.dofmap.index_map_bs == reference.dofmap.index_map_bs

    assert _conformity_jump(_random_function(V), family) < 1e-20


@pytest.mark.skipdolfinx
@pytest.mark.parametrize("cell_name,name", _cases)
def test_dolfinx_interpolation_is_exact(cell_name, name):
    """Interpolating a function from the element's space reproduces it."""
    import ufl
    from dolfinx import fem
    triple, family, deg = _fuse_element(cell_name, name)
    msh = _shuffled_mesh(cell_name)
    V = _function_space(msh, triple)
    f = _polynomial(family, deg, msh.geometry.dim)

    u = fem.Function(V)
    u.interpolate(f)
    x = ufl.SpatialCoordinate(msh)
    exact = f(x) if family == "Lagrange" else ufl.as_vector(list(f(x)))
    error = fem.assemble_scalar(fem.form(ufl.inner(u - exact, u - exact) * ufl.dx))
    assert error < 1e-20


@pytest.mark.skipdolfinx
@pytest.mark.parametrize("cell_name", ["triangle", "tetrahedron"])
def test_dolfinx_conformity_check_detects_missing_realignment(monkeypatch, cell_name):
    """Without the realignment of reoriented FUSE entities, DOLFINx builds a
    non-conforming space, so the conformity check above is not vacuous."""
    import fuse_basix.basix_interface as bi
    original = bi.basix_entity_alignment

    def unaligned(triple, cell_type):
        return [[(entity, e_F, entity.group.identity) for entity, e_F, _ in aligned]
                for aligned in original(triple, cell_type)]

    monkeypatch.setattr(bi, "basix_entity_alignment", unaligned)
    triple, family, _ = _fuse_element(cell_name, "cg3")
    V = _function_space(_shuffled_mesh(cell_name), triple)
    assert _conformity_jump(_random_function(V), family) > 1e-2
