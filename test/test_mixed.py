import pytest
import numpy as np
import sympy as sp
from firedrake import *
from fuse import *
from FIAT.mixed import MixedElement as FIATMixedElement
from FIAT.quadrature_schemes import create_quadrature
from test_2d_examples_docs import construct_rt
from test_3d_examples_docs import construct_tet_rt
from test_convert_to_fiat import create_cg1, create_cg2_tri, create_dg1_tet
from fuse.mixed import MixedTriple
from fuse.vectorisation import VectorTriple


def taylor_hood_like(cell):
    """Vector RT velocity block + scalar CG1 pressure block (mixed, block value shape)."""
    return MixedTriple(construct_rt(cell), create_cg1(cell))


def test_value_shape_and_num_dofs():
    cell = polygon(3)
    rt = construct_rt(cell)
    cg1 = create_cg1(cell)
    me = MixedTriple(rt, cg1)

    # value shape is the flattened sum of the sub-element value sizes
    assert me.get_value_shape() == (int(np.prod(rt.get_value_shape())) + 1,)
    assert me.get_value_shape() == (3,)
    assert me.num_dofs() == rt.num_dofs() + cg1.num_dofs()


def test_requires_common_cell():
    tri = polygon(3)
    quad = polygon(4)
    with pytest.raises(ValueError):
        MixedTriple(create_cg1(tri), create_cg1(quad))


def test_entity_ids_offset_concatenation():
    cell = polygon(3)
    rt = construct_rt(cell)
    cg1 = create_cg1(cell)
    me = MixedTriple(rt, cg1)

    flat = [i for dim in me.entity_ids for ent in me.entity_ids[dim]
            for i in me.entity_ids[dim][ent]]
    assert sorted(flat) == list(range(me.num_dofs()))
    # the second block's ids are all offset above the first block's count
    second_block_ids = [i for i in flat if i >= rt.num_dofs()]
    assert len(second_block_ids) == cg1.num_dofs()


def test_to_ufl_is_mixed_with_per_block_pullback():
    cell = polygon(3)
    me = taylor_hood_like(cell)
    ue = me.to_ufl()

    assert type(ue).__name__ == "MixedElement"
    assert ue.reference_value_shape == (3,)
    sub_pullbacks = [type(s.pullback).__name__ for s in ue.sub_elements]
    assert sub_pullbacks == ["ContravariantPiola", "IdentityPullback"]


def test_to_fiat_block_diagonal_tabulation():
    cell = polygon(3)
    rt = construct_rt(cell)
    cg1 = create_cg1(cell)
    me = MixedTriple(rt, cg1)

    fe = me.to_fiat()
    assert isinstance(fe, FIATMixedElement)
    assert fe.value_shape() == (3,)
    assert fe.space_dimension() == 6

    ref = cell.to_fiat()
    pts = create_quadrature(ref, 2).get_points()
    npts = len(pts)

    mixed_tab = fe.tabulate(0, pts)[(0, 0)]
    assert mixed_tab.shape == (6, 3, npts)

    rt_tab = rt.to_fiat().tabulate(0, pts)[(0, 0)].reshape(3, 2, npts)
    cg_tab = cg1.to_fiat().tabulate(0, pts)[(0, 0)].reshape(3, 1, npts)

    # velocity block occupies dof rows 0:3, components 0:2
    assert np.allclose(mixed_tab[0:3, 0:2, :], rt_tab)
    assert np.allclose(mixed_tab[0:3, 2:3, :], 0.0)
    # pressure block occupies dof rows 3:6, component 2
    assert np.allclose(mixed_tab[3:6, 2:3, :], cg_tab)
    assert np.allclose(mixed_tab[3:6, 0:2, :], 0.0)


def _mixed_l2_project(W, gvec, gscal):
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    a = (inner(sigma, tau) + inner(u, v)) * dx
    L = (inner(gvec, tau) + inner(gscal, v)) * dx
    w = Function(W)
    solve(a == L, w, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})
    return w


# The six relative orientations of the two tetrahedra sharing a face.
_TWO_TET_PERMS = [sp.combinatorics.Permutation(p) for p in
                  ([0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2],
                   [0, 1, 3, 2], [0, 3, 2, 1], [0, 2, 1, 3])]


@pytest.mark.parametrize("perm", _TWO_TET_PERMS)
def test_two_tet_mixed_orientation(perm):
    """A mixed RT x DG element must reproduce a field it can represent exactly
    on every relative orientation of the two cells - the decisive check that the
    per-block custom orientation is applied correctly through the mixed element."""
    from firedrake.utility_meshes import TwoTetMesh
    cell = make_tetrahedron()
    me = MixedTriple(construct_tet_rt(cell), create_dg1_tet(cell))

    mesh = TwoTetMesh(perm=perm, use_fuse=True)
    x = SpatialCoordinate(mesh)
    gvec = as_vector([1.0, 2.0, 3.0])    # constant vector, reproduced by RT1
    gscal = x[0]                         # linear scalar, reproduced by DG1

    W = FunctionSpace(mesh, me.to_ufl())
    sigma_h, u_h = _mixed_l2_project(W, gvec, gscal).subfunctions

    assert errornorm(gvec, sigma_h) < 1e-10
    assert errornorm(gscal, u_h) < 1e-10


def taylor_hood(cell):
    """FUSE Taylor-Hood element: vector CG2 velocity x scalar CG1 pressure."""
    return MixedTriple(VectorTriple(create_cg2_tri(cell)), create_cg1(cell))


# Direct solver options robust to the pressure constant nullspace of a Stokes
# saddle-point system (mumps null-pivot detection), so the solve does not depend
# on the fragile default handling of the singular matrix.
_STOKES_PARAMS = {'mat_type': 'aij', 'ksp_type': 'gmres', 'ksp_rtol': 1e-13,
                  'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps',
                  'mat_mumps_icntl_24': 1, 'mat_mumps_icntl_25': 0}


def test_taylor_hood_stokes():
    """Solve Stokes with the FUSE Taylor-Hood mixed element on a manufactured
    solution that lies exactly in the discrete space, and check it is recovered.

    With u = (y, -x) (divergence free, degree 1) and p = x (degree 1), the forcing
    is f = -div(grad(u)) + grad(p) = (1, 0). Taylor-Hood contains this solution
    exactly, so the mixed element must reproduce it to solver tolerance."""
    cell = polygon(3)
    mesh = UnitSquareMesh(4, 4, use_fuse=True)
    x = SpatialCoordinate(mesh)
    u_exact = as_vector([x[1], -x[0]])
    p_exact = x[0]
    f = as_vector([1.0, 0.0])

    W = FunctionSpace(mesh, taylor_hood(cell).to_ufl())
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    a = (inner(grad(u), grad(v)) - p * div(v) - q * div(u)) * dx
    L = inner(f, v) * dx
    bc = DirichletBC(W.sub(0), u_exact, "on_boundary")
    nullspace = MixedVectorSpaceBasis(
        W, [W.sub(0), VectorSpaceBasis(constant=True, comm=W.comm)])
    up = Function(W)
    solve(a == L, up, bcs=bc, nullspace=nullspace, solver_parameters=_STOKES_PARAMS)
    uh, ph = up.subfunctions

    # pressure is determined only up to a constant, so compare with the mean removed
    area = assemble(Constant(1.0) * dx(domain=mesh))
    ph0 = ph - assemble(ph * dx) / area
    pex0 = p_exact - assemble(p_exact * dx(domain=mesh)) / area

    assert sqrt(assemble(inner(uh - u_exact, uh - u_exact) * dx)) < 1e-10
    assert sqrt(assemble((ph0 - pex0) ** 2 * dx)) < 1e-10
