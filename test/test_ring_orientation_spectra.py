"""
Orientation/permutation tests for H(curl) (first-kind Nedelec, k=1) and
H(div) (Raviart-Thomas, k=2) elements on the non-orientable tetrahedral and
hexahedral rings.

Tetrahedral backend switch
--------------------------
Change only:

    TET_BACKEND = "fuse"
    TET_BACKEND = "fiat"

or set the TET_BACKEND environment variable.

Tetrahedral elements:
    FUSE: elem = periodic_table(0, 3, k, deg)
          V = FunctionSpace(mesh, elem.to_ufl())

    FIAT: V = FunctionSpace(mesh, {1: "N1curl", 2: "RT"}[k], deg)

The cell backend is selected on Mesh:

    Mesh(..., cell_backend=CellBackend.FUSE)
    Mesh(..., cell_backend=CellBackend.FIAT)

Hex tests always use FUSE:

    elem = periodic_table(2, 3, k, deg)
    V = FunctionSpace(mesh, elem.to_ufl())
    Mesh(..., cell_backend=CellBackend.FUSE)

The tests compare spectra of assembled operators rather than raw matrices,
because a local cell relabelling can change the matrix representation through
a signed/permuted global DOF basis without changing the underlying operator.
The operators are the mass matrix and mass + curl-curl (k=1) or
mass + div-div (k=2).

Note that Firedrake canonicalises each cell's closure ordering itself, so
relabelling cell c does not change cell c's own entity orientations; it
changes the plex orientation of the entities it shares, and hence the
orientation keys seen by its neighbours.
"""

from __future__ import annotations

import functools
import itertools
import os
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

from firedrake import (
    FunctionSpace,
    TrialFunction,
    TestFunction,
    assemble,
    FacetNormal,
    Function,
    avg,
    cross,
    curl,
    div,
    dS,
    dot,
    dx,
    inner,
)
from firedrake.mesh import Mesh, plex_from_cell_list

from finat.ufl import CellBackend
from fuse import periodic_table


COMM = MPI.COMM_WORLD

# ============================================================================
# ONE-LINE TETRAHEDRAL BACKEND SWITCH
# ============================================================================

TET_BACKEND = os.environ.get("TET_BACKEND", "fuse")  # or "fiat"

assert TET_BACKEND in ("fuse", "fiat")


# ============================================================================
# MESH CONSTRUCTION
# ============================================================================

def _read_gmsh_nodes_and_cells(filename, element_type):
    """Read coordinates and selected Gmsh 2.2 cells."""
    lines = Path(filename).read_text().splitlines()

    inode = lines.index("$Nodes")
    nnode = int(lines[inode + 1])

    coords = []
    for line in lines[inode + 2:inode + 2 + nnode]:
        f = line.split()
        coords.append([float(f[1]), float(f[2]), float(f[3])])

    ielem = lines.index("$Elements")
    nelem = int(lines[ielem + 1])

    cells = []
    for line in lines[ielem + 2:ielem + 2 + nelem]:
        f = [int(x) for x in line.split()]
        if f[1] != element_type:
            continue
        ntags = f[2]
        cells.append([v - 1 for v in f[3 + ntags:]])

    return np.asarray(coords, dtype=np.double), np.asarray(cells, dtype=np.int32)


def _make_mesh(coords, cells, cell_backend):
    """
    Build the DMPlex and select the cell backend on Mesh.

    There is deliberately NO cell_backend argument anywhere near
    FunctionSpace construction.
    """
    plex = plex_from_cell_list(
        3,
        cells,
        coords,
        COMM,
    )
    return Mesh(
        plex,
        reorder=False,
        cell_backend=cell_backend,
    )


GMSH_TO_PETSC_HEX = [0, 3, 2, 1, 4, 5, 6, 7]


def _hex_ring_mesh(cell_perms=None):
    coords, base_cells = _read_gmsh_nodes_and_cells(
        Path(__file__).with_name("nonorientable_hex_ring.msh"),
        5,
    )

    cells = base_cells.copy()

    if cell_perms is not None:
        if len(cell_perms) != len(cells):
            raise ValueError("Wrong number of hex permutations")
        for i, p in enumerate(cell_perms):
            cells[i] = cells[i, list(p)]

    # Rotations are defined in Gmsh vertex ordering; DMPlexCreateFromCellList
    # uses PETSc's ordering, whose top face is reversed relative to Gmsh
    # (v5 above v3, v7 above v1). Passing Gmsh ordering directly yields
    # crossed (zero signed volume) hexes.
    cells = cells[:, GMSH_TO_PETSC_HEX]

    # Hexes always use FUSE.
    return _make_mesh(coords, cells, CellBackend.FUSE)


def _tet_ring_mesh(cell_perms=None):
    coords, base_cells = _read_gmsh_nodes_and_cells(
        Path(__file__).with_name("tet_twisted_ring.msh"),
        4,
    )

    cells = base_cells.copy()

    if cell_perms is not None:
        if len(cell_perms) != len(cells):
            raise ValueError("Wrong number of tet permutations")
        for i, p in enumerate(cell_perms):
            cells[i] = cells[i, list(p)]

    backend = (
        CellBackend.FUSE
        if TET_BACKEND == "fuse"
        else CellBackend.FIAT
    )

    return _make_mesh(coords, cells, backend)


# ============================================================================
# FINITE-ELEMENT SPACES
# ============================================================================

FIAT_TET_FAMILY = {0: "CG", 1: "N1curl", 2: "RT"}


def _tet_space(mesh, k, deg):
    """
    Tet k-form element. FUSE and FIAT are switched here, while the cell
    backend remains a property of the Mesh object.
    """
    if TET_BACKEND == "fuse":
        elem = periodic_table(0, 3, k, deg)
        return FunctionSpace(mesh, elem.to_ufl())

    return FunctionSpace(mesh, FIAT_TET_FAMILY[k], deg)


def _hex_space(mesh, k, deg):
    """Hex k-form element (Q, NCE, NCF for k=0, 1, 2), always FUSE."""
    elem = periodic_table(2, 3, k, deg)
    return FunctionSpace(mesh, elem.to_ufl())


RINGS = {
    "hex": (_hex_ring_mesh, _hex_space),
    "tet": (_tet_ring_mesh, _tet_space),
}


# ============================================================================
# PROPER LOCAL CELL SYMMETRIES
# ============================================================================

_REF_HEX_VERTS = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
], dtype=int)

_HEX_LOOKUP = {tuple(x): i for i, x in enumerate(_REF_HEX_VERTS)}


def _hex_rotations():
    """24 orientation-preserving symmetries of the reference cube."""
    result = []

    for axes in itertools.permutations(range(3)):
        for flips in itertools.product((0, 1), repeat=3):
            p = []

            for x in _REF_HEX_VERTS:
                y = [x[a] for a in axes]
                y = tuple(
                    1 - y[j] if flips[j] else y[j]
                    for j in range(3)
                )
                p.append(_HEX_LOOKUP[y])

            X = _REF_HEX_VERTS[list(p)].astype(float)
            a = X[1] - X[0]
            b = X[3] - X[0]
            c = X[4] - X[0]

            det = round(np.linalg.det(np.column_stack((a, b, c))))
            if det == 1 and tuple(p) not in result:
                result.append(tuple(p))

    assert len(result) == 24
    return tuple(result)


def _tet_rotations():
    """12 orientation-preserving symmetries of the reference tetrahedron."""
    result = []

    for p in itertools.permutations(range(4)):
        inversions = sum(
            p[i] > p[j]
            for i in range(4)
            for j in range(i + 1, 4)
        )
        if inversions % 2 == 0:
            result.append(tuple(p))

    assert len(result) == 12
    return tuple(result)


HEX_ROTATIONS = _hex_rotations()
TET_ROTATIONS = _tet_rotations()

IDENTITY_HEX = tuple(range(8))
IDENTITY_TET = tuple(range(4))


NCELLS = {"hex": 7, "tet": 42}
ROTATIONS = {"hex": HEX_ROTATIONS, "tet": TET_ROTATIONS}


# ============================================================================
# OPERATOR ASSEMBLY
# ============================================================================

def _dense_matrix(A):
    """Convert a small PETSc matrix to a dense NumPy matrix."""
    M = A.petscmat.convert("dense")
    return np.array(M.getDenseArray(), copy=True)


def _assemble_operators(V, k):
    """Mass matrix and the graph-norm operator for the k-form space."""
    u = TrialFunction(V)
    v = TestFunction(V)
    d = curl if k == 1 else div
    M = _dense_matrix(assemble(inner(u, v) * dx))
    A = _dense_matrix(assemble(inner(u, v) * dx + inner(d(u), d(v)) * dx))
    return M, A


def _spectrum(A):
    """Sorted eigenvalues of the symmetrised matrix."""
    A = 0.5 * (A + A.T)
    return np.sort(np.linalg.eigvalsh(A))


def _relative_difference(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0)


def _spectra(kind, k, deg, cell_perms=None):
    make_mesh, make_space = RINGS[kind]
    V = make_space(make_mesh(cell_perms), k, deg)
    M, A = _assemble_operators(V, k)
    assert np.all(np.isfinite(M)) and np.all(np.isfinite(A))
    return _spectrum(M), _spectrum(A)


@functools.lru_cache(maxsize=None)
def _baseline_spectra(kind, k, deg, backend):
    # backend is part of the cache key only.
    return _spectra(kind, k, deg)


def _check_invariance(kind, k, deg, cell_perms, label=""):
    backend = "fuse" if kind == "hex" else TET_BACKEND
    sM0, sA0 = _baseline_spectra(kind, k, deg, backend)
    sM1, sA1 = _spectra(kind, k, deg, cell_perms)

    dM = _relative_difference(sM1, sM0)
    dA = _relative_difference(sA1, sA0)

    print(
        f"{kind.upper()} backend={backend.upper()} k={k} deg={deg} {label} "
        f"mass_spectrum_rel={dM:.3e} graph_spectrum_rel={dA:.3e}"
    )

    assert dM < 1.0e-11
    assert dA < 1.0e-11


def _one_cell_perms(kind, cell_index, rotation_index):
    perms = [ROTATIONS[kind][0]] * NCELLS[kind]
    perms[cell_index] = ROTATIONS[kind][rotation_index]
    return perms


def _random_perms(kind, seed):
    rng = np.random.default_rng(seed)
    rots = ROTATIONS[kind]
    return [rots[i] for i in rng.integers(len(rots), size=NCELLS[kind])]


# ============================================================================
# BASELINE TESTS
# ============================================================================

@pytest.mark.parametrize("kind", ["hex", "tet"])
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("deg", [1, 2, pytest.param(3, marks=pytest.mark.slow)])
def test_ring_baseline(kind, k, deg):
    make_mesh, make_space = RINGS[kind]
    V = make_space(make_mesh(), k, deg)
    M, A = _assemble_operators(V, k)

    assert M.shape[0] == M.shape[1]
    assert A.shape == M.shape
    assert np.all(np.isfinite(M))
    assert np.all(np.isfinite(A))


def test_hex_ring_mesh_is_valid():
    """Every hex has a positive Jacobian (guards the Gmsh->PETSc reorder)."""
    from firedrake import JacobianDeterminant, conditional

    mesh = _hex_ring_mesh()
    J = JacobianDeterminant(mesh)
    vol = assemble(1 * dx(domain=mesh))
    negative = assemble(conditional(J < 0, 1, 0) * dx(domain=mesh))
    assert vol > 20.0
    assert negative == 0.0


def _interior_face_keys(mesh):
    """Orientation keys seen by cells on their interior (shared) faces."""
    closure = np.asarray(mesh.topology.cell_closure)
    orient = np.asarray(mesh.topology.entity_orientations)
    faces = slice(20, 26)  # hex closure: 8 vertices, 12 edges, 6 faces, cell
    shared = np.isin(closure[:, faces],
                     [f for f, n in zip(*np.unique(closure[:, faces],
                                                   return_counts=True))
                      if n == 2])
    return set(orient[:, faces][shared].tolist())


def test_hex_fast_rotations_cover_face_keys():
    """The fast hex rotations exercise every hex face orientation key."""
    seen = set()
    for r in FAST_HEX_ROTATIONS:
        seen |= _interior_face_keys(_hex_ring_mesh(_one_cell_perms("hex", 0, r)))
    assert seen == set(range(8))


# ============================================================================
# FAST INVARIANCE TESTS (cell 0, a few rotations)
# ============================================================================

# Tet rotations cover several vertex 3-cycles and a double transposition.
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("rotation_index", [0, 1, 2, 5, 8, 9])
def test_tet_ring_one_cell_spectrum_invariance(k, rotation_index):
    perms = _one_cell_perms("tet", 0, rotation_index)
    _check_invariance("tet", k, 1, perms,
                      f"cell=0 rotation={rotation_index:02d}")


# Hex rotations by image of the reference z axis:
#   0, 2: +z; 1, 8: -z; 4: +y; 5: -y; 16: +x
# Rotating cell 0 by this set puts every face orientation key 0-7 on an
# interior face somewhere (see test_hex_fast_rotations_cover_face_keys).
FAST_HEX_ROTATIONS = [0, 1, 2, 4, 5, 8, 16]


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("rotation_index", FAST_HEX_ROTATIONS)
def test_hex_ring_one_cell_spectrum_invariance(k, rotation_index):
    perms = _one_cell_perms("hex", 0, rotation_index)
    _check_invariance("hex", k, 1, perms,
                      f"cell=0 rotation={rotation_index:02d}")


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("rotation_index", [4, 5, 16])
def test_hex_ring_axis_swap_spectrum_invariance(k, rotation_index):
    """Axis-swapping cube symmetries at deg 2 (interior face DOFs)."""
    perms = _one_cell_perms("hex", 0, rotation_index)
    _check_invariance("hex", k, 2, perms,
                      f"cell=0 rotation={rotation_index:02d}")


# ============================================================================
# CONFORMITY TESTS (direct, no baseline comparison)
# ============================================================================
# One all-cell relabelling per ring. For the hex ring, seed 7 puts every face
# orientation key 0-7 on an interior face of a single mesh, so each test
# below needs only one mesh.
CONFORMITY_SEED = {"hex": 7, "tet": 0}


def _conformity_mesh(kind):
    return RINGS[kind][0](_random_perms(kind, CONFORMITY_SEED[kind]))


def test_hex_conformity_mesh_covers_face_keys():
    assert _interior_face_keys(_conformity_mesh("hex")) == set(range(8))


@pytest.mark.parametrize("kind", ["hex", "tet"])
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("deg", [1, 2, pytest.param(3, marks=pytest.mark.slow)])
def test_ring_interior_jump_vanishes(kind, k, deg):
    """
    A random function in the assembled space has zero tangential (k=1) or
    normal (k=2) jump across every interior face. A wrong orientation
    transform on any shared entity makes this nonzero.
    """
    V = RINGS[kind][1](_conformity_mesh(kind), k, deg)
    u = Function(V)
    rng = np.random.default_rng(0)
    u.dat.data_wo[...] = rng.standard_normal(u.dat.data_ro.shape)

    n = FacetNormal(V.mesh())
    if k == 1:
        j = cross(n("+"), u("+")) + cross(n("-"), u("-"))
    else:
        j = dot(u("+"), n("+")) + dot(u("-"), n("-"))

    jump2 = assemble(inner(j, j) * dS)
    size2 = assemble(inner(avg(u), avg(u)) * dS)
    rel = np.sqrt(abs(jump2) / size2)
    print(f"{kind.upper()} k={k} deg={deg} relative interior jump={rel:.3e}")
    assert size2 > 0
    assert rel < 1.0e-10


@pytest.mark.parametrize("kind", ["hex", "tet"])
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("deg", [1, pytest.param(2, marks=pytest.mark.slow)])
def test_ring_discrete_kernel_dimension(kind, k, deg):
    """
    Count zero eigenvalues of the curl-curl (k=1) or div-div (k=2) operator
    against the mass matrix. The ring is a solid torus (b0 = b1 = 1,
    b2 = b3 = 0), so for a conforming discrete de Rham complex
        dim ker curl = dim V0 - 1 + b1 = dim V0,
        dim ker div  = dim V2 - dim V3 = dim V1 - dim V0   (Euler char. 0).
    Reuses the mass and graph-norm kernels of the invariance tests.
    """
    from scipy.linalg import eigh

    mesh = _conformity_mesh(kind)
    make_space = RINGS[kind][1]
    V = make_space(mesh, k, deg)
    M, A = _assemble_operators(V, k)
    K = A - M
    lam = eigh(0.5 * (K + K.T), 0.5 * (M + M.T), eigvals_only=True)

    dim0 = make_space(mesh, 0, deg).dim()
    expected = dim0 if k == 1 else make_space(mesh, 1, deg).dim() - dim0

    zero = lam < 1.0e-8 * lam.max()
    nzero = int(zero.sum())
    gap = lam[~zero].min() / lam.max()
    print(f"{kind.upper()} k={k} deg={deg} dim={V.dim()} "
          f"zero eigenvalues={nzero} expected={expected} "
          f"smallest nonzero/max={gap:.3e}")
    assert gap > 1.0e-6
    assert nzero == expected


# ============================================================================
# SLOW SWEEPS
# ============================================================================
# CI runs these too (8 workers, 600 s per-test timeout), so they are kept
# deliberately small: one cell, one random seed, deg 3 on hexes only.

# Rotating cell 0 (which touches the twist seam) through all rotations puts
# every hex face orientation key on an interior face. Firedrake canonicalises
# a cell's own closure, so this exercises the orientations its neighbours see.
@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("rotation_index", range(24))
def test_hex_ring_all_rotations(k, rotation_index):
    perms = _one_cell_perms("hex", 0, rotation_index)
    _check_invariance("hex", k, 1, perms,
                      f"cell=0 rotation={rotation_index:02d}")


@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("rotation_index", range(12))
def test_tet_ring_all_rotations(k, rotation_index):
    perms = _one_cell_perms("tet", 0, rotation_index)
    _check_invariance("tet", k, 1, perms,
                      f"cell=0 rotation={rotation_index:02d}")


@pytest.mark.slow
@pytest.mark.parametrize("kind,deg", [("hex", 1), ("hex", 2), ("hex", 3),
                                      ("tet", 1), ("tet", 2)])
@pytest.mark.parametrize("k", [1, 2])
def test_ring_random_multi_cell(kind, deg, k):
    """Every cell relabelled by a seeded random rotation."""
    seed = CONFORMITY_SEED[kind]
    perms = _random_perms(kind, seed)
    _check_invariance(kind, k, deg, perms, f"seed={seed}")


@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2])
def test_hex_ring_deg3_axis_swap(k):
    perms = _one_cell_perms("hex", 0, 4)
    _check_invariance("hex", k, 3, perms, "cell=0 rotation=04")
