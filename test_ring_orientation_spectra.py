"""
Orientation/permutation tests for first-kind Nedelec elements on the
non-orientable tetrahedral and hexahedral rings.

Tetrahedral backend switch
--------------------------
Change only:

    TET_BACKEND = "fuse"
    TET_BACKEND = "fiat"

Tetrahedral Nedelec (k=1):
    FUSE: elem = periodic_table(0, 3, 1, deg)
          V = FunctionSpace(mesh, elem.to_ufl())

    FIAT: V = FunctionSpace(mesh, "N1curl", deg)

The cell backend is selected on Mesh:

    Mesh(..., cell_backend=CellBackend.FUSE)
    Mesh(..., cell_backend=CellBackend.FIAT)

Hex tests always use FUSE:

    elem = periodic_table(2, 3, 1, deg)
    V = FunctionSpace(mesh, elem.to_ufl())
    Mesh(..., cell_backend=CellBackend.FUSE)

The tests compare spectra of assembled operators rather than raw matrices,
because a local cell relabelling can change the matrix representation through
a signed/permuted global DOF basis without changing the underlying operator.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

from firedrake import (
    FunctionSpace,
    TrialFunction,
    TestFunction,
    assemble,
    curl,
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

TET_BACKEND = "fuse"  # change to "fiat"

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

def _tet_space(mesh, deg):
    """
    First-kind Nedelec, k=1.

    FUSE and FIAT are intentionally switched here, while the cell backend
    remains a property of the Mesh object.
    """
    if TET_BACKEND == "fuse":
        elem = periodic_table(0, 3, 1, deg)
        return FunctionSpace(mesh, elem.to_ufl())

    return FunctionSpace(mesh, "N1curl", deg)


def _hex_space(mesh, deg):
    """First-kind Nedelec, k=1, always FUSE."""
    elem = periodic_table(2, 3, 1, deg)
    return FunctionSpace(mesh, elem.to_ufl())


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


# ============================================================================
# OPERATOR ASSEMBLY
# ============================================================================

def _dense_matrix(A):
    """Convert a small PETSc matrix to a dense NumPy matrix."""
    M = A.petscmat.convert("dense")
    return np.array(M.getDenseArray(), copy=True)


def _assemble_mass(V):
    u = TrialFunction(V)
    v = TestFunction(V)
    return _dense_matrix(assemble(inner(u, v) * dx))


def _assemble_hcurl(V):
    u = TrialFunction(V)
    v = TestFunction(V)
    return _dense_matrix(
        assemble(
            inner(u, v) * dx
            + inner(curl(u), curl(v)) * dx
        )
    )


def _spectrum(A):
    """Sorted eigenvalues of the symmetrised matrix."""
    A = 0.5 * (A + A.T)
    return np.sort(np.linalg.eigvalsh(A))


def _relative_difference(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0)


# ============================================================================
# BASELINE TESTS
# ============================================================================

@pytest.mark.parametrize("deg", [1, 2, 3])
def test_hex_ring_baseline(deg):
    mesh = _hex_ring_mesh()
    V = _hex_space(mesh, deg)

    M = _assemble_mass(V)
    A = _assemble_hcurl(V)

    assert M.shape[0] == M.shape[1]
    assert A.shape == M.shape
    assert np.all(np.isfinite(M))
    assert np.all(np.isfinite(A))


@pytest.mark.parametrize("deg", [1, 2, 3])
def test_tet_ring_baseline(deg):
    mesh = _tet_ring_mesh()
    V = _tet_space(mesh, deg)

    M = _assemble_mass(V)
    A = _assemble_hcurl(V)

    assert M.shape[0] == M.shape[1]
    assert A.shape == M.shape
    assert np.all(np.isfinite(M))
    assert np.all(np.isfinite(A))


# ============================================================================
# TETRAHEDRAL RING
# ============================================================================

@pytest.mark.parametrize("cell_index", [0])
@pytest.mark.parametrize("rotation_index", [0, 1, 2, 5, 8, 9])
@pytest.mark.parametrize("deg", [1])
def test_tet_ring_one_cell_spectrum_invariance(
    cell_index, rotation_index, deg
):
    baseline_mesh = _tet_ring_mesh()
    baseline_V = _tet_space(baseline_mesh, deg)

    M0 = _assemble_mass(baseline_V)
    A0 = _assemble_hcurl(baseline_V)

    sM0 = _spectrum(M0)
    sA0 = _spectrum(A0)

    perms = [IDENTITY_TET] * 42
    perms[cell_index] = TET_ROTATIONS[rotation_index]

    rotated_mesh = _tet_ring_mesh(perms)
    rotated_V = _tet_space(rotated_mesh, deg)

    M1 = _assemble_mass(rotated_V)
    A1 = _assemble_hcurl(rotated_V)

    sM1 = _spectrum(M1)
    sA1 = _spectrum(A1)

    dM = _relative_difference(sM1, sM0)
    dA = _relative_difference(sA1, sA0)

    print(
        f"TET backend={TET_BACKEND.upper()}, "
        f"cell={cell_index}, rotation={rotation_index:02d}, "
        f"perm={TET_ROTATIONS[rotation_index]}, "
        f"mass_spectrum_rel={dM:.6e}, "
        f"hcurl_spectrum_rel={dA:.6e}"
    )

    assert dM < 1.0e-11
    assert dA < 1.0e-11


# ============================================================================
# HEXAHEDRAL RING -- FUSE ONLY
# ============================================================================

@pytest.mark.parametrize("cell_index", range(1))
@pytest.mark.parametrize("rotation_index", [0, 1, 4, 8, 16])
@pytest.mark.parametrize("deg", [1])
def test_hex_ring_one_cell_spectrum_invariance(
    cell_index, rotation_index, deg
):
    baseline_mesh = _hex_ring_mesh()
    baseline_V = _hex_space(baseline_mesh, deg)

    M0 = _assemble_mass(baseline_V)
    A0 = _assemble_hcurl(baseline_V)

    sM0 = _spectrum(M0)
    sA0 = _spectrum(A0)

    perms = [IDENTITY_HEX] * 7
    perms[cell_index] = HEX_ROTATIONS[rotation_index]

    rotated_mesh = _hex_ring_mesh(perms)
    rotated_V = _hex_space(rotated_mesh, deg)

    M1 = _assemble_mass(rotated_V)
    A1 = _assemble_hcurl(rotated_V)

    sM1 = _spectrum(M1)
    sA1 = _spectrum(A1)

    dM = _relative_difference(sM1, sM0)
    dA = _relative_difference(sA1, sA0)

    print(
        f"HEX backend=FUSE, "
        f"cell={cell_index}, rotation={rotation_index:02d}, "
        f"perm={HEX_ROTATIONS[rotation_index]}, "
        f"mass_spectrum_rel={dM:.6e}, "
        f"hcurl_spectrum_rel={dA:.6e}"
    )

    assert dM < 1.0e-11
    assert dA < 1.0e-11


@pytest.mark.parametrize("rotation_index", [4, 16])
def test_hex_ring_axis_swap_spectrum_invariance(rotation_index):
    """Focused test for representative cube axis-swapping symmetries."""
    deg = 2

    baseline_mesh = _hex_ring_mesh()
    baseline_V = _hex_space(baseline_mesh, deg)
    A0 = _assemble_hcurl(baseline_V)
    s0 = _spectrum(A0)

    perms = [IDENTITY_HEX] * 7
    perms[0] = HEX_ROTATIONS[rotation_index]

    rotated_mesh = _hex_ring_mesh(perms)
    rotated_V = _hex_space(rotated_mesh, deg)
    A1 = _assemble_hcurl(rotated_V)
    s1 = _spectrum(A1)

    d = _relative_difference(s1, s0)

    print(
        f"HEX axis-swap rotation={rotation_index}, "
        f"perm={HEX_ROTATIONS[rotation_index]}, "
        f"relative spectrum difference={d:.6e}"
    )

    assert d < 1.0e-11
