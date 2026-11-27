# Twisted ring orientation tests: handoff

Branch: `indiamai/twisted_mesh` (draft commit `561aa80` "twisted ring draft").
Written 2026-11-27. **None of the ring tests have been run successfully yet.**
That is the first job.

## Goal

Check that FUSE's orientation transforms for first-kind Nédélec (H(curl),
`k=1`) elements, and later H(div) (`k=2`), are correct on a mesh where **no
consistent edge-direction labelling exists**. On a mesh like that, Firedrake
cannot avoid non-trivial entity orientations, including the hex axis-swap
orientations (face keys 4–7). The earlier two-hex tests (`TwoHexMesh` in
`test/test_convert_to_fiat.py`) only ever exercised those orientations one
shared face at a time.

The check is **invariance under local relabelling**: give one cell's vertices
a proper rotation (24 for a hex, 12 for a tet). The physical mesh stays the
same, so the assembled operator must also stay the same, up to a signed
permutation of the global DOFs. Comparing raw matrices is therefore wrong.
Compare the **sorted eigenvalues** of the mass matrix and of the
mass + curl-curl matrix instead.

## Files on this branch

| File | What it is |
|---|---|
| `generate_nonorientable_hex_ring.py` | Builds a 7-hex ring with a half twist (the 3-D counterexample in Agelek, Anderson, Bangerth & Barth, *On orienting edges of unstructured 2-D/3-D meshes*). It includes a parity solver that proves no edge orientation satisfies "parallel reference edges point the same way", and it refuses to write the mesh if the construction ever becomes orientable. Writes Gmsh 2.2. Pure Python with no Firedrake. |
| `nonorientable_hex_ring.msh` | Output of the generator. Running `python generate_nonorientable_hex_ring.py` from the repo root reproduces it byte for byte. The test reads it from the directory the test lives in. |
| `tet_twisted_ring.msh` | 42-tet version of the same ring (28 nodes, the same vertices as the hex ring, 6 tets per hex). It is the control: FIAT's N1curl is known to be correct on tets. The script that generated it was not kept. |
| `test_ring_orientation_spectra.py` | The test harness. It builds meshes directly with `plex_from_cell_list(3, cells, coords, COMM)` and `Mesh(plex, reorder=False, cell_backend=CellBackend.FUSE/FIAT)`, and applies per-cell vertex permutations to the connectivity. |

`test_ring_orientation_spectra.py` contains:

- **The tet backend switch.** Set `TET_BACKEND = "fuse"` or `"fiat"` at the
  top. FUSE uses `periodic_table(0, 3, 1, deg)`; FIAT uses `"N1curl"`.
- **The hex element.** Hexes always use FUSE: `periodic_table(2, 3, 1, deg)`.
- **Tests:**
  - `test_{hex,tet}_ring_baseline` checks that assembly works and the results are finite.
  - `test_tet_ring_one_cell_spectrum_invariance` rotates cell 0 with rotations {0, 1, 2, 5, 8, 9}, at deg 1.
  - `test_hex_ring_one_cell_spectrum_invariance` rotates cell 0 with rotations {0, 1, 4, 8, 16}, at deg 1.
  - `test_hex_ring_axis_swap_spectrum_invariance` uses rotations 4 and 16, at deg 2. These are the axis-swapping ones.
- **Tolerances.** Every test asserts a relative spectrum difference below `1e-11`.

## Environment requirements

- Firedrake, FInAT and FIAT on the branches listed in `.github/branches`.
  Currently those are FIAT `indiamai/new_quads` and Firedrake
  `indiamai/move_matrix_arrays` (base `indiamai/fuse`).
- The test needs `from finat.ufl import CellBackend` and the
  `Mesh(..., cell_backend=...)` keyword. Older checkouts used
  `use_fuse=True` instead.
- Make sure `fuse` imports from **this** working tree. Older Firedrake venvs
  have been known to contain a stale installed copy. Check with
  `python -c "import fuse; print(fuse.__file__)"`.
- The `plex_from_cell_list` signature varies between Firedrake versions. If
  it errors, adjust `_make_mesh` only.

## Suggested plan

1. Check the environment, regenerate the hex mesh, and run the cheap tests first:
   `pytest test_ring_orientation_spectra.py -k baseline -s`
2. Get the tet control passing. Run the tet invariance test with
   `TET_BACKEND = "fiat"`; it should pass, which validates the harness itself.
   Then run it with `"fuse"`. If FIAT fails, the harness is wrong (for
   example the mesh or the permutations), not FUSE.
3. Run the hex tests: first `test_hex_ring_one_cell_spectrum_invariance`, then
   the axis-swap test. Record which rotation indices fail.
   - A useful split is the axis image of each rotation, meaning where it sends
     the reference z edge (vertex 0 → 4). Rotations that keep the z axis lie in
     the D4-style subgroup that the earlier work covered. Rotations that send z
     to x or y exercise the axis-swap keys.
   - The script `test_nonorientable_hex_fuse.py::test_axis_image` (untracked,
     on the old desktop) printed this classification.
4. Widen the coverage once it's green: all 7 cells, all 24 rotations (or a
   seeded random multi-cell sample), degrees 1 to 3, and `k=2` (H(div), where
   the relevant spectrum is mass + div-div).
5. Move the finished tests into `test/` and put the mesh generator somewhere
   sensible, for example a test fixture that generates the mesh into
   `tmp_path`, so no `.msh` needs committing. Mark the expensive cases
   `@pytest.mark.slow`.

## If a hex case fails

Narrow it down with the smallest mesh that reproduces it. The old desktop has
untracked single-hex and two-hex axis-swap scripts
(`test_single_hex_axis_swaps_fuse.py`, `test_two_hex_axis_swap_fuse.py`).
They project the constant field (2, 3, 5) onto N1curl, which should be exact
on affine cubes, using the rotations below:

| Rotation | Permutation | Image of reference z |
|---|---|---|
| identity | `(0,1,2,3,4,5,6,7)` | +z |
| z flip | `(7,6,5,4,3,2,1,0)` | −z |
| z to y | `(4,5,1,0,7,6,2,3)` | +y |
| z to x | `(0,3,7,4,1,2,6,5)` | +x |

Recreating them from that description is quick.

Background from earlier orientation work, which may be stale, so check against
the current code:

- FUSE's hypercube orientation keys must match FIAT/dmcommon's
  `o = 2**d * eo + io`.
- Hex face matrices for the axis-swap keys (4–7) were the last part to be
  filled in.
- Previous DOF-ordering bugs came from the order of the matrices FUSE builds
  differing from Firedrake's packing order. Those bugs stayed hidden at low
  degree, so test deg ≥ 2 as well as deg 1.

## Pitfalls

- The ring is *non-orientable in the edge-direction sense only*. As a solid it
  is orientable, and every cell has a positive Jacobian (the generator reverses
  the end faces to make sure of that).
- The ring cells are trilinear and distorted, so projection errors are **not**
  expected to reach machine precision. Use invariance (spectra), not absolute
  error.
- Don't compare raw global matrices. Relabelling a cell changes the DOF
  numbering and signs, so the matrices differ even when everything is correct.
- The user prefers that expensive or long test sweeps are left to CI. Keep
  local runs small, at deg ≤ 2 with a few rotations.
