# Changelog

## Unreleased

### Breaking changes

- `DOFGenerator(X, A)` takes two arguments. The third group (G2) is removed, and the attribute `g1` is renamed `transversal`.
- `ElementTriple(cell, (V, W_I), dofs)` takes two spaces. The pullback is no longer given: `Fid`, `Fcurl`, `Fdiv` and `fuse.spaces.pullbacks` are removed. Use `ElementTriple.pullback()` and `ElementTriple.sobolev_space()`, which are derived from the kernel actions.
- Kernels declare how they transform with `action=` (`H1`, `HCurl`, `HDiv`, `L2` or a `KernelAction`). A vector kernel without an action is treated as `HCurl`, with a `DeprecationWarning`.
- `immerse(cell, triple)` no longer takes `TrH1`, `TrHCurl` or `TrHDiv`, which are removed. The trace comes from each kernel's action. `TrGrad` and `TrHess` are still passed.
- `DOF.eval(pullback=...)` is renamed `DOF.eval(trace=...)`.
- The serialisation format has changed. Elements saved by earlier versions cannot be loaded.

`python scripts/migrate_api.py PATH ...` rewrites most of these. Run it with `--check` to list the changes without writing them.

### Behaviour changes

- Orientation matrices are built from the DOF generators' symmetries, and key 0 is always the identity. On tetrahedron faces generated with an S3 transversal, the global DOFs are in a different basis, so checkpoints of those spaces will not load identically.
- Some elements' DOFs are reordered or change sign: BDM on the triangle (degree 3 and above), the second-kind Nédélec faces on the tetrahedron, and several cell interiors. Each DOF set is the same up to order and sign.
- Crouzeix–Raviart reports `L2` as its Sobolev space (previously `H1`).
