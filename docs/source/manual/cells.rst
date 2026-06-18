:tocdepth: 0

Cells
============

A cell complex :math:`C` in FUSE is constructed inductively from 0-cells (points) to :math:`n`-skeletons via attachment maps :math:`\phi^n_\alpha`. This representation is compatible with PETSc's DMPlex, allowing for flexible and robust geometric descriptions of finite elements.

FUSE provides classes to represent different types of cells and their mappings to standard reference elements used in finite element libraries.

Key Classes
-----------

- :class:`fuse.cells.Point`: Represents a cell in the complex.
- :class:`fuse.cells.Edge`: Represents a connection in the complex.
- :class:`fuse.cells.CellComplexToFiatSimplex`: Interface to FIAT's Simplex reference elements.
- :class:`fuse.cells.CellComplexToUFL`: Interface to UFL's Cell descriptions.

