# FUSE - a unified, serialisable finite element

![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/indiamai/8d09e14999153441dba99d1759e90707/raw/covbadge.json)

FUSE describes a finite element by its cell, its spaces and the symmetries that generate its degrees of freedom.
From that description it derives the orientation matrices, pullback and Sobolev space, so the element can be used directly in Firedrake (through FIAT and FInAT).
Elements can be saved to and loaded from JSON.

```bash
pip install fuse-element
```

Documentation: https://firedrakeproject.github.io/fuse/
