Groups
============

FUSE uses symmetry groups to explicitly handle orientations and transformations of finite elements. Supported groups include symmetric groups :math:`S_n` and cyclic groups :math:`C_n`.

These groups are used to:
- Map between sub-entities of a cell complex.
- Reconcile orientations across entity boundaries.
- Define transformations for degrees of freedom.

Key Classes
-----------

- :class:`fuse.groups.PermutationSetRepresentation`: Represents a group as a set of permutations.
- :class:`fuse.groups.GroupRepresentation`: A high-level representation of symmetry groups for finite elements.

