Triples
==============

The fundamental object in FUSE is the **Expanded Triple**, which provides a constructive definition for a finite element.

Expanded Triple
---------------

An expanded triple is defined as :math:`(C, \mathcal{U}, \mathcal{E})`, where:
- :math:`C` is the cell complex.
- :math:`\mathcal{U}` is a tuple of spaces :math:`(V, W, W_I)` (e.g., function space, trace space).
- :math:`\mathcal{E}` is the DOF generation description.

DOF Triple
----------

A **DOF Triple** is defined as :math:`(\mathcal{X}, \mathcal{G}_1, \mathcal{G}_2)`, where:
- :math:`\mathcal{X}` is a set of DOF generators.
- :math:`\mathcal{G}_1` is the generator group (varied to generate DOFs).
- :math:`\mathcal{G}_2` is the transformation group.

Key Classes
-----------

- :class:`fuse.triples.ElementTriple`: The main class for defining an expanded triple.
- :class:`fuse.triples.DOFGenerator`: Represents a linear functional used to generate DOFs.
- :class:`fuse.triples.ImmersedDOFs`: Handles the immersion of DOFs from sub-entities into the main element.
