Triples
==============

An element is ``ElementTriple(cell, (V, W_I), generators)``: a cell, the
polynomial space ``V`` with the interpolation space ``W_I``, and a list of
``DOFGenerator``.

The map from the reference cell to a physical cell, and the Sobolev space,
are derived from the kernel actions rather than given:

* ``pullback()`` is the identity for 0-forms, covariant Piola for ``HCurl``,
  contravariant Piola for ``HDiv`` and L2 Piola for top forms. All DOFs must
  agree; mixing pullbacks raises an error.
* ``sobolev_space()`` is the space matching the kernels' form degree k
  (``H1``, ``HCurl`` or ``HDiv``) if the element has DOFs on the
  k-dimensional entities, and ``L2`` otherwise.
