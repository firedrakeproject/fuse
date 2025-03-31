Examples in 3D
===========================

This page will describe the foundational element definitions in three dimensions and how they are written in this software.

Describing a tetrahedron
--------------------------
A tetrahedron is initially set up like this:

.. literalinclude:: ../../test/test_3d_examples_docs.py
    :language: python3
    :dedent:
    :start-after: [make_tet 0]
    :end-before: [make_tet 1]

and then components of the tetrahedron may be extracted using either the helper functions :any:`Point.vertices` and :any:`Point.edges` or the generic function :any:`Point.d_entities`.

.. literalinclude:: ../../test/test_3d_examples_docs.py
    :language: python3
    :dedent:
    :start-after: [make_tet 1]
    :end-before: [make_tet 2]

CG3 on a tetrahedron
--------------------------

.. literalinclude:: ../../test/test_3d_examples_docs.py
    :language: python3
    :dedent:
    :start-after: [test_tet_cg3 0]
    :end-before: [test_tet_cg3 1]

Raviart Thomas on a tetrahedron
------------------------------------

.. literalinclude:: ../../test/test_3d_examples_docs.py
    :language: python3
    :dedent:
    :start-after: [test_tet_rt 0]
    :end-before: [test_tet_rt 1]


Nedelec on a tetrahedron
-----------------------------------

.. literalinclude:: ../../test/test_3d_examples_docs.py
    :language: python3
    :dedent:
    :start-after: [test_tet_ned 0]
    :end-before: [test_tet_ned 1]