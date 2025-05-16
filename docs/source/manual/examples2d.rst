Examples
===========================

This page will describe the foundational element definitions and how they are written in this software.

DG0  
--------------------------
.. literalinclude:: ../../../test/test_2d_examples_docs.py
   :language: python3
   :dedent:
   :start-after: [test_dg0 0]
   :end-before: [test_dg0 1]


DG1 on interval  
--------------------------

.. literalinclude:: ../../../test/test_2d_examples_docs.py
   :language: python3
   :dedent:
   :start-after: [test_dg1_int 0]
   :end-before: [test_dg1_int 1]


..
     .. plot::

..         from fuse import *
..         edge = Point(1, [Point(0, group=S1), Point(0, group=S1)], vertex_num=2, group=S2)
..         xs = [DOF(DeltaPairing(), PointKernel((-1,)))]
..         dg1 = ElementTriple(edge, (P1, CellL2, C0), DOFGenerator(xs, S2, S1))
        dg1.plot()

 
..    plot:: ../../test/test_2d_examples_docs.py plot_dg1

DG1 on triangle
--------------------------
.. literalinclude:: ../../../test/test_2d_examples_docs.py
   :language: python3
   :dedent:
   :start-after: [test_dg1_tri 0]
   :end-before: [test_dg1_tri 1]

.. plot:: ../../test/test_2d_examples_docs.py plot_dg1_tri

CG1 on interval
--------------------------

.. literalinclude:: ../../../test/test_2d_examples_docs.py
   :language: python3
   :dedent:
   :start-after: [test_cg1 0]
   :end-before: [test_cg1 1]

.. plot:: ../../test/test_2d_examples_docs.py plot_cg1

CG3 on triangle
--------------------------

.. literalinclude:: ../../../test/test_2d_examples_docs.py
   :language: python3
   :dedent:
   :start-after: [test_cg3 0]
   :end-before: [test_cg3 1]

.. plot:: ../../test/test_2d_examples_docs.py plot_cg3