DOFs
==============

A degree of freedom pairs a kernel :math:`\kappa` with the input function
:math:`v` over the entity :math:`E` it is defined on:

.. math::

   \ell(v) = \langle \kappa, v \rangle_{P, E}

In FUSE this is ``DOF(pairing, kernel)``. The pairing is ``DeltaPairing``
(point evaluation) or ``L2Pairing`` (integral moment). The kernel is one of
``PointKernel``, ``VectorKernel``, ``PolynomialKernel`` or
``BarycentricPolynomialKernel``.

Kernel actions
--------------

A kernel's action says how its value transforms under a map of the cell: a
symmetry, or the map immersing an entity into a larger cell. It is given with
the kernel:

.. code-block:: python3

   DOF(L2Pairing(), VectorKernel((1, 0), action=HCurl))

``H1``, ``HCurl``, ``HDiv`` and ``L2`` are shorthands for
:class:`~fuse.actions.FormAction`, which reads the value as a k-form:

.. list-table::
   :header-rows: 1

   * - Shorthand
     - Action
     - Value
   * - ``H1``
     - ``FormAction(0, "scalar")``
     - a scalar function (the default for scalar kernels)
   * - ``HCurl``
     - ``FormAction(1, "flat")``
     - a 1-form; immersed along the entity (tangential)
   * - ``HDiv``
     - ``FormAction(None, "hodge")``
     - a (d-1)-form; immersed across the entity (normal)
   * - ``L2``
     - ``FormAction(None, "scalar")``
     - a top form (a density)

``None`` means the degree is taken from the entity the kernel is defined on.
Other actions can be added by subclassing :class:`~fuse.actions.KernelAction`.

Generation
----------

``DOFGenerator(X, A)`` applies each member of the generation transversal
``A`` to each DOF in ``X``. ``A`` must be a left transversal of the stabiliser
of the span of ``X``. This is not checked on construction; call
``check_transversal()`` to check it.

Immersion
---------

``immerse(cell, triple)`` attaches the DOFs of a triple defined on a
sub-entity to ``cell``. Each DOF is lifted into the cell by its kernel's
action, and the trace applied to the input function is the dual of that lift,
so no trace is passed. Derivative DOFs are the exception: they pass
``TrGrad`` or ``TrHess``.
