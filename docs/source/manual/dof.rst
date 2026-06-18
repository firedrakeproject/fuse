DOFs
==============
Each degree of freedom :math:`\mathcal{X}_i` can be represented in an integral form:

.. math::

   \mathcal{X}_i (\vec{f}, g) =  \langle  \kappa(x, g),  \vec{f}(x) \rangle_{\mathcal{V}, gE}

.. list-table:: Integral Form of a DOF
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Description
   * - :math:`E`
     - Entity the DOF is defined over
   * - :math:`\vec{f}``
     - Input function to the linear functional, in the space :math:`\mathcal{V}` on :math:`E`
   * - :math:`x`
     - Local coordinate vector on :math:`E`
   * - :math:`\mathcal{V}`
     - Space from the definition triple (:math:`\mathcal{C}`, :math:`\mathcal{V}`, :math:`\mathcal{E}`)
   * - :math:`g`
     - A member of :math:`\mathcal{G}_1` that will be varied to generate the DOFs.
   * - :math:`\kappa` 
     - Kernel - the linear functional that transforms under the group action
   * - :math:`\langle \cdot, \cdot \rangle_{\mathcal{V}, E}`
     - A pairing over the entity that combines the kernel and the function.

The classes documented on this page provide options for the kernel :math:`\kappa` and the pairing :math:`\langle \cdot, \cdot \rangle_{\mathcal{V}, E}`.

Pairing Classes
---------------

- :class:`fuse.dof.DeltaPairing`: Point evaluation pairing.
- :class:`fuse.dof.L2Pairing`: :math:`L^2` inner product pairing.

Kernel Classes
--------------

- :class:`fuse.dof.PointKernel`: A kernel defined at a specific point.
- :class:`fuse.dof.VectorKernel`: A vector-valued kernel.
- :class:`fuse.dof.PolynomialKernel`: A kernel defined as a polynomial.
- :class:`fuse.dof.BarycentricPolynomialKernel`: A kernel defined as a polynomial in barycentric coordinates.

