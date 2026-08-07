Mixed elements
==============

A **mixed element** is the Cartesian product of a sequence of element triples
defined on a common cell. It is the object used to discretise systems of
equations in which different fields are approximated in different spaces, for
example the velocity--pressure pair of the Stokes equations (Taylor--Hood,
:math:`[P_2]^d \times P_1`) or the flux--potential pair of mixed Poisson
(:math:`RT_1 \times DG_0`).

Given sub-elements with dual bases
:math:`\{\mathcal{X}^{(k)}_i\}` and value shapes :math:`s_k`, the mixed element
has

* dual basis the **disjoint union** :math:`\bigcup_k \{\mathcal{X}^{(k)}_i\}`
  (the blocks are independent -- there is no coupling between fields at the
  element level),
* value shape the flattened **sum** :math:`\left(\sum_k \prod s_k\right)`, with
  each sub-element occupying its own block of components,
* the pullback of each block taken from that sub-element (identity, covariant or
  contravariant Piola), applied per block.

A mixed element is therefore **not** a Ciarlet element: it has no single
polynomial space and no single nodal basis. In FUSE it is represented by the
:class:`~fuse.mixed.MixedTriple` class rather than by
:class:`~fuse.triples.ElementTriple`.

This is distinct from **enrichment** (the direct sum of spaces sharing the *same*
value shape and mapping, such as the :math:`P_1 \oplus \text{bubble}` velocity of
the MINI element), which is expressed with addition of triples and yields an
ordinary :class:`~fuse.triples.ElementTriple`.

Construction
------------

A mixed element is built from any existing element triples on the same cell.
Here ``velocity`` is a vector-valued :math:`H(\mathrm{div})` triple (a Raviart--Thomas
element) and ``pressure`` is a scalar Lagrange triple, both constructed on the same
cell as shown in :doc:`examples2d`::

    from fuse.mixed import MixedTriple

    mixed = MixedTriple(velocity, pressure)

    mixed.get_value_shape()            # (3,)  = 2 (velocity) + 1 (pressure)
    mixed.num_dofs()                   # sum of the sub-element dof counts

The element converts to UFL/FInAT and FIAT as a single object::

    ufl_element = mixed.to_ufl()       # finat.ufl.MixedElement
    fiat_element = mixed.to_fiat()     # FIAT.mixed.MixedElement

In Firedrake the resulting element defines a mixed function space directly::

    W = FunctionSpace(mesh, mixed.to_ufl())

A block may itself be vector valued. The Taylor--Hood element pairs a vector
Lagrange velocity with a scalar Lagrange pressure, the velocity being obtained by
vectorising a scalar triple with :class:`~fuse.vectorisation.VectorTriple`::

    from fuse.vectorisation import VectorTriple

    taylor_hood = MixedTriple(VectorTriple(cg2), cg1)   # [P2]^d x P1

Each sub-element keeps its own mapping, so the contravariant Piola pullback of the
:math:`RT` velocity and the identity pullback of the pressure are applied per
block. FUSE's custom entity orientation is likewise applied independently to each
block during assembly, so a mixed element built from FUSE sub-elements behaves
exactly as those sub-elements do on their own.

Composability
-------------

A mixed element **can** be combined with:

* ``to_ufl`` / ``to_fiat`` conversion as a unit, and Firedrake assembly,
  interpolation, projection and solves via ``FunctionSpace(mesh, mixed.to_ufl())``;
* any FUSE element as a block -- Lagrange, DG, vector-valued, :math:`RT`,
  Nédélec and BDM elements, and enriched triples formed by adding triples
  (enabling, e.g., the MINI element);
* plotting and introspection of the combined degrees of freedom.

A mixed element **cannot** be:

* treated as a Ciarlet element -- it has no single polynomial space, nodal basis
  or well-defined single degree;
* immersed onto a facet or passed to a trace (:class:`~fuse.traces.TrH1`,
  :class:`~fuse.traces.TrHDiv`, ...): the sub-elements have heterogeneous Sobolev
  spaces and no common trace. Immerse the sub-elements *before* mixing;
* enriched (added) with another element -- enrichment requires an identical value
  shape and mapping;
* used as a factor in a tensor product.

FUSE does not check the well-posedness (inf--sup / LBB condition) of the mixed
system; that remains the responsibility of the user.
