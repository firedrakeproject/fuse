from fuse.dof import ComponentKernel
from fuse.triples import ElementTriple


class VectorTriple(ElementTriple):
    """A vector valued element formed from a scalar valued one.

    The DOFs of the base element are repeated once per component, each
    restricted to that component, and the polynomial space gains a shape. The
    number of components is the spatial dimension of the cell.
    """

    def __init__(self, base, perm=True):
        if str(base.spaces[1]) in ("HDiv", "HCurl"):
            raise ValueError(
                f"Cannot vectorise an element in {base.spaces[1]}. Its pullback mixes "
                "components, so the components do not transform independently.")
        if base.spaces[0].set_shape:
            raise ValueError("Cannot vectorise an element that is already vector valued.")
        if base.flat:
            raise ValueError("Cannot vectorise a flattened element.")

        # ElementTriple.__init__ is deliberately not called: it adds the cell to
        # each DOFGenerator, which would modify the base element's generators.
        self.base = base
        self.N = base.cell.get_spatial_dimension()
        self.cell = base.cell
        self.spaces = (base.spaces[0].to_vector(), base.spaces[1], base.spaces[2])
        self.DOFGenerator = base.DOFGenerator
        self.flat = False

        self.ref_el = None

        self.dofs = None
        self.dofs = self.generate()
        self.perm = perm

    def generate(self):
        """Clone each DOF once per component, tagging it with a ComponentKernel.

        Builds ``(new_dofs, comp_map)``, where ``comp_map`` sends each new DOF id
        to the ``(original dof id, component)`` it came from.

        Ordering is component innermost, matching the convention of
        ``finat.TensorFiniteElement`` with ``shape_innermost``.
        Each clone keeps the entity of the DOF it came from, so the entity grouping of the original
        element carries over unchanged.
        """
        if self.dofs is None:
            new_dofs = []
            comp_map = {}
            for dof in self.base.generate():
                for c in range(self.N):
                    new_dof = dof.with_kernel(ComponentKernel((c,), base_kernel=dof.kernel))
                    new_dof.id = len(new_dofs)
                    comp_map[new_dof.id] = (dof.id, c)
                    new_dofs.append(new_dof)
            self.dofs, self.comp_map = new_dofs, comp_map
        return self.dofs

    def num_dofs(self):
        return self.N * self.base.num_dofs()

    def sub_elements(self):
        return [self.base]

    def setup_matrices(self):
        raise NotImplementedError(
            "Orientation matrices for a VectorTriple are not implemented yet. They "
            "cannot be produced by make_dof_perms, which sizes its blocks from the "
            "generating groups and so would silently return identity matrices.")

    def to_ufl(self):
        # Guarded here as well as in setup_matrices because ElementTriple.to_ufl
        # assigns ref_el before building the matrices, so failing partway through
        # would leave a half set up element that a second call would accept.
        return self.setup_matrices()

    def __repr__(self):
        return "Vector(%s)" % repr(self.base)
