"""Module for handling tensor products of finite element triples."""

from fuse.triples import ElementTriple
from fuse.cells import TensorProductPoint
from finat.ufl import TensorProductElement, FuseElement


def tensor_product(A, B):
    """Compute the tensor product of two element triples.

    Parameters
    ----------
    A : ElementTriple
        The first component element triple.
    B : ElementTriple
        The second component element triple.

    Returns
    -------
    TensorProductTriple
        The resulting tensor product triple.

    Raises
    ------
    ValueError
        If either `A` or `B` is not an instance of `ElementTriple`.
    """
    if not (isinstance(A, ElementTriple) and isinstance(B, ElementTriple)):
        raise ValueError("Both components of Tensor Product need to be a Fuse Triple.")
    return TensorProductTriple(A, B)


class TensorProductTriple(ElementTriple):
    """Represent a tensor product of two element triples.

    Parameters
    ----------
    A : ElementTriple
        The first component element triple.
    B : ElementTriple
        The second component element triple.
    flat : bool, default False
        Whether the resulting tensor product element should be flattened.

    Attributes
    ----------
    A : ElementTriple
        The first component element triple.
    B : ElementTriple
        The second component element triple.
    spaces : list
        List of Sobolev spaces chosen as the component-wise maximum.
    DOFGenerator : list
        List containing the degree of freedom generators of the components.
    cell : TensorProductPoint
        The underlying tensor product cell geometry.
    flat : bool
        Flag indicating if the element is flattened.
    apply_matrices : bool
        Flag indicating whether orientation/transformation matrices should be applied.
    """

    def __init__(self, A, B, flat=False):
        self.A = A
        self.B = B
        self.spaces = []
        for (a, b) in zip(self.A.spaces, self.B.spaces):
            self.spaces.append(a if a >= b else b)

        self.DOFGenerator = [A.DOFGenerator, B.DOFGenerator]
        self.cell = TensorProductPoint(A.cell, B.cell)
        self.flat = flat
        self.apply_matrices = False

    def sub_elements(self):
        """Get the component element triples of the tensor product.

        Returns
        -------
        list of ElementTriple
            The two component element triples `[A, B]`.
        """
        return [self.A, self.B]

    def __repr__(self):
        return "TensorProd(%s, %s)" % (repr(self.A), repr(self.B))

    def setup_matrices(self):
        """Set up transformation/orientation matrices for the tensor product element.

        Returns
        -------
        dict
            The dictionary of entity transformation/orientation matrices.
        """
        oriented_mats_by_entity, flat_by_entity = self._initialise_entity_dicts(self.A.generate() + self.B.generate())
        breakpoint()
        for dim in range(self.cell.dimension):
            for dimA in range(self.A.cell.dimension):
                pass
            for dimB in range(self.B.cell_dimension):
                pass
        return super().setup_matrices()

    def to_ufl(self):
        """Convert the tensor product triple to its corresponding UFL element representation.

        Returns
        -------
        TensorProductElement or FuseElement
            The UFL element object representing this tensor product.
        """
        if self.flat:
            return FuseElement(self, self.cell.flatten().to_ufl())
        ufl_sub_elements = [e.to_ufl() for e in self.sub_elements()]
        # self.setup_matrices()
        # breakpoint()
        return TensorProductElement(*ufl_sub_elements, cell=self.cell.to_ufl())

    def flatten(self):
        """Create a flattened version of this tensor product triple.

        Returns
        -------
        TensorProductTriple
            A new TensorProductTriple instance with `flat=True`.
        """
        return TensorProductTriple(self.A, self.B, flat=True)

    def unflatten(self):
        """Create an unflattened version of this tensor product triple.

        Returns
        -------
        TensorProductTriple
            A new TensorProductTriple instance with `flat=False`.
        """
        return TensorProductTriple(self.A, self.B, flat=False)
