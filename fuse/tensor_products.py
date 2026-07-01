from fuse.triples import ElementTriple
from fuse.cells import TensorProductPoint
from finat.ufl import TensorProductElement, FuseElement


def tensor_product(A, B):
    if not (isinstance(A, ElementTriple) and isinstance(B, ElementTriple)):
        raise ValueError("Both components of Tensor Product need to be a Fuse Triple.")
    return TensorProductTriple(A, B)


class TensorProductTriple(ElementTriple):

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
        return [self.A, self.B]

    def __repr__(self):
        return "TensorProd(%s, %s)" % (repr(self.A), repr(self.B))

    def setup_matrices(self):
        oriented_mats_by_entity, flat_by_entity = self._initialise_entity_dicts(self.A.generate() + self.B.generate())
        breakpoint()
        for dim in range(self.cell.dimension):
            for dimA in range(self.A.cell.dimension):
                pass
            for dimB in range(self.B.cell_dimension):
                pass
        return super().setup_matrices()

    def to_ufl(self):
        if self.flat:
            return FuseElement(self, self.cell.flatten().to_ufl())
        ufl_sub_elements = [e.to_ufl() for e in self.sub_elements()]
        # self.setup_matrices()
        # breakpoint()
        return TensorProductElement(*ufl_sub_elements, cell=self.cell.to_ufl())

    def flatten(self):
        return TensorProductTriple(self.A, self.B, flat=True)

    def unflatten(self):
        return TensorProductTriple(self.A, self.B, flat=False)
