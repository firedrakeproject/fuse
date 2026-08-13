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
        (poly_a, wi_a, pullback_a) = A.spaces
        (poly_b, wi_b, pullback_b) = B.spaces
        if pullback_a != pullback_b:
            raise ValueError("Tensor product factors must share the same pullback.")
        self.spaces = [poly_a if poly_a >= poly_b else poly_b,
                       wi_a if wi_a >= wi_b else wi_b,
                       pullback_a]

        self.DOFGenerator = [A.DOFGenerator, B.DOFGenerator]
        self.cell = TensorProductPoint(A.cell, B.cell)
        self.flat = flat
        self.apply_matrices = False

    def sub_elements(self):
        return [self.A, self.B]

    def __repr__(self):
        return "TensorProd(%s, %s)" % (repr(self.A), repr(self.B))

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
