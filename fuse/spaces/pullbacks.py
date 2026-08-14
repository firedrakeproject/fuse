from ufl.sobolevspace import H1, HDiv, HCurl, L2


class Pullback(object):
    """Symbolic representation of a finite element pullback F.

    F is the isomorphism induced by the cell map that transforms degrees of
    freedom between the reference and physical cells.
    """

    name = None

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Pullback) and self.name == other.name

    def __hash__(self):
        return hash(("Pullback", self.name))

    def mapping(self):
        return self.name

    def ufl_sobolev_space(self, form_degree, tdim):
        """The UFL Sobolev space this pullback naturally maps into.
        """
        raise NotImplementedError

    def _to_dict(self):
        return {"pullback": self.name}

    def dict_id(self):
        return "Pullback"

    def _from_dict(obj_dict):
        return pullback_from_name(obj_dict["pullback"])


class IdentityPullback(Pullback):

    name = "identity"

    def ufl_sobolev_space(self, form_degree, tdim):
        return L2 if form_degree == tdim else H1


class CovariantPiola(Pullback):

    name = "covariant Piola"

    def ufl_sobolev_space(self, form_degree, tdim):
        return HCurl


class ContravariantPiola(Pullback):

    name = "contravariant Piola"

    def ufl_sobolev_space(self, form_degree, tdim):
        return HDiv


Fid = IdentityPullback()
Fcurl = CovariantPiola()
Fdiv = ContravariantPiola()


def pullback_from_name(name):
    for p in (Fid, Fcurl, Fdiv):
        if p.name == name:
            return p
    raise ValueError("Pullback not found")
