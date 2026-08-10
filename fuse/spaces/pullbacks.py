import numpy as np
from ufl.sobolevspace import H1, HDiv, HCurl, L2


class Pullback(object):
    """Symbolic representation of a finite element pullback F.

    F is the isomorphism induced by the cell map that transforms degrees of
    freedom between the reference and physical cells. The correct choice
    depends on the form of the DOFs: point/scalar DOFs use the identity,
    tangential DOFs the covariant Piola map, and normal DOFs the
    contravariant Piola map.
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
        """The UFL pullback (mapping) string for this transform."""
        return self.name

    def visualise_dof(self, J, v):
        """Push a reference DOF direction (normal or tangent) onto the physical cell.

        This is used purely to draw DOF glyphs (the normal/tangent arrows in the
        element diagrams): it maps the geometric direction the DOF represents, not
        the basis functions. The two transform oppositely -- a tangent pushes
        forward by J and a normal (conormal) by the inverse transpose -- so H(curl)
        tangential DOFs use J and H(div) normal DOFs use J^{-T}. It is not the UFL
        pullback used for assembly (see ``mapping``).

        :param: J: Jacobian of the reference-to-physical cell map.
        :param: v: reference-cell direction.
        """
        raise NotImplementedError

    def ufl_sobolev_space(self, form_degree, tdim):
        """The UFL Sobolev space this pullback naturally maps into.

        The identity pullback is natural for both H1 and L2; the two are
        distinguished by the form degree, which is derived from where the DOFs
        live (all-interior DOFs give an L2/discontinuous element).
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

    def visualise_dof(self, J, v):
        return np.asarray(v, dtype=float)


class CovariantPiola(Pullback):

    name = "covariant Piola"

    def ufl_sobolev_space(self, form_degree, tdim):
        return HCurl

    def visualise_dof(self, J, v):
        # Tangential (H(curl)) DOF direction: tangents push forward by J.
        J = np.asarray(J, dtype=float)
        return J @ np.asarray(v, dtype=float)


class ContravariantPiola(Pullback):

    name = "contravariant Piola"

    def ufl_sobolev_space(self, form_degree, tdim):
        return HDiv

    def visualise_dof(self, J, v):
        # Normal (H(div)) DOF direction: normals push forward by the inverse transpose.
        J = np.asarray(J, dtype=float)
        return np.linalg.inv(J).T @ np.asarray(v, dtype=float)


Fid = IdentityPullback()
Fcurl = CovariantPiola()
Fdiv = ContravariantPiola()


def pullback_from_name(name):
    for p in (Fid, Fcurl, Fdiv):
        if p.name == name:
            return p
    raise ValueError("Pullback not found")
