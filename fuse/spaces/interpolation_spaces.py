"""Interpolation spaces module.

This module provides symbolic representations of interpolation function spaces
such as Sobolev and Continuous spaces, enabling inclusion testing.
"""

from functools import total_ordering


@total_ordering
class InterpolationSpace(object):
    """Symbolic representation of an interpolation function space.

    This implements a subset of the methods of a Python set so that
    other spaces can be tested for inclusion.

    Parameters
    ----------
    name : str
        The name of this space.
    shape : tuple, optional
        The shape of the space. Defaults to None.
    parents : set or list, optional
        A set of spaces of which this space is a subspace. Defaults to None.

    Attributes
    ----------
    name : str
        The name of this space.
    shape : tuple or None
        The shape of the space.
    parents : frozenset
        A set of spaces of which this space is a subspace.
    """

    def __init__(self, name, shape=None, parents=None):
        """Instantiate an InterpolationSpace object.

        Parameters
        ----------
        name : str
            The name of this space.
        shape : tuple, optional
            The shape of the space. Defaults to None.
        parents : set or list, optional
            A set of spaces of which this space is a subspace. Defaults to None.
        """
        self.name = name
        self.shape = shape
        p = frozenset(parents or [])
        # Ensure that the inclusion operations are transitive.
        self.parents = p.union(*[p_.parents for p_ in p])

    def __str__(self):
        """Format as a string.

        Returns
        -------
        str
            The name of the space.
        """
        return self.name

    def __repr__(self):
        """Representation.

        Returns
        -------
        str
            The string representation of the object.
        """
        return f"InterpolationSpace({self.name!r}, {list(self.parents)!r})"

    def __eq__(self, other):
        """Check equality.

        Parameters
        ----------
        other : object
            The other object to compare.

        Returns
        -------
        bool
            True if equal, False otherwise.
        """
        return isinstance(other, InterpolationSpace) and self.name == other.name and self.shape == other.shape

    def __ne__(self, other):
        """Check inequality.

        Parameters
        ----------
        other : object
            The other object to compare.

        Returns
        -------
        bool
            True if not equal, False otherwise.
        """
        return not self == other

    def __hash__(self):
        """Compute the hash value.

        Returns
        -------
        int
            The hash value.
        """
        return hash(("InterpolationSpace", self.name))

    def __lt__(self, other):
        """Check if this space is a proper subset of another space.

        In common with intrinsic Python sets, < indicates "is a proper subset of".

        Parameters
        ----------
        other : InterpolationSpace
            The other space to check for inclusion.

        Returns
        -------
        bool
            True if this space is a proper subset of `other`, False otherwise.
        """
        return other in self.parents

    def _to_dict(self):
        """Convert the space to a dictionary representation.

        Returns
        -------
        dict
            Dictionary containing the string representation of the space.
        """
        return {"space": str(self)}

    def dict_id(self):
        """Get the dictionary identifier for this space class.

        Returns
        -------
        str
            The identifier string 'InterpolationSpace'.
        """
        return "InterpolationSpace"

    @staticmethod
    def _from_dict(obj_dict):
        """Reconstruct the space class from a dictionary representation.

        Parameters
        ----------
        obj_dict : dict
            The dictionary containing space information.

        Returns
        -------
        InterpolationSpace
            The reconstructed InterpolationSpace object.
        """
        return InterpolationSpace(obj_dict["space"])


class Sobolev(InterpolationSpace):
    """Describes the Sobolev Space W_m,p.

    Parameters
    ----------
    derivatives : int
        The number of derivatives that are required to exist (m).
    lebesgue : int
        The L_p space the derivatives are required to be in (p).
    shape : tuple, optional
        The shape of the space. Defaults to None.
    name : str, optional
        The name of the space. Defaults to None.
    parents : list, optional
        Parent spaces. Defaults to [].

    Attributes
    ----------
    derivatives : int
        The number of derivatives required to exist.
    lebesgue : int
        The Lebesgue integration parameter p.
    shape : tuple or None
        The shape of the space.
    """

    def __init__(self, derivatives, lebesgue, shape=None, name=None, parents=[]):
        """Initialize the Sobolev space.

        Parameters
        ----------
        derivatives : int
            The number of derivatives required to exist (m).
        lebesgue : int
            The L_p space the derivatives are required to be in (p).
        shape : tuple, optional
            The shape of the space. Defaults to None.
        name : str, optional
            The name of the space. Defaults to None.
        parents : list, optional
            Parent spaces. Defaults to [].
        """
        self.derivatives = derivatives
        self.lebesgue = lebesgue
        self.shape = shape

        if name is None:
            if derivatives == 0:
                name = "L_" + str(lebesgue)
            elif lebesgue == 2:
                name = "H_" + str(derivatives)
            else:
                name = "W_" + str(derivatives) + ", " + str(lebesgue)

        super(Sobolev, self).__init__(name, parents)

    def __lt__(self, other):
        """Check if this space is a proper subset of another space.

        In common with intrinsic Python sets, < indicates "is a proper subset of".

        Parameters
        ----------
        other : InterpolationSpace
            The other space to check for inclusion.

        Returns
        -------
        bool
            True if this space is a proper subset of `other`, False otherwise.
        """
        if isinstance(other, Sobolev):
            if self.lebesgue >= other.lebesgue:
                return other.derivatives < self.derivatives or other in self.parents

        return other in self.parents

    def __call__(self, shape):
        """Call to return a new Sobolev space with a given shape.

        Parameters
        ----------
        shape : tuple
            The target shape.

        Returns
        -------
        Sobolev
            The new Sobolev space instance.
        """
        return Sobolev(self.derivatives, self.lebesgue, shape, name=self.name, parents=self.parents)


class Continuous(InterpolationSpace):
    """Describes the continuous space C_n.

    Parameters
    ----------
    derivatives : int
        The number of times the functions can be continuously differentiated (n).
    shape : tuple, optional
        The shape of the space. Defaults to None.
    parents : list, optional
        Parent spaces. Defaults to [].

    Attributes
    ----------
    derivatives : int
        The differentiation degree parameter n.
    shape : tuple or None
        The shape of the space.
    """

    def __init__(self, derivatives, shape=None, parents=[]):
        """Initialize the Continuous space.

        Parameters
        ----------
        derivatives : int
            The number of times the functions can be continuously differentiated (n).
        shape : tuple, optional
            The shape of the space. Defaults to None.
        parents : list, optional
            Parent spaces. Defaults to [].
        """
        self.derivatives = derivatives
        self.shape = shape
        name = "C_" + str(derivatives)
        super(Continuous, self).__init__(name, parents)

    def __lt__(self, other):
        """Check if this space is a proper subset of another space.

        In common with intrinsic Python sets, < indicates "is a proper subset of".

        Parameters
        ----------
        other : InterpolationSpace
            The other space to check for inclusion.

        Returns
        -------
        bool
            True if this space is a proper subset of `other`, False otherwise.
        """
        if isinstance(other, Continuous):
            return other.derivatives > self.derivatives

        return other in self.parents

    def __call__(self, shape):
        """Call to return a new Continuous space with a given shape.

        Parameters
        ----------
        shape : tuple
            The target shape.

        Returns
        -------
        Continuous
            The new Continuous space instance.
        """
        return Continuous(self.derivatives, shape=shape, parents=self.parents)


C0 = Continuous(0)
L2 = Sobolev(0, 2)
H1 = Sobolev(1, 2)
HDiv = Sobolev(0, 2, name="HDiv", parents=[L2])
HCurl = Sobolev(0, 2, name="HCurl", parents=[L2])
