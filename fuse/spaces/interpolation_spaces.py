from functools import total_ordering


@total_ordering
class InterpolationSpace(object):
    """Symbolic representation of an interpolation function space

    This implements a subset of the methods of a Python set so that
    other spaces can be tested for inclusion.
    """

    # Subclasses set an instance attribute that shadows this. Deserialised
    # spaces are built from the base class, so the default keeps __eq__ total.
    shape = None

    def __init__(self, name, shape=None, parents=None):
        """Instantiate a InterpolationSpace object.

        Args:
            name: The name of this space,
            parents: A set of spaces of which this
            space is a subspace.
        """
        self.name = name
        p = frozenset(parents or [])
        # Ensure that the inclusion operations are transitive.
        self.parents = p.union(*[p_.parents for p_ in p])

    def __str__(self):
        """Format as a string."""
        return self.name

    def __repr__(self):
        """Representation."""
        return f"InterpolationSpace({self.name!r}, {list(self.parents)!r})"

    def __eq__(self, other):
        """Check equality."""
        return isinstance(other, InterpolationSpace) and self.name == other.name and self.shape == other.shape

    def __ne__(self, other):
        """Not equal."""
        return not self == other

    def __hash__(self):
        """Hash."""
        return hash(("InterpolationSpace", self.name))

    def __lt__(self, other):
        """In common with intrinsic Python sets, < indicates "is a proper subset of"."""
        return other in self.parents

    def _to_dict(self):
        return {"space": str(self)}

    def dict_id(self):
        return "InterpolationSpace"

    def _from_dict(obj_dict):
        return InterpolationSpace(obj_dict["space"])


class Sobolev(InterpolationSpace):
    """
    Describes the Sobolev Space W_m,p

    Arguments:
    - Derivatives (m): the numbers of derivatives that are required to exist
    - Lebesgue (p): The L_p space the derivatives are required to be in
    """

    def __init__(self, derivatives, lebesgue, shape=None, name=None, parents=[]):
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
        """In common with intrinsic Python sets, < indicates "is a proper subset of"."""

        if isinstance(other, Sobolev):
            if self.lebesgue >= other.lebesgue:
                return other.derivatives < self.derivatives or other in self.parents

        return other in self.parents

    def __call__(self, shape):
        return Sobolev(self.derivatives, self.lebesgue, shape, name=self.name, parents=self.parents)


class Continuous(InterpolationSpace):
    """
    Describes the continuous space C_n

    Arguments:
    - Derivatives(n): the number of times the functions can be continously differentiated
    """

    def __init__(self, derivatives, shape=None, parents=[]):
        self.derivatives = derivatives
        self.shape = shape
        name = "C_" + str(derivatives)
        super(Continuous, self).__init__(name, parents)

    def __lt__(self, other):
        """In common with intrinsic Python sets, < indicates "is a proper subset of"."""

        if isinstance(other, Continuous):
            return other.derivatives > self.derivatives

        return other in self.parents

    def __call__(self, shape):
        return Continuous(self.derivatives, shape=shape, parents=self.parents)


C0 = Continuous(0)
L2 = Sobolev(0, 2)
H1 = Sobolev(1, 2)
HDiv = Sobolev(0, 2, name="HDiv", parents=[L2])
HCurl = Sobolev(0, 2, name="HCurl", parents=[L2])

# Want to have
# c_0
# c_n
# w_m,p st mp < 1

# h_n ~ w_n,2
# l_2 ~ h_0
# h_1

# inclusions
# c_0
