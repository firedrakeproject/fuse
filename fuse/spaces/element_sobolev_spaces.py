"""Element Sobolev spaces module.

This module defines representations of various Sobolev spaces on a single cell,
such as H1, H2, HDiv, HCurl, and L2.
"""

from functools import total_ordering


@total_ordering
class ElementSobolevSpace(object):
    """Representation of a Sobolev space on a single cell.

    Parameters
    ----------
    parents : list
        The parent spaces that contain this space.
    domain : Point, optional
        The cell or domain defined over. Defaults to None.

    Attributes
    ----------
    parents : list
        The parent spaces that contain this space.
    domain : Point or None
        The cell or domain defined over.
    """

    def __init__(self, parents, domain=None):
        """Initialize the ElementSobolevSpace.

        Parameters
        ----------
        parents : list
            The parent spaces that contain this space.
        domain : Point, optional
            The cell or domain defined over. Defaults to None.
        """
        self.domain = domain
        self.parents = parents

    def __lt__(self, other):
        """Check if this space is a proper subset of another space.

        In common with intrinsic Python sets, < indicates "is a proper subset of".

        Parameters
        ----------
        other : ElementSobolevSpace
            The other Sobolev space to compare against.

        Returns
        -------
        bool
            True if this space is a proper subset of `other`, False otherwise.
        """
        return any([isinstance(other, p) for p in self.parents])

    def __eq__(self, other):
        """Check if this space is equal to another space (ignoring domain).

        Parameters
        ----------
        other : object
            The object to compare against.

        Returns
        -------
        bool
            True if the string representations match, False otherwise.
        """
        return repr(self) == repr(other)

    def __hash__(self):
        """Compute the hash value of the space.

        Returns
        -------
        int
            The hash value.
        """
        return hash(("ElementSobolevSpace", str(self)))

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
            The identifier string 'SobolevSpace'.
        """
        return "SobolevSpace"

    @staticmethod
    def _from_dict(obj_dict):
        """Reconstruct the space class from a dictionary representation.

        Parameters
        ----------
        obj_dict : dict
            The dictionary containing space information.

        Returns
        -------
        type
            The corresponding ElementSobolevSpace subclass.
        """
        space_name = obj_dict["space"]
        if space_name == "L2":
            return CellL2
        elif space_name == "H1":
            return CellH1
        elif space_name == "HDiv":
            return CellHDiv
        elif space_name == "HCurl":
            return CellHCurl
        elif space_name == "H2":
            return CellH2


class CellH1(ElementSobolevSpace):
    """Cell H1 Sobolev space representation."""

    def __init__(self, cell):
        """Initialize the H1 space on a cell.

        Parameters
        ----------
        cell : Point
            The cell defined over.
        """
        super(CellH1, self).__init__([CellL2, CellHDiv, CellHCurl], cell)

    def __repr__(self):
        """Get the string representation.

        Returns
        -------
        str
            'H1'
        """
        return "H1"


class CellHDiv(ElementSobolevSpace):
    """Cell HDiv Sobolev space representation."""

    def __init__(self, cell):
        """Initialize the HDiv space on a cell.

        Parameters
        ----------
        cell : Point
            The cell defined over.
        """
        super(CellHDiv, self).__init__([CellL2], cell)

    def __repr__(self):
        """Get the string representation.

        Returns
        -------
        str
            'HDiv'
        """
        return "HDiv"


class CellHCurl(ElementSobolevSpace):
    """Cell HCurl Sobolev space representation."""

    def __init__(self, cell):
        """Initialize the HCurl space on a cell.

        Parameters
        ----------
        cell : Point
            The cell defined over.
        """
        super(CellHCurl, self).__init__([CellL2], cell)

    def __repr__(self):
        """Get the string representation.

        Returns
        -------
        str
            'HCurl'
        """
        return "HCurl"


class CellH2(ElementSobolevSpace):
    """Cell H2 Sobolev space representation."""

    def __init__(self, cell):
        """Initialize the H2 space on a cell.

        Parameters
        ----------
        cell : Point
            The cell defined over.
        """
        super(CellH2, self).__init__([CellL2, CellHDiv, CellHCurl, CellH1], cell)

    def __repr__(self):
        """Get the string representation.

        Returns
        -------
        str
            'H2'
        """
        return "H2"


class CellL2(ElementSobolevSpace):
    """Cell L2 Sobolev space representation."""

    def __init__(self, cell):
        """Initialize the L2 space on a cell.

        Parameters
        ----------
        cell : Point
            The cell defined over.
        """
        super(CellL2, self).__init__([], cell)

    def __repr__(self):
        """Get the string representation.

        Returns
        -------
        str
            'L2'
        """
        return "L2"
