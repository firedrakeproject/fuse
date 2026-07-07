from functools import total_ordering
from typing import Any
@total_ordering
class ElementSobolevSpace(object):
    """
    Representation of a Sobolev space on a single cell

    :param: *underlying_space*: The UFL representation of the Sobolev Space
    :param: *domain*: (Optional) the cell defined over- if not originally provided it should be provided during use.

    """

    def __init__(self, parents, domain=None):
        self.domain: Any = domain
        self.parents: list[type[ElementSobolevSpace]] = parents

    def __lt__(self, other):
        """In common with intrinsic Python sets, < indicates "is a proper subset of"."""
        return any([isinstance(other, p) for p in self.parents])

    def __eq__(self, other):
        """ specifically do not check domain"""
        return repr(self) == repr(other)

    def __hash__(self):
        """Hash."""
        return hash(("ElementSobolevSpace", str(self)))

    def _to_dict(self):
        return {"space": str(self)}

    def dict_id(self):
        return "SobolevSpace"

    @staticmethod
    def _from_dict(obj_dict: dict[str, Any]) -> type["ElementSobolevSpace"] | None:
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
        return None


class CellH1(ElementSobolevSpace):

    def __init__(self, cell):
        super(CellH1, self).__init__([CellL2, CellHDiv, CellHCurl], cell)

    def __repr__(self):
        return "H1"


class CellHDiv(ElementSobolevSpace):

    def __init__(self, cell):
        super(CellHDiv, self).__init__([CellL2], cell)

    def __repr__(self):
        return "HDiv"


class CellHCurl(ElementSobolevSpace):

    def __init__(self, cell):
        super(CellHCurl, self).__init__([CellL2], cell)

    def __repr__(self):
        return "HCurl"


class CellH2(ElementSobolevSpace):

    def __init__(self, cell):
        super(CellH2, self).__init__([CellL2, CellHDiv, CellHCurl, CellH1], cell)

    def __repr__(self):
        return "H2"


class CellL2(ElementSobolevSpace):

    def __init__(self, cell):
        super(CellL2, self).__init__([], cell)

    def __repr__(self):
        return "L2"
