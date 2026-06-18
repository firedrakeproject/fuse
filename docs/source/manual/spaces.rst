Spaces
===================

FUSE defines various spaces used in the construction of finite elements, including local function spaces and trace spaces.

Interpolation Spaces
---------------------

Interpolation spaces define how functions are represented on the cell.

- :class:`fuse.spaces.interpolation_spaces.Continuous`: Represents continuous function spaces.

Element Sobolev Spaces
------------------------

FUSE supports standard Sobolev spaces on cells:

- :class:`fuse.spaces.element_sobolev_spaces.CellH1`: The :math:`H^1` Sobolev space.
- :class:`fuse.spaces.element_sobolev_spaces.CellHDiv`: The :math:`H(\text{div})` Sobolev space.
- :class:`fuse.spaces.element_sobolev_spaces.CellHCurl`: The :math:`H(\text{curl})` Sobolev space.
- :class:`fuse.spaces.element_sobolev_spaces.CellL2`: The :math:`L^2` space.

Polynomial Spaces
---------------------

Polynomial spaces are used to define the basis functions of the finite element.

- :class:`fuse.spaces.polynomial_spaces.PolynomialSpace`: Represents a general polynomial space.
- :class:`fuse.spaces.polynomial_spaces.ConstructedPolynomialSpace`: A polynomial space constructed from a set of basis functions.
