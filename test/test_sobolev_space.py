from fuse import *
from fuse.spaces.element_sobolev_spaces import ElementSobolevSpace
from fuse.spaces.interpolation_spaces import InterpolationSpace
import pytest


def test_comparison():
    cell = Point(0)
    l2 = CellL2(cell)
    h1 = CellH1(cell)
    hdiv = CellHDiv(cell)
    hcurl = CellHCurl(cell)

    assert h1 < l2
    assert hdiv < l2
    assert hcurl < l2
    assert not h1 > hcurl


def test_interpolation_space_round_trip_comparison():
    decoded = InterpolationSpace._from_dict(L2._to_dict())
    assert decoded == L2
    assert decoded != H1


def test_element_sobolev_space_from_dict_rejects_unknown():
    with pytest.raises(ValueError):
        ElementSobolevSpace._from_dict({"space": "not a space"})
