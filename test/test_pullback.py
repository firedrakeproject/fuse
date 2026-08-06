from fuse import *
from fuse.spaces.pullbacks import Pullback
from ufl.sobolevspace import H1, L2 as UFL_L2, HDiv as UFL_HDiv, HCurl as UFL_HCurl
import pytest


@pytest.mark.parametrize("pullback,expected", [
    (Fid, "identity"),
    (Fcurl, "covariant Piola"),
    (Fdiv, "contravariant Piola"),
])
def test_mapping(pullback, expected):
    assert pullback.mapping() == expected


def test_identity_natural_space_depends_on_form_degree():
    # tdim 2: an all-interior (form degree 2) identity element is L2,
    # otherwise H1.
    assert Fid.ufl_sobolev_space(2, 2) == UFL_L2
    assert Fid.ufl_sobolev_space(0, 2) == H1


def test_piola_natural_spaces():
    assert Fcurl.ufl_sobolev_space(1, 3) == UFL_HCurl
    assert Fdiv.ufl_sobolev_space(2, 3) == UFL_HDiv


def test_round_trip():
    for p in (Fid, Fcurl, Fdiv):
        assert Pullback._from_dict(p._to_dict()) is p
