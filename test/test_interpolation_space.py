from fuse import *
from fuse.spaces.interpolation_spaces import InterpolationSpace


def test_interpolation_space_round_trip_comparison():
    decoded = InterpolationSpace._from_dict(L2._to_dict())
    assert decoded == L2
    assert decoded != H1
