"""Pin FUSE's interval-product (quad/hex) orientation keys to FIAT/dmcommon.

These are lightweight unit tests (no Firedrake): they check the canonical key
helper in ``fuse.utils`` and the group numbering of flattened quad/hex cells
against FIAT's ``make_entity_permutations_tensorproduct`` and the dmcommon
tensor-product orientation convention ``o = (2**d) * eo + io``.
"""
import itertools
import pytest
import numpy as np
from fuse.cells import line, TensorProductPoint
from fuse.utils import (canonical_tensor_orientation_key,
                        inverse_canonical_tensor_orientation_key)
from FIAT.reference_element import UFCInterval
from FIAT.orientation_utils import make_entity_permutations_tensorproduct


def _fiat_vertex_perm_to_key(d):
    """FIAT vertex-image permutation -> dmcommon integer key for the
    ``d``-fold interval product."""
    o_p_maps = [{0: [0, 1], 1: [1, 0]}] * d
    tuple_perm_map = make_entity_permutations_tensorproduct(
        [UFCInterval()] * d, [1] * d, o_p_maps)
    out = {}
    for tup, vperm in tuple_perm_map.items():
        eo = tup[0]
        io = sum(b * 2 ** (d - 1 - i) for i, b in enumerate(tup[1:]))
        out[tuple(vperm)] = (2 ** d) * eo + io
    return out


@pytest.mark.parametrize("d", [1, 2, 3])
def test_canonical_key_round_trip(d):
    axis_perms = sorted(itertools.permutations(range(d)))
    seen = set()
    for eo, axis_perm in enumerate(axis_perms):
        for io in range(2 ** d):
            flips = tuple((io >> (d - 1 - i)) & 1 for i in range(d))
            key = canonical_tensor_orientation_key(axis_perm, flips, d)
            assert key == (2 ** d) * eo + io
            assert inverse_canonical_tensor_orientation_key(key, d) == (axis_perm, flips)
            seen.add(key)
    assert seen == set(range(2 ** d * len(axis_perms)))


@pytest.mark.parametrize("d", [2, 3])
def test_canonical_key_matches_fiat(d):
    """Every FIAT tuple key (eo, o_1, ..., o_d) flattens to the same integer
    the helper produces from (axis_perm, flips)."""
    axis_perms = sorted(itertools.permutations(range(d)))
    o_p_maps = [{0: [0, 1], 1: [1, 0]}] * d
    tuple_perm_map = make_entity_permutations_tensorproduct(
        [UFCInterval()] * d, [1] * d, o_p_maps)
    for tup in tuple_perm_map:
        eo, flips = tup[0], tup[1:]
        expected = (2 ** d) * eo + sum(b * 2 ** (d - 1 - i) for i, b in enumerate(flips))
        assert canonical_tensor_orientation_key(axis_perms[eo], flips, d) == expected


def test_flattened_quad_keys_match_dmcommon():
    """The flattened quad's 8 group members carry exactly the dmcommon keys
    0..7, matching FIAT identity-to-identity (by vertex-image permutation)."""
    interval = line()
    quad = TensorProductPoint(interval, interval).flatten()
    fiat = _fiat_vertex_perm_to_key(2)
    keys = {}
    for m in quad.group.members():
        keys[tuple(m.array_form)] = m.numeric_rep()
    # Every member's key equals the dmcommon key for its vertex image perm.
    for vperm, key in keys.items():
        assert key == fiat[vperm]
    assert sorted(keys.values()) == list(range(8))
    # Pin the reflection (eo == 0) block against the dmcommon docstring table:
    # identity -> 0, flip y -> 1, flip x -> 2, flip both -> 3.
    assert keys[(0, 1, 2, 3)] == 0
    assert keys[(1, 0, 3, 2)] == 1
    assert keys[(2, 3, 0, 1)] == 2
    assert keys[(3, 2, 1, 0)] == 3


def test_flattened_hex_keys_match_dmcommon():
    """The flattened hex cell group carries exactly dmcommon keys 0..47."""
    interval = line()
    hexf = TensorProductPoint(interval, interval, interval).flatten()
    fiat = _fiat_vertex_perm_to_key(3)
    keys = {}
    for m in hexf.group.members():
        keys[tuple(m.array_form)] = m.numeric_rep()
    for vperm, key in keys.items():
        assert key == fiat[vperm]
    assert sorted(keys.values()) == list(range(48))


def test_flattened_hex_face_keys_match_dmcommon():
    """Each quad face of the hex numbers its own D4 group with dmcommon keys
    0..7, agreeing with FIAT identity-to-identity."""
    interval = line()
    hexf = TensorProductPoint(interval, interval, interval).flatten()
    fiat = _fiat_vertex_perm_to_key(2)
    face_dims = [dt for dt in hexf.all_subpoints if sum(dt) == 2]
    assert face_dims, "expected 2D face sub-entities"
    for dt in face_dims:
        for face in hexf.all_subpoints[dt]:
            keys = {tuple(m.array_form): m.numeric_rep() for m in face.group.members()}
            for vperm, key in keys.items():
                assert key == fiat[vperm]
            assert sorted(keys.values()) == list(range(8))


def test_component_orientations_hit_subentity_numbering():
    """The structural keys emitted by ``component_orientations`` are always
    valid keys of the corresponding flattened sub-entity's group numbering
    (this is what dissolves the historical KeyError entanglement)."""
    interval = line()
    hex_tp = TensorProductPoint(interval, interval, interval)
    hexf = hex_tp.flatten()
    comp = hex_tp.component_orientations()
    for dimtuple, table in comp.items():
        if sum(dimtuple) == 0:
            continue
        subgroup_keys = set()
        for sub in hexf.all_subpoints[dimtuple]:
            subgroup_keys |= {m.numeric_rep() for m in sub.group.members()}
        assert set(table.values()) <= subgroup_keys
