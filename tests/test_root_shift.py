"""Amendment C: the root shift is exact at its anchor, un-shift inverts shift, and the sampler
applies the shift only to root channels and only for chunks with a prefix.

Exactness statement (measured on the whole pair cache, RESULTS.md Phase 1b): in float64 the
round trip unshift(shift(w)) == w is bitwise on every entry; in float32 it is exact to one ulp
(max abs error 2^-24 on values in [-1, 1]) because IEEE subtraction of a shift much larger than
the value cannot be inverted exactly. Both are asserted below as stated."""
import os

import torch

from lattice_smc.data.pairs import CACHE_DIR, PairSet
from lattice_smc.model.window import CHUNK_LEN, ROOT, root_shift, shift_window, unshift_window

B = 64


def windows():
    if os.path.exists(os.path.join(CACHE_DIR, "pairs_train.npz")):
        return PairSet("pairs_train", n=B).motion
    return torch.rand(B, 300, 151) * 2 - 1


def test_anchor_is_exactly_zero_and_uncond_rows_unshifted():
    w = windows()
    uncond = torch.zeros(B, dtype=torch.bool)
    uncond[::3] = True
    s = root_shift(w, uncond)
    rel = shift_window(w, s)
    assert torch.all(rel[~uncond, CHUNK_LEN - 1, ROOT] == 0)
    assert torch.equal(rel[uncond], w[uncond])
    assert torch.equal(rel[..., :4], w[..., :4]) and torch.equal(rel[..., 7:], w[..., 7:])


def test_round_trip_bitwise_in_float64_and_one_ulp_in_float32():
    w = windows()
    s = root_shift(w)
    w64 = w.double()
    assert torch.equal(unshift_window(shift_window(w64, s.double()), s.double()), w64)
    back32 = unshift_window(shift_window(w, s), s)
    assert torch.equal(back32[..., :4], w[..., :4]) and torch.equal(back32[..., 7:], w[..., 7:])
    assert (back32 - w).abs().max().item() <= 2 ** -24
    # and the shift is exactly the identity on a window whose anchor root is already zero
    rel = shift_window(w, s)
    assert torch.equal(shift_window(rel, root_shift(rel)), rel)


def test_shift_then_unshift_of_a_generated_chunk_keeps_the_prefix_frame():
    """The sampler's use: chunk k-1 (absolute) -> shifted prefix; a generated relative chunk whose
    frame 0 root is 0 lands exactly on the prefix's frame-149 root after un-shift."""
    w = windows()
    prev = w[:, CHUNK_LEN:]  # some clean chunk in absolute coordinates
    s = root_shift(prev)
    rel_chunk = torch.zeros(B, CHUNK_LEN, 151)
    out = unshift_window(rel_chunk, s)
    assert torch.equal(out[:, 0, ROOT], prev[:, CHUNK_LEN - 1, ROOT])


def test_disp_reconstruction_bitwise_float64():
    """Amendment E: reconstructing positions from the ground-truth displacements reproduces the
    ground-truth root bitwise in float64 (exact normalized displacements, cumulative sum from the
    clean frame 149); through the metres / s scaling the round trip is exact to float64 rounding;
    the prefix half and the non-root channels are untouched by chunk_to_disp."""
    from lattice_smc.model.window import (DISP_CLIP, chunk_from_disp, chunk_to_disp, clip_x0, positions_from_disp,
                                          positions_from_disp_norm)
    w = windows().double()
    p = w[..., ROOT]
    d_norm = p[:, CHUNK_LEN:] - p[:, CHUNK_LEN - 1:-1]
    rec = positions_from_disp_norm(d_norm, p[:, CHUNK_LEN - 1])
    assert torch.equal(rec, p[:, CHUNK_LEN:])
    disp = chunk_to_disp(w)
    assert torch.equal(disp[:, :CHUNK_LEN], w[:, :CHUNK_LEN])
    assert torch.equal(disp[..., :4], w[..., :4]) and torch.equal(disp[..., 7:], w[..., 7:])
    back = positions_from_disp(disp[:, CHUNK_LEN:, ROOT], p[:, CHUNK_LEN - 1])
    assert torch.allclose(back, p[:, CHUNK_LEN:], rtol=1e-12, atol=1e-14)
    assert torch.allclose(chunk_from_disp(disp[:, CHUNK_LEN:], p[:, CHUNK_LEN - 1]), w[:, CHUNK_LEN:], rtol=1e-12, atol=1e-14)
    # clip: displacement channels to +-5, everything else to [-1, 1]
    x = torch.zeros(2, CHUNK_LEN, 151)
    x[..., 4] = 7.0
    x[..., 9] = 3.0
    c = clip_x0(x, {"repr": "disp"})
    assert c[0, 0, 4] == DISP_CLIP and c[0, 0, 9] == 1.0
    c2 = clip_x0(x, {"repr": "relative"})
    assert c2[0, 0, 4] == 1.0
