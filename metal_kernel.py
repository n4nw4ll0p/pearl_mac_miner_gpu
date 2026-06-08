"""GPU transcript bound-check and noise kernels."""
from __future__ import annotations

import mlx.core as mx

LROT_PER_TILE = 13
JACKPOT_SIZE = 16
TH = TW = 16
TILE_ELEMS = TH * TW


import pearlk_rt as _pk
_PRECOMPILED = _pk.available()
if _PRECOMPILED:
    _b3_kernel = _pk.Kernel("bound_digest")
    _b3_kernel_mask = _pk.Kernel("bound_mask")
    _noise2_kernel = _pk.Kernel("noise")


def dual_noise(msg_a, msg_b, key_a, key_b):
    """Generate a draw's two noise operand blocks from (msg_a, msg_b, key_a, key_b)."""
    nb = msg_a.shape[0]
    n_msgs = mx.array([nb], dtype=mx.uint32)
    tg = 256 if nb >= 256 else (nb if nb > 0 else 1)
    grid = ((nb + tg - 1) // tg) * tg
    ea, eb = _noise2_kernel(
        inputs=[msg_a.astype(mx.uint32), msg_b.astype(mx.uint32),
                key_a.astype(mx.uint32), key_b.astype(mx.uint32), n_msgs],
        grid=(grid, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(nb, 32), (nb, 32)],
        output_dtypes=[mx.int32, mx.int32],
    )
    return ea, eb


def fused_blake3_bound(transcripts, a_seed_u32_8, bound_u32_8, return_digest: bool = True):
    """Keyed-hash each (T, 16) transcript and compare to `bound` (u32x8). Returns (mask, digest); mask[t] is True iff transcript t is at or below the bound. return_digest=False returns (mask, None)."""
    T = transcripts.shape[0]
    if T == 0:
        return mx.zeros((0,), dtype=mx.bool_), (mx.zeros((0, 8), dtype=mx.uint32) if return_digest else None)
    msg = transcripts.astype(mx.uint32)
    key = a_seed_u32_8.astype(mx.uint32)
    bound = bound_u32_8.astype(mx.uint32)
    n_msgs = mx.array([T], dtype=mx.uint32)
    tg = 256 if T >= 256 else (T if T > 0 else 1)
    grid = ((T + tg - 1) // tg) * tg
    if not return_digest:
        (mask,) = _b3_kernel_mask(
            inputs=[msg, key, bound, n_msgs],
            grid=(grid, 1, 1), threadgroup=(tg, 1, 1),
            output_shapes=[(T,)], output_dtypes=[mx.bool_])
        return mask, None
    digest, mask = _b3_kernel(
        inputs=[msg, key, bound, n_msgs],
        grid=(grid, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(T, 8), (T,)],
        output_dtypes=[mx.uint32, mx.bool_],
    )
    return mask, digest


