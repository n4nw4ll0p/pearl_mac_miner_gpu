"""GPU per-tile fold kernels."""
from __future__ import annotations

import mlx.core as mx

JACKPOT_SIZE = 16
TH = TW = 16
_PAIR_NSG = 16


import pearlk_rt as _pk
_PRECOMPILED = _pk.available()
if _PRECOMPILED:
    _fold_kernel = _pk.Kernel("fold")
    _pair_kernel = _pk.Kernel("fold_pair")


def fold_cumulative(P, m: int, n: int, nc: int):
    nti, ntj = m // TH, n // TW
    T = nti * ntj
    Pf = P.astype(mx.float32)
    if ntj % 2 == 0:
        dims = mx.array([m, n, nc, ntj, T], dtype=mx.uint32)
        ntg = (T // 2 + _PAIR_NSG - 1) // _PAIR_NSG
        (transcripts,) = _pair_kernel(
            inputs=[Pf, dims],
            grid=(ntg * _PAIR_NSG * 32, 1, 1),
            threadgroup=(_PAIR_NSG * 32, 1, 1),
            output_shapes=[(T, JACKPOT_SIZE)],
            output_dtypes=[mx.uint32],
        )
        return transcripts
    dims = mx.array([m, n, nc, ntj], dtype=mx.uint32)
    (transcripts,) = _fold_kernel(
        inputs=[Pf, dims],
        grid=(T * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(T, JACKPOT_SIZE)],
        output_dtypes=[mx.uint32],
    )
    return transcripts


def transcripts_batched(a_eff, b_eff, r: int):
    """Compute (T, 16) transcripts via batched matmul + per-tile fold (fallback path). Bit-exact with the reference."""
    m, k = a_eff.shape
    n = b_eff.shape[0]
    assert r <= 128, f"rank must be <= 128 for fp32-exact chunk matmul; got {r}"
    assert m % TH == 0 and n % TW == 0, f"m,n must be multiples of {TH}; got m={m} n={n}"
    assert k % r == 0, f"k must be a multiple of rank; got k={k} r={r}"
    nc = k // r
    af = a_eff.astype(mx.float32).reshape(m, nc, r).transpose(1, 0, 2)
    bf = b_eff.astype(mx.float32).reshape(n, nc, r).transpose(1, 2, 0)
    P = mx.matmul(af, bf)
    return fold_cumulative(P, m, n, nc)


