"""GPU transcript-kernel dispatch."""
from __future__ import annotations

import os

import mlx.core as mx

from fold_kernel import transcripts_batched

JACKPOT_SIZE = 16
TH = TW = 16


import pearlk_rt as _pk
_PRECOMPILED = _pk.available()


def _spec_name(base, k, r, ntj, nchk, bt):
    return f"{base}_{k}_{r}_{ntj}_{nchk}" + ("_t" if bt else "")


def _transcripts_precompiled(a_eff, b_eff, r: int, b_transposed: bool = False):
    m, k = a_eff.shape
    assert r <= 128, f"rank must be <= 128 for fp32-exact chunk accumulate; got {r}"
    assert k % r == 0, f"k must be a multiple of rank; got k={k} r={r}"
    n = b_eff.shape[1] if b_transposed else b_eff.shape[0]
    assert m % TH == 0 and n % TW == 0, f"m,n must be multiples of {TH}; got m={m} n={n}"
    nti, ntj = m // TH, n // TW
    nchk = k // r
    T = nti * ntj
    if nti % 2 == 0 and ntj % 2 == 0:
        nib, njb = nti // 2, ntj // 2
        sq_pow2 = (nib == njb) and (nib & (nib - 1)) == 0
        base = "transcripts_s" if sq_pow2 else "transcripts_b"
        grid = nib * njb * 32
    elif ntj % 2 == 0:
        base = "transcripts_p"
        grid = nti * (ntj // 2) * 32
    else:
        if b_transposed:
            b_eff = mx.contiguous(b_eff.T)
        return transcripts_batched(a_eff, b_eff, r)
    af = a_eff if a_eff.dtype == mx.float16 else a_eff.astype(mx.float16)
    spec = _spec_name(base, k, r, ntj, nchk, b_transposed)
    if _pk.has(spec):
        bf = b_eff if b_eff.dtype == mx.float16 else b_eff.astype(mx.float16)
        (t,) = _pk.run(spec, [af, bf], [[T, JACKPOT_SIZE]], ["uint32"], [grid, 1, 1], [32, 1, 1])
        return t
    if b_transposed:
        b_eff = mx.contiguous(b_eff.T)
    bf = b_eff if b_eff.dtype == mx.float16 else b_eff.astype(mx.float16)
    dims = mx.array([m, n, k, r, ntj, nchk], dtype=mx.uint32)
    (t,) = _pk.run(base, [af, bf, dims], [[T, JACKPOT_SIZE]], ["uint32"], [grid, 1, 1], [32, 1, 1])
    return t


def transcripts_fused(a_eff, b_eff, r: int, b_transposed: bool = False):
    """Compute one draw's (T, 16) transcripts from float16 operands a_eff/b_eff at rank r. b_transposed: b_eff is in (k, n) layout. Bit-exact with the reference."""
    return _transcripts_precompiled(a_eff, b_eff, r, b_transposed)


