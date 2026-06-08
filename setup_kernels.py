"""GPU per-draw signal and operand-build kernels."""
from __future__ import annotations

import numpy as np
import mlx.core as mx


def signal_fused(base_seed: bytes, draw_index: int, label: int, rows: int, cols: int):
    """Generate one draw's signal matrix (rows x cols, int8) on the GPU."""
    if len(base_seed) != 32:
        raise ValueError(f"base_seed must be 32 bytes, got {len(base_seed)}")
    seed = np.frombuffer(base_seed, dtype="<u4")
    s0, s1 = int(seed[0]), int(seed[1])
    M = 0xFFFFFFFF
    C = (s0 ^ ((draw_index & M) * 0x9E3779B1 & M) ^ ((label & M) * 0x85EBCA77 & M)
         ^ (s1 * 0x27D4EB2F & M)) & M
    dims = mx.array([rows, cols], dtype=mx.uint32)
    consts = mx.array([C], dtype=mx.uint32)
    tx, ty = 64, 4
    gx = ((cols + tx - 1) // tx) * tx
    gy = ((rows + ty - 1) // ty) * ty
    (out,) = _signal_kernel(
        inputs=[dims, consts],
        grid=(gx, gy, 1),
        threadgroup=(tx, ty, 1),
        output_shapes=[(rows, cols)],
        output_dtypes=[mx.int8],
    )
    return out


import pearlk_rt as _pk
_PRECOMPILED = _pk.available()
if _PRECOMPILED:
    _signal_kernel = _pk.Kernel("signal")
    _build_kernel = _pk.Kernel("build_ab")
    _build_kernel_bt = _pk.Kernel("build_ab_t")


_DIMS_CACHE = {}


def build_ab_fused(Ag, e_al, e_br, pa0, pa1, pb0, pb1, k: int, rank: int,
                   b_transposed: bool = False):
    """Build effective operands a_eff/b_eff from the signal (Ag), noise (e_*), and permutation columns (p*). b_transposed returns b_eff in (k, m) layout."""
    m = Ag.shape[0]
    dkey = (m, k, rank)
    dims = _DIMS_CACHE.get(dkey)
    if dims is None:
        dims = mx.array(list(dkey), dtype=mx.uint32); _DIMS_CACHE[dkey] = dims
    tx, ty = 64, 16
    gx = ((k + tx - 1) // tx) * tx
    gy = ((m + ty - 1) // ty) * ty
    a_eff, b_eff = (_build_kernel_bt if b_transposed else _build_kernel)(
        inputs=[Ag, e_al.astype(mx.int32), e_br.astype(mx.int32),
                pa0.astype(mx.uint32), pa1.astype(mx.uint32),
                pb0.astype(mx.uint32), pb1.astype(mx.uint32), dims],
        grid=(gx, gy, 1),
        threadgroup=(tx, ty, 1),
        output_shapes=[(m, k), ((k, m) if b_transposed else (m, k))],
        output_dtypes=[mx.float16, mx.float16],
    )
    return a_eff, b_eff


