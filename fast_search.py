"""GPU search: build a draw's transcripts and bound-check them."""
from __future__ import annotations

import mlx.core as mx

from fused_lp import transcripts_fused
from metal_kernel import fused_blake3_bound


def search_mask_fast(a_eff, b_eff, a_seed_u32_8, bound_u32_8, r: int, b_transposed: bool = False,
                     want_digest: bool = True):
    """Search one draw on the GPU. a_eff/b_eff: effective operands; a_seed_u32_8: commitment key (u32x8); bound_u32_8: difficulty target (u32x8); r: rank. b_transposed: b_eff is in (k,n) layout. Returns (mask, transcripts, digest) where mask[t] is True iff tile t's transcript hashes at or below `bound`; want_digest=False returns digest as None."""
    tr = transcripts_fused(a_eff, b_eff, r, b_transposed=b_transposed)
    mask, dig = fused_blake3_bound(tr, a_seed_u32_8.astype(mx.uint32),
                                   bound_u32_8.astype(mx.uint32), return_digest=want_digest)
    return mask, tr, dig


