"""Overlapped GPU search pipeline (CPU commitment alongside GPU search)."""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import mlx.core as mx

from pearl_core import protocol

from fast_search import search_mask_fast
from fast_setup import (
    uniform_random_gpu_dual, commitment_one_tree, _u32x8_le,
)
from setup_kernels import signal_fused, build_ab_fused


def search_draws_overlapped(base_seed: bytes, job_key: bytes, m: int, k: int,
                            rank: int, bound_u32x8, draw_indices, cert_version: int = 0):
    """Yield (draw_index, winning_tile_indices) for each index in `draw_indices`, overlapping the next draw's CPU commitment with the current draw's GPU search. Winners are identical to the non-overlapped path. bound_u32x8 is the difficulty target; m/k/rank the draw shape. cert_version >= 3 selects the V3 salted noise seeds."""
    bound = mx.array(bound_u32x8)
    idxs = list(draw_indices)
    if not idxs:
        return

    N = len(idxs)
    DEPTH = 2
    dims = (m, m) if cert_version >= 3 else None

    def stage_S(n):
        Ag = signal_fused(base_seed, n, protocol.SIGNAL_LABEL_A, m, k)
        mx.async_eval(Ag)
        return Ag

    def stage_C(Ag):
        mx.eval(Ag)
        A_view = np.frombuffer(memoryview(Ag), dtype=np.int8)
        a_pad = (A_view if A_view.nbytes % protocol.BLAKE3_CHUNK_LEN == 0
                 else protocol.pad_to_chunk_boundary(bytes(A_view)))
        b_seed, a_seed = commitment_one_tree(job_key, a_pad, dims=dims)
        e_ar = protocol.generate_permutation_matrix(protocol.SEED_LABEL_A, a_seed, k, rank)
        e_bl = protocol.generate_permutation_matrix(protocol.SEED_LABEL_B, b_seed, k, rank)
        return b_seed, a_seed, e_ar, e_bl

    def stage_G(Ag, cpu):
        b_seed, a_seed, e_ar, e_bl = cpu
        e_al, e_br = uniform_random_gpu_dual(protocol.SEED_LABEL_A, protocol.SEED_LABEL_B,
                                             a_seed, b_seed, m, rank)
        pm = mx.array(np.stack([e_ar[:, 0], e_ar[:, 1], e_bl[:, 0], e_bl[:, 1]], axis=0).astype(np.uint32))
        a_eff, b_eff = build_ab_fused(Ag, e_al, e_br, pm[0], pm[1], pm[2], pm[3], k, rank,
                                      b_transposed=True)
        mask, _, _ = search_mask_fast(a_eff, b_eff, mx.array(_u32x8_le(a_seed)), bound, rank,
                                      b_transposed=True, want_digest=False)
        cnt = mask.sum()
        mx.async_eval(cnt)
        return mask, cnt

    with ThreadPoolExecutor(max_workers=1) as pool:
        sig = {}
        cfut = {}

        def ensure_S(i):
            if 0 <= i < N and i not in sig:
                sig[i] = stage_S(idxs[i])

        def ensure_C(i):
            if 0 <= i < N and i not in cfut:
                ensure_S(i)
                cfut[i] = pool.submit(stage_C, sig[i])

        pending = []

        def issue_G(i):
            ensure_C(i)
            mask, cnt = stage_G(sig[i], cfut[i].result())
            pending.append((i, mask, cnt))
            ensure_S(i + DEPTH + 1); ensure_C(i + DEPTH)

        for j in range(min(DEPTH + 1, N)):
            ensure_C(j)
        ensure_S(min(DEPTH + 1, N))

        issued = 0
        while issued < N and len(pending) < DEPTH:
            issue_G(issued); issued += 1
        for _ in range(N):
            if issued < N:
                issue_G(issued); issued += 1
            i, mask, cnt = pending.pop(0)
            c = int(cnt.item())
            winners = [] if c == 0 else [int(t) for t in np.nonzero(np.array(mask))[0]]
            sig.pop(i, None); cfut.pop(i, None)
            yield idxs[i], winners


