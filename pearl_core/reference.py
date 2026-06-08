"""NumPy reference oracle for the jackpot computation."""

from __future__ import annotations

import numpy as np

from .rotate import rotl32_np

JACKPOT_SIZE = 16
LROT_PER_TILE = 13


def compute_jackpot(
    secret_a: np.ndarray,
    secret_b: np.ndarray,
    noise_a: np.ndarray,
    noise_b: np.ndarray,
    r: int,
) -> np.ndarray:
    """Canonical compute_jackpot.

    Args:
        secret_a: int8, shape (h, k).
        secret_b: int8, shape (w, k). (B is stored as B^T — rows index N.)
        noise_a:  int8, shape (h, k).
        noise_b:  int8, shape (w, k).
        r:        chunk size along K. Must divide k.

    Returns:
        jackpot_msg: uint32, shape (JACKPOT_SIZE,).
    """
    assert secret_a.dtype == np.int8 and secret_b.dtype == np.int8
    assert noise_a.dtype == np.int8 and noise_b.dtype == np.int8
    h, k = secret_a.shape
    w, kb = secret_b.shape
    assert k == kb
    assert noise_a.shape == (h, k) and noise_b.shape == (w, k)
    assert k % r == 0, f"k={k} not divisible by r={r}"

    a = secret_a.astype(np.int32) + noise_a.astype(np.int32)
    b = secret_b.astype(np.int32) + noise_b.astype(np.int32)

    jackpot = np.zeros((h, w), dtype=np.int32)
    jackpot_msg = np.zeros(JACKPOT_SIZE, dtype=np.uint32)

    for chunk_idx, ll in enumerate(range(r, k + 1, r)):
        a_chunk = a[:, ll - r : ll]
        b_chunk = b[:, ll - r : ll]
        jackpot += a_chunk @ b_chunk.T

        # ascontiguousarray ensures .view() works even if jackpot ever becomes
        # a non-contiguous slice in the future.
        xored_tile = np.bitwise_xor.reduce(
            np.ascontiguousarray(jackpot).view(np.uint32).reshape(-1)
        )
        tid = (ll // r - 1) % JACKPOT_SIZE
        jackpot_msg[tid] = rotl32_np(jackpot_msg[tid], LROT_PER_TILE) ^ np.uint32(xored_tile)

    return jackpot_msg


def compute_jackpot_pure(
    secret_a: np.ndarray,
    secret_b: np.ndarray,
    noise_a: np.ndarray,
    noise_b: np.ndarray,
    r: int,
) -> np.ndarray:
    """Pure-Python int oracle — no numpy matmul or reductions.

    A genuinely independent reference: every arithmetic step uses Python's
    unbounded ints with explicit masking to i32 / u32. Used in tests to
    cross-check `compute_jackpot` (which goes through numpy semantics).
    """
    h, k = secret_a.shape
    w = secret_b.shape[0]
    jackpot = [[0] * w for _ in range(h)]
    jackpot_msg = [0] * JACKPOT_SIZE
    I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1
    U32_MASK = 0xFFFFFFFF

    def to_u32(x: int) -> int:
        return x & U32_MASK

    def to_i32(x: int) -> int:
        x &= U32_MASK
        return x - (1 << 32) if x >= (1 << 31) else x

    for chunk_idx, ll in enumerate(range(r, k + 1, r)):
        for u in range(h):
            for v in range(w):
                acc = jackpot[u][v]
                for l in range(ll - r, ll):
                    a_eff = int(secret_a[u, l]) + int(noise_a[u, l])
                    b_eff = int(secret_b[v, l]) + int(noise_b[v, l])
                    acc = to_i32(acc + a_eff * b_eff)
                jackpot[u][v] = acc
        xored_tile = 0
        for u in range(h):
            for v in range(w):
                xored_tile ^= to_u32(jackpot[u][v])
        tid = (ll // r - 1) % JACKPOT_SIZE
        rotated = ((jackpot_msg[tid] << LROT_PER_TILE) | (jackpot_msg[tid] >> (32 - LROT_PER_TILE))) & U32_MASK
        jackpot_msg[tid] = rotated ^ xored_tile

    return np.array(jackpot_msg, dtype=np.uint32)
