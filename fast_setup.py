"""GPU per-draw setup: on-device signal, commitment, and noise."""
from __future__ import annotations

import os
import sys

import numpy as np
import mlx.core as mx

from pearl_core import protocol

from metal_kernel import fused_blake3_bound, dual_noise

_DIFF_BOUND = 2 * protocol.ZERO_POINT_TRANSLATION - 1
assert _DIFF_BOUND < 128, "uniform-random range no longer fits int8; GPU wrap needed"

_RANGE_MASK = int(protocol.RANGE_MASK)
_ZERO_PT = int(protocol.ZERO_POINT_TRANSLATION)
try:
    from blake3 import blake3 as _blake3_mod
    _COMMIT_THREADS = int(os.environ["PEARL_COMMIT_THREADS"]) if "PEARL_COMMIT_THREADS" in os.environ else _blake3_mod.AUTO
except Exception:
    _COMMIT_THREADS = -1


def _u32(x):
    return mx.array(np.uint32(x), dtype=mx.uint32)


def _u32x8_le(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<u4").copy()


def commitment_one_tree(job_key: bytes, a_pad: bytes, commit_threads=None):
    from blake3 import blake3
    mt = _COMMIT_THREADS if commit_threads is None else commit_threads
    h = blake3(a_pad, key=job_key, max_threads=mt).digest()
    b_noise_seed = protocol.blake3_digest(job_key + h, key=None)
    a_noise_seed = protocol.blake3_digest(b_noise_seed + h, key=None)
    return b_noise_seed, a_noise_seed


def signal_gpu(base_seed: bytes, draw_index: int, label: int, rows: int, cols: int):
    if len(base_seed) != 32:
        raise ValueError(f"base_seed must be 32 bytes, got {len(base_seed)}")
    seed = np.frombuffer(base_seed, dtype="<u4")
    s0, s1 = int(seed[0]), int(seed[1])
    di = draw_index & 0xFFFFFFFF
    lab = label & 0xFFFFFFFF
    i = mx.arange(rows, dtype=mx.uint32).reshape(rows, 1)
    j = mx.arange(cols, dtype=mx.uint32).reshape(1, cols)
    v = (_u32(s0) ^ (_u32(di) * _u32(0x9E3779B1)) ^ (_u32(lab) * _u32(0x85EBCA77))
         ^ (i * _u32(0xC2B2AE3D)) ^ (j * _u32(0x165667B1)) ^ (_u32(s1) * _u32(0x27D4EB2F)))
    v = v ^ (v >> _u32(15))
    v = v * _u32(0x2C1B3C6D)
    v = v ^ (v >> _u32(13))
    v = v * _u32(0x297A2D39)
    v = v ^ (v >> _u32(16))
    return (v & _u32(0x7F)).astype(mx.int32) - 64


_SHIFTS = mx.array(np.array([0, 8, 16, 24], dtype=np.uint32))
_ZERO8 = mx.zeros(8, dtype=mx.uint32)
_MSG_CACHE = {}


def _noise_message(seed_label: bytes, nb: int, prepend_index: int = 0):
    mkey = (seed_label, nb, prepend_index)
    msg = _MSG_CACHE.get(mkey)
    if msg is None:
        seed_u32 = np.frombuffer(seed_label, dtype="<u4")
        idx = mx.arange(nb, dtype=mx.uint32) + _u32(1)
        cols = []
        for w in range(16):
            if w == prepend_index:
                cols.append(idx)
            elif 8 <= w < 16:
                cols.append(mx.broadcast_to(_u32(int(seed_u32[w - 8])), (nb,)))
            else:
                cols.append(mx.zeros((nb,), dtype=mx.uint32))
        msg = mx.stack(cols, axis=1)
        mx.eval(msg)
        _MSG_CACHE[mkey] = msg
    return msg


def uniform_random_gpu(seed_label: bytes, key: bytes, num_rows: int, num_cols: int,
                       prepend_index: int = 0):
    total = num_rows * num_cols
    nb = (total + 31) // 32
    msg = _noise_message(seed_label, nb, prepend_index)
    _, dig = fused_blake3_bound(msg, mx.array(_u32x8_le(key)), _ZERO8)
    b = (dig[:, :, None] >> _SHIFTS[None, None, :]) & _u32(_RANGE_MASK)
    vals = b.reshape(nb * 32).astype(mx.int32) - _ZERO_PT
    return vals[:total].reshape(num_rows, num_cols)


def uniform_random_gpu_dual(label_a: bytes, label_b: bytes, key_a: bytes, key_b: bytes,
                            num_rows: int, num_cols: int, prepend_index: int = 0):
    total = num_rows * num_cols
    nb = (total + 31) // 32
    msg_a = _noise_message(label_a, nb, prepend_index)
    msg_b = _noise_message(label_b, nb, prepend_index)
    ka = mx.array(_u32x8_le(key_a)); kb = mx.array(_u32x8_le(key_b))
    ea, eb = dual_noise(msg_a, msg_b, ka, kb)
    ea = ea.reshape(nb * 32)[:total].reshape(num_rows, num_cols)
    eb = eb.reshape(nb * 32)[:total].reshape(num_rows, num_cols)
    return ea, eb


def setup_symmetric_gpu(base_seed: bytes, draw_index: int, job_key: bytes,
                        m: int, k: int, rank: int):
    """Build one draw's operands on the GPU from (base_seed, draw_index, job_key) at shape (m, k, rank): returns (a_eff, b_eff, a_seed) bit-exact with the CPU setup. a_seed is the commitment key the search hashes with."""
    Ag = signal_gpu(base_seed, draw_index, protocol.SIGNAL_LABEL_A, m, k)
    A_i8 = np.array(Ag.astype(mx.int8))
    a_pad = protocol.pad_to_chunk_boundary(A_i8.tobytes())
    b_seed, a_seed = commitment_one_tree(job_key, a_pad)

    e_al = uniform_random_gpu(protocol.SEED_LABEL_A, a_seed, m, rank)
    e_br = uniform_random_gpu(protocol.SEED_LABEL_B, b_seed, m, rank)
    e_ar_t = protocol.generate_permutation_matrix(protocol.SEED_LABEL_A, a_seed, k, rank)
    e_bl = protocol.generate_permutation_matrix(protocol.SEED_LABEL_B, b_seed, k, rank)
    pa0 = mx.array(e_ar_t[:, 0].astype(np.uint32))
    pa1 = mx.array(e_ar_t[:, 1].astype(np.uint32))
    pb0 = mx.array(e_bl[:, 0].astype(np.uint32))
    pb1 = mx.array(e_bl[:, 1].astype(np.uint32))

    a_eff = Ag + mx.take(e_al, pa0, axis=1) - mx.take(e_al, pa1, axis=1)
    b_eff = Ag + mx.take(e_br, pb0, axis=1) - mx.take(e_br, pb1, axis=1)
    return a_eff, b_eff, _u32x8_le(a_seed)


