"""Re-derive and re-verify a winning draw on the CPU — the consensus authority that gates every share and block."""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

import numpy as np

from . import protocol
from .miner import threads_partition
from .autonomous import tile_index_arrays
from .reference import compute_jackpot
from .streaming import Draw, Winner, snapshot_from_template, build_and_submit, TemplateSnapshot


def _u32x8(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<u4").copy()


def _bound_u32x8(bound: int) -> np.ndarray:
    return np.frombuffer(int(bound).to_bytes(32, "little"), dtype="<u4").copy()


def reconstruct_winner(
    snap: TemplateSnapshot, base_seed: bytes, draw_index: int, tile_index: int,
    a_parts, b_parts, m: int, n: int, k: int, rank: int,
    log: Callable[[str], None] = print,
) -> Optional[Winner]:
    """Regenerate the flagged draw locally and re-verify it. Returns a Winner the
    proof path can consume, or None if the device's hit fails local re-verify."""
    A = protocol.signal_matrix_cheap(base_seed, draw_index, protocol.SIGNAL_LABEL_A, m, k)
    B_t = A
    a_pad = protocol.pad_to_chunk_boundary(A.tobytes())
    b_pad = protocol.pad_to_chunk_boundary(B_t.tobytes())
    dims = (m, n) if getattr(snap, "cert_version", 0) >= 3 else None
    b_seed, a_seed = protocol.compute_commitment_hash(snap.job_key, a_pad, b_pad, dims=dims)
    noise = protocol.compute_noise_for_indices(
        k, rank, (b_seed, a_seed), list(range(m)), list(range(n)),
    )
    a_rows = a_parts[tile_index // len(b_parts)]
    b_cols = b_parts[tile_index % len(b_parts)]
    msg = compute_jackpot(A[a_rows], B_t[b_cols], noise.a[a_rows], noise.b[b_cols], rank)
    jp_hash = protocol.compute_jackpot_hash(msg, a_seed)
    if protocol.hash_to_u256_le(jp_hash) > snap.bound:
        log(f"  device flagged draw={draw_index} tile={tile_index} but local re-verify "
            f"failed (hash > bound); rejecting as false positive")
        return None
    draw = Draw(
        A=A, B_t=B_t, a_padded=a_pad, b_padded=b_pad, a_seed=a_seed,
        a_partitions=[list(a_rows)], b_partitions=[list(b_cols)], batch_start=0, batch_end=1,
    )
    return Winner(snapshot=snap, draw=draw, a_rows=list(a_rows), b_cols=list(b_cols),
                  jackpot_msg=msg, jackpot_hash=jp_hash)


def _tip_changed(rpc, prev_hash: Optional[str],
                 log: Callable[[str], None] = lambda *_: None) -> bool:
    """True if the chain tip moved since this job's template was fetched (so the
    job — and any winner found for it — is now for an orphan tip). Uses the cheap
    `getbestblockhash` (the template's `previousblockhash` IS the tip at fetch
    time) rather than rebuilding a full template every poll. Best-effort: on an RPC
    error, assume unchanged (don't abandon a live job over a transient blip) but LOG
    it — a sustained failure would otherwise silently pin us to a stale tip."""
    if prev_hash is None:
        return False
    try:
        return rpc.call("getbestblockhash", []) != prev_hash
    except Exception as e:
        log(f"  (tip-check RPC failed; assuming tip unchanged: {e})")
        return False


