"""Thread-partition helper and mining dataclasses."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .protocol import (
    IncompleteBlockHeader,
    MiningConfiguration,
    compute_commitment_hash,
    compute_jackpot_hash,
    compute_job_key,
    compute_noise_for_indices,
    extract_difficulty_bound,
    hash_to_u256_le,
    pad_to_chunk_boundary,
    signal_draw,
)
from .reference import compute_jackpot as numpy_compute_jackpot


@dataclass
class Solution:
    """A valid PoW solution. Mirrors the essential fields of PlainProof."""

    a_rows_indices: List[int]
    b_cols_indices: List[int]
    jackpot_msg: np.ndarray  # uint32 (16,)
    jackpot_hash: bytes      # 32 bytes
    jackpot_bound: int       # U256
    tile_index: int          # which (a_partition, b_partition) tile satisfied the target
    tiles_tried: int         # total tiles attempted in this attempt


@dataclass
class MiningAttempt:
    """Result of one (A, B) draw — either a Solution or None if no tile beat the target.

    Carries A and B_t so callers (e.g. live_miner building Merkle proofs) don't have
    to replay the rng to recover the winning matrices.
    """

    solution: Solution | None
    tiles_tried: int
    elapsed_seconds: float
    A: np.ndarray | None = None     # (m, k) int8, the matrices drawn for this attempt
    B_t: np.ndarray | None = None   # (n, k) int8


def threads_partition(pattern, total_dim: int) -> List[List[int]]:
    """Public: enumerate every (h,)-shaped tile of indices the pattern partitions
    `total_dim` into. Mirrors mine.rs::threads_partition.
    """
    period = pattern.period()
    if total_dim % period != 0:
        raise ValueError(f"total_dim={total_dim} must be divisible by pattern period={period}")
    base = pattern.to_list()
    return [
        [offset + d for d in base]
        for offset in range(total_dim)
        if pattern.offset_is_valid(offset)
    ]


class SolutionVerificationError(Exception):
    """Raised when a Solution fails to verify against the public inputs."""


