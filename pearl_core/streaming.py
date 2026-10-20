"""Template-snapshot, draw, and block-submission helpers (solo mining)."""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from . import block_builder
from .miner import threads_partition
from .pearld_rpc import PearldRpc
from .reference import compute_jackpot
from .protocol import (
    MMA_INT7_INT7_INT32,
    IncompleteBlockHeader,
    MiningConfiguration,
    PeriodicPattern,
    compute_commitment_hash,
    compute_jackpot_hash,
    compute_job_key,
    compute_noise_for_indices,
    extract_difficulty_bound,
    hash_to_u256_le,
    pad_to_chunk_boundary,
    signal_draw,
)


# --- Dataclasses --------------------------------------------------------------


@dataclass
class TemplateSnapshot:
    """One generation of pearld state. Immutable once built."""

    generation: int
    template: dict
    header: IncompleteBlockHeader
    config: MiningConfiguration
    rs_config: Any  # pearl_mining.MiningConfiguration
    coinbase_tx: Any
    raw_txns: list[bytes]
    bound: int
    job_key: bytes
    cert_version: int = 0


@dataclass
class Draw:
    """A single random (A, B) draw and its per-partition tile bookkeeping."""

    A: np.ndarray
    B_t: np.ndarray
    a_padded: bytes
    b_padded: bytes
    a_seed: bytes
    a_partitions: list[list[int]]
    b_partitions: list[list[int]]
    batch_start: int   # index of this draw's first tile in the parent Batch
    batch_end: int     # exclusive


@dataclass
class Batch:
    """A unit of work. Tagged with a generation."""

    generation: int
    secret_a: np.ndarray  # (T, h, k) int8
    secret_b: np.ndarray  # (T, w, k) int8
    noise_a: np.ndarray   # (T, h, k) int8
    noise_b: np.ndarray   # (T, w, k) int8
    r: int
    draws: list[Draw]
    # Per-tile keyed-hash key (T, 8) uint32 — each draw's a_seed broadcast across
    # its tiles. Only consumed by the fused dispatch path (compute_winners_batch).
    a_seeds: np.ndarray | None = None


@dataclass
class Result:
    """Search output, tagged with the generation it was prepared for.

    Exactly one of `jackpot_msgs` (non-fused: the full per-tile transcripts) or
    `matches` (fused: a per-tile bool mask, with transcripts kept on-device) is
    set, depending on which dispatch path produced it.
    """

    generation: int
    batch: Batch
    elapsed: float
    jackpot_msgs: np.ndarray | None = None  # (T, 16) uint32 — non-fused path
    matches: np.ndarray | None = None       # (T,) bool — fused path

    @property
    def num_tiles(self) -> int:
        arr = self.matches if self.matches is not None else self.jackpot_msgs
        return 0 if arr is None else int(arr.shape[0])


@dataclass
class Winner:
    """A jackpot hash that beat the bound for some generation."""

    snapshot: TemplateSnapshot
    draw: Draw
    a_rows: list[int]
    b_cols: list[int]
    jackpot_msg: np.ndarray
    jackpot_hash: bytes


# --- Helpers ------------------------------------------------------------------


def _system_ram_bytes() -> int | None:
    """Total physical RAM in bytes, or None if the platform won't report it."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def _cgroup_mem_limit_bytes() -> int | None:
    """The cgroup memory limit (the cap that actually applies inside a container),
    or None if unlimited / unavailable. os.sysconf('SC_PHYS_PAGES') reports HOST
    RAM and ignores this cap, so a worker in a memory-limited container would size
    its budget to half the host and OOM-kill. Checks cgroup v2 then v1."""
    for path in ("/sys/fs/cgroup/memory.max",                    # cgroup v2
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):  # cgroup v1
        try:
            with open(path) as f:
                raw = f.read().strip()
        except OSError:
            continue
        if raw == "max":            # cgroup v2 "unlimited"
            return None
        try:
            val = int(raw)
        except ValueError:
            continue
        # cgroup v1 "unlimited" is a sentinel near INT64_MAX; treat any absurdly
        # large (or non-positive) value as no real limit.
        if val <= 0 or val >= (1 << 62):
            return None
        return val
    return None


def _effective_ram_bytes() -> int | None:
    """min(physical RAM, cgroup limit) — the memory actually available to us."""
    candidates = [x for x in (_system_ram_bytes(), _cgroup_mem_limit_bytes()) if x]
    return min(candidates) if candidates else None


def _template_key(tmpl: dict) -> tuple:
    """Identity of a template w.r.t. mining work.

    Two getblocktemplate responses describe the same work iff they have the
    same chain tip, same difficulty bits, and the same ordered list of txids.
    Comparing full transaction dicts (as the old watcher did) caused spurious
    'new template' decisions on every mempool fee/sigops fluctuation.
    """
    return (
        tmpl.get("previousblockhash"),
        tmpl.get("bits"),
        tuple(tx["txid"] for tx in tmpl.get("transactions", [])),
    )


def build_config(
    common_dim: int, rank: int,
    rows: list[int] | None = None, cols: list[int] | None = None,
) -> tuple[MiningConfiguration, Any]:
    """Rust-verifier-compatible config. `rows`/`cols` are the partition patterns
    (h=len(rows), w=len(cols)); they default to the h=4,w=8 dev tile but the
    controller passes the active preset's patterns (e.g. range(256))."""
    import pearl_mining as rust

    rows = [0, 1, 2, 3] if rows is None else list(rows)
    cols = [0, 1, 2, 3, 4, 5, 6, 7] if cols is None else list(cols)
    py = MiningConfiguration(
        common_dim=common_dim, rank=rank, mma_type=MMA_INT7_INT7_INT32,
        rows_pattern=PeriodicPattern.from_list(rows),
        cols_pattern=PeriodicPattern.from_list(cols),
    )
    ru = rust.MiningConfiguration(
        common_dim=common_dim, rank=rank, mma_type=rust.MMAType.Int7xInt7ToInt32,
        rows_pattern=rust.PeriodicPattern.from_list(rows),
        cols_pattern=rust.PeriodicPattern.from_list(cols),
        reserved=bytes(32),
    )
    return py, ru


def snapshot_from_template(
    tmpl: dict, mining_address: str, generation: int,
    k: int, rank: int,
    rows: list[int] | None = None, cols: list[int] | None = None,
) -> TemplateSnapshot:
    """Build a TemplateSnapshot from a pearld getblocktemplate response."""
    prev_hash = bytes.fromhex(tmpl["previousblockhash"])
    nbits = int(tmpl["bits"], 16)
    timestamp = int(tmpl["curtime"])
    version = int(tmpl["version"])

    coinbase_tx = block_builder.make_coinbase_tx(
        height=int(tmpl["height"]),
        coinbase_value=int(tmpl["coinbasevalue"]),
        mining_address=mining_address,
        coinbase_aux_flags_hex=tmpl.get("coinbaseaux", {}).get("flags"),
        default_witness_commitment_hex=tmpl.get("default_witness_commitment"),
    )
    coinbase_txid = coinbase_tx.get_txid()
    raw_txns = [bytes.fromhex(tx["data"]) for tx in tmpl.get("transactions", [])]
    txids = [coinbase_txid] + [tx["txid"] for tx in tmpl.get("transactions", [])]
    merkle_root = block_builder.compute_merkle_root([bytes.fromhex(t)[::-1] for t in txids])

    header = IncompleteBlockHeader(
        version=version, prev_block=prev_hash, merkle_root=merkle_root,
        timestamp=timestamp, nbits=nbits,
    )
    py_config, rs_config = build_config(k, rank, rows, cols)
    return TemplateSnapshot(
        generation=generation,
        template=tmpl, header=header, config=py_config, rs_config=rs_config,
        coinbase_tx=coinbase_tx, raw_txns=raw_txns,
        bound=extract_difficulty_bound(nbits, py_config),
        job_key=compute_job_key(header, py_config),
        cert_version=3,  # mainnet is on the salted noise-seed (V3) cert since Aug 2026
    )




# --- Pipeline -----------------------------------------------------------------


# --- Finalization -------------------------------------------------------------


def build_and_submit(
    winner: Winner,
    m: int, n: int, k: int, rank: int,
    rpc: PearldRpc,
    log: Callable[[str], None] = print,
) -> tuple[bool, str]:
    """Take a Winner, build the Plonky2 ZK proof, and submitblock to pearld."""
    import pearl_mining as rust

    snap = winner.snapshot
    rs_header = rust.IncompleteBlockHeader(
        snap.header.version, snap.header.prev_block, snap.header.merkle_root,
        snap.header.timestamp, snap.header.nbits,
    )

    # The fast prepare path defers chunk-boundary padding (it's only needed for
    # the winning draw, ~1 in N). Recompute now from the matrices we kept.
    a_padded = winner.draw.a_padded or pad_to_chunk_boundary(winner.draw.A.tobytes())
    b_padded = winner.draw.b_padded or pad_to_chunk_boundary(winner.draw.B_t.tobytes())

    tree_a = rust.MerkleTree(a_padded, snap.job_key)
    leaves_a = rust.MerkleTree.compute_leaf_indices_from_rows(winner.a_rows, (m, k))
    proof_a = rust.MatrixMerkleProof(tree_a.get_multileaf_proof(leaves_a), winner.a_rows)

    tree_b = rust.MerkleTree(b_padded, snap.job_key)
    leaves_b = rust.MerkleTree.compute_leaf_indices_from_rows(winner.b_cols, (n, k))
    proof_b = rust.MatrixMerkleProof(tree_b.get_multileaf_proof(leaves_b), winner.b_cols)

    plain_proof = rust.PlainProof(m, n, k, rank, proof_a, proof_b)
    ok, msg = rust.verify_plain_proof(rs_header, plain_proof)
    if not ok:
        return False, f"verify_plain_proof rejected: {msg}"
    log("  ✓ Rust verify_plain_proof accepted")

    log("→ generating ZK proof (Plonky2)...")
    t0 = time.time()
    zk = rust.generate_proof(rs_header, plain_proof)
    log(f"  ✓ ZK proof generated in {time.time() - t0:.1f}s")
    ok, msg = rust.verify_proof(rs_header, zk)
    if not ok:
        return False, f"verify_proof rejected: {msg}"

    cert_bytes, header_with_commit, header_hash = block_builder.build_zk_cert(
        rs_header.to_bytes(), bytes(zk.public_data), bytes(zk.proof_data),
    )
    coinbase_bytes = snap.coinbase_tx.to_bytes(snap.coinbase_tx.has_segwit)
    block_bytes = block_builder.assemble_block_bytes(
        cert_bytes, header_with_commit, coinbase_bytes, snap.raw_txns,
    )
    log(f"→ submitblock ({len(block_bytes)} bytes)")
    result = rpc.call("submitblock", [block_bytes.hex()])
    if result is None:
        log("  ✓ Block accepted!")
        return True, header_hash.hex()
    return False, f"submitblock rejected: {result}"
