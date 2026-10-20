"""Pearl protocol reference: signal, commitment, noise, and jackpot hashing."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from blake3 import blake3

# --- Constants ----------------------------------------------------------------

JACKPOT_SIZE = 16
LROT_PER_TILE = 13

# Noise constants (from pearl_noise.rs)
NOISE_RANGE = 128
IDXS_PER_COL = 2
UNIFORM_NOISE_RANGE = NOISE_RANGE // IDXS_PER_COL  # 64
ZERO_POINT_TRANSLATION = UNIFORM_NOISE_RANGE // 2  # 32
RANGE_MASK = UNIFORM_NOISE_RANGE - 1               # 0x3F
BLAKE3_DIGEST_SIZE = 32
BLAKE3_CHUNK_LEN = 1024

SEED_LABEL_A = b"A_tensor" + b"\x00" * 24
SEED_LABEL_B = b"B_tensor" + b"\x00" * 24

MMA_INT7_INT7_INT32 = 0

# --- Random signal matrices ---------------------------------------------------


def signal_draw(rng: np.random.Generator, shape: Tuple[int, ...]) -> np.ndarray:
    """Draw a random int8 matrix with values in the int7 signal range.

    Returns values in [-64, 63] (one less than the reference's [-64, 64] but
    consensus-equivalent: the protocol only requires uniform random matrices
    in this range — it doesn't pin specific values from a particular RNG).

    ~10× faster than rng.integers(-64, 65, size=shape, dtype=int8) because
    np.frombuffer(rng.bytes(N), dtype=int8) is a near-memcpy from the PCG64
    state, with the >> 1 reduction folded in. Use this helper everywhere
    (miner, streaming pipeline, verifier) to keep replay invariants.
    """
    n = 1
    for d in shape:
        n *= d
    raw = np.frombuffer(rng.bytes(n), dtype=np.int8)
    # Arithmetic right-shift on int8: maps [-128,127] → [-64,63], uniformly.
    return (raw >> 1).reshape(shape).copy()


# Label bytes distinguishing the A and B streams in seed-based generation.
SIGNAL_LABEL_A = 0
SIGNAL_LABEL_B = 1


def signal_matrix_from_seed(
    base_seed: bytes, draw_index: int, label: int, rows: int, cols: int
) -> np.ndarray:
    """Deterministic int8 signal matrix in [-64, 63] from (base_seed, draw_index, label).

    The miner generates A/B from a compact seed (so the
    controller uploads no matrices), and the controller regenerates the WINNING
    draw's A/B with this exact function to build the ZK proof. A/B are the miner's
    free choice (not consensus-pinned), so the only requirement is that this matches
    the reference implementation byte-for-byte.

    Scheme: a stream of single-block keyed BLAKE3 hashes (key=base_seed), block i's
    64-byte message = [u32 i | u32 draw_index | u32 label | zeros | base_seed], whose
    bytes (reinterpreted int8, arithmetic >>1) give the signal range.
    """
    if len(base_seed) != 32:
        raise ValueError(f"base_seed must be 32 bytes, got {len(base_seed)}")
    n = rows * cols
    n_blocks = (n + BLAKE3_DIGEST_SIZE - 1) // BLAKE3_DIGEST_SIZE
    out = bytearray()
    msg = bytearray(64)
    msg[32:64] = base_seed
    msg[4:8] = (draw_index & 0xFFFFFFFF).to_bytes(4, "little")
    msg[8:12] = (label & 0xFFFFFFFF).to_bytes(4, "little")
    for i in range(n_blocks):
        msg[0:4] = i.to_bytes(4, "little")
        out += blake3_digest(bytes(msg), key=base_seed)
    raw = np.frombuffer(bytes(out)[:n], dtype=np.int8)
    return (raw >> 1).reshape(rows, cols)


def signal_matrix_cheap(
    base_seed: bytes, draw_index: int, label: int, rows: int, cols: int
) -> np.ndarray:
    """Cheap int7 signal matrix in [-64, 63] — numpy twin of
    the reference implementation (MUST be bit-identical; the controller
    regenerates the winning draw with this).

    A/B values are consensus-free (circuit only range-checks int7 [-64,64] and
    verifies commitment + jackpot over them — no seed regeneration), so we skip
    the signal BLAKE3 PRNG entirely and generate A directly, row-major, with a
    counter-based mix. The per-draw commitment + the jackpot BLAKE3 supply the
    search entropy; A only needs to vary per draw and be a valid int7 matrix.
    All arithmetic is uint32 (wraps).
    """
    if len(base_seed) != 32:
        raise ValueError(f"base_seed must be 32 bytes, got {len(base_seed)}")
    seed = np.frombuffer(base_seed, dtype="<u4")
    s0 = np.uint32(seed[0]); s1 = np.uint32(seed[1])
    di = np.uint32(draw_index & 0xFFFFFFFF); lab = np.uint32(label & 0xFFFFFFFF)
    i = np.arange(rows, dtype=np.uint32)[:, None]
    j = np.arange(cols, dtype=np.uint32)[None, :]
    with np.errstate(over="ignore"):
        v = (s0 ^ (di * np.uint32(0x9E3779B1)) ^ (lab * np.uint32(0x85EBCA77))
             ^ (i * np.uint32(0xC2B2AE3D)) ^ (j * np.uint32(0x165667B1)) ^ (s1 * np.uint32(0x27D4EB2F)))
        v = v ^ (v >> np.uint32(15))
        v = v * np.uint32(0x2C1B3C6D)
        v = v ^ (v >> np.uint32(13))
        v = v * np.uint32(0x297A2D39)
        v = v ^ (v >> np.uint32(16))
    return ((v & np.uint32(0x7F)).astype(np.int8) - np.int8(64))  # [-64, 63]


# --- Dataclasses --------------------------------------------------------------


@dataclass
class PeriodicPattern:
    """3-level periodic index pattern. See reference/proof_utils.rs."""

    shape: List[Tuple[int, int]]  # list of (stride, length), length 3

    NUM_DIMS = 3

    @classmethod
    def from_list(cls, pattern: List[int]) -> "PeriodicPattern":
        if not pattern:
            raise ValueError("Pattern cannot be empty")
        if pattern[0] != 0:
            raise ValueError("Pattern must start at 0")
        if any(pattern[i] >= pattern[i + 1] for i in range(len(pattern) - 1)):
            raise ValueError("Pattern must be sorted with no duplicates")

        p = list(pattern)
        shape_vec: List[Tuple[int, int]] = []

        while len(p) > 1:
            found = False
            for period in range(1, len(p)):
                if len(p) % period != 0:
                    continue
                s = p[period]
                if all(p[i] + s == p[i + period] for i in range(len(p) - period)):
                    shape_vec.append((s, len(p) // period))
                    p = p[:period]
                    found = True
                    break
            if not found:
                raise ValueError("Pattern is not periodic")

        shape_vec.reverse()
        period = shape_vec[-1][0] * shape_vec[-1][1] if shape_vec else 1
        while len(shape_vec) < cls.NUM_DIMS:
            shape_vec.append((period, 1))
        return cls(shape=shape_vec)

    def to_list(self) -> List[int]:
        res = [0]
        for stride, length in self.shape:
            res = [r + i * stride for i in range(length) for r in res]
        return res

    def to_bytes(self) -> bytes:
        out = bytearray(2 * self.NUM_DIMS)
        min_stride = 1
        for i, (stride, length) in enumerate(self.shape):
            factor = stride // min_stride
            out[2 * i] = factor - 1
            out[2 * i + 1] = length - 1
            min_stride = stride * length
        return bytes(out)

    def size(self) -> int:
        prod = 1
        for _, length in self.shape:
            prod *= length
        return prod

    def period(self) -> int:
        stride, length = self.shape[-1]
        return stride * length

    def offset_is_valid(self, offset: int) -> bool:
        for stride, length in reversed(self.shape):
            offset %= stride * length
            if offset >= stride:
                return False
        return True


@dataclass
class MiningConfiguration:
    """Mining configuration (52 bytes serialized)."""

    common_dim: int               # u32
    rank: int                     # u16
    mma_type: int                 # u16, only 0 supported
    rows_pattern: PeriodicPattern
    cols_pattern: PeriodicPattern
    reserved: bytes = field(default_factory=lambda: b"\x00" * 32)

    SERIALIZED_SIZE = 52
    RESERVED_SIZE = 32

    def to_bytes(self) -> bytes:
        if len(self.reserved) != self.RESERVED_SIZE:
            raise ValueError(
                f"reserved must be {self.RESERVED_SIZE} bytes, got {len(self.reserved)}"
            )
        return (
            struct.pack("<I", self.common_dim)
            + struct.pack("<H", self.rank)
            + struct.pack("<H", self.mma_type)
            + self.rows_pattern.to_bytes()
            + self.cols_pattern.to_bytes()
            + self.reserved
        )

    def dot_product_length(self) -> int:
        return self.common_dim - (self.common_dim % self.rank)


@dataclass
class IncompleteBlockHeader:
    """76-byte block header without the proof commitment."""

    version: int      # u32
    prev_block: bytes  # 32 bytes
    merkle_root: bytes  # 32 bytes
    timestamp: int    # u32
    nbits: int        # u32

    SERIALIZED_SIZE = 76

    def to_bytes(self) -> bytes:
        if len(self.prev_block) != 32:
            raise ValueError(f"prev_block must be 32 bytes, got {len(self.prev_block)}")
        if len(self.merkle_root) != 32:
            raise ValueError(f"merkle_root must be 32 bytes, got {len(self.merkle_root)}")
        # prev_block and merkle_root are stored byte-reversed in the wire format.
        return (
            struct.pack("<I", self.version)
            + bytes(reversed(self.prev_block))
            + bytes(reversed(self.merkle_root))
            + struct.pack("<I", self.timestamp)
            + struct.pack("<I", self.nbits)
        )


# --- Hashing helpers ----------------------------------------------------------


def blake3_digest(data: bytes, key: bytes | None = None) -> bytes:
    """Match the Rust pearl_blake3::blake3_digest signature: 32-byte digest, optional 32-byte key."""
    if key is None:
        return blake3(data).digest()
    if len(key) != 32:
        raise ValueError(f"key must be 32 bytes, got {len(key)}")
    return blake3(data, key=key).digest()


def pad_to_chunk_boundary(data: bytes) -> bytes:
    """Pad to a multiple of BLAKE3_CHUNK_LEN (1024) with zeros."""
    rem = len(data) % BLAKE3_CHUNK_LEN
    if rem == 0:
        return data
    return data + b"\x00" * (BLAKE3_CHUNK_LEN - rem)


def compute_job_key(header: IncompleteBlockHeader, config: MiningConfiguration) -> bytes:
    return blake3_digest(header.to_bytes() + config.to_bytes(), key=None)


# --- V3 (salted noise-seed) certificate derivation ---------------------------
# Consensus hardfork "salted noise-seed" (Aug 2026). Bind each Merkle root to its
# declared dimension before the legacy commitment chain, so the noise seeds (and
# therefore every derived matrix) differ from V1/V2.
SEED_SALT_A = blake3_digest(b"pearl/cert-v3/noise-seed/A", key=None)
SEED_SALT_B = blake3_digest(b"pearl/cert-v3/noise-seed/B", key=None)

_BIND_PAD = b"\x00" * 28


def _bind_root(root: bytes, dim: int, salt: bytes) -> bytes:
    return blake3_digest(root + dim.to_bytes(4, "little") + _BIND_PAD, key=salt)


def compute_commitment_hash(
    job_key: bytes, a_row_major_padded: bytes, b_col_major_padded: bytes,
    dims: Tuple[int, int] | None = None,
) -> Tuple[bytes, bytes]:
    """Returns (b_noise_seed, a_noise_seed).

    ``dims=(m, n)`` selects the V3 salted derivation (cert_version >= 3); ``None``
    keeps the legacy V1/V2 chain used by offline/old-format jobs.
    """
    hash_a = blake3_digest(a_row_major_padded, key=job_key)
    hash_b = blake3_digest(b_col_major_padded, key=job_key)
    if dims is not None:
        m, n = dims
        hash_a = _bind_root(hash_a, m, SEED_SALT_A)
        hash_b = _bind_root(hash_b, n, SEED_SALT_B)
    b_noise_seed = blake3_digest(job_key + hash_b, key=None)
    a_noise_seed = blake3_digest(b_noise_seed + hash_a, key=None)
    return b_noise_seed, a_noise_seed



# --- Outer hash + target check ------------------------------------------------


def compute_jackpot_hash(jackpot_msg: np.ndarray, commitment_hash: bytes) -> bytes:
    """blake3 of jackpot_msg (16 u32 words, little-endian) with the commitment hash as the key.

    Matches zk-pow/src/api/proof_utils.rs::compute_jackpot_hash.
    """
    if jackpot_msg.shape != (JACKPOT_SIZE,):
        raise ValueError(f"jackpot_msg shape must be ({JACKPOT_SIZE},), got {jackpot_msg.shape}")
    if jackpot_msg.dtype != np.uint32:
        raise ValueError(f"jackpot_msg dtype must be uint32, got {jackpot_msg.dtype}")
    if len(commitment_hash) != 32:
        raise ValueError(f"commitment_hash must be 32 bytes, got {len(commitment_hash)}")
    msg = jackpot_msg.astype("<u4").tobytes()
    if len(msg) != 64:
        raise ValueError(f"serialized jackpot_msg must be 64 bytes, got {len(msg)}")
    return blake3_digest(msg, key=commitment_hash)


def nbits_to_difficulty(nbits: int) -> int:
    """Bitcoin compact difficulty → U256 target."""
    exponent = (nbits >> 24) & 0xFF
    mantissa = nbits & 0x00FFFFFF
    if mantissa == 0 or exponent == 0:
        return 0
    if mantissa & 0x00800000:
        return 0  # invalid/negative
    target = mantissa
    if exponent <= 3:
        target >>= 8 * (3 - exponent)
    else:
        target <<= 8 * (exponent - 3)
    return target & ((1 << 256) - 1)


USIZE_MASK = (1 << 64) - 1
U256_MAX = (1 << 256) - 1


def difficulty_factor(config: MiningConfiguration) -> int:
    """The reward-neutral difficulty multiplier ``h * w * dot_product_length``.

    Computed as ``usize`` (u64 wrap) to match Rust release-mode arithmetic. The
    jackpot hash clears when ``hash <= target * factor`` — for BOTH the nbits block
    target and a pool's share target. A zero factor (common_dim < rank or an empty
    pattern) is a misconfiguration that Rust would panic on; we raise.
    """
    h = config.rows_pattern.size()
    w = config.cols_pattern.size()
    factor = (h * w * config.dot_product_length()) & USIZE_MASK
    if factor == 0:
        raise ValueError(
            f"difficulty factor is zero: h={h} w={w} dot_product_length={config.dot_product_length()}; "
            "this means common_dim < rank or an empty pattern — Rust panics here"
        )
    return factor


def apply_difficulty_factor(target: int, config: MiningConfiguration) -> int:
    """Scale a raw U256 ``target`` by the difficulty factor (clamped to U256_MAX).

    Use for the nbits-derived block bound AND the pool's share target — a pool's
    ``mining.notify`` target is the RAW (pre-factor) value, so the share clears at
    ``target * difficulty_factor``, not ``target`` directly.
    """
    factor = difficulty_factor(config)
    if target > U256_MAX // factor:
        return U256_MAX
    return target * factor


def extract_difficulty_bound(nbits: int, config: MiningConfiguration) -> int:
    """Returns the adjusted U256 bound that hash_jackpot must be <= for a valid solution.

    Matches Rust's release-mode arithmetic via `difficulty_factor` (usize wrap).
    """
    return apply_difficulty_factor(nbits_to_difficulty(nbits), config)


def hash_to_u256_le(hash_bytes: bytes) -> int:
    """Interpret 32 hash bytes as a little-endian U256, the same way the Rust verifier does."""
    if len(hash_bytes) != 32:
        raise ValueError(f"hash_bytes must be 32 bytes, got {len(hash_bytes)}")
    return int.from_bytes(hash_bytes, "little")


# --- Noise generation ---------------------------------------------------------


def _get_random_hash(index: int, seed: bytes, key: bytes, prepend_index: int) -> bytes:
    """Reproduces pearl_noise.rs::get_random_hash.

    Rust does `(1 + index) as i32` which silently wraps on overflow. We mask to
    32 bits and emit unsigned little-endian bytes so we match Rust for any
    index (struct.pack('<i', v) raises OverflowError outside [-2^31, 2^31-1]).
    """
    if len(seed) != 32:
        raise ValueError(f"seed must be 32 bytes, got {len(seed)}")
    if len(key) != 32:
        raise ValueError(f"key must be 32 bytes, got {len(key)}")
    if not 0 <= prepend_index < 8:
        raise ValueError(f"prepend_index must be in [0, 8), got {prepend_index}")
    message = bytearray(64)
    prepend_value = (1 + index) & 0xFFFFFFFF
    message[prepend_index * 4 : prepend_index * 4 + 4] = prepend_value.to_bytes(4, "little", signed=False)
    message[32:64] = seed
    return blake3_digest(bytes(message), key=key)


def _hash_blocks_into(
    out: bytearray, num_blocks: int, seed: bytes, key: bytes, prepend_index: int
) -> None:
    """Fill `out` with `num_blocks` consecutive blake3 digests indexed 0..num_blocks-1.

    Inner-loop equivalent of N calls to `_get_random_hash`: reuses a single
    64-byte message buffer instead of allocating one per call, and avoids the
    `bytes(message)` copy + parameter-validation overhead of the wrapper.
    """
    if len(out) < num_blocks * BLAKE3_DIGEST_SIZE:
        raise ValueError("output buffer too small")
    if len(seed) != 32 or len(key) != 32:
        raise ValueError("seed and key must be 32 bytes")
    # Same invariant _get_random_hash enforced: the 4-byte prepend must land in
    # the first 32 bytes, not overwrite the seed region msg[32:64].
    if not 0 <= prepend_index < 8:
        raise ValueError(f"prepend_index must be in [0, 8), got {prepend_index}")

    msg = bytearray(64)
    msg[32:64] = seed
    off = prepend_index * 4
    for i in range(num_blocks):
        prepend_value = (1 + i) & 0xFFFFFFFF
        msg[off : off + 4] = prepend_value.to_bytes(4, "little", signed=False)
        # bytes(msg) here is unavoidable — blake3_digest needs an immutable bytes-like.
        # The 64-byte copy is ~50 ns and the blake3 work dwarfs it.
        h = blake3_digest(bytes(msg), key=key)
        base = i * BLAKE3_DIGEST_SIZE
        out[base : base + BLAKE3_DIGEST_SIZE] = h


def _is_contiguous_from_zero(row_indices: List[int]) -> bool:
    """Hot fast-path predicate: is this just range(len(row_indices))?"""
    if isinstance(row_indices, range):
        return row_indices.start == 0 and row_indices.step == 1
    n = len(row_indices)
    if n == 0:
        return False
    # Cheap structural check before iterating.
    if row_indices[0] != 0 or row_indices[-1] != n - 1:
        return False
    for i, v in enumerate(row_indices):
        if v != i:
            return False
    return True


def generate_uniform_random_matrix(
    seed: bytes, key: bytes, row_indices: List[int], num_cols: int
) -> np.ndarray:
    """Returns int8 array of shape (len(row_indices), num_cols).

    Values are byte-stream-derived: byte & 0x3F - 32 → int8 in [-32, 32).

    Fast path: when `row_indices == range(len(row_indices))` (the case in
    every mining draw), we hash the contiguous block range once, slice into
    a (rows, num_cols) numpy view, and apply the transformation vectorized.
    """
    # --- Fast path: contiguous rows starting at zero ---------------------
    if _is_contiguous_from_zero(row_indices):
        num_rows = len(row_indices)
        total_bytes = num_rows * num_cols
        # Hash exactly the blocks spanning [0, total_bytes) once. The
        # [:total_bytes] slice is a no-op when total_bytes is a multiple of the
        # digest size, so this single path covers both aligned and unaligned
        # num_cols (ceil division gives the block count either way).
        num_blocks = (total_bytes + BLAKE3_DIGEST_SIZE - 1) // BLAKE3_DIGEST_SIZE
        buf = bytearray(num_blocks * BLAKE3_DIGEST_SIZE)
        _hash_blocks_into(buf, num_blocks, seed, key, prepend_index=0)
        arr = np.frombuffer(bytes(buf), dtype=np.uint8)[:total_bytes]
        arr = ((arr & np.uint8(RANGE_MASK)).astype(np.int16) - ZERO_POINT_TRANSLATION).astype(np.int8)
        return arr.reshape(num_rows, num_cols)

    # --- General path (arbitrary row_indices) ----------------------------
    rows = []
    for row_idx in row_indices:
        start = row_idx * num_cols
        end = start + num_cols
        first_block = start // BLAKE3_DIGEST_SIZE
        last_block = (end - 1) // BLAKE3_DIGEST_SIZE
        chunks = []
        for block in range(first_block, last_block + 1):
            h = _get_random_hash(block, seed, key, 0)
            base = block * BLAKE3_DIGEST_SIZE
            lo = max(0, start - base)
            hi = min(BLAKE3_DIGEST_SIZE, end - base)
            chunks.append(h[lo:hi])
        raw = b"".join(chunks)
        arr = np.frombuffer(raw, dtype=np.uint8)
        arr = ((arr & np.uint8(RANGE_MASK)).astype(np.int16) - ZERO_POINT_TRANSLATION).astype(np.int8)
        rows.append(arr)
    return np.stack(rows, axis=0)


def generate_permutation_matrix(
    seed: bytes, key: bytes, k: int, noise_rank: int
) -> np.ndarray:
    """Returns array of shape (k, 2) of uint32: each row [first_idx, second_idx].

    Optimization vs the literal Rust port: the inner j-loop over the 8 uint32
    words per blake3 hash is replaced by vectorized numpy ops on the full
    (k,) uint32 stream. The Python work per hash drops from ~8 µs to ~0.5 µs.
    """
    rank_mask = np.uint32(noise_rank - 1)
    rank_mul = np.uint64(noise_rank - 1)
    lines_per_hash = BLAKE3_DIGEST_SIZE // 4  # 8
    num_hashes = (k + lines_per_hash - 1) // lines_per_hash

    # Generate all num_hashes blake3 digests into one contiguous buffer.
    buf = bytearray(num_hashes * BLAKE3_DIGEST_SIZE)
    _hash_blocks_into(buf, num_hashes, seed, key, prepend_index=1)

    # Reinterpret as <u4 and trim to exactly k uint32 values.
    ru_all = np.frombuffer(bytes(buf), dtype="<u4")[:k]

    # Vectorized index derivation (matches the Rust formula exactly).
    first = ru_all & rank_mask
    mul_hi = ((rank_mul * ru_all.astype(np.uint64)) >> np.uint64(32)).astype(np.uint32)
    second = first ^ (np.uint32(1) + mul_hi)
    return np.stack([first, second], axis=1)


def _matvec_sparse_perm(perm: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """For each row [a, b] of perm: result[i] = vec[a] - vec[b], cast to int8 with wrap.

    Matches reference/pearl_noise.rs::matvec_sparse_perm.
    """
    if vec.dtype != np.int8:
        raise ValueError(f"vec must be int8, got {vec.dtype}")
    pos = vec[perm[:, 0]].astype(np.int32)
    neg = vec[perm[:, 1]].astype(np.int32)
    # The Rust code casts (pos - neg) as i32 to i8, which is a truncating cast (mod 256, signed wrap).
    diff = (pos - neg).astype(np.int32)
    return ((diff + 128) % 256 - 128).astype(np.int8)


def _matvec_sparse_perm_batch(perm: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """Batched matvec_sparse_perm. `vecs` is (num_rows, noise_rank) int8.

    Returns (num_rows, k) int8. Equivalent to stacking _matvec_sparse_perm(perm, vecs[i])
    across i but computed in one numpy fancy-index op — eliminates the per-row
    Python loop and reuses the same gather indices for every row.
    """
    if vecs.dtype != np.int8:
        raise ValueError(f"vecs must be int8, got {vecs.dtype}")
    if vecs.ndim != 2:
        raise ValueError(f"vecs must be 2-D, got shape {vecs.shape}")
    # vecs[:, idx] gathers along columns for every row in one go.
    pos = vecs[:, perm[:, 0]].astype(np.int32)
    neg = vecs[:, perm[:, 1]].astype(np.int32)
    diff = pos - neg
    return ((diff + 128) % 256 - 128).astype(np.int8)


@dataclass
class MMSlice:
    a: np.ndarray  # int8 (h, k)  — rows of noise for A, one per requested a_row_index
    b: np.ndarray  # int8 (w, k)  — rows of noise for B^t, one per requested b_col_index


def compute_noise_for_indices(
    k: int,
    noise_rank: int,
    commitment_hash: Tuple[bytes, bytes],
    a_rows_indices: List[int],
    b_cols_indices: List[int],
) -> MMSlice:
    """Port of reference/pearl_noise.rs::compute_noise_for_indices.

    Args:
        k: common dimension.
        noise_rank: power of two, divisible by 32.
        commitment_hash: (b_noise_seed, a_noise_seed) — 32 bytes each.
        a_rows_indices: rows of A to materialize.
        b_cols_indices: columns of B to materialize.

    Returns:
        MMSlice with .a (len(a_rows), k) int8 and .b (len(b_cols), k) int8.
    """
    if noise_rank <= 0 or (noise_rank & (noise_rank - 1)) != 0:
        raise ValueError(f"noise_rank must be a positive power of two, got {noise_rank}")
    if noise_rank % BLAKE3_DIGEST_SIZE != 0:
        raise ValueError(
            f"noise_rank must be divisible by {BLAKE3_DIGEST_SIZE}, got {noise_rank}"
        )
    if not a_rows_indices:
        raise ValueError("a_rows_indices is empty")
    if not b_cols_indices:
        raise ValueError("b_cols_indices is empty")
    b_noise_seed, a_noise_seed = commitment_hash

    e_al = generate_uniform_random_matrix(SEED_LABEL_A, a_noise_seed, a_rows_indices, noise_rank)
    e_ar_t = generate_permutation_matrix(SEED_LABEL_A, a_noise_seed, k, noise_rank)
    e_bl = generate_permutation_matrix(SEED_LABEL_B, b_noise_seed, k, noise_rank)
    e_br_t = generate_uniform_random_matrix(SEED_LABEL_B, b_noise_seed, b_cols_indices, noise_rank)

    noise_a = _matvec_sparse_perm_batch(e_ar_t, e_al)
    noise_b_t = _matvec_sparse_perm_batch(e_bl, e_br_t)

    return MMSlice(a=noise_a, b=noise_b_t)
