"""Stratum pool client for Pearl mining (controller side).

Lets the controller mine **to a pool** (share-based) instead of **solo** via
pearld `getblocktemplate`/`submitblock`. The wire format was determined from
live pool traffic and is documented below.

Two pool endpoints use this wire format:

  * **LuckyPool** (`mining.*`, object params).
      authorize : ``mining.authorize {"wallet": "prl1p….worker"}``  -> ``{result:true}``
      job       : server pushes ``mining.notify {header, height, job_id, target}``
      share     : ``mining.submit {"job_id", "plain_proof"}``  (base64 PlainProof)
  * **AlphaPool plain Stratum** (port 5571) uses the same object-param messages.
      Its fixed difficulty is selected with authorize password ``x;d=N``.

AlphaPool's legacy port 5566 front-loads an anti-DDoS ``pearl.challenge`` PoW
gate. We detect that older dialect and raise `AlphaPoolChallengeUnsupported`.

Key fact that shapes this client: a **share is a cheap `PlainProof`** (the
Merkle openings of the winning A/B columns), NOT a Plonky2 proof. The POOL runs
the expensive prover and assembles the block (matching the upstream gateway's
`submitPlainProof` -> `ProofGenerator`). So a pool miner never calls
`generate_proof` and never needs the prover warmup.
"""

from __future__ import annotations

import itertools
import json
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class StratumError(RuntimeError):
    """A pool returned an error response, or the transport failed."""


class AlphaPoolChallengeUnsupported(StratumError):
    """The pool opened with a ``pearl.challenge`` auth-PoW gate (AlphaPool's
    DDoS-mitigation handshake). The challenge solver and post-auth ``pearl.*``
    job/submit frames are not implemented."""


def parse_pool_url(url: str) -> tuple[str, int, bool]:
    """``stratum+tcp://host:port`` / ``stratum+ssl://host:port`` / ``host:port``
    -> ``(host, port, use_tls)``."""
    use_tls = False
    rest = url
    if "://" in url:
        scheme, rest = url.split("://", 1)
        use_tls = scheme.lower() in ("stratum+ssl", "stratum+tls", "ssl", "tls")
    rest = rest.rstrip("/")
    if ":" not in rest:
        raise StratumError(f"pool url must be host:port (got {url!r})")
    host, port = rest.rsplit(":", 1)
    return host, int(port), use_tls


def header_from_wire(b: bytes):
    """Parse the 76-byte ``mining.notify.header`` into an `IncompleteBlockHeader`.

    The wire header is the *serialized* form (version LE, prev_block and
    merkle_root byte-reversed, timestamp LE, nbits LE), i.e. the exact output of
    `IncompleteBlockHeader.to_bytes()`. We invert that so the reconstructed
    header round-trips byte-for-byte — `compute_job_key` then hashes precisely
    the bytes the pool committed to.
    """
    import struct

    from .protocol import IncompleteBlockHeader

    if len(b) != IncompleteBlockHeader.SERIALIZED_SIZE:  # 76
        raise StratumError(
            f"header must be {IncompleteBlockHeader.SERIALIZED_SIZE} bytes, got {len(b)}"
        )
    header = IncompleteBlockHeader(
        version=struct.unpack_from("<I", b, 0)[0],
        prev_block=bytes(reversed(b[4:36])),
        merkle_root=bytes(reversed(b[36:68])),
        timestamp=struct.unpack_from("<I", b, 68)[0],
        nbits=struct.unpack_from("<I", b, 72)[0],
    )
    if header.to_bytes() != b:  # invariant: parsing is the exact inverse of to_bytes
        raise StratumError("header round-trip mismatch; wire layout differs from expectation")
    return header


@dataclass
class Job:
    """One ``mining.notify`` work unit."""

    job_id: str
    header_bytes: bytes  # 76-byte incomplete header (version‖prev‖merkle‖time‖nbits)
    target: int          # U256 share bound; jackpot hash must be <= this
    height: Optional[int]
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_notify(cls, params: dict) -> "Job":
        return cls(
            job_id=str(params["job_id"]),
            header_bytes=bytes.fromhex(params["header"]),
            target=int(params["target"], 16),
            height=params.get("height"),
            raw=params,
        )


class StratumPoolClient:
    """Line-delimited JSON-RPC Stratum client. A background thread reads server
    frames; ``mining.notify`` jobs are latched into ``self._job`` and requests
    (``authorize``/``submit``) block on an id-keyed response."""

    def __init__(
        self,
        url: str,
        wallet: str,
        worker: Optional[str] = None,
        difficulty: Optional[int] = None,
        password: Optional[str] = None,
        log: Callable[[str], None] = print,
        timeout: float = 30.0,
        keepalive_interval: float = 20.0,
    ):
        self.host, self.port, self.use_tls = parse_pool_url(url)
        self.wallet = wallet
        self.worker = worker
        self.difficulty = difficulty
        # AlphaPool 5571 reads static difficulty from the standard password field;
        # LuckyPool reads the numeric difficulty field. Sending both makes
        # --difficulty portable while an explicit password can also carry t=<PRL>.
        self.password = password if password is not None else (
            f"x;d={int(difficulty)}" if difficulty is not None else None
        )
        self.log = log
        self.timeout = timeout
        self.keepalive_interval = max(0.0, float(keepalive_interval))

        self._sock: Optional[socket.socket] = None
        self._reader: Optional[threading.Thread] = None
        self._send_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._pending: dict[int, tuple[threading.Event, list]] = {}
        self._ignored_ids: set[int] = set()
        self._job: Optional[Job] = None
        self._job_lock = threading.Lock()
        self._job_event = threading.Event()
        self._challenge_event = threading.Event()
        self._closed = False
        self._authorized = False
        self._last_send = 0.0
        self._keepalive_thread: Optional[threading.Thread] = None

    # --- login string -------------------------------------------------------
    def _login(self) -> str:
        return f"{self.wallet}.{self.worker}" if self.worker else self.wallet

    # --- connection ---------------------------------------------------------
    def connect(self) -> None:
        self._closed = False
        raw = socket.create_connection((self.host, self.port), timeout=self.timeout)
        if self.use_tls:
            ctx = ssl.create_default_context()
            raw = ctx.wrap_socket(raw, server_hostname=self.host)
        self._sock = raw
        self._reader = threading.Thread(target=self._read_loop, args=(raw,),
                                         name="stratum-reader", daemon=True)
        self._reader.start()

    def close(self) -> None:
        self._closed = True
        with self._send_lock:
            sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    # --- transport ----------------------------------------------------------
    def _send(self, obj: dict) -> None:
        line = (json.dumps(obj) + "\n").encode()
        with self._send_lock:
            if self._sock is None:
                raise StratumError("not connected")
            try:
                self._sock.sendall(line)
                self._last_send = time.monotonic()
            except OSError as e:
                # Surface a dead socket as StratumError so callers like
                # submit_share (which only catches StratumError) don't crash on it.
                raise StratumError(f"send failed: {e}") from e

    def _read_loop(self, sock: socket.socket) -> None:
        buf = b""
        try:
            while not self._closed:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        self._handle_line(line)
        except OSError:
            pass
        finally:
            if not self._closed:
                self.log("stratum: connection lost; attempting reconnect")
                self._reconnect()

    def _handle_line(self, line: bytes) -> None:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            self.log(f"stratum: non-JSON frame {line!r}")
            return
        method = msg.get("method")
        if method == "mining.notify":
            self._on_notify(msg.get("params") or {})
        elif method in ("pearl.challenge", "pearl.notify"):
            # AlphaPool dialect: anti-DDoS challenge gate (and pearl.* work).
            self._challenge_event.set()
            self.log(f"stratum: AlphaPool {method} seen (challenge-gated, unsupported)")
        elif method == "mining.set_difficulty":
            self.log(f"stratum: set_difficulty {msg.get('params')}")
        elif method is None and msg.get("id") in self._pending:
            ev, holder = self._pending.pop(msg["id"])
            holder.append(msg)
            ev.set()
        elif method is None and msg.get("id") in self._ignored_ids:
            self._ignored_ids.discard(msg["id"])
        else:
            self.log(f"stratum: unhandled frame {line!r}")

    def _on_notify(self, params: dict) -> None:
        try:
            job = Job.from_notify(params)
        except (KeyError, ValueError) as e:
            self.log(f"stratum: bad mining.notify ({e}): {params!r}")
            return
        with self._job_lock:
            self._job = job
        self._job_event.set()
        self.log(
            f"stratum: job {job.job_id} height={job.height} "
            f"target~2^{job.target.bit_length() - 1}"
        )

    # --- request/response ---------------------------------------------------
    def _request(self, method: str, params: dict, timeout: Optional[float] = None):
        rid = next(self._ids)
        ev = threading.Event()
        holder: list = []
        self._pending[rid] = (ev, holder)
        self._send({"id": rid, "method": method, "params": params})
        if not ev.wait(timeout or self.timeout):
            self._pending.pop(rid, None)
            if self._challenge_event.is_set():
                raise AlphaPoolChallengeUnsupported(
                    f"pool {self.host}:{self.port} requires a pearl.challenge auth-PoW "
                    "(AlphaPool); not implemented in this release"
                )
            raise StratumError(f"timeout waiting for {method} response")
        resp = holder[0]
        if resp.get("error"):
            raise StratumError(f"{method} error: {resp['error']}")
        return resp.get("result")

    def authorize(self) -> Any:
        """Send ``mining.authorize`` and wait for the ack. Raises
        `AlphaPoolChallengeUnsupported` if the server is challenge-gated."""
        params: dict = {"wallet": self._login()}
        if self.difficulty is not None:
            params["difficulty"] = int(self.difficulty)
        if self.password is not None:
            params["password"] = self.password
        result = self._request("mining.authorize", params, timeout=min(self.timeout, 12.0))
        self._authorized = True
        if self.keepalive_interval and not (
            self._keepalive_thread and self._keepalive_thread.is_alive()
        ):
            self._keepalive_thread = threading.Thread(
                target=self._keepalive_loop, name="stratum-keepalive", daemon=True
            )
            self._keepalive_thread.start()
        return result

    def keepalive_if_idle(self) -> None:
        """Keep send-idle Stratum sessions alive while a low-rate share is searched.

        Both observed public pools close an otherwise healthy socket after roughly
        30–45 seconds without a client frame. ``mining.configure`` is harmless on the
        Pearl gateways and is also what AlphaPool's legacy client uses as its ping.
        """
        if (
            self._closed
            or not self._authorized
            or not self.keepalive_interval
            or time.monotonic() - self._last_send < self.keepalive_interval
        ):
            return
        rid = next(self._ids)
        if len(self._ignored_ids) >= 64:
            self._ignored_ids.clear()
        self._ignored_ids.add(rid)
        try:
            self._send({"id": rid, "method": "mining.configure", "params": [["pearl/v1"], {}]})
        except StratumError:
            self._ignored_ids.discard(rid)

    def _keepalive_loop(self) -> None:
        while not self._closed:
            time.sleep(min(5.0, self.keepalive_interval))
            self.keepalive_if_idle()

    def submit_share(self, job_id: str, plain_proof_b64: str) -> tuple[bool, Any]:
        """Submit ``mining.submit {job_id, plain_proof}``. Returns (ok, result-or-error)."""
        try:
            return True, self._request(
                "mining.submit", {"job_id": job_id, "plain_proof": plain_proof_b64}
            )
        except StratumError as e:
            return False, str(e)

    # --- jobs ---------------------------------------------------------------
    def current_job(self, timeout: Optional[float] = None) -> Optional[Job]:
        """Latest job. With ``timeout`` blocks until the first job arrives;
        ``timeout=0`` returns immediately (latest or None)."""
        if timeout == 0:
            if not self._job_event.is_set():
                return None
        elif not self._job_event.wait(timeout):
            return None
        with self._job_lock:
            return self._job

    # --- reconnect ----------------------------------------------------------
    def _reconnect(self, attempts: int = 6) -> None:
        backoff = 1.0
        for _ in range(attempts):
            if self._closed:
                return
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            try:
                self.connect()
                if self._authorized:
                    self.authorize()
                self.log("stratum: reconnected")
                return
            except (OSError, StratumError) as e:
                self.log(f"stratum: reconnect failed ({e})")
        # Reconnect exhausted: mark the client dead so mining loops can detect it
        # (current_job keeps returning the last latched job, so without this they'd
        # spin forever on a stale job).
        self._closed = True
        self.log("stratum: giving up reconnect")


# --- pool-mode helpers (heavy imports are lazy so the transport stays light) --

def snapshot_from_pool_job(job: Job, k: int, rank: int, rows, cols, generation: int = 0):
    """Build a `streaming.TemplateSnapshot` from a pool `Job`.

    Unlike `snapshot_from_template`, there is no coinbase/merkle to build — the
    pool already baked a per-miner coinbase into ``job.header_bytes`` — so
    coinbase_tx/raw_txns are unused (the pool, not us, assembles the block).

    The pool's ``target`` is the RAW (pre-factor) share target: a jackpot hash
    clears at ``target * difficulty_factor`` (h*w*dot_product_length, ~2^19 for the
    16x16/k2048 config), exactly the factor `extract_difficulty_bound` applies to
    the nbits target. We apply it here so every consumer (search bound AND
    `reconstruct_winner`'s re-verify) uses the consensus share bound by
    construction; using ``target`` verbatim is ~500,000x too strict.
    """
    from .protocol import apply_difficulty_factor, compute_job_key
    from .streaming import TemplateSnapshot, build_config

    header = header_from_wire(job.header_bytes)
    py_config, rs_config = build_config(k, rank, rows, cols)
    return TemplateSnapshot(
        generation=generation,
        template=job.raw,
        header=header,
        config=py_config,
        rs_config=rs_config,
        coinbase_tx=None,
        raw_txns=[],
        bound=apply_difficulty_factor(job.target, py_config),
        job_key=compute_job_key(header, py_config),
    )


def build_plain_proof_base64(winner, m: int, n: int, k: int, rank: int,
                             log: Callable[[str], None] = print,
                             verify_header_target: bool = True) -> tuple[Optional[str], Optional[str]]:
    """Build the share payload for a winning draw: the Merkle openings of the
    winning A/B rows/cols, packaged as a `PlainProof`, base64-encoded.

    This is the cheap half of `streaming.build_and_submit` (steps up to
    `verify_plain_proof`) — it does NOT run `generate_proof`; the pool proves.
    ``verify_header_target`` is used by solo/self-test paths. Pool jobs deliberately
    set it false: the header contains network block difficulty, while
    ``reconstruct_winner`` has already checked the candidate against the easier pool
    share target. Rechecking a normal share against the block target would discard it.
    Returns ``(base64, None)`` on success or ``(None, error_message)``.
    """
    import pearl_mining as rust

    from .protocol import pad_to_chunk_boundary

    snap = winner.snapshot
    rs_header = rust.IncompleteBlockHeader(
        snap.header.version, snap.header.prev_block, snap.header.merkle_root,
        snap.header.timestamp, snap.header.nbits,
    )
    a_padded = winner.draw.a_padded or pad_to_chunk_boundary(winner.draw.A.tobytes())
    b_padded = winner.draw.b_padded or pad_to_chunk_boundary(winner.draw.B_t.tobytes())

    tree_a = rust.MerkleTree(a_padded, snap.job_key)
    leaves_a = rust.MerkleTree.compute_leaf_indices_from_rows(winner.a_rows, (m, k))
    proof_a = rust.MatrixMerkleProof(tree_a.get_multileaf_proof(leaves_a), winner.a_rows)

    tree_b = rust.MerkleTree(b_padded, snap.job_key)
    leaves_b = rust.MerkleTree.compute_leaf_indices_from_rows(winner.b_cols, (n, k))
    proof_b = rust.MatrixMerkleProof(tree_b.get_multileaf_proof(leaves_b), winner.b_cols)

    plain_proof = rust.PlainProof(m, n, k, rank, proof_a, proof_b)
    if verify_header_target:
        ok, msg = rust.verify_plain_proof(rs_header, plain_proof)
        if not ok:
            return None, f"verify_plain_proof rejected: {msg}"
    return plain_proof.to_base64(), None
