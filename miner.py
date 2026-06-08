"""Pearl miner for Apple Silicon: mine to a Stratum pool (--pool) or solo to a pearld node (--rpc).

Both modes share one validated MLX-GPU + ARM-SME search core. A disclosed dev fee is pool-mined
to the developer address (see dev_fee.py and the README). The pool transport is Stratum V1
(mining.*); LuckyPool is supported. Run `python miner.py --selftest` for an offline end-to-end
check."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

import numpy as np
import mlx.core as mx

import dev_fee

from pearl_core import protocol  # noqa: E402
from pearl_core.miner import threads_partition  # noqa: E402
from pearl_core.autonomous_mine import reconstruct_winner  # noqa: E402
from pearl_core.pearld_rpc import public_rpc_url  # noqa: E402
from pearl_core.stratum import (  # noqa: E402
    StratumPoolClient, Job, AlphaPoolChallengeUnsupported,
    snapshot_from_pool_job, build_plain_proof_base64,
)

from fast_search import search_mask_fast as search_mask  # noqa: E402
from fast_setup import setup_symmetric_gpu  # noqa: E402  (bit-exact on-device draw setup)
from stream_search import search_draws_overlapped  # noqa: E402  (CPU commit ∥ GPU search)
from cpu_search import search_draws_cpu_auto as search_draws_cpu  # noqa: E402  (CPU+SME engine, bit-exact)
try:                                     # no-sudo thermal/power telemetry for the status line
    from thermal import ThermalMonitor, pressure_name, power_mode_name
except Exception:                        # never let telemetry break mining
    ThermalMonitor = None
    def pressure_name(x): return str(x)
    def power_mode_name(x): return str(x)

# Draws streamed per generator instance before re-checking the pool job. Bounds the
# pipeline's lifetime (and its worker thread) so job rotation stays responsive.
STREAM_BATCH = 64
# Overlapped draws per burst when --compute-pct throttles: run a burst, drain, then idle. The
# burst size trades GPU availability for other apps (shorter = the GPU is free sooner) against how
# precisely the duty-cycle hits the target (longer = amortizes the per-burst setup).
# PEARL_THROTTLE_BURST overrides the default.
THROTTLE_BURST = max(1, int(os.environ.get("PEARL_THROTTLE_BURST", "4")))
# CPU/SME throttle burst — the opposite of the GPU: the CPU prefers LARGER bursts (its idle floor
# is low and each burst has engine setup, so tiny bursts waste time and power).
# PEARL_THROTTLE_BURST_CPU overrides the default.
THROTTLE_BURST_CPU = max(1, int(os.environ.get("PEARL_THROTTLE_BURST_CPU", "16")))
# CPU+SME draws use a disjoint nonce range so they never collide with the GPU's draws in `both`
# mode. 2^30 keeps the two ranges disjoint while fitting both engines' index constraints; a session
# never reaches a billion draws, so the ranges never overlap in practice.
CPU_DRAW_OFFSET = 1 << 30
_ADAPT_MCPU = os.environ.get("PEARL_ADAPT_MCPU", "1") != "0"
_MCPU_HOT = int(os.environ.get("PEARL_MCPU_HOT", "16384"))
if _MCPU_HOT % 32 != 0:   # the SME unified engine needs m%32==0; fall back to the validated default
    _MCPU_HOT = 16384
# The GPU search only PROPOSES candidate winners; reconstruct_winner independently re-verifies
# every hit on the CPU consensus oracle, so a fast kernel can never submit a bad share.

# 16x16 contiguous tiles: the unit the search folds.
TILE = 16
ROWS = list(range(TILE))
COLS = list(range(TILE))
__version__ = "1.0.0"


def _u32x8_le(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<u4").copy()


def _bound_u32x8(bound: int) -> np.ndarray:
    return np.frombuffer(int(bound).to_bytes(32, "little"), dtype="<u4").copy()


# --- operational resilience (failover / keepalive / watchdog / log-to-file) ---------
class PoolConnectionLost(Exception):
    """mine_pool raises this when the pool link is unusable (reconnect exhausted, or the
    job stream stalled past --stall-timeout). The supervisor catches it to fail over to the
    next pool / auto-restart, instead of silently sitting on a dead connection."""


def _make_logger(log_file: Optional[str]) -> Callable[..., None]:
    """Return the log callback: always prints to stdout, and (if log_file is set) also appends
    timestamped lines to the file. A miner runs unattended for days, so a file log is table stakes."""
    if not log_file:
        return print

    def _log(msg: str = "") -> None:
        print(msg)
        try:
            with open(log_file, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        except OSError:
            pass  # never let logging break mining

    return _log


def _enable_tcp_keepalive(client, idle_s: int = 60) -> None:
    """Turn on OS-level TCP keepalive on the pool socket so a silently-dropped connection
    (router NAT timeout, pool crash with no FIN) is detected and the reader thread reconnects,
    instead of the miner hanging on a half-open socket. Best-effort; never fatal."""
    sock = getattr(client, "_sock", None)
    if sock is None:
        return
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        if hasattr(socket, "TCP_KEEPALIVE"):      # macOS: idle seconds before the first probe
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, idle_s)
        elif hasattr(socket, "TCP_KEEPIDLE"):     # linux
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, idle_s)
    except OSError:
        pass


def validate_shape(m: int, k: int, rank: int) -> None:
    """Consensus/tiling invariants the kernels assume — fail loudly on a bad combo.
    The difficulty factor (h*w*dot_product_length) is now applied at the source in
    `stratum.snapshot_from_pool_job`, so the bound the search and re-verify use is
    already the consensus share bound; nothing here re-scales it."""
    if m % TILE != 0:
        raise ValueError(f"m (and n=m) must be a multiple of {TILE}; got {m}")
    if rank <= 0 or k % rank != 0:
        raise ValueError(f"k must be a positive multiple of rank; got k={k} rank={rank}")
    if k < 16 * rank:
        raise ValueError(f"consensus requires k >= 16*rank; got k={k} rank={rank}")


def setup_symmetric(base_seed: bytes, draw_index: int, job_key: bytes,
                    m: int, k: int, rank: int):
    """Effective (a_eff, b_eff, a_seed_u32) for the symmetric draw (B = A, n = m).

    Mirrors `autonomous_mine.reconstruct_winner` exactly: cheap int7 A, one
    commitment tree (b_pad == a_pad), noise from the commitment seeds. Returns the
    full-(m,k) effective operands so the kernel can fold every 16x16 tile.
    """
    A = protocol.signal_matrix_cheap(base_seed, draw_index, protocol.SIGNAL_LABEL_A, m, k)
    a_pad = protocol.pad_to_chunk_boundary(A.tobytes())
    b_seed, a_seed = protocol.compute_commitment_hash(job_key, a_pad, a_pad)
    noise = protocol.compute_noise_for_indices(
        k, rank, (b_seed, a_seed), list(range(m)), list(range(m)))
    a_eff = A.astype(np.int32) + noise.a.astype(np.int32)
    b_eff = A.astype(np.int32) + noise.b.astype(np.int32)
    return a_eff, b_eff, _u32x8_le(a_seed)


def search_draw(base_seed: bytes, draw_index: int, job_key: bytes,
                m: int, k: int, rank: int, bound_u32x8: np.ndarray) -> list[int]:
    """Run one symmetric draw on the GPU; return winning tile indices (t=i*ntj+j).

    Setup also runs on the GPU (setup_symmetric_gpu): a_eff/b_eff are built on-device
    from small CPU-hashed inputs, avoiding the operand transfers. Bit-exact with the
    CPU setup."""
    a_eff, b_eff, a_seed_u32 = setup_symmetric_gpu(base_seed, draw_index, job_key, m, k, rank)
    mask, _, _ = search_mask(a_eff, b_eff, mx.array(a_seed_u32), mx.array(bound_u32x8), rank,
                             want_digest=False)  # digest discarded; skip its DRAM write
    return [int(t) for t in np.nonzero(np.array(mask))[0]]


def _throttled(make_chunk_gen, indices, pct: int, burst: int, should_stop=None):
    """Duty-cycle a per-(draw_index, winners) generator for background/low-heat mining: run `burst`
    draws, then idle for active·(100/pct − 1) so the engine is busy ~pct% of the wall (~pct% avg
    power; the die/cluster power-gates in the gap). `make_chunk_gen(chunk)` returns a FRESH (di,wins)
    generator over that index slice — it must DRAIN (finish all in-flight work) when the slice ends,
    so the idle is genuinely idle (GPU: a bounded search_draws_overlapped drains; CPU: the engine
    finishes its batch). pct>=100 is a straight passthrough. `should_stop()` (optional) lets a concurrent
    teardown abort the idle promptly (the idle sleeps in small slices and rechecks it).

    Per-engine (not global): so `both` mode throttles GPU and CPU INDEPENDENTLY with their own pct +
    burst (the two have opposite optima — GPU wants tiny bursts, the CPU wants large ones). Pure
    timing; winners are unchanged → still bit-exact."""
    idxs = list(indices)
    if pct >= 100 or not idxs:
        yield from make_chunk_gen(idxs)
        return
    i = 0
    while i < len(idxs):
        if should_stop is not None and should_stop():
            return
        t0 = time.perf_counter()
        for item in make_chunk_gen(idxs[i:i + burst]):   # drains at the slice end → GPU/CPU idle after
            yield item
        idle_until = time.perf_counter() + (time.perf_counter() - t0) * (100.0 - pct) / pct
        while True:                                       # idle in slices so teardown interrupts it
            rem = idle_until - time.perf_counter()
            if rem <= 0 or (should_stop is not None and should_stop()):
                break
            time.sleep(min(rem, 0.2))
        i += burst


def _draw_generator(engine: str, base_seed: bytes, job_key: bytes, m: int, k: int,
                    rank: int, bound8: np.ndarray, start: int, batch: int, m_cpu: int = 0,
                    gpu_pct: int = 100, cpu_pct: int = 100):
    """Yield (draw_index, winners, m_draw) for one batch from the selected engine(s). m_draw
    is the draw side that engine used (so the consumer reconstructs each winner at the right m).
      gpu  — the MLX GPU pipeline (search_draws_overlapped) at m
      cpu  — the CPU+SME path (search_draws_cpu) at m
      both — BOTH concurrently on DISJOINT nonce ranges, EACH AT ITS OWN OPTIMUM m: the GPU at m
             (large), the CPU/SME at m_cpu (default 8192, its throughput peak; at small m the CPU
             is setup-bound). The CPU is a near-free additive engine on a disjoint nonce range
             (see CPU_DRAW_OFFSET).
    gpu_pct/cpu_pct (<100) duty-cycle each engine for background mining (see _throttled), each with
    its own burst (THROTTLE_BURST / THROTTLE_BURST_CPU). Every yielded winner is re-verified by
    reconstruct_winner regardless of which engine proposed it, so neither engine can submit a bad share."""
    idxs = range(start, start + batch)
    if engine == "gpu":
        def mk(chunk):
            return search_draws_overlapped(base_seed, job_key, m, k, rank, bound8, chunk)
        for di, wins in _throttled(mk, idxs, gpu_pct, THROTTLE_BURST):
            yield di, wins, m
    elif engine == "cpu":
        # cpu-only: the GPU isn't competing for cores, so oversubscribe the CPU engine's
        # thread count to fully hide the fold under the matmul.
        import cpu_search as _cs

        def mk(chunk):
            return search_draws_cpu(base_seed, job_key, m, k, rank, bound8, chunk,
                                    n_tr=_cs.PL_ENGINE_THREADS_CPU)
        for di, wins in _throttled(mk, idxs, cpu_pct, THROTTLE_BURST_CPU):
            yield di, wins, m
    else:  # both — GPU at m, CPU/SME at m_cpu, disjoint nonce ranges, throttled independently
        yield from _merged_gpu_cpu(base_seed, job_key, m, m_cpu, k, rank, bound8, start, batch,
                                   gpu_pct=gpu_pct, cpu_pct=cpu_pct)


def _merged_gpu_cpu(base_seed: bytes, job_key: bytes, m_gpu: int, m_cpu: int, k: int, rank: int,
                    bound8: np.ndarray, start: int, batch: int, gpu_pct: int = 100, cpu_pct: int = 100):
    """GPU (at m_gpu) and CPU+SME (at m_cpu) mine the same job simultaneously — independent
    engines on disjoint nonce ranges, each at its own optimal draw size — merged onto one stream
    as draws finish. Yields (draw_index, winners, m_draw) tagged with the producing engine's m.
    Each engine is throttled INDEPENDENTLY (gpu_pct/cpu_pct, each with its own burst) inside its pump,
    so e.g. the GPU can run at 20% while the CPU runs at 50% — the idle is stop-aware so teardown is prompt."""
    import queue
    q: "queue.Queue" = queue.Queue(maxsize=16)
    DONE, ERR = object(), object()
    stop = threading.Event()

    def mk_g(chunk):
        return search_draws_overlapped(base_seed, job_key, m_gpu, k, rank, bound8, chunk)

    def mk_c(chunk):
        return search_draws_cpu(base_seed, job_key, m_cpu, k, rank, bound8, chunk)
    ggen = _throttled(mk_g, range(start, start + batch), gpu_pct, THROTTLE_BURST,
                      should_stop=stop.is_set)
    cgen = _throttled(mk_c, range(start + CPU_DRAW_OFFSET, start + CPU_DRAW_OFFSET + batch),
                      cpu_pct, THROTTLE_BURST_CPU, should_stop=stop.is_set)

    def pump(g, m_draw):
        # Iterate AND close g in this same thread — closing a generator from another thread while
        # this one is mid-iteration raises "generator already executing". A stop-aware put keeps
        # us from hanging on a full queue once the consumer stops draining.
        try:
            for di, wins in g:
                if stop.is_set():
                    break
                while not stop.is_set():
                    try:
                        q.put((di, wins, m_draw), timeout=0.2); break   # tag with the engine's m
                    except queue.Full:
                        continue
        except Exception as e:                 # surface to the consumer; don't deadlock
            try: q.put((ERR, e), timeout=0.2)
            except queue.Full: pass
        finally:
            g.close()                          # OUR generator, OUR thread (no cross-thread race)
            try: q.put(DONE, timeout=0.2)
            except queue.Full: pass

    gt = threading.Thread(target=pump, args=(ggen, m_gpu), daemon=True)
    ct = threading.Thread(target=pump, args=(cgen, m_cpu), daemon=True)
    gt.start()
    ct.start()
    done = 0
    try:
        while done < 2:
            item = q.get()
            if item is DONE:
                done += 1
                continue
            if item[0] is ERR:                  # (ERR, exc) sentinel
                raise item[1]
            yield item
    finally:                                    # tear down both engines on close/rotation
        stop.set()                              # pumps see this, break, and close their own gens
        deadline = time.time() + 3.0
        while (gt.is_alive() or ct.is_alive()) and time.time() < deadline:
            try:
                while True:
                    q.get_nowait()              # drain so a pump blocked on a full put can exit
            except queue.Empty:
                pass
            gt.join(timeout=0.1); ct.join(timeout=0.1)


def _status_line(gpu_tiles: int, cpu_tiles: int, k: int, engine: str, shares: int,
                 draws: int, search_s: float, uptime_s: float, mon, unit: str = "shares") -> str:
    """A standard-miner-style status line: per-engine + total Pearl TH/s, GPU/CPU watts,
    power mode, thermal pressure, GPU util, shares, uptime. Telemetry is no-sudo (thermal.py);
    fields degrade gracefully to '?' if unavailable."""
    hwk = TILE * TILE * k
    dt = max(search_s, 1e-9)
    gpu_th = gpu_tiles * hwk / dt / 1e12
    cpu_th = cpu_tiles * hwk / dt / 1e12
    total_th = gpu_th + cpu_th
    up = uptime_s
    upstr = f"{int(up // 3600)}h{int(up % 3600 // 60):02d}m" if up >= 3600 else f"{int(up // 60)}m{int(up % 60):02d}s"
    rate = f"{total_th:6.3f} TH/s"
    if engine == "both":
        rate += f" (GPU {gpu_th:.2f} + CPU {cpu_th:.2f})"
    therm = ""
    if mon is not None:
        s = mon.sample()
        g = f"{s.gpu_w:.0f}W" if s.gpu_w is not None else "?W"
        c = f"{s.cpu_w:.0f}W" if s.cpu_w is not None else "?W"
        therm = (f" | {g} GPU {c} CPU | power={power_mode_name(s.pmode)} "
                 f"pressure={s.pressure}({pressure_name(s.pressure)}) util={s.util_pct}%")
        rec, burst, pkg, bpkg, iv = mon.throttle(s, gpu_tiles + cpu_tiles, hwk, dt)
        if rec is not None and burst > 0:
            seg = f" | throttle: now {rec:.2f} vs burst {burst:.2f} TH/s ({rec / burst - 1:+.0%})"
            if bpkg > 0:
                seg += f", pkg {pkg:.0f}/{bpkg:.0f}W"
            therm += seg + f", ctxsw {iv:.0f}/s"
    return f"  {rate} | {unit} {shares} | {draws} draws | up {upstr}{therm}"


def mine_pool(pool_url: str, address: str, worker: Optional[str],
              m: int, k: int, rank: int, difficulty: Optional[int] = None,
              max_shares: int = 0, max_draws: int = 0,
              status_every: int = 50, engine: str = "gpu", m_cpu: int = 0,
              compute_pct: int = 100, compute_pct_cpu: Optional[int] = None,
              stall_timeout: float = 0.0, run_seconds: float = 0.0, dev_fee_scheduler=None,
              search_time_callback: Optional[Callable[[float], None]] = None,
              log: Callable[[str], None] = print) -> float:
    """Connect, authorize, then loop: per job, search draws on the selected engine(s)
    and submit a PlainProof for every verified winner. n is forced to m (symmetric draw).

    run_seconds>0 stops after that long (used to mine a bounded dev-fee slice; see
    _mine_dev_fee_slice). dev_fee_scheduler, if given, periodically pool-mines the dev fee to the
    dev wallet (the fee slice itself calls back in with run_seconds set + scheduler None).
    engine ∈ {gpu, cpu, both} (see _draw_generator). In `both`, the GPU runs at m and the
    CPU/SME at m_cpu (each at its own optimum, disjoint nonce ranges); winners are reconstructed
    at the producing engine's m.

    compute_pct / compute_pct_cpu (1-100): background/thermal throttle, PER ENGINE. <100 duty-cycles
    that engine — run a burst, drain, then idle so the engine is busy ~pct% of the wall, drawing ~pct%
    of the power (avg) so it never heat-soaks (see _throttled). compute_pct is the GPU's; compute_pct_cpu
    is the CPU/SME's and DEFAULTS TO compute_pct (so `--compute-pct 20` throttles whichever engine(s) run;
    `--compute-pct 20 --compute-pct-cpu 50` runs the GPU at 20% and the CPU at 50% in `both` mode). The
    two have OPPOSITE burst optima (the GPU power-gates between bursts so it prefers small bursts;
    the CPU has per-burst setup overhead so it prefers large ones) — see THROTTLE_BURST /
    THROTTLE_BURST_CPU. Pure timing; winners stay bit-exact."""
    n = m  # B = A
    compute_pct = max(1, min(100, int(compute_pct)))
    compute_pct_cpu = compute_pct if compute_pct_cpu is None else max(1, min(100, int(compute_pct_cpu)))
    validate_shape(m, k, rank)
    if engine == "both":
        if m_cpu <= 0:
            m_cpu = m
        validate_shape(m_cpu, k, rank)
    client = StratumPoolClient(pool_url, address, worker, difficulty, log=log)
    log(f"connecting to {client.host}:{client.port} (tls={client.use_tls}) as {address}"
        + (f".{worker}" if worker else ""))
    client.connect()
    _enable_tcp_keepalive(client)   # detect silently-dropped links so reconnect can fire
    try:
        client.authorize()
        log("authorized")
    except AlphaPoolChallengeUnsupported as e:
        log(f"FATAL: {e}")
        log("This pool is challenge-gated; target a LuckyPool endpoint (mining.* dialect) "
            "instead; AlphaPool's post-challenge protocol is not implemented.")
        client.close()
        raise

    base_seed = os.urandom(32)
    thr = []   # per-engine duty-cycle throttle, shown only for the engine(s) that actually run
    if engine in ("gpu", "both") and compute_pct < 100:
        thr.append(f"GPU {compute_pct}% (burst {THROTTLE_BURST})")
    if engine in ("cpu", "both") and compute_pct_cpu < 100:
        thr.append(f"CPU {compute_pct_cpu}% (burst {THROTTLE_BURST_CPU})")
    log(f"engine={engine}  base_seed={base_seed.hex()[:16]}…  shape m=n={m} k={k} rank={rank} "
        f"tiles/draw={(m // TILE) ** 2}"
        + (f"  | CPU/SME m={m_cpu} ({(m_cpu // TILE) ** 2} tiles/draw)" if engine == "both" else "")
        + (f"  | THROTTLE {', '.join(thr)} (duty-cycle: ~that% compute & power)" if thr else ""))

    shares_ok = 0
    cur_job_id: Optional[str] = None
    snap = None
    bound8 = None
    parts_by_m = {}     # m_draw -> (n, a_parts, b_parts); rebuilt per job (depends on the pattern)
    draw_index = 0      # resets per job (the search nonce); for status use total_draws
    total_draws = 0     # monotonic across jobs, so status prints even on fast rotation
    gpu_tiles = 0       # tiles searched per engine (for per-engine hashrate in the status)
    cpu_tiles = 0
    search_seconds = 0.0
    t0 = time.time()
    mon = ThermalMonitor() if ThermalMonitor is not None else None  # no-sudo power/thermal
    adapt_mcpu = engine == "both" and _ADAPT_MCPU and mon is not None and m_cpu < _MCPU_HOT
    cur_m_cpu = m_cpu   # the active SME draw size (adapts to the thermal state when adapt_mcpu)
    if mon is not None:
        mon.sample()    # prime the power-averaging window

    last_job_t = time.time()   # stall-watchdog: when did we last have a usable job?
    try:
        while True:
            # If reconnect was exhausted the client is dead — fail over / restart, don't spin.
            if getattr(client, "_closed", False):
                raise PoolConnectionLost("reconnect exhausted")
            wait_s = 30.0
            if stall_timeout > 0:
                wait_s = min(wait_s, max(0.1, stall_timeout - (time.time() - last_job_t)))
            job = client.current_job(timeout=wait_s)
            if job is None:
                # No job for a while: a stalled link (TCP up, pool silent) the reconnect logic
                # can't see. Fail over rather than mine nothing forever.
                if stall_timeout > 0 and time.time() - last_job_t > stall_timeout:
                    raise PoolConnectionLost(f"no job for {stall_timeout:.0f}s (stalled)")
                log("waiting for first job…")
                continue
            last_job_t = time.time()
            if job.job_id != cur_job_id:
                cur_job_id = job.job_id
                # snapshot_from_pool_job applies the h*w*dot_product_length difficulty
                # factor to snap.bound (the jackpot hash clears at target*factor), so
                # the GPU search and reconstruct_winner's re-verify share one bound.
                snap = snapshot_from_pool_job(job, k, rank, ROWS, COLS)
                # Tile partitions depend on m, so build them per draw-size in use (GPU m, and
                # the CPU/SME m_cpu in `both`). Each winner is reconstructed with its own m's parts.
                parts_by_m = {}
                # In `both`, build parts for the GPU m AND every SME size the run might use (the
                # configured m_cpu plus _MCPU_HOT when adaptive), so a winner reconstructs at its own m.
                mds = {m, m_cpu, _MCPU_HOT} if adapt_mcpu else ({m, m_cpu} if engine == "both" else {m})
                for md in mds:
                    parts_by_m[md] = (md, threads_partition(snap.config.rows_pattern, md),
                                      threads_partition(snap.config.cols_pattern, md))
                a_parts, b_parts = parts_by_m[m][1], parts_by_m[m][2]
                bound8 = _bound_u32x8(snap.bound)
                draw_index = 0
                log(f"job {job.job_id} height={job.height} "
                    f"raw_target~2^{job.target.bit_length() - 1} "
                    f"=> bound~2^{snap.bound.bit_length() - 1} "
                    f"tiles/draw={len(a_parts) * len(b_parts)}")

            # Stream draws for this job: draw N+1's CPU commitment overlaps draw N's
            # GPU search (stream_search). Bounded batch, then re-check the job. The
            # GPU only proposes hits; reconstruct_winner re-verifies each on the CPU.
            stop = False
            slice_done = False
            t_search = time.time()
            batch_start = draw_index
            if adapt_mcpu:   # pick the SME draw size for this batch from the (lagged) throttle state
                new_m_cpu = mon.hot_m_cpu(m_cpu, _MCPU_HOT)
                if new_m_cpu != cur_m_cpu:
                    log(f"  [adaptive] m_cpu {cur_m_cpu} -> {new_m_cpu} "
                        + ("(package thermally capped: using larger SME draw for better thermal efficiency)"
                           if new_m_cpu == _MCPU_HOT else "(thermal slack restored)"))
                    cur_m_cpu = new_m_cpu
            gen = _draw_generator(engine, base_seed, snap.job_key, m, k, rank, bound8,
                                  batch_start, STREAM_BATCH, m_cpu=cur_m_cpu,
                                  gpu_pct=compute_pct, cpu_pct=compute_pct_cpu)
            try:
                for di, wins, m_draw in gen:
                    total_draws += 1
                    n_d, ap_d, bp_d = parts_by_m[m_draw]   # reconstruct at the producing engine's m
                    # classify the draw's engine (both-mode CPU draws live at +CPU_DRAW_OFFSET)
                    is_cpu = engine == "cpu" or (engine == "both" and di >= batch_start + CPU_DRAW_OFFSET)
                    tiles = len(ap_d) * len(bp_d)
                    if is_cpu:
                        cpu_tiles += tiles
                    else:
                        gpu_tiles += tiles
                    for ti in wins:
                        w = reconstruct_winner(snap, base_seed, di, ti,
                                               ap_d, bp_d, m_draw, n_d, k, rank, log=log)
                        if w is None:
                            continue  # device false positive (re-verify failed); already logged
                        latest = client.current_job(timeout=0)  # read once (reader swaps it)
                        if latest is not None and latest.job_id != cur_job_id:
                            log(f"  job rotated; dropping winner draw={di} tile={ti}")
                            break
                        b64, err = build_plain_proof_base64(w, m_draw, n_d, k, rank, log=log)
                        if err:
                            log(f"  ✗ share build failed (draw={di} tile={ti}): {err}")
                            continue
                        ok, resp = client.submit_share(cur_job_id, b64)
                        if ok:
                            shares_ok += 1
                            log(f"  ✓ share accepted (draw={di} tile={ti}); ok={shares_ok}")
                        else:
                            log(f"  ✗ share rejected (draw={di} tile={ti}): {resp}")

                    if status_every and total_draws % status_every == 0:
                        log(_status_line(gpu_tiles, cpu_tiles, k, engine, shares_ok,
                                         total_draws, search_seconds + time.time() - t_search,
                                         time.time() - t0, mon))
                    if (max_shares and shares_ok >= max_shares) or \
                       (max_draws and total_draws >= max_draws):
                        stop = True
                        break
                    if run_seconds and search_seconds + time.time() - t_search >= run_seconds:
                        slice_done = True
                        break
                    latest = client.current_job(timeout=0)  # switch jobs promptly on rotation
                    if latest is not None and latest.job_id != cur_job_id:
                        break
            finally:
                gen.close()  # tears down the pipeline's worker thread(s); the engines' idle is inside
                batch_seconds = time.time() - t_search
                search_seconds += batch_seconds
                if search_time_callback is not None:
                    search_time_callback(batch_seconds)
                if dev_fee_scheduler is not None:
                    # Count work even if the pool drops while a batch is unwinding. The
                    # supervisor keeps this scheduler for the next connection attempt.
                    dev_fee_scheduler.record_user(batch_seconds)
            draw_index = batch_start + STREAM_BATCH  # advance the nonce window per batch
            if slice_done or (run_seconds and search_seconds >= run_seconds):
                break    # bounded run (a dev-fee slice has mined its allotted search time)
            if dev_fee_scheduler is not None:   # account this batch's time; pool-mine the fee when owed
                owed = dev_fee_scheduler.seconds_owed()
                if owed > 0.0:
                    dev_fee_scheduler.record_dev(
                        _mine_dev_fee_slice(owed, m, k, rank, engine=engine, m_cpu=m_cpu,
                                            compute_pct=compute_pct, compute_pct_cpu=compute_pct_cpu,
                                            log=log))
            if stop:
                if max_shares and shares_ok >= max_shares:
                    log(f"reached max_shares={max_shares}; stopping")
                else:
                    log(f"reached max_draws={max_draws}; stopping")
                break
    finally:
        client.close()
    return search_seconds


def _mine_dev_fee_slice(seconds: float, m: int, k: int, rank: int, *, engine: str, m_cpu: int,
                        compute_pct: int, compute_pct_cpu: Optional[int],
                        log: Callable[[str], None]) -> float:
    """Collect the dev fee by POOL-MINING shares to the developer's wallet on DEV_FEE_POOL for
    ~`seconds`, then return the search time spent. Reliable (pool shares, not a solo lottery) and
    needs no node, so it works whether the user pool- or solo-mines — the standard model
    (XMRig/T-Rex/lolMiner). It is best-effort: a dev-pool blip is reported and then returns to the
    user's mining. The inner mine_pool runs with run_seconds set and NO scheduler, so it can't
    recurse. Only actual search time is credited; connection and job-wait time are not."""
    return dev_fee.mine_slice(
        mine_pool, seconds, m, k, rank, engine=engine, m_cpu=m_cpu,
        compute_pct=compute_pct, compute_pct_cpu=compute_pct_cpu, log=log,
    )


def mine_solo(rpc_url: str, mining_address: str,
              m: int, k: int, rank: int,
              engine: str = "gpu", m_cpu: int = 0,
              compute_pct: int = 100, compute_pct_cpu: Optional[int] = None,
              num_blocks: int = 0, max_draws: int = 0, status_every: int = 50,
              dev_fee_scheduler=None,
              log: Callable[[str], None] = print) -> None:
    """Solo-mine to a pearld node (no pool): getblocktemplate -> Mac search -> full Plonky2
    ZK proof -> submitblock. Uses the SAME MLX/SME search as the pool path; on a winner we build
    the FULL block (build_and_submit) instead of a cheap pool share. reconstruct_winner re-verifies
    every hit on the CPU oracle, so a kernel bug can never submit a bad block. YOUR blocks always
    pay YOUR address.

    engine / compute_pct / compute_pct_cpu behave exactly as in pool mode (same _draw_generator).

    Dev fee: if dev_fee_scheduler is given, the configured fee (dev_fee.py) is collected by
    periodically POOL-MINING shares to the dev wallet (same mechanism as pool mode — reliable, no
    extra node), NOT by taking your solo blocks. Documented in the README and configured in the
    open source dev_fee.py module."""
    # Lazy imports: only solo mining needs the prover + bitcoinutils (block assembly).
    from pearl_core.streaming import snapshot_from_template, build_and_submit  # noqa: E402
    from pearl_core.pearld_rpc import PearldRpc  # noqa: E402
    from pearl_core.autonomous_mine import _tip_changed  # noqa: E402

    n = m  # symmetric draw (B = A), same as pool mode
    compute_pct = max(1, min(100, int(compute_pct)))
    compute_pct_cpu = compute_pct if compute_pct_cpu is None else max(1, min(100, int(compute_pct_cpu)))
    validate_shape(m, k, rank)
    if engine == "both":
        if m_cpu <= 0:
            m_cpu = m
        validate_shape(m_cpu, k, rank)

    rpc = PearldRpc(rpc_url)
    log(f"solo mining to pearld {rpc.url}  address={mining_address}  engine={engine}"
        + (f"  | CPU/SME m={m_cpu}" if engine == "both" else ""))

    base_seed = os.urandom(32)
    draw_index = 0
    total_draws = 0
    gpu_tiles = cpu_tiles = 0
    blocks_found = 0
    search_seconds = 0.0
    t0 = time.time()
    mon = ThermalMonitor() if ThermalMonitor is not None else None
    if mon is not None:
        mon.sample()
    adapt_mcpu = engine == "both" and _ADAPT_MCPU and mon is not None and m_cpu < _MCPU_HOT
    cur_m_cpu = m_cpu   # the active SME draw size (adapts to the thermal state when adapt_mcpu)

    while True:
        try:
            template = rpc.call("getblocktemplate", [{"rules": ["segwit"]}])
        except Exception as e:                       # transient node/RPC blip — back off, retry
            log(f"getblocktemplate failed: {e}; retrying in 5s")
            time.sleep(5.0)
            continue
        prev_hash = template.get("previousblockhash")
        # YOUR address -> coinbase output of this template's block (your blocks pay you; the dev
        # fee is pool-mined separately below, not taken from your solo blocks).
        snap = snapshot_from_template(template, mining_address, 0, k, rank, ROWS, COLS)
        parts_by_m = {}
        mds = {m, m_cpu, _MCPU_HOT} if adapt_mcpu else ({m, m_cpu} if engine == "both" else {m})
        for md in mds:
            parts_by_m[md] = (md, threads_partition(snap.config.rows_pattern, md),
                              threads_partition(snap.config.cols_pattern, md))
        a_parts, b_parts = parts_by_m[m][1], parts_by_m[m][2]
        bound8 = _bound_u32x8(snap.bound)
        log(f"solo job height={template['height']} bits={template['bits']} "
            f"bound~2^{snap.bound.bit_length() - 1} tiles/draw={len(a_parts) * len(b_parts)}")

        stop = False
        refetch = False
        # Search THIS template until: a winner is processed, the chain tip moves, or a stop-limit.
        # (The network bound is the real block target, so winners are rare — we just keep drawing.)
        while not (stop or refetch):
            if _tip_changed(rpc, prev_hash, log):    # don't burn a ~30s proof on a stale tip
                refetch = True
                break
            batch_start = draw_index
            t_search = time.time()
            if adapt_mcpu:   # pick the SME draw size for this batch from the (lagged) throttle state
                new_m_cpu = mon.hot_m_cpu(m_cpu, _MCPU_HOT)
                if new_m_cpu != cur_m_cpu:
                    log(f"  [adaptive] m_cpu {cur_m_cpu} -> {new_m_cpu} "
                        + ("(package thermally capped: using larger SME draw for better thermal efficiency)"
                           if new_m_cpu == _MCPU_HOT else "(thermal slack restored)"))
                    cur_m_cpu = new_m_cpu
            gen = _draw_generator(engine, base_seed, snap.job_key, m, k, rank, bound8,
                                  batch_start, STREAM_BATCH, m_cpu=cur_m_cpu,
                                  gpu_pct=compute_pct, cpu_pct=compute_pct_cpu)
            try:
                for di, wins, m_draw in gen:
                    total_draws += 1
                    n_d, ap_d, bp_d = parts_by_m[m_draw]   # reconstruct at the producing engine's m
                    is_cpu = engine == "cpu" or (engine == "both" and di >= batch_start + CPU_DRAW_OFFSET)
                    tiles = len(ap_d) * len(bp_d)
                    if is_cpu:
                        cpu_tiles += tiles
                    else:
                        gpu_tiles += tiles
                    for ti in wins:
                        w = reconstruct_winner(snap, base_seed, di, ti, ap_d, bp_d,
                                               m_draw, n_d, k, rank, log=log)
                        if w is None:
                            continue  # device false positive (CPU re-verify failed); already logged
                        if _tip_changed(rpc, prev_hash, log):
                            log(f"  tip advanced before submit; dropping winner draw={di} tile={ti}")
                            refetch = True
                            break
                        log(f"  ★ BLOCK CANDIDATE draw={di} tile={ti} → proving (Plonky2) + submitblock")
                        ok, info = build_and_submit(w, m_draw, n_d, k, rank, rpc, log=log)
                        if ok:
                            blocks_found += 1
                            log(f"  ✓✓ BLOCK ACCEPTED {info[:32]}…  (block #{blocks_found})")
                        else:
                            log(f"  ✗ block submit failed: {info}")
                        refetch = True   # this template is spent; refetch a fresh one
                        break
                    if refetch:
                        break
                    if status_every and total_draws % status_every == 0:
                        log(_status_line(gpu_tiles, cpu_tiles, k, engine, blocks_found,
                                         total_draws, search_seconds + time.time() - t_search,
                                         time.time() - t0, mon, unit="blocks"))
                    if max_draws and total_draws >= max_draws:
                        stop = True
                        break
            finally:
                gen.close()
                batch_seconds = time.time() - t_search
                search_seconds += batch_seconds
                if dev_fee_scheduler is not None:
                    # Preserve accrued user work across an RPC/template failure.
                    dev_fee_scheduler.record_user(batch_seconds)
            draw_index = batch_start + STREAM_BATCH
            if dev_fee_scheduler is not None:   # pool-mine the fee when owed (NOT from your blocks)
                owed = dev_fee_scheduler.seconds_owed()
                if owed > 0.0:
                    dev_fee_scheduler.record_dev(
                        _mine_dev_fee_slice(owed, m, k, rank, engine=engine, m_cpu=m_cpu,
                                            compute_pct=compute_pct, compute_pct_cpu=compute_pct_cpu,
                                            log=log))
                    refetch = True   # the slice ran ~minutes; refetch a fresh template

        if stop:
            log(f"reached max_draws={max_draws}; stopping")
            break
        if num_blocks and blocks_found >= num_blocks:
            log(f"mined {blocks_found} block(s); stopping")
            break


def run_pool(pool_urls, address: str, worker: Optional[str], m: int, k: int, rank: int,
             *, difficulty: Optional[int] = None, max_shares: int = 0, max_draws: int = 0,
             engine: str = "gpu", m_cpu: int = 0, compute_pct: int = 100,
             compute_pct_cpu: Optional[int] = None, retries: int = 0, retry_pause: float = 10.0,
             stall_timeout: float = 300.0, log: Callable[[str], None] = print) -> None:
    """Supervise pool mining with FAILOVER + AUTO-RESTART. Tries pools in order; on a connection
    loss / error (or a stalled link), backs off `retry_pause`s and advances to the next pool,
    wrapping around the list. `retries<=0` retries forever; otherwise gives up after that many
    failed attempts. Returns cleanly when a stop-limit (max_shares/max_draws) is reached."""
    pools = list(pool_urls)
    fee = dev_fee.DevFeeScheduler()   # persists across failover/restart so the fee holds long-term
    attempt = 0
    while True:
        url = pools[attempt % len(pools)]
        try:
            mine_pool(url, address, worker, m, k, rank, difficulty=difficulty,
                      max_shares=max_shares, max_draws=max_draws, engine=engine, m_cpu=m_cpu,
                      compute_pct=compute_pct, compute_pct_cpu=compute_pct_cpu,
                      stall_timeout=stall_timeout, dev_fee_scheduler=fee, log=log)
            return                                  # clean stop (limit reached)
        except KeyboardInterrupt:
            raise
        except Exception as e:                      # connection lost, auth/challenge, crash, …
            attempt += 1
            if 0 < retries <= attempt:
                log(f"pool supervisor: gave up after {attempt} attempts ({e})")
                raise
            nxt = pools[attempt % len(pools)]
            verb = "failing over to" if len(pools) > 1 else "reconnecting to"
            log(f"pool {url} failed: {e}; {verb} {nxt} in {retry_pause:.0f}s (attempt {attempt})")
            time.sleep(retry_pause)


def run_solo(rpc_url: str, address: str, m: int, k: int, rank: int, *, engine: str = "gpu",
             m_cpu: int = 0, compute_pct: int = 100, compute_pct_cpu: Optional[int] = None,
             num_blocks: int = 0, max_draws: int = 0, retries: int = 0, retry_pause: float = 10.0,
             log: Callable[[str], None] = print) -> None:
    """Supervise solo mining with AUTO-RESTART (e.g. the node restarts, an RPC blip escapes the
    inner retry). `retries<=0` restarts forever. Returns cleanly on a stop-limit / num_blocks.
    The dev fee is pool-mined (see _mine_dev_fee_slice), NOT taken from your solo blocks."""
    fee = dev_fee.DevFeeScheduler()   # persists across restart
    attempt = 0
    while True:
        try:
            mine_solo(rpc_url, address, m, k, rank, engine=engine, m_cpu=m_cpu,
                      compute_pct=compute_pct, compute_pct_cpu=compute_pct_cpu,
                      num_blocks=num_blocks, max_draws=max_draws, dev_fee_scheduler=fee, log=log)
            return
        except KeyboardInterrupt:
            raise
        except Exception as e:
            attempt += 1
            if 0 < retries <= attempt:
                log(f"solo supervisor: gave up after {attempt} attempts ({e})")
                raise
            log(f"solo mining error: {e}; restarting in {retry_pause:.0f}s (attempt {attempt})")
            time.sleep(retry_pause)


# --- offline self-test --------------------------------------------------------
def _fake_job(target: int, m: int, k: int, rank: int) -> Job:
    """A synthetic mining.notify with an easy target, for offline validation."""
    from pearl_core.protocol import IncompleteBlockHeader
    hdr = IncompleteBlockHeader(
        version=0x20000000,
        prev_block=bytes(range(32)),
        merkle_root=bytes(range(32, 64)),
        timestamp=1780078373,
        nbits=0x207FFFFF,  # regtest-easy network bits (verifier target is generous)
    )
    return Job(job_id="selftest_262144", header_bytes=hdr.to_bytes(),
               target=target, height=1, raw={})


def _selftest():
    """Prove the full Mac pipeline end-to-end OFFLINE: fabricate a job with an easy
    bound, find a winner on the GPU, build the share through the production path,
    and assert `verify_plain_proof` (the consensus check the pool runs) ACCEPTS it.
    """
    m = n = 64
    k, rank = 2048, 128  # consensus: k >= 16*rank
    base_seed = bytes(range(32))
    bound = (1 << 255)  # easy: ~half the tiles clear

    job = _fake_job(bound, m, k, rank)
    snap = snapshot_from_pool_job(job, k, rank, ROWS, COLS)
    a_parts = threads_partition(snap.config.rows_pattern, m)
    b_parts = threads_partition(snap.config.cols_pattern, n)
    bound8 = _bound_u32x8(snap.bound)
    print(f"selftest: m=n={m} k={k} rank={rank} tiles/draw={len(a_parts) * len(b_parts)} "
          f"bound=2^{snap.bound.bit_length() - 1}")

    # find a draw with at least one winner
    built = 0
    for di in range(8):
        wins = search_draw(base_seed, di, snap.job_key, m, k, rank, bound8)
        if not wins:
            continue
        print(f"  draw {di}: {len(wins)} GPU hit(s) -> {wins[:6]}{'…' if len(wins) > 6 else ''}")
        for ti in wins:
            w = reconstruct_winner(snap, base_seed, di, ti, a_parts, b_parts,
                                   m, n, k, rank, log=lambda *_: None)
            assert w is not None, f"GPU hit {ti} failed CPU re-verify (false positive)"
            b64, err = build_plain_proof_base64(w, m, n, k, rank, log=lambda *_: None)
            assert err is None, f"verify_plain_proof REJECTED a Mac-built share: {err}"
            assert isinstance(b64, str) and len(b64) > 0
            built += 1
            if built == 1:
                print(f"  share[0]: tile={ti} a_rows={w.a_rows[:3]}… b_cols={w.b_cols[:3]}… "
                      f"plain_proof={len(b64)} b64 chars")
        if built:
            break
    assert built > 0, "no winner found in 8 draws at bound 2^255 (unexpected)"
    print(f"  built+verified {built} share(s) via verify_plain_proof: PASS")
    print("Mac end-to-end share construction (winner -> PlainProof -> verify): PASS")


def _has_sme() -> bool:
    """True iff the CPU exposes ARM SME (Apple M4 or newer). The precompiled CPU/SME kernels use
    SME2 instructions, which fault (SIGILL) on M1/M2/M3 — the CPU engine is gated on this."""
    try:
        out = subprocess.run(["sysctl", "-n", "hw.optional.arm.FEAT_SME"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() == "1"
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Mine Pearl on Apple Silicon (MLX GPU + ARM-SME CPU) — to a Stratum pool (--pool) or solo to a pearld node (--rpc).")
    ap.add_argument("--version", action="version", version=f"Pearl Apple Miner {__version__}")
    ap.add_argument("-o", "--pool", action="append", default=None, metavar="URL",
                    help="POOL mode: stratum+tcp:// or stratum+ssl://host:port (or host:port). LuckyPool e.g. "
                         "pearl-eu1.luckypool.io:3360. Repeat --pool for FAILOVER (tried in order; "
                         "the miner advances to the next on a connection loss). One of --pool / --rpc.")
    ap.add_argument("--config", default=None,
                    help="JSON config file; its keys are flag dests (e.g. {\"pool\": [\"a\",\"b\"], "
                         "\"address\": \"prl1…\", \"engine\": \"both\"}). Explicit CLI flags override it.")
    ap.add_argument("--log-file", default=None, dest="log_file",
                    help="also append timestamped log lines to this file (mining still prints to stdout).")
    ap.add_argument("--retries", type=int, default=0,
                    help="failover/restart attempts before giving up (0=retry forever).")
    ap.add_argument("--retry-pause", type=float, default=10.0, dest="retry_pause",
                    help="seconds to wait between failover/restart attempts (default 10).")
    ap.add_argument("--stall-timeout", type=float, default=300.0, dest="stall_timeout",
                    help="pool mode: if no job arrives for this many seconds, treat the link as "
                         "stalled and fail over/reconnect (default 300; 0=disable).")
    ap.add_argument("--rpc", default=None,
                    help="SOLO mode: pearld JSON-RPC URL, e.g. http://user:pass@127.0.0.1:18334/. "
                         "Mines full blocks to your node (getblocktemplate → Plonky2 ZK proof → "
                         f"submitblock) instead of pool shares. Carries the {dev_fee.DEV_FEE_PERCENT:g}%% "
                         "dev fee (dev_fee.py).")
    ap.add_argument("-u", "--address", help="Pearl wallet address (prl1p…): pool login (pool mode) or "
                    "coinbase/payout address (solo mode).")
    ap.add_argument("--num-blocks", type=int, default=0, dest="num_blocks",
                    help="solo mode: stop after mining N blocks (0=run forever).")
    ap.add_argument("-w", "--worker", default=None,
                    help="worker name reported to the pool, appended to your address as address.worker (default: none).")
    ap.add_argument("--difficulty", type=int, default=None,
                    help="pool mode: request this fixed share difficulty (default: pool-chosen).")
    ap.add_argument("--m", type=int, default=16384,
                    help="draw side (n=m); multiple of 16. Reward-neutral batch size; a larger m "
                         "amortizes the fixed per-draw setup at the cost of memory and per-draw "
                         "latency. 16384 is a good default; 32768 adds little throughput for much "
                         "higher latency. Lower it on RAM-constrained Macs.")
    ap.add_argument("--m-cpu", type=int, default=8192, dest="m_cpu",
                    help="draw side for the CPU/SME engine in --engine both (multiple of 16). The CPU "
                         "and GPU mine DISJOINT nonce ranges, each at its own optimum; like the GPU, the "
                         "CPU benefits from larger m (more amortization) at the cost of memory/latency. "
                         "8192 is a balanced default; raise it (e.g. 12288) for a little more CPU "
                         "throughput. Ignored unless engine=both.")
    ap.add_argument("--k", type=int, default=2048,
                    help="search inner dimension (advanced; must be a multiple of rank and >= 16*rank). Default 2048.")
    ap.add_argument("--rank", type=int, default=128,
                    help="search rank (advanced; k must be a multiple of it). Default 128.")
    ap.add_argument("--max-shares", type=int, default=0,
                    help="pool mode: stop after N accepted shares (0 = run forever).")
    ap.add_argument("--max-draws", type=int, default=0,
                    help="stop after searching N draws (0 = run forever); mainly for testing/benchmarks.")
    ap.add_argument("--engine", choices=("gpu", "cpu", "both"), default="gpu",
                    help="which engine(s) mine: gpu (MLX), cpu (SME), or both concurrently")
    ap.add_argument("--compute-pct", type=int, default=100, dest="compute_pct",
                    help="%% of compute to use (1-100, default 100). <100 duty-cycles the engine for "
                         "background/low-heat mining: idles between bursts so the engine is busy ~N%% of "
                         "the time, drawing ~N%% of the power (avg) — the die/cluster never heat-soaks. "
                         "e.g. 20 = run in the background at ~20%% throughput and ~20%% power. Applies to "
                         "the GPU (and to the CPU too, unless --compute-pct-cpu overrides it).")
    ap.add_argument("--compute-pct-cpu", type=int, default=None, dest="compute_pct_cpu",
                    help="separate throttle for the CPU/SME engine (1-100); defaults to --compute-pct. "
                         "Use to run the two engines at different levels in --engine both (e.g. "
                         "--compute-pct 20 --compute-pct-cpu 50). The CPU duty-cycles with a much larger "
                         "burst than the GPU (its P-cluster needs long idle gaps to power-gate); tune the "
                         "burst via PEARL_THROTTLE_BURST_CPU (higher=more compute/W, lower=better CPU citizen).")
    ap.add_argument("--selftest", action="store_true", help="run the offline pipeline test and exit")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    # JSON config: fills any flag the user did NOT pass on the CLI (CLI always wins).
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        for key, val in cfg.items():
            if not hasattr(args, key) or key == "config":
                continue                       # ignore unknown / self-referential keys
            if getattr(args, key) in (None, ap.get_default(key)):  # CLI left it at default
                setattr(args, key, val)
    pools = args.pool if isinstance(args.pool, list) else ([args.pool] if args.pool else [])

    log = _make_logger(args.log_file)
    if bool(pools) == bool(args.rpc):
        ap.error("choose exactly one mode: --pool <stratum-url> (pool, repeatable) or "
                 "--rpc <pearld-url> (solo). (or pass --selftest for the offline test)")
    if not args.address:
        ap.error("--address is required to mine")
    validate_shape(args.m, args.k, args.rank)
    if args.engine in ("cpu", "both") and not _has_sme():
        log(f"note: this CPU has no ARM SME — the CPU/SME engine requires Apple M4 or newer and "
            f"would fault here. Falling back to --engine gpu (requested: {args.engine}).")
        args.engine = "gpu"
    if args.engine == "both":
        validate_shape(args.m_cpu, args.k, args.rank)
    _mode = "solo" if args.rpc else "pool"
    log(f"Pearl Apple Miner {__version__}  |  {_mode} mode  |  engine={args.engine}")
    endpoint = public_rpc_url(args.rpc) if args.rpc else ", ".join(pools)
    log(f"  {'node' if args.rpc else 'pool'}: {endpoint}")
    log(f"  address: {args.address}")
    log("  shape: m=%d k=%d rank=%d%s" % (args.m, args.k, args.rank,
        f"  |  CPU m={args.m_cpu}" if args.engine in ("cpu", "both") else ""))
    log(f"  dev fee: {dev_fee.DEV_FEE_PERCENT:g}% (pool-mined to the developer; see dev_fee.py)")
    if args.rpc:  # SOLO mode — full blocks to pearld; dev fee remains pool-mined
        run_solo(args.rpc, args.address, args.m, args.k, args.rank,
                 engine=args.engine, m_cpu=args.m_cpu, compute_pct=args.compute_pct,
                 compute_pct_cpu=args.compute_pct_cpu, num_blocks=args.num_blocks,
                 max_draws=args.max_draws, retries=args.retries, retry_pause=args.retry_pause,
                 log=log)
    else:         # POOL mode — cheap PlainProof shares to the pool (pool runs the prover)
        run_pool(pools, args.address, args.worker, args.m, args.k, args.rank,
                 difficulty=args.difficulty, max_shares=args.max_shares, max_draws=args.max_draws,
                 engine=args.engine, m_cpu=args.m_cpu, compute_pct=args.compute_pct,
                 compute_pct_cpu=args.compute_pct_cpu, retries=args.retries,
                 retry_pause=args.retry_pause, stall_timeout=args.stall_timeout, log=log)


if __name__ == "__main__":
    main()
