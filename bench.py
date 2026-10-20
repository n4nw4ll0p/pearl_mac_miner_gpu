"""Offline hashrate benchmark for the MLX GPU path (no pool, no wallet).

Usage:
    python bench.py --m 8192 --k 2048 --rank 128 --draws 8 --warmup 1

Reports draws/s, tiles/s and Pearl TH/s using the same accounting as miner.py
(TH/s = tiles * 16*16*k / seconds / 1e12).
"""
from __future__ import annotations

import argparse
import time

from fast_search import search_mask_fast  # noqa: F401  (import side effects)
from stream_search import search_draws_overlapped

import miner as M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=8192)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="run for this long instead of --draws; reports per-chunk rate")
    ap.add_argument("--chunk", type=int, default=32, help="draws per reporting chunk")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--bound-bits", type=int, default=1,
                    help="log2 of the synthetic share bound (1 = essentially no winners)")
    args = ap.parse_args()

    m, k, rank = args.m, args.k, args.rank
    if m % M.TILE:
        raise SystemExit(f"m must be a multiple of {M.TILE}")
    if k % rank or k < 16 * rank:
        raise SystemExit("k must be a multiple of rank and >= 16*rank")

    base_seed = bytes(range(32))
    bound = 1 << args.bound_bits
    job = M._fake_job(bound, m, k, rank)
    snap = M.snapshot_from_pool_job(job, k, rank, M.ROWS, M.COLS)
    bound8 = M._bound_u32x8(snap.bound)
    tiles_per_draw = (m // M.TILE) ** 2

    def run(n: int) -> float:
        t0 = time.perf_counter()
        for _di, _wins in search_draws_overlapped(
            base_seed, snap.job_key, m, k, rank, bound8, range(n)
        ):
            pass
        return time.perf_counter() - t0

    if args.warmup:
        run(args.warmup)

    if args.seconds > 0:
        try:
            from thermal import ThermalMonitor, pressure_name
            mon = ThermalMonitor()
        except Exception:
            mon = None

            def pressure_name(x):  # type: ignore[misc]
                return str(x)

        t_end = time.perf_counter() + args.seconds
        total_draws = total_tiles = 0
        t_start = time.perf_counter()
        while time.perf_counter() < t_end:
            n = args.chunk
            t0 = time.perf_counter()
            for _di, _wins in search_draws_overlapped(
                base_seed, snap.job_key, m, k, rank, bound8, range(n)
            ):
                pass
            dt = time.perf_counter() - t0
            total_draws += n
            total_tiles += n * tiles_per_draw
            therm = ""
            if mon is not None:
                s = mon.sample()
                g = f"{s.gpu_w:.0f}W" if s.gpu_w is not None else "?W"
                therm = f" | {g} GPU pressure={pressure_name(s.pressure)} util={s.util_pct}%"
            print(
                f"  chunk {n} draws {dt:.2f}s -> {n / dt:.2f} draws/s "
                f"{n * tiles_per_draw * M.TILE * M.TILE * k / dt / 1e12:.3f} TH/s{therm}",
                flush=True,
            )
        dt = time.perf_counter() - t_start
        print(
            f"m={m} k={k} rank={rank} tiles/draw={tiles_per_draw} "
            f"draws={total_draws} time={dt:.1f}s "
            f"average TH/s={total_tiles * M.TILE * M.TILE * k / dt / 1e12:.3f}"
        )
        return

    dt = run(args.draws)

    draws_s = args.draws / dt
    tiles_s = draws_s * tiles_per_draw
    th = tiles_s * M.TILE * M.TILE * k / 1e12

    print(
        f"m={m} k={k} rank={rank} tiles/draw={tiles_per_draw} "
        f"draws={args.draws} time={dt:.3f}s "
        f"draws/s={draws_s:.4f} tiles/s={tiles_s:.1f} TH/s={th:.4f}"
    )


if __name__ == "__main__":
    main()
