"""Empirical check that the GPU bound test matches theory: run N draws at an easy bound
and compare observed winners with tiles * bound/2^256."""
from __future__ import annotations

import argparse

import miner as M
from stream_search import search_draws_overlapped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--bound-bits", type=int, default=240)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    base_seed = bytes(range(32))
    factor = M.TILE * M.TILE * args.k
    target = (1 << args.bound_bits) // factor  # snapshot re-applies `factor`
    job = M._fake_job(target, args.m, args.k, args.rank)
    snap = M.snapshot_from_pool_job(job, args.k, args.rank, M.ROWS, M.COLS)
    bound8 = M._bound_u32x8(snap.bound)
    tiles = args.draws * (args.m // M.TILE) ** 2
    p = snap.bound / (1 << 256)
    print(f"bound=2^{snap.bound.bit_length()-1} tiles={tiles} p={p:.3e} expected={tiles*p:.2f}")
    for t in range(args.trials):
        hits = tiles_hit = 0
        start = t * args.draws
        for _di, wins in search_draws_overlapped(
            base_seed, snap.job_key, args.m, args.k, args.rank, bound8,
            range(start, start + args.draws),
        ):
            hits += len(wins)
            tiles_hit += 1
        print(f"  trial {t}: observed wins={hits} (draws={tiles_hit})")


if __name__ == "__main__":
    main()
