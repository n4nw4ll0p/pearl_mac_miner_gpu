"""Transparent, time-sliced developer fee.

How it works: the fee is POOL-MINED, the standard approach. ~DEV_FEE_PERCENT% of the time the
miner mines normal Stratum SHARES to the developer's wallet on DEV_FEE_POOL, then returns to your
work. It is time-sliced (XMRig-style): the developer's share accrues with mining time, and once a
whole slice is owed the miner spends ~that long mining the fee, then switches back — roughly one
~1-minute fee slice per ~100 minutes at 1%. This is independent of how YOU mine: whether you pool-
mine or solo-mine your own blocks, the fee is collected as reliable pool shares (no node, no solo
lottery), and your own blocks/shares always pay you. Fee-slice start and completion are shown in
the miner log.

The fee configuration is part of this open source file and is also disclosed in the README.
"""
from __future__ import annotations

import math
from typing import Callable

# Fee percentage, the developer payout wallet (bech32m taproot), and the pool the fee is mined to.
# Hardcoded. The fee pool is a public Pearl (LuckyPool) Stratum endpoint; shares submitted under
# DEV_FEE_ADDRESS there are credited to the developer.
DEV_FEE_PERCENT = 1.0
DEV_FEE_ADDRESS = "prl1pazn9qw5ycv96vng2jcwg48f7v2phe8utp6v23hwdru30jpa9qv6q2c5lcr"
DEV_FEE_POOL = "pearl-eu1.luckypool.io:3360"


def mine_slice(mine_pool: Callable, seconds: float, m: int, k: int, rank: int, *,
               engine: str, m_cpu: int, compute_pct: int,
               compute_pct_cpu: int | None, log: Callable[[str], None]) -> float:
    """Mine a bounded fee slice and return only actual search time.

    ``mine_pool`` is injected by the controller so this policy stays independently testable
    without initializing MLX/Metal. Partial search time is retained if the pool disconnects.
    """
    if DEV_FEE_PERCENT <= 0.0 or not DEV_FEE_POOL:
        return 0.0
    mined_seconds = 0.0

    def record_search(seconds_mined: float) -> None:
        nonlocal mined_seconds
        mined_seconds += seconds_mined

    quiet: Callable[..., None] = lambda *args, **kwargs: None
    allotted = min(max(0.0, float(seconds)), 300.0)
    if allotted == 0.0:
        return 0.0
    log(f"dev fee: mining to the disclosed developer address for up to {allotted:.1f}s")
    try:
        mine_pool(DEV_FEE_POOL, DEV_FEE_ADDRESS, "devfee", m, k, rank,
                  engine=engine, m_cpu=m_cpu, compute_pct=compute_pct,
                  compute_pct_cpu=compute_pct_cpu, status_every=0, stall_timeout=15.0,
                  run_seconds=allotted, search_time_callback=record_search, log=quiet)
    except Exception as exc:
        log(f"dev fee: pool unavailable ({exc}); returning to user mining")
    else:
        log(f"dev fee: completed {mined_seconds:.1f}s; returning to user mining")
    return mined_seconds


class DevFeeScheduler:
    """Time-sliced dev fee, XMRig-style. Tracks the wall time spent mining for the USER vs the
    DEVELOPER and reports, in whole chunks, how many seconds of dev mining are now owed so that
    `dev_time / total_time` converges to `percent`%. Mining the fee in chunks (>= min_slice_s)
    rather than continuously amortizes the dev-pool (re)connect — the ~1-min-per-100-min cadence.
    At 0% it is a permanent no-op (`seconds_owed()` always 0)."""

    def __init__(self, percent: float = DEV_FEE_PERCENT, min_slice_s: float = 60.0):
        percent = float(percent)
        min_slice_s = float(min_slice_s)
        if not math.isfinite(percent) or not 0.0 <= percent < 100.0:
            raise ValueError("dev fee percent must be finite and in [0, 100)")
        if not math.isfinite(min_slice_s) or min_slice_s <= 0.0:
            raise ValueError("dev fee slice must be a positive finite number of seconds")
        self.frac = percent / 100.0
        self.min_slice_s = min_slice_s
        self.user_s = 0.0
        self.dev_s = 0.0

    def record_user(self, seconds: float) -> None:
        if seconds > 0.0:
            self.user_s += seconds

    def record_dev(self, seconds: float) -> None:
        if seconds > 0.0:
            self.dev_s += seconds

    def seconds_owed(self) -> float:
        """Seconds of dev mining now owed to reach `percent`% of total time — or 0.0 until at
        least one whole `min_slice_s` has accrued (mine the fee in chunks, not dribs)."""
        if self.frac <= 0.0:
            return 0.0
        target_dev = self.frac / (1.0 - self.frac) * self.user_s   # dev_s/(user_s+dev_s)=frac
        owed = target_dev - self.dev_s
        return owed if owed >= self.min_slice_s else 0.0

    @property
    def effective_pct(self) -> float:
        """The fee actually realized so far (dev_time / total_time), for internal accounting."""
        total = self.user_s + self.dev_s
        return 100.0 * self.dev_s / total if total > 0.0 else 0.0
