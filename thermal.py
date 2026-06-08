"""No-sudo thermal/power monitor for the Apple Silicon Pearl miner.

The sustained-throughput limiter is GPU thermal throttle (throughput falls as the die heats and
the firmware lowers the GPU clock). This module exposes the signals needed to SEE that happen,
all readable WITHOUT root:

  - GPU power (W) and CPU power (W)  via IOReport "Energy Model" group (libIOReport.dylib)
  - thermal pressure level (0..)     via libSystem notify com.apple.system.thermalpressurelevel
  - GPU utilization %                via ioreg IOAccelerator PerformanceStatistics
  - per-draw ms (the direct clock proxy) is tracked by the caller (rising ms = falling clock)

powermetrics gives the same numbers but needs sudo; everything here is unprivileged.
Use `ThermalMonitor().sample()` for a snapshot, or run `python thermal.py` for a live demo.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import re
import resource
import subprocess
import time
from ctypes import POINTER, byref, c_char_p, c_int, c_int64, c_uint32, c_void_p
from dataclasses import dataclass

_UTF8 = 0x08000100

# --- thermal pressure (libSystem notify) ----------------------------------------------------
_libc = ctypes.CDLL(ctypes.util.find_library("System"))
_PRESSURE_TOK = None
_PRESSURE_NAMES = {0: "nominal", 1: "moderate", 2: "heavy", 3: "trapping", 4: "sleeping"}


def thermal_pressure() -> int:
    """com.apple.system.thermalpressurelevel as an int (0 nominal -> higher = more throttle
    pressure). Diverges from / is finer than NSProcessInfo.thermalState. No sudo."""
    global _PRESSURE_TOK
    try:
        if _PRESSURE_TOK is None:
            tok = c_int(0)
            if _libc.notify_register_check(b"com.apple.system.thermalpressurelevel", byref(tok)) != 0:
                return -1
            _PRESSURE_TOK = tok
        st = ctypes.c_uint64(0)
        _libc.notify_get_state(_PRESSURE_TOK, byref(st))
        return int(st.value)
    except Exception:
        return -1


def pressure_name(level: int) -> str:
    return _PRESSURE_NAMES.get(level, f"level{level}")


_POWERMODE_NAMES = {0: "automatic", 1: "low-power", 2: "high-power"}


def power_mode() -> int:
    """macOS powermode (no sudo to READ; sudo to set). 0=automatic/normal, 1=Low Power,
    2=High Power. HPM (2) is the #1 sustained-throttle lever; -1 if unreadable."""
    try:
        out = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=3).stdout
        m = re.search(r"powermode\s+(\d+)", out)
        return int(m.group(1)) if m else -1
    except Exception:
        return -1


def power_mode_name(mode: int) -> str:
    return _POWERMODE_NAMES.get(mode, f"mode{mode}")


def gpu_util_pct() -> int:
    """GPU 'Device Utilization %' from ioreg IOAccelerator (no sudo). Note: this field can read
    sticky/peak rather than instantaneous on some drivers — treat as a coarse busy indicator."""
    try:
        out = subprocess.run(["ioreg", "-r", "-c", "IOAccelerator", "-d", "1"],
                             capture_output=True, text=True, timeout=3).stdout
        m = re.search(r'"Device Utilization %"=(\d+)', out)
        return int(m.group(1)) if m else -1
    except Exception:
        return -1


# --- GPU/CPU power (IOReport "Energy Model", no sudo) ----------------------------------------
def _load_ioreport():
    IOR = ctypes.CDLL("libIOReport.dylib")
    CF = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    for f, res, arg in [
        ("IOReportCopyChannelsInGroup", c_void_p, [c_void_p, c_void_p, c_uint32, c_uint32, c_uint32]),
        ("IOReportCreateSubscription", c_void_p, [c_void_p, c_void_p, POINTER(c_void_p), c_uint32, c_void_p]),
        ("IOReportCreateSamples", c_void_p, [c_void_p, c_void_p, c_void_p]),
        ("IOReportCreateSamplesDelta", c_void_p, [c_void_p, c_void_p, c_void_p]),
        ("IOReportChannelGetChannelName", c_void_p, [c_void_p]),
        ("IOReportChannelGetUnitLabel", c_void_p, [c_void_p]),
        ("IOReportSimpleGetIntegerValue", c_int64, [c_void_p, c_int]),
    ]:
        fn = getattr(IOR, f); fn.restype = res; fn.argtypes = arg
    CF.CFStringCreateWithCString.restype = c_void_p
    CF.CFStringCreateWithCString.argtypes = [c_void_p, c_char_p, c_uint32]
    CF.CFDictionaryGetValue.restype = c_void_p; CF.CFDictionaryGetValue.argtypes = [c_void_p, c_void_p]
    CF.CFArrayGetCount.restype = c_int; CF.CFArrayGetCount.argtypes = [c_void_p]
    CF.CFArrayGetValueAtIndex.restype = c_void_p; CF.CFArrayGetValueAtIndex.argtypes = [c_void_p, c_int]
    CF.CFStringGetCStringPtr.restype = c_char_p; CF.CFStringGetCStringPtr.argtypes = [c_void_p, c_uint32]
    CF.CFStringGetCString.restype = c_int; CF.CFStringGetCString.argtypes = [c_void_p, c_char_p, c_int, c_uint32]
    CF.CFRelease.argtypes = [c_void_p]
    return IOR, CF


# energy unit label -> joules-per-unit
_J = {"nJ": 1e-9, "uJ": 1e-6, "mJ": 1e-3, "J": 1.0}


@dataclass
class ThermalSample:
    gpu_w: float | None      # GPU power, watts (None if IOReport unavailable)
    cpu_w: float | None      # CPU power, watts
    pressure: int            # thermalpressurelevel (0 nominal..; -1 unknown)
    util_pct: int            # GPU utilization % (coarse; -1 unknown)
    pmode: int               # powermode (0 auto, 1 low, 2 high-power; -1 unknown)
    window_s: float          # the wall-time the power was averaged over

    def line(self) -> str:
        g = f"{self.gpu_w:5.1f}W" if self.gpu_w is not None else "  ?  "
        c = f"{self.cpu_w:5.1f}W" if self.cpu_w is not None else "  ?  "
        return (f"GPU {g}  CPU {c}  pressure={self.pressure}({pressure_name(self.pressure)})  "
                f"util={self.util_pct}%  power={power_mode_name(self.pmode)}")


class ThermalMonitor:
    """Persistent IOReport subscription + pressure/util readers. Call sample() repeatedly;
    power is the AVERAGE over the wall-time since the previous sample() (or since construction)."""

    def __init__(self):
        try:
            self._ior, self._cf = _load_ioreport()
            grp = self._cf.CFStringCreateWithCString(None, b"Energy Model", _UTF8)
            chans = self._ior.IOReportCopyChannelsInGroup(grp, None, 0, 0, 0)
            sc = c_void_p()
            self._sub = self._ior.IOReportCreateSubscription(None, chans, byref(sc), 0, None)
            self._sc = sc
            self._key_channels = self._cf.CFStringCreateWithCString(None, b"IOReportChannels", _UTF8)
            self._prev = self._ior.IOReportCreateSamples(self._sub, self._sc, None)
            self._prev_t = time.perf_counter()
            self._ok = bool(self._sub and self._prev)
        except Exception:
            self._ok = False
        self._pmode = power_mode()   # a setting, not a live signal — read once
        self._burst_th = 0.0
        self._burst_pkg = 0.0     # peak GPU+CPU watts (cold boost); current/peak < 1 ⇒ clocks dropped
        self._tt_prev = None      # (cum_tiles, cum_search_s) at the previous throttle() call
        self._ru_prev = resource.getrusage(resource.RUSAGE_SELF)
        self._ru_t = time.perf_counter()
        self._last_ratio = None   # last recent/burst rate ratio (None until a burst is established)
        self._adapt_cur = None    # last adaptive m_cpu pick (hysteresis state for hot_m_cpu())

    def throttle(self, s: "ThermalSample", cum_tiles: float, hwk: int, cum_search_s: float):
        """Windowed throttle signal — call once per status tick with cumulative tiles + search-seconds
        and the just-taken `s`. Returns (recent_th, burst_th, pkg_w, burst_pkg_w, ivcsw_per_s): each
        *_th is the rate over THIS window, *burst* the peak (cold) ever seen, so recent/burst < 1 means
        the clock has dropped. ivcsw_per_s = involuntary context switches/s (oversubscription proxy)."""
        now = time.perf_counter()
        rec_th = None
        if self._tt_prev is not None:
            d_tiles = cum_tiles - self._tt_prev[0]
            d_s = cum_search_s - self._tt_prev[1]
            if d_s > 1e-6:
                rec_th = d_tiles * hwk / d_s / 1e12
                self._burst_th = max(self._burst_th, rec_th)
        self._tt_prev = (cum_tiles, cum_search_s)
        pkg = (s.gpu_w or 0.0) + (s.cpu_w or 0.0)
        self._burst_pkg = max(self._burst_pkg, pkg)
        ru = resource.getrusage(resource.RUSAGE_SELF)
        dtw = now - self._ru_t
        ivcsw = (ru.ru_nivcsw - self._ru_prev.ru_nivcsw) / dtw if dtw > 0 else 0.0
        self._ru_prev = ru
        self._ru_t = now
        self._last_ratio = (rec_th / self._burst_th) if (rec_th is not None and self._burst_th > 0) else None
        return rec_th, self._burst_th, pkg, self._burst_pkg, ivcsw

    def hot_m_cpu(self, base_m: int, hot_m: int, lo: float = 0.93, hi: float = 0.97) -> int:
        """Adaptive m_cpu policy for `both` mode (read-only on the last throttle()): return hot_m when
        the package is thermally capped (last windowed rate < lo*burst), base_m once recovered
        (> hi*burst). The hysteresis band [lo, hi] holds the prior pick so it doesn't flap. Returns
        base_m until a burst rate is established (when capped, the larger size runs the same
        work with less heat; otherwise the small size is used)."""
        r = self._last_ratio
        cur = self._adapt_cur if self._adapt_cur is not None else base_m
        if r is not None:
            if r < lo:
                cur = hot_m
            elif r > hi:
                cur = base_m
        self._adapt_cur = cur
        return cur

    def _name_unit(self, ch):
        def pystr(r):
            if not r:
                return None
            p = self._cf.CFStringGetCStringPtr(r, _UTF8)
            if p:
                return p.decode()
            b = ctypes.create_string_buffer(256)
            return b.value.decode() if self._cf.CFStringGetCString(r, b, 256, _UTF8) else None
        return pystr(self._ior.IOReportChannelGetChannelName(ch)), pystr(self._ior.IOReportChannelGetUnitLabel(ch))

    def _power(self):
        if not self._ok:
            return None, None, 0.0
        now = self._ior.IOReportCreateSamples(self._sub, self._sc, None)
        t = time.perf_counter()
        dt = t - self._prev_t
        delta = self._ior.IOReportCreateSamplesDelta(self._prev, now, None)
        self._cf.CFRelease(self._prev)
        self._prev = now
        self._prev_t = t
        gpu_w = cpu_w = None
        arr = self._cf.CFDictionaryGetValue(delta, self._key_channels) if delta else None
        n = self._cf.CFArrayGetCount(arr) if arr else 0
        for i in range(n):
            ch = self._cf.CFArrayGetValueAtIndex(arr, i)
            name, unit = self._name_unit(ch)
            if name not in ("GPU Energy", "GPU", "CPU Energy") or dt <= 0:
                continue
            joules = self._ior.IOReportSimpleGetIntegerValue(ch, 0) * _J.get(unit, 0.0)
            w = joules / dt
            if name in ("GPU Energy", "GPU"):   # prefer "GPU Energy" (nJ, finest); both ~agree
                gpu_w = w if (gpu_w is None or unit == "nJ") else gpu_w
            elif name == "CPU Energy":
                cpu_w = w
        if delta:
            self._cf.CFRelease(delta)
        return gpu_w, cpu_w, dt

    def sample(self) -> ThermalSample:
        gpu_w, cpu_w, dt = self._power()
        return ThermalSample(gpu_w=gpu_w, cpu_w=cpu_w, pressure=thermal_pressure(),
                             util_pct=gpu_util_pct(), pmode=self._pmode, window_s=dt)


def _demo():
    """Live demo: drive the production GPU search and watch power/pressure/clock-proxy move."""
    import os
    import sys
    import numpy as np
    from pearl_core import protocol  # noqa
    import stream_search as ss
    from miner import _bound_u32x8

    m, K, RANK = 16384, 2048, 128
    T = (m // 16) ** 2
    base, jk, bound = os.urandom(32), os.urandom(32), _bound_u32x8(0)
    mon = ThermalMonitor()
    print(f"thermal monitor: IOReport power = {'available' if mon._ok else 'UNAVAILABLE'} "
          f"(no sudo). Driving m={m} search; watch GPU W rise, pressure climb, ms/draw grow:\n")
    print(f"  {'draw':>4} {'ms/draw':>8} {'TH/s':>7}   thermals")
    mon.sample()  # prime the power window
    off = 0; ms_min = 1e9
    for blk in range(16):
        t = time.perf_counter()
        for _ in ss.search_draws_overlapped(base, jk, m, K, RANK, bound, range(off, off + 2)):
            pass
        off += 2
        ms = (time.perf_counter() - t) / 2 * 1e3
        ms_min = min(ms_min, ms)
        th = T * 256 * K / (ms * 1e-3) / 1e12
        s = mon.sample()
        clk = ms_min / ms  # clock proxy: 1.0 at cold-burst, drops as it throttles
        print(f"  {off:>4} {ms:>8.1f} {th:>7.3f}   {s.line()}  clk~{clk*100:3.0f}%")


if __name__ == "__main__":
    _demo()
