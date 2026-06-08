"""Rotate-left-32 primitive used by the reference oracle."""

from __future__ import annotations

import numpy as np


def rotl32_np(x, n: int):
    """Rotate-left a uint32 by n bits. Numpy/Python int safe for any n."""
    n = n & 31
    x = np.uint32(x)
    # Widen to uint64 so left-shift can't overflow and the right-shift-by-32
    # case (when n==0) doesn't hit UB.
    return np.uint32(((np.uint64(x) << np.uint64(n)) | (np.uint64(x) >> np.uint64(32 - n))) & np.uint64(0xFFFFFFFF))


