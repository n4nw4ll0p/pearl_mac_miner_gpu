"""Tile-index helpers for the search."""
from __future__ import annotations
import os
from typing import List
import numpy as np
def tile_index_arrays(a_parts: List[List[int]], b_parts: List[List[int]]):
    """(a_parts × b_parts) -> tile_a_idx (T, h), tile_b_idx (T, w) int32, where
    tile t = (a_parts[t // len(b_parts)], b_parts[t % len(b_parts)])."""
    ta, tb = [], []
    for a in a_parts:
        for b in b_parts:
            ta.append(list(a))
            tb.append(list(b))
    return np.asarray(ta, dtype=np.int32), np.asarray(tb, dtype=np.int32)
