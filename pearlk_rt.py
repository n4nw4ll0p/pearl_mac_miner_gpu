"""Loader for the precompiled GPU kernels.

The GPU kernels ship as compiled bytecode (pearlk.metallib) loaded by a small MLX C++
primitive (_pearlk). This module exposes them behind a drop-in callable (`Kernel`, the
same call interface as an mx.fast.metal_kernel object) plus `has()`/`available()`, so the
per-shape dispatch can fall back to a generic kernel when a specialized variant isn't
present.

`available()` is True once the kernels load. This distribution requires compiled kernels;
point PEARL_KERNELS_DIR at a directory containing a complete replacement set (see KERNELS.md).
"""
from __future__ import annotations

import json
import os

# When True, a failure to load the kernels is a clear, fatal error at import rather than a
# silent disable. Set by the distribution build.
_REQUIRE_PRECOMPILED = True

_lib = None
_manifest = None
_DTYPE_STR = None
_loaded = False


def _locate_manifest(lib) -> str:
    # pearlk_manifest.json ships beside _pearlk.so / pearlk.metallib
    cand = []
    if getattr(lib, "__file__", None):
        cand.append(os.path.join(os.path.dirname(os.path.abspath(lib.__file__)),
                                 "pearlk_manifest.json"))
    env = os.environ.get("PEARL_KERNELS_DIR")
    if env:
        cand.append(os.path.join(env, "pearlk_manifest.json"))
    cand.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "gpu_ext", "pearlk_manifest.json"))
    for p in cand:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"pearlk_manifest.json not found (looked in {cand})")


def _load():
    global _lib, _manifest, _DTYPE_STR, _loaded
    if _loaded:
        return
    import mlx.core as mx
    # ensure the directory holding _pearlk.so / pearlk.metallib is importable (dist: next to
    # this file; dev: pass via PYTHONPATH=gpu_ext/build). Location-independent of cwd.
    import sys
    _self_dir = os.path.dirname(os.path.abspath(__file__))
    for _d in (os.environ.get("PEARL_KERNELS_DIR"), _self_dir):
        if _d and _d not in sys.path and os.path.exists(os.path.join(_d, "pearlk_manifest.json")):
            sys.path.insert(0, _d)
    import _pearlk  # raises ImportError if the extension isn't on sys.path
    _lib = _pearlk
    with open(_locate_manifest(_pearlk)) as fh:
        _manifest = set(json.load(fh)["kernels"])
    _DTYPE_STR = {mx.uint32: "uint32", mx.int32: "int32", mx.float32: "float32",
                  mx.bool_: "bool", mx.uint8: "uint8", mx.int8: "int8",
                  mx.uint64: "uint64", mx.float16: "float16"}
    _loaded = True


def available() -> bool:
    """True iff the precompiled kernel path should be used. When required (env
    PEARL_PRECOMPILED=1, or a build that sets _REQUIRE_PRECOMPILED), a load failure raises a
    clear error instead of silently returning False. This binary-only distribution cannot disable
    its compute kernels."""
    flag = os.environ.get("PEARL_PRECOMPILED")
    if flag == "0":
        raise RuntimeError(
            "PEARL_PRECOMPILED=0 is not supported by this binary distribution; "
            "the GPU kernel source is not included. Remove the setting or provide a "
            "complete replacement through PEARL_KERNELS_DIR."
        )
    required = flag == "1" or _REQUIRE_PRECOMPILED
    try:
        _load()
    except Exception as e:
        if required:
            raise RuntimeError(
                "Pearl GPU kernels could not be loaded (the _pearlk extension and "
                "pearlk.metallib). This build requires them. Verify the supported runtime: "
                "CPython 3.12 on arm64 macOS, with the kernel binaries present beside the "
                f"package. Underlying error: {e!r}") from e
        return False
    return True


def has(name: str) -> bool:
    """Whether a kernel of this exact name was precompiled into the metallib."""
    _load()
    return name in _manifest


def run(name, inputs, output_shapes, output_dtypes, grid, threadgroup):
    _load()
    import mlx.core as mx
    # The primitive indexes input buffers flat, so inputs must be row-contiguous; a
    # non-contiguous view would be read with wrong strides. (no-op when already contiguous)
    inputs = [mx.contiguous(a) for a in inputs]
    shapes = [list(s) for s in output_shapes]
    dts = [d if isinstance(d, str) else _DTYPE_STR[d] for d in output_dtypes]
    return _lib.run(name, inputs, shapes, dts, list(grid), list(threadgroup))


class Kernel:
    """Drop-in for an mx.fast.metal_kernel callable, routed to the precompiled metallib.
    Same call interface: kernel(inputs=, grid=, threadgroup=, output_shapes=, output_dtypes=)."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, *, inputs, grid, threadgroup, output_shapes, output_dtypes, **_):
        return run(self.name, inputs, output_shapes, output_dtypes, grid, threadgroup)
