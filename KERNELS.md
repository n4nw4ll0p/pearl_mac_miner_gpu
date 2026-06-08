# Kernel interface (swapping in your own kernels)

The compute kernels ship as binary artifacts and are addressed by **role name** through a
small, stable interface. The orchestration source (open in this repo) calls them; you can
provide your own implementations by building drop-in binaries that export the same role names
and pointing the miner at them:

```bash
PEARL_KERNELS_DIR=/path/to/your/kernels python miner.py --pool <url> --address <addr>
```

The replacement directory must contain the complete set: `libcpu_search.dylib` (CPU/SME),
plus `pearlk.metallib`, `_pearlk*.so`, and `pearlk_manifest.json` (GPU). Run
`python miner.py --selftest` after selecting a replacement; the test validates its output
against the consensus path. The miner does not run the self-test automatically at startup.

## GPU kernels — `pearlk.metallib`, invoked via `pearlk_rt.run(name, inputs, out_shapes, out_dtypes, grid, threadgroup)`

Listed in `pearlk_manifest.json`. Exact input/output dtypes, shapes, grid, and threadgroup
for each call are in the orchestration source (`fast_search.py`, `fast_setup.py`,
`stream_search.py`, `metal_kernel.py`, `fold_kernel.py`, `setup_kernels.py`, `fused_lp.py`).

| role | computes |
|------|----------|
| `signal`        | per-draw signal matrix (on device) |
| `build_ab` / `build_ab_t` | effective operands A/B (row-major / transposed-B) |
| `noise`         | per-draw noise operands |
| `transcripts_s` / `transcripts_b` / `transcripts_p` | draw transcripts (square / block / pair tiling); generic, shape via a `dims` buffer |
| `transcripts_s_<k>_<r>_<ntj>_<nchk>[_t]` | shape-specialized transcript kernels (optional fast path; the generic `transcripts_*` is the fallback) |
| `transcripts_ref` | reference transcript kernel |
| `fold` / `fold_pair` | per-tile fold |
| `bound_digest` / `bound_mask` | keyed hash + difficulty-bound check (with / without digest) |

## CPU/SME kernels — `libcpu_search.dylib`, invoked via `ctypes`

Role names are bound in `cpu_search.py` (which documents each `argtypes`/`restype`):
`cpu_transcripts` / `cpu_transcripts2` / `cpu_transcripts2f`, `cpu_pack`, `cpu_winners`,
`cpu_commit` / `cpu_commit_par`, `cpu_perm`, `cpu_signal`, `cpu_noise`,
`cpu_build_both` / `cpu_build_both_f`, `cpu_pipe_worker`, `cpu_e_*` (engine).

Read `cpu_search.py` for the exact C signature each role expects.
