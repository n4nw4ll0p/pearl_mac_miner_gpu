"""CPU + ARM-SME search engine, bit-exact with the GPU search."""
from __future__ import annotations

import ctypes
import os
import subprocess
import threading

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_KDIR = os.environ.get("PEARL_KERNELS_DIR", _HERE)
_LIB = os.path.join(_KDIR, "libcpu_search.dylib")

SME_THREADS = int(os.environ.get("PEARL_SME_THREADS", "8"))
TR_THREADS = int(os.environ.get("PEARL_TR_THREADS", "8"))
SETUP_THREADS = int(os.environ.get("PEARL_SETUP_THREADS", "3"))

from concurrent.futures import ThreadPoolExecutor
_POOL = ThreadPoolExecutor(max_workers=max(2, SME_THREADS), thread_name_prefix="k")
_SETUP_POOL = ThreadPoolExecutor(max_workers=max(2, SETUP_THREADS), thread_name_prefix="setup")
_PREFETCH = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")

import threading


def _parallel(worker, parts, pool=None):
    if len(parts) <= 1:
        for p in parts:
            worker(*p)
        return
    for _ in (pool or _POOL).map(lambda p: worker(*p), parts):
        pass


def _build():
    if not os.path.exists(_LIB):
        script = os.path.join(_HERE, "build_stripped_dylib.sh")
        if os.path.exists(script):
            subprocess.run([script, _LIB], check=True)
        else:
            raise FileNotFoundError(
                f"CPU kernel library not found: {_LIB}. Set PEARL_KERNELS_DIR to its directory.")
    return ctypes.CDLL(_LIB)


_lib = _build()
_lib.cpu_transcripts.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4
_lib.cpu_transcripts2.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4
_lib.cpu_transcripts2.restype = None
_lib.cpu_transcripts2f.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4
_lib.cpu_transcripts2f.restype = None
_lib.cpu_pipe_worker.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
_lib.cpu_pipe_worker.restype = None
_lib.cpu_transcripts.restype = None
_lib.cpu_pack.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.cpu_pack.restype = None
_lib.cpu_winners.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
_lib.cpu_winners.restype = ctypes.c_int
_lib.cpu_signal.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 4 + [ctypes.c_int] * 3
_lib.cpu_signal.restype = None
_lib.cpu_build_both.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 3
_lib.cpu_build_both.restype = None
_lib.cpu_noise.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
_lib.cpu_noise.restype = None
_lib.cpu_build_both_f.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 4
_lib.cpu_build_both_f.restype = None
_lib.cpu_commit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                              ctypes.c_void_p, ctypes.c_void_p]
_lib.cpu_commit.restype = None
_lib.cpu_commit_par.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
_lib.cpu_commit_par.restype = None
_lib.cpu_perm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_int, ctypes.c_int]
_lib.cpu_perm.restype = None


def cpu_perm(seed_label: bytes, key: bytes, k: int, rank: int) -> np.ndarray:
    out = np.empty((k, 2), np.uint32)
    sl = np.frombuffer(seed_label, "<u4").copy(); ky = np.frombuffer(key, "<u4").copy()
    _lib.cpu_perm(out.ctypes.data_as(ctypes.c_void_p), sl.ctypes.data_as(ctypes.c_void_p),
                       ky.ctypes.data_as(ctypes.c_void_p), k, rank)
    return out


def cpu_commit(job_key: bytes, a_pad, parts: int = 1) -> tuple[bytes, bytes]:
    a = np.ascontiguousarray(a_pad, np.int8)
    b_seed = np.empty(32, np.uint8); a_seed = np.empty(32, np.uint8)
    jk = np.frombuffer(job_key, np.uint8)
    _cp = ctypes.c_void_p
    if parts > 1:
        _lib.cpu_commit_par(jk.ctypes.data_as(_cp), a.ctypes.data_as(_cp), a.size, parts,
                              b_seed.ctypes.data_as(_cp), a_seed.ctypes.data_as(_cp))
    else:
        _lib.cpu_commit(jk.ctypes.data_as(_cp), a.ctypes.data_as(_cp), a.size,
                          b_seed.ctypes.data_as(_cp), a_seed.ctypes.data_as(_cp))
    return b_seed.tobytes(), a_seed.tobytes()

_lib.cpu_e_create.argtypes = ([ctypes.c_int] * 6 + [ctypes.c_uint32] * 3 + [ctypes.c_void_p] * 4)
_lib.cpu_e_create.restype = ctypes.c_void_p
_lib.cpu_e_run.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]; _lib.cpu_e_run.restype = None
_lib.cpu_e_collect.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
_lib.cpu_e_collect.restype = ctypes.c_int
_lib.cpu_e_stats.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]; _lib.cpu_e_stats.restype = None
_lib.cpu_e_stop.argtypes = [ctypes.c_void_p]; _lib.cpu_e_stop.restype = None
_lib.cpu_e_join.argtypes = [ctypes.c_void_p]; _lib.cpu_e_join.restype = None
_lib.cpu_e_destroy.argtypes = [ctypes.c_void_p]; _lib.cpu_e_destroy.restype = None


def uniform_random(seed_label: bytes, key: bytes, rows: int, rank: int) -> np.ndarray:
    seed = np.frombuffer(seed_label, dtype="<u4").copy()
    keyw = np.frombuffer(key, dtype="<u4").copy()
    out = np.empty(rows * rank, np.int8)
    _lib.cpu_noise(out.ctypes.data_as(ctypes.c_void_p), seed.ctypes.data_as(ctypes.c_void_p),
                           keyw.ctypes.data_as(ctypes.c_void_p), rows * rank, 0)
    return out.reshape(rows, rank)


def _threaded_rowblocks(nti, threads):
    threads = max(1, min(threads, nti))
    q = nti // threads
    bnd = [i * q for i in range(threads)] + [nti]
    return [(bnd[i], bnd[i + 1]) for i in range(threads) if bnd[i] < bnd[i + 1]]


def cpu_signal(base_seed: bytes, draw_index: int, label: int, m: int, k: int,
                    threads: int = SME_THREADS, pool=None) -> np.ndarray:
    seed = np.frombuffer(base_seed, dtype="<u4")
    s0, s1 = int(seed[0]), int(seed[1])
    A = np.empty((m, k), np.int8)
    Ai = A.ctypes.data
    parts = _threaded_rowblocks(m // 16, threads)

    def w(b0, b1):
        _lib.cpu_signal(ctypes.c_void_p(Ai + b0 * 16 * k), s0, s1,
                             draw_index & 0xFFFFFFFF, label & 0xFFFFFFFF, (b1 - b0) * 16, k, b0 * 16)
    _parallel(w, parts, pool)
    return A


def cpu_build_both(A, e_a, perm_a, e_b, perm_b, m, k, rank, threads=SME_THREADS, pool=None):
    A = np.ascontiguousarray(A, np.int8)
    e_a = np.ascontiguousarray(e_a, np.int8); e_b = np.ascontiguousarray(e_b, np.int8)
    perm_a = np.ascontiguousarray(perm_a, np.uint32); perm_b = np.ascontiguousarray(perm_b, np.uint32)
    Pa = np.empty((m // 16, k // 4, 64), np.int8); Pb = np.empty((m // 16, k // 4, 64), np.int8)
    Pai, Pbi, Ai = Pa.ctypes.data, Pb.ctypes.data, A.ctypes.data
    eai, pai, ebi, pbi = e_a.ctypes.data, perm_a.ctypes.data, e_b.ctypes.data, perm_b.ctypes.data
    parts = _threaded_rowblocks(m // 16, threads)

    def w(b0, b1):
        off_p = b0 * (k // 4) * 64
        off_e = b0 * 16 * rank
        _lib.cpu_build_both(ctypes.c_void_p(Pai + off_p), ctypes.c_void_p(Pbi + off_p),
                        ctypes.c_void_p(Ai + b0 * 16 * k),
                        ctypes.c_void_p(eai + off_e), ctypes.c_void_p(pai),
                        ctypes.c_void_p(ebi + off_e), ctypes.c_void_p(pbi),
                        (b1 - b0) * 16, k, rank)
    _parallel(w, parts, pool)
    return Pa, Pb


def cpu_build_both_f(A, a_seed: bytes, b_seed: bytes, perm_a, perm_b, m, k, rank,
                     threads=SME_THREADS, pool=None, out=None):
    from pearl_core import protocol
    A = np.ascontiguousarray(A, np.int8)
    perm_a = np.ascontiguousarray(perm_a, np.uint32); perm_b = np.ascontiguousarray(perm_b, np.uint32)
    a_key = np.frombuffer(a_seed, "<u4").copy(); b_key = np.frombuffer(b_seed, "<u4").copy()
    lab_a = np.frombuffer(protocol.SEED_LABEL_A, "<u4").copy()
    lab_b = np.frombuffer(protocol.SEED_LABEL_B, "<u4").copy()
    if out is not None:
        Pa, Pb = out
    else:
        Pa = np.empty((m // 16, k // 4, 64), np.int8); Pb = np.empty((m // 16, k // 4, 64), np.int8)
    Pai, Pbi, Ai = Pa.ctypes.data, Pb.ctypes.data, A.ctypes.data
    aki, bki, lai, lbi = a_key.ctypes.data, b_key.ctypes.data, lab_a.ctypes.data, lab_b.ctypes.data
    pai, pbi = perm_a.ctypes.data, perm_b.ctypes.data
    parts = _threaded_rowblocks(m // 16, threads)

    def w(b0, b1):
        off_p = b0 * (k // 4) * 64
        _lib.cpu_build_both_f(ctypes.c_void_p(Pai + off_p), ctypes.c_void_p(Pbi + off_p),
                              ctypes.c_void_p(Ai + b0 * 16 * k),
                              ctypes.c_void_p(aki), ctypes.c_void_p(bki),
                              ctypes.c_void_p(lai), ctypes.c_void_p(lbi),
                              ctypes.c_void_p(pai), ctypes.c_void_p(pbi),
                              (b1 - b0) * 16, k, rank, b0 * 16)
    _parallel(w, parts, pool)
    return Pa, Pb


def pack(M: np.ndarray) -> np.ndarray:
    rows, k = M.shape
    M = np.ascontiguousarray(M, dtype=np.int8)
    P = np.empty((rows // 16, k // 4, 64), np.int8)
    _lib.cpu_pack(M.ctypes.data_as(ctypes.c_void_p), P.ctypes.data_as(ctypes.c_void_p), rows, k)
    return P


def _even_rowblock_split(nti, threads):
    pairs = nti // 2
    threads = max(1, min(threads, pairs))
    q = pairs // threads
    bnd = [i * q * 2 for i in range(threads)] + [nti]
    return [(bnd[i], bnd[i + 1]) for i in range(threads) if bnd[i] < bnd[i + 1]]


_PC_BUFS = {}


def _transcripts_pc(Ap, Bp, ntj, kg, cg, threads, pool):
    nti = Ap.shape[0]
    out = np.empty((nti * ntj, 16), np.uint32)
    api, bpi, opi, arow = Ap.ctypes.data, Bp.ctypes.data, out.ctypes.data, Ap.strides[0]
    parts = _even_rowblock_split(nti, threads)

    def w(b0, b1):
        buf = _PC_BUFS.get(threading.get_ident())
        if buf is None:
            buf = _PC_BUFS[threading.get_ident()] = np.empty(16 * 4 * 256, np.int32)
        _lib.cpu_pipe_worker(ctypes.c_void_p(api + b0 * arow), ctypes.c_void_p(bpi),
                       ctypes.c_void_p(opi + b0 * ntj * 64), 0, b1 - b0, ntj, kg, cg,
                       buf.ctypes.data_as(ctypes.c_void_p))
    if len(parts) <= 1:
        w(0, nti)
    else:
        _parallel(w, parts, pool)
    return out


def transcripts(Ap: np.ndarray, Bp: np.ndarray, ntj: int, kg: int, cg: int,
                threads: int = SME_THREADS, pool=None) -> np.ndarray:
    nti = Ap.shape[0]
    use2 = (nti % 2 == 0 and ntj % 2 == 0)
    nc16 = (kg // cg == 16)
    if use2 and nc16:
        return _transcripts_pc(Ap, Bp, ntj, kg, cg, threads, pool)
    out = np.empty((nti * ntj, 16), np.uint32)
    api, bpi, opi, arow = Ap.ctypes.data, Bp.ctypes.data, out.ctypes.data, Ap.strides[0]
    if use2 and nc16:
        fn = _lib.cpu_transcripts2f
    elif use2:
        fn = _lib.cpu_transcripts2
    else:
        fn = _lib.cpu_transcripts
    parts = _even_rowblock_split(nti, threads) if use2 else _threaded_rowblocks(nti, threads)
    if len(parts) <= 1:
        fn(ctypes.c_void_p(api), ctypes.c_void_p(bpi), ctypes.c_void_p(opi), nti, ntj, kg, cg)
        return out

    def w(b0, b1):
        fn(ctypes.c_void_p(api + b0 * arow), ctypes.c_void_p(bpi),
           ctypes.c_void_p(opi + b0 * ntj * 64), b1 - b0, ntj, kg, cg)
    _parallel(w, parts, pool)
    return out


def winners(tr: np.ndarray, a_seed_u32: np.ndarray, bound_u32x8: np.ndarray,
            threads: int = 4) -> list[int]:
    T = tr.shape[0]
    tr = np.ascontiguousarray(tr, np.uint32)
    key = np.ascontiguousarray(a_seed_u32, np.uint32)
    bound = np.ascontiguousarray(bound_u32x8, np.uint32)
    tri, ki, bi = tr.ctypes.data, key.ctypes.data, bound.ctypes.data
    if threads <= 1 or T < 4096:
        buf = np.empty(T, np.int32)
        nw = _lib.cpu_winners(ctypes.c_void_p(tri), ctypes.c_void_p(ki), ctypes.c_void_p(bi),
                                 0, T, buf.ctypes.data_as(ctypes.c_void_p))
        return buf[:nw].tolist()
    bufs = [np.empty(T, np.int32) for _ in range(threads)]
    counts = [0] * threads
    q = T // threads
    bnd = [i * q for i in range(threads)] + [T]

    def w(i):
        counts[i] = _lib.cpu_winners(ctypes.c_void_p(tri), ctypes.c_void_p(ki), ctypes.c_void_p(bi),
                                        bnd[i], bnd[i + 1], bufs[i].ctypes.data_as(ctypes.c_void_p))
    ths = [threading.Thread(target=w, args=(i,)) for i in range(threads)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    out: list[int] = []
    for i in range(threads):
        out.extend(bufs[i][:counts[i]].tolist())
    return sorted(out)


def setup_packed(base_seed: bytes, draw_index: int, job_key: bytes,
                 m: int, k: int, rank: int, threads: int = SME_THREADS, pool=None,
                 cert_version: int = 0):
    from pearl_core import protocol
    from fast_setup import commitment_one_tree
    A = cpu_signal(base_seed, draw_index, protocol.SIGNAL_LABEL_A, m, k, threads, pool)
    a_buf = A if (A.size % protocol.BLAKE3_CHUNK_LEN == 0) else \
        np.frombuffer(protocol.pad_to_chunk_boundary(A.tobytes()), np.int8)
    dims = (m, m) if cert_version >= 3 else None
    b_seed, a_seed = commitment_one_tree(job_key, a_buf, dims=dims)
    e_ar = protocol.generate_permutation_matrix(protocol.SEED_LABEL_A, a_seed, k, rank)
    e_bl = protocol.generate_permutation_matrix(protocol.SEED_LABEL_B, b_seed, k, rank)
    Ap, Bp = cpu_build_both_f(A, a_seed, b_seed, e_ar, e_bl, m, k, rank, threads, pool)
    return Ap, Bp, np.frombuffer(a_seed, dtype="<u4").copy()


def search_draw_cpu(base_seed: bytes, draw_index: int, job_key: bytes,
                    m: int, k: int, rank: int, bound_u32x8: np.ndarray,
                    cert_version: int = 0) -> list[int]:
    Ap, Bp, a_seed_u32 = setup_packed(base_seed, draw_index, job_key, m, k, rank,
                                      cert_version=cert_version)
    tr = transcripts(Ap, Bp, m // 16, k // 4, rank // 4)
    return winners(tr, a_seed_u32, bound_u32x8)


def search_draws_cpu(base_seed: bytes, job_key: bytes, m: int, k: int, rank: int,
                     bound_u32x8, draw_indices, cert_version: int = 0):
    idxs = list(draw_indices)
    if not idxs:
        return
    bound = np.asarray(bound_u32x8, np.uint32)
    def _setup(di):
        return setup_packed(base_seed, di, job_key, m, k, rank,
                            threads=SETUP_THREADS, pool=_SETUP_POOL,
                            cert_version=cert_version)

    fut = _PREFETCH.submit(_setup, idxs[0])
    for i, di in enumerate(idxs):
        Ap, Bp, a_seed = fut.result()
        if i + 1 < len(idxs):
            fut = _PREFETCH.submit(_setup, idxs[i + 1])
        tr = transcripts(Ap, Bp, m // 16, k // 4, rank // 4, threads=TR_THREADS)
        yield di, winners(tr, a_seed, bound, threads=min(4, TR_THREADS))


PL_ENGINE_THREADS_CPU = int(os.environ.get("PEARL_PL_TR_CPU", "12"))
U_TR_CAP = int(os.environ.get("PEARL_U_TRCAP", "0"))
PL_UNIFIED_THREADS = int(os.environ.get("PEARL_U_TR", "14"))
PL_UNIFIED_SLOTS = int(os.environ.get("PEARL_U_SLOTS", "6"))


def search_draws_cpu_unified(base_seed: bytes, job_key: bytes, m: int, k: int, rank: int,
                             bound_u32x8, draw_indices, n_slots=None, n_tr=None, tr_cap=None):
    from pearl_core import protocol
    _cp = ctypes.c_void_p
    idxs = list(draw_indices)
    if not idxs:
        return
    if (m % 32 != 0) or (k // rank != 16):
        yield from search_draws_cpu(base_seed, job_key, m, k, rank, bound_u32x8, idxs)
        return
    n_slots = n_slots or PL_UNIFIED_SLOTS
    n_tr = n_tr or PL_UNIFIED_THREADS
    tr_cap = U_TR_CAP if tr_cap is None else tr_cap
    T = (m // 16) ** 2
    seed = np.frombuffer(base_seed, "<u4")
    s0, s1 = int(seed[0]), int(seed[1])
    lab_a = np.frombuffer(protocol.SEED_LABEL_A, "<u4").copy()
    lab_b = np.frombuffer(protocol.SEED_LABEL_B, "<u4").copy()
    bound = np.asarray(bound_u32x8, np.uint32)
    jk = np.frombuffer(job_key, np.uint8).copy()
    draw_map = np.asarray(idxs, np.int32)
    eng = _lib.cpu_e_create(n_slots, n_tr, tr_cap, m, k, rank, s0 & 0xFFFFFFFF, s1 & 0xFFFFFFFF,
                        protocol.SIGNAL_LABEL_A & 0xFFFFFFFF, jk.ctypes.data_as(_cp),
                        lab_a.ctypes.data_as(_cp), lab_b.ctypes.data_as(_cp),
                        bound.ctypes.data_as(_cp))
    out = np.empty(T, np.int32)
    _lib.cpu_e_run(eng, len(idxs), draw_map.ctypes.data_as(_cp))
    try:
        for i, di in enumerate(idxs):
            cnt = _lib.cpu_e_collect(eng, i, out.ctypes.data_as(_cp), T)
            if cnt < 0:
                break
            yield di, sorted(out[:cnt].tolist())
    finally:
        _lib.cpu_e_stop(eng)
        _lib.cpu_e_join(eng)
        _lib.cpu_e_destroy(eng)


def search_draws_cpu_auto(base_seed: bytes, job_key: bytes, m: int, k: int, rank: int,
                          bound_u32x8, draw_indices, n_tr=None, cert_version: int = 0):
    """CPU + ARM-SME search for `draw_indices`, yielding (draw_index, winning_tile_indices); bit-exact with the GPU search. m/k/rank: draw shape; bound_u32x8: difficulty target. Selects the fastest available CPU engine. cert_version >= 3 uses the V3 salted Python path (the native engine only implements the legacy derivation)."""
    if cert_version >= 3:
        yield from search_draws_cpu(base_seed, job_key, m, k, rank, bound_u32x8,
                                    draw_indices, cert_version=cert_version)
        return
    yield from search_draws_cpu_unified(base_seed, job_key, m, k, rank, bound_u32x8,
                                        draw_indices, n_tr=n_tr)


