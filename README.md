# Pearl Apple Miner

Pearl proof-of-work mining for Apple Silicon using the MLX GPU and, on M4 or
newer, the ARM SME CPU engine. Supports Stratum V1 pool mining and solo mining
through `pearld`.

## Requirements

- Apple Silicon with macOS 15 or newer
- Arm64 CPython 3.12
- M4 or newer for the optional CPU/SME engine; the GPU works on M1 and newer

## Install

```bash
git clone https://github.com/MoonShiesty/pearl-apple-miner.git
cd pearl-apple-miner
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Pool mining

```bash
python miner.py --pool pearl-eu1.luckypool.io:3360 \
  --address <your-prl-address> --worker mymac
```

Repeat `--pool URL` to configure failover pools.

## Solo mining

```bash
python miner.py --rpc http://user:pass@127.0.0.1:8332 \
  --address <your-prl-address>
```

Solo mode requires a reachable `pearld` node and creates the full block and ZK
proof locally.

## M4 Max benchmark

Measured September 3, 2026 on an Apple M4 Max with a 14-core CPU, 32-core GPU,
36 GB RAM, Python 3.12, and MLX 0.31.2. Results are the median of three 8-draw
runs after warm-up at `--k 2048 --rank 128` and 100% compute.

| Engine | Draw size | Throughput |
|---|---:|---:|
| GPU | `m=16384` | **5.35 TH/s** |
| CPU/SME | `m=16384` | **2.68 TH/s** |
| GPU + CPU/SME | `m=16384`, `m-cpu=8192` | **5.53 TH/s** (4.42 + 1.10) |

Short runs vary with temperature, power mode, and other system activity.

## Engines and tuning

- `--engine {gpu,cpu,both}` selects the GPU, CPU/SME, or both.
- `--m` sets the GPU draw size; `--m-cpu` sets the CPU size in `both` mode.
- `--compute-pct N` reduces compute use and heat through duty-cycle throttling.
- Defaults are tuned for an M4 Max: `--m 16384 --m-cpu 8192 --k 2048 --rank 128`.
- See [KERNELS.md](KERNELS.md) to use replacement kernels.

Thread counts can be tuned with `PEARL_SME_THREADS`, `PEARL_TR_THREADS`, and
`PEARL_SETUP_THREADS` (all default to `8`, `8`, and `3`, respectively).

## Developer fee

The miner includes a disclosed 1% developer fee in pool and solo modes. It
mines for the developer for approximately one minute after every 99 minutes of
user mining. User pool shares and solo blocks always use the user's address.

- Address: `prl1pazn9qw5ycv96vng2jcwg48f7v2phe8utp6v23hwdru30jpa9qv6q2c5lcr`
- Pool: `pearl-eu1.luckypool.io:3360`
- Implementation: [dev_fee.py](dev_fee.py)

## License

The Python source is MIT licensed. The three bundled compute binaries use the
separate [binary license](BINARY-LICENSE), so the complete distribution is
source-available rather than entirely open source. The bundled `pearl_mining`
verifier is ISC licensed; see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

## Verify

```bash
shasum -a 256 --check SHA256SUMS
python -m unittest discover -s tests -v
```

## Self-test

```bash
python miner.py --selftest
```

The offline self-test runs a draw through GPU search, winner reconstruction,
`PlainProof` creation, and consensus verification. It should finish with
`PASS`.
