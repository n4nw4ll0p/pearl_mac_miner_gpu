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

Recorded May 31–June 1, 2026 on an M4 Max at `--k 2048 --rank 128` and 100%
compute. The peak run used the production pipeline, searched disjoint GPU and
CPU nonce ranges, and passed the end-to-end self-test.

| Measurement | Throughput |
|---|---:|
| GPU, cold burst | **~5.3 TH/s** |
| CPU/SME, unified engine | **3.53 TH/s** |
| GPU + CPU/SME, sustained | **7.92 TH/s** |
| GPU + CPU/SME, typical cool burst | **8.0–8.8 TH/s** |
| GPU + CPU/SME, highest observed | **9.04 TH/s** |

The highest run was approximately 5.76 TH/s GPU plus 3.31 TH/s CPU/SME.
Performance varies with temperature, power mode, and other system activity.

## Engines and tuning

- `--engine {gpu,cpu,both}` selects the GPU, CPU/SME, or both. It defaults to
  `both` on M4 and newer and automatically falls back to `gpu` on M1–M3.
- `--m` sets the GPU draw size; `--m-cpu` sets the CPU size in `both` mode.
- `--compute-pct N` reduces compute use and heat through duty-cycle throttling.
- M4 defaults match the peak configuration: `--engine both --m 16384
  --m-cpu 8192 --k 2048 --rank 128 --compute-pct 100`.
- See [KERNELS.md](KERNELS.md) to use replacement kernels.

The unified CPU engine defaults to 14 threads and six slots. Override them with
`PEARL_U_TR` and `PEARL_U_SLOTS`.

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
