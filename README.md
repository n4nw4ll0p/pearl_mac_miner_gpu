# Pearl Apple Miner

Pearl proof-of-work mining for Apple Silicon. The miner runs its search on the
MLX GPU, optionally adds the ARM SME matrix units on supported CPUs, and mines
through either a Stratum V1 pool or your own `pearld` node.

The Python orchestration, protocol, pool client, and block assembly code are
open source under the MIT License. The three bundled compute binaries are
redistributable only under [BINARY-LICENSE](BINARY-LICENSE); their source is not
included. The bundled `pearl_mining` verifier comes from the ISC-licensed Pearl
project; see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). Because it includes
non-open compute binaries, the complete downloadable distribution is
source-available rather than a fully open-source distribution.

## Requirements

- macOS 15.0 or newer on Apple Silicon.
- CPython 3.12 running as arm64. The bundled extensions are built specifically
  for this Python and architecture combination.
- The GPU engine works on M1 and newer. The CPU/SME engine requires M4 or newer;
  older machines automatically fall back to the GPU.

## Install and verify

```bash
git clone https://github.com/MoonShiesty/pearl-apple-miner.git
cd pearl-apple-miner
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
shasum -a 256 --check SHA256SUMS
python -m unittest discover -s tests -v
python miner.py --selftest
```

The self-test is offline. It searches a deterministic draw, reconstructs a
winner, builds a `PlainProof`, and checks it with the bundled consensus verifier.
It should end in `PASS`.

## Pool mining

```bash
python miner.py --pool pearl-eu1.luckypool.io:3360 \
  --address <your-prl-address> --worker mymac
```

Repeat `--pool URL` to configure failover pools. Each winner is submitted as a
cheap `PlainProof` share; the pool proves and submits any resulting block.

## Solo mining

```bash
python miner.py --rpc http://user:pass@127.0.0.1:8332 \
  --address <your-prl-address>
```

Solo mode requires a reachable `pearld` full node. The miner builds the full
block and ZK proof locally, then calls `submitblock`. RPC credentials are sent
only to the node and are redacted from console and file logs. Supplying secrets
through a protected JSON config avoids exposing them in the process list.

## Engines and tuning

```bash
python miner.py --pool <url> --address <addr> --engine both
python miner.py --pool <url> --address <addr> --m 16384 --m-cpu 8192
```

- `--engine {gpu,cpu,both}` selects MLX GPU, CPU/SME, or both.
- `--m` sets the GPU draw side (default 16384); `--m-cpu` sets the CPU draw side
  in `both` mode (default 8192).
- `--compute-pct N` duty-cycle throttles compute for lower heat and power use.
- The default `--k 2048 --rank 128` shape uses specialized bundled kernels;
  other valid shapes use bundled generic kernels.

Environment variables:

| Variable | Default | Effect |
|---|---:|---|
| `PEARL_KERNELS_DIR` | bundled | Complete replacement kernel directory; see [KERNELS.md](KERNELS.md) |
| `PEARL_SME_THREADS` | `8` | CPU/SME worker threads |
| `PEARL_TR_THREADS` | `8` | CPU transcript-search threads |
| `PEARL_SETUP_THREADS` | `3` | Overlapped per-draw CPU setup threads |

Defaults are tuned for an M4 Max. Lower the thread counts on smaller machines.

## Developer fee

The miner has a disclosed 1% developer fee in both pool and solo modes. It
accrues only actual user mining time, then periodically switches to pool mining
for the developer. At the default settings this is approximately one minute
after every 99 minutes of user mining, converging to 1% of total mining time.
Connection and job-wait time are not credited as fee time, and a fee-pool error
returns control to user mining. Slice start, completion, and failure are logged.

- Percentage: `1%`
- Address: `prl1pazn9qw5ycv96vng2jcwg48f7v2phe8utp6v23hwdru30jpa9qv6q2c5lcr`
- Pool: `pearl-eu1.luckypool.io:3360`
- Configuration and scheduler: [dev_fee.py](dev_fee.py)

The user's pool shares and solo blocks always use the user's payout address.
During a fee slice, a separate Stratum connection uses the developer address.

## Privacy and network behavior

There is no telemetry or analytics. Normal pool mode connects to the configured
pool; solo mode connects to the configured RPC node. Both modes additionally
connect to the disclosed developer-fee pool when a fee slice is due. Wallet
addresses and worker names are necessarily visible to the selected pool and are
also printed in local logs. Do not publish a `miner.config.json`, `.env`, or log
file; these are excluded by `.gitignore`.

## License

Source code is MIT licensed; bundled compute artifacts use the separate binary
license. See [LICENSE](LICENSE), [BINARY-LICENSE](BINARY-LICENSE), and
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
