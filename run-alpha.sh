#!/usr/bin/env bash
# AlphaPool (share every ~1-3 min at m=4096). Usage: ./run-alpha.sh
set -euo pipefail
cd "$(dirname "$0")"
ADDR=${ADDR:-prl1paq33sk7pmxwxmdmdktskxlhrlgsr9satfwul4m352w6vl6najh2stduuqg}
WORKER=${WORKER:-m1max}
M=${M:-4096}
PCT=${PCT:-100}
source .venv/bin/activate
exec python miner.py \
  --pool us2.alphapool.tech:5571 --difficulty 4096 \
  --address "$ADDR" --worker "$WORKER" \
  --engine gpu --m "$M" --k 2048 --rank 128 \
  --compute-pct "$PCT" --log-file logs/alpha.log
