#!/usr/bin/env bash
# HeroMiners (fixed diff ~2^202, first share ~1-2 h at ~3 TH/s). Usage: ./run-herominers.sh
set -euo pipefail
cd "$(dirname "$0")"
ADDR=${ADDR:-prl1paq33sk7pmxwxmdmdktskxlhrlgsr9satfwul4m352w6vl6najh2stduuqg}
WORKER=${WORKER:-m1max}
M=${M:-4096}
PCT=${PCT:-100}
source .venv/bin/activate
exec python miner.py \
  --pool sg.pearl.herominers.com:1200 \
  --address "$ADDR" --worker "$WORKER" \
  --engine gpu --m "$M" --k 2048 --rank 128 \
  --compute-pct "$PCT" --log-file logs/herominers.log
