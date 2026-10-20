# Pearl (PRL) บน MacBook Pro M1 Max — สรุป Phase 0–3

วันที่: 20 ต.ค. 2026
เครื่อง: MacBook Pro M1 Max (32-core GPU, 64 GB), macOS 27.0.1
Miner: `pearl-apple-miner` v1.0.3 (MLX GPU + Metal)
Address: `prl1paq33sk7pmxwxmdmdktskxlhrlgsr9satfwul4m352w6vl6najh2stduuqg`
Worker: `m1max`

---

## 1. ผลลัพธ์หลัก

| รายการ | สถานะ |
|---|---|
| ติดตั้ง + ตรวจสอบโค้ด/ไบนารี (Phase 0–1) | ✅ ผ่าน (17/17 tests, selftest PASS) |
| ขุดจริงกับ pool (Phase 2) | ✅ **มี accepted share จริง** |
| Tuning (Phase 3) | ✅ ได้ค่าที่ดีที่สุดสำหรับเครื่องนี้ |
| ปัญหาใหญ่ที่เจอและแก้ | ✅ **miner v1.0.3 สร้าง proof เป็น cert v1/v2 แต่ mainnet ใช้ cert v3 → pool reject** |

### ตัวเลขที่พิสูจน์แล้ว
- AlphaPool (d=4096): **6 shares accepted**, 0 consensus reject (มี 1 อัน timeout ของ network)
- HeroMiners: authorize/job ได้, กำลังขุดด้วย cert v3 (share แรกใช้เวลา ~1–2 ชม. เพราะ diff สูงกว่า 64 เท่า)

---

## 2. บั๊ก cert v3 (สำคัญ)

Pool ทุกตัวส่ง `cert_version: 3` มาใน `mining.notify` แต่ miner v1.0.3:
- ไม่อ่าน field นี้เลย
- คำนวณ noise seed แบบเดิม (V1/V2): `blake3(job_key || merkle_root)`
- mainnet หลัง hardfork ส.ค. 2026 ใช้ **salted noise-seed**: bind merkle root กับ dimension ก่อน
  ```
  salt_A = blake3("pearl/cert-v3/noise-seed/A")
  root_A' = blake3(root_A || m_le32 || 28 zero bytes, key=salt_A)
  (ทำ аналогиกับ B ด้วย salt_B และ n)
  ```
- ผลคือ proof ที่ส่งไปผิดทั้งชุด → pool ตอบ `code 20 rejected`

### สิ่งที่แก้ (patch 8 ไฟล์)
- `pearl_core/protocol.py` — เพิ่ม V3 salted commitment hash
- `fast_setup.py` — `commitment_one_tree(..., dims=...)`
- `stream_search.py`, `cpu_search.py`, `miner.py` — ส่ง `cert_version` ผ่าน search pipeline
- `pearl_core/stratum.py` — parse `cert_version` จาก notify
- `pearl_core/streaming.py` — เก็บ `cert_version` ใน snapshot
- `pearl_core/autonomous_mine.py` — reconstruct winner ด้วย V3 seeds

ตรวจสอบความถูกต้อง: เทียบกับ reference implementation ของ `pearl-research-labs/pearl` (`commitment_hash.py`) แล้ว **ตรงกันทุกไบต์** และ pool ยอมรับ share จริง

> เก็บ diff ไว้ที่ `cert-v3-fix.patch` — ถ้า clone ใหม่ให้ `git apply cert-v3-fix.patch`
> หมายเหตุ: `--selftest` ยัง PASS เพราะเป็น offline path แบบ legacy (ตัว verify ที่ bundle มาเป็นเวอร์ชันเก่า) การทดสอบจริงคือการส่ง share เข้า pool

---

## 3. Tuning บนเครื่องจริง (pool mode, k=2048, rank=128)

วัดจาก draws จริง ÷ เวลาจริง (ไม่ใช่แค่ตัวเลข burst ใน log)

| Config | TH/s (effective) | GPU power | Shares | หมายเหตุ |
|---|---|---|---|---|
| m=2048, pct=100 | 2.59 | 27–30 W | 0/5 นาที | overhead สูง ไม่แนะนำ |
| **m=4096, pct=100** | **3.11** | **31 W** | **3 accepted / 5 นาที** | **ดีที่สุดโดยรวม** |
| m=8192, pct=100 | 3.04 (burst 4.18) | 37–47 W | 2 accepted / 7.5 นาที | ช่วงแรกแรง แต่ throttle 20–40% |
| m=8192, pct=80 | 2.95 | 33 W | 1 accepted / 5 นาที | เย็น/ประหยัดไฟกว่า |
| m=16384 (offline bench) | 2.04 | — | — | ไม่แนะนำ |

- Thermal pressure อยู่ที่ระดับ `heavy` ตลอด (ปกติของ M1 Max เมื่อภาระต่อเนื่อง) แต่ไม่มี throttling ระดับวิกฤต
- ค่าที่แนะนำ: **m=4096, compute-pct=100** (หรือ m=8192/pct=80 ถ้าอยากให้เครื่องเงียบขึ้น)

---

## 4. สถานะปัจจุบัน

กำลังขุดอยู่ที่ **HeroMiners (Singapore)**
```
pool:    sg.pearl.herominers.com:1200
worker:  m1max
config:  --engine gpu --m 4096 --k 2048 --rank 128 --compute-pct 100
log:     logs/herominers.log
pid:     25643
```
- hashrate ~3.0–3.1 TH/s, GPU ~30–32 W
- `cert=v3` ถูกส่งถูกต้องแล้ว
- HeroMiners ใช้ diff คงที่ (~2^202) สูงกว่า AlphaPool d=4096 ราว 64 เท่า → **share แรกประมาณ 1–2 ชั่วโมง**
- ดูสถิติ: https://pearl.herominers.com/?address=prl1paq33sk7pmxwxmdmdktskxlhrlgsr9satfwul4m352w6vl6najh2stduuqg
- ยอดเดิมในกระเป๋า HeroMiners: ~0.105 PRL (จาก session ก่อนหน้า)

### หยุด / สลับ pool
```bash
cd /Users/xnxnx/Documents/Working/ai/pearlmine/pearl-apple-miner

pkill -f "miner.py"        # หยุด

./run-alpha.sh             # AlphaPool — share ทุก ~1–3 นาที
./run-herominers.sh        # HeroMiners — share ทุก ~1–2 ชั่วโมง

tail -f logs/herominers.log
```

---

## 5. เศรษฐศาสตร์ (ตามจริง)

- ที่ ~3 TH/s คาดได้ **~0.05 PRL/วัน**
- มูลค่าปัจจุบันน้อยมาก (~$0.05/วัน) — **ไม่คุ้มค่าไฟ** แต่เพียงพอสำหรับทดลอง/เรียนรู้
- HeroMiners มี balance 0.105 PRL, ขั้นต่ำการถอนมักจะ 1 PRL → ต้องขุดต่อเนื่องหลายวัน
- Dev fee ของ miner: 1% (time-sliced, ระบุไว้ใน `dev_fee.py`)

---

## 6. ไฟล์ที่เพิ่ม/แก้

| ไฟล์ | หน้าที่ |
|---|---|
| `cert-v3-fix.patch` | diff ของการแก้ cert v3 (8 ไฟล์) |
| `run-alpha.sh` | รัน AlphaPool ด้วย config ที่แนะนำ |
| `run-herominers.sh` | รัน HeroMiners ด้วย config ที่แนะนำ |
| `bench.py` | benchmark offline (ไม่ต้องใช้ pool/wallet) |
| `stat_check.py` | ตรวจ статистика การเทียบ hash กับ bound |
| `logs/` | log การขุดทั้งหมด |

---

## 7. ขั้นต่อไป (ถ้าต้องการ)

1. **Solo mining (Phase 5)**: ต้องรัน `pearld` + `pearl-gateway` ในเครื่อง แล้วชี้ miner ไปที่ RPC
   - ระวัง: README ของ miner เขียน `--rpc ...:8332` แต่ official mainnet RPC port คือ **44107** — ต้องตรวจก่อน
   - solo จะได้ block reward เต็ม แต่โอกาสพบ block น้อยมากที่ 3 TH/s
   - snapshot สำหรับ solo ถูกตั้ง `cert_version=3` ไว้แล้วในแพตช์นี้
2. **Path C (Metal/MLX miner ของตัวเอง)**: ตอนนี้ทราบจุดที่ต้องระวังแล้ว (V3 commitment, tile/hash semantics) — ถ้าจะต่อยอดสามารถใช้ `cert-v3-fix.patch` เป็น reference ได้
3. **Optimization เพิ่ม**: ลอง `--compute-pct 70/85`, ทดสอบ m=4096 กับ k/rank อื่น, หรือรัน long-run 24 ชม. เพื่อวัดเสถียรภาพ
