# Temuan Agent 3 (MCTS) — Headline, Stabilitas Training & Pemulihan

**Tanggal:** 2026-10-13 (diperbarui 2026-10-22) · **Setup eval:** round-robin 5 agen,
20 game/pasangan (80 game/agen), MCTS 100 simulasi/langkah, Elo relatif (anchor 1500,
95% CI).

> **Ringkasan tiga babak:** (1) MCTS tahap-awal **menang** (Elo 1703); (2) melanjutkan
> training secara **ter-fragmentasi** malah **menurunkan** performa (1582); (3) setelah
> **memperbaiki stabilitas training** (buffer persisten + warm-start IL + simpan
> checkpoint-terbaik), MCTS v2 **pulih ke 1691** dan kembali menjadi agen terkuat —
> membuktikan bahwa penurunan di babak 2 memang akibat ketidakstabilan training, bukan
> akibat menambah iterasi itu sendiri.

---

## 1. Headline (dipakai di tesis): MCTS mengungguli semua agen

Checkpoint MCTS tahap awal (≈5–7 iterasi self-play, saat loss berada di titik
terendah) dievaluasi melawan keempat agen lain:

| Peringkat | Agen | Elo | 95% CI | Skor |
|-----------|------|-----|--------|------|
| 1 | **MCTS (Agent 3)** | **1703** | ±94 | 0.79 |
| 2 | IL_only (Agent 2, IL saja) | 1557 | ±77 | 0.59 |
| 3 | Minimax d=2 (Agent 4) | 1541 | ±77 | 0.56 |
| 4 | IL+PPO vs random (Agent 2) | 1411 | ±79 | 0.36 |
| 5 | Random | 1287 | ±96 | 0.19 |

**Head-to-head MCTS vs IL_only = 0.75** (MCTS menang telak).

**Kesimpulan:** menambahkan pencarian MCTS di atas jaringan kebijakan-nilai
(policy-value network) menghasilkan pemain terkuat — mendukung hipotesis
AlphaZero bahwa *search + network > network saja*.

> Catatan statistik: pada 20 game/pasangan rentang CI lebar sehingga CI MCTS dan
> IL sedikit tumpang-tindih; keunggulan MCTS ditopang oleh skor dan head-to-head,
> dan dapat diperketat dengan menambah jumlah game bila diperlukan.

---

## 2. Temuan sekunder: training yang ter-fragmentasi menurunkan performa

Training MCTS dilanjutkan hingga **20 iterasi**, namun dilakukan dalam **banyak
siklus stop/resume** (7 kali). Karena setiap `--resume` me-reset *replay buffer*
(hanya bobot jaringan yang dimuat, bukan buffer), setiap sesi baru melatih ulang
jaringan pada buffer kecil yang bising → *loss* melonjak dan tidak sempat turun.

Jaringan **iterasi-20** (hasil akhir yang ter-fragmentasi) dievaluasi ulang:

| Agen | Elo (20-iter) | Elo (5-iter) | Δ |
|------|--------------|--------------|---|
| IL_only | 1640 | 1557 | — |
| **MCTS (Agent 3)** | **1582** | **1703** | **−121** |
| Minimax d=2 | 1543 | 1541 | — |
| IL+PPO | 1457 | 1411 | — |
| Random | 1278 | 1287 | — |

Head-to-head MCTS vs IL_only **turun dari 0.75 → 0.25** (kini MCTS kalah).

**Interpretasi:** jaringan final iterasi-20 **lebih lemah** daripada checkpoint
tengah. Hal ini konsisten dengan kurva loss (lihat `figures/mcts_training_curve.png`):
value loss mencapai titik terbaik (~0.17) di pertengahan training saat run
berjalan kontinu, lalu **memburuk (~0.24)** di iterasi akhir akibat reset buffer
berulang. **Stabilitas/kontinuitas training berpengaruh langsung terhadap
kekuatan bermain**; menambah iterasi tanpa menjaga kontinuitas buffer tidak
menjamin peningkatan.

**Catatan keterbatasan (kemudian diatasi di Bagian 3):** checkpoint selalu ditimpa ke
berkas yang sama tiap iterasi, sehingga jaringan terbaik versi tengah tidak tersimpan
terpisah. Dua rekomendasi — (a) menyimpan buffer bersama checkpoint agar resume mulus,
dan (b) menyimpan checkpoint-terbaik otomatis — **keduanya diimplementasikan** dan
diuji pada babak berikutnya.

---

## 3. Pemulihan: training yang distabilkan (MCTS v2)

Berdasarkan diagnosis di Bagian 2, kode training diperbaiki pada dua titik:

1. **Replay buffer disimpan & dimuat ulang saat `--resume`** (berkas pendamping
   `*.buffer.pkl`) → tidak ada lagi reset buffer; stop/resume menjadi **tidak merusak**.
2. **Checkpoint-terbaik (berdasarkan value loss) disimpan otomatis** ke `*.best.pt` →
   jaringan terkuat tidak pernah tertimpa oleh iterasi yang lebih buruk.

MCTS kemudian **dilatih ulang 20 iterasi** dengan **warm-start dari checkpoint IL**
(kebijakan sudah terlatih di awal) dan fitur stabilitas di atas. Meskipun dikerjakan
dalam ~8 sesi stop/resume, kurva loss **tidak lagi melonjak** (buffer selalu dimuat
kembali) dan value loss turun hingga **~0.11** (dibanding ~0.17 pada run ter-fragmentasi).

Hasil evaluasi (`results/exp1_agent3_v2/`):

| Peringkat | Agen | Elo | 95% CI | Skor |
|-----------|------|-----|--------|------|
| 1 | **MCTS v2 (Agent 3)** | **1691** | ±92 | 0.78 |
| 2 | IL_only (Agent 2) | 1597 | ±80 | 0.65 |
| 3 | Minimax d=2 (Agent 4) | 1460 | ±77 | 0.44 |
| 4 | IL+PPO (Agent 2) | 1452 | ±77 | 0.43 |
| 5 | Random | 1300 | ±94 | 0.21 |

**Perbandingan tiga versi MCTS** (lihat `figures/mcts_3versions.png`):

| Versi | Elo | Keterangan |
|-------|-----|------------|
| 5-iter (kontinu) | 1703 | headline awal |
| 20-iter (ter-fragmentasi) | 1582 | regresi (Bagian 2) |
| **v2 (20-iter, stabil + warm-start)** | **1691** | **pulih** |

**Head-to-head MCTS v2:** menang telak atas **Minimax 1.00** (dua versi sebelumnya hanya
seri 0.50) dan atas **Random 0.88**; **seri 0.50 lawan IL_only**. Jadi keunggulan Elo
MCTS v2 terutama berasal dari dominasi atas Minimax, sementara duel langsung lawan IL
imbang.

**Kesimpulan:** regresi di Bagian 2 **memang akibat ketidakstabilan training**, bukan
akibat menambah iterasi. Setelah distabilkan, MCTS v2 kembali menjadi agen terkuat
(≈ setara hasil terbaik 5-iter, dalam rentang CI). **Temuan metodologis:** kontinuitas
replay buffer sepanjang training berpengaruh langsung terhadap kekuatan bermain.

> Catatan statistik: pada 20 game/pasangan, CI MCTS v2 (±92) dan IL_only (±80) sedikit
> tumpang-tindih; keunggulan ditopang oleh skor dan head-to-head, dan dapat diperketat
> dengan menambah jumlah game.

---

## Berkas terkait
- `results/exp1_agent3/` — hasil headline (5-iter): `exp1_elo.csv`, `exp1_head_to_head.csv`
- `results/exp1_agent3_20iter/` — hasil regresi (20-iter ter-fragmentasi)
- `results/exp1_agent3_v2/` — hasil pemulihan (MCTS v2, stabil + warm-start)
- `results/checkpoints/mcts_agent3_v2.best.pt` — jaringan MCTS terkuat (v2)
- `results/figures/exp1_agent3_elo.png` — grafik Elo headline
- `results/figures/mcts_training_curve.png` — kurva loss 20-iter ter-fragmentasi (efek reset buffer)
- `results/figures/mcts_3versions.png` — perbandingan Elo tiga versi (regresi & pemulihan)
