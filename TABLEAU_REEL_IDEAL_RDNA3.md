# Tableau réel / idéal — RDNA3

Mesures sur la Radeon PRO W7900 (gfx1100), 48 GiB, ROCm 10.1, carte au repos sauf
mention. Médiane de 5 lots ; chaque ligne cite la configuration et l'état de
l'arbre. Les cibles « idéal » viennent de `RAPPORT_RDNA3.md` §7.

## W0 — GEMM IQR dense (commit `c9f7d51`, 2026-11-15)

Forme `q|gate` 12 288 × 2 560 u5, `kRowF32`, arbre `rdna3-iqr`, 50–100 lancements
par lot, 25–41 °C, pas d'autre processus GPU.

| M | Réel | Idéal | η | Note |
|---|---|---|---|---|
| 1 (GEMV) | 26,41 µs, 744,5 GB/s | 22,8 µs, 864 GB/s | **86,3 %** | cible ≥65 % tenue |
| 16 | 239,1 µs, 82,2 GB/s, 4,2 TFLOP/s | borne WMMA ≈ 0,43 ms pour M=4096 | — | échafaudage `v_dot4`, W3 doit le porter |

Forme `down` 5 120 × 17 408 u4, `kRowF32` :

| M | Réel | Idéal | η | Note |
|---|---|---|---|---|
| 1 | 100,2 µs, 444,8 GB/s | 864 GB/s | 51,5 % | 320 tuiles seulement ; split-K en W3/W4 |
| 16 | 710,6 µs, 62,7 GB/s | — | — | idem |

## Lecture

- Le GEMV atteint la cible sur la forme large. La forme `down` (N=5 120) manque de
  parallélisme : 320 tuiles de 16 lignes pour 96 CU ; un split-K de 4 à 8 lèverait
  la limite.
- Le chemin M=16 n'est pas une performance : il décode la tuile en LDS puis fait
  un `v_dot4` par (token, ligne). La cible du rapport pour le préfill suppose la
  matrice 16×16×16 IU8 ; c'est le travail de W3/W4, pas un écart du décodeur.
- Aucune mesure HC/PLE, attention h256, MoE ni vision n'est encore faite : une
  seule carte, benchmarks séquentiels.

## À remplir

| Lot | Forme | Réel | Idéal | η |
|---|---|---|---|---|
| W3 | GEMV 27B, split-K | | | |
| W4 | `gemm_iqr_tiled` M=512/4096 | | | |
| W5 | `moe_gemm_iqr` préfill/décodage | | | |
| W6 | `attn_paged` h256 decode/prefill | | | |
| W8/W9 | `hc_read`, `ple_gate`, `ple_conv` | | | |
| W10 | `attn_dense` hd72 M=16384 | | | |
| M1 | 27B tokens/s (préfill, décodage) | | | |
| M5 | Flash-Next tokens/s | | | |
