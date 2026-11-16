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

## W12 — fusion et préfill (commit `d8418c1`, 2026-11-15)

27B `qwen3.8-27b-iq4r-l-v4`, 64 tokens greedy, 3 lots, serveur `max-num-seqs 1`,
`max-num-batched-tokens 512`, KV i8. Base = phase 4 (`57e7bcf`) ; W12 = même carte,
mêmes requêtes, mêmes scripts.

| prompt | base préfill | W12 préfill | base décodage | W12 décodage | base TTFT | W12 TTFT |
|---|---|---|---|---|---|---|
| 16 tok | 71,6 tok/s | 75,9 tok/s | 27,3 tok/s | 27,2 tok/s | 224 ms | 211 ms |
| 33 tok | 69,6 tok/s | 123,5 tok/s | 27,0 tok/s | 27,0 tok/s | 474 ms | 267 ms |
| 819 tok | 79,9 tok/s | **303,2 tok/s** | 20,4 tok/s | **20,8 tok/s** | 10,25 s | **2,70 s** |

Attribution par A/B sur la même carte (décodage 16 tok / préfill 819 tok), chaque
variante ne retirant qu'un commit :

| variante | décodage | préfill |
|---|---|---|
| base phase 4 | 27,3 | 79,9 |
| W12 complet (tête avant pitch) | 26,3 | 306 |
| sans gated fusionné | 25,9 | 303 |
| sans tête 16 bits | 27,0 | 300 |
| W12 final (pitch 33) | 27,2 | 303 |

Lecture : le **GEMM IQR résident** porte le préfill (+223 tok/s, −74 % de TTFT à
819) ; le **gated fusionné** gagne ~1,5 % en décodage ; la **tête 16 bits** coûtait
3 % à cause d'un conflit de bancs LDS (stride 32 dwords), supprimé par un pitch de
33 dwords — elle est désormais à parité, avec l'accumulation f32.

## W13 — split-K, tête, h256, vision (commit `f6be390`, 2026-11-15)

27B, **builds propres** (répertoires neufs ; un build incrémental après rsync peut
garder des objets périmés et fausser l'A/B), 64 tokens greedy, 3 lots, mêmes
requêtes que W12. W12 = `9558549` (les chiffres W12 de la section précédente ont
été re-mesurés ici sur build propre ; écarts < 1,5 %).

| prompt | préfill W12 | préfill W13 | décodage W12 | décodage W13 | TTFT W12 | TTFT W13 |
|---|---|---|---|---|---|---|
| 16 tok | 103,1 | 103,4 | 27,0 | **28,4** | 150 ms | 155 ms |
| 33 tok | 122,0 | 122,1 | 26,8 | **28,2** | 271 ms | 270 ms |
| 819 tok | 299,7 | 301,0 | 20,7 | **21,2** | 2,73 s | 2,72 s |

Noyaux (mesures propres) :

| noyau | forme | W12 | W13 |
|---|---|---|---|
| GEMV `down` | 5120×17408, M=1, split-K 8 | 104,0 µs, 428,5 GB/s | **67,3 µs, 662,4 GB/s** (49,5 → 76,6 % de 864) |
| `attn_dense` hd72 | M=512 | 684,7 µs | **350,7 µs (−49 %)** |
| `attn_dense` hd72 | M=256 / 128 / 64 / 32 | 342,5 / 175,6 / 92,1 / 48,4 µs | 179,6 / 94,3 / 48,7 / 26,7 µs |
| `gemm_iqr_tiled` résident | 12288×2560, M=512 / 4096 | 2083 / 17 410 µs (15,5 / 14,8 TFLOP/s) | idem |
| tuile WMMA (construite, éteinte) | mêmes formes | — | 2801 / 23 110 µs (11,5 / 11,2 TFLOP/s) |

Lecture : le **split-K** porte le décodage (+5,2 % à 16 et 33 tokens, +2,4 % à
819) et 1,55× sur le GEMV `down` ; la **vision** gagne 45–49 % de M=32 à M=512
(les petites tailles restent au plancher de lancement) ; la **tête** (un lancement,
pitch 33, bit-exact M=1/M=2) et **h256** (cache de scores LDS, pitch 272) sont
validés et neutres en préfill ; la **tuile WMMA** est 25–32 % plus lente que le
`v_dot4` et le préfill 27B tombe de 301 à 231 tok/s avec elle — construite,
testée, **désactivée** (`kIqrWmmaMinM` au-delà de tout M), à reprendre en W14.

## Flash-Next (W16, 2026-11-16)

Conteneur `qwen3.8-next-flash-iqr-v4` (121 GB), `placement: expert_tiered`,
`host-pool-mib: 43000`, KV i8, `num-speculative-tokens: 0`,
`max-model-len: 8192`, `max-num-batched-tokens: 2048`. Serveur réel, 1 séquence,
greedy, 64 tokens générés :

| prompt | préfill | décodage | TTFT |
|---|---|---|---|
| 30 tok | 20,6 tok/s | 4,6 tok/s | 1,46 s |
| 2290 tok | **42,1 tok/s** | **4,0 tok/s** | 54,3 s |
| 4570 tok | **41,5 tok/s** | **4,0 tok/s** | 110,2 s |
| 4 × 30 tok concurrents | 17,8 tok/s (par requête) | 4,0–4,5 tok/s | 1,68–3,38 s |

Lecture : premier **bring-up fonctionnel** du 121 GB, tous les ops résolus en
device (0 bande device manquante sur 5 074 questions). Le décodage à 4 tok/s
reflète le tiering des experts (SSD/host), l'absence de MTP et un premier chemin
sans optimisation ; le préfill tient 42 tok/s jusqu'à 4 570 tokens.
`kbench` sur ce conteneur : après W17, **856/856 cas corrects, 0 échec,
0 divergence** (`--kernels libr3d:libref`, pire rel_l2 1,2e-2). Les deux défauts
trouvés par le premier run sont corrigés : la tête `logits_gemm_bf16` indexait
son échelle par ligne (1×128) comme une grille 128×128 (rel_l2 0,53 → 1,7e-3),
et trois sorties QSA divergeaient de libref. Robustesse : un prompt de
12 001 tokens est refusé proprement (HTTP 400, message explicite) et le serveur
reste vivant ; les redémarrages sont propres.

Contrôle final de non-régression 27B sur la pointe W17 : préfill **303,3 tok/s**
à 819 tokens (médiane de 3), décodage **28,6 tok/s** à 16 tokens et **21,4** à
819, sorties identiques — les rows ajoutées ne changent pas le chemin dense.

Les timings par row du premier run kbench ne sont **pas exploitables** : la
carte a throttlé (PPT0/PPT1) pendant les 8 minutes de mesure ; HC/PLE restent à
mesurer sur carte froide. Les 78 cas non couverts par kbench sont hors v1 :
MTP 48 (`mtp.draft_head`, `blk.48`), `ngram_hint` sans oracle libref.

## Lecture

- Le GEMV large atteint 86 % ; la forme `down` était à 49,5 %, le split-K la
  porte à 76,6 % (662 GB/s sur 864).
- Le chemin M≥64 reste un `v_dot4` : la matrice 16×16×16 IU8 construite en W13
  est 25–32 % **plus lente** et reste éteinte. La cible « WMMA » du rapport n'est
  pas atteinte par cette forme ; c'est la piste W14 (émission/staging, pas la
  résidence).
- La vision `attn_dense` hd72 gagne 45–49 % de M=32 à M=512 ; la mesure à
  M=16384 reste à faire (kbench s'arrête à 512 sur ce conteneur).
- h256 et la tête sont mesurés de bout en bout (préfill 301, décodage 21,2 à 819),
  pas isolément. HC/PLE et MoE restent non mesurés. Une seule carte, benchmarks
  séquentiels, builds propres.

## À remplir / différé

| Lot | Forme | Réel | Idéal | η |
|---|---|---|---|---|
| W3 | GEMV 27B, split-K | 662,4 GB/s (76,6 %) sur `down` | 864 GB/s | **fait** |
| W4 | `gemm_iqr_tiled` M=512/4096 | 15,5 / 14,8 TFLOP/s (résident) ; WMMA 11,5 / 11,2 | borne WMMA | WMMA plus lente, éteinte |
| W5 | `moe_gemm_iqr` préfill/décodage | | | aucun modèle MoE |
| W6 | `attn_paged` h256 decode/prefill | inclus 27B : préfill 301, décodage 21,2 | | à isoler |
| W8/W9 | `hc_read`, `ple_gate`, `ple_conv` | | | à mesurer |
| W10 | `attn_dense` hd72 | −49 % à M=512 | | M=16384 non mesuré |
| M1 | 27B tokens/s (préfill, décodage) | 301,0 / 21,2–28,4 | | **dépassé** |
| M5 | Flash-Next tokens/s | | | aucun modèle |

## Optimisations différées (W14+)

Ces noyaux sont livrés et validés, mais pas à leur optimum ; ils restent dans
l'arbre et sont marqués à optimiser, pas à supprimer.

- **Tuile WMMA du préfill** : construite, bit-identique, couverte par les tests,
  **éteinte** car 25–32 % plus lente que le `v_dot4` à toutes les formes mesurées.
  Le problème est l'émission et le staging (decode par une seule wave, fragments),
  pas la résidence : W14 doit la reprendre et la rallumer si elle gagne.
- **`gemm_iqr_gated_i8a8h` fusionné** : 512 threads, 136 blocs pour 96 CU (1,4
  bloc/CU) ; essayer un découpage à 256 threads ou un kernel persistant. Gain
  actuel ~1,5 % en décodage, à ne pas régresser.
- **`attn_dense` hd72 à M=16384** : le gain est mesuré jusqu'à M=512 ; ajouter un
  cas kbench à 16384 avant d'aller plus loin.
- **`attn_paged` h256** : le cache de scores et le pitch 272 sont livrés ; le gain
  isolé n'est pas mesuré et le split du préfill reste interdit par l'ordre exact
  que les tests exigent.
- **HC/PLE** : aucune mesure.
- **E4M3/rotated** : aucun modèle non-IQR sur l'hôte.


