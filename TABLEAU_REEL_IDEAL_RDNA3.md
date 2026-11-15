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

## Lecture

- Le GEMV atteint la cible sur la forme large. La forme `down` (N=5 120) manque de
  parallélisme : 320 tuiles de 16 lignes pour 96 CU ; un split-K de 4 à 8 lèverait
  la limite.
- Le chemin M=16 n'est pas une performance : il décode la tuile en LDS puis fait
  un `v_dot4` par (token, ligne). La cible du rapport pour le préfill suppose la
  matrice 16×16×16 IU8 ; c'est le travail de W3/W4, pas un écart du décodeur.
- Aucune mesure HC/PLE ni MoE n'est encore faite, et les changements W12 de
  h256/vision ne sont pas isolés (seuls les chiffres 27B les incluent). Une seule
  carte, benchmarks séquentiels ; voir « Optimisations différées ».

## À remplir / différé

| Lot | Forme | Réel | Idéal | η |
|---|---|---|---|---|
| W3 | GEMV 27B, split-K | 51,5 % sur `down` (W0) | 864 GB/s | split-K non fait |
| W4 | `gemm_iqr_tiled` M=512/4096 | bout en bout seul (303 tok/s à 819) | borne WMMA | noyau non mesuré ; `v_dot4` |
| W5 | `moe_gemm_iqr` préfill/décodage | | | aucun modèle MoE |
| W6 | `attn_paged` h256 decode/prefill | inclus dans les chiffres 819, non isolé | | à mesurer |
| W8/W9 | `hc_read`, `ple_gate`, `ple_conv` | | | à mesurer |
| W10 | `attn_dense` hd72 M=16384 | non isolé | | à mesurer |
| M1 | 27B tokens/s (préfill, décodage) | 303,2 / 20,8–27,2 | | **atteint** |
| M5 | Flash-Next tokens/s | | | aucun modèle |

## Optimisations différées (W13+)

Ces noyaux sont livrés et validés, mais pas à leur optimum ; ils restent dans
l'arbre et sont marqués à optimiser, pas à supprimer.

- **`gemm_iqr_tiled` résident** : le préfill est passé à 303 tok/s, mais le cœur
  reste un échafaudage `v_dot4` ; la matrice 16×16×16 IU8/WMMA n'est pas utilisée.
  C'est le premier levier du préfill.
- **`down` 5 120 × 17 408** : 320 tuiles pour 96 CU, 51,5 % ; split-K (4 à 8) non
  fait.
- **`gemm_iqr_gated_i8a8h` fusionné** : 512 threads et 136 groupes pour 136 blocs
  (1,4 bloc/CU sur 96 CU) ; gagnerait à un bloc de 2 groupes ou 256 threads.
  Gain actuel ~1,5 % en décodage.
- **Tête IQR 16 bits** : à parité après le pitch 33, mais deux passes int8 et un
  lancement par classe ; la parallélisation des classes et le recouvrement du
  décodage restent ouverts.
- **`attn_paged` h256** : staging i8 en 16 octets et passe max sans V livrés ;
  aucun gain isolé mesuré. Split du préfill non fait.
- **`attn_dense` hd72** : tuilage LDS et chargements vectorisés livrés, ordre
  inchangé ; coût à M=16384 non mesuré.
- **HC/PLE** : aucune mesure.

