# Suivi des lots — RDNA3

État au 2026-11-15. Les estimations sont celles du plan ; « reste » est mis à jour
à chaque session. Ne pas réécrire `PLAN_IMPLEMENTATION_RDNA3.md`.

| Lot | État | Estimation | Reste à faire |
|---|---|---|---|
| Préparation | terminé | — | journal, suivi, ADR, environnement, reproduction |
| W0 — spike IQR 27B | **terminé (GO)** | 2,5 sem | parité et η atteintes ; mémo M0 |
| W1 — formes et hooks | en cours | 1,5 sem | forme table/grid/`kE4m3`/`tcq_rb` en kernel ; refus en place |
| W2 — décodeur device | en cours | 1,5 sem | treillis calculé gfx11 fait ; table/grid à faire |
| W3 — GEMM M ≤ 16 | en cours | 1,5 sem | GEMV 86 % ; m16 `v_dot4` à porter |
| W4 — m64 et préfill | à faire | 2,5 sem | m64, tuile préfill, bout en bout 27B |
| W5 — MoE IQR | à faire | 4 sem | run map, formes mixtes, experts bf16 |
| W6 — attention h256/GQA/KV i8 | **terminé (revu, 15/15 device)** | 3,5 sem | + kv_store i8, rope_table, imrope |
| W7 — QSA | à faire | 2,5 sem | après W6 |
| W8 — HC | **terminé (revu, device vert)** | 3,5 sem | 6 rows ; perf à mesurer en W12 |
| W9 — PLE | **terminé (revu, device vert)** | 1,5 sem | 4 rows ; `opd_shape` en W11 |
| W10 — vision | **terminé (revu, 142 rows vertes)** | 3 sem | coût attn_dense à optimiser en W12 |
| W11 — intégration | **terminé (2026-11-15)** | 1,5 sem | tête vocabulaire validée et servie par défaut |
| W12 — optimisation | **terminé (2026-11-15)** | 3–4 sem | préfill 27B ×3,8 ; gated fusionné ; tête 16 bits ; h256/vision ; W13 = WMMA/IU8, split-K `down`, mesures isolées |
| M0 — Go/No-Go | **GO (2026-11-15)** | fin S3 | `MEMO_M0_RDNA3.md` |
| M1 — dense IQR | **atteint et dépassé : 27B texte, décodage 27,2 tok/s (16 tok) / 20,8 (819), préfill 303 tok/s à 819** | fin S8 | W13 : préfill WMMA, split-K |

## Risques ouverts

- **R1** — *résolu* : W6 (attn h256 gqa 6, kv_store i8, imrope) est vert.
- **R2** — *résolu* : la tête vocabulaire était correcte ; la référence du test
  lisait la queue d'échelles au mauvais offset. Le test utilise désormais la
  référence GEMM validée et passe ; la row est servie par défaut (plus de
  `RAD_IQR_VOCAB`).
- **R3** — Relayout hôte scalaire : le chargement du 27B peut être long ;
  mesurer, puis vectoriser si nécessaire.
- **R4** — Aucune fixture MoE : référence hôte à produire en W5.
- **R5** — *résolu* : la vision est fusionnée et verte (142 rows).
- **R6** — *résolu* : `rmsnorm_had_quant_i8g` est déclarée, fusionnée (add +
  norme + Hadamard + int8), validée 7/7 contre la paire (dont l'absence de
  résidu et `wadd=0`), et le fold FFN `gemm_iqr_gated_i8a8h` est en place.
- **R7** — *résolu* : le préfill > 16 tokens segfaultait parce que
  `rad_gemm_run` lisait l'en-tête `rad.planes` sur un poids bf16 en VRAM ; le
  peek ne s'applique plus qu'aux poids privés. 33/83/819 tokens verts.
- **R8** — Les gains de `attn_paged` h256 et `attn_dense` hd72 ne sont pas isolés :
  les chiffres 27B les incluent sans les distinguer. Les noyaux restent dans
  l'arbre, marqués à mesurer puis optimiser en W13 (voir
  `TABLEAU_REEL_IDEAL_RDNA3.md`, « Optimisations différées »).
- **R9** — La vérification des formes E4M3/rotated sur un modèle non-IQR reste
  impossible : aucun 2B/4B sur l'hôte, seulement le 27B et le Flash-Next.
- **R10** — `gemm_iqr_gated_i8a8h` fusionné n'utilise que 1,4 bloc par CU
  (136 groupes, 96 CU) ; son gain est réel (~1,5 % en décodage) mais son
  occupation est à revoir en W13 (deux groupes par bloc ou 256 threads).
