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
| W11 — intégration | en cours | 1,5 sem | **`rmsnorm_had_quant_i8g` manquant** (bloque M1) ; tête vocabulaire IQR |
| W12 — optimisation | à faire | 3–4 sem | |
| M0 — Go/No-Go | **GO (2026-11-15)** | fin S3 | `MEMO_M0_RDNA3.md` |
| M1 — dense IQR | **chargé et à l'écoute, premier pas bloqué** | fin S8 | norme int8 `rmsnorm_had_quant_i8g` |

## Risques ouverts

- **R1** — *résolu* : W6 (attn h256 gqa 6, kv_store i8, imrope) est vert.
- **R2** — Tête vocabulaire IQR : le kernel existe mais échoue la référence ; il
  se refuse au run. À corriger avant M1.
- **R3** — Relayout hôte scalaire : le chargement du 27B peut être long ;
  mesurer, puis vectoriser si nécessaire.
- **R4** — Aucune fixture MoE : référence hôte à produire en W5.
- **R5** — *résolu* : la vision est fusionnée et verte (142 rows).
- **R6** — La norme int8 du trunk 27B (`rmsnorm_had_quant_i8g`) manque à
  libr3d ; l'architecture retombe sur la norme E4M3, qui refuse un q int8.
  Bloque M1. Une première tentative de composite a été annulée (`RAD_E_STATE`
  au plan de buffers) ; voir le journal.
