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
| W10 — vision | en cours (W10-lite) | 3 sem | reste `attn_dense` hd72 et `rope` axial |
| W11 — intégration | à faire | 1,5 sem | tête vocabulaire IQR à corriger |
| W12 — optimisation | à faire | 3–4 sem | |
| M0 — Go/No-Go | **GO (2026-11-15)** | fin S3 | `MEMO_M0_RDNA3.md` |
| M1 — dense IQR | bloqué par la vision uniquement | fin S8 | texte : toutes bandes résolvent |

## Risques ouverts

- **R1** — *résolu* : W6 (attn h256 gqa 6, kv_store i8, imrope) est vert.
- **R2** — Tête vocabulaire IQR : le kernel existe mais échoue la référence ; il
  se refuse au run. À corriger avant M1.
- **R3** — Relayout hôte scalaire : le chargement du 27B peut être long ;
  mesurer, puis vectoriser si nécessaire.
- **R4** — Aucune fixture MoE : référence hôte à produire en W5.
- **R5** — La seule dépendance restante de M1 est la tour vision (`attn_dense`
  hd72, `rope` axial). W10-lite la couvre ; ADR-5 reste le repli.
