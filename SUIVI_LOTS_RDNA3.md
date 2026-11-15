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
| W6 — attention h256/GQA/KV i8 | **à démarrer** | 3,5 sem | nécessaire à M1 (GQA 6) |
| W7 — QSA | à faire | 2,5 sem | |
| W8 — HC | à faire | 3,5 sem | |
| W9 — PLE | à faire | 1,5 sem | |
| W10 — vision | à faire | 3 sem | conteneur sans média autorisé (ADR-5) |
| W11 — intégration | à faire | 1,5 sem | |
| W12 — optimisation | à faire | 3–4 sem | |
| M0 — Go/No-Go | **GO (2026-11-15)** | fin S3 | `MEMO_M0_RDNA3.md` |
| M1 — dense IQR | bloqué par W4 (+ W6, embed, tête) | fin S8 | |

## Risques ouverts

- **R1** — Attention h256 GQA 6 absente de `libr3d` : M1 dépend de W6 ou d'un
  repli libref. Décision : démarrer W6 dès M0 Go.
- **R2** — Embedding à table L12 et tête vocabulaire IQR absentes : à planifier
  avant M1 (opérations hors W0/W1).
- **R3** — Relayout hôte scalaire : le chargement du 27B peut être long ;
  mesurer, puis vectoriser si nécessaire.
- **R4** — Aucune fixture MoE : référence hôte à produire en W5.
