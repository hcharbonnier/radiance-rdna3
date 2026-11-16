# Suivi des lots — RDNA3

État au 2026-11-16. Les estimations sont celles du plan ; « reste » est mis à jour
à chaque session. Le suivi détaillé des tâches restantes est dans
`CHECKLIST_RDNA3.md` ; l'audit du conteneur Flash-Next est dans
`FLASH_NEXT_AUDIT.md`. Ne pas réécrire `PLAN_IMPLEMENTATION_RDNA3.md`.

| Lot | État | Estimation | Reste à faire |
|---|---|---|---|
| Préparation | terminé | — | journal, suivi, ADR, environnement, reproduction |
| W0 — spike IQR 27B | **terminé (GO)** | 2,5 sem | parité et η atteintes ; mémo M0 |
| W1 — formes et hooks | en cours | 1,5 sem | audit fait ; table/grille denses fait ; **`kSigned` 8 et `w4nl` restants** |
| W2 — décodeur device | en cours | 1,5 sem | treillis + table/grille faits ; `kSigned`/`w4nl` en W15 |
| W3 — GEMM M ≤ 16 | **terminé (2026-11-15)** | 1,5 sem | GEMV 86 % ; split-K `down` 76,6 % (1,55×) |
| W4 — m64 et préfill | **terminé (résident) ; WMMA éteinte** | 2,5 sem | tuile résidente 15,5 TFLOP/s ; WMMA 25–32 % plus lente, à reprendre en W14 |
| W5 — MoE IQR | **en cours** | 4 sem | treillis + run map + référence hôte verts ; **`w4nl` livré** (MTP 48 inclus) ; restent formes mixtes par couche et experts bf16 protégés |
| W6 — attention h256/GQA/KV i8 | **terminé (revu, 15/15 device)** | 3,5 sem | kv_store i8 block 4 et rope imrope h128 livrés en W15 |
| W7 — QSA | **terminé (2026-11-16, 4 ops)** | 2,5 sem | tests libref verts ; perf à mesurer |
| W8 — HC | **terminé (revu, device vert)** | 3,5 sem | refus runtime `M=5` à corriger en bring-up |
| W9 — PLE | **terminé (revu, device vert)** | 1,5 sem | 4 rows ; perf à mesurer |
| W10 — vision | **terminé (revu, 142 rows vertes)** | 3 sem | `gemm_nt_bias`/`grid_embed` device livrés en W15 |
| W11 — intégration | **terminé (2026-11-15)** | 1,5 sem | tête vocabulaire validée et servie par défaut |
| W12 — optimisation | **terminé (2026-11-15)** | 3–4 sem | préfill 27B ×3,8 ; gated fusionné ; tête 16 bits ; h256/vision |
| W13 — optimisation | **terminé (2026-11-15)** | 2 sem | split-K `down` 1,55× ; vision −49 % à M=512 ; tête un lancement ; h256 ; WMMA mesurée plus lente et éteinte |
| W14 — audit et rows | **terminé** | 1 sem | audit Flash-Next ; 4 ops QSA ; MoE treillis + référence ; formes denses table/grille |
| W15 — rows Flash-Next | **terminé** | 2 sem | `kSigned` 8, `gdn_conv_recurrent_update_had`, `kv_store` b4, `rope` h128, router fusionné, `gemm_nt` N=1, rows vision, `moe_gemm_w4nl` ; **159 rows / 0 échec / 15 skips** |
| W16 — bring-up Flash-Next | **terminé (2026-11-16)** | 1 sem | serveur 121 GB servi : préfill 42 tok/s, décodage 4,0–4,6 tok/s, 4 séquences ; 0 bande device manquante |
| W17 — corrections kbench | **en cours (2 agents)** | 1 sem | `logits_gemm_bf16` (6 FAIL) et 3 formes QSA divergentes |
| M0 — Go/No-Go | **GO (2026-11-15)** | fin S3 | `MEMO_M0_RDNA3.md` |
| M1 — dense IQR | **atteint et dépassé : 27B texte, décodage 28,4 tok/s (16 tok) / 21,2 (819), préfill 301 tok/s à 819** | fin S8 | W15 : WMMA, occupation gated, mesure M=16384 |

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
- **R8** — *partiellement résolu* : `attn_dense` hd72 est mesuré (−49 % à M=512,
  M=32–512) ; `attn_paged` h256 reste mesuré de bout en bout seulement (préfill
  301, décodage 21,2 à 819). Les noyaux restent dans l'arbre, marqués à mesurer
  isolément et optimiser en W14.
- **R9** — La vérification des formes E4M3/rotated sur un modèle non-IQR reste
  impossible : aucun 2B/4B sur l'hôte, seulement le 27B et le Flash-Next.
- **R10** — `gemm_iqr_gated_i8a8h` fusionné n'utilise que 1,4 bloc par CU
  (136 groupes, 96 CU) ; son gain est réel (~1,5 % en décodage) mais son
  occupation est à revoir en W14 (deux groupes par bloc ou 256 threads).
- **R11** — *résolu, à garder en tête* : un build incrémental après `rsync -a`
  peut garder des objets périmés (mtimes préservés) et fausser un A/B de 25 %
  (préfill 301 contre 229). Les comparaisons se font dans des répertoires de
  build neufs, et le serveur doit être vérifié (`/proc/<pid>/exe`) avant de
  mesurer.
- **R12** — La tuile WMMA du préfill est construite, testée et bit-identique,
  mais 25–32 % plus lente que le `v_dot4` ; elle est éteinte par
  `kIqrWmmaMinM` au-delà de tout M, pas supprimée. W15/W16 doit la reprendre
  (émission/staging) avant de la rallumer.
- **R13** — *résolu* : toutes les rows manquantes sont livrées (W15) et le
  déclaratif Flash-Next ne laisse **aucune bande device non résolue** sur
  5 074 questions. Les 372 combinaisons non résolues sont des replis hôte.
- **R14** — Déploiement Flash-Next : le plan `qsa_score` fait 512 Mio à 2048
  lignes et 1 Gio à 4096 (le plugin refuse) ; garder `max_mixed_tok ≤ 2048` ou
  baisser `max-model-len`. La v1 fixe `num-speculative-tokens: 0` (MTP hors
  périmètre), `placement: expert_tiered`, `host-pool-mib: 43000`.
- **R15** — *résolu* : `moe_iqr_test` demandait une classe impaire plus étroite
  que la row refuse par contrat ; les plans carte sont passés à classes égales
  et le refus est épinglé. 10 formes, 0 écart (tolérance 2 ulp).
- **R16** — *résolu* : Flash-Next sert le texte. Les refus runtime (`hc_read`
  M=5, `moe_gather`, `moe_iqr`, router, tête vocabulaire) sont corrigés et le
  serveur répond sur 30/2290/4570 tokens et 4 séquences.
- **R17** — `logits_gemm_bf16` échoue sur les formes Flash-Next
  (`n_vocab=248320 n_embd=2560`, M=1..32) : 6 cas kbench, rel_l2 ≈ 0,5. Agent
  W17 : reproduire, corriger le chemin bf16 (le chemin quantifié et la tête IQR
  ne doivent pas changer), test de non-régression.
- **R18** — Trois descriptions de sortie QSA divergent de libref
  (`qsa_select.nsel` rang 2 au lieu de 1, `qsa_work.nc` `[8]` au lieu de
  `[seqs]`, `qsa_tail_store.state` `[8,…]` au lieu de `[M,…]`). Agent W17 :
  corriger et vérifier toutes les sorties QSA.
- **R19** — Les timings par row du kbench Flash-Next ne sont pas exploitables :
  la carte a throttlé (PPT0/PPT1) pendant les 8 minutes. HC/PLE et MoE/QSA
  restent à mesurer sur carte froide (`--kernels libr3d:libref`, petits lots).
