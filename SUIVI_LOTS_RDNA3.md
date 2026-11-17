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
| W5 — MoE IQR | **en cours** | 4 sem | treillis + run map + référence hôte verts ; **`w4nl` livré** (MTP 48 inclus) ; restent formes mixtes par couche et experts bf16 protégés ; temps par row mesuré en W23 (η 70,5 % IQR à M=32, 51 % w4nl à M=16) |
| W6 — attention h256/GQA/KV i8 | **terminé (revu, 15/15 device)** | 3,5 sem | kv_store i8 block 4 et rope imrope h128 livrés en W15 |
| W7 — QSA | **terminé (2026-11-16, 4 ops)** | 2,5 sem | tests libref verts ; perf mesurée en W18 (`qsa_score` 83,4 ms à M=512, 103 GFLOP/s) |
| W8 — HC | **terminé (revu, device vert)** | 3,5 sem | refus `M=5` corrigé ; coûts mesurés en W18 (`hc_read` 1,77 ms à M=1, η 0,6 %) |
| W9 — PLE | **terminé (revu, device vert)** | 1,5 sem | 4 rows ; perf mesurée en W18 (69 TFLOP/s à M=2 048) |
| W10 — vision | **terminé (revu, 142 rows vertes)** | 3 sem | `gemm_nt_bias`/`grid_embed` device livrés en W15 |
| W11 — intégration | **terminé (2026-11-15)** | 1,5 sem | tête vocabulaire validée et servie par défaut |
| W12 — optimisation | **terminé (2026-11-15)** | 3–4 sem | préfill 27B ×3,8 ; gated fusionné ; tête 16 bits ; h256/vision |
| W13 — optimisation | **terminé (2026-11-15)** | 2 sem | split-K `down` 1,55× ; vision −49 % à M=512 ; tête un lancement ; h256 ; WMMA mesurée plus lente et éteinte |
| W14 — audit et rows | **terminé** | 1 sem | audit Flash-Next ; 4 ops QSA ; MoE treillis + référence ; formes denses table/grille |
| W15 — rows Flash-Next | **terminé** | 2 sem | `kSigned` 8, `gdn_conv_recurrent_update_had`, `kv_store` b4, `rope` h128, router fusionné, `gemm_nt` N=1, rows vision, `moe_gemm_w4nl` ; **159 rows / 0 échec / 15 skips** |
| W16 — bring-up Flash-Next | **terminé (2026-11-16)** | 1 sem | serveur 121 GB servi : préfill 42 tok/s, décodage 4,0–4,6 tok/s, 4 séquences ; 0 bande device manquante |
| W17 — corrections kbench | **terminé (2026-11-16)** | 1 sem | `logits_gemm_bf16` corrigé (échelle 1×128), 3 formes QSA alignées ; kbench **856/856, 0 échec, 0 divergence** |
| W18 — mesures à froid | **terminé (2026-11-16)** | — | HC/PLE/QSA par row, 0 échec : `hc_read` 1,77 ms à M=1 (η 0,6 %), η lien 1,7 %, MoE par row mesuré en W23 |
| W19 — démo M6, robustesse, 16k | **terminé (2026-11-16)** | — | prompts distincts, 1/4/6 séquences, cache de préfixe, 13 421 tokens à 16k, contention, erreurs ; tier KV hôte off et 0 % en rejeu multi-agents documentés |
| W20 — contrat capacité/étendue | **terminé (2026-11-16)** | — | R24 corrigé (`logits_gemm_vocab_iqr`), même faute trouvée et corrigée dans 3 rows router ; mode `kbench --band-capacity` ; selftest 159/0/15, kbench 856/856 dans les deux modes, 27B `SEQS=4` à 67,3 tok/s |
| M0 — Go/No-Go | **GO (2026-11-15)** | fin S3 | `MEMO_M0_RDNA3.md` |
| M1 — dense IQR | **atteint et dépassé : 27B texte, décodage 28,4 tok/s (16 tok) / 21,2 (819), préfill 301 tok/s à 819** | fin S8 | W15 : WMMA, occupation gated, mesure M=16384 |
| M2 — η lien ≥ 80 % | **non tenu : 1,7 %** (42 tok/s contre 2 545) | fin S10 | le plafond mesuré est `hc_read` (89 tok/s), pas le MoE |
| M3 — QSA | **atteint** : équivalent au dense, 103 GFLOP/s à M=512 | fin S10 | `qsa_score` reste le poste QSA à optimiser |
| M4 — coûts documentés | **atteint** : HC, PLE, QSA par row à froid, MoE par row en W23 | fin S10 | η MoE 70,5 % à M=32 (IQR), 51 % à M=16 (w4nl), 10–12 % à M=1 |
| M5 — Flash-Next texte | **atteint** : 1/4/6 séquences, 8k et 16k, robustesse ; **vision livrée** (W26) | fin S12 | image 512²/1024² servie, `capabilities.image=true` |
| M6 — démo serveur | **atteint** | fin S12 | R21 (tier KV) et R22 (snapshots courts) résolus ; vision livrée (W26) |

## Prochains lots (W21–W24)

État : W0–W20 livrés, **W22 (R21 et R22) livré**, **W23 (MoE mesurable)
livré**, **W25 (MTP) livré**, **W26 (vision) livré** ; M0, M1, M3, M4, M5, M6
atteints ; **M2 non tenu** (η lien 1,7 %). **Périmètre révisé le 2026-11-16**
(ADR-7) : la vision et le MTP de Flash-Next entrent dans la cible ; tous deux
sont livrés, l'ADR-6 est supersédé et `num-speculative-tokens: 0` n'est plus
qu'un repli. L'ordre ci-dessous est celui de la valeur décroissante.

| lot | contenu | entrée | critère de sortie |
|---|---|---|---|
| **W21 — `hc_read` (R20)** | profil `--profile-ops true` sur Flash-Next, attribution du coût (lecture relue par ligne ? décodage E4M3 ? latence ?), correction | mesures W18 (1,77 ms à M=1), A/B deux-grilles déjà fait (mêmes temps) | `hc_read` M=1 ramené vers la borne de la row (≈20 µs) ou cause écrite ; décodage Flash-Next ≥ ×2 ; kbench 856/856 |
| **W22 — cache multi-tour (R21, R22)** — **fait** | R21 : rows u8 `gather_rows`/`scatter_rows` (fragments KV opaques, 4 128 octets), test `gather_rows_u8_device_test` ; R22 : ancre de queue pour un prompt sous le premier intervalle, rejeu 9 tours 24 % → **52,6 %** de tokens servis, `no_snapshot` 6 → 2, checkpoints 1 → 7 | tier actif sur Flash-Next (2 groupes, 51,38 Kio/entrée) et 27B (516 Kio/entrée) ; **0 % en rejeu agent** avant R22 | tier KV actif ; conversation de 3 tours > 0 token servi ; 32k tenu ; `no_snapshot` disparaît (hors premiers tours de sous-agents) |
| **W23 — MoE mesurable (R23)** — **fait** | vue complète des 5 plans (sans `fwht128`), échelle IQR ligne×bloc, `TablePointers` (slab + pointeurs hôte/device) container/replay, timer et marges ; formes IQR non servies refusées au layout | 110 cas `moe_gemm_q` sautés | kbench **968/968** (0 échec, 0 divergence) ; η MoE : IQR 70,5 % à M=32, w4nl 51 % à M=16 ; M=1 borné par le treillis (10–12 %) |
| **W24 — QSA et clôture doc** | `qsa_score` à M=512 (3.8), doc des 5 ops QSA (3.9), `docs/OPS.md` + `libr3d/README.md` (1.5/5.7/8.6), `RAPPORT` (5.6), handover final (8.7) | 83,4 ms à M=512 (2 % du pic) | `qsa_score` ≥ 20 % du pic ou cause écrite ; doc publiée ; décision d'ajout du RAPPORT prise |
| **W25 — MTP Flash-Next (R25, ADR-7)** — **fait** | rows `mtp_enter_e4m3`, `logit_rerank_iqr`/`_i8`, tests device ; corrige `gemm_nt_q_w2a8_m16` (en-tête joint lu sur l'hôte, chunk de K à M=1) ; `NS=3` servi | plugin `qwen4exp_fp8` déclare les ops MTP à `NS > 0` ; libr3d n'avait aucune des rows ; tables MTP 48 non posées par kbench | décodage **4,51 → 8,21 tok/s (1,82×)**, texte identique, acceptation 40,3 % ;   kbench **968/968** (FN) et 683/683 (27B) ; les 14 cas `mtp.draft_head` et
  les tables MTP 48 sont couverts (R23) |
| **W26 — vision servie (R26, ADR-7)** — **fait** | build `-DRAD_WITH_FFMPEG=ON` (libavformat 62.3 / ffmpeg 8.0), corpus de 4 PNG sur l'hôte, tours image 512² et 1024² (`data:` URL), `capabilities.image=true` | tour 856,68 Mio en VRAM, 48,27 Mio de staging pour 16384 patches | 512² : 256 tokens image, TTFT 8,19 s ; 1024² : 1024 tokens, TTFT 28,73 s ; réponses correctes ; 856/856 kbench dans les deux modes, 480/480 cas vision |

Hors v1 et non planifié : codebook `w4nl` dans `w4a8` seulement. Le MTP et la
vision sont revenus dans la cible le 2026-11-16 (ADR-7).

## Risques ouverts

- **R1** — *résolu* : W6 (attn h256 gqa 6, kv_store i8, imrope) est vert.
- **R2** — *résolu* : la tête vocabulaire était correcte ; la référence du test
  lisait la queue d'échelles au mauvais offset. Le test utilise désormais la
  référence GEMM validée et passe ; la row est servie par défaut (plus de
  `RAD_IQR_VOCAB`).
- **R3** — *résolu* : le chargement du 27B prend **2,7 s** pour 15,87 GiB
  (6,2 GB/s, dont 1,7 s de disque à 9,8 GB/s), modèle prêt en 3,3 s ; le
  relayout scalaire n'est pas un goulot sur cet hôte.
- **R24** — *résolu (W20)* : le 27B échouait avec `max-num-seqs > 1` parce que
  `logits_gemm_vocab_iqr` lisait le paramètre rangé comme le nombre de lignes au
  lieu d'une capacité de bande. La row lit maintenant l'étendue sur `x`, comme
  `libr4d`. L'audit `rad-kbench --band-capacity` a trouvé la même faute dans
  `router_topk`, `router_topk_scatter` et `router_sigmoid_topk` (corrigés). Test
  device épinglé ; kbench Flash-Next 856/856 dans les deux modes ; 27B `SEQS=4`
  à préfill 317,5 tok/s et 67,3 tok/s agrégés sur 4 requêtes. Contrat écrit dans
  `AGENTS.md` (`--band-capacity`) : le paramètre rangé est une capacité, l'étendue
  est dans les tenseurs.
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
  **Dérogation documentée (2026-11-16)** : inventaire de l'hôte vérifié —
  `qwen3.8-27b-iq4r-l-v4.rad` et `qwen3.8-next-flash-iqr-v4.rad`, tous deux IQR.
  La couverture actuelle est celle des rows et des tests device sur ces deux
  conteneurs ; la validation croisée non-IQR reste à faire dès qu'un conteneur
  est disponible.
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
  baisser `max-model-len`. La v1 fixe `placement: expert_tiered`,
  `host-pool-mib: 43000`. `num-speculative-tokens: 0` (ADR-6) n'est plus la
  configuration cible : le MTP revient avec l'ADR-7 et R25, et 0 reste le repli
  jusqu'à livraison des rows.
- **R15** — *résolu* : `moe_iqr_test` demandait une classe impaire plus étroite
  que la row refuse par contrat ; les plans carte sont passés à classes égales
  et le refus est épinglé. 10 formes, 0 écart (tolérance 2 ulp).
- **R16** — *résolu* : Flash-Next sert le texte. Les refus runtime (`hc_read`
  M=5, `moe_gather`, `moe_iqr`, router, tête vocabulaire) sont corrigés et le
  serveur répond sur 30/2290/4570 tokens et 4 séquences.
- **R17** — *résolu* : la tête `logits_gemm_bf16` indexait l'échelle par ligne
  (1×128) comme une grille 128×128 ; `HeadPlan`/`HeadQ` portent maintenant
  `(sbr, sbc)`, rel_l2 0,53 → 1,7e-3, test `logits_bf16_device_test` ajouté.
- **R18** — *résolu* : `qsa_select.nsel`/`seqused` en rang 1, `qsa_work.nc` en
  `[S]`, `qsa_tail_store.state` en `[M,1,ring,n]` ; kbench 0 divergence.
- **R19** — *résolu* : les timings par row ont été repris à froid (W18), sans
  throttler ; 182 + 147 cas, 0 échec. Voir `TABLEAU_REEL_IDEAL_RDNA3.md`.
- **R20** — `hc_read_e4m3` coûte **1,77 ms à M=1** (6,55 Mo de poids, η 0,6 %)
  alors que le commentaire de la row annonce 18,7 µs, et 0,23 ms par ligne
  au-delà de M=8 (matrices relues par ligne). À 48 appels par pas : ~93 ms du
  décodage et 47 % du préfill. L'A/B deux-grilles (g_hc_decode_form = 1) donne
  les mêmes temps et chaud = froid : ce n'est ni l'attente par tickets ni la
  bande passante. C'est le premier poste à instrumenter (hors périmètre de
  l'optimisation noyau demandée pour l'instant).
- **R21** — *résolu* : libr3d sert les fragments KV opaques avec les rows
  `gather_rows_u8`/`scatter_rows_u8` (dword par thread, aucun passage par le
  codec E4M3). Le tier se configure sur Flash-Next (2 groupes paginés,
  51,38 Kio/entrée, lots de 318) et sur le 27B (1 groupe, 516 Kio/entrée,
  lots de 64) ; en rejeu de 3 tours, `to_host` 471, `promoted` 245, 0 échec et
  0 coupe, avec `RADIANCE_DEBUG_TIER_VERIFY=1` (hash des octets copiés puis
  restaurés). Le test `gather_rows_u8_device_test` couvre les octets opaques
  (toutes les valeurs, indices -1/hors plan, pitches larges, M = 1/8/2 048).
- **R22** — *résolu* : sous `--prefix-resume exact` un prompt plus court qu'un
  intervalle n'avait ni point de grille ni snapshot, donc le tour suivant ne
  reprenait aucun état linéaire. Une ancre de queue (dernière frontière de
  quantum, un quantum plus bas hors grille, dès 256 tokens d'ancre) donne au
  rejeu 9 tours **52,6 %** de tokens servis (24 % avant), `no_snapshot`
  6 → 2 et 7 checkpoints écrits (1 avant). Compromis : sous le premier
  intervalle la reprise est hors grille, comme `nearest`, donc l'accord bit à
  bit n'est garanti que pour un prompt répété ; au-delà, `exact` reste strict.
  Les 2 `no_snapshot` restants sont les premiers tours des sous-agents (rien à
  reprendre avant leur propre ancre).
- **R23** — *résolu* (W23) : la vue complète des 5 plans est décodée (sans
  `fwht128`), l'échelle IQR est le produit ligne×bloc, et les tables MoE sont
  passées comme **table de pointeurs** (slab + pointeurs hôte/device) au
  conteneur comme au replay. kbench passe à 968/968, 0 échec, 0 divergence ;
  η MoE : IQR 70,5 % à M=32, w4nl 51 % à M=16, M=1 à 10–12 % (borné par le
  décodage treillis).
- **R25** — *résolu* (W25) : rows `mtp_enter_e4m3` et `logit_rerank_iqr`/`_i8`
  en device, plus deux corrections de `gemm_nt_q_w2a8_m16` (en-tête du poids
  joint lu sur l'hôte alors qu'il vit en VRAM ; MT dont le chunk ne divise pas
  K à M=1/K=2560). Flash-Next sert à `NS=3` : **4,51 → 8,21 tok/s (1,82×)**,
  texte identique, acceptation 40,3 % (66/61/37 % aux profondeurs 1–3).
  kbench 968/968 (FN) et 683/683 (27B) dans les deux modes ; les tables
  MTP 48 et le draft head sont couverts par kbench (R23). Le 27B reste à
  `NS=0`.
- **R26** — *résolu* (W26) : build `-DRAD_WITH_FFMPEG=ON` (libavformat 62.3,
  ffmpeg 8.0), tour de 27 blocs (856,68 Mio) placée en VRAM, staging média de
  48,27 Mio pour 16 384 patches. `capabilities.image=true` ; image 512² servie
  (256 tokens image, 281 avec le texte, TTFT 8,19 s) et 1024² (1 024 tokens,
  TTFT 28,73 s), réponses correctes. La vidéo est annoncée par le processeur mais
  n'a pas été testée. `kbench` 856/856 dans les deux modes et 480/480 cas
  vision, 0 divergence.
