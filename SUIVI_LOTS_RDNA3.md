# Suivi des lots — RDNA3

État au 2026-11-17. Les estimations sont celles du plan ; « reste » est mis à
jour à chaque session. Le suivi détaillé des tâches restantes est dans
`CHECKLIST_RDNA3.md` ; l'audit du conteneur Flash-Next est dans
`FLASH_NEXT_AUDIT.md`. Ne pas réécrire `PLAN_IMPLEMENTATION_RDNA3.md`.

| Lot | État | Estimation | Reste à faire |
|---|---|---|---|
| Préparation | terminé | — | journal, suivi, ADR, environnement, reproduction |
| W0 — spike IQR 27B | **terminé (GO)** | 2,5 sem | parité et η atteintes ; mémo M0 |
| W1 — formes et hooks | **terminé** | 1,5 sem | audit fait ; table/grille, **`kSigned` 8**, **`w4nl`** et **IQ8R groupe 256** (W27) livrés |
| W2 — décodeur device | **terminé** | 1,5 sem | treillis + table/grille, `kSigned`/`w4nl` (W15) et le groupe 256 (W27) |
| W3 — GEMM M ≤ 16 | **terminé (2026-11-15)** | 1,5 sem | GEMV 86 % ; split-K `down` 76,6 % (1,55×) |
| W4 — m64 et préfill | **terminé (résident) ; WMMA éteinte** | 2,5 sem | tuile résidente 15,5 TFLOP/s ; WMMA 25–32 % plus lente, éteinte |
| W5 — MoE IQR | **terminé** | 4 sem | treillis + run map + référence hôte verts ; `w4nl` livré (MTP 48 inclus) ; formes mixtes par couche couvertes par R23 ; experts bf16 protégés absents de Flash-Next (non exercés) |
| W6 — attention h256/GQA/KV i8 | **terminé (revu, 15/15 device)** | 3,5 sem | kv_store i8 block 4 et rope imrope h128 livrés en W15 |
| W7 — QSA | **terminé (2026-11-16, 4 ops)** | 2,5 sem | tests libref verts ; `qsa_score` à M=512 **2,23 ms** (36×, W28) |
| W8 — HC | **terminé (revu, device vert)** | 3,5 sem | refus `M=5` corrigé ; `hc_read` corrigé en W28 (**428 µs à M=1, 4,5×**) |
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
| M2 — η lien ≥ 80 % | **à re-mesurer** : 1,7 % (42 tok/s) mesuré en W18, **avant** la correction `hc_read` de W28 | fin S10 | le plafond de W18 était `hc_read` (89 tok/s) ; le corriger (4,5× à M=1, décodage 4,48 → 12,73 tok/s) rend le chiffre caduc |
| M3 — QSA | **atteint et dépassé** : équivalent au dense ; `qsa_score` à M=512 **2,23 ms** (36×, W28) | fin S10 | la mesure isolée de l'indexeur par pas reste à faire |
| M4 — coûts documentés | **atteint** : HC, PLE, QSA par row à froid, MoE par row en W23 | fin S10 | η MoE 70,5 % à M=32 (IQR), 51 % à M=16 (w4nl), 10–12 % à M=1 |
| M5 — Flash-Next texte | **atteint** : 1/4/6 séquences, 8k et 16k, robustesse ; **vision livrée** (W26) ; décodage **12,73 tok/s** après W28 | fin S12 | image 512²/1024² servie, `capabilities.image=true` |
| M6 — démo serveur | **atteint** | fin S12 | R21 (tier KV) et R22 (snapshots courts) résolus ; vision livrée (W26) |
| M7 — performance | **atteint pour les deux postes mesurés** : `hc_read` 4,5× à M=1 (W28), `qsa_score` 36× (W28) | fin S29 | reste le second passage `hc_read` et le η lien de bout en bout |

## Prochains lots (W21–W28)

État : W0–W28 livrés, **W22 (R21 et R22) livré**, **W23 (MoE mesurable)
livré**, **W25 (MTP) livré**, **W26 (vision) livré**, **W27 (IQ8R) livré**,
**W28 (hc_read, qsa_score, DSpark, folds, collectifs) livré** ; M0,
M1, M3, M4, M5, M6 et M7 atteints ; **M2 caduc, à re-mesurer** (η lien de W18,
antérieur au correctif `hc_read`). **Périmètre
révisé le 2026-11-16** (ADR-7) : la vision et le MTP de Flash-Next entrent dans
la cible ; tous deux sont livrés, l'ADR-6 est supersédé et
`num-speculative-tokens: 0` n'est plus qu'un repli. L'ordre ci-dessous est
celui de la valeur décroissante.

| lot | contenu | entrée | critère de sortie |
|---|---|---|---|
| **W21 — `hc_read` (R20)** — **fait (W28)** : staging LDS coalescé des matrices de mélange (16 lignes × 256 colonnes en `down`, 64 × 32 k en `up`), sommes f32 ascendantes inchangées ; **M=1 1 941 → 428 µs (4,5×)**, M=8 6,2×, M=128 7,9×, **M=512 114 → 12,4 ms (9,2×)** ; décodage Flash-Next **4,48 → 12,73 tok/s (2,84×)**. Profil : 96 sites, 1,93 ms/appel, poids en VRAM. Reste un écart ~40× au plancher de 10,5 µs (second passage multi-lignes) | mesures W18 (1,77 ms à M=1), A/B deux-grilles déjà fait (mêmes temps) | `hc_read` M=1 ramené vers la borne de la row (≈20 µs) ou cause écrite ; décodage Flash-Next ≥ ×2 ; kbench 856/856 |
| **W22 — cache multi-tour (R21, R22)** — **fait** | R21 : rows u8 `gather_rows`/`scatter_rows` (fragments KV opaques, 4 128 octets), test `gather_rows_u8_device_test` ; R22 : ancre de queue pour un prompt sous le premier intervalle, rejeu 9 tours 24 % → **52,6 %** de tokens servis, `no_snapshot` 6 → 2, checkpoints 1 → 7 | tier actif sur Flash-Next (2 groupes, 51,38 Kio/entrée) et 27B (516 Kio/entrée) ; **0 % en rejeu agent** avant R22 | tier KV actif ; conversation de 3 tours > 0 token servi ; 32k tenu ; `no_snapshot` disparaît (hors premiers tours de sous-agents) |
| **W23 — MoE mesurable (R23)** — **fait** | vue complète des 5 plans (sans `fwht128`), échelle IQR ligne×bloc, `TablePointers` (slab + pointeurs hôte/device) container/replay, timer et marges ; formes IQR non servies refusées au layout | 110 cas `moe_gemm_q` sautés | kbench **968/968** à l'époque (HEAD W28 : 948/384/653), 0 échec, 0 divergence ; η MoE : IQR 70,5 % à M=32, w4nl 51 % à M=16 ; M=1 borné par le treillis (10–12 %) |
| **W24 — QSA et clôture doc** — **fait (W28)** | réécriture coalescée/tuilée de `qsa_score` (tuile 32×64, pages et lignes de q en LDS, chaînes f32 ascendantes) : M=512 **82,27 → 2,23–2,30 ms** (**36–37×**, bit-identique à libref) ; `qsa_score_device_test` 72 checks exacts ; `--bench` 4 946/4 946 dans les deux modes ; doc et `RAPPORT_RDNA3.md` au dépôt | 83,4 ms à M=512 (2 % du pic) | **≥ 20 % du pic** (atteint : 36–37× la mesure de départ, 2,23 ms) |
| **W25 — MTP Flash-Next (R25, ADR-7)** — **fait** | rows `mtp_enter_e4m3`, `logit_rerank_iqr`/`_i8`, tests device ; corrige `gemm_nt_q_w2a8_m16` (en-tête joint lu sur l'hôte, chunk de K à M=1) ; `NS=3` servi | plugin `qwen4exp_fp8` déclare les ops MTP à `NS > 0` ; libr3d n'avait aucune des rows ; tables MTP 48 non posées par kbench | décodage **4,51 → 8,21 tok/s (1,82×)**, texte identique, acceptation 40,3 % ;   kbench **968/968** (FN) et 683/683 (27B) à l'époque (HEAD W28 : 948/948, 653/653) ; les 14 cas `mtp.draft_head` et
  les tables MTP 48 sont couverts (R23) |
| **W26 — vision servie (R26, ADR-7)** — **fait** | build `-DRAD_WITH_FFMPEG=ON` (libavformat 62.3 / ffmpeg 8.0), corpus de 4 PNG sur l'hôte, tours image 512² et 1024² (`data:` URL), `capabilities.image=true` | tour 856,68 Mio en VRAM, 48,27 Mio de staging pour 16384 patches | 512² : 256 tokens image, TTFT 8,19 s ; 1024² : 1024 tokens, TTFT 28,73 s ; réponses correctes ; 856/856 kbench dans les deux modes, 480/480 cas vision |
| **W27 — IQ8R groupe 256 (D1, P0)** — **fait** | `gemm_nt_q` `i8a8h` groupe 256 : encodage IQ8R au layout (produit bf16 des deux niveaux u8/bf16, une échelle par 256), noyaux m16/résident/WMMA templatés sur le groupe (un seul `sub` i32 sur deux blocs 128), rows `gemm_iqr_nt_m16_w256`/`gemm_iqr_tiled_w256`, split-K refusé ; producteurs `rmsnorm_had_quant_i8g`, `had_quant_act_i8g`, `gated_had_quant_i8g` en groupe 256 (deux rotations 128, un amax/scale par 256) et leurs rows `*_256_bf16` | `nq_h.declare` échouait (groupe 256 contre 128) ; le 2B ne chargeait pas | selftest **169 rows / 0 échec / 15 skips** à la livraison (au HEAD W28 : **190/0/16**) ; `act_i8g_group256_device_test` 67 checks puis **103** (W28), `iqr_device_test` 75 checks (8 cas 256 bit-exacts) ; kbench FN **968/968** et 27B **683/683** à l'époque (au HEAD W28 : **948/948**, **384/384** 2B et **653/653** 27B, 0 échec, 0 divergence) ; 2B IQ8R déclaré (523 ops, 423 poids) puis **servi à NS=0** (KV bf16 : texte cohérent, 40,2 tok/s à 60 tokens) ; kbench ciblé 2B **110/110** (pire rel_l2 4,018e-05, GEMM bit-exact) ; conteneur 2B **entier 384/384** (pire 1,810e-03, 30 rows device) |
| **W28 — performance, drafter, collectifs** — **fait** | `hc_read` : staging LDS coalescé des deux noyaux (M=1 4,5×, M=512 9,2×, décodage FN 2,84×) ; `qsa_score` : tuile 32×64 en LDS (M=512 36×, bit-identique) ; 2B à `NS=3` (DSpark : 1,56×, texte identique, cas `w2a8` M=7 réparé) ; `gate_had_quant_i8g` 128/256 + fold gated 256 ; 18 rows `all_reduce`/`all_gather` (test 2 rangs sur une carte) | W21 (décodage 4,5 tok/s), W24 (83,4 ms), D2, D3 | selftest **190/0/16** ; qsa 72, tp2 12, act_i8g 103, ple_hc 48, sched 548 782, e2e 909 (3 skips), fixture 92 385 (7 skips) ; kbench FN **948/948**, 2B **384/384**, 27B **653/653**, 0 échec, 0 divergence ; 27B texte identique ; FN **12,73 tok/s** |

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
- **R9 — résolu (W27/W28)** : un conteneur non-IQR est sur l'hôte
  (`minicpm5-2b-iq8r.rad`) ; il sert à `NS=0` et à `NS=3` (DSpark, 1,56×),
  et ses rows IQ8R (producteurs, GEMM groupe 256, `gate_had_quant_i8g` et fold
  gated 256) sont couvertes par `iqr_device_test` (cas 256 bit-exacts) et
  `act_i8g_group256_device_test` (103 checks), plus kbench 384/384 sur le
  conteneur entier. Reste non couvert : les variantes `ar_*` de tp > 1.
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
- **R20** — *résolu (W28)* : `hc_read_e4m3` coûtait **1,77 ms à M=1** (6,55 Mo
  de poids, η 0,6 %) alors que le commentaire de la row annonce 18,7 µs, et
  0,23 ms par ligne au-delà de M=8 (matrices relues par ligne). Le profil
  engine (96 sites, 1,93 ms/appel, poids en VRAM) a montré que le coût était
  dans le noyau : une lane par ligne au même offset (32× d'amplification de
  secteur en `down`, 8× en `up`). Les deux noyaux stagent maintenant leur
  tranche de poids en LDS : **M=1 1 941 → 428 µs (4,5×)**, M=512
  114 → 12,4 ms (9,2×), décodage Flash-Next **4,48 → 12,73 tok/s (2,84×)**.
  Reste un écart ~40× au plancher de bande (prochain passage : plusieurs lignes
  par tranche de poids).
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
  conteneur comme au replay. kbench passe à 968/968 (à l'époque ; HEAD W28 :
  948/384/653), 0 échec, 0 divergence ;
  η MoE : IQR 70,5 % à M=32, w4nl 51 % à M=16, M=1 à 10–12 % (borné par le
  décodage treillis).
- **R25** — *résolu* (W25) : rows `mtp_enter_e4m3` et `logit_rerank_iqr`/`_i8`
  en device, plus deux corrections de `gemm_nt_q_w2a8_m16` (en-tête du poids
  joint lu sur l'hôte alors qu'il vit en VRAM ; MT dont le chunk ne divise pas
  K à M=1/K=2560). Flash-Next sert à `NS=3` : **4,51 → 8,21 tok/s (1,82×)**,
  texte identique, acceptation 40,3 % (66/61/37 % aux profondeurs 1–3).
  kbench 968/968 (FN) et 683/683 (27B) à l'époque (HEAD W28 : 948/948 et
  653/653) ; les tables
  MTP 48 et le draft head sont couverts par kbench (R23). Le 27B reste à
  `NS=0`.
- **R26** — *résolu* (W26) : build `-DRAD_WITH_FFMPEG=ON` (libavformat 62.3,
  ffmpeg 8.0), tour de 27 blocs (856,68 Mio) placée en VRAM, staging média de
  48,27 Mio pour 16 384 patches. `capabilities.image=true` ; image 512² servie
  (256 tokens image, 281 avec le texte, TTFT 8,19 s) et 1024² (1 024 tokens,
  TTFT 28,73 s), réponses correctes. La vidéo est annoncée par le processeur mais
  n'a pas été testée. `kbench` 856/856 dans les deux modes et 480/480 cas
  vision, 0 divergence.
- **R27** — *résolu (W28), règle conservée* : déclarer une op change le
  **graphe** de l'arch. `gate_had_quant_i8g` fait prendre à Flash-Next la
  branche fusionnée, émise sur **toute la projection** (`[T, heads*head_dim]`)
  avec une porte en vue 3D de pas de tête 512 ; la row refusait la géométrie et
  le moteur s'arrêtait à l'étape 0. Corrigé et couvert par un cas device
  (103 checks) ; la règle « kbench vert ne prouve pas le graphe, servir le
  modèle après le merge » est dans `AGENTS.md`.
- **R28** — *ouvert* : les 18 rows `all_reduce`/`all_gather` sont livrées et
  testées à 2 rangs sur une carte (W28), mais les **jonctions `ar_*`
  fusionnées** restent des stubs (leurs producteurs ont des signatures
  différentes de libr3d) : un modèle `tp>1` n'est pas encore servable.
- **R29** — *résolu (W29)* : `main` 2.1.0 retire `ngram_ids_i32` de libr3d ;
  Flash-Next (M=2048 sur device) échoue alors au démarrage sur « no kernel
  resolved ». Row restaurée (`c11c451`), service de nouveau correct.
- **R30** — *résolu (W29)* : la généralisation `GS=1/2` du noyau IQR (W27,
  IQ8R) coûtait **17 %** au préfill 27B ; la spécialisation `GS=1`
  (`faa89fd`, noyau resident W26) redonne 3 839 µs / 288 tok/s, IQ8R inchangé.
- **R31** — *résolu pour `build-clean`* : les caches 27B + Flash-Next sont
  fusionnés dans `build-clean/radiance_home/tune` ; le kbench y sert `impl=3`
  (132,3 µs contre 776,8 en `impl=0`, **5,9×**). Artefact machine non
  versionné : à livrer avec chaque déploiement (ou rejouer `rad-tune`).
- **R32** — *amont* : `down_w5_fp8` non déclaré dans
  `arch/common/rad_block_moe_fp8.h` (casse `flint`/`bailing_hybrid`) et
  retrait de `ngram_ids_i32` cassant RDNA3 ; correctifs locaux `5e98eb9` et
  `c11c451`, à proposer en amont.
- **R33** — *décision (W29)* : `mtp_enter` **n'est pas un chantier 2–4×**.
  La grille est `(n/8 canaux, M rangées)`, un bloc par (8 canaux, rangée) ; la
  linéarité en M est le travail d'un GEMV par rangée (normalisation + décodage
  E4M3 + dot), pas un défaut de parallélisme. À M≤4 : 186–512 µs, <0,5 % du
  token ; profiler le graphe servi avant toute refonte.
- **R34** — *couvert (W29)* : `attn_paged_device_test`
  couvre les rows h256 i8 (gqa6 bs16, gqa12 bs4) en décodage/préfill, court et
  split : **8 checks/9 cas** (le 9e est le mode temps), accord à **1 bf16 ULP**
  de libref, **pas bit-exact** mais écart **documenté** (polynôme `exp` à
  1 f32 ulp, parenthèses du split-KV). `RAD_ATTN_BENCH=1`, **scratch/split-KV
  alloué**, donne **4,06 ms** (gqa6) / **2,37 ms** (gqa12) à 32K q=1 (sans
  scratch, repli `splits=1` non servi : 29,1/33,3 ms). Le **préfill** est
  passé au split-KV en W30 (voir ci-dessous) ; reste le multi-rangées verify
  M=4, en préservant l'ordre par rangée.

## W29 — rebase `main` 2.1.0 + GS=1 (2026-11-17)

- Rebase 144 commits, 4 conflits additifs, `ple_conv` dédoublonné au profit de
  `main`, `moe_gather` corrigé (capacité), ABI 40, QSA `kQsaUnread`/`kQsaOnce`.
- 195 rows, 0 échec, 16 skips ; kbench 27B 3 181/3 181, Flash-Next
  4 786/4 786, MiniCPM 1 802/1 802 ; 27B 288 tok/s (GS=1), Flash-Next long
  contexte 3 252 tokens 234,8 tok/s.
- Détail et suites : `RAPPORT_REBASE_V2_1_0.md`.

## W30 — préfill attention h256 i8 en split-KV (2026-11-18)

- **Cause mesurée** : q=17 coûtait le même temps que q=128 à 8K (12,1 vs
  12,5 ms) pour 7,5× moins de travail — 8 workgroups, marche séquentielle sur
  tout le contexte. Le préfill était le dernier chemin à `splits=1`.
- **Livré** : l'entrée de préfill prend `splits`/scratch ; échelle ciblant
  **128 workgroups**, plafonnée au nombre de blocs du cache, puis divisée
  jusqu'à tenir dans l'aréna (repli `splits=1` inchangé, décodeur inchangé) ;
  les rows de préfill déclarent `attn_h256_scratch`.
- **Gain** (build-p0 → build-opt, `RAD_ATTN_BENCH=1`) : q=17 **5,1–9,0×**
  (2 376 µs à 8K gqa6, 8 508 à 32K ; 1 795 / 6 364 gqa12) ; q=128 **1,39–1,49×** ;
  décodage inchangé. Une cible à 256 workgroups ne donne rien : le noyau
  sature, borné par l'instruction (conversion int8 + FMA par lane et par dim).
- **Porte tenue** : `attn_paged_device_test` **8 checks/9 cas**, selftest
  **195/0/16**, `sched_test` 548 952, `engine_e2e_test` 1 076 (3 skips),
  `fixture_test` 109 671, tests device verts ; texte **identique** Flash-Next
  3 252 tokens (32 tokens, drafts 19/39) et 27B/Flash-Next à 372 tokens.
  Le prélude long ne bouge pas (chunks de 2 048 tokens → `splits=1`) : le gain
  porte sur les chunks courts.
- **Reste** : au-delà, changer l'arithmétique (`v_dot4_i32_i8` après
  quantification de q, ou softmax en une passe) — nouveau contrat numérique, à
  mesurer séparément.
- **R35** — *décision (W30)* : le split du préfill est autorisé au même titre
  que celui du décodage. Le test ne pinçait pas « préfill = une passe » : il
  pinçait la tolérance **1 bf16 ULP**, et les tranches sont sommées en ordre
  ascendant puis réduites, comme pour le décodage. Un test qui exigerait le
  bit-exact sur un préfill multi-tranches serait faux.
- **R36** — *clos par la mesure (W30)* : le **multi-rangées verify M=4** n'est
  pas un levier. gqa6 (27B) q=4 coûte le même temps que q=1 (969,8 vs 921,3 µs
  à 8K ; 3 915,0 vs 3 872,6 à 32K) ; gqa12 paie +23–31 % d'un pas (707,1 vs
  539,9 µs à 8K), soit le travail de 4× plus de rangées réelles dans la même
  tuile. Même conclusion que R33 : ne pas rouvrir sans profiler le graphe.
- **Service 4 séquences (27B, W30)** : validé. Quatre requêtes distinctes
  (~90 tokens, 64 générés) lancées ensemble, `max-num-seqs 4`, KV i8, NS=0 :
  **49,4 / 49,2** tok/s agrégés sur `build-p0` (W29) contre **50,3 / 48,9** sur
  `build-opt` (W30), ~16 tok/s par flux — aucune régression du split-KV du
  préfill. **Réserve** : la reproductibilité du texte suppose une composition
  de lot fixe ; le nombre de tranches dépend de `n_seq`, comme le split du
  décodage. Ne pas promettre un texte bit-identique à travers deux
  compositions de lot.

## W31 — vision et préfill, trois noyaux (2026-11-18)

Trois agents en worktrees séparés, chacun sur son arbre hôte, tous les runs
sous le même `flock` (aucun bench concurrent).

- **Livré `hc_read`** : tuilage par rangées (32 sorties × 8 rangées en `down`,
  32 colonnes × 8 rangées en `up`), noyaux W28 à une rangée conservés sous une
  tuile (décodage). **2,6–2,9×** à M ≥ 128 (128 : 3 766 → 1 298 µs ; 4 096 :
  107 976 → 38 264), M=1 inchangé (400,3 → 399,2), **bit-exact**.
- **Livré `gemm_nt_bias`** : staging bf16 k-major, tuile 32×64, 4×4 par thread,
  pitchs 36/68 halfwords (plus de conflit 16 voies), mul/add séparés.
  **8,7–11×** (1 045×1152×1152 : 3 332 → 385 µs, 7,20 TFLOP/s), **0 ulp**,
  kbench 140/140 `rel_l2 = 0` ; la row est désormais 2,3–2,5× plus rapide que
  libavx.
- **Mesuré `attn_dense`, aucun noyau changé** : la forme servie est 16 384
  patches × 16 têtes, la row fait **5,30 TFLOP/s** (efficacité croissante avec
  M), soit 53 % de sa borne d'issue VALU ; les variantes de scheduling sont
  neutres. Mode temps ajouté (`RAD_ATTN_DENSE_BENCH=1`).
- **Effet servi** : Flash-Next texte 972 tok 4,78 → **3,13 s** ; 3 252 tok
  13,56 → **8,19 s** ; image 1 024² 9,58 → **4,66 s** ; image 2 048² 39,2 →
  **19,1 s** ; 27B inchangé (IQR 88 %). Texte servi identique au bit.
- **Validation** : tous les tests verts, `r3d_selftest` 195/0/16.
- **R37** — *correction (W31)* : le « 19 GFLOP/s à M=4096 » de W30 était un
  comptage faux (16 têtes et 16 384 patches oubliés). `attn_dense` n'est pas
  effondré : 5,30 TFLOP/s à la forme servie. Ne pas rouvrir un chantier
  scheduling sur cette base ; pour aller plus vite il faut le FMA ou une
  softmax en une passe (contrat).
- **R38** — *décision (W31)* : le contrat **bit-exact** de `gemm_nt_bias` est
  tenu en gagnant 8–11×. La référence libr4d (`gemm_bf16_nt_bias_tiled`) est
  tuilée fp32 et ne promet pas le bit-exact, mais rien n'oblige à s'en écarter :
  le plafond bit-exact est ~11 TFLOP/s (mix 2,75 instr/MAC). Ne pas passer au
  chemin `dot2` (23,5 TFLOP/s) sans décision explicite de contrat.
- **Reste** : `moe_gemm_q` est devenu le premier poste du préfill Flash-Next
  (46 % à 972 tokens) ; `hc_read` garde 1,7× vers son plancher ; `attn_dense`
  1,3–1,9× sans contrat ; le 27B reste borné par la tuile IQR.

## W32 — `hc_read` 2ᵉ passe, `attn_dense` clos, MoE rejeté (2026-11-18)

Quatre agents en worktrees séparés, un seul run GPU à la fois.

- **Livré `hc_read`** : deux rangées par lane en `down` (1,5 LDS/MAC) et
  2 colonnes × 2 rangées en `up` (1,0 LDS/MAC), produits formés avant des
  additions ascendantes ; `hc_norm` 64 rangées par bloc (8,5 → 0,46 ms) ;
  `q`/`rq` amax par réduction de bloc et rotation étage par étage (9,2 →
  0,8 ms). **1,23× à M=1**, **1,78× à 128**, **2,94× à 512**, 2,8× à 4 096 ;
  cumulé depuis W30 : **7,7–7,9×**. Bit-exact, `ple_hc_device_test` 58/10.
  Servi : texte 972 tok 3 133 → **2 560 ms**, 3 252 tok 8 186 → **6 316 ms**,
  image 1 024² 4 660 → **4 049 ms**, texte identique au bit.
- **`attn_dense` : clos sous contrat.** 437 VALU/clé + 182 non-VALU = 619 ;
  plancher 115 ms, 100 % d'issue 163 ms, mesuré 233 ms (~70 %). lane = clé
  **+23 %**, kStage 64 ×9,9, launch_bounds(128,3) ×3,8, waves 3/5 bruit.
  Aucun noyau modifié ; plancher réaliste ~5,3 TFLOP/s à la forme servie.
- **Rejeté : MoE run map** (tuile 16/32/64). kbench **1,27× à M=128** mais
  servi **+8,5 %** (2 550 → 2 768 µs) : à M=1204 chaque expert porte ~23
  rangées, une tuile de 32 double les blocs et **décode deux fois** chaque
  unité 16×128. Le commit cassait aussi le service (scratch déclaré avec la
  tuile de la bande, capacité trop petite → `RAD_E_SCRATCH` au premier pas du
  prompt de 3 252 tokens). Retiré ; l'agent reprend avec la géométrie servie
  comme porte.
- **Validation** : `sched_test` 548 952, `engine_e2e_test` 1 076 (3 skips),
  `fixture_test` 109 671, `ple_hc_device_test` 58/10, `moe_iqr_device_test`
  15/15, `vision_rows_device_test` 66/10, `iqr_device_test` 76/11,
  `attn_paged_device_test` 8/9, `r3d_selftest` 195/0/16.
- **R39** — *décision (W32)* : un gain kbench à M ≤ 512 ne vaut rien pour une
  row du graphe servi. Toute optimisation de `moe_gemm_q` se juge à
  **M ≈ 1024–2048** (la géométrie réelle) et, si possible, sur le TTFT servi ;
  une variante qui perd à cette géométrie ne rentre pas, même si elle gagne
  ailleurs.
- **R40** — *clos (W32), confirmé par la mesure (W40a)* : `attn_dense` hd72
  sous contrat bit-exact est à son plancher d'issue (~5,3 TFLOP/s). Le poste
  est borné par la **latence** de la chaîne du dot, pas par les slots :
  **FMA = 3,3 % plus lent** (437 → 221 VALU/clé, 241,3 contre 233,6 ms à
  16 384 × 16). Une **softmax en une passe** avec scores de tuile en LDS donne
  **1,46× au noyau et 1,15× au TTFT 2048²** (14,63 contre 16,79 s), mais le
  texte servi change sur les deux images → **rejeté par le gate** (`9577eda3`
  ne livre que la documentation). Rouvrir demande un store exact qui tient, ou
  un nouveau contrat de texte servi.
- **Reste** : `moe_gemm_q` (56,5 % du préfill 972 tokens) via la continuation
  de trellis, puis `hc_read` (17,6 %).
- **`hc_read`, W40b** : tuile down **2×2×64** au-delà de 256 rangées
  (`a4ed7281`), oracle `M=257` ajouté (`8401e6fd`, 58 → **76 checks**),
  **5,3–8,7 %** bit-exacts sur la plage servie 512–4 096 et TTFT −0,1 à
  −0,9 %, textes identiques. Les plans f32 pré-décodés, le prefetch registre,
  P2Q4/P4Q2 et WMMA/`v_dot2` sont **rejetés sur mesure** ; la marge de **1,7×**
  n'est pas atteignable sous contrainte bit-exacte (dot à 2,75 instr/MAC
  contre un plancher de 2,0, le reste est de la latence à 24 vagues). Ce n'est
  plus un reste à faire.
- **MoE / chaleur, W40c** : budget d'échanges piloté par la résidence
  (`49f9317a`) — 16 après un broad pass sur un plan ≥ 60 % résident, 64 sinon,
  latch 512 étapes, `--expert-moves N`. **−21 % d'octets de promotion**
  (160,4 → 57,5 Mio/token), décodage 3 252 **−1,75 %** (NS=0) et −0,7 % (NS=3),
  TTFT −1,5 %, textes identiques, kbench Flash-Next 4 786/4 786. Rejetés :
  `kXpSlots 4`, budget 16 ou 48 global, sans chaleur. Le décodage à 3 252 est
  **latency-bound** (92,4 ms de pas : `hc_read` 28,6, `attn_paged` 25,4,
  `moe_gemm_q` 20,7) : ne pas rouvrir « octets/token » sans profil.
- **IQR — tuile 2×4 rejetée (W32)** : 1,94× réel à M=1626 (32,91 → 16,94 ms,
  34,2 TFLOP/s, 0 ulp) mais sur la grille **i8 signée**, gardée par
  `F::KIND == kSigned && F::B == 8` ; le 27B est **trellis u4/u5**
  (`rad-info --recipe` : `tcq=16 tcq_code=mad`), donc la tuile ne tourne pas.
  Servi inchangé (TTFT 11 244 ms, `gemm_nt_q` 10 048 ms). La piste trellis est
  le **recouvrement** du décodage K+1 avec la consommation de K (double buffer),
  pas une copie parallèle d'octets, la chaîne d'états n'étant pas parallèle.
- **R41** — *décision (W32)* : une optimisation d'une row ne se juge que sur
  la **forme que les conteneurs servis déclarent**. Avant de toucher à
  `gemm_iqr_tiled`, vérifier `rad-info --recipe` (trellis vs signé) ; une tuile
  gardée par `kSigned` ne peut pas gagner sur le 27B, le Flash-Next ou
  MiniCPM. Une variante device-only se refuse même si elle est plus rapide et
  exacte.

## W33 — `gemm_nt_q` trellis en tuile registre 2×4 (2026-11-18)

- **Cause mesurée** : le décodage trellis sur la vague 0 n'est pas le limiteur
  (parallélisé : 1,04× à M=1626, 0,997× à M=512, déjà recouvert par 4 blocs/CU).
  Le limiteur est le **flux d'activation et la chaîne `dot4`** : l'ancienne
  tuile faisait deux chargements par `dot4` et consommait chaque dword
  d'activation une seule fois.
- **Livré** : 2 tokens × 4 lignes par voie (0,75 chargement/`dot4`, 8 sorties),
  décodage de l'unité par les quatre vagues en parallèle (`state1`, additif),
  226 VGPR, 2 blocs/CU, **bit-identique**, réservé à `kTrellisMad`/`LRB=0` et
  `M >= 128`.
- **Device** : M=512 10 557 → **6 098 µs**, 1 024 21 355 → **11 969**,
  1 626 36 901 → **21 744**, 2 048 46 667 → **28 549** (1,63–1,78×,
  30,5 TFLOP/s à M=1 024).
- **Servi** : 27B 3 252 tokens **TTFT 11 225 → 7 222 ms (1,55×)**, `gemm_nt_q`
  1,69× (préfill **288 → 450 tok/s**) ; 372 tokens 1 326 → 986 ms ; décodage
  inchangé (28,93 tok/s) ; texte identique au bit.
- **`hc_read`, N en scratch : mesuré, non livré.** N est reconstruit 12× par
  rangée ; le sortir du `down` rend 10,8–12,6 % du core, mais le scratch net
  **~8 % (≈1,09× à 4 096)**, sous la barre de 1,15×.
- **R42** — *livré (W33)* : la tuile registre 2×4 est le bon levier pour le
  trellis, et le décodage parallèle n'est qu'un appoint (il fait tomber le gs1
  de 110 à 87 VGPR et évite le spill de la tuile). Ne pas rejouer la piste
  « décodage distribué » seule : elle vaut ≤ 4 %. La borne restante est
  l'arithmétique (37 % du plafond f16 WMMA).
- **Reste** : 27B `attn_paged` devient le 2ᵉ poste (12,8 %) ; préfill Flash-Next
  toujours dominé par `moe_gemm_q` (PCIe) ; `hc_read` 1,7× vers son plancher.

## W34 — stager de préfill (2026-11-18, essayé, non livré)

- **Cause confirmée** du préfill 972 tokens : deux chunks par site (896 + 76)
  sous l'ancien seuil du stager, donc 19,08 GiB d'experts relus sur PCIe sur le
  chemin critique (35,5 GiB par préfill, 81 % du lien x16).
- **Tentative** : ring réservé aux niveaux d'arène intermédiaires, publié pour
  les passes ≥ 256 rangées, seuil du planificateur inchangé (aucun quantum ni
  ancre R22 touché).
- **Résultat** : préfill 972 tok **2 497 → 2 102 ms (1,19×)**,
  `moe_gemm_q` 1,30× ; neutre à 3 252 et 92 tokens ; mais **décodage −7 % à
  92 tokens et −5 % à 3 252** (30,5 → 28,3 et 19,35 → 18,4 tok/s),
  reproductible à 1 ms près. Texte identique.
- **Cause du coût** : le prêt de la dalle n'était jamais **rendu** après un
  chunk stagé ; `arena_step` ne faisait que le rétrécir. Après 64 pas de
  décodage la dalle tenait **16 441 experts au lieu de 16 802**, et les 361
  manquants étaient relus sur PCIe.
- **Correction** : `set_loan` rend le prêt plein au premier pas dont la passe
  ne stagère pas. Résidence restaurée (16 802), décodage **19,25 tok/s** à
  3 252 et **30,13** à 92 tokens (≤ 1,2 % de l'arbre sans stager), préfill
  972 tok **2 120–2 133 ms (1,18×)**, texte identique.
- **Décision** : **fusionné** (`52fd19d`, correctif compris). Le levier rentre
  dans R43.
- **R43** — *règle de mesure (W34)* : tout levier moteur qui touche l'arène se
  juge sur **les deux** : TTFT des prompts concernés **et** débit de décodage ;
  > 2 % de décodage = refus. Le stager est passé sous cette barre après
  correction.
- **Reste** : la proposition chiffrée sur le chunk de queue (99 tokens = 86 %
  des experts), sans toucher à `core/sched/` ; et la marge `hc_read` (1,7× vers
  son plancher).

## W36 — chunk de queue : proposition examinée, non retenue (2026-11-18)

- **Correction du chiffre** : pour 972 tokens les chunks sont **896 + 76** (le
  704 + 99 est le cas d'un prompt de 803). Le 76 relit **77 % des experts**
  (760 slots sur 512), soit **14,0 GiB** en place, ~0,65–0,75 s des 2,04 s ;
  le chunk de 896 stage 18,2 GiB.
- **Où** : `Scheduler::tail_checkpoint` (`core/sched/scheduler.cpp:1292`,
  ancre 1297–1305, `kShortTailMin` 256 à `scheduler.h:507`) et
  `plan_chunk` (`scheduler.cpp:1104`, coupe en 1131/1133). La coupe naît de
  `--prefix-resume exact` (ancre 896 pour 972), pas du quantum.
- **Ce qu'une fusion coûterait** : l'état linéaire n'est capturé qu'en **fin**
  de chunk ; un snapshot à 896 au milieu d'un chunk de 972 demande une
  capacité moteur nouvelle. L'invariant « les chunks du run froid » tombe pour
  la queue : un hit `exact` à 896 recalculerait 76 tokens dans une forme jamais
  exécutée à froid, donc le texte n'est plus garanti identique. Relever
  `kShortTailMin` par défaut change la politique pour tous les prompts sous le
  seuil et supprime l'ancre courte. Aucun signal « pas de reprise possible »
  n'existe (le cache de préfixe VRAM reste actif même à `prefix-cache-mib: 0`).
- **Mesure qui tranche** : un chunk plus gros **ne paie pas** ici. Budget
  4096 + `checkpoint-interval` 4096 sur le prompt de 3 252 (un chunk au lieu de
  2048 + 1204) : **5 453 → 6 486 ms (+19 %)**, texte identique, décodage
  inchangé. L'arithmétique « un chunk au lieu de deux = une copie d'experts en
  moins » est donc fausse sur ce modèle : le coût d'arène et de résidence
  dépasse la copie épargnée, comme le GUIDE §4.5 l'annonce pour les gros
  contextes.
- **Décision** : **ne pas toucher à R22**. Le gain projeté de la fusion
  (~0,65 s sur 972) est du même ordre que ce que la mesure ci-dessus détruit,
  et il s'achète en cassant l'exactitude de reprise. Si le produit veut un jour
  le gain *single-shot*, la seule forme acceptable est un **drapeau**
  (`--tail-anchor off`) qui fait rendre -1 au cas court de `tail_checkpoint`,
  jamais le défaut, avec trois tests à écrire d'abord (ancre courte linéaire à
  I=2048 dans `sched_test`, e2e froid/hit/tour appendu à I=2048, fixtures
  `--strict`).
- **R44** — *décision (W36)* : le chunk de queue est une **conséquence de
  l'ancre de reprise**, pas un gaspillage. Ne pas le fusionner par défaut ;
  toute variante passe par un drapeau et une revue R22. Et vérifier par la
  mesure, pas par l'arithmétique, qu'un chunk plus gros réduit le TTFT : sur
  Flash-Next le contraire est mesuré (+19 % à 3 252 tokens en un chunk).

## W37 — `--tail-anchor off` (2026-11-18, livré opt-in)

- **Livré** : `--tail-anchor on|off` (défaut `on`). Avec `off`, le cas court de
  `tail_checkpoint` rend -1 : un prompt sous le premier intervalle s'exécute en
  **un chunk** et n'écrit **aucun snapshot hors grille**. Aucun autre
  changement (intervalle, quantum, `--prefix-resume`).
- **Effet servi** (Flash-Next, même build, texte identique) : préfill 972 tok
  **2 068 → 1 488 ms (1,39×)**, décodage −0,9 % (bruit) ; 3 252 tok neutre et
  décodage identique.
- **Prix** : la conversation courte re-préfille son prompt au tour suivant
  (0 snapshot, `reason="no_snapshot"`), d'où le **défaut `on`**.
- **Tests** : `sched_test` 548 964 (+12), `engine_e2e_test` 1 087 (+11),
  `config_test` 2 927, suite complète verte, selftest 195/0/16.
- **R45** — *décision (W37)* : le drapeau est la forme livrable de la fusion
  de queue, et il reste **opt-in**. Un serveur conversationnel le laisse à
  `on` (l'ancre courte sert M6/R22) ; un serveur à prompt unique ou orienté
  préfill sur MoE le passe à `off` et gagne 1,39× sur les prompts de 256–1 024
  tokens. Ne pas changer le défaut sans mesurer la reprise d'un tour appendu.

## W38 — jalon validation : kbench complet, strict, `--tail-anchor` (2026-11-18, aucun code)

- **But** : solder la dette de mesure de W30–W37 sur un build neuf (`6d9084b`
  rsyncé, `build-w38` Release/gfx1100/FFmpeg, cache de tune recopié) : les
  compteurs kbench d'`AGENTS.md` dataient de W29, et seuls des runs ciblés
  avaient été refaits après `hc_read` (W31/W32), `gemm_nt_bias` (W31) et la
  tuile trellis de `gemm_nt_q` (W33).
- **kbench complet, `--bench`, un conteneur à la fois** : 27B **3 181/3 181**,
  MiniCPM **1 802/1 802**, Flash-Next **4 786/4 786** — 0 échec,
  0 divergence, comptes **identiques** à W29, pires `rel_l2` identiques
  (3,619e-03 / 3,485e-03 / 1,520e-02). 542 + 368 + 2 310 s de carte.
- **Replay strict des fixtures** (forme du ctest `kbench_ref`, `libref` seul,
  `--strict --coverage data/kbench_rows`) : **1 134/1 134**, 0 stale,
  0 divergence, pire `rel_l2` 3,442e-03, exit 0, 80 s. `fixture_test` n'a pas
  d'option `--strict` : la commande juste est celle du ctest.
- **Suite de tests** : compteurs de W37 reproduits au chiffre près
  (`sched_test` 548 964, `engine_e2e_test` 1 087/3 skips, `fixture_test`
  109 671, `config_test` 2 927, tous les device verts, selftest **195/0/16**).
- **`--tail-anchor off` mesuré à 1 536/2 047/2 048** (payloads identiques entre
  modes, nonce neuf par probe, `cache_n = 0`, `prompt_n` exact par bissection,
  deux réplicats) : Flash-Next **1,16–1,18×** à 1 536, **1,16×** à 2 047,
  **1,02×** à 2 048 ; 27B **1,07–1,08×**, **1,06×**, **1,01×**. Textes
  identiques sur les 12 paires ; décodage inchangé (NS=0, 8 tokens).
- **Décision** : jalon validation de la §6.A du handover W37 **clos**. Le
  prochain levier est la recette 27B en `NS=3` (gain servi potentiel ~1,5×,
  effort faible), puis `hc_read` (1,7× vers son plancher) ; la vision reste
  une décision de contrat.
- **R46** — *règle (W38)* : le gain de `--tail-anchor off` vit dans
  `(kShortTailMin, I)` quand le dernier chunk est partiel (1,39× à 972,
  1,16–1,18× à 1 536, 1,16× à 2 047) et **disparaît à exactement I**
  (1,01–1,02×) : un intervalle plein n'a pas de queue partielle à ancrer.
  Ne pas attendre de gain au-delà de I, ne pas conclure à une régression.

## W38b — 27B `NS=3` : bande KV corrigée, placement bloquant (2026-11-18)

- **Mesuré d'abord** (`build-w38`, prompt 2 047 tokens, 64 générés, serveur
  neuf par configuration) : KV i8 `NS=0` sert à **15,4 tok/s** ; KV i8 `NS=3`
  refuse (rows `attn_paged`/`kv_store` du drafter h128 non servies en int8) ;
  KV bf16 refuse aussi (le tronc h256 n'a pas de row bf16). Un dtype global ne
  peut pas servir les deux groupes : leurs têtes diffèrent (256 vs 128).
- **Livré** (`ee291d2`) : le groupe KV du drafter prend sa propre dtype ;
  `qwen35_fp8` le garde en **bf16** quand le tronc est en i8 (note au
  démarrage), contrat porté par `arch_test`. `build-w39` : suite complète verte
  (sched 548 964, e2e 1 087/3 skips, fixture 109 671, config 2 927, device
  verts, selftest 195/0/16), kbench ciblé 27B **854/854**, 0 divergence.
- **Blocage suivant** : `dflash_conv` n'a **pas de row device** dans `libr3d`,
  donc le groupe de poids du drafter est épinglé sur le *mapped tier*, et ses
  normes (déclarées f32, stockées bf16) exigent un élargissement au chargement
  → refus « widened at load ». `--placement all_vram` ne change rien.
- **R47** — *constat (W38)* : le 27B en `NS=3` n'est pas une recette ; il faut
  une row device `dflash_conv`. Le prérequis KV est livré, le reste est du
  noyau (hors périmètre sans feu vert). Le 27B reste servi à `NS=0`
  (**15,4 tok/s** à 2 047 tokens). **Clos en W38c** (row livrée).
- **Contrôles du changement** : texte 27B `NS=0` identique entre `build-w38` et
  `build-w39` ; sweep 27B entier sur `build-w39` **3 181/3 181**, 0 échec,
  0 divergence, même pire `rel_l2` ; ciblé 854/854.
- **Vision re-mesurée** (`build-w38`, sans `profile-ops`) : 1 024² →
  **TTFT 3,28 s** (W32 : 4,05 s), 2 048² → **16,33 s** (W32 : 16,97 s) ; le
  gain du petit cas vient de W33–W34 (trellis, stager, ancre de queue).
- **Flash-Next décodage re-mesuré** (`build-w38`, 3 252 tokens, 64 générés) :
  `NS=0` 11,03 tok/s, `NS=3` **19,29 tok/s** (1,75×), texte identique ; le
  préfill paie **+9,7 %** (5 473 → 6 003 ms) pour la passe de contexte du
  drafter.

## W38c — `dflash_conv` sur la carte, 27B en `NS=3` (2026-11-18)

- **Livré** (`f63dffd`) : `libr3d/r3dh_dflash_conv_body.h` (calcul sans HIP,
  transcrit de `ref_dflash_conv`, taps croissants, un arrondi bf16) et
  `libr3d/r3d_dflash_conv.hip` (8 canaux par thread, masque de bloc par
  modulo donc tout `block_size >= 1`, `delta` au pas de la projection
  `2*taps*NG` sinon refus, alias `x`/`out` refusé). Row `taps 2`,
  `group_size 16`, `hidden_size % 128`, bf16.
- **Tests** : `dflash_conv_device_test` **37 checks / 3 cas**, bit-exact sur
  8 géométries (bande 27B, les deux moitiés de la projection, bloc 4 sur
  `T = 10`, bloc 3, bloc 1, bloc 32, `T = 1`, `H = 128`) et les deux refus ;
  selftest **196 rows, 0 échec, 16 skips** ; suite complète verte `build-w39`.
- **27B servi**, 64 tokens générés, deux serveurs par point : à 2 047 tokens
  `NS=0` 15,35 / `NS=3` 15,77 (**1,03×**, acceptation 70 %, préfill +1,7 %) ;
  à 135 tokens 27,8 → 34,2 (**1,23×**, 64 %). Textes identiques (4 paires).
  `NS=2` refusé par l'arch (bloc entraîné = 4).
- **R47 clos** : la row lève le blocage de placement.
- **R48** — *mesure (W38)* : `NS=3` sur le 27B vaut **1,23× à 135 tokens,
  1,03× à 2 047**. L'arithmétique des pas (64 tokens à 70 % d'acceptation
  ≈ 21 verify + 21 drafter en 4,04 s, contre 65 ms/token à `NS=0`) situe le
  verify à 4 rangées vers 2,3× un pas à une rangée et le drafter vers 45 ms :
  le « 1,5× » de W37 supposait un verify quasi gratuit, faux sur un tronc
  dense compute-bound. Contexte long → `NS=0`, contexte court → `NS=3` (texte
  glouton inchangé). **Caduc en W40d** : la cause était l'attention du drafter
  privée de split-KV (aréna jamais allouée) ; à 2 047, `NS=3` vaut **1,74×**.
- **R49** — *constat (W40d)* : le tronc **h128 causal** de MiniCPM tournait à
  `splits=1` depuis toujours (même aréna à 0) : **2,48 tok/s à 969 tokens**
  contre 28,15 quand le split est actif, et 22,18 contre 63,01 au prompt de 59.
  Le « 40,2 tok/s » d'`AGENTS.md` n'est pas reproduit au point documenté. Avec
  la garde `causal == 0`, les textes restent identiques au pristine et le
  tronc reste lent : **décision produit** — accepter un re-baseline du texte
  (tolérance 1 bf16 ULP) pour 10–11×, ou traiter la lenteur autrement.
- **Couverture** : kbench laisse les 112 cas `dflash_conv` du 27B en
  « skipped » (la déclaration porte la plane entière, le site d'issue la
  tranche par `wrow_slice`), mais le sweep reste 3 181/3 181, 0 échec,
  0 divergence ; la row est couverte par `dflash_conv_device_test`.

## W41a — décodage 27B : 4 chaînes i32 et grille split-K (2026-11-19)

- **Profil** : ttn_paged **34,4 ms (50,5 %)**, GEMM IQR **27,5 ms (40,4 %)**,
  reste 9 %. Le GEMM lit 15,25 Gio à **~555 Go/s** (plafond carte 726–734,
  motif de tuile 611–676) ; le trellis n'est que 4–8 % du temps.
- **Livré** (651d0bfc) : 4 chaînes i32 à M ∈ {2,4} et grille split-K
  tranche-major, bit-exactes. Servi **+0,7 à +2,3 %** ; kbench gemm_nt_q
  1,044× à M=1 et 1,058× à M=2/4 (5 120×6 144 M=1 1,19×) ; 27B 3 181/3 181.
- **R50** — *constat* : le pas 27B est à moitié de l'attention, dont le split
  est interdit par le contrat de texte ; le GEMM est à 76 % du plafond. Sous
  contrat bit-exact, plus de levier > 1,06× sur ce GEMM. Ne pas rouvrir sans
  accepter un changement d'arithmétique.
