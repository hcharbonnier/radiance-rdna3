# Checklist de clôture — RDNA3 / Qwen3.8 Flash-Next IQR-V4

État au 2026-11-16, branche `rdna3-iqr`, état code `de81ffe` : **R21** (tier
KV actif), **R22** (snapshots courts), **R23** (tables MoE dans kbench),
**R25** (MTP servi) et **R26** (vision servie) livrés. Référence : **164
rows**, selftest 0 échec, `sched_test` 548 782 checks, `engine_e2e_test` 909
checks, kbench Flash-Next **968/968** et 27B **683/683** dans les deux modes,
0 divergence. **Périmètre révisé le 2026-11-16** : la vision et le MTP de
Flash-Next entrent dans la cible (ADR-7, qui supersède l'ADR-6). Sert à suivre
l'avancement ; cocher au fur et à mesure
et dater la ligne. Le détail des décisions et mesures est dans
`TABLEAU_REEL_IDEAL_RDNA3.md`, `JOURNAL_RDNA3.md`, `SUIVI_LOTS_RDNA3.md`,
`HANDOVER_RDNA3_PHASE7.md`, `FLASH_NEXT_AUDIT.md` et `docs/adr/`.

Légende : `[x]` fait, `[ ]` à faire, `[~]` en cours, `[!]` bloqué.

## Reste à faire — synthèse (2026-11-16)

Ordre recommandé. Les fiches de lot sont dans `SUIVI_LOTS_RDNA3.md`
(« Prochains lots »), les chiffres dans `TABLEAU_REEL_IDEAL_RDNA3.md`.

**P0 — performance Flash-Next (M2 non tenu : η lien 1,7 % de 2 545 tok/s)**
- [ ] **R20 `hc_read`** — 1,77 ms à M=1 pour 6,55 Mo de poids (η 0,6 % ; la row
      documente 18,7 µs) et 0,23 ms par ligne au-delà de M=8 : ~40 % du décodage
      et 47 % du préfill. Action suivante : `--profile-ops true` sur Flash-Next
      (temps device par op) pour trancher lecture/décodage/latence, puis
      corriger. **Feu vert requis** : optimisation noyau en pause.
- [ ] **W24a `qsa_score`** — 83,4 ms à M=512 (103 GFLOP/s, 2 % du pic), ~4 s par
      préfill 2 048 (item 3.8).

**P1 — cache multi-tour, périmètre servi, couverture MoE**
- [x] **R22 snapshots multi-agents** — cause : sous `--prefix-resume exact`
      un prompt plus court qu'un intervalle n'avait ni point de grille ni
      snapshot, donc le tour suivant ne reprenait aucun état linéaire
      (`no_snapshot`). Correctif : ancre de queue au dernier frontière de
      quantum (un quantum plus bas hors grille), dès que l'ancre atteint 256
      tokens. Rejeu de 9 tours : tokens servis 24 % → **52,6 %**, préfill
      recalculé −37 %, `no_snapshot` 6 → **2** (premiers tours de
      sous-agents), checkpoints écrits 1 → **7** ; `sched_test` 548 782
      checks, `engine_e2e_test` et tests device verts. Compromis documenté :
      sous le premier intervalle, la reprise est hors grille (comme
      `nearest`), donc l'accord bit à bit n'est plus garanti pour un tour
      appendu court, seulement pour un prompt répété.
- [x] **R23 tables MoE** — cause double : la row IQR ne voit qu'une vue de
      3 des 5 plans (le reader refusait la sélection) et `kf_lay_table`
      empilait les entrées au lieu de la table de pointeurs du moteur.
      Correctif : décodage de la vue complète (sans `fwht128`), échelle
      IQR produit ligne×bloc, `TablePointers` (slab + pointeurs hôte/device)
      pour conteneur et replay, rotation froide et marges ; refus au layout
      des formes IQR non servies. kbench passe de 856 à **952/952** puis
      **968/968** après fusion W25, 0 échec, 0 divergence ; temps par row et
      η mesurés (IQR 70,5 % à M=32, w4nl 51 % à M=16 ; M=1 borné par le
      décodage treillis, pas par la bande).
- [x] **R25 MTP Flash-Next (ADR-7)** — rows device `mtp_enter_e4m3`,
      `logit_rerank_iqr` et `logit_rerank_i8` (contrats `libr4d`), tests
      device (22 + 36 checks), selftest **164 rows**. Deux bugs préexistants
      de `gemm_nt_q_w2a8_m16` corrigés (en-tête du poids joint lu sur l'hôte
      alors qu'il vit en VRAM ; chunk de K incompatible à M=1/K=2560). Service
      à `NS=3` : décodage **4,51 → 8,21 tok/s (1,82×)**, texte identique,
      acceptation 40,3 % (66/61/37 % aux profondeurs 1–3). Les 14 cas
      `mtp.draft_head` sont couverts et les tables MTP 48 le sont par R23
      (968/968 au total).
- [ ] 2.11/2.12 — formes mixtes par couche et rows `mod4` (seulement si `tp=4`).

**P2 — couverture, doc, clôture**
- [ ] 5.5 R9 — modèle non-IQR absent de l'hôte (dérogation écrite, à reprendre
      si un conteneur 2B/4B arrive).
- [ ] 5.6 — décider l'ajout de `RAPPORT_RDNA3.md` au dépôt (à jour, hors dépôt).
- [x] 1.5 / 5.7 / 8.6 — `docs/OPS.md` (rows, dont les cinq ops QSA) et
      `libr3d/README.md` (inventaire 164 rows / 101 schémas, limites connues) ;
      2.9 la row MoE est déjà documentée dans `docs/OPS.md` (§`moe_gemm_q`,
      w4nl) ; 3.9 les cinq ops QSA sont documentées (W24).
- [ ] `qsa_tail_store` / `qsa_work` — refuser explicitement un index qui n'est
      pas i32 (le shim caste sans vérifier) et ajouter le test de refus ; écart
      relevé pendant W24, sans impact sur le moteur qui alloue i32.
- [x] 8.7 — handover final de phase (`HANDOVER_RDNA3_PHASE7.md`, état `de81ffe`).
- [ ] 6.1–6.6 — optimisations différées (non bloquantes).
- [ ] **Écart RDNA4 → RDNA3** — **238 rows / 179 ops** absents de `libr3d`
      (toutes architectures) ; la cible v1 n'en manque aucun sur ses chemins
      exercés. Périmètre à décider, feuille de route **§D** et liste complète
      `RAPPORT_RDNA3.md` §10.
- Hors v1 : codebook `w4nl` dans `w4a8` seulement (MTP et vision sont revenus
  dans la cible par l'ADR-7).

## 0. Déjà clos (pour situer le reste)

- [x] W0 / M0 — spike IQR 27B, parité, GO (`MEMO_M0_RDNA3.md`)
- [x] W1/W2 (partiel) — hooks hôte, tag de layout, décodeur `kTrellisMad` L=16
- [x] W3 — GEMV M≤16, split-K `down` (662 GB/s, 76,6 %)
- [x] W4 — tuile préfill résidente (301 tok/s à 819) ; WMMA construite, éteinte
- [x] W6 — attention h256/GQA/KV i8, 15/15 identique à libref
- [x] W8 / W9 — HC et PLE, device verts (perf non mesurée)
- [x] W10 — tour vision, 142 rows vertes, `attn_dense` −49 % à M=512
- [x] W11 (dense) — 27B servi, suite verte, démo texte
- [x] W12 / W13 — optimisations, **144 rows / 0 échec / 14 skips**, device 6/27
- [x] W14 — audit Flash-Next (`FLASH_NEXT_AUDIT.md`), 4 ops QSA device (tests
      libref verts), row MoE IQR treillis, référence hôte MoE, formes denses
      table/grille ; **151 rows / 0 échec / 14 skips**, puis `moe_iqr_test`
      réconcilié (10 formes, 0 écart)
- [x] W15 — `kSigned` 8, `gdn_conv_recurrent_update_had`, `kv_store` block 4,
      `rope` imrope h128, router fusionné, `gemm_nt` N=1, `gemm_nt_bias`,
      `grid_embed`, `moe_gemm_w4nl` ; **159 rows / 0 échec / 15 skips**, tous
      les tests device W15 verts, `moe_iqr_test` et `moe_w4nl` verts.
- [x] M1 — 27B texte : décodage 28,4 tok/s (16 tok), 21,2 (819), préfill 301
- [x] M7 (dense) — tableau rempli pour les rows denses
- [x] ADR-1 à ADR-7 — écrits dans `docs/adr/` (ADR-7 du 2026-11-16 : vision et
      MTP de Flash-Next dans la cible ; supersède l'ADR-6 et complète l'ADR-5)
- [x] W16/W17 — Flash-Next servi par `libr3d` seul ; tête `logits_gemm_bf16`
      (échelle 1×128) et formes QSA ; kbench 856/856, 0 divergence
- [x] W18 — mesures à froid HC/PLE/QSA par row ; η lien calculé (1,7 %)
- [x] W19 — démo M6 (prompts, 1/4/6 séquences, cache, contention, 16k),
      robustesse étendue, rejeu 27B et R3 (chargement 2,7 s)
- [x] W20 — contrat capacité/étendue : R24 corrigé, même faute corrigée dans
      3 rows router, mode `kbench --band-capacity`, test device épinglé
- [x] W22 (R21) — rows u8 `gather_rows`/`scatter_rows` pour les fragments KV
      opaques (4 128 octets) ; tier KV hôte actif sur Flash-Next (2 groupes,
      51,38 Kio/entrée) et 27B (516 Kio/entrée), copie et restauration
      vérifiées par hash (`RADIANCE_DEBUG_TIER_VERIFY=1`), 0 échec ; test
      `gather_rows_u8_device_test` ; 161 rows au selftest
- [x] W26 (R26) — vision servie : build `-DRAD_WITH_FFMPEG=ON` (libavformat
      62.3, ffmpeg 8.0), tour 856,68 Mio en VRAM, `capabilities.image=true`,
      tour image 512² et 1024² accepté et répondu ; 856/856 kbench dans les
      deux modes et 480/480 cas vision, 0 divergence
- [x] W25 (R25) — MTP de Flash-Next : `mtp_enter_e4m3` et
      `logit_rerank_iqr`/`_i8` en device, `gemm_nt_q_w2a8_m16` corrigé pour
      un poids joint en VRAM et pour M=1/K=2560 ; **164 rows** au selftest,
      **968/968** kbench Flash-Next (après R23) et 683/683 27B dans les deux
      modes ; NS=3 servi, **1,82×** le décodage à texte identique
- [x] W23 (R23) — tables MoE posées dans kbench : vue complète des 5 plans
      (sans `fwht128`), échelle IQR ligne×bloc, `TablePointers`
      (slab + pointeurs hôte/device) pour conteneur et replay, timer et
      marges ; formes IQR non servies refusées au layout. 968/968, 0 échec,
      0 divergence ; η mesuré (IQR 70,5 % à M=32, w4nl 51 % à M=16)
- [x] M3 (QSA) — équivalent au dense, 4 ops device vertes
- [x] M4 — coûts HC/PLE/QSA documentés et MoE par row mesuré (R23 : η 70,5 %
      IQR à M=32, 51 % w4nl à M=16 ; M=1 borné par le treillis)
- [x] M5 / M6 — Flash-Next texte 1/4/6 séquences, 8k et 16k, démo,
      robustesse et vision ; R21 et R22 résolus
- [!] M2 — η lien **1,7 %** contre 80 % : cause mesurée (`hc_read`), lot W21

## 1. W1/W2 — formes IQR restantes (prérequis MoE)

- [x] 1.1 Auditer les encodages réels du conteneur Flash-Next — fait dans
      `FLASH_NEXT_AUDIT.md` (recette mixte, 591 lignes, commandes et incertitudes).
- [x] 1.2 Formes à ajouter identifiées : **`kSigned` 8** (i8 tourné, tronc),
      **`w4nl`** (table E4M3/64 + f32, tous les `down`, MTP), treillis
      `kTrellisMad` u3/u4/u5 (gate_up, déjà servi), bf16 protégé (déjà servi).
- [x] 1.3 `form_supported` et noyaux denses : `kTable`/`kGrid` ajoutés et
      testés (W14), **`kSigned` 8 livré (W15)**, `w4nl` livré (W15, 2.10).
- [x] 1.4 Tests device des formes : table/grille, `kSigned` et `w4nl` couverts
      par leurs lots (W14/W15).
- [ ] 1.5 Mettre à jour `docs/OPS.md`, `libr3d/README.md` et le tableau des
      tailles de plans pour les formes ajoutées.
- [x] 1.6 **`kSigned` 8** : `RAD_I8 → kSigned, B=8`, `X(kSigned,8,0,0,0,kBf16)`
      dans `R3D_IQR_FORMS`, acceptation dans les trois familles, décodage i8
      direct, bandes mixtes `kSigned`+`kTrellisMad`. Agent W15 dédié.
- [!] 1.7 Vérifier l'agrégation par bande : une row qui accepte les deux formes
      couvre `N=512 K=2560` et `N=2560 K=6144` (MTP 48). **Requis depuis
      l'ADR-7** : MTP 48 est servi, donc la bande existe dans le run.

## 2. W5 — MoE IQR / jalon M2 (~4 semaines, chemin critique)

- [x] 2.1 **Référence hôte** de `moe_gemm_iqr` (R4) : `tests/iqr_ref.h` +
      `tests/moe_iqr_test.cpp` écrits et **réconciliés en W14** (10 formes,
      0 écart, tolérance 2 ulp).
- [x] 2.2 Row `moe_gemm_iqr` déclarée (`libr3d/r3d_moe_iqr.hip`), schéma
      `moe_gemm_q`, classes paires/impaires, scratch.
- [x] 2.3 Run map : tri des lignes, portée par expert, bande ≤ 16 lignes et
      préfill (`moe_iqr_device_test` 15/15 vert).
- [x] 2.4 Noyau par expert, tuile de 16 lignes, résidence en préfill.
- [~] 2.5 Mélange de formes : treillis et **`w4nl` livrés (W15)** ; restent les
      formes mixtes dans une même couche (2.11) et la couverture des experts
      bf16 protégés.
- [x] 2.6 Échelles par expert/ligne/bloc, lignes abandonnées, experts à zéro
      ligne, top-k variable (couvert par `moe_iqr_device_test`).
- [~] 2.7 Tests : synthétiques et référence hôte **réconciliés (W14/W15)** ;
      la couche réelle du conteneur reste à faire, bloquée par le layout des
      tables MoE dans kbench (R23).
- [~] 2.8 Mesure M2 : décodage 1 et 4 séquences, préfill 2048/4096, **η lien
      ≥ 80 %**. Fait à froid le 2026-11-16 : η lien = **1,7 %** (42 tok/s), le
      plafond étant `hc_read` (89 tok/s à lui seul), pas le MoE ; le temps par
      row MoE n'est **pas mesurable** sur ce conteneur (`kbench` refuse le format
      des tables, 110 cas sautés) — voir `TABLEAU_REEL_IDEAL_RDNA3.md`.
- [ ] 2.9 Doc de la row MoE et des limites connues.
- [x] 2.10 **`moe_gemm_w4nl`** : forme `kW4NL` (table f32[16], E4M3/64 + f32,
      `group=64`), row device, `a_order` token/sorted, N=1280/2560,
      K=2560/640, MTP 48. **Requis pour Flash-Next** (25 590 poids `down`).
- [ ] 2.11 MoE : formes mixtes dans une couche (treillis gate_up + w4nl down)
      et experts bf16 protégés via `moe_gemm_bf16`.
- [ ] 2.12 `moe_gemm_iqr_mod4`/`moe_gemm_w4nl_mod4` seulement si `tp=4`.

## 3. W7 — QSA / jalon M3 (livré en W14, tests verts)

- [x] 3.1 `qsa_tail_store`
- [x] 3.2 `qsa_work` (7 cas libref, 529 checks)
- [x] 3.3 `qsa_block_key` (16 cas, 5504 valeurs identiques)
- [x] 3.4 `qsa_score` (781 checks)
- [x] 3.5 `qsa_select` (14 cas, 781 checks)
- [x] 3.6 Sélection identique à la référence, égalité dense sous le seuil
- [x] 3.7 Cas : plusieurs séquences, blocs incomplets, topk large, signed
- [ ] 3.8 Mesure : temps de l'indexeur par pas et surcoût de lancement
- [ ] 3.9 Doc des cinq ops QSA et de leurs refus

## 4. Rows et formes découvertes par l'audit (W15)

- [x] 4.1 `gdn_conv_recurrent_update_had` — bloque le déclaratif Flash-Next
      (`libr3d` seul) ; agent W15.
- [x] 4.2 `kv_store` i8 **block 4** (head_dim 256, n_head_kv 2, M=4096) ;
      agent W15.
- [x] 4.3 `rope` **imrope h128** (sections 11/11/10, rotary 64) ; agent W15.
- [x] 4.4 `router_gemm_topk_scatter` device (M≤64, E=512, top_k=10, protect) ;
      agent W15 (fusion ou paire selon ce que le plugin demande).
- [x] 4.5 `gemm_nt` **N=1** device (shared gate, K=2560, M=2048) ; agent W15.
- [x] 4.6 `gemm_nt_bias` et `grid_embed` vision device ; agent W15.
- [x] 4.7 `kSigned` 8 (voir 1.6 et 1.7).
- [x] 4.8 `moe_gemm_w4nl` (voir 2.10).

## 4bis. M5/M6 — intégration Flash-Next (~1,5 semaine)

- [x] 4bis.1 Audit du conteneur 121 GB (`FLASH_NEXT_AUDIT.md`), graphe obtenu
      avec deux contournements de debug.
- [x] 4bis.2 Déclaratif avec `libr3d` seul : **0 bande device non résolue**
      sur 5 074 questions ; le serveur démarre (44,35 GiB/carte, 21 s) avec
      `/tmp/fn-v1.yaml`. Toutes les rows W15 sont résolues.
- [x] 4bis.2b Premier pas réel : le refus `hc_read_e4m3 M=5` a été corrigé
      (numel remplacé par la forme logique), ainsi que `moe_gather`,
      `moe_iqr`, le router et la tête vocabulaire (plan du déclaratif au lieu
      d'un peek VRAM). Flash-Next sert le texte.
- [x] 4bis.2c `kbench` de contrôle après W17 : **856/856 cas, 0 échec,
      0 divergence** (pire rel_l2 1,2e-2). Tête bf16 corrigée (échelle 1×128,
      rel_l2 0,53 → 1,7e-3) et trois formes QSA alignées. Les 78 cas non
      couverts étaient les tables MTP 48 et `ngram_hint` : **MTP 48 est
      exigible depuis l'ADR-7** (R25), `ngram_hint` reste sans oracle et n'est
      pas une régression. Depuis R23/W25, le total est **968/968**
      (Flash-Next) et **683/683** (27B), 0 échec, 0 divergence ; restent 10
      `gemm_nt_q_gated` i8a8h sans référence libref et 8 `ngram_hint`.
- [x] 4bis.3 Configuration v1 : `placement: expert_tiered`, `host-pool-mib:
      43000`, KV i8, **`max_mixed_tok ≤ 2048`** (le plan `score` fait 512 MiB
      à 2048 lignes, 1 Gio à 4096 et le plugin refuse), `max-model-len: 8192`.
      Vérifiée en W19 (serveur démarré avec ces réglages) ; le tier KV hôte est
      actif depuis R21, avec les deux groupes paginés de Flash-Next. Depuis
      R25, la configuration servie est **`num-speculative-tokens: 3`** (MTP
      actif, 1,82× le décodage) ; 0 reste le repli pour comparer ou si un
      drafter n'est pas servi. Depuis
      R26, le build servi exige `-DRAD_WITH_FFMPEG=ON`, `--mm-max-patches`
      reste `auto` (16384 patches, 2048²) et la tour tient en VRAM
      (856,68 Mio) dans le budget `host-pool-mib: 43000` / marge 96 Mio.
- [x] 4bis.4 Texte : 1 séquence (30 / 2290 / 4570 tokens) et 4 séquences
      concurrentes servies ; préfill 42 tok/s, décodage 4,0–4,6 tok/s.
- [x] 4bis.5 Image / vision : **livrée** (R26, ADR-7). Build
      `-DRAD_WITH_FFMPEG=ON` ; `capabilities.image=true` ; image 512² →
      256 tokens (281 avec le texte), TTFT 8,19 s ; image 1024² →
      1024 tokens (1045), TTFT 28,73 s ; réponses correctes (couleurs et
      disposition, dégradé). La tour (856,68 Mio) est en VRAM, la vidéo est
      annoncée mais non testée. **Robustesse** : 18 cas, 2048² (4 098
      tokens, 121,5 s), 3000² redimensionnée, 2 × 1024², alpha/JPEG, 4
      erreurs 400 explicites, VRAM constante, aucune fuite ; limites
      consignées dans `TABLEAU_REEL_IDEAL_RDNA3.md` § Vision.
- [x] 4bis.6 Robustesse : prompt de 12 002 tokens refusé proprement (HTTP 400,
      message explicite avec le nombre de tokens à couper) et serveur toujours
      vivant ; `max_tokens` hors contexte plafonné à 8 191 (pas de 400) et
      `finish=abort` à la déconnexion, slot rendu ; abandon à 3 s puis requête
      suivante en 3,57 s ; **6 admissions simultanées 6/6** ; redémarrages
      propres ; `/health` 200 et 0 requête en vol à la fin.
- [x] 4bis.7 Aucun nouvel avertissement au démarrage ; les avertissements
      présents (PCIe sous max, swap, governor) sont ceux de l'hôte.
- [x] 4bis.8 Démo reproductible (script, prompts, lots) et documentation :
      `m6demo.py`/`m6tail.py`/`m6prefix.py`/`m6grow.py` + `agentreplay.py` +
      `fnconc.sh` ; résultats dans `TABLEAU_REEL_IDEAL_RDNA3.md` §M6. Limites
      documentées : tier KV hôte (résolu depuis R21) et 0 % de cache en rejeu
      multi-agents (`no_snapshot`).
- [x] 4bis.9 Mise à jour de `SUIVI_LOTS_RDNA3.md` (W18/W19, M2→M6, risques
      R19–R23) et du journal.

## 5. Mesures et rapport final (M4/M7)

- [x] 5.1 HC : coût de `hc_read`/`hc_write`/`hc_enter`/`hc_project` mesuré à
      froid : `hc_read` 1,77 ms à M=1, 470 ms à M=2 048 ;
      `hc_enter`/`hc_write` au plancher ; `hc_project` sans bande dans ce
      conteneur).
- [x] 5.2 PLE : `ple_gate` 23,1 µs à M=1 / 69 TFLOP/s à M=2 048 ; `ple_conv`
      5,8 µs à M=1 / 160 GB/s à M=2 048 ; `ngram_hint` reste sans oracle.
- [x] 5.3 Lignes MoE, QSA, HC, PLE, Flash-Next remplies à froid. Le MoE par row
      reste bloqué par le format des tables du conteneur (110 cas sautés par
      kbench, « container format error ») et η lien est calculé de bout en bout
      (1,7 %).
- [x] 5.4 R3 — mesurer le temps de relayout/chargement du 27B : **15,87 GiB en
      2,7 s** (6,2 GB/s, 1,7 s de disque), prêt en 3,3 s ; pas de vectorisation
      nécessaire. Rejeu : préfill 318,8 tok/s à 844, décodage 23,2/29,4 tok/s.
- [x] 5.8 R24 — 27B multi-séquence : **corrigé**. Le paramètre rangé est une
      capacité de bande, pas le nombre de lignes ; `logits_gemm_vocab_iqr` lit
      l'étendue sur `x` (comme `libr4d`), et l'audit `rad-kbench --band-capacity`
      a trouvé la même faute dans `router_topk`, `router_topk_scatter` et
      `router_sigmoid_topk` (corrigés aussi). Preuve : test device
      `iqr_vocab_head_accepts_a_narrow_step_in_a_wide_band` (échec sur l'ancienne
      `.so`, vert sur la nouvelle), kbench 27B 605/605 et Flash-Next 856/856 dans
      les deux modes (par défaut et `--band-capacity`), 27B `SEQS=4` à 67,3 tok/s
      agrégés.
- [~] 5.5 R9 — valider E4M3/rotated sur un modèle non-IQR : **inexécutable sur
      l'hôte** (inventaire vérifié : seuls le 27B IQR et Flash-Next IQR).
      Dérogation documentée dans `SUIVI_LOTS_RDNA3.md` §R9 ; à reprendre dès
      qu'un conteneur non-IQR 2B/4B est présent.
- [~] 5.6 `RAPPORT_RDNA3.md` **mis à jour** (encadré, §7.3 corrigée, §7.5
      mesures réelles, §9 révisée) ; reste la **décision d'ajout au dépôt**, qui
      appartient à l'utilisateur.
- [ ] 5.7 Mettre à jour `docs/OPS.md`, `libr3d/README.md` et les limites connues
      (dont le contrat capacité/étendue, aujourd'hui dans `AGENTS.md`).

## 6. Optimisations différées (non bloquantes pour M5/M6)

- [ ] 6.1 WMMA préfill : reprendre l'émission/staging, rallumer seulement si elle
      bat le `v_dot4` (actuellement 25–32 % plus lente, éteinte).
- [ ] 6.2 Gated fusionné : occupation 1,4 bloc/CU (kernel persistant ou 256 threads).
- [ ] 6.3 `attn_dense` hd72 : mesurer M=16384 (kbench s'arrête à 512).
- [ ] 6.4 `attn_paged` h256 : mesure isolée ; split du préfill interdit par
      l'ordre exact exigé.
- [ ] 6.5 Déduplication `dot4_i8`/`i8_codes4`/`wave_max_f` (risque de dérive).
- [ ] 6.6 Vérifier la reproductibilité des builds (builds neufs pour tout A/B).

## 7. Périmètre v1 — points tranchés par les ADR

- [x] 7.1 MTP / spéculation — **réintégré dans la cible le 2026-11-16**
      (ADR-7, qui supersède l'ADR-6) ; le travail est suivi en R25 et le repli
      `num-speculative-tokens: 0` reste valable jusqu'à livraison des rows.
- [x] 7.2 `w4nl` **n'est plus hors périmètre** : l'audit montre 25 590 poids
      experts `down` en `w4nl` ; la row est requise pour servir Flash-Next par
      `libr3d` (voir 2.10). Seul le codebook `w4nl` dans `w4a8` reste optionnel.

## 8. Critères de clôture du projet

- [x] 8.1 `r3d_selftest` **164/0/15**, `iqr_device_test`, `logits_bf16_device_test`,
      `moe_iqr_test`, `moe_w4nl`, `gather_rows_u8`, `mtp_enter_e4m3`,
      `logit_rerank_iqr` et les quatre tests QSA verts ; kbench Flash-Next
      **968/968** et 27B **683/683**, 0 échec, 0 divergence.
- [x] 8.2 Flash-Next démarre et génère (texte, 1 et 4 séquences, préfills
      2048/4096) sans fallback ni avertissement nouveau.
- [~] 8.3 MoE conforme (treillis + `w4nl` + bf16 protégé, écart borné documenté)
      ; le **η lien ≥ 80 %** est mesuré et **non tenu** (1,7 % global) — cause
      identifiée (`hc_read`, 47 % du préfill), pas le MoE ; le temps par row
      MoE est mesuré par R23 (η 70,5 % IQR à M=32, 51 % w4nl à M=16 ; M=1
      borné par le décodage treillis, 10–12 %).
- [x] 8.4 QSA identique à la référence et égal au dense sous le seuil (781/819
      checks, kbench 0 divergence).
- [~] 8.5 Tableau réel/idéal complet, écarts expliqués, RAPPORT à jour :
      Flash-Next, kbench, HC/PLE, QSA et MoE par row mesurés ; η lien calculé ;
      reste la décision d'ajout du RAPPORT au dépôt (5.6).
- [x] 8.6 Documentation des rows, architectures, limites et perf à jour
      (`docs/OPS.md` : cinq ops QSA, `gather_rows_u8` ; `libr3d/README.md` :
      inventaire 164 rows / 101 schémas, familles, limites connues).
- [x] 8.7 Handover final écrit (`HANDOVER_RDNA3_PHASE7.md`) et journal tenu à
      la date de clôture ; §8 donne la reprise immédiate.
- [x] 8.8 **Vision servie (ADR-7)** : build `RAD_WITH_FFMPEG=ON`,
      `capabilities.image=true`, tours image 512²/1024²/2048² acceptés et
      répondus, 18 cas de robustesse (multi-images, formats, erreurs 400),
      limite retirée de `TABLEAU_REEL_IDEAL_RDNA3.md` (W26).
- [x] 8.9 **MTP Flash-Next (ADR-7)** : `mtp_enter` et `logit_rerank` en device,
      run à `NS=3` et gain mesuré (4,51 → 8,21 tok/s, 1,82×, texte identique,
      40,3 % d'acceptation) ; les 14 cas `mtp.draft_head` et les tables MTP 48
      sont couverts par kbench (968/968, 0 échec, 0 divergence).

## 9. Ordre de marche suggéré (parallélisable)

1. **W14 (fait)** : audit, 4 ops QSA, MoE treillis + référence, formes denses
   table/grille ; **151 rows** puis **158 rows** après W15.
2. **W15 (fait sauf w4nl)** : `kSigned` 8, `gdn_conv_recurrent_update_had`,
   `kv_store` b4, `rope` h128 imrope, router fusionné, `gemm_nt` N=1,
   rows vision (`gemm_nt_bias`, `grid_embed`), `moe_iqr_test` réconcilié.
3. **W15b** : `moe_gemm_w4nl` + formes mixtes/bf16 protégé (en cours).
4. **Intégration** : démarrage Flash-Next réel, config v1 (`num-spec 0`,
   `max_mixed_tok ≤ 2048`, `expert_tiered`), 1/4 séquences, préfills 2048/4096.
5. **Mesures** : HC/PLE, MoE, QSA, R3, tableau + RAPPORT.

Si la cible se limite au **27B texte**, seuls 5.1–5.7 et 8.1/8.5–8.7 restent ;
le reste concerne Flash-Next.

## D. Écart RDNA4 → RDNA3 : kernels à porter ou recréer (2026-11-17)

Mesure à l'état `de81ffe` : `libr4d/r4d_rows.cpp` (**457 rows, 274 ops**) contre
`libr3d/r3d_rows.cpp` (**164 rows, 101 ops**), soit **238 rows / 179 ops** sans
équivalent RDNA3. La liste complète (op par op, famille par famille) est dans
`RAPPORT_RDNA3.md` §10 ; cette section est la feuille de route.

- **La cible v1 actuelle n'a aucun op manquant** sur ses chemins exercés
  (kbench 968/968, 683/683 ; selftest 164/0/15). Les lignes ci-dessous sont
  conditionnées par le périmètre multi-architectures.
- [ ] **D0 — décider le périmètre** : toutes les architectures `arch/` (238
      rows) ou seulement la cible v1 + l'artefact IQ8R ? Sans cette décision,
      D5–D9 restent hors engagement.
- [ ] **D1 — IQ8R groupe 256 (MiniCPM5-2B, non-IQR)** : `rmsnorm_had_quant_iq8r`,
      `ar_rmsnorm_had_quant_iq8r`, `gate_had_quant_iq8r`, `had_quant_act_iq8r`,
      `gated_had_quant_iq8r`, le producteur non groupé `rmsnorm_had_quant_i8`,
      `rmsnorm_quant_i8g`, et les variantes **groupe 256** de
      `gemm_nt_q`/`gemm_nt_q_gated` en `i8a8h` (`gemm_iq8r_nt_m16`,
      `gemm_iq8r_tiled` ; les rows actuelles n'acceptent que 128). Test : le
      2B charge et sert à `NS=0`. Le `rad_arch_declare` échoue aujourd'hui sur
      `nq_h.declare` (groupe 256 contre 128) — cause exacte dans
      `RAPPORT_RDNA3.md` §10.4.
- [ ] **D2 — DSpark (MiniCPM5 à NS > 0)** : `dflash_conv`,
      `dflash_markov_greedy`, `dflash_markov_greedy_f32`, `dflash_tree` ; plus
      `row_topk_merge`, `sample_merge_topk`, `pair_argmax` si le sélecteur les
      demande. Inutile pour R9 à `NS=0`.
- [ ] **D3 — multi-carte (`tp>1`)** : `all_reduce`, `all_gather` et les fusions
      `ar_*` (**18 rows**, 2 ops). Sans objet sur une carte, requis pour
      `tp=2/4`.
- [ ] **D4 — streaming KV** : `kv_store_streamed` (fp8/i8/bf16) et
      `kv_resolve_i32` ; le tier hôte actuel n'en a pas besoin.
- [ ] **D5 — attention/KV des autres architectures (30 ops, 58 rows)** :
      `attn_cross`, `attn_local_rel`, `attn_na3d`, `attn_paged_gate_{quant,had_quant}`,
      `attn_paged_qrow{,_tree}`, `attn_prefix_self`, `attn_vsa`, `dsa_{pool,query,score}`,
      `embed_attn`, `kv_resolve`, `kv_store_streamed`, `ministral_rope`, `mla_*`,
      `pixtral_rope`, `position_scale`, `qk_norm_rope*`, `qkv_norm_rope*`,
      `rope_rows`, `sam_attn`, `vsa_select`.
- [ ] **D6 — normes et quantisation (21 + 11 ops, 36 rows)** :
      `add_rmsnorm`, `group_norm`, `join_add_scale`, `l2norm_chan`, `ln_mod*`,
      `mhc_*`, `norm_add_norm*`, `pixel_norm`, `ple_conv_bias`, `rms_mod*`,
      `rmsnorm_p`, `rmsnorm_rows`, `gate_{quant_fp8,had_quant_i8g}`,
      `gelu_mul_quant_i8g`, `gram_accum`, `rmsnorm_had_quant_i8`, `rmsnorm_quant_i8g`.
- [ ] **D7 — MoE/router inverses (4 ops)** : `moe_gather_inv`, `moe_scatter_inv`,
      `router_sigmoid_group_topk`, `router_sigmoid_topk_scatter`.
- [ ] **D8 — sampler/sélection (4 ops)** : `sample_chain`, `sample_merge_topk`,
      `cfg_topk_sample`, `pair_argmax`.
- [ ] **D9 — média et génération (hors v1 sauf cible)** : `elem` (**85 ops**,
      dont `gelu_mul`, `gelu_quick`, `gelu_tanh`, `mul_weight`, `scale_rows`,
      `seg_*`, `snake*`, `swiglu_*`, `flow_*`, `conv2d`, `patch_*`, `vtile_*`,
      `tile_*`), `conv` (6), `vision` (`pixtral_merge`), `draft`/`dflash` (8),
      `gemm`/`lora` (4), `gdn` (`kda_update`, `ssd_update`), `misc` (`prefetch`).
- [ ] **D10 — rows RDNA4-only à recréer pour gfx11** : `lora_*_wmma`,
      `mla_attn_rows/matrix_bf16` et variantes `_fp8kv`, `dsa_score_fp8`
      (`dsa_pool`/`dsa_query` à vérifier), `dequant_w4_bf16` (fragments WMMA),
      `moe_gemm_{w4a8,i8a8,w4a8i8}*` et `_mod4` (le RDNA3 a des équivalents
      IQR/`w4nl`, pas ces formats). Écrire
      `v_dot4`/`v_mfma_f32_16x16x16_bf16` ou déclarer la forme non servie.
      Liste non exhaustive : vérifier le fichier `libr4d/r4d_*.hip` de chaque
      row manquante avant de la classer portable ou à recréer.

Ordre recommandé : **D1 → D2** (artefact non-IQR), **D3** si multi-carte,
**D5–D8** si une architecture correspondante entre dans la cible, **D9–D10**
en dernier. Ne pas rouvrir les rows déjà servies (IQR, `w4nl`, paged h128/h256,
QSA, vision Qwen, MTP) : elles ne font pas partie de l'écart.
