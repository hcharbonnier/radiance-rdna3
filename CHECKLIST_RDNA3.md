# Checklist de clôture — RDNA3 / Qwen3.8 Flash-Next IQR-V4

État au 2026-11-16, branche `rdna3-iqr`, pointe `7338226` (R21 : rows u8
`gather_rows`/`scatter_rows`, tier KV hôte actif ; R26 : vision servie sur un
build FFmpeg ; l'état W20 856/856 et 605/605 reste la référence,
`gather_rows`/`scatter_rows` re-mesurés 48/48 et 84/84 sans divergence).
**Périmètre révisé le 2026-11-16** : la vision et le
MTP de Flash-Next entrent dans la cible (ADR-7, qui supersède l'ADR-6), suivis
en R25 et R26. Sert à suivre l'avancement ; cocher au fur et à mesure
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
- [ ] **R22 snapshots multi-agents** — 0 % de cache servi en rejeu agent
      (`no_snapshot` ×4, 1 checkpoint/5 tours) ; revoir `geometric-backoff` et la
      rétention des snapshots.
- [ ] **R23 tables MoE** — kbench ne pose pas les tables du conteneur (110 cas
      sautés) ; aligner les tags de format ou passer par une fixture
      `--record`, puis remplir le temps par row MoE et η lien par noyau.
- [ ] **R25 MTP Flash-Next (ADR-7)** — écrire `mtp_enter` et `logit_rerank`
      en device dans `libr3d` (mêmes contrats que `libr4d`), puis servir avec
      `num-speculative-tokens > 0`, couvrir les cas MTP 48 de kbench et mesurer
      le gain de décodage. Le 27B reste à 0 (drafter h128 en bf16).
- [ ] 2.11/2.12 — formes mixtes par couche et rows `mod4` (seulement si `tp=4`).

**P2 — couverture, doc, clôture**
- [ ] 5.5 R9 — modèle non-IQR absent de l'hôte (dérogation écrite, à reprendre
      si un conteneur 2B/4B arrive).
- [ ] 5.6 — décider l'ajout de `RAPPORT_RDNA3.md` au dépôt (à jour, hors dépôt).
- [x] 1.5 / 5.7 / 8.6 — `docs/OPS.md` (rows, dont les cinq ops QSA) et
      `libr3d/README.md` (inventaire 161 rows / 99 schémas, limites connues) ;
      2.9 la row MoE est déjà documentée dans `docs/OPS.md` (§`moe_gemm_q`,
      w4nl) ; 3.9 les cinq ops QSA sont documentées (W24).
- [ ] `qsa_tail_store` / `qsa_work` — refuser explicitement un index qui n'est
      pas i32 (le shim caste sans vérifier) et ajouter le test de refus ; écart
      relevé pendant W24, sans impact sur le moteur qui alloue i32.
- [ ] 8.7 — handover final de phase.
- [ ] 6.1–6.6 — optimisations différées (non bloquantes).
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
- [x] M3 (QSA) — équivalent au dense, 4 ops device vertes
- [x] M4 — coûts HC/PLE/QSA documentés (MoE par row bloqué, cf. R23)
- [x] M5 / M6 — Flash-Next texte 1/4/6 séquences, 8k et 16k, démo et
      robustesse ; R21 résolu, R22 encore documentée
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
      pas une régression.
- [x] 4bis.3 Configuration v1 : `placement: expert_tiered`, `host-pool-mib:
      43000`, KV i8, **`max_mixed_tok ≤ 2048`** (le plan `score` fait 512 MiB
      à 2048 lignes, 1 Gio à 4096 et le plugin refuse), `max-model-len: 8192`.
      Vérifiée en W19 (serveur démarré avec ces réglages) ; le tier KV hôte est
      actif depuis R21, avec les deux groupes paginés de Flash-Next. Le
      `num-speculative-tokens: 0` de l'ADR-6 n'est plus la configuration servie
      dès que R25 est livré (ADR-7) ; il reste le repli entre-temps. Depuis
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

- [x] 8.1 `r3d_selftest` **161/0/15**, `iqr_device_test`, `logits_bf16_device_test`,
      `moe_iqr_test`, `moe_w4nl`, `gather_rows_u8` et les quatre tests QSA
      verts ; kbench Flash-Next **856/856, 0 échec, 0 divergence**.
- [x] 8.2 Flash-Next démarre et génère (texte, 1 et 4 séquences, préfills
      2048/4096) sans fallback ni avertissement nouveau.
- [~] 8.3 MoE conforme (treillis + `w4nl` + bf16 protégé, écart borné documenté)
      ; le **η lien ≥ 80 %** est mesuré et **non tenu** (1,7 %) — cause
      identifiée (`hc_read`, 47 % du préfill), pas le MoE ; le temps par row MoE
      reste non mesurable sur ce conteneur (R23, lot W23).
- [x] 8.4 QSA identique à la référence et égal au dense sous le seuil (781/819
      checks, kbench 0 divergence).
- [~] 8.5 Tableau réel/idéal complet, écarts expliqués, RAPPORT à jour :
      Flash-Next, kbench, HC/PLE et QSA par row faits à froid ; η lien calculé ;
      MoE par row bloqué par le format des tables du conteneur ; reste la
      décision d'ajout du RAPPORT au dépôt (5.6).
- [ ] 8.6 Documentation des rows, architectures, limites et perf à jour
      (`docs/OPS.md`, `libr3d/README.md`).
- [ ] 8.7 Handover final (phase suivante) et journal à la date de clôture ; le
      handover courant (`HANDOVER_RDNA3_PHASE7.md` §8) couvre la reprise
      immédiate.
- [x] 8.8 **Vision servie (ADR-7)** : build `RAD_WITH_FFMPEG=ON`,
      `capabilities.image=true`, tours image 512²/1024²/2048² acceptés et
      répondus, 18 cas de robustesse (multi-images, formats, erreurs 400),
      limite retirée de `TABLEAU_REEL_IDEAL_RDNA3.md` (W26).
- [ ] 8.9 **MTP Flash-Next (ADR-7)** : `mtp_enter` et `logit_rerank` en device,
      run avec `num-speculative-tokens > 0`, cas MTP 48 couverts par kbench
      (0 « Not covered » sur le MTP), gain de décodage mesuré et consigné.

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
