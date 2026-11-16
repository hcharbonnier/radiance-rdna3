# Checklist de clôture — RDNA3 / Qwen3.8 Flash-Next IQR-V4

État au 2026-11-16, branche `rdna3-iqr`, pointe `fb07ba3` (W14 fusionné, `moe_iqr_test`
en cours de réconciliation). Sert à suivre l'avancement ; cocher au fur et à
mesure et dater la ligne. Le détail des décisions et mesures est dans
`TABLEAU_REEL_IDEAL_RDNA3.md`, `JOURNAL_RDNA3.md`, `SUIVI_LOTS_RDNA3.md`,
`HANDOVER_RDNA3_PHASE6.md` et `FLASH_NEXT_AUDIT.md`.

Légende : `[x]` fait, `[ ]` à faire, `[~]` en cours, `[!]` bloqué.

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
      table/grille ; **151 rows / 0 échec / 14 skips** (hors `moe_iqr_test`,
      en cours de réconciliation)
- [x] M1 — 27B texte : décodage 28,4 tok/s (16 tok), 21,2 (819), préfill 301
- [x] M7 (dense) — tableau rempli pour les rows denses
- [x] ADR-1 à ADR-6 — écrits dans `docs/adr/`

## 1. W1/W2 — formes IQR restantes (prérequis MoE)

- [x] 1.1 Auditer les encodages réels du conteneur Flash-Next — fait dans
      `FLASH_NEXT_AUDIT.md` (recette mixte, 591 lignes, commandes et incertitudes).
- [x] 1.2 Formes à ajouter identifiées : **`kSigned` 8** (i8 tourné, tronc),
      **`w4nl`** (table E4M3/64 + f32, tous les `down`, MTP), treillis
      `kTrellisMad` u3/u4/u5 (gate_up, déjà servi), bf16 protégé (déjà servi).
- [~] 1.3 `form_supported` et noyaux denses : `kTable`/`kGrid` ajoutés et
      testés (W14), **`kSigned` 8 en cours (W15)**, `w4nl` séparé (2.10).
- [~] 1.4 Tests device des formes : table/grille couverts ; `kSigned` et `w4nl`
      à couvrir avec leurs lots.
- [ ] 1.5 Mettre à jour `docs/OPS.md`, `libr3d/README.md` et le tableau des
      tailles de plans pour les formes ajoutées.
- [ ] 1.6 **`kSigned` 8** : `RAD_I8 → kSigned, B=8`, `X(kSigned,8,0,0,0,kBf16)`
      dans `R3D_IQR_FORMS`, acceptation dans les trois familles, décodage i8
      direct, bandes mixtes `kSigned`+`kTrellisMad`. Agent W15 dédié.
- [ ] 1.7 Vérifier l'agrégation par bande : une row qui accepte les deux formes
      couvre `N=512 K=2560` et `N=2560 K=6144` (MTP 48).

## 2. W5 — MoE IQR / jalon M2 (~4 semaines, chemin critique)

- [~] 2.1 **Référence hôte** de `moe_gemm_iqr` (R4) : `tests/iqr_ref.h` +
      `tests/moe_iqr_test.cpp` écrits ; la comparaison carte échoue encore
      (`RAD_E_UNSUPPORTED`, écarts top-k/parts) — agent d'intégration en cours.
- [x] 2.2 Row `moe_gemm_iqr` déclarée (`libr3d/r3d_moe_iqr.hip`), schéma
      `moe_gemm_q`, classes paires/impaires, scratch.
- [x] 2.3 Run map : tri des lignes, portée par expert, bande ≤ 16 lignes et
      préfill (`moe_iqr_device_test` 15/15 vert).
- [x] 2.4 Noyau par expert, tuile de 16 lignes, résidence en préfill.
- [~] 2.5 Mélange de formes : treillis servi ; **`w4nl` et experts bf16
      protégés restent à faire** (2.10/2.11).
- [x] 2.6 Échelles par expert/ligne/bloc, lignes abandonnées, experts à zéro
      ligne, top-k variable (couvert par `moe_iqr_device_test`).
- [~] 2.7 Tests : synthétiques verts ; référence hôte en cours de
      réconciliation ; couche réelle du conteneur à faire après 2.10.
- [ ] 2.8 Mesure M2 : décodage 1 et 4 séquences, préfill 2048/4096, **η lien
      ≥ 80 %**.
- [ ] 2.9 Doc de la row MoE et des limites connues.
- [ ] 2.10 **`moe_gemm_w4nl`** : forme `kW4NL` (table f32[16], E4M3/64 + f32,
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

- [ ] 4.1 `gdn_conv_recurrent_update_had` — bloque le déclaratif Flash-Next
      (`libr3d` seul) ; agent W15.
- [ ] 4.2 `kv_store` i8 **block 4** (head_dim 256, n_head_kv 2, M=4096) ;
      agent W15.
- [ ] 4.3 `rope` **imrope h128** (sections 11/11/10, rotary 64) ; agent W15.
- [ ] 4.4 `router_gemm_topk_scatter` device (M≤64, E=512, top_k=10, protect) ;
      agent W15 (fusion ou paire selon ce que le plugin demande).
- [ ] 4.5 `gemm_nt` **N=1** device (shared gate, K=2560, M=2048) ; agent W15.
- [ ] 4.6 `gemm_nt_bias` et `grid_embed` vision device ; agent W15.
- [ ] 4.7 `kSigned` 8 (voir 1.6 et 1.7).
- [ ] 4.8 `moe_gemm_w4nl` (voir 2.10).

## 4bis. M5/M6 — intégration Flash-Next (~1,5 semaine)

- [x] 4bis.1 Audit du conteneur 121 GB (`FLASH_NEXT_AUDIT.md`), graphe obtenu
      avec deux contournements de debug.
- [ ] 4bis.2 Démarrage réel avec `libr3d` seul : résoudre
      `gdn_conv_recurrent_update_had`, `moe_gemm_iqr`, `moe_gemm_w4nl`, les
      quatre QSA, `kSigned`, `kv_store` b4, `rope` h128, `router`.
- [ ] 4bis.3 Configuration v1 : `num-speculative-tokens: 0` (MTP hors v1),
      `placement: expert_tiered`, `host-pool-mib: 43000`, KV i8,
      **`max_mixed_tok ≤ 2048`** (le plan `score` fait 512 MiB à 2048 lignes,
      1 Gio à 4096 et le plugin refuse), `max-model-len` ajusté.
- [ ] 4bis.4 Texte : 1 et 4 séquences, préfill 2048 et 4096, sorties conformes.
- [ ] 4bis.5 Image / vision si le conteneur la sert (dépend de 4.6).
- [ ] 4bis.6 Robustesse : formes refusées, erreurs de plan, arrêt propre,
      reprise après redémarrage.
- [ ] 4bis.7 Aucun nouvel avertissement au démarrage ; rapport de startup relu.
- [ ] 4bis.8 Démo reproductible (script, prompts, lots) et documentation.
- [ ] 4bis.9 Mise à jour de `SUIVI_LOTS_RDNA3.md` (M2→M6) et du journal.

## 5. Mesures et rapport final (M4/M7)

- [ ] 5.1 HC : coût de `hc_read`/`hc_write`/`hc_enter`/`hc_project` (M4 exige
      « coûts documentés »).
- [ ] 5.2 PLE : coût de `ple_gate`/`ple_conv` et de l'n-gram.
- [ ] 5.3 Remplir les lignes MoE, QSA, HC, PLE, Flash-Next de
      `TABLEAU_REEL_IDEAL_RDNA3.md`.
- [ ] 5.4 R3 — mesurer le temps de relayout/chargement du 27B ; vectoriser si
      nécessaire.
- [ ] 5.5 R9 — valider E4M3/rotated sur un modèle non-IQR (bloqué : aucun 2B/4B
      sur l'hôte ; documenter la dérogation si le modèle reste absent).
- [ ] 5.6 Mettre à jour `RAPPORT_RDNA3.md` (hors dépôt aujourd'hui) et décider
      de son ajout au dépôt.
- [ ] 5.7 Mettre à jour `docs/OPS.md`, `libr3d/README.md` et les limites connues.

## 6. Optimisations différées (non bloquantes pour M5/M6)

- [ ] 6.1 WMMA préfill : reprendre l'émission/staging, rallumer seulement si elle
      bat le `v_dot4` (actuellement 25–32 % plus lente, éteinte).
- [ ] 6.2 Gated fusionné : occupation 1,4 bloc/CU (kernel persistant ou 256 threads).
- [ ] 6.3 `attn_dense` hd72 : mesurer M=16384 (kbench s'arrête à 512).
- [ ] 6.4 `attn_paged` h256 : mesure isolée ; split du préfill interdit par
      l'ordre exact exigé.
- [ ] 6.5 Déduplication `dot4_i8`/`i8_codes4`/`wave_max_f` (risque de dérive).
- [ ] 6.6 Vérifier la reproductibilité des builds (builds neufs pour tout A/B).

## 7. Hors périmètre v1 (acté par les ADR)

- [ ] 7.1 MTP / `mtp_enter` / `logit_rerank` / spéculation (ADR-6 : hors v1 ;
      le run v1 fixe `num-speculative-tokens: 0`, ce qui retire aussi la couche 48).
- [x] 7.2 `w4nl` **n'est plus hors périmètre** : l'audit montre 25 590 poids
      experts `down` en `w4nl` ; la row est requise pour servir Flash-Next par
      `libr3d` (voir 2.10). Seul le codebook `w4nl` dans `w4a8` reste optionnel.

## 8. Critères de clôture du projet

- [ ] 8.1 `r3d_selftest` **151/0/14** (144 + les 7 rows QSA/MoE), `iqr_device_test`,
      `moe_iqr_test` et les quatre tests QSA verts sur la pointe.
- [ ] 8.2 Flash-Next démarre et génère (texte, 1 et 4 séquences, préfills
      2048/4096) sans fallback ni avertissement nouveau.
- [ ] 8.3 MoE conforme (treillis + `w4nl` + bf16 protégé, écart borné documenté)
      et **η lien ≥ 80 %** en préfill.
- [ ] 8.4 QSA identique à la référence et égal au dense sous le seuil.
- [ ] 8.5 Tableau réel/idéal complet, écarts expliqués, RAPPORT à jour.
- [ ] 8.6 Documentation des rows, architectures, limites et perf à jour.
- [ ] 8.7 Handover final (phase suivante) et journal à la date de clôture.

## 9. Ordre de marche suggéré (parallélisable)

1. **W14 (fait)** : audit, 4 ops QSA, MoE treillis + référence, formes denses
   table/grille, 151 rows vertes hors `moe_iqr_test`.
2. **W15 (en cours)** : `kSigned` 8, `gdn_conv_recurrent_update_had`,
   `kv_store` b4, `rope` h128 imrope, router fusionné, `gemm_nt` N=1,
   rows vision (`gemm_nt_bias`, `grid_embed`), réconciliation `moe_iqr_test`.
3. **W15b** : `moe_gemm_w4nl` + formes mixtes/bf16 protégé (après la
   réconciliation du schéma MoE).
4. **Intégration** : démarrage Flash-Next réel, config v1 (`num-spec 0`,
   `max_mixed_tok ≤ 2048`, `expert_tiered`), 1/4 séquences, préfills 2048/4096.
5. **Mesures** : HC/PLE, MoE, QSA, R3, tableau + RAPPORT.

Si la cible se limite au **27B texte**, seuls 5.1–5.7 et 8.1/8.5–8.7 restent ;
le reste concerne Flash-Next.
