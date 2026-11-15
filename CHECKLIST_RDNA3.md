# Checklist de clôture — RDNA3 / Qwen3.8 Flash-Next IQR-V4

État au 2026-11-15, branche `rdna3-iqr`, pointe `2cdff40`. Sert à suivre
l'avancement ; cocher au fur et à mesure et dater la ligne. Le détail des
décisions et mesures est dans `TABLEAU_REEL_IDEAL_RDNA3.md`,
`JOURNAL_RDNA3.md`, `SUIVI_LOTS_RDNA3.md` et `HANDOVER_RDNA3_PHASE6.md`.

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
- [x] M1 — 27B texte : décodage 28,4 tok/s (16 tok), 21,2 (819), préfill 301
- [x] M7 (dense) — tableau rempli pour les rows denses
- [x] ADR-1 à ADR-6 — écrits dans `docs/adr/`

## 1. W1/W2 — formes IQR restantes (prérequis MoE)

- [ ] 1.1 Auditer les encodages réels du conteneur Flash-Next (bandes, classes,
      formes, échelles) et consigner le résultat dans `docs/` ou le journal.
- [ ] 1.2 Confirmer si le conteneur n'utilise que `kTrellisMad` L=16, lrb=0 ;
      sinon, lister exactement les formes à ajouter (`kTable`, `kGrid`,
      `kE4m3Tensor`, `kE4m3Row`, `tcq_rb`, `w4nl`).
- [ ] 1.3 Étendre `form_supported` et les noyaux denses aux formes retenues
      (`libr3d/r3d_gemm_iqr_m16.hip:382`, `r3d_gemm_iqr_gated.hip:214`,
      `r3d_iqr_logits.hip:286`), avec refus explicites pour les autres.
- [ ] 1.4 Couvrir chaque forme ajoutée par un test device (décodage et GEMM
      comparés à la référence hôte), y compris les cas limites de bits et les zéros.
- [ ] 1.5 Mettre à jour la documentation des rows/formes (`docs/OPS.md`,
      `libr3d/README.md`) et le tableau des tailles de plans.

## 2. W5 — MoE IQR / jalon M2 (~4 semaines, chemin critique)

- [ ] 2.1 Écrire la **référence hôte** de `moe_gemm_iqr` (R4) : `libref` ne
      l'implémente pas ; s'appuyer sur l'ordre de `libr4d/r4d_moe_iqr.hip`
      (ADR-4) et produire une fixture reproductible.
- [ ] 2.2 Déclarer la row `moe_gemm_iqr` dans `libr3d/r3d_rows.cpp` (schéma
      `moe_gemm_q`, classes paires/impaires `N_odd/K_odd`, scratch).
- [ ] 2.3 Porter la **run map** : tri des lignes, portée par expert, bande de
      décodage ≤ 16 lignes, chemin de préfill au-delà.
- [ ] 2.4 Noyau de décodage par expert et tuile de 16 lignes ; réutilisation de
      la tuile de poids sur plusieurs tuiles de tokens en préfill.
- [ ] 2.5 Mélange de formes dans une même couche (treillis + table `w4nl`,
      autres formes de 1.2) et **experts bf16 protégés** via la row bf16.
- [ ] 2.6 Repli des échelles par expert / ligne / bloc 128 ; lignes abandonnées,
      experts à zéro ligne, top-k variable, nombre d'experts variable.
- [ ] 2.7 Tests : couche MoE synthétique contre la référence hôte ; couche réelle
      du conteneur ; cas top-k/experts/protégés ; aucun accès hors plan.
- [ ] 2.8 Mesure : décodage 1 et 4 séquences, préfill 2048 et 4096, **η lien
      ≥ 80 %** (critère M2).
- [ ] 2.9 Doc de la row MoE et des limites connues.

## 3. W7 — QSA / jalon M3 (~2,5 semaines)

- [x] 3.1 `qsa_tail_store` (déjà déclaré, `libr3d/r3d_rows.cpp:2151`)
- [ ] 3.2 `qsa_work` — work (taille/ring), staging, page/block partial, rope.
- [ ] 3.3 `qsa_block_key` — pooling, norme, rotary au premier token, `mode`
      `mrope`/`imrope`, sections, yarn.
- [ ] 3.4 `qsa_score` — scores par bloc et par requête, bornes, `cu`/`nc`.
- [ ] 3.5 `qsa_select` — top-k (signé/non signé), sélection dense sous le seuil
      de budget, pages partielles, plusieurs séquences.
- [ ] 3.6 Égalité **au bit près** avec le chemin dense sous le seuil ; sélection
      identique à la référence (`libref/ref_qsa.cpp`, `ref_attn.cpp`).
- [ ] 3.7 Cas : spéculation désactivée, contextes longs, blocs incomplets,
      séquences multiples dans un même pas.
- [ ] 3.8 Mesure : temps de l'indexeur par pas et surcoût de lancement.
- [ ] 3.9 Doc des cinq ops QSA et de leurs refus.

## 4. M5/M6 — intégration Flash-Next (~1,5 semaine)

- [ ] 4.1 Démarrage du conteneur 121 GB
      (`<work-root>/models/qwen3.8-next-flash-iqr-v4.rad`) : toutes les rows
      résolues, aucun fallback inattendu.
- [ ] 4.2 Texte : 1 séquence et 4 séquences, préfill 2048 et 4096, mêmes
      sorties que la référence sur les prompts de démonstration.
- [ ] 4.3 Image / tour vision si le conteneur la sert (W10 déjà livré).
- [ ] 4.4 Robustesse : formes refusées, erreurs de plan, arrêt propre, reprise
      après redémarrage.
- [ ] 4.5 Aucun nouvel avertissement au démarrage ; rapport de startup relu.
- [ ] 4.6 Démonstration reproductible (script + prompts + lots) et documentation.
- [ ] 4.7 Mise à jour de `SUIVI_LOTS_RDNA3.md` (M2→M6) et du journal.

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

- [ ] 7.1 MTP / `mtp_enter` / spéculation (ADR-6 : hors v1).
- [ ] 7.2 Codebook `w4nl` dans `w4a8` (si `w4nl64-i8lm.rad` devient une cible).

## 8. Critères de clôture du projet

- [ ] 8.1 `r3d_selftest` 144/0/14 et `iqr_device_test` verts sur la pointe.
- [ ] 8.2 Flash-Next démarre et génère (texte, 1 et 4 séquences, préfills
      2048/4096) sans fallback ni avertissement nouveau.
- [ ] 8.3 MoE conforme (écart borné documenté) et **η lien ≥ 80 %** en préfill.
- [ ] 8.4 QSA identique à la référence et égal au dense sous le seuil.
- [ ] 8.5 Tableau réel/idéal complet, écarts expliqués, RAPPORT à jour.
- [ ] 8.6 Documentation des rows, architectures, limites et perf à jour.
- [ ] 8.7 Handover final (phase suivante) et journal à la date de clôture.

## 9. Ordre de marche suggéré (parallélisable)

1. **W5 MoE** (chemin critique) : 1.1–1.3 puis 2.1–2.9.
2. **W7 QSA** en parallèle : 3.2–3.9 (oracle libref déjà présent).
3. **Audit Flash-Next** en parallèle : 1.1, 4.1, 4.6 (lister les rows/formes
   réellement demandées avant de figer W1/W2).
4. **Mesures HC/PLE + R3** (sérialisées sur le GPU) pendant 1–2.
5. **M5/M6**, puis 5.1–5.7 et 8.1–8.7.

Si la cible se limite au **27B texte**, seuls 5.1–5.7 et 8.1/8.5–8.7 restent ;
le reste concerne Flash-Next.
