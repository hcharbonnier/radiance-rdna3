# Journal du programme — support RDNA3 (gfx1100)

Une entrée par jour travaillé. Chaque mesure cite l'état de l'arbre, la configuration,
les horloges, les températures, la médiane, le p95 et le nombre de répétitions.

## 2026-11-15 — J1 : démarrage, vérification d'environnement, reproduction de l'échec

**Lot** : préparation, W0.1 (inventaire).

**Environnement vérifié**

- Hôte `ai` joignable (`ssh <gpu-user>@<gpu-host>`), noyau 7.0.0-43-generic, 122 GiB RAM.
- W7900 `gfx1100`, 48 301 604 864 octets de VRAM, 26,7 Mo utilisés au repos ; iGPU
  `gfx1036` présent et ignoré par le moteur.
- `/opt` : 279–296 Go libres (le rapport citait 424 Go ; écart noté, pas bloquant).
- ROCm 10.1, cmake 4.2.3, ninja 1.13.2, Docker 29.9.3 ; `open-webui` et `cloudflared`
  tournent, non touchés ; aucun serveur radiance résiduel.
- Clone de référence `<work-root>/radiance` propre sur `ddcb9c6`; `build/bin`
  complet (radiance, rad-info, rad-kbench, rad-yaml, r3d_selftest).
- `RADIANCE_HOME=<work-root>/radiance/build/radiance_home` fonctionne.
- Conteneur Flash-Next présent : 121 678 372 864 octets (113 GiB).
- 27B téléchargé : 18 700 496 896 octets (17,42 GiB), terminé à 120 Mo/s ;
  renommé `qwen3.8-27b-iq4r-l-v4.rad`.

**Reproduction de l'échec (confirmée)**

Configuration une carte générée par `rad-yaml` (tp 1, 4 séquences, 200K,
`expert_tiered`, pool 43 000 MiB, KV i8, 3 tokens spéculatifs, chunk 4096,
port 8100), lancée hors conteneur, 300 s de délai. Journal :

```
E qwen4exp_fp8: rad_arch_declare returned invalid argument
E startup stopped: invalid argument
```

Noyaux chargés : `libavx@0.1.0,libr3d@0.1.0,spechub@0.1.0,libref@0.1.0` ;
plugin `qwen4exp_fp8@0.1.0`. Même monde que le handover.

**Inventaire 27B (`rad-info`, `--recipe`)**

- `qwen35`, 64 couches, `n_embd` 5120, `n_head` 24, `n_head_kv` 4,
  **`head_dim` 256**, FFN 17408, 262144 de contexte, 0 expert.
- Trunk : `kTrellisMad` u4/u5, `tcq_rows` 256/512/1024, `tcq_order=1`,
  `scale=f32` par ligne (`kRowF32`) ou `scale=bf16` par (ligne, 128) pour
  `attn_output`/`ssm_out`.
- `token_embd` : ancien treillis à table L12 (u4, échelle E4M3 par 128 + f32).
- Tête : `output.k4..k7`, `kTrellisMad` u4..u7, `kRowF32`, vocabulaire pondéré
  (quatre classes), plus `dflash.draft_head` u2 et un drafter MXFP4.
- **Aucune forme table `w4nl` dans les GEMM denses du 27B** : la table est propre
  aux experts `ffn_down` du Flash-Next.

**Écarts par rapport au plan, à traiter**

1. Le 27B a `head_dim=256`, GQA 6, pas h128. `libr3d` ne sert que h128
   (`r3d_rows.cpp:1437-1446`). M1 (« 27B servi ») exige donc l'attention h256
   (W6) ou un repli libref (correct mais lent). W6 dépend seulement de W1 et
   peut démarrer dès le Go, en parallèle de W3/W4.
2. Le trunk du 27B est entièrement IQR : aucun GEMM int8 classique à ajouter.
   En revanche l'embedding (treillis à table L12) et la tête vocabulaire
   (`logits_gemm_vocab_iqr`) manquent aussi à `libr3d` ; ils ne sont pas dans
   W0 mais conditionnent M1.
3. Le plan décrit W0 comme « treillis et table » ; pour le 27B le besoin réel
   est treillis calculé (`kTrellisMad`) u4/u5, `kRowF32` et `kBf16`. La table
   `w4nl` reste à couvrir en W5 (MoE Flash-Next).

**Décisions**

- Branche locale `rdna3-iqr` créée (aucun push).
- Répertoire de build dédié : `<work-root>/work/radiance` (synchronisé depuis
  le dépôt local, `.git`, `build`, `external-docs` exclus).
- Référence IQR dupliquée dans `libr3d` (`r3d_iqr.h`), sans include de `libr4d`
  (ADR-1) ; tags privés `0x60000000`/`0x70000000` (ADR-2).
- W0 commence par les hooks hôte et un relayout scalaire vérifié contre
  `libr4d` ; l'optimisation du transpose hôte viendra si le chargement est
  trop lent.

**Prochaine étape**

- Écrire `libr3d/r3d_iqr.h`, les hooks `layout/relayout/unrelayout`, le
  décodeur `gfx11` (treillis calculé), les rows GEMV et 16 tokens, puis
  rejouer `rad-kbench --op gemm_iqr_nt_m64`.

**Blocages** : aucun.

## 2026-11-15 — J1 (suite) : W0 dense, décision M0

**Lot** : W0/W1/W2/W3 (tranche dense) ; M0.

**Livré**

- `libr3d/r3d_iqr.h` : définition neutre des formes IQR (copie contrôlée de libr4d,
  ADR-1), tags privés `0x60000000`/`0x70000000` (ADR-2).
- `libr3d/r3d_iqr_layout.cpp` : hooks layout/relayout/unrelayout, transpose treillis
  scalaire parallélisé par tuiles, refus des formes non lues.
- `libr3d/r3d_iqr_dec.h` : décodeur gfx11 du treillis calculé (états, `sudot4` byte-sum,
  échelles `kRowF32`/`kBf16`, `tcq_rb` préparé).
- `libr3d/r3d_gemm_iqr_m16.hip` : row `gemm_iqr_nt_m64`, GEMV (une wave par tuile de
  16 lignes, réduction LDS) et 1–16 tokens (tuile décodée en LDS, paires (token, row)).
- `tests/iqr_parity_test.cpp` : 24 formes, 285 vérifications, octets identiques à
  libr4d, aller-retour.
- `tests/iqr_device_test.cpp` : référence hôte indépendante, 4 cas bit-exacts, plus un
  mode `RAD_IQR_BENCH=1`.

**Mesures** (carte au repos, 25–41 °C, médiane de 5 lots)

| Forme | M | Temps | GB/s | η |
|---|---|---|---|---|
| 12 288×2 560 u5 | 1 | 26,41 µs | 744,5 | 86,3 % |
| 12 288×2 560 u5 | 16 | 239,1 µs | 82,2 | 9,5 % |
| 5 120×17 408 u4 | 1 | 100,2 µs | 444,8 | 51,5 % |
| 5 120×17 408 u4 | 16 | 710,6 µs | 62,7 | 7,3 % |

**Constats**

- Aucune fixture IQR dans `data/kernels.rkb` : le critère de rejeu est remplacé par la
  référence hôte (mémo M0, §Écarts). `strings` ne trouve ni `gemm_iqr` ni `trellis`.
- `sudot4` gfx11 est signé × non signé : le produit signé exige `(true, w, true, a)`
  comme libr4d. Le premier essai `(false, a, false, w)` produisait des sorties fausses.
- GEMV : `a_scale` est par (token, 128), pas par ligne de poids ; corriger l'index a
  fait passer les 16 lignes de la tuile.
- M=16 par `v_dot4` reste un échafaudage (4,2 TFLOP/s) ; W3 doit le porter vers la
  matrice 16×16×16 IU8.
- Down 5 120×17 408 : 51,5 % seulement (320 tuiles, split-K à ajouter).

**Décision** : **M0 = GO** (`MEMO_M0_RDNA3.md`).

**Parallélisation** (autorisée par l'utilisateur, plan §6.5)

- Deux sessions Agent Manager ouvertes depuis `rdna3-iqr` (c9f7d51) :
  `rdna3-w6-attn-h256` (W6 attention h256/GQA/KV i8) et
  `rdna3-w8-w9-hc-ple` (W9 PLE puis W8 HC). Travail d'implémentation seulement :
  aucun benchmark ni test GPU concurrent, build device autorisé. Revue et fusion
  assurées par le coordinateur.
- Les lots IQR (W1–W5) restent séquentiels : point de contention du décodeur
  (plan §6.5). Les benchmarks restent séquentiels (une carte).

**Prochaine étape** : W1/W2 (table, grid, `kE4m3`, `tcq_rb`, refus), W3 (m16), W4
(m64/préfill), et W6 en parallèle (attention h256/GQA 6).

**Blocages** : aucun. Points d'attention : fixture IQR absente ; M1 dépend de W6.

## 2026-11-15 — J1 (suite) : declare du 27B, revue W6

**Lot** : W1/W2/M1 (préparation), revue du sous-agent W6.

**Ce qui débloque le declare du 27B**

Le 27B échouait avant toute requête de layout (`rad_arch_declare` → E_INVAL sans
message). Instrumentation opt-in ajoutée : `RAD_ARCH_TRACE` (étape `RAD_ARCH_TRY`
qui échoue), `RAD_BUILD_TRACE` (refus de poids par row, erreurs du builder),
`RAD_R3D_TRACE` (refus des hooks IQR). Causes trouvées et corrigées :

1. `gated_had_quant_i8g` et `had_quant_act_i8g` de libr3d n'acceptaient que
   `n == 128` ; le 27B a n=5120, 6144, 17408. Les kernels servent désormais une
   wave par (token, groupe de 128), grille (M, n/128), ce qui évite aussi le bug
   DPP de gfx1102 dans une boucle.
2. `embed_lookup` IQR : row `embed_lookup_iqr` ajoutée (décodage par ligne
   demandée, Hadamard 128 en f32, échelles E4M3/f32), port de libr4d.
3. `logits_gemm_vocab` : schéma device-only (libref ne le déclare pas) et row
   `logits_gemm_vocab_iqr` avec les hooks de layout ; le kernel refuse encore
   (`RAD_E_UNSUPPORTED`) jusqu'à son écriture.
4. `gemm_iqr_tiled` : row M ≥ 1 (declare résout l'architecture à son bucket le
   plus large) ; aujourd'hui un lancement m16 par tranche de 16, correct mais
   re-lecteur du poids.

Après ces correctifs, le 27B **déclare** sur gfx1100 (rapport de sélection
complet, 135 bandes device non résolues) :

| Op manquant | Paramètres | Lot |
|---|---|---|
| `attn_paged` | h256, gqa 6, kv i8, block 16, q_len 16/512 | W6 |
| `kv_store` | h256, n_head_kv 4, kv i8 | W6 |
| `rope` | h256, n_head 24/4, mode **imrope**, sections 11 11 10 | W6 |
| `rope_table` | rot 64, theta 1e7, f32 | W6 |
| `attn_dense` | hd72, vison (ViT) | W10 |
| `rope` | hd72, mode **axial**, vison | W10 |

**Revue W6**

Le sous-agent a livré `aa86e04` (4 rows h256, 784 lignes, selftest de 15 cas).
Fusion locale faite puis **annulée** : `r3d_selftest` sur la carte montre 13 cas
faux sur 15 (decode causal/ragged dès dim 0 ; prefill faux à partir de row 16,
donc frontière de tuile ; max |diff| jusqu'à 178). Seuls « windowed » et
« prefill empty » passent. Le sous-agent a reçu les cas exacts, l'autorisation
d'exécuter `r3d_selftest` (pas de benchmark ni serveur) et le complément de
périmètre kv_store i8 / rope_table / imrope. La carte lui est laissée pour cette
passe.

**Décisions**

- `r3d_selftest` : l'invariant « aucune row ne porte de hook de layout » est
  assoupli pour les rows IQR (ADR-1 exige l'inverse), le reste inchangé.

**Prochaine étape** : attendre la correction W6, puis (1) écrire le kernel
`logits_gemm_vocab_iqr`, (2) W5 table/grid, (3) mesurer le 27B et M1.

**Blocages** : aucun ; carte occupée par W6.

