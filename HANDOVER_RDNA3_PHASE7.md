# Handover — RDNA3, phase 7 (W14–W17 livrés)

2026-11-16. Branche `rdna3-iqr`, pointe de code `a772d70`. Ce document remplace
`HANDOVER_RDNA3_PHASE6.md` pour l'état courant.

## 1. État vérifié

| Vérification | Résultat |
|---|---|
| `r3d_selftest` complet | **159 rows, 0 échec, 15 skips** |
| Tests device | `iqr_device_test`, `logits_bf16_device_test`, 4 tests QSA, `moe_iqr_test`, `moe_iqr_device_test`, `moe_w4nl`, `kv_store_b4`, `rope` h128, router, `gemm_nt` N=1, vision, GDN had : **tous verts** |
| kbench Flash-Next (`--kernels libr3d:libref`) | **856/856, 0 échec, 0 divergence**, pire rel_l2 1,2e-2 |
| 27B texte | décodage 28,4 tok/s (16 tok) / 21,2 (819), préfill 301 tok/s à 819 |
| Flash-Next texte | préfill 42 tok/s (2 290 et 4 570 tok), décodage 4,0–4,6 tok/s, 4 séquences concurrentes OK |
| Robustesse Flash-Next | prompt 12 001 tok → HTTP 400 clair, serveur vivant ; redémarrages propres |

## 2. Ce que W14–W17 ont livré

- **QSA complet** : `qsa_work`, `qsa_block_key`, `qsa_score`, `qsa_select`
  (plus `tail_store`), tests libref ; formes alignées sur libref (`nsel`
  rang 1, `nc` `[S]`, `state` `[M,1,ring,n]`).
- **MoE IQR** : row `moe_gemm_iqr` (treillis), run map triée, bande 16 lignes
  et préfill, référence hôte `tests/iqr_ref.h` + `moe_iqr_test` (10 formes,
  0 écart).
- **`moe_gemm_w4nl`** : forme `w4nl64a8i8h` (codes u4, table f32[16] vérifiée,
  E4M3/64 sous niveau f32 2^−13), MTP 48 couvert, `a_group` 128/256.
- **Formes denses** : `kTable`/`kGrid` (u2..u5) ; **`kSigned` 8** (tronc i8
  tourné), bandes mixtes.
- **Rows Flash-Next** : `gdn_conv_recurrent_update_had`, `kv_store` block 4,
  `rope` imrope h128, router fusionné, `gemm_nt` N=1, `gemm_nt_bias`,
  `grid_embed`.
- **Bring-up Flash-Next** : déclaratif **0 bande device manquante** (5 074
  questions), serveur 44,35 GiB/carte chargé en 21 s ; correction des refus
  runtime (`hc_read` forme logique, `moe_gather` lignes du pas, tables
  `moe_iqr` initialisées, plan de tête au lieu d'un peek VRAM).
- **W17** : tête `logits_gemm_bf16` — échelle par ligne `1x128` indexée comme
  une grille 128×128 (rel_l2 0,53 → **1,7e-3**) ; formes QSA. Nouveau test
  `logits_bf16_device_test`.

## 3. Flash-Next — configuration v1

`/tmp/fn-v1.yaml` (ou les scripts existants avec les env ci-dessous) :

```
P=8200 M=<work-root>/models/qwen3.8-next-flash-iqr-v4.rad \
TP=1 NS=0 MLEN=8192 HOSTPOOL=43000 KVHOST=1024 MNBT=2048 \
scripts/fnserve.sh
```

- `num-speculative-tokens: 0` : **MTP hors v1** (la couche 48 et
  `mtp.draft_head` ne sont pas chargés ; les 78 cas kbench non couverts sont
  MTP et `ngram_hint`).
- `max-num-batched-tokens: 2048` : le plan `qsa_score` fait 512 Mio à 2 048
  lignes et 1 Gio à 4 096, que le plugin refuse.
- `placement: expert_tiered`, `host-pool-mib: 43000`, `kv-cache-dtype: i8`.
- `max-model-len: 8192` pour la démo ; au-delà, redimensionner le pool KV.

Démo mesurée : prompts 30 / 2 290 / 4 570 tokens (1 séquence) et 4 requêtes
concurrentes de 30 tokens ; `curl /v1/completions`, champs `timings`.

## 4. Commandes

Build neuf (obligatoire pour un A/B ; les objets d'un build incrémental après
`rsync -a` peuvent être périmés) :

```bash
rsync -a --exclude /.git --exclude /build --exclude /external-docs --exclude /.kilo \
  ./ <gpu-user>@<gpu-host>:<work-root>/work/w17-main/
ssh <gpu-user>@<gpu-host> "cd <work-root>/work/w17-main && \
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAD_GPU_TARGETS=gfx1100 && \
  cmake --build build -j 16"
```

Tests (un seul GPU, `flock`) :

```bash
ssh <gpu-user>@<gpu-host> "flock -w 7200 /tmp/radiance-gpu.lock -c '\
  cd <work-root>/work/w17-main && ./build/tests/iqr_device_test && \
  ./build/tests/logits_bf16_device_test && ./build/tests/moe_w4nl_device_test && \
  ./build/bin/r3d_selftest ./build/radiance_home/kernels/libr3d.so'"
```

kbench de contrôle (sans `--bench`, ~3 min) :

```bash
rad-kbench -m qwen3.8-next-flash-iqr-v4.rad --home <home> \
  --kernels libr3d:libref --max-cases 1
```

## 5. Pièges

- **Builds périmés** : `rsync -a` préserve les mtimes ; un rebuild incrémental
  peut ne rien reconstruire (faux A/B de 25 %). A/B = répertoire neuf.
- **Home vivant** : les `.so` s'écrivent dans `radiance_home/kernels` ; tuer le
  serveur avant de reconstruire.
- **Port 8200/8100** : vérifier qu'il est libre et que le processus répondant
  est bien le binaire visé (`/proc/<pid>/exe`).
- **kbench** : la carte throttle (PPT0/PPT1) sur un run `--bench` long ; les
  timings par row d'un run de 8 min ne sont pas exploitables. Faire les mesures
  de perf sur carte froide et en petits lots.
- **Échelles de tête** : une tête quantifiée peut avoir une échelle par ligne
  (`1x128`), pas seulement `128x128` ; l'indexation doit venir du bloc encodé,
  pas d'une grille supposée.
- **kbench « Not covered »** : les 78 cas MTP/`ngram_hint` ne sont pas des
  régressions.

## 6. Reste à faire

1. **Mesures à froid** : HC/PLE, MoE/QSA par row et **η lien ≥ 80 %** en
   préfill 2048/4096 (M2/M4) ; le kbench de 8 min throttle.
2. **Optimisations différées** : WMMA préfill (éteinte, 25–32 % plus lente),
   occupation du gated fusionné (1,4 bloc/CU), `attn_dense` à M=16384,
   `attn_paged` h256 isolé.
3. **Robustesse étendue** : erreurs de plan, arrêt/reprise en cours de
   génération, plusieurs admissions.
4. **R9** : vérifier E4M3/rotated sur un modèle non-IQR (aucun 2B/4B sur
   l'hôte).
5. **Rapport final** : `RAPPORT_RDNA3.md` (hors dépôt), tableau et limites.
6. **Hors v1** : MTP, codebook `w4nl` dans `w4a8`.

## 7. Non ajouté volontairement

`API_DISTANTE.md`, `HANDOVER_RDNA3.md`, `PLAN_IMPLEMENTATION_RDNA3.md`,
`RAPPORT_RDNA3.md` et `external-docs/` restent hors dépôt (déjà untracked ;
PDF ISA AMD sous copyright).
