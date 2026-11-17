# Handover final — RDNA3, phase 7 (R21, R22, R23, R25, R26 et W27 livrés)

2026-11-17. Branche `rdna3-iqr`, état code `de81ffe` plus W27 (IQ8R groupe
256 : `8a3e75e`). Rappel des livraisons : W20 (contrat capacité/étendue), R21
(tier KV hôte), R22 (snapshots courts), R23 (tables MoE dans kbench), R25 (MTP
servi), R26 (vision servie), W27 (MiniCPM5-2B IQ8R servi à `NS=0`). **Périmètre
révisé le 2026-11-16** : la vision et le MTP de Flash-Next entrent dans la cible
(ADR-7, qui supersède l'ADR-6) et sont **livrés**. Ce document remplace
`HANDOVER_RDNA3_PHASE6.md` pour l'état courant ; §8 donne la reprise immédiate.

## 1. État vérifié

| Vérification | Résultat |
|---|---|
| `r3d_selftest` complet | **169 rows, 0 échec, 15 skips** |
| Tests device | `iqr_device_test`, `logits_bf16_device_test`, 4 tests QSA, `moe_iqr_test`, `moe_iqr_device_test`, `moe_w4nl`, `kv_store_b4`, `gather_rows_u8`, `mtp_enter_e4m3`, `logit_rerank_iqr`, `rope` h128, router, `gemm_nt` N=1, vision, GDN had, `act_i8g_group256` : **tous verts** |
| kbench Flash-Next (`--kernels libr3d:libref`) | **968/968, 0 échec, 0 divergence**, pire rel_l2 1,206e-2 ; **968/968 aussi en `--band-capacity`** (R23 + W25) |
| kbench 27B | **683/683, 0 échec** dans les deux modes (R23 ; 605 avant), pire rel_l2 3,619e-3 |
| Contrat capacité (W20) | `iqr_vocab_head_accepts_a_narrow_step_in_a_wide_band` : rouge sur `a772d70`, vert sur `fae6c44` |
| R21 tier KV (W22a) | rows u8 `gather_rows`/`scatter_rows` ; Flash-Next **2 groupes, 51,38 Kio/entrée**, 27B **1 groupe, 516 Kio** ; rejeu 27B : `to_host` 471, `promoted` 245, 0 échec (hash de vérification actif) |
| R25 MTP (W25) | rows `mtp_enter_e4m3`, `logit_rerank_iqr`/`_i8` ; selftest 164 ; kbench **968/968** (FN) et 683/683 (27B) dans les deux modes ; NS=3 servi : **4,51 → 8,21 tok/s (1,82×)**, texte identique, acceptation 40,3 % (66/61/37 % aux profondeurs 1–3) |
| R23 MoE kbench (W23) | tables MoE posées (5 plans, échelle IQR ligne×bloc, table de pointeurs) ; kbench 952 → **968/968** après W25 ; η MoE : IQR 70,5 % à M=32, w4nl 51 % à M=16 ; M=1 borné par le treillis (10–12 %) |
| R26 vision (W26) | build `-DRAD_WITH_FFMPEG=ON` (libavformat 62.3, ffmpeg 8.0) ; tour 856,68 Mio en VRAM, 48,27 Mio de staging pour 16 384 patches ; `capabilities.image=true` ; 512² : 256 tokens image (281 avec le texte), TTFT 8,19 s ; 1024² : 1 024 tokens, TTFT 28,73 s ; 2048² : 4 096 tokens, TTFT 121,5 s ; 3000² redimensionnée ; 18 cas de robustesse (2 images, alpha/JPEG, 4 × 400) sans fuite ; 480/480 cas vision et 968/968 kbench, 0 divergence |
| R22 snapshots (W22b) | ancre de queue sous le premier intervalle ; rejeu 9 tours : tokens servis 24 % → **52,6 %**, préfill recalculé −37 %, `no_snapshot` 6 → 2, checkpoints 1 → 7 ; `sched_test` 548 782 checks, `engine_e2e_test` et tests device verts |
| R9 IQ8R groupe 256 (W27) | rows `gemm_iqr_nt_m16_w256`/`gemm_iqr_tiled_w256` et `*_256_bf16` des trois producteurs ; selftest **169/0/15** ; `act_i8g_group256_device_test` 67 checks, `iqr_device_test` 75 checks (8 cas 256 bit-exacts) ; kbench FN 968/968 et 27B 683/683 dans les deux modes ; **MiniCPM5-2B IQ8R chargé (523 ops, 423 poids) et servi à `NS=0`** : texte cohérent, 40,2 tok/s à 60 tokens (KV bf16), kbench ciblé **110/110**, pire rel_l2 4,018e-05 ; conteneur 2B entier **384/384** (pire 1,810e-03) |
| Périmètre v1 (2026-11-16) | vision **livrée** (W26) et MTP Flash-Next **livré** (W25 : NS=3, 1,82×) ; le 27B reste à `NS=0` |
| Mesures à froid W18 (HC/PLE/QSA par row) | 182 + 147 cas, 0 échec ; `hc_read` **1,77 ms à M=1** (η 0,6 %), η lien **1,7 %** ; MoE par row mesuré en W23 (η 70,5 % à M=32, 51 % w4nl à M=16, 10–12 % à M=1) |
| Démo serveur M6 / W19 | préfill 38,7–41,9 tok/s, décodage 4,41 (1 séq.) / 15,5 (4) / 22,1 (6) tok/s ; cache ×12 sur prompt identique ; **R22 résolu** : rejeu 9 tours 24 % → 52,6 % de tokens servis, `no_snapshot` 6 → 2 ; 4 décodeurs ×7,0 à côté d'un préfill |
| Rejeu 27B / W19–W20 | chargement 15,87 GiB en 2,7 s ; `SEQS=1` préfill 318,8 tok/s (844), décodage 23,2 / 29,4 ; **`SEQS=4` corrigé** (W20) : préfill 317,5 tok/s, 4 requêtes à 67,3 tok/s agrégés |
| Contexte 16k | KV 1,23 GiB (4 × 16 384) ; prompt de 13 421 tokens servi à 40,9 tok/s ; >16k → 400 |
| 27B texte | décodage 28,4 tok/s (16 tok) / 21,2 (819), préfill 301 tok/s à 819 |
| Flash-Next texte | préfill 42 tok/s (2 290 et 4 570 tok), décodage 4,0–4,6 tok/s, 4 séquences concurrentes OK |
| Robustesse Flash-Next | prompt 12 001 tok → HTTP 400 clair, serveur vivant ; redémarrages propres |

## 2. Ce que W14–W27 et R21–R26 ont livré

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
- **W18** : mesures à froid HC/PLE/QSA par row, 0 échec, sans throttler ;
  η lien calculé (1,7 %) et cause attribuée à `hc_read`.
- **W19** : démo M6 (prompts distincts, 1/4/6 séquences, cache de préfixe,
  contention, 16k), robustesse étendue, rejeu 27B (R3 clos).
- **W20** : **contrat capacité/étendue** (R24) — le paramètre rangé d'une row est
  un plafond de bande, l'étendue est sur les tenseurs. Corrigé dans
  `logits_gemm_vocab_iqr` et dans `router_topk`, `router_topk_scatter`,
  `router_sigmoid_topk` (trouvés par le nouveau `kbench --band-capacity`).
  Test device épinglé ; 27B multi-séquence servi (67,3 tok/s agrégés).
- **W22a (R21)** : rows u8 `gather_rows`/`scatter_rows` pour les fragments KV
  opaques (dword par thread, pas de passage par le codec E4M3), contrainte
  `dtype == u8`, `n % 4 == 0` ; test `gather_rows_u8_device_test` (octets
  opaques, indices -1/hors plan, pitches larges, M = 1/8/2 048). Le tier KV
  hôte se configure et déplace réellement des entrées sur Flash-Next et le 27B,
  vérification par hash active.
- **W22b (R22)** : ancre de queue pour un prompt sous le premier intervalle.
  Sous `--prefix-resume exact` il n'existait ni point de grille ni snapshot
  sous l'intervalle, donc un tour d'agent court ne reprenait aucun état
  linéaire (`no_snapshot`). L'ancre est la dernière frontière de quantum, un
  quantum plus bas si elle est hors grille, dès 256 tokens d'ancre ; le préfill
  froid coupe là. Rejeu 9 tours : 24 % → **52,6 %** de tokens servis, préfill
  recalculé −37 %, `no_snapshot` 6 → 2 (premiers tours de sous-agents),
  checkpoints 1 → 7. Compromis : sous le premier intervalle la reprise est
  hors grille, comme `nearest` ; l'accord bit à bit n'y est plus garanti pour
  un tour appendu (un prompt répété reste exact), et au-delà de l'intervalle
  `exact` reste strict. `sched_test` 548 782 checks, `engine_e2e_test` et
  tests device verts.
- **W23 (R23)** : les tables MoE du conteneur sont posées dans kbench. La row
  IQR ne voit qu'une vue de 3 des 5 plans (`codes, scale, trellis,
  trellis.mad, scale.col`) ; le reader reconstruit la vue complète (sans
  `fwht128`, que le `Dec` n'applique pas), décode l'échelle IQR comme
  `scale[ligne] × scale.col[bloc 128]`, et le moteur passe une **table de
  pointeurs** que kbench reproduit (slab + pointeurs hôte/device, container et
  replay). Les formes IQR non servies sont refusées au layout. kbench passe de
  856 à **952/952**, puis **968/968** avec W25 (0 échec, 0 divergence) ; le
  temps par row et l'η sont mesurés (IQR 70,5 % à M=32, w4nl 51 % à M=16 ;
  10–12 % à M=1, borné par le décodage treillis).
- **W26 (R26)** : vision servie. Build `-DRAD_WITH_FFMPEG=ON` (l'hôte a
  libavformat 62.3 / ffmpeg 8.0 ; il exige les paquets `libav*-dev`, `sudo`
  n'est pas accordé à `<gpu-user>`). La tour (856,68 Mio) est en VRAM, le staging
  média prend 48,27 Mio pour 16 384 patches ; `capabilities.image=true` et les
  tours image 512²/1024² sont acceptés et répondus (le texte est correct :
  couleurs et positions, dégradé). Aucun changement de code : le build et la
  configuration ont suffi.
- **W27 (R9 / D1)** : IQ8R groupe 256. Le layout accepte l'encodage IQ8R et
  range l'échelle en bf16 par (row, 256), produit u8×bf16 ; le GEMM `i8a8h`
  templatise le groupe (deux blocs 128 dans un `sub` i32 exact, un seul fold,
  split-K refusé) ; les trois producteurs rotated font deux FWHT 128 par groupe
  avec un amax/une échelle sur 256. Le 2B (523 ops, 423 poids) charge et sert à
  `NS=0` en KV bf16 : texte cohérent, 40,2 tok/s à 60 tokens, kbench ciblé
  110/110 (GEMM bit-exact), selftest 169/0/15, kbench FN/27B inchangés dans les
  deux modes.

## 3. Flash-Next — configuration v1

`/tmp/fn-v1.yaml` (ou les scripts existants avec les env ci-dessous) :

```
P=8200 M=<work-root>/models/qwen3.8-next-flash-iqr-v4.rad \
TP=1 NS=3 MLEN=8192 HOSTPOOL=43000 KVHOST=1024 MNBT=2048 \
scripts/fnserve.sh
```

- `num-speculative-tokens: 3` : configuration servie depuis W25/R25 (MTP du
  Flash-Next, 1,82× le décodage à texte identique, acceptation 40,3 %).
  `NS=0` reste le repli de comparaison. Les tables MTP 48 et le draft head sont
  couverts par kbench (968/968).
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
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAD_GPU_TARGETS=gfx1100 -DRAD_WITH_FFMPEG=ON && \
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
- **kbench « Not covered »** : `ngram_hint` reste sans oracle ; les tables
  MTP 48 et le draft head sont couverts (968/968) ; restent 10 cas
  `gemm_nt_q_gated` i8a8h sans référence libref.
- **Spéculation** : Flash-Next à `NS=3`, acceptation ~40 % (66/61/37 % aux
  profondeurs 1–3), 1,82× le décodage, texte identique. Le 27B reste à
  `NS=0`. Un poids **joint** (`rad.planes`) d'une row device peut vivre en
  VRAM : un shim ne doit pas lire son en-tête sur l'hôte (SIGSEGV ; W25 a
  corrigé `gemm_nt_q_w2a8_m16`). Le bras `w2a8` doit aussi choisir un MT dont
  le chunk divise K (M=1/K=2560 → MT=4).
- **Média** : un build sans `-DRAD_WITH_FFMPEG=ON` refuse toute image
  (`capabilities.image=false`) ; l'hôte exige les paquets `libav*-dev`
  (`pkg-config --modversion libavformat`), et `<gpu-user>` n'a pas `sudo`.
- **Passes média** : au-delà de `image-max-pixels` (4 194 304 px ici) l'image
  est **réduite** à 2048², jamais refusée ; seul le nombre de tokens image le
  révèle. Une passe complète fait ~121 s (≈34 tok/s de préfill, ~135
  patches/s) et le watchdog peut afficher « nothing has been logged for
  20/40 s » (`core/engine.cpp:1307`) : c'est du bruit, la requête aboutit.
  `THROTTLE_STATUS` d'`amd-smi` vaut `THROTTLED` même au repos sur cet hôte :
  suivre puissance et fréquence.
- **Paramètre rangé = capacité** : `M` dans les args est le plafond de la bande
  (`max_seqs × (1 + spec)`), jamais le nombre de lignes du pas ; l'étendue est
  sur les tenseurs. Une row qui compare les deux refuse un pas plus étroit que sa
  bande et le moteur arrête le modèle (R24). Tester avec `--band-capacity`.
- **Tier KV hôte** : actif depuis R21. Les fragments sont des octets opaques et
  passent par les rows u8 (`n` et pitches en octets, multiples de 4) ; ne pas
  les router vers les rows bf16/f32, dont le codec E4M3 ne préserve pas un
  payload NaN. Vérifier avec `RADIANCE_DEBUG_TIER_VERIFY=1` (hash après gather
  et après scatter) et `/sessions` (`moved.to_host`, `promoted`, `failures`).

## 6. Reste à faire

1. **Mesures à froid — faites (W18)** : HC/PLE/QSA par row sur `w15-main`
   (`a772d70`), 0 échec, aucun throttler. Résultats dans
   `TABLEAU_REEL_IDEAL_RDNA3.md`. Deux constats :
   - **`hc_read` est le premier poste de Flash-Next** : 1,77 ms à M=1
     (η ≈ 0,6 % de la borne bande) et 470 ms à M=2 048, soit 0,23 ms par ligne
     ajoutée — la paire de matrices E4M3 est relue par ligne à ~28 GB/s. À 48
     appels par pas : ~93 ms/token en décodage (40 % du pas) et 47 % du préfill
     2 048. La row annonce 18,7 µs au décodage : écart ×95 à profiler côté
     engine. **C'est la piste n°1** pour le décodage à 4 tok/s.
   - **η lien = 1,7 %** (42 tok/s contre 2 545 tok/s), donc M2 non tenu, et le
     plafond mesuré est `hc_read`, pas le MoE. Le **temps par row MoE est
     mesuré depuis R23** : IQR 70,5 % de la bande à M=32, w4nl 51 % à M=16,
     10–12 % à M=1 (borné par le décodage treillis) — voir §10 du RAPPORT.
2. **Optimisations différées** : `hc_read` (ci-dessus) d'abord ; WMMA préfill
   (éteinte, 25–32 % plus lente),
   occupation du gated fusionné (1,4 bloc/CU), `attn_dense` à M=16384,
   `attn_paged` h256 isolé. **`qsa_score` à M=512 : cause mesurée (W24)** —
   79,0/82,7/83,3 ms à froid selon trois passes (103 GFLOP/s, 258,5 MiB
   d'opérandes, aucun throttler), contre **9,0 ms pour la row hôte AVX** sur
   32 threads. Un thread traite un bloc, donc une warp lit **32 pages
   distinctes** au même offset : 16× d'amplification de secteur et une row
   bornée par L2, pas par le calcul. Un déroulage 4 blocs/thread (chaînes
   indépendantes) a donné **163 ms, 2× plus lent** et a été retiré. La suite est
   une réécriture coalescée/tuilée (pages et lignes de q en LDS), à faire avec
   un A/B sur ce protocole.
3. **Robustesse étendue — faite (W19)** : 12 002 tokens → 400 en 0,00 s avec le
   nombre de tokens à couper ; `max_tokens` hors contexte **plafonné** à 8 191
   (pas de 400) et `finish=abort` à la déconnexion, slot rendu ; abandon à 3 s
   puis requête suivante en 3,57 s ; 6 admissions 6/6 ; `/health` 200 et
   0 requête en vol. Démo M6 dans `TABLEAU_REEL_IDEAL_RDNA3.md` §M6.
   Une limite restante : le rejeu multi-agents n'a **rien** réutilisé (0 %,
   `no_snapshot` ×4, 1 checkpoint pour 5 tours) alors qu'un prompt identique
   réutilise 4 096 tokens (TTFT 108,4 → 9,0 s) et qu'une conversation qui
   grandit réutilise 2 048 par tour. **R21 et R22 sont résolus** (W22) : le
   tier KV hôte déplace des entrées et, depuis l'ancre de queue, le rejeu
   9 tours sert 52,6 % de ses tokens (24 % avant), `no_snapshot` 6 → 2 et
   7 checkpoints écrits.
   Contention (`fnconc`) : 4 décodeurs 12,6 s seuls → 87,6 s à côté d'un préfill
   de 3 000 tokens (+590 %), préfill 35 tok/s (vs 42 seul).
4. **R9 — résolu (W27)** : `minicpm5-2b-iq8r.rad` (2B, IQ8R rotated i8,
   groupe 256) **charge et sert** à `NS=0`. La cause était bien la largeur de
   groupe : `libr3d` n'avait ni producteur rotated ni GEMM `i8a8h` à 256.
   Livré : producteurs groupe 256 (`rmsnorm_had_quant_i8g`, `had_quant_act_i8g`,
   `gated_had_quant_i8g` ; deux rotations de 128, un amax et une échelle par
   256), GEMM groupe 256 (`gemm_iqr_nt_m16_w256`, `gemm_iqr_tiled_w256` ;
   encodage IQ8R au layout avec l'échelle bf16 = produit u8×bf16 par (row,256),
   un seul `sub` i32 par 256, split-K refusé). Service en **KV bf16** : les rows
   `attn_paged`/`kv_store` de cet arch ne prennent pas i8 à `head_dim=128`.
   Chiffres : 523 ops / 423 poids déclarés, texte cohérent, 40,2 tok/s à
   60 tokens, kbench ciblé 110/110 (pire 4,018e-05, GEMM bit-exact). Restent
   pour un usage complet : variantes `ar_*`/non fusionnées (tp > 1), fold
   `gemm_nt_q_gated` 256 (non déclaré pour un poids i8), puis DSpark pour
   `NS > 0` (checklist D2).
4bis. **R24 — corrigé (W20)** : le paramètre rangé est une **capacité de bande**,
   pas le nombre de lignes du pas. `logits_gemm_vocab_iqr` lit l'étendue sur `x`
   (comme `libr4d`), et l'audit `rad-kbench --band-capacity` a trouvé la même
   faute dans `router_topk`, `router_topk_scatter` et `router_sigmoid_topk`
   (corrigés). Épinglé par `iqr_vocab_head_accepts_a_narrow_step_in_a_wide_band`
   (échec sur l'ancienne `.so`, vert sur la nouvelle) ; kbench Flash-Next 856/856
   dans les deux modes ; `r3d_selftest` 159/0/15. Le 27B se sert donc à
   `--max-num-seqs 4` (`--kv-cache-dtype i8`, `--num-speculative-tokens 0` : sans
   `NS=0`, le drafter fenêtré réclame bf16) et 4 requêtes donnent 67,3 tok/s
   agrégés. Pour trancher R20 sans toucher au noyau : `--profile-ops true` donne
   le temps device par op.

   Arbre de référence corrigé : **`<work-root>/work/w20-r24`** (même source que
   `a772d70` plus W20, build ccache 1 min 27). `w15-main` reste l'état `a772d70`
   non corrigé : ne pas s'en servir pour un A/B avec le correctif.
5. **Rapport final — mis à jour (2026-11-16)** : encadré de mise à jour en tête,
   note sur la §7.3, nouvelle §7.5 « Mesures réelles » et §9 révisée. Il reste
   **hors dépôt** (décision du §7 ci-dessous inchangée) ; le tableau et le journal
   portent les chiffres dans le dépôt.
6. **Image / vision — livrée (ADR-7, W26)** : build
   `-DRAD_WITH_FFMPEG=ON`, tour en VRAM, `capabilities.image=true`, images
   512²/1024²/2048² servies. Robustesse : 18 cas (2 images par requête,
   alpha/JPEG, ordre des parts, 4 erreurs 400 explicites), VRAM constante,
   aucune fuite. Limites : resize silencieux au-delà de 4 194 304 px,
   avertissements de watchdog pendant les passes longues, pass complet
   ~121 s, type MIME reconnu aux octets, 2 × 2048² non testé. Reste non
   testé : la vidéo (annoncée par le processeur) et l'audio. Le corpus de
   démonstration vit hors dépôt dans `<work-root>/images` ; les paquets
   `libav*-dev` sont requis et `sudo` n'est pas accordé à `<gpu-user>`.
7. **Nettoyage code — livré** : `dot4_i8`/`i8_codes4` partagés dans
   `r3d_common.h`, traces de bring-up retirées (W25, branche
   `rdna3-cleanup-helpers`).
8. **Hors v1** : codebook `w4nl` dans `w4a8` seulement. Le MTP de Flash-Next
   est **livré** (W25) : rows device, service à `NS=3`, 1,82× le décodage ;
   la couverture kbench des tables MTP 48 et du draft head est faite
   (968/968). Le 27B
   garde `NS=0` (drafter h128 bf16).

## 7. Non ajouté volontairement

`API_DISTANTE.md`, `HANDOVER_RDNA3.md`, `PLAN_IMPLEMENTATION_RDNA3.md` et
`external-docs/` restent hors dépôt (déjà untracked ; PDF ISA AMD sous
copyright). `RAPPORT_RDNA3.md` **est entré dans le dépôt le 2026-11-17**
(décision prise en W24).

## 8. Pour l'agent suivant

Lire, dans cet ordre : ce fichier, `CHECKLIST_RDNA3.md` § « Reste à faire —
synthèse », `SUIVI_LOTS_RDNA3.md` § « Prochains lots », puis
`TABLEAU_REEL_IDEAL_RDNA3.md` § Flash-Next et § M6. `AGENTS.md` porte le
protocole kbench et les consignes de travail.

**Où en est le programme.** Flash-Next texte et vision sont servis, le 27B
aussi, kbench 968/968 et 683/683 dans les deux modes, le tier KV hôte fonctionne
(R21), un tour image passe (R26), le MTP de Flash-Next est servi (R25, 1,82× le
décodage) et **MiniCPM5-2B (IQ8R, groupe 256) charge et sert à `NS=0`** (W27).
Selftest **169 rows**. Le périmètre v1 a été **élargi le 2026-11-16** (ADR-7) :
la vision et le MTP sont **livrés**. Ne pas rouvrir les contrats clos
(R17–R19, R24) sans une mesure qui les contredit.

**Ordre de travail recommandé.** L'optimisation noyau reste en pause faute de
feu vert ; W21 reprend la première place dès que le feu vert est donné.

1. **W21 `hc_read` (R20)** — le plus gros levier de performance : 1,77 ms à
   M=1 (η 0,6 %) et 0,23 ms/ligne, ~40 % du décodage et 47 % du préfill.
   Commencer par `--profile-ops true` (temps device par op) sur Flash-Next ; la
   forme deux-grilles et la température sont déjà écartées.
2. **W24 — fait** : `qsa_score` à M=512 mesuré (79,0/82,7/83,3 ms), cause
   écrite (warp = 32 pages à offset constant, 16× d'amplification de secteur ;
   la row hôte AVX fait 9,0 ms), essai ILP 4 blocs retiré (2× plus lent).
   `RAPPORT_RDNA3.md` est au dépôt. Reste une réécriture coalescée/tuilée.
3. **Écart RDNA4 → RDNA3** — **233 rows / 176 ops** absents de `libr3d` (les
   rows IQ8R du tronc sont livrées en W27 ; il reste les `ar_*`, le fold gated
   256 et DSpark sur le chemin IQ8R). Feuille de route checklist **§D**, liste
   complète RAPPORT §10. P0 restant : DSpark si `NS>0` ; P1 : `tp>1` ; le reste
   est conditionné à la décision de périmètre (D0).

**Décisions qui appartiennent à l'utilisateur** : (1) feu vert pour
l'optimisation noyau (W21). La vision et le MTP sont tranchés (ADR-7,
2026-11-16 : inclus) et livrés ; l'ajout du RAPPORT au dépôt est tranché
(2026-11-17 : inclus).

**Discipline de mesure.** Un chiffre sans commande, sans horloge et sans état de
la carte ne compte pas. Carte froide, `flock`, un seul processus, build neuf pour
tout A/B, serveur arrêté avant reconstruction, `/proc/<pid>/exe` vérifié.
