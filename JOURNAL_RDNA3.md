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

## 2026-11-15 — J1 (suite) : revue et fusion W8/W9 (HC, PLE)

**Lot** : W8, W9.

Le sous-agent a livré `05fd026` (11 rows, 2728 lignes, tests hôte et device) sur
`rdna3-w8-w9-hc-ple`. Revue faite : rows append-only, schémas copiés slot pour
slot (75 opérations déclarées par plus d'un plugin, 0 désaccord selon
`rad-schemas`), aucune modification de row existante, `ar_*` limité à
`world_size=1` et refusé au-delà.

Validation device après correction d'un **bug du test** (`ngram_hint` passait
`eos=-1`, que libref et le shim refusent ; `eos=1000` désormais) :

- `hc_layout_test` : 26 vérifications, vert.
- `ple_hc_device_test` : 8 cas / 48 vérifications, verts — `ple_conv`,
  `ple_gate`, `embed_lookup_q` E4M3, `hc_read` bf16, `hc_write`/`enter`/`project`,
  `ar_hc_write` et `ar_gather_hc_write` à un rang, `ngram_hint`.
- `ngram_hint` a d'abord renvoyé `RAD_E_SHAPE` sur `eos=-1` : comportement
  conforme à libref (`ref_ngram.cpp:122`), c'était le test qui était faux.

**Fusion** : `merge W8/W9` dans `rdna3-iqr`, puis commit de la correction du
test et de l'enregistrement CTest `ple_hc_device_test_card` (label gpu,
`RUN_SERIAL`). Build et tests refaits depuis l'arbre fusionné : verts.

**Décisions**

- Le tag de layout `r4d.hc.e4m3.*` est conservé dans libr3d : c'est le même
  format stocké, et le réutiliser évite un re-layout des conteneurs produits
  pour RDNA4. ADR-2 ne couvre que les tags de dtype IQR.
- Les hooks `opd_shape` et les entrées `data/kbench_rows` de HC/PLE sont
  repoussés à W11 : ils ne bloquent ni M1 ni M2.

**Reste** : performance HC/PLE non mesurée (aucun benchmark pendant la passe) ;
le double `exp` de `ple_gate` est à profiler ; variantes `q/rq` int8 non
couvertes par un test device.

**Blocages** : aucun.

## 2026-11-15 — J1 (suite) : tête vocabulaire, garde-fou

Le kernel `logits_gemm_vocab_iqr` est écrit (quantisation FWHT/int8 par groupe,
un GEMM IQR par classe, scatter des ids) avec son scratch, mais il **échoue la
référence** : à q/s et poids identiques, la ligne device rend `0.0449` là où la
référence calcule `-0.279` (forme tête 27B u4 L16 kRowF32, N=256, K=512, M=2).
Le `m16` et le lancement par tranches donnent la même valeur, donc le GEMM est
cohérent avec lui-même ; l'écart est dans la chaîne q/s/poids ou dans la
référence du test, pas dans le chunking. Le scratch et le scatter sont corrects
(la valeur device est bien écrite à l'id attendu).

Décision : la row **refuse** (`RAD_E_UNSUPPORTED`) tant que le test
`RAD_IQR_VOCAB=1` n'est pas vert, pour qu'un conteneur déclare puis échoue
franchement au premier pas au lieu d'émettre des logits faux. À reprendre en
W11, avec le kernel de tête dédié (accumulation f32).

## 2026-11-15 — J1 (suite) : W6 revu et fusionné, 27B à la vision près

**W6** : trois commits (`aa86e04`, `3a1a2ff`, `ed99cde`). Revue indépendante :
`r3d_selftest` complet depuis un build séparé, avec `libref.so` à côté du plugin.

- 15/15 cas h256 verts contre libref : decode/prefill gqa12 et gqa6, causal,
  ragged, windowed, empty, verify, split-KV ; `max |diff|` 0 à 0,125 bf16 selon
  le cas.
- Ajouts vérifiés : `kv_store` int8 head_dim 256 block 16, `rope_table` f32
  rot 64, `rope` mode imrope h256.
- Le seul échec du selftest était l'invariant statique « pas de hook de layout »
  (rows IQR/HC) ; exception étendue à `hc_read` (tag `r4d.hc.e4m3` partagé).
- Fusion `merge W6` dans `rdna3-iqr`, build et selftest depuis l'arbre fusionné :
  **140 rows, 0 échec**, 13 skips (shims).

**27B après W6**

Il ne reste **que la tour vision** : 54 bandes, toutes `attn_dense` hd72
(gqa 1, causal 0, seqlen 16384) et `rope` mode **axial** hd72 (sections 18 18).
Toutes les bandes texte résolvent, y compris `attn_paged` h256 gqa 6 kv i8,
`kv_store` i8, `imrope` et `rope_table`.

Deux contraintes trop étroites corrigées : `rope_table` exigeait `M == 64` alors
que la bande a `M=63` (désormais 1..64), et `rope imrope` exigeait la forme
fusionnée 24/4 alors que le 27B appelle q seul (24, n_kv 0) puis k seul (4, n_kv
0). Le shim lisait déjà la largeur du couple.

**Prochaine étape** : W10-lite (attn_dense hd72 + rope axial), seule dépendance
restante de M1 ; puis correction de la tête vocabulaire et premier lancement du
27B.

## 2026-11-15 — J1 (suite) : W10 fusionné, premier lancement du 27B

**W10-lite** : branche fusionnée (merge `9ca90f7`) puis son suivi `efa1361`
(hc_up, absmax kv_store, borne embed, gate fusionnée). Le worktree local avait
été ramené à l'état pré-W10 par un checkout inattendu ; restauré depuis HEAD,
puis fusion du suivi. Build depuis l'arbre fusionné et `r3d_selftest` avec
libref : **142 rows, 0 échec**, 13 skips de shims. HEAD `1892845`.

**27B** : déclare sans aucune bande device, chargement **3,0 s**, 21,16 GiB sur
la carte, écoute sur 8100, `/v1/models` répond. La première complétion échoue
au premier pas :

```
E kernel rmsnorm_quant_fp8_bf16 (libr3d) refused op 'rmsnorm_quant_fp8' at M=5
  ... {n=5120 eps=1e-06 group=128 dtype=bf16 wadd=1 M=512}: invalid argument
```

Cause : le trunk du 27B est int8 ; l'architecture demande
`rmsnorm_had_quant_i8g`, que libr3d ne déclare pas, et retombe sur
`rmsnorm_quant_fp8` (E4M3), dont le shim refuse un q int8. C'est le seul
bloqueur M1 observé.

**Tentative** : composite `rmsnorm_had_quant_i8g` (norme fp8 puis
`had_quant_act_i8g`) avec schéma/row/scratch. Le startup échoue alors avec
« called in the wrong phase » (`RAD_E_STATE`, plan de buffers), sans trace de
builder ni d'arch. Changements annulés ; l'arbre est revenu à l'état vérifié
`1892845`.

**Prochaine étape** : implémenter proprement `rmsnorm_had_quant_i8g` (noyau
fusionné, ou composite via la row fp8 existante avec dispatch sur le dtype de
q), puis relancer le 27B et vérifier la tête vocabulaire.

## 2026-11-15 — J2 : W11, norme int8, fold FFN, premier texte du 27B

**`rmsnorm_had_quant_i8g`** : noyau fusionné (add résiduel → variance de la
somme écrite bf16 → gain → arrondi bf16 → Hadamard 128 → int8 row-major), row,
schéma et shape hook (q en I8), déclaré dans `r3d_plugin.h` et compilé comme
unité séparée. La norme est prise sur la somme écrite, comme la paire. Le
résidu est optionnel indépendamment de `wadd`, comme `rmsnorm_quant_fp8` et
libref.

Validation : `r3d_selftest --norm-only` compare le noyau à `add` + `rmsnorm` +
`had_quant_act_i8g` sur M=1/5/512, avec et sans résidu : **6/6 identiques au
bit** (codes, échelles, `out_bf16`). Selftest complet : **144 rows, 0 échec**,
14 skips (le nouvel op n'a pas de schéma dans libref).

**Fold FFN i8a8h** : la déclaration de la norme active `q8_fed` dans le bloc
MLP, qui demande alors `gemm_nt_q_gated` en i8a8h. libr3d ne l'avait qu'en
fp8a8 ; la seule row était libavx (host), ce qui faisait basculer toute l'unité
de couche sur le tier mapped et déclenchait le refus « widened at load » sur
les gains de norme f32 (le `RAD_E_STATE` de la veille). Ajout de
`gemm_iqr_gated_i8a8h` : `gemm_iqr_m16` calcule le gate_up bf16 dans le scratch
de l'op, puis `gated_had_quant_i8g` fait silu + rotation + int8. C'est la même
forme intermédiaire que libr4d stage dans `r4d_gemm_iqr_gated`.

**Convention d'échelle** : libr3d pliait `1/sqrt(128)` dans les quantiseurs
tournés `*_i8g` et `*_fp8` alors que docs/OPS et libref ne plient rien (le
1/128 est porté par le plan de poids, `libquant` stocke `scale/128`). Corrigé
pour `had_quant_act_i8g`, `gated_had_quant_i8g`, `had_quant_act_fp8`,
`gated_had_quant_fp8`, `rmsnorm_had_quant_i8g` et la tête IQR : échelle
`amax/127` (ou `/448`), groupe nul → 1, et le produit gaté est arrondi bf16
avant la rotation comme la paire non fusionnée. Deux bugs de shim corrigés au
passage : le plan `b_scale` d'une forme `block=1x*` n'est pas `[N, K/128]`
(extent privé, plus de refus de forme), et le scratch de la tête était
calculé avec `n_t` au lieu des seuls paramètres (donc jamais alloué).

**27B** : démarre, charge 21,17 GiB en 3,0 s, et génère « Paris.\nThe capital of
Germany is » en 8 tokens, **28,7 tok/s**, TTFT 83 ms. La tête tournait alors
sous `RAD_IQR_VOCAB=1` ; le désaccord de son test synthétique (0,507812 contre
-3,15625) venait de la **référence du test**, qui lisait la queue d'échelles au
mauvais offset (`align256(rows*4)` au lieu de `cf_off`) et appliquait le facteur
de ligne après coup : la référence validée `reference()`, celle que le test m16
contrôle, donne exactement 0,507812. Le test de la tête utilise maintenant cette
référence ; il est vert, et la variable d'environnement a été supprimée.

**W11 terminé** : la tête vocabulaire est servie par défaut et validée.

## 2026-11-15 — J2 (suite) : revue des changements et préfill > 16 tokens

Revue indépendante des changements non commités ; sept points corrigés.

- **IQR tiled** : le shim `r3d_rad_gemm_nt_q_iqr_tiled` gardait le contrôle
  d'extension exact sur `b_scale` que le m16 venait d'assouplir ; il refuse
  désormais lui aussi la forme `block=1x*`. Un cas M=33 a été ajouté à
  `iqr_device_test` (le tiled passe, 33 rows en 16+16+1).
- **Selftest de la norme** : les cas « sans résidu » liaient quand même le
  buffer résiduel ; depuis que le shim lit `res` dès qu'il est liée, ils
  échouaient (3 FAIL dans la dernière exécution). Le slot est maintenant
  absent quand `residual=false`, et un cas « résidu lié, `wadd=0` » pin la
  sémantique de libref. **7/7 cas verts, selftest complet RC=0, 144 rows,
  0 échec.**
- **gfx1102** : le FWHT du noyau fusionné tournait dans la boucle de groupes,
  le motif que le dépôt documente comme mal compilé sur gfx1102. Il passe
  désormais par une tranche LDS par wave (`fwht128_lds`, même réseau et même
  ordre d'opérandes que `r3d_fwht_wave`, donc bit-identique) : plus de DPP
  dans la boucle, tous targets.
- **Tête vocabulaire** : la référence du test lisait la queue des échelles au
  mauvais offset ; remplacée par la référence GEMM validée. Le test est vert et
  la variable d'environnement a été supprimée.
- **Gated IQR** : la description de row ne prétend plus économiser un launch
  device ; le fold garde les deux noyaux et le scratch (fusion d'épilogue en
  W12).
- **Shape hook gated** : `N % 256` (au lieu de 128) pour coller à la
  contrainte de row et au quantiseur.
- **out_bf16** : les quatre bf16 par lane sont écrits en deux dwords au lieu
  de quatre stores 2 octets, avec contrôle d'alignement.

**Préfill > 16 tokens** : les prompts de 15 et 16 tokens passent et génèrent
(« Paris. The quick », « city of Paris. », 28,4 tok/s). À 33 et 83 tokens, le
processus meurt en SIGSEGV ; `rocgdb` pointe
`r3d_rad_gemm_nt_bf16_tiled` (`rad_gemm_run`) appelé par `GdnFP8::step`. Le
chemin IQR tiled est validé séparément au M=33.

**Cause et correctif** : `rad_gemm_run` lisait les premiers octets de tout
poids comme un en-tête `rad.planes`, y compris pour un poids bf16 simple
(`ssm_ab`, [96, 5120]) placé en VRAM : la lecture hôte d'un pointeur device
provoque le SIGSEGV. Le peek d'en-tête ne s'applique plus qu'aux poids privés
(`w->dtype != RAD_BF16`), c'est-à-dire aux têtes quantisées, qui sont mappées
et lisibles par l'hôte. Vérifié : prompts de **33, 83 et 819 tokens** verts,
préfill 68–79 tok/s, décodage 21–28 tok/s. Le 819 tokens (« the Cook Islands
is ») traverse les bandes tiled M ≤ 64/127/512.

**Selftest après correctif** : RC=0, 144 rows, 0 échec, 14 skips ; norme 7/7.

## 2026-11-15 — W12 : préfill résident, gated fusionné, tête 16 bits, h256/vision

Cinq chants menés en parallèle dans des worktrees Agent Manager, revus par leurs
auteurs puis fusionnés sur `rdna3-iqr` (10 commits, `d4de0ca`→`d8418c1`).

- **`gemm_iqr_tiled` résident** : une tuile de 16×128 est décodée une fois par
  (tuile, bloc K) et réutilisée pour toutes les tranches de 16 tokens ; l'ordre
  numérique par (token, ligne) ne change pas, donc **bit-identique** au tiled
  précédent (cas M=33/160/512 ajoutés au test device). Préfill 819 tokens :
  **79,9 → 303,2 tok/s**, TTFT **10,25 → 2,70 s**.
- **Gated IQR fusionné** : decode gate/up, gate, arrondi bf16, Hadamard et codes
  en **un seul launch** (16 waves par groupe de 128 colonnes), sans scratch
  `[M, 2n]`. Test dédié : **codes, échelles et `out_bf16` identiques octet pour
  octet** à la paire m16 + `gated_had_quant_i8g` pour M=1/5/16, silu/sigmoid,
  avec et sans `out_bf16`. Gain mesuré ~1,5 % en décodage.
- **Tête IQR 16 bits** : activation `q = 128·hi + lo`, deux passes int8 exactes,
  repli f32 unique, **plus aucun intermédiaire bf16** ; la row garde son contrat
  (`vocab_offset`, classes 0..3, ids hors rang ignorés). Premier jet 3 % plus
  lent au décodage : les 16 lignes de la tuile LDS étaient lues au même banc à
  chaque `c4` (stride 32 dwords). Pitch de **33 dwords** + split hi/lo sur les
  32 lanes : décodage à parité (27,2 contre 27,3 tok/s à 16 tokens ; 20,8
  contre 20,4 à 819).
- **`attn_paged` h256** : page i8 staged en blocs de 16 octets, `nu` limite les
  slots visibles, passe max sans V. 15/15 identiques à libref.
- **`attn_dense` hd72** : tuilage LDS de K/V, chargements bf16 16 octets,
  élargissement exact et même ordre de dots (sortie identique ou dans la
  tolérance libref), garde `grid.y` et index de ligne 64 bits. 142 rows vision
  vertes.

**Validation** : `iqr_device_test` 4 cas/11 checks verts ; `r3d_selftest`
**144 rows, 0 échec, 14 skips**. 27B : mêmes sorties greedy que la base
(« the Cook Islands is Avarua… »).

**Attribution A/B** (décodage 16 tok, préfill 819) : base 27,3/79,9 ; sans gated
25,9/303 ; sans tête 27,0/300 ; final 27,2/303. Le préfill vient du résident,
le gated est un gain, la tête est neutre après le pitch.

**Pièges** : (1) un stride de 32 dwords en LDS met 16 lignes sur le même banc —
pitch impair obligatoire ; (2) un `build` neuf ne contient pas les plugins
`architectures/`, il faut construire la cible par défaut ou le serveur refuse
« no architecture plugin claims 'qwen35' » ; (3) chaque agent doit avoir son
propre répertoire distant et tout test GPU passe par
`flock /tmp/radiance-gpu.lock` ; (4) un lancement de benchmark par arbre, même
requêtes, mêmes lots.

**Reste à optimiser (W13+)** : cœur WMMA/IU8 du préfill, split-K `down`,
occupation du gated (1,4 bloc/CU), classes parallèles de la tête, mesure isolée
h256/vision, HC/PLE. Détail dans `HANDOVER_RDNA3_PHASE5.md` et
`TABLEAU_REEL_IDEAL_RDNA3.md`.

## 2026-11-15 — W13 : split-K, tête 16 bits, h256, vision, WMMA éteinte

Cinq chants en worktrees Agent Manager (le sixième, occupation du gated, n'a rien
livré), revus par leurs auteurs puis fusionnés sur `rdna3-iqr` (`f6be390`).

- **Split-K `down`** : huit tranches de K pour le GEMV M=1 à N ≤ 8192, partiels
  i32 par tranche, réduction en ordre croissant, scratch 256-aligné fourni par le
  hook de la row `gemm_iqr_nt_m64` (repli sans scratch si absent). GEMV
  5120×17408 M=1 : **104,0 → 67,3 µs, 428,5 → 662,4 GB/s** (49,5 → 76,6 % de 864).
  Décodage 27B : **+5,2 %** à 16/33 tokens. Test device dédié : 1 ulp bf16 sur
  1 sortie de 5120 (borne 2).
- **Tête 16 bits, un lancement** : les quatre classes passent en un seul grid
  (`blockIdx.z`), repli f32 factorisé, `M=1` hi/lo sur 32 lanes et **pitch 33** ;
  test **bit-exact M=1 vs M=2** ajouté, classes à comptes inégaux couvertes.
- **`attn_paged` h256** : cache LDS des scores de la passe max (5120 f32) et
  pitch de 272 octets pour étaler les bancs ; la passe deux ne rejoue que V si le
  cache tient, sinon elle recalcule le dot dans le même ordre. 15/15 identiques
  à libref.
- **`attn_dense` hd72** : quatre clés par pas dans la marche (chaînes
  indépendantes, ordre inchangé), et test M=16384 à segments uniformes
  (18 874 368 valeurs exactes). kbench : **684,7 → 350,7 µs à M=512 (−49 %)**,
  342 → 180 à M=256, 176 → 94 à M=128, 92 → 49 à M=64 ; M≤16 au plancher.
- **Tuile WMMA du préfill** : construite et bit-identique, mais **25–32 % plus
  lente** que le `v_dot4` résident sur builds propres (12288×2560 M=512 2801 contre
  2083 µs ; M=4096 23 110 contre 17 410 ; down 5120×17408 M=512 7387 contre
  5683). Le préfill 27B tombe de 301 à 231 tok/s avec elle. Elle reste dans
  l'arbre, **éteinte** (`kIqrWmmaMinM = 1 << 30`), marquée W14 : le problème est
  l'émission/le staging, pas la résidence.

**Validation** : `iqr_device_test` 6 cas / 27 checks verts (dont split 1 ulp,
tête bit-exacte M=1/M=2, vision M=16384 exacte) ; `r3d_selftest` **144 rows,
0 échec, 14 skips** ; 27B mêmes sorties greedy.

**Mesures 27B, builds propres, 3 lots** : préfill 16/33/819 = 103,4 / 122,1 /
**301,0** tok/s (W12 : 103,1 / 122,0 / 299,7) ; décodage 16/33/819 = **28,4 /
28,2 / 21,2** tok/s (W12 : 27,0 / 26,8 / 20,7) ; TTFT 819 2,72 s.

**Piège majeur** : un `rsync -a` dans un répertoire de build existant préserve
les mtimes et peut laisser des objets périmés ; les A/B incrémentaux ont d'abord
annoncé un préfill à 229 tok/s, puis un build **neuf** a redonné 301. Toute
comparaison se fait dans un build neuf, avec vérification du binaire du serveur
(`/proc/<pid>/exe`) et du port 8100 libre. Les .so sont écrits directement dans
`radiance_home/kernels` (`LIBRARY_OUTPUT_DIRECTORY`), donc un serveur lancé
pendant un build charge un mélange.

**Reste (W14)** : WMMA (émission/staging), occupation du gated fusionné, mesure
`attn_dense` à M=16384 dans kbench, mesure isolée h256, HC/PLE, E4M3 non-IQR.

## 2026-11-16 — W14 : audit Flash-Next, QSA, MoE IQR, formes

Huit chants en worktrees Agent Manager, revus par leurs auteurs puis fusionnés
sur `rdna3-iqr`.

- **Audit Flash-Next** (`FLASH_NEXT_AUDIT.md`) : le conteneur mélange treillis
  `kTrellisMad` (gate_up, u3/u4/u5), codebook `w4nl` (tous les `down`, 25 590
  poids), **`kSigned` 8 tourné** (k/v/output, `ssm_out`, partagés, PLE) et bf16
  protégé ; il demande les cinq QSA, `gdn_conv_recurrent_update_had`,
  `kv_store` block 4, `rope` imrope h128, `router_gemm_topk_scatter`,
  `gemm_nt` N=1, et impose `max_mixed_tok ≤ 2048` (plan de scores).
- **QSA** : `qsa_work`, `qsa_block_key`, `qsa_score`, `qsa_select` (en plus de
  `tail_store`) déclarés et testés contre libref — 529, 5 504, 781 et 781
  valeurs/checks identiques, 7/16/14 cas.
- **MoE IQR** : row `moe_gemm_iqr` (`libr3d/r3d_moe_iqr.hip`), run map triée,
  bande 16 lignes et préfill, classes paires/impaires, scratch ; `moe_iqr_device_test`
  15/15 vert. La **référence hôte** (`tests/iqr_ref.h`, `tests/moe_iqr_test.cpp`)
  est écrite mais la comparaison carte échoue (`RAD_E_UNSUPPORTED`, écarts
  top-k/parts) : agent d'intégration en cours (R15).
- **Formes denses** : `kTable` (u2..u5) et `kGrid` (V=4) acceptées, échelles
  bf16, tests contre `reference()` ; le treillis reste bit-identique. Ce n'est
  pas encore la forme `w4nl` du conteneur.
- **Revue W12** : pitch gated MT>1, borne `max_seqlen` vision, store V h256.

**Validation** : `iqr_device_test` et les quatre tests QSA verts ; `r3d_selftest`
**151 rows, 0 échec, 14 skips** ; `moe_iqr_test` en échec (voir R15).

**Lancement W15** : `kSigned` 8, `gdn_conv_recurrent_update_had`, `kv_store`
block 4, `rope` h128 imrope, router fusionné, `gemm_nt` N=1, rows vision
(`gemm_nt_bias`, `grid_embed`), et réconciliation MoE ; `moe_gemm_w4nl` suivra.

## 2026-11-16 — W15 : rows Flash-Next

Huit agents en worktrees, revus par leurs auteurs puis fusionnés sur
`rdna3-iqr` (`2177619`). `moe_gemm_w4nl` reste en cours.

- **`kSigned` 8** : `RAD_I8 → kSigned, B=8`, `X(kSigned,8,0,0,0,kBf16)` dans
  `R3D_IQR_FORMS`, acceptation dans les trois familles denses, échelle bf16 par
  128, codes i8 directs ; le treillis reste bit-identique. Débloque k/v/output,
  `ssm_out`, partagés et PLE du tronc Flash-Next.
- **`gdn_conv_recurrent_update_had`** : row device, conv + recurrent + decode i8
  tourné, testée contre la paire non fusionnée sur quatre séquences, fenêtre
  pleine et groupe 256 (46 checks).
- **`kv_store` block 4** : i8, head_dim 256, n_head_kv 2, M=4096, pages
  partielles, block 16 bit-identique (test device dédié).
- **`rope` imrope h128** : sections 11/11/10, rotary 64, 2 097 152 valeurs
  exactes à M=4096, h256/h72 inchangés.
- **Router fusionné** : GEMM + top-k + tri en device, `protect`, égalités
  déterministes, 38 checks.
- **`gemm_nt` N=1** : row GEMV bf16 dédiée, sélection et refus testés.
- **Vision** : `gemm_nt_bias` et `grid_embed` en device, 66 checks, 142 rows
  vision toujours vertes.
- **MoE** : `moe_iqr_test` réconcilié — le test demandait une classe impaire
  plus étroite que la row refuse par contrat ; plans à classes égales, refus
  épinglé. 10 formes, 0 écart (tolérance 2 ulp), `moe_iqr_device_test` 15/15.

**Validation** : `iqr_device_test`, les quatre tests QSA, `moe_iqr_device_test`,
`moe_iqr_test`, `kv_store_b4`, `rope` h128, router, `gemm_nt` N=1, vision et
GDN had verts ; `r3d_selftest` **158 rows, 0 échec, 15 skips**.

**Reste** : `moe_gemm_w4nl` (25 590 poids `down` + MTP 48), puis démarrage réel
de Flash-Next avec `num-speculative-tokens: 0`, `max_mixed_tok ≤ 2048`,
`expert_tiered`, et les mesures HC/PLE.

## 2026-11-16 — `moe_gemm_w4nl` et premier démarrage Flash-Next

- **`moe_gemm_w4nl`** : forme `w4nl64a8i8h` (codes u4, table f32[16] vérifiée
  valeur par valeur, échelle E4M3 par 64 sous niveau f32 2^−13), row au run map
  du MoE IQR, hooks de layout/relayout dédiés, MTP 48 couvert (80 slots,
  48 experts, N=1280 K=2560 : 102 400 valeurs bf16 exactes), `a_group` 128/256.
  `r3d_selftest` : **159 rows, 0 échec, 15 skips**.
- **Déclaratif Flash-Next** avec `libr3d` seul : 5 074 questions,
  **0 bande device non résolue** ; les 372 non résolues sont des replis hôte.
  Le serveur démarre en 21 s, 44,35 GiB par carte, `expert_tiered`.
- **Premier pas refusé** : `hc_read_e4m3` refuse `M=5` alors que la bande est
  `M≤2048` (`shape this kernel does not serve`), et le moteur s'arrête. Agent
  de bring-up dédié pour identifier la garde exacte et corriger la row (ou
  déclarer la contrainte pour retomber sur la row hôte), puis les refus
  suivants jusqu'au texte 1/4 séquences.

## 2026-11-16 — Flash-Next servi

L'agent de bring-up a corrigé les refus runtime successifs :

- `hc_read` : la garde `numel` des poids E4M3 comparait le nombre de codes au
  row joint ; remplacée par la forme logique déclarée (`hc_dims`).
- `moe_gather`/`moe_gather_inv` : le `M` de la bande était pris pour le nombre
  de lignes du pas ; il vient maintenant de la sortie.
- `moe_iqr` : les tables d'arguments étaient partiellement non initialisées
  (le moteur compare les octets d'un lancement enregistré) ; `memset` ajouté.
- Router et tête vocabulaire : formes de pas et plan de tête dérivé du
  déclaratif au lieu d'un peek `rad.planes` sur un poids VRAM.

**Résultat** : Flash-Next sert le texte. 1 séquence : 30 tok → 20,6 tok/s de
préfill, 4,6 tok/s de décodage, TTFT 1,46 s ; 2 290 tok → 42,1 tok/s, 4,0,
54,3 s ; 4 570 tok → 41,5 tok/s, 4,0, 110,2 s. 4 requêtes concurrentes de 30
tokens : 4,0–4,5 tok/s, TTFT 1,68–3,38 s. `r3d_selftest` 159/0/15.

**kbench de contrôle** sur le conteneur (495 s) : 1 743/1 749 cas corrects,
**6 FAIL `logits_gemm_bf16`** (n_vocab 248 320, n_embd 2 560, M=1..32, rel_l2
0,50–0,53) et **22 divergences de forme QSA** (`qsa_select.nsel` en rang 2,
`qsa_work.nc` `[8]` au lieu de `[seqs]`, `qsa_tail_store.state` `[8,…]` au lieu
de `[M,…]`). Les timings par row ne sont pas exploitables : la carte a throttlé
(PPT0/PPT1) pendant la mesure.

Deux agents W17 corrigent la tête bf16 et les formes QSA ; re-vérification
kbench attendue à 0 FAIL et 0 divergence.

## 2026-11-16 — W17 : corrections kbench, Flash-Next conforme

- **`logits_gemm_bf16`** : la tête Flash-Next stocke une échelle par ligne et
  128 de K (`i8*bf16[1x128]`) ; le chemin du plan l'indexait comme une grille
  128×128, donc toutes les lignes au-delà de 128 empruntaient les échelles d'un
  autre bloc (rel_l2 0,50–0,53). `HeadPlan`/`HeadQ` portent désormais
  `(sbr, sbc)`, les têtes 128×128 gardent leur indexation exacte, et les blocs
  non positifs ou K non multiple de 128 sont refusés. Nouveau
  `tests/logits_bf16_device_test.cpp` (M=1/2/8/16/32, forme à blocs, tête bf16
  simple, refus). rel_l2 retombé à **1,7e-3**.
- **QSA** : `nsel`/`seqused` en rang 1, `qsa_work.nc` en `[S]`,
  `qsa_tail_store.state` en `[M,1,ring,n]` ; le test QSA compare maintenant
  toutes les descriptions à libref.
- **kbench final** (arbre fusionné, `--kernels libr3d:libref`) :
  **856/856 cas, 0 échec, 0 divergence**, pire rel_l2 1,2e-2.
- **Robustesse** : un prompt de 12 001 tokens est refusé avec un HTTP 400 clair
  et le serveur reste vivant ; les redémarrages sont propres.
- **Validation** : `r3d_selftest` 159/0/15, `iqr_device_test`,
  `logits_bf16_device_test`, les quatre tests QSA, `moe_iqr_test` et
  `moe_w4nl` verts.

Reste non mesuré : HC/PLE, MoE/QSA par row et η lien, à prendre sur carte froide
(le kbench de 8 min throttle en PPT0/PPT1).

## 2026-11-16 — W18 : mesures à froid, `hc_read` identifié

Runs `rad-kbench --kernels libr3d:libref --bench` sur
`<work-root>/work/w15-main` (code `a772d70`), carte au repos, lots de 1 ms,
`--max-cases 1..3`, M jusqu'à 2 048. Horloge 2 882–2 894 MHz, **aucun
throttler** sur HC/PLE ; la passe QSA a tourné à 2 789 MHz (PPT0 au warm-up).

- **HC** : `hc_read_e4m3` (n 2 560, hc 4, lr 320) = **1,77 ms à M=1** pour
  6,55 Mo de poids (3,9 GB/s, **η ≈ 0,6 %** de la borne bande de 10,5 µs) et
  **470 ms à M=2 048**, soit 0,23 ms par ligne ajoutée : la paire de matrices
  est relue par ligne à ~28 GB/s au lieu d'être amortie par tuile. À 48 appels
  par pas (96 matrices) : **~93 ms/token** en décodage (40 % du pas) et **22,9 s
  d'un préfill de 2 048 tokens** (47 % des 48,6 s mesurés, plafond 89 tok/s).
  Le commentaire de la row annonce 18,7 µs au décodage : écart ×95 à profiler
  côté engine. `hc_enter`/`hc_write` sont au plancher de lancement ; `hc_project`
  n'a pas de bande.
- **PLE** : `ple_gate` 23,1 µs à M=1 (plancher) et 69 TFLOP/s à M=2 048 ;
  `ple_conv` 5,8 µs à M=1, lié aux activations (160 GB/s à M=2 048).
- **QSA** : `qsa_score` 119 µs à M=1, **83,4 ms à M=512** (103 GFLOP/s, 2 % du
  pic, 258 Mio d'opérandes) ; `qsa_select` 73 µs/946 µs ; les autres rows sont
  au plancher. 147 cas, 0 échec, rel_l2 0.
- **MoE — bloqué** : `kbench` refuse de poser les tables du conteneur
  (`moe_gemm_iqr` : « container format error » sur `trellis:…/fwht128` ×50 ;
  `moe_gemm_w4nl` : « shape this kernel does not serve » sur `affine:…/fwht128`
  ×60). Seuls `moe_gather`/`moe_scatter` (poids synthétiques) sont couverts. Le
  temps par row MoE et η lien par noyau restent donc hors de portée de kbench
  sur cet artefact.
- **η lien (M2)** : 42 tok/s mesurés contre 2 545 tok/s de borne → **1,7 %**,
  critère des 80 % non tenu ; l'attribution montre `hc_read` comme plafond,
  pas le MoE.
- Vérification : 182 cas HC/PLE et 147 cas QSA, **0 échec** (pire rel_l2
  1,5e-2 sur `hc_read`).
- **A/B de la forme de décodage** : arbre neuf `w18-hc2grid` avec
  `g_hc_decode_form = 1` (les deux noyaux deux-grilles au lieu du fused à
  tickets). Résultat identique (M=1 : 1 782 / 1 778 µs ; M=2 048 : 471 / 477 ms),
  0 échec. L'attente par tickets n'est donc pas la cause ; le coût est dans le
  corps du read (lecture/décodage des matrices). Le binaire est **jetable** et
  `w15-main` reste la référence.
- Piste suivante pour `hc_read` : instrumenter le read seul (compteur de
  transactions ou micro-benchmark dédié) — chaud et froid donnent le même temps,
  donc ni la bande passante ni la résidence ne l'expliquent.

## 2026-11-16 — W19 : démo serveur M6

Serveur `w15-main`, configuration v1 (`TP=1`, `NS=0`, `MLEN=8192`,
`HOSTPOOL=43000`, `KVHOST=1024`, `MNBT=2048`, `WIRE=wht6`), poids 41,12 GiB,
KV 1,67 GiB.

- **Prompts distincts** (71–113 tokens) : préfill 38,7–41,9 tok/s, décodage
  4,55–4,59 tok/s, TTFT 1,8–2,7 s (égal au préfill, pas de coût fixe visible).
- **Séquences** : 1 → 4,41 tok/s ; 4 → 3,86–3,93 chacune (15,5 tok/s agrégés) ;
  6 → 3,66–3,74 chacune (22,1 tok/s agrégés).
- **Cache de préfixe** : identique 4 468 tokens, 2<sup>e</sup> envoi **4 096
  servis**, TTFT 108,4 → 9,0 s ; même préfixe + 1 354 tokens → 4 096 servis,
  42,4 s ; conversation qui grandit → 2 048 servis par tour (58,9 → 19,7 →
  29,6 s). **Rejeu agent (5 tours, sous-agents) : 0 % servi**, 1 checkpoint
  écrit, `no_snapshot` ×4, alors que l'attention détenait 4 268 tokens.
- **Tier KV hôte** : refusé au démarrage (`no device gather/scatter for group 1
  at a 4128-byte fragment`), donc une conversation finie ne quitte pas la VRAM ;
  le cache est en VRAM seule. À corriger pour le multi-tour long.
- **Contention** (`fnconc`, 4 décodeurs × 48 tokens, préfill ~3 000) : 12,6 s
  seuls → **87,6 s** à côté du préfill (+590 %) ; préfill 35 tok/s (vs 42 seul).
- **Robustesse** : 12 002 tokens → 400 en 0,00 s avec le nombre de tokens à
  couper ; `max_tokens: 20000` plafonné à 8 191 (pas de 400) et `finish=abort`
  à la déconnexion, slot rendu ; abandon à 3 s puis requête servie en 3,57 s ;
  6 admissions 6/6 ; `/health` 200 et 0 requête en vol à la fin.
- Limites M6 : image/vision (`capabilities.image=false`) et cache de préfixe
  au-delà de 2 048 tokens restent hors démonstration.
- **16k** : serveur relancé avec `MLEN=16384 SEQS=4` (KV 1,23 GiB, 65 596
  tokens, 4 × 16 384, 24 snapshots). Prompt de **13 421 tokens accepté** en
  328,5 s (40,9 tok/s, 8 tokens générés) ; prompt de 18 756 tokens → **400
  immédiat**. Le débit de préfill ne dépend donc que du nombre de tokens, pas de
  la longueur du contexte réservé.
- **R9** : inventaire de l'hôte vérifié (27B IQR + Flash-Next IQR seulement) ;
  la validation E4M3/rotated sur un modèle non-IQR reste impossible, dérogation
  documentée dans `SUIVI_LOTS_RDNA3.md` §R9.
- **27B rejoué** (R3) : chargement **15,87 GiB en 2,7 s** (6,2 GB/s, dont 1,7 s
  de disque), prêt en 3,3 s, 17,57 GiB tenus. À `max-num-seqs 1`, `kv i8`,
  `NS=0` : préfill **318,8 tok/s** à 844 tokens, décodage **23,2 tok/s** (16) et
  **29,4** (64) ; 4 requêtes simultanées se sérialisent (5,41 s). Sans `NS=0`, le
  drafter h128 fenêtré réclame bf16 et le déclaratif refuse.
- **R24 trouvé** : `max-num-seqs 4` sur le 27B **échoue au premier pas** —
  `logits_gemm_vocab_iqr` (`libr3d/r3d_iqr_logits.hip:362`) exige que le `M` des
  args égale le nombre de lignes de `x` ; le moteur passe le `M` de bande (4)
  pour un pas d'une ligne, d'où `RAD_E_SHAPE` et l'arrêt du modèle. Même classe
  que le refus `hc_read M=5` corrigé en 4bis.2b ; corrigé en W20 (ci-dessous).
- **Outil pour la suite** : `--profile-ops true` donne le temps device par op
  (le log en donne la réserve) ; c'est le moyen prévu pour trancher R20
  (`hc_read`) sans toucher au noyau.

## 2026-11-16 — W20 : contrat capacité/étendue (R24) corrigé

Décision : le paramètre rangé (`M`) est une **capacité de bande**, pas le nombre
de lignes du pas. Le moteur prépare les args une fois par (op, bande, domaine) et
y laisse le plafond ; l'étendue est portée par les tenseurs (`o.rows`, narrow à
l'issue), et c'est ce que fait déjà `libr4d` (`r4d_logits_vocab.hip:408-412` :
`M = x->shape[0]`, sortie `>= M`). Trois corrections, plus un mode de test.

1. **`logits_gemm_vocab_iqr`** (`libr3d/r3d_iqr_logits.hip`) : `M` vient de `x`,
   la capacité n'est plus qu'un plancher (`band_M < M` → refus). C'est le bug
   R24 : un pas d'une ligne dans une bande `M ≤ max_logit_rows` était refusé.
2. **Audit de la classe** : `--band-capacity` (nouveau mode `rad-kbench`, voir
   `AGENTS.md`) issue chaque cas avec `M = plafond de bande` et les opérandes à
   la taille du point. Sur Flash-Next il a trouvé une **seconde instance** :
   `router_topk`, `router_topk_scatter` et `router_sigmoid_topk` lisaient `M`
   dans les args et lançaient `grid(M)` — lignes en trop lues/écrites (le
   rapport le voit comme 8 échecs et un débordement de 256 octets). Corrigés :
   lignes prises sur `L`, contrôle de capacité, et vérification que `I`/`W`
   (et `B`, `ST`) couvrent `M*top_k`.
3. **Test de non-régression** : `tests/iqr_device_test.cpp`
   `iqr_vocab_head_accepts_a_narrow_step_in_a_wide_band` appelle la row avec
   `M = 1, 3, 2` dans des capacités 4/4/8, vérifie la justesse contre la
   référence f64, **et** que les lignes au-delà du pas restent NaN (la capacité
   ne doit pas être écrite). Contre l'ancienne `.so` : 5 échecs (`rc=-1`) ;
   contre la nouvelle : vert.

Validation (arbre neuf `w20-r24`, ccache, build 1 min 27) :
`r3d_selftest` **159 rows, 0 échec, 15 skips** ; `iqr_device_test` 67 checks
verts ; kbench Flash-Next par défaut **856/856, 0 échec, 0 divergence, pire
rel_l2 1,206e-2** (identique à la référence) ; kbench Flash-Next
`--band-capacity` **856/856, 0 échec, 0 débordement** ; 27B kbench **605/605**
dans les deux modes (pire rel_l2 3,619e-3, identique) ; 27B à
`--max-num-seqs 4` : préfill 317,5 tok/s, décodage 23,1/29,3 tok/s, **4 requêtes
concurrentes 67,3 tok/s agrégés** (contre 29,4 sérialisées à `SEQS=1`).

## 2026-11-16 — W22a : tier KV hôte (R21) activé

Le tier refusait de se configurer sur Flash-Next : la row `gather_rows`/`scatter_rows`
demandée pour le groupe `kv_qsa_bkey` (fragment de 4 128 octets) n'existait
qu'en bf16/f32, dont `ld_elem`/`st_elem` font passer chaque octet par le codec
E4M3 — un payload NaN ne survit pas à l'aller-retour. Le tier déplace des
fragments **opaques**, pas des valeurs.

1. **Rows u8** (`libr3d/r3d_rows.cpp`, `r3d_model_bf16.hip`) :
   `gather_rows_u8` et `scatter_rows_u8`, un dword par thread, contrainte
   `dtype == u8` et `n % 4 == 0` ; `n` et les deux pitches sont des octets, et
   le lanceur refuse un pitch qui n'est pas un dword entier. Index négatif :
   ligne zéro au gather, rien au scatter ; index au-delà du plan : idem, aucune
   lecture/écriture hors du plan. Le shim branche sur `dt == R3D_DT_U8` avant
   le chemin élément.
2. **Test** : `tests/gather_rows_u8_device_test.cpp` — 6 géométries par row,
   M = 1/8/2 048, fragment 4 128 octets et petites lignes, pitches larges,
   toutes les valeurs d'octet, indices -1 et hors plan, destination du scatter
   partiellement écrite. Oracle local octet à octet (pas libref, dont le chemin
   u8 passe par float) ; contraintes et description d'opérandes vérifiées.
3. **Selftest** : deux géométries u8 (M = 8, n = 4 128) ; **161 rows, 0 échec,
   15 skips**.

Validation (arbre neuf `w21-kvtier`, build complet) : `gather_rows_u8_device_test`
33 checks verts ; `iqr_device_test` 67 verts ; `logits_bf16_device_test` 8 verts ;
`kv_store_b4_device_test` 42 verts. kbench ciblé `gather_rows`/`scatter_rows` :
27B **48/48** et Flash-Next **84/84**, 0 échec, 0 divergence.

Tier en service, `RADIANCE_DEBUG_TIER_VERIFY=1` (hash des octets après gather et
après scatter) :
- Flash-Next (`--prefix-cache-mib 1024`, 8k) : **2 groupes paginés**, 51,38 Kio
  par entrée, lots de 318 ; requête de 25 tokens servie ; 8 copies vers l'hôte,
  0 échec.
- 27B (même réglage, 4k, `max-num-seqs 1`) : 1 groupe, 516 Kio par entrée, lots
  de 64. Conversation A de 3 899 + 3 927 tokens, conversation B de 3 562
  tokens pour évincer A, puis A au tour 3 : `to_host` 471, `dropped` 586,
  `promoted` **245**, `restore_cuts` 0, `failures` 0 ; tours 2 et 3 servis avec
  **2 048 tokens cachés** chacun. La copie et la restauration passent donc par
  les deux rows, et le hash confirme les octets.

Reste de W22 : R22 (snapshots en rejeu multi-agents), inchangé.

## 2026-11-16 — W22b : snapshots des tours courts (R22)

**Cause racine.** Sous `--prefix-resume exact` (défaut), la grille de reprise
vaut l'intervalle (2 048) à partir de 0. `Scheduler::tail_checkpoint()` ne
rendait un snapshot que si `grid_floor(target − 1) > intervalle`, donc **tout
prompt sous 2 048 tokens n'écrivait aucun checkpoint**. Le tour suivant
matchait ses blocs d'attention mais n'avait pas de slot d'état linéaire
(`checkpoint_slot < 0`), d'où `reuse = 0` et `HitCut::NoSnapshot`. Ce n'était
pas la rétention : 1 checkpoint **écrit** pour 9 tours.

**Correctif.** Ancre de queue (branche `rdna3-r22-snapshots`, commit
`7188fe2`) : pour `target ≤ intervalle`, snapshot à la dernière frontière de
quantum ; si elle n'est pas une frontière froide (multiple de l'intervalle ou
point de grille), reculer d'un quantum pour qu'un client qui re-rend les
derniers tokens reste sur le préfixe haché. Seuil `kShortTailMin = 256` tokens
d'ancre, seulement si le cache de préfixe est actif. Le préfill froid coupe à
l'ancre (chunk final court) ; le tour suivant reprend là. Au-delà du premier
intervalle, `exact` reste strictement sur la grille.

**Mesures** (Flash-Next, `max-model-len 8192`, KV i8, `prefix-cache-mib 1024`,
`max-num-seqs 1`, GPU sous `flock`, même graine ; rejeu
`agentreplay.py --parents 1 --turns 5 --system 600 --turn 400,700 --reply 32
--sub-every 2 --sub-turns 2 --seed 3`, 9 tours / 17 k tokens) :

| | avant (`06226d1`) | après (`7188fe2`) |
|---|---|---|
| tokens servis | 4 096 / 17 027 (24 %) | **9 088 / 17 265 (52,6 %)** |
| préfill recalculé | 12 931 | 8 177 (−37 %) |
| `no_snapshot` | 6 | **2** |
| checkpoints écrits | 1 | **7** |
| TTFT parent / après sous-agent / sous-agent suivant | 31,8 / 51,6 / 32,7 s | 22,9 / 28,9 / 10,6 s |

Les 2 `no_snapshot` restants sont les premiers tours des sous-agents (rien à
reprendre avant leur propre ancre) ; leurs tours suivants servent 960 et 896
tokens.

**Validation.** `sched_test` 548 782 checks / 196 cas, 0 échec ;
`engine_e2e_test` complet 0 échec ; `iqr_device_test`,
`logits_bf16_device_test`, `gather_rows_u8_device_test`,
`vision_rows_device_test` et `r3d_selftest` (161/0/15) verts. Tests ajoutés :
ancre planifiée sous `exact` et `nearest`, reprise sans arrondi à la grille,
propriété « tout checkpoint au-delà du premier intervalle est un début de
chunk froid », et un test e2e de trois tours de 600 tokens dont le tour 2
sert ≥ 256 tokens avec `no_snapshot = 0` (vérifié rouge sans le correctif).

**Compromis et reste.** Tout prompt de 256…2 048 tokens paie un snapshot et
souvent un chunk final court ; sous le premier intervalle la reprise est hors
grille (comme `nearest`), donc l'accord bit à bit n'est plus garanti pour un
tour appendu court — un prompt répété reste exact, et au-delà de l'intervalle
`exact` reste strict. Non couvert : premier tour d'un sous-agent (pas d'ancre
partagée), tours réécrits dont la divergence précède l'ancre.

## 2026-11-16 — décision de périmètre : vision et MTP de Flash-Next dans la v1

Décision utilisateur, consignée dans **`docs/adr/ADR-0007-v1-vision-and-mtp.md`**
(qui supersède l'ADR-6 et complète l'ADR-5) :

- **Vision incluse.** La tour est portée depuis W10 et complétée par W15
  (`gemm_nt_bias_bf16`, `grid_embed_bf16`) ; ses rows sont vertes. Le blocage
  restant n'est pas un noyau mais le décodage média : le build servi n'a pas
  `RAD_WITH_FFMPEG` (`capabilities.image=false`). Lot **W26** : rebuild
  `-DRAD_WITH_FFMPEG=ON`, corpus d'images, un tour image de bout en bout,
  `capabilities.image` à vrai. L'artefact sans média de l'ADR-5 reste permis
  comme artefact de test, il ne satisfait plus la cible.
- **MTP de Flash-Next inclus.** Le plugin `qwen4exp_fp8` déclare `mtp_enter`,
  `mtp_couple_propose` et `logit_rerank` dès `num-speculative-tokens > 0` ;
  `libr3d` n'a de row device ni pour `mtp_enter` ni pour `logit_rerank`
  (`libr4d`, `libavx` et `libref` les ont). Lot **W25** : écrire les deux rows
  sur les contrats de `libr4d`, les tester, servir à `NS > 0`, et couvrir les
  cas MTP 48 de kbench — qui deviennent un critère de clôture au lieu d'un
  « Not covered ». Les experts du MTP 48 (`w4nl` `u4`) sont déjà servis par
  `moe_gemm_w4nl`.
- **27B inchangé** : `num-speculative-tokens: 0` et KV i8, son drafter h128
  fenêtré exigeant un KV bf16. La décision porte sur le MTP de Flash-Next.
- `ngram_hint` reste sans oracle : son « Not covered » n'est pas une
  régression ; celui du MTP 48 en est un désormais.

Conséquences pratiques : `num-speculative-tokens: 0` n'est plus « la
configuration v1 » mais le repli jusqu'à W25 ; la clôture gagne deux critères
(8.8 vision servie, 8.9 MTP servi et couvert). Les docs de suivi (checklist,
suivi des lots, handover, audit, plan, rapport) portent la révision.

## 2026-11-16 — W26 : vision servie (R26)

FFmpeg 8.0.1 et les paquets `libav*-dev` sont installés sur l'hôte (avformat
62.3.100, avcodec 62.11.100, swscale 9.1.100 ; `pkg-config` les voit). Arbre
neuf `w26-vision`, build complet avec `-DRAD_WITH_FFMPEG=ON` ; l'exécutable
lie `libavformat.so.62`, `libavcodec.so.62` et `libswscale.so.9`.

**Configuration** — celle de W19, sans changement de code ni de réglage :
`placement: expert_tiered`, `host-pool-mib: 43000`, marge 96 Mio, KV i8,
`NS=0`, `max-model-len 8192`, `max-num-batched-tokens 2048`,
`--mm-max-patches` reste `auto`. Le démarrage décide :

```
I placement: encoder towers (856.68 MiB a rank) vram (default)
I media staging: 10.00 MiB a rank for a step's encoder rows and 48.27 MiB for an encoder pass of up to 16384 patches
I media: images and video through 'vision'; an image resizes to at most 4194304 pixels (16384 patches a pass)
```

Carte : 44,35 GiB tenus sur 44,98 (la tour est dans les poids statiques) ; le
pool hôte reste à 20,02 GiB réservés. `capabilities.image=true` dans
`/v1/models` (la vidéo est aussi annoncée).

**Tours servis** (corpus `<work-root>/images`, PNG générés avec ffmpeg ;
requête `chat/completions` en `data:image/png;base64`, `enable_thinking:false`) :

| tour | tokens de prompt | dont image | TTFT | réponse |
|---|---|---|---|---|
| texte, 234 tok | 234 | — | 5,99 s | référence |
| image 512² | 281 | 256 | 8,19 s | « three solid-colored squares — red, green, and blue — … light gray background », positions exactes |
| image 1024² | 1 045 | 1 024 | 28,73 s | dégradé « deep navy blue … warm golden orange », exact |

Le préfill texte tourne à ~39 tok/s ; la différence de TTFT donne ~1,0 s de
tour pour 1 024 patches et ~1,9 s pour 4 096 (estimation, pas de profil par
op). Un premier essai sans `enable_thinking:false` avait consommé les 192
tokens en raisonnement : la perception était déjà juste, mais le contenu était
vide (`finish=length`).

**Non-régression** — `r3d_selftest` 161/0/15 ; `iqr_device_test` 67 checks,
`logits_bf16_device_test`, `gather_rows_u8_device_test` et
`vision_rows_device_test` verts ; `kbench` Flash-Next
`--kernels libr3d:libref --max-cases 1` **856/856** en défaut et en
`--band-capacity`, 0 divergence, pire rel_l2 1,206e-2 (identique à W20) ; run
complet 2 189/2 189 et ciblé vision **480/480**, 0 divergence.

Reste non testé : la vidéo et l'audio (annoncés par le processeur).

### W26 — robustesse du chemin média

Campagne de 18 cas dans une seule session serveur (même build, même recette,
sous `flock`) : aucun crash, aucun 5xx, aucune fuite — **VRAM constante à
45 896 MiB** sur toute la campagne, rendue à l'arrêt, texte de contrôle
identique avant/après.

- **Bord haut** : 2048² (16 384 patches, un pass) → 4 098 tokens image
  (4 096 + 2 marqueurs), TTFT 121,5 s. 3000² (9 Mpx) **n'est pas refusée** :
  elle est redimensionnée à 2048² (mêmes 4 098 tokens, 121,9 s).
- **Plusieurs images** : 2 × 512² → 516 tokens, 15,1 s ; 2 × 1024² →
  2 052 tokens, 58,5 s ; les deux images sont distinguées.
- **Ordre, formats** : texte→image et image→texte donnent le même prompt ;
  PNG RGBA (alpha) et JPEG décodés, descriptions correctes.
- **Erreurs** : base64 invalide, `data:text/plain`, PNG tronqué, type de part
  inconnu → 400 en 0,0 s avec champ fautif, aucun travail GPU ; une requête
  valide juste après passe (200, 7,8 s).
- **Carte** : edge 36→68 °C, hotspot 42→84 °C, mémoire 46→80 °C, 22–158 W,
  gfx jusqu'à 2 896 MHz, 100 % d'utilisation pendant les passes. Le champ
  `THROTTLE_STATUS` d'`amd-smi` vaut `THROTTLED` même au repos sur cet hôte :
  il n'est pas exploitable ; la puissance et la fréquence le sont.
- **À documenter** : redimensionnement silencieux au-delà de 4 194 304 px ;
  12 avertissements de watchdog (`core/engine.cpp:1307`, seuil 20 s) pendant
  les passes longues, toutes abouties ; coût d'un pass complet ~121 s
  (≈34 tok/s de préfill, ≈135 patches/s) puis ~4 tok/s de décodage ;
  `decode_media_url` reconnaît le format aux octets et ne vérifie pas le type
  MIME déclaré ; 2 × 2048² (32 768 patches) non testé.
- Aucun défaut de contrat à corriger : les deux améliorations possibles sont
  de diagnostic (une ligne de log par image : taille après resize, patches,
  tokens ; un message de watchdog adapté aux passes média).

## 2026-11-16 — W25 : MTP de Flash-Next servi (R25)

**Rows.** `libr3d` sert deux nouvelles rows, sur les contrats de `libr4d` et
`libref` :
- `mtp_enter_e4m3` : norme groupée du flux large (`hc` sous-flux), norme de
  l'embedding, deux produits E4M3 `[n,n]` (scale f32 par 128) et le mélange
  `wadd` ; noyau un warp par canal, norme en LDS. Test device : 22 checks sur
  3 cas, contre `libref`, M=1..32, hc=4, n=2560.
- `logit_rerank_iqr` et `logit_rerank_i8` : rescore exact des `R=8`
  candidats du draft head contre la tête `lm_head` du tronc ; IQR (u3/u4) et
  i8 (`i8*bf16[1x128]`), `vocab_off` géré, `pairs` optionnel (monde > 1).
  Test device : 36 checks sur 5 cas, M=1..8, N=248320, K=2560, dont la
  forme réelle du conteneur. Selftest : **164 rows, 0 échec, 15 skips**.

**Deux bugs préexistants de `gemm_nt_q_w2a8_m16`, révélés par le draft head.**
1. **En-tête lu sur l'hôte** : le shim et le launcher appelaient
   `rad_planes_header` sur le poids `b_scale`, qui vit en **VRAM** quand l'op
   est sur la carte ; le CPU déréférençait donc un pointeur GPU → SIGSEGV au
   premier pas de spéculation. Preuve : `rocgdb` sur le serveur NS=3,
   `si_addr=0x7fe1c4347000`, `cmpl $0x534e4c50,(%rax)` (le magic
   `rad.planes`) dans `r3d_rad_gemm_nt_q_w2a8+2114`. Correctif : les offsets
   des plans (scale f16, zero u8) se déduisent de la règle de layout
   (`header + 2 entries`, aligné 64), sans lire l'en-tête ; le hook de layout
   impose désormais un zero `u8` (les deux bras le refusaient déjà).
2. **Chunk de K contre M** : le MT était choisi par M seul ; à M=1,
   `chunk = 2048`, et `K=2560` n'est pas un multiple → refus « shape ».
   Correctif : plus petit MT ≥ M dont le chunk divise K (MT=4 pour
   M=1, K=2560), le noyau stockant seulement `m < M`.

**Validation.** `sched_test` et `engine_e2e_test` verts ; `iqr_device_test`,
`logits_bf16`, `gather_rows_u8`, `vision_rows`, les deux tests MTP/rerank et
`r3d_selftest` verts. kbench Flash-Next `--kernels libr3d:libref
--max-cases 1` : **872/872** dans les deux modes (16 cas de plus qu'avant :
les rows MTP sont servies et vérifiées), 0 divergence, pire rel_l2 1,206e-2 ;
27B **605/605** dans les deux modes, pire rel_l2 3,619e-3. Les tables MTP 48
(`moe_gemm_q`) restent « Not covered » : c'est la couverture de R23.

**Service et gain.** Flash-Next avec `--num-speculative-tokens 3`, même
prompt et greedy (`max_tokens 256`, 1 séquence, 8k, KV i8, host pool 43 000,
build FFmpeg) :

| | NS=0 | NS=3 |
|---|---|---|
| décodage | 4,508 tok/s (255 pas) | **8,213 tok/s** (115 pas) |
| TTFT | 1,92 s | 2,03 s |
| texte | 1322 car. | **identique** (1322 car.) |
| acceptation | — | 40,3 % global (66,1 / 60,5 / 37,0 % aux profondeurs 1–3) |

Soit **1,82×** le décodage à texte identique, sans changer le préfill. Le 27B
reste à `NS=0` (son drafter h128 exige un KV bf16).

## 2026-11-16 — W23 : tables MoE posées dans kbench (R23)

**Cause double, toute côté outillage.** Le poids IQR du conteneur a cinq plans
(`codes`, `scale`, `trellis`, `trellis.mad`, `scale.col`) ; la row en déclare
une **vue** de trois plans, et `enc_selection_logical` refuse « 3 of 5 planes,
which decode to nothing on their own ». kbench retombait sur un tirage
synthétique, puis le relayout comparait la table `trellis` à des
`trellis.mad` tirés indépendamment → `RAD_E_FORMAT`. Pour `w4nl`,
`kf_lay_table` empilait les entrées à la taille déclarée (codes u4) alors que
le layout joint est plus grand : `RAD_E_SHAPE`. Le moteur, lui, passe une
**table de pointeurs** (`stride[0] == 0`), forme que les noyaux IQR/w4nl
exigent.

**Correctif** (branche `rdna3-r23-kbench`, commits `7720703`, `93d9e2b`) :
- `tools/kfixture` : complète la vue codes d'un treillis avec tous les plans ;
  décode la vue d'échelle comme `scale[ligne] × scale.col[bloc 128]` ; n'applique
  pas `fwht128` (le `Dec` des rows IQR ne le fait pas) ;
  `kf_lay_entries` dispose une entrée par expert.
- `tools/rad_kbench` : `TablePointers` construit un slab d'entrées alignées et
  un tableau de pointeurs hôte/device, pour le chemin conteneur et le replay ;
  marges et rotation froide suivent la taille pointée ; une référence dont le
  hook de description **décline** (`RAD_E_UNSUPPORTED`) ne fait plus décrire
  les opérandes par la row device (cas `gemm_nt_q_gated` i8a8h).
- `tools/kbench_timer` : la table de pointeurs fait tourner son slab et
  régénère ses têtes (`R > 1`) ; `TimedOperand` porte `slab/slab_bytes/
  entry_stride`, `bytes` reste le buffer pointé.
- `libr3d/r3d_iqr_layout.cpp` : refuse les formes MoE IQR que `form_supported`
  ne sert pas, au lieu de les accepter puis de les faire refuser au lancement.
- `tests/fixture_test.cpp` : deux tests de la vue codes/échelle IQR et un des
  entrées séparées.

**Couverture et mesure.** Sur le conteneur Flash-Next, le run demandé fait
**832/832** vérifiés, 0 échec, 0 divergence (pire rel_l2 3,162e-3) ; la
non-régression `--kernels libr3d:libref --max-cases 1` passe de 856 à
**952/952** dans les deux modes, pire rel_l2 1,206e-2 ; tous libs
**4930/4930**, pire 1,520e-2. Après fusion avec W25, l'arbre final fait
**968/968** (les 14 cas `mtp.draft_head` sont servis par `logit_rerank_i8`,
les 110 cas MoE par R23) et 27B **683/683** ; il reste 10 cas
`gemm_nt_q_gated` i8a8h sans référence libref et 8 `ngram_hint` sans oracle.

Temps par row et η lien (η = experts lus × octets / temps / 603 GB/s, débit de
copie froide à 256 MiB ; experts lus = borne haute `min(n_expert, M·top_k)`) :

| `moe_gemm_iqr` M | µs device | η | `moe_gemm_w4nl` M | µs device | η |
|---:|---:|---:|---:|---:|---:|
| 1 | 230,2 | 0,118 | 1 | 292,6 | 0,096 |
| 4 | 382,8 | 0,283 | 4 | 499,2 | 0,225 |
| 16 | 696,0 | 0,625 | 16 | 876,7 | 0,512 |
| 32 | 1 229,4 | **0,705** | 32 | 1 801,6 | 0,498 |
| 64 | 2 311,2 | 0,601 | 64 | 2 929,7 | 0,491 |
| 512 | 3 989,3 | 0,350 | 512 | 8 757,0 | 0,164 |

`gemm_iqr` dense : N=2560/K=2560 M=1 → 17,86 µs, 373 GB/s, η 0,619 ;
N=16384/K=2560 M=1 → 57,19 µs, 460 GB/s, η 0,763 ; M=16 → 70,61 µs, η 0,159.
Lecture : le décodage MoE M=1 n'est pas borné par la bande (10–12 %) mais par
le décodage treillis et le faible parallélisme par expert ; l'IQR atteint
425 GB/s (70 %) à M=32 puis redevient compute-bound (35 % à M=512) ; le w4nl
plafonne à 309 GB/s (51 %) à M=16.

Validation : `fixture_test`, `iqr_device_test` 67/67, `moe_iqr_device_test`
15/15, `moe_w4nl_device_test` 18/18, `moe_iqr_test` 45/45, `logits_bf16`,
`gather_rows_u8`, `vision`, selftest 164/0/15 sur l'arbre fusionné ; kbench
**968/968** (Flash-Next) et **683/683** (27B) dans les deux modes, 0 échec,
0 divergence. Carte : 2 613 MHz moyens, aucun throttler, plancher 3,06 µs,
copie froide 603 GB/s.

Restent non couverts : 10 `gemm_nt_q_gated` i8a8h (libref n'a pas la
référence), 8 `ngram_hint` (pas d'oracle), 20 formes IQR des fixtures non
servies (refusées à la déclaration), et la rotation froide `R > 1` d'une table
de pointeurs (revue, non mesurée sur ce conteneur).

## 2026-11-17 — W27 : IQ8R groupe 256, le 2B charge et sert (R9 / D1)

**Cause.** `minicpm5-2b-iq8r.rad` s'arrêtait sur
`llama_dense_fp8: rad_arch_declare -> invalid argument` (`rad_block_attn_fp8.h:110`,
`nq_h.declare`) : l'arch demande `group=256` (IQ8R), `libr3d` n'avait que des
rows à 128 — ni producteur rotated, ni GEMM `i8a8h`, dont l'échelle est un
**produit** u8 (par row, 256) × bf16 (par ligne). Aucun défaut du fichier.

**Correctif** (deux worktrees, `rdna3-iq8r-gemm` commit `2444126`,
`rdna3-iq8r-prod` commit `4b1cec6`, merges `ca302df` / `8a3e75e`) :
- `r3d_iqr_layout.cpp` reconnaît l'encodage IQ8R à `group=256` (affine, 3 plans,
  `fwht128`, codes i8 tiled, `scale` u8 tiled `[1,256]`, `scale.1` bf16 tiled
  `[1,0]`) ; les codes gardent l'ordre `kSigned` 8 par unité 16×128 et le plan
  d'échelle est rangé tile-major `[N/16][K/256][16]` bf16, valeur = produit
  arrondi une fois, comme `r4d_fp8_layout.cpp`.
- `r3d_iqr.h` / `r3d_iqr_dec.h` : `ngrp = K/256` pour l'indexation des échelles
  (`tile_major(r, kb/2, ngrp)`), mot de forme inchangé.
- `r3d_gemm_iqr_m16.hip` : `GS` (128/256) templatisé sur bande-décodage,
  résident et WMMA ; deux blocs de 128 s'accumulent dans **un seul** `sub` i32
  exact, puis un seul fold `(sub × a_scale) × w_scale` — l'ordre de
  `ref_gemm.h`. Le chemin 128 est bit-identique (tests + kbench inchangés).
  Split-K refusé à 256 (le hook scratch renvoie 0).
- Rows `gemm_iqr_nt_m16_w256` / `gemm_iqr_tiled_w256` (`C_EQ group 256`,
  `K % 256`), mêmes bandes que la paire 128.
- Producteurs `rmsnorm_had_quant_i8g`, `had_quant_act_i8g`,
  `gated_had_quant_i8g` : `HG = group/128` (8 colonnes par lane et par moitié),
  deux FWHT 128 indépendantes par groupe, **un** amax et **une** échelle par
  256 ; rows `*_256_bf16`. Le facteur 1/128 reste sur le poids.

**Couverture.** Selftest **169 rows / 0 échec / 15 skips** ; kbench Flash-Next
**968/968** et 27B **683/683** dans les deux modes (pire rel_l2 inchangés) ;
`iqr_device_test` 75 checks sur 10 cas (8 cas 256 : M 1/4/8/16/64/512, zéros,
codes −128/127) et `act_i8g_group256_device_test` 67 checks sur 4 cas
(256 + non-régression 128, silu/sigmoid, `out_bf16`, formes refusées).

Sur l'hôte, le 2B déclare **523 ops / 423 poids** ; les quatre ops IQ8R
résolvent sur les nouvelles rows (`gemm_iqr_nt_m16_w256` à M ≤ 16, tiled
au-delà ; `had_quant_act_i8g` a aussi un repli AVX, les deux producteurs
fusionnés restent device). kbench ciblé **110/110**, pire rel_l2 **4,018e-05**,
GEMM **bit-exact**. Les deux producteurs fusionnés affichent « Not covered »
(pas d'oracle libref) et sont couverts par le test device. Le conteneur 2B
**rejoué en entier** (`--kernels libr3d:libref --max-cases 1`, 18,5 s) donne
**384/384**, 0 échec, 0 divergence, pire rel_l2 **1,810e-03**, 30 rows device
exercées — `attn_paged` gqa=8, `rope` neox h128, `kv_store`, `embed_lookup`, la
tête à 130 560 tokens et les sampleurs sont donc couverts sur ce conteneur.

**Service.** `radiance --tp 1 --max-model-len 4096 --max-num-seqs 1
--num-speculative-tokens 0 --kv-cache-dtype bf16` : `/health` 200, complétion
greedy cohérente (« The capital of France is » → « Paris. » ; planètes dans
l'ordre), **40,2 tok/s** sur 60 tokens, 69,0 tok/s sur 16, préfill
184–367 tok/s sur ces prompts courts. Le KV **i8** est refusé par cet arch à
`head_dim=128` (`attn_paged`/`kv_store` : bf16/f16/fp8 seulement) : le service
du 2B utilise bf16.

## 2026-11-17 — W24 : `qsa_score` à M=512, cause mesurée (optimisation livrée en W28)

Mesure à froid sur `w27-merge`, `--op qsa_score --bench` : médianes device
**79,0 / 82,7 / 83,3 ms** sur trois passes (2 842–2 852 MHz, aucun throttler,
258,5 MiB d'opérandes), soit 103 GFLOP/s = 2 % du pic ; la row hôte AVX fait
**9,0 ms** sur 32 threads (9× moins).

**Cause.** Un thread traite un (row, bloc) ; une warp de 32 lanes lit donc
**32 pages distinctes au même offset**, et chaque lane n'utilise que 2 octets
d'un secteur de 32 (amplification ×16). La row est bornée par L2, pas par le
calcul ni par le lien.

**Essai écarté.** Quatre blocs par thread, quatre chaînes f32 indépendantes,
arithmétique par (row, bloc) inchangée (test device 72 checks, 12 cas, exact) :
**163,5 / 163,0 / 163,0 ms**, 2× plus lent — davantage de pages distinctes par
instruction. Retiré de l'arbre. La suite est une réécriture coalescée/tuilée
(pages et lignes de `q` en LDS, un thread par (row, bloc) conservé), à mesurer
sur ce même protocole.

## 2026-11-17 — W21 préparation : `hc_read` attribué au noyau (profil engine)

Serveur Flash-Next `--profile-ops`, `NS=0`, `MLP=8192`, `SEQS=1`,
`expert_tiered`, pool hôte 43 000, KV i8, une complétion greedy de 4 tokens
(arbre `w27-merge`). Le tableau final (cumulatif sur 4 pas, non comparable à un
run non profilé) donne :

- **96 sites `hc_read`** (attn + ffn par couche, 4 appels chacun) à
  **1,93 ms/appel**, `wait` 2,8–7,9 µs ; total 7,72–7,84 ms par site.
- Les autres postes : `embed_lookup` **9,50 ms/appel** (pool hôte, 38 ms au
  total), `gemm_nt_q` `ssm_inz` 5,27 ms/appel, `gdn_conv_recurrent_update_had`
  3,11 ms/appel.
- Décodage profilé : 161,9 ms/token (les synchronisations du profil changent le
  débit ; à ne pas comparer à 4,0–4,6 tok/s non profilé).

Comparaisons et placement :
- kbench, **même (op, kernel, géométrie)** (`M=1 n=2560 hc=4 lowrank=320
  mix=e4m3`) : **1,77 ms** → l'engine n'ajoute pas de coût propre.
- `--debug-placement` : `blk.*.attn_hc_norm.weight` et `blk.*.ffn_hc_norm.weight`
  sont **en VRAM** (48 × 1,88 Mio chacun, plus `output_hc_norm.weight` 40 Kio) ;
  poids par rang : 42,55 GiB VRAM, 20,02 GiB pool hôte, 0 sur disque.

Conclusion : ni le lien ni le tiering n'expliquent les 1,93 ms ; la row RDNA3
relit la paire de matrices **par ligne** au lieu d'amortir par la tuile de
128 lignes que son contrat décrit. À 6,55 Mo par appel (chiffre W18) en
1,77 ms, le débit est **~3,7 GB/s** (plancher 10,5 µs à 622 GB/s, référence
libr4d 18,7 µs). La correction est un travail de noyau (W21) et attend le feu
vert ; `hc_enter`, `hc_write` et le MoE ne sont pas en cause.

## 2026-11-17 — W28 : hc_read, qsa_score, DSpark NS=3, folds IQ8R et collectifs tp>1

Cinq lots menés en parallèle (worktrees Agent Manager), fusionnés dans `rdna3-iqr`,
puis validés ensemble dans un build neuf (`w28-merge`).

**W21 `hc_read` (`96617a4`).** Diagnostic (profil `--profile-ops`) : 96 sites à
**1,93 ms/appel** en engine contre 1,77 ms en kbench au même géométrique, poids
`hc_norm` en VRAM — le coût est dans le noyau. Les deux noyaux lisaient les
matrices de mélange une ligne par lane au même offset (32× d'amplification de
secteur en `down`, 8× en `up`, 30 blocs à M=1). Ils stagèrent maintenant une
tranche en LDS avec les 256 threads (16 lignes × 256 colonnes en `down`, 64
lignes × 32 k en `up`), les lanes propriétaires gardant la somme f32 ascendante
de la référence. Cold card, 3 passes alternées, 2 801 MHz : M=1 **1 941 → 428 µs**
avec inject (4,5×) et **1 774 → 351 µs** sans (5,0×) ; M=8 6,2× ; M=128 7,9× ;
M=512 **114 → 12,4 ms** (9,2×). `ple_hc_device_test` 48 checks verts, rel_l2
kbench inchangé. En service, décodage Flash-Next **4,48 → 12,73 tok/s**
(**2,84×**, même prompt, 64 tokens).

**W24 `qsa_score` (`bb08e16`).** Une warp lisait 32 pages de blocs distinctes au
même offset (16× d'amplification). Réécriture : un workgroup stage en LDS les
pages et les lignes de q d'une tuile 32×64, chaque thread garde des chaînes
f32 ascendantes. M=512 **82,27 → 2,23–2,30 ms** (36–37×), **bit-identique** à
libref ; `qsa_score_device_test` 72 checks exacts.

**W25/D2 DSpark (`e7a5b26`).** `minicpm5-2b-iq8r.rad` à `NS=3` sur la build
`w28` : les rows du drafter existaient ; le seul défaut était le cas selftest
`w2a8` déclaré `U8` au lieu de `F16`, donc toujours sauté. Cas numérique ajouté
au bande du drafter (**M=7**, plan joint en mémoire device, rel_l2 1,72e-03).
Service : **32,20 tok/s contre 20,64** (1,56×), **texte identique au bit**,
33 drafts acceptés sur 210 (15,7 %) ; sur sa branche, kbench 2B 384/384 et FN
968/968 (avant le fix d'intégration `gate_had_quant_i8g` ; l'arbre fusionné
donne 948/948, voir plus bas).

**W28b folds IQ8R (`4cbcd73`).** `gate_had_quant_i8g` 128/256 (la porte dans son
propre opérande, deux rotations 128 et un amax par 256) et le fold
`gemm_nt_q_gated` `i8a8h` 256 (un `sub` i32 exact sur deux blocs, un fold).
Tests device étendus (91 checks sur 5 cas).

**W28c collectifs tp>1 (`038d78b`).** 18 rows `all_reduce`/`all_gather` portées
de libhalo (`r3dh_ar_*`, renommage `halo_`→`r3dh_`), schemas et bandes de
libr4d, `device.p2p` requis, paramètre rangé = plafond de bande ; les jonctions
`ar_*` fusionnées restent des stubs (producteurs à signer). Test
`tp2_collectives_device_test` : deux rangs sur une carte, deux streams, somme
f32 exacte et gather étiqueté (12 checks, 2 cas). Selftest **190 rows**.

**Trouvé à l'intégration (corrigé, `a8bc412`).** Déclarer une op change le
**graphe** de l'arch : avec `gate_had_quant_i8g` déclarée, Flash-Next prend la
branche fusionnée et l'émet sur **toute la projection** — `q`/`x` en
`[T, heads*head_dim]` alors que le paramètre `n` nomme la largeur de tête, et la
porte est une vue `[T, heads, head_dim]` de pas de tête 512 ([q|porte]
entrelacé). La row refusait cette géométrie. Corrigé : pas de ligne et de tête
portés, un bloc de rotation reste dans une tête, cas device dédié (103 checks
sur 6 cas). Texte servi correct ; le 27B voit 30 cas `sigmoid`/`mul`/
`quant_act_fp8` remplacés par la branche fusionnée (sans oracle libref) mais
**répond au bit près** comme avant (29,63 vs 29,50 tok/s).

**Validation combinée.** Selftest **190 rows, 0 échec, 16 skips** (les 16
schemas sans oracle libref) ; `sched_test` 548 782, `engine_e2e_test` 909
(3 skips), `fixture_test` 92 385 (7 skips), qsa 72, tp2 12, act_i8g 103,
ple_hc 48, iqr, logits, gather_u8, vision, mtp, rerank verts ; kbench FN
**948/948**, 2B **384/384**, 27B **653/653**, 0 échec, 0 divergence, pires
rel_l2 inchangés (1,206e-02 / 1,810e-03 / 3,619e-03) — le nombre de cas suit
l'énumération des rows résolues, pas la couverture.

**Doc.** `RAPPORT_RDNA3.md` entre dans le dépôt (décision prise) et ses
§7.6/§10.4–10.5 sont révisés ; `docs/OPS.md` et `libr3d/README.md` étaient déjà
à jour.

## 2026-11-17 — rebase `main` 2.1.0 (W29) et retour du préfill IQR (GS=1)

**Rebase.** `rdna3-iqr` rejouée sur `main` `3045979` (v2.1.0) : 144 commits
rejoués, 4 conflits additifs (CMake + 3 blocs `r3d_rows.cpp`), `ple_conv`
dédoublonné au profit du noyau entrant, `moe_gather` adopte la garde de
capacité, `down_w5_fp8` restauré (défaut amont après `d1b5a45`), `ngram_ids_i32`
restaurée (sa suppression cassait Flash-Next M=2048). Détail :
`RAPPORT_REBASE_V2_1_0.md`.

**Selftest.** **195 rows, 0 échec, 16 skips** ; `sched_test` 548 952,
`engine_e2e_test` 1 076 (3 skips), `build_test` 969, `shipped_plugins_test`
1 010, `arch_test` 183 527, `device_test` 61 808, tous les tests device RDNA3
verts. kbench **27B 3 181/3 181**, **Flash-Next 4 786/4 786**, MiniCPM
**1 802/1 802**, 0 échec, pires rel_l2 3,619e-03 / 1,520e-02 / 3,485e-03.

**Écart de préfill, corrigé.** L'A/B `final-all` (W26) vs rebasé montrait
285 → 249 tok/s ; un build neuf de la pré-rebase (W27) donne 250 tok/s, donc
le rebase est neutre. La dette est la généralisation `GS=1/2` du noyau IQR
pour IQ8R : `gemm_iqr_resident_kernel` groupé coûte 17 % sur le 27B. La
spécialisation `GS=1` (`faa89fd`, noyau resident restauré de W26) redonne
**3 839 µs** (16,8 TFLOP/s) et **288,4/287,3 tok/s** sur 532 tokens ; texte
27B identique, IQ8R (`GS=2`) inchangé.

**Autres postes (build patché).** Flash-Next IQR tiled 2 053,8 µs
(12288×2560 M=512) / 2 675,9 µs (16384×2560) ; MoE 231,2 / 4 009 µs ;
bf16 défaut 776,8 / 788,4 µs ; `hc_read` **355,1 / 12 323,5 µs** (W21, 5–9×
mieux que W26) ; `mtp_enter` 186,2 / 59 385 µs (linéaire : grille
`(n/8, M)`, un bloc par (8 canaux, rangée) — la linéarité est le travail d'un
GEMV par rangée, pas un défaut de parallélisme ; l'hypothèse 2–4× est retirée,
à profiler dans le graphe) ; `logit_rerank`
32,9 µs. `rad-tune` 27B : bf16 tiled `impl=3` 199,5 µs contre 789,0
(3,95×) ; Flash-Next tiled 65,2/67,2/71,2 µs (N=96/512/640 K=2560 M=512).

**QSA long contexte.** Flash-Next à 3 252 tokens (> borne 2 051) : préfill
**234,8 tok/s**, TTFT 13,85 s, rows `qsa_block_key`/`qsa_score`/`qsa_select`/
`qsa_tail_store`/`qsa_work` sélectionnées en libr3d et avx. Le chemin dense
`by_table` reste exercé à 58 tokens.

**Attention couverte.** `attn_paged_device_test` (nouveau) compare les rows
h256 i8 gqa6 bs16 et gqa12 bs4, décodage et préfill, court et split, à
libref : **8/8** à **1 bf16 ULP**, pas bit-exact. La row revendique le
bit-exact ; écart documenté (polynôme `exp` à 1 f32 ulp, parenthèses du
split-KV). `RAD_ATTN_BENCH=1`, **scratch/split-KV alloué** : gqa6/bs16 q=1
**58,9 / 235,6 / 976,5 / 4 059,4 µs** à ctx 512/2048/8192/32768 ; gqa12/bs4
**39,9 / 560,9 / 2 365,8 µs** ; préfill q=17 **3 166,4 / 12 052,4 µs**.
Sans scratch (repli `splits=1`, non servi) les décodages donnent
437,7 / 1 973,8 / 7 313,6 / 29 133 µs et 499,4 / 8 505,2 / 33 285 µs — à ne
pas utiliser comme baseline. Coût ~0,12 µs/token, ~8 GB/s de KV effectif ;
levier restant : préfill q=17 (splits=1) + multi-rangées verify (M=4), en
préservant l'ordre par rangée. kbench ne chronomètre toujours pas l'attention.

**Amont.** `down_w5_fp8` et `ngram_ids_i32` sont deux régressions de `main`
2.1.0 (compilation et service RDNA3) ; correctifs locaux `5e98eb9` et
`c11c451`, à proposer en amont.

## 2026-11-18 — W30 : préfill `attn_paged` h256 i8 en split-KV

**Constat.** Le préfill h256 i8 était le seul chemin encore à `splits=1` :
un premier pas de 17 tokens fait deux tuiles de 16 (gqa 6) ou trois de 8
(gqa 12), soit **8 à 12 workgroups** pour une carte de 96 CU, et chaque
workgroup marche **tout le contexte** en série. La preuve est dans la mesure :
à 8 192 de contexte, q=17 et q=128 coûtent le même temps (12,1 vs 12,5 ms)
alors que q=128 fait **7,5× plus de travail** — le poste est borné par le
parallélisme, pas par l'arithmétique. Baseline `build-p0`
(`RAD_ATTN_BENCH=1`, carte froide, meilleur de 3 lots de 20) :

| préfill | 2 048 | 8 192 | 32 768 |
|---|---:|---:|---:|
| gqa6 bs16 q=17 | 3 153,8 µs | 12 122,4 µs | 46 886,4 µs |
| gqa12 bs4 q=17 | 3 539,4 µs | 13 634,6 µs | 52 937,9 µs |
| gqa6 bs16 q=128 | 3 305,0 µs | 12 517,6 µs | — |
| gqa12 bs4 q=128 | 3 625,6 µs | 14 379,7 µs | — |

**Correctif** (`libr3d`, `r3d_attn_paged_h256.hip` + `r3d_rows.cpp`). L'entrée
de préfill prend `splits` et le scratch, comme le décodage : l'échelle de
splits cible **128 workgroups** (deux par CU), est plafonnée au nombre de blocs
du cache (une tranche au-delà ne porte aucune clé) puis divisée par deux
jusqu'à tenir dans l'aréna offerte, jusqu'à `splits=1` — le repli sans scratch
reste donc le comportement d'avant. Les rows de préfill déclarent
`attn_h256_scratch` (indice à 16 tranches, plafond 16 Mio), le décodeur est
inchangé. Le noyau n'a pas bougé : c'est la même marche en deux passes, la même
arithmétique par tranche, et les mêmes parenthèses documentées du split-KV.
Test `attn_paged_device_test` étendu au mode temps : préfill q=17 à 32 768 et
q=128 à 2 048/8 192 pour les deux géométries.

**Mesures** (`build-opt`, même protocole ; carte 199 W, 2 315 MHz, aucun
throttler — `THROTTLE_STATUS` vaut `THROTTLED` au repos sur cet hôte) :

| préfill | 2 048 | 8 192 | 32 768 |
|---|---:|---:|---:|
| gqa6 bs16 q=17 | **558,0 µs (5,7×)** | **2 375,6 µs (5,1×)** | **8 507,7 µs (5,5×)** |
| gqa12 bs4 q=17 | **394,7 µs (9,0×)** | **1 795,3 µs (7,6×)** | **6 364,3 µs (8,3×)** |
| gqa6 bs16 q=128 | **2 379,6 µs (1,39×)** | **8 994,2 µs (1,39×)** | — |
| gqa12 bs4 q=128 | **2 537,0 µs (1,43×)** | **9 637,5 µs (1,49×)** | — |

Décodage inchangé (gqa6 q=1 : 56,4 / 243,4 / 956,1 / 3 859,4 µs à ctx
512/2048/8192/32768 ; gqa12 : 38,3 / 544,1 / 2 321,8 µs). Dispersion entre deux
runs du même binaire : ~5 % (q=17 à 8K : 2 253–2 376 µs) ; les chiffres retenus
sont les plus prudents. Une cible à 256 workgroups (donc 32 tranches quand
l'aréna le permet) ne change rien : à 128 workgroups le noyau sature — il est
borné par l'instruction (une conversion int8 et une FMA par lane et par
dimension), pas par le parallélisme. q=17 garde en plus 47 % de rangées de
remplissage (17 tokens sur 32 emplacements) ; elles ne paient que la passe un.

**Validation** (`build-opt`, build neuf) : `attn_paged_device_test`
**8 checks/9 cas** (le cas préfill « split » exerce désormais une vraie coupe
à ctx=32), selftest **195 rows, 0 échec, 16 skips**, `sched_test` 548 952,
`engine_e2e_test` 1 076 (3 skips), `fixture_test` 109 671, tous les tests device
verts (iqr, logits, gather_u8, vision, mtp, rerank, act_i8g 103, ple_hc 48,
tp2 12). Texte servi **identique** : Flash-Next, prompt de 3 252 tokens
(chemin QSA sparse) 32 tokens générés, même acceptation de drafts (19/39) ;
Flash-Next et 27B à 372 tokens, texte identique aussi. Le prélude de
3 252 tokens ne gagne rien (241,1 → 239,9 tok/s) : le découpage est en chunks de
2 048 tokens, où `tuiles × têtes KV` dépasse déjà 128, donc `splits=1` ; le gain
de la row porte sur les **chunks courts** (< 512 tokens, typiquement la queue
d'un prompt), où le préfill est dominé par les GEMM et non par l'attention.

**Verify M=4, mesuré et clos.** Le mode temps couvre désormais q=4 (NS=3) dans
la bande décodage : gqa6 q=4 coûte le **même temps** que q=1 (969,8 vs 921,3 µs
à 8K ; 3 915,0 vs 3 872,6 à 32K ; 235,9 vs 269,0 à 2K), donc la vérification
des 4 rangées est absorbée par la marche KV et le lancement. gqa12 paie
+23–31 % d'un pas (707,1 vs 539,9 µs à 8K ; 2 885,0 vs 2 343,3 à 32K) parce que
ses tuiles de 4 tokens passent de 25 % à 100 % de rangées réelles ; c'est le
prix du travail, pas un défaut de parallélisme. **Pas de chantier multi-rangées**
(comme R33 pour `mtp_enter`).

**Reste.** Au-delà du split, il faut changer l'arithmétique — dot
`v_dot4_i32_i8` après quantification de q, ou softmax en une passe (online) —
ce qui déplace le contrat numérique et demande une mesure dédiée.

