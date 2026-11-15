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



