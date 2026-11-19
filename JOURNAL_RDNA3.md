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

**Service multi-séquence (27B).** `max-num-seqs 4`, KV i8, `NS=0`, quatre
requêtes **distinctes** de 83–92 tokens, 64 tokens chacune, lancées ensemble,
`build-p0` contre `build-opt`, deux runs par build : **49,4 / 49,2** (p0) et
**50,3 / 48,9** (opt) tok/s agrégés, 256 tokens en 5,08–5,31 s, soit
**~16 tok/s par flux**. Le split-KV du préfill ne dégrade donc pas le service
multi-séquence ; la recette `max-num-seqs 4` reste valide.

Réserve mesurée : le texte n'est reproductible qu'à **composition de lot
fixe**. Un warm-up sur un prompt distinct rend chaque build déterministe d'un
run à l'autre (une exception sur `build-opt`, où la composition a changé), mais
le nombre de tranches de préfill dépend de `n_seq` (`per = ceil(rows/n_seq)`
dans le shim) : deux compositions différentes déplacent des bits bas et
peuvent retourner un argmax serré. C'est la même classe d'écart que le split du
décodage, déjà documentée ; le shim ne connaît pas les longueurs par séquence,
donc une politique indépendante du lot n'est pas disponible à cet endroit.

**Profil servi (mesure, pas optimisation).** `profile-ops: true`,
`RADIANCE_PROFILE_EVERY=1`, `build-opt`, une requête par cas, NS=0. Les temps
sont gonflés par les barrières du mode ; les **parts** sont la lecture.

| cas | TTFT | poste | part | détail |
|---|---:|---|---:|---|
| Flash-Next, image 1024² (1 045 tok) | 9,58 s | `gemm_nt_bias` | 36,7 % | 3,71 s, 111 appels, N=1152 K=1152 M=1045 : **11,8 ms/appel = 0,23 TFLOP/s** |
| | | `hc_read` | 29,8 % | 3,01 s, 873 appels |
| | | `moe_gemm_q` | 20,5 % | 2,07 s |
| | | `attn_dense` | 4,4 % | 0,44 s |
| Flash-Next, image 2048² (4 098 tok) | 39,2 s | `gemm_nt_bias` | 37,0 % | 14,7 s |
| | | `hc_read` | 28,4 % | 11,3 s |
| | | `attn_dense` | 16,8 % | 6,69 s, **251 ms/couche** à 4 096 tokens |
| | | `moe_gemm_q` | 7,6 % | 3,04 s |
| | | `attn_paged` | 4,7 % | 1,87 s |
| Flash-Next, texte 972 tok | 4,78 s | `hc_read` | 53,4 % | 2,84 s |
| | | `moe_gemm_q` | 32,1 % | 1,71 s |
| 27B, texte 3 252 tok | 11,44 s | `gemm_nt_q` | 88,3 % | 10,22 s = **17,2 TFLOP/s** (roofline de son propre noyau, 16,8) |

Lecture :

- **La tour vision coûte 21,4 s des 39,2 s** d'une image 2048². `gemm_nt_bias`
  est la row **bit-exacte** de la tour (tuile 16×32, opérandes f32 en LDS, un
  mul + un add par pas de k) : 0,83–0,96 TFLOP/s à kbench sur toutes les formes,
  et **libavx à 32 threads fait 2,8× mieux** que la carte sur la même forme. Ce
  n'est pas un noyau lent par accident, c'est le contrat de la row.
- **`attn_dense` hd72 s'effondre au-delà de la bande mesurée** : 215 GFLOP/s à
  M=512 (350,7 µs) mais **19 GFLOP/s à 4 096** (251 ms/couche, 11× pire par
  FLOP), non causal, `gqa=1`, `max_seqlen=16384` déclaré. Aucun changement de
  contrat n'est nécessaire pour corriger ça (même ordre arithmétique).
  **Corrigé en W31 : ce chiffre est faux** — la forme servie est 16 384 patches
  × 16 têtes, la row fait **5,30 TFLOP/s** et son efficacité monte avec M ; il
  n'y a pas d'effondrement, et les variantes de scheduling sont neutres. Voir
  l'entrée W31.
- **`hc_read` reste un poste de préfill** : 2,84 s par chunk de 972 tokens
  (53 %), 11,3 s à 4 098 tokens, soit 28 µs par rangée et par site — ~4,6× le
  plancher de ses 630 Mo de matrices relues.
- **Le 27B n'a pas de marge de graphe** : 88 % de son préfill est `gemm_nt_q`
  à 17,2 TFLOP/s, c'est-à-dire son noyau ; le gagner demande de gagner le
  noyau IQR tiled (16,8 → 23,5 TFLOP/s pour notre propre tuile bf16).

## 2026-11-18 — W31 : vision et préfill, trois noyaux (agents parallèles)

**Correction d'abord.** Le profil de W30 annonçait `attn_dense` à « 19 GFLOP/s
à M=4096 » puis un effondrement : c'était une **erreur de comptage**. La forme
servie par la tour est **16 384 patches × 16 têtes** (4 096 est le nombre de
tokens après le merger), soit 1,237 TFLOP par couche. La row fait **5,30
TFLOP/s** à cette forme, et son efficacité **monte** avec M (3,03 à 512×16,
4,71 à 4 096×16, 5,30 à 16 384×16). Ce n'est pas un effondrement : c'est 53 %
de sa borne d'issue VALU (~435 instructions VALU par clé et par thread, sans
FMA, avec le recalcul du max imposé par le bit-exact). Variantes mesurées et
**rejetées** : 256 threads/bloc 238,2 ms, kStage 32 235,4, waves/EU 2 239,5,
waves/EU 6 239,7, VOPD 236,1 (baseline 235,3). Aucun noyau modifié ; seul un
**mode temps** est ajouté (`RAD_ATTN_DENSE_BENCH=1`), qui couvre 1 et 16 têtes
et les segments `cu`, avec le compte de FLOPs correct. Pour ≤ 50 ms il faudrait
le FMA (ou une softmax en une passe) : **décision de contrat**, pas de
scheduling.

**`hc_read` — tuilage par rangées (bit-exact).** Cause mesurée : le coût était
**par rangée**, pas par octet (hot et cold à moins de 3 %) ; `hc_down` ne
gardait que 16 lanes actives sur 256 et chaque rangée relisait la matrice
entière (108 octets lus par octet de matrice à M=108). Un workgroup `down`
couvre maintenant 32 sorties × 8 rangées, un `up` 32 colonnes × 8 rangées ;
la tranche stagée sert les 8 rangées ; les noyaux W28 à une rangée par bloc
sont **conservés** sous une tuile (M < 8, décodage).

| `hc_read` M | avant (hot, µs) | après | gain |
|---:|---:|---:|---:|
| 1 | 400,3 | 399,2 | 1,00 |
| 128 | 3 766,1 | **1 297,5** | 2,9× |
| 512 | 14 894,5 | **5 656,8** | 2,6× |
| 1 024 | 27 929,6 | **10 787,4** | 2,6× |
| 4 096 | 107 975,6 | **38 263,7** | 2,8× |

`ple_hc_device_test` **58 checks/10 cas** (dont un oracle E4M3, M=10 pour la
tuile partielle), bit-identique sur x/inj/q/rq/scales.

**`gemm_nt_bias_bf16` — tuile bf16 (bit-exact).** Cause : l'ancien noyau stagait
les opérandes en **f32** avec un pitch de 128 f32 (multiple de 32 mots), soit
un conflit de banques 16 voies, 3 LDS par 2 MAC et une seule chaîne de deux
colonnes par thread. Nouveau : staging **bf16** k-major, pitchs 36/68
halfwords, tuile 32×64, 128 threads, **4 rangées × 4 colonnes par thread**
(16 chaînes indépendantes), mul et add séparés (`-ffp-contract=off`).

| forme | avant | après | gain |
|---|---:|---:|---:|
| 512×1152×1536 | 2 857,2 µs (0,63 TF/s) | **259,8 µs (6,97)** | 11,0× |
| 1 045×1152×1152 + res | 3 332,0 µs (0,83) | **385,1 µs (7,20)** | 8,7× |
| 1 024×4 608×4 608 gelu | 44 971 µs (0,97) | **6 251,9 µs (6,96)** | 7,2× |

kbench ciblé : **140/140**, `rel_l2` libr3d **0,00e+00** sur tous les M ; la row
est désormais **2,3–2,5× plus rapide que libavx** (« libavx 2,8× mieux » est
inversé). Le plafond bit-exact est ~11 TF/s (mix 2,75 instructions/MAC) ; on en
tient 63–66 %. Le repère 23,5 TF/s est le chemin `dot2`, **non** bit-exact.

**Effet servi** (`profile-ops`, même protocole, NS=0, `build-w31`) :

| cas | avant (W30) | après (W31) | |
|---|---:|---:|---|
| Flash-Next texte 972 tok, TTFT | 4,78 s | **3,13 s** | −34 % |
| Flash-Next texte 3 252 tok, préfill | 13,56 s | **8,19 s** | −40 % |
| Flash-Next image 1 024², TTFT | 9,58 s | **4,66 s** | −51 % |
| Flash-Next image 2 048², TTFT | 39,2 s | **19,1 s** | −51 % |
| 27B texte 3 252 tok, TTFT | 11,44 s | 11,39 s | — (IQR 88 %) |

Détail de la passe 2 048² : `attn_dense` 6,54 s (inchangé), `hc_read` 11,28 →
**4,01 s**, `gemm_nt_bias` 14,71 → **2,12 s**, `moe_gemm_q` 2,98 s,
`attn_paged` 1,82 s. Texte servi **identique au bit** : Flash-Next 3 252
tokens (32 tokens générés, drafts 19/39) et 27B 372 tokens.

**Validation** : `sched_test` 548 952, `engine_e2e_test` 1 076 (3 skips),
`fixture_test` 109 671, `iqr_device_test` 76, `vision_rows_device_test` 66/10
(79 avec les deux modes temps), `ple_hc_device_test` 58/10, tous les autres
tests device verts, `r3d_selftest` **195 rows, 0 échec, 16 skips**. Branches
agents fusionnées : `735ed0f` (bench attn_dense), `d724d55` (hc_read),
`6c26c24` (gemm vision), `e80e87d` (bench vision).

**Reste.** Vision : `attn_dense` (6,5 s d'une passe 2 048²) ne peut descendre
qu'avec un changement de contrat (FMA ou softmax en une passe) ; la piste
« lane = clé » avec scores en LDS viserait ~124 ms au mieux. Préfill
Flash-Next : `moe_gemm_q` est désormais 46 % d'un préfill de 972 tokens, et
`hc_read` 33 % ; le 27B reste borné par la tuile IQR.

## 2026-11-18 — W32 : `hc_read` deuxième passe, `attn_dense` clos, MoE rejeté

**`hc_read` — deux rangées par lane, norm et quant parallèles (livré, bit-exact).**
Causes mesurées au profil (rocprofv3, M=4 096) : `down` et `up` coûtaient **2 LDS
par MAC** (un pour le poids, un pour l'activation) ; `hc_norm` prenait 8,5 ms
avec un bloc par rangée et **4 lanes actives sur 256** sur une chaîne `double`
dépendante ; `quant`+`inj` 9,2 ms, FWHT et amax sérialisés dans la lane 0 sur
20 blocs par rangée.

Correctif : une lane possède **2 rangées** en `down` et **2 colonnes × 2
rangées** en `up`, donc un poids stagé sert deux MAC et une activation deux
colonnes (1,5 puis 1,0 LDS/MAC), les quatre produits d'un groupe étant formés
avant leurs additions **ascendantes** (aucun FMA, `-ffp-contract=off`) ;
`hc_norm` couvre 64 rangées par bloc avec huit chargements émis avant la somme ;
`q`/`rq` font l'amax par réduction de bloc et la rotation étage par étage sur
128 threads, dans l'ordre d'étages de la référence.

| `hc_read` M | W31 | W32 | gain | vs W30 |
|---:|---:|---:|---:|---:|
| 1 | 399,2 | **324,1** | 1,23× | 1,23× |
| 128 | 1 297,5 | **728,4** | 1,78× | 5,17× |
| 512 | 5 656,8 | **1 927,3** | 2,94× | 7,73× |
| 1 024 | 10 787,4 | **4 045,7** | 2,67× | 6,90× |
| 4 096 | 38 263,7 | **13 702,5** | 2,79× | 7,88× |

`core` (down+up seuls) 28 601 → **13 005 µs** à M=4 096. Cumulé depuis W30, la
row est **7,7–7,9×** plus rapide au préfill.

Effet servi (`profile-ops`, même protocole, NS=0) :

| cas | W31 | W32 |
|---|---:|---:|
| texte 972 tok, TTFT | 3 133 ms | **2 560 ms** (−18 %) |
| texte 3 252 tok, préfill | 8 186 ms | **6 316 ms** (−23 %) |
| image 1 024², TTFT | 4 660 ms | **4 049 ms** (−13 %) |
| image 2 048², TTFT | 19 100 ms | **16 967 ms** (−11 %) |
| `hc_read` dans le préfill 972 tok | 1 197 ms (32,6 %) | **534 ms** (17,6 %) |

Texte servi **identique au bit** (Flash-Next 3 252 tokens, drafts 19/39).

**`attn_dense` hd72 — piste « lane = clé » mesurée et close.** Comptage
d'instructions du noyau : **437 VALU par clé et par thread** (marche 1 : 145,
marche 2 : 292), soit exactement la borne bit-exact, plus 182 créneaux
d'émission non-VALU (54 LDS, 53 `s_waitcnt`, 75 `s_delay_alu`) → 619. Plancher
VALU pur 115 ms à 2,6 GHz ; à 100 % d'issue 163 ms ; mesuré **233 ms**, donc
**~70 % d'issue**. Variantes mesurées : lane = clé avec tuile de scores 32 en
LDS **286,2 ms (+23 %)**, `kStage` 64 **2 315,6 ms** (36 KiB de LDS, occupancy
détruite), `__launch_bounds__(128,3)` **879,2 ms**, waves/EU 3 **232,9** et 5
**233,8** (bruit). La carte « lane = clé » est une **transposée**, pas une
réduction : q passe du registre au LDS, K devient par-lane, 56 instructions de
plus par clé et 28 spills. **Aucun noyau modifié** ; le plancher réaliste de
cette row sous contrat est ~5,3 TFLOP/s à 16 384 patches × 16 têtes, et les
6,5 s de la tour restent. Le seul gain restant exigerait le FMA, un tampon de
scores de 64 KiB par ligne (changement de schéma) ou une softmax en une passe.

**MoE run map — livré sur kbench, rejeté en service.** Le noyau `moe_iqr` donnait
à un bloc une tuile fixe de 64 rangées alors qu'un préfill route une ou deux
rangées par expert ; la variante choisissait la tuile la plus étroite parmi
16/32/64 (`T ≤ 8·E` → 16, `≤ 24·E` → 32, sinon 64), ce qui divise la LDS par
deux et supprime un barrage par bloc-K. kbench : **M=128 2 591 → 2 033 µs
(1,27×)**, M=256 1,19×, M=512 1,10×.

En service, c'est une **régression** (profil, deux runs alternés par arbre) :
**2 550 / 2 554 µs → 2 768 / 2 766 µs (+8,5 %)**. La géométrie servie est
`M=1204` (bande M ≤ 2048), soit `T = 12 040` et **~23,5 rangées par expert** :
la tuile 32 double le nombre de blocs, donc chaque unité de poids 16×128 est
**décodée deux fois** au lieu d'être réutilisée sur quatre tuiles de 16
rangées. Le gain ne vaut que pour des runs de 1–2 rangées.

Le commit a en outre cassé le service : le prompt de 3 252 tokens a **arrêté le
moteur** au premier pas (`kernel moe_gemm_iqr refused op 'moe_gemm_q' at M=1204
[band M<=2048] … scratch buffer absent or too small`). Cause : le hook de
déclaration dimensionnait la carte de runs avec la tuile du **M de la bande**
(T=20 480 → tuile 64 → capacité 833) alors que le pas utilisait M=1204 → tuile
32 → capacité 890. La capacité `n_expert + ceil(T/tuile)` **croît quand la
tuile rétrécit** : une déclaration doit se dimensionner pour la tuile la plus
étroite possible, jamais pour celle d'un pas particulier. Commit retiré de la
branche (l'historique est aplati : il n'y a pas de commit mort), résultat
consigné ici ; l'agent reprend le sujet avec la géométrie servie comme porte.

**IQR — tuile registre 2×4 : réelle sur la grille signée, inapplicable au 27B.**
L'agent a diagnostiqué juste (le décodage de l'unité 16×128 tourne sur la
vague 0 pendant que trois vagues attendent au barrage, 110 VGPR, 4 blocs/CU) et
livré une tuile registre 2 tokens × 4 lignes avec **copie parallèle des
octets** de l'unité par les quatre vagues. Sur la grille **i8 signée** :
34816×5120 M=1626 **32,91 → 16,94 ms (1,94×)**, 34,2 TFLOP/s, 0 ulp ;
M=1 inchangé ; `iqr_device_test` 78/12. Mais la garde est
`if constexpr (F::KIND == kSigned && F::B == 8)` et **`rad-info --recipe` montre
que les poids denses du 27B sont trellis u4/u5/u6 (`tcq=16`, `tcq_code=mad`)**,
donc `kTrellisMad` : la tuile n'est **jamais sélectionnée**. Mesure servi
(prompt 3 252 tokens) : **TTFT 11 244 ms**, `gemm_nt_q` 10 048 ms — inchangé
(11 390 / 10 180 avant). Un gain de kbench sur une forme absente du graphe ne
compte pas (R39) : commits retirés, sujet rendu à l'agent avec la forme trellis.
La piste correcte pour le trellis n'est pas la copie parallèle (chaîne d'états)
mais le **recouvrement** du décodage du bloc K+1 avec la consommation du bloc K
(double buffer entre blocs K, ordre d'accumulation inchangé).

**Validation (`build-w32`, build neuf)** : `sched_test` 548 952,
`engine_e2e_test` 1 076 (3 skips), `fixture_test` 109 671,
`ple_hc_device_test` **58 checks/10 cas**, `moe_iqr_device_test` 15/15,
`vision_rows_device_test` 66/10, `iqr_device_test` 76/11,
`attn_paged_device_test` 8/9, `r3d_selftest` **195 rows, 0 échec, 16 skips**.

**Reste.** Le préfill Flash-Next est maintenant dominé par `moe_gemm_q`
(56,5 % à 972 tokens, 1 711 ms inchangés) puis `hc_read` (17,6 %) ; le vrai
levier MoE est la **continuation de trellis** (0,80 ms des 2,59 ms de plancher
diagnostiqué) et, au-delà, le chunking du planificateur (9 appels par site
relisent les 512 experts), qui n'est pas du ressort de la row. `hc_read` garde
1,7× vers son plancher de matrices. `attn_dense` est clos sous contrat.

## 2026-11-18 — W33 : `gemm_nt_q` trellis en tuile registre 2×4 (livré)

**Cause mesurée.** Le décodage de l'unité 16×128 sur la vague 0 n'est **pas**
le limiteur : en le distribuant sur les quatre vagues (une colonne par voie),
on obtient **1,04×** à M=1626 et **0,997×** à M=512 — il est déjà recouvert par
les quatre blocs par CU, et un double buffer ne viserait que ≤ 4 %. Le
limiteur est le **flux d'activation et la chaîne `dot4`** : l'ancienne tuile
donnait à chaque voie une ligne de poids et huit tokens, soit deux chargements
par `dot4` et chaque dword d'activation consommé par une seule ligne.

**Correctif** (`libr3d/r3d_gemm_iqr_m16.hip`, `r3d_iqr_dec.h`). Une voie
possède **2 tokens × 4 lignes** (8 sorties) : un pas `c4` alimente 8 `dot4`
avec 2 dwords d'activation et 4 de poids, **0,75 chargement par `dot4`**, et le
dword d'activation est consommé par quatre lignes en registres. L'unité est
décodée par les quatre vagues en parallèle (`state1` dans le décodeur, ajout
purement additif : extraction d'état, hash masqué et somme d'octets identiques
à `rows()`, chacun écrit à sa place). 226 VGPR, aucun spill, 2 blocs/CU.
L'arithmétique par (token, ligne) est inchangée — codes entiers exacts, groupes
ascendants, deux facteurs f32 de gauche à droite, épilogue `row_scale` — donc
**sortie bit-identique**. Réservé à `kTrellisMad` avec `LRB = 0` et `M >= 128` ;
le décodage M=1 (`m16`) et la voie IQ8R groupe 256 ne sont pas touchés.

| 34816×5120 trellis | avant | après | gain | TFLOP/s |
|---|---:|---:|---:|---:|
| M=512 | 10 557 | **6 098** | 1,73× | 17,3 → 29,9 |
| M=1 024 | 21 355 | **11 969** | 1,78× | 17,1 → 30,5 |
| M=1 626 | 36 901 | **21 744** | 1,70× | 15,7 → 26,7 |
| M=2 048 | 46 667 | **28 549** | 1,63× | 15,6 → 25,6 |

**Servi** (27B, KV i8, NS=0, `profile-ops`, build neuf) : prompt de 3 252
tokens **TTFT 11 225 → 7 222 ms (1,55×)**, `gemm_nt_q` **10 048 → 5 954 ms
(1,69×)** sur 1 248 appels (préfill 288 → **450 tok/s**) ; prompt de 372 tokens
1 326 → **986 ms** ; décodage **28,84 → 28,93 tok/s** (inchangé) ; texte
**identique au bit**. `attn_paged` devient le deuxième poste (12,8 %) du
préfill 27B.

**Validation** : `iqr_device_test` 76/11, `sched_test` 548 952,
`engine_e2e_test` 1 076 (3 skips), `fixture_test` 109 671,
`ple_hc_device_test` 58/10, `moe_iqr_device_test` 15/15,
`vision_rows_device_test` 66/10, `attn_paged_device_test` 8/9,
`r3d_selftest` **195 rows, 0 échec, 16 skips** ; kbench ciblé `--op gemm_nt_q`
194/194, `rel_l2` **2,911e-03** inchangé.

**`hc_read`, N une fois par rangée : mesuré, non livré.** N est reconstruit
**12× par rangée et par appel** (11× en `down` — dix tuiles de sortie plus le
bloc d'injection — et 1× dans l'épilogue `up`). Le supprimer du `down` rend
**10,8–12,6 %** du core (M=4 096 : 12 825 → 11 446 µs), mais un scratch
`M × 10 240` bf16 coûte l'écriture (84 Mo), la relecture de `h` et une
reconstruction restante : **~8 % net, ≈1,09× à 4 096 et ≈1,11× à 1 024**, sous
la barre de 1,15×. Aucun code livré.

**Reste.** La tuile est à 30,5 TFLOP/s, soit 37 % du plafond f16 WMMA (81,9) et
au-dessus de notre tuile bf16 (23,5) ; au-delà, il faudrait une tuile 4×4
(spill de 88 VGPR mesuré), une autre arithmétique, ou un double buffer si
l'occupation baisse.

## 2026-11-18 — W34 : stager de préfill (livré, coût de décodage corrigé)

**Contexte.** Le préfill Flash-Next est dominé par `moe_gemm_q` (51 % à 972
tokens) et 19,08 GiB d'experts vivent dans le host pool : 35,5 GiB relus par
préfill, 1,86 s de MoE pour un plancher PCIe de 1,51 s (81 % du lien x16 à
25,2 GB/s). Le prompt de 972 tokens est coupé en **deux chunks par site**
(896 + 76), tous deux sous l'ancien seuil du stager : chaque couche routée
lisait ses experts sur PCIe **sur le chemin critique**.

**Tentative.** Réserver le ring de staging aux niveaux d'arène intermédiaires
(`arena_lent` exclut le ring, `arena_lent_full` le rend à la dalle pour les pas
qui ne stagèrent pas), ring publié pour les passes ≥ 256 rangées, seuil du
planificateur inchangé (aucun chunk, quantum ni ancre de `prefix-resume` ne
bouge). Mesuré : un chunk de 82 tokens stagé coûte plus qu'il ne rend
(0,81 s contre 0,72 s) et la queue de 67 rangées d'un prompt de 963 tokens
double les octets stagés sans gain — d'où le plancher à 256.

| servi, Flash-Next | sans stager | avec stager |
|---|---:|---:|
| préfill 972 tok | 2 497 ms | **2 102 ms (1,19×)** |
| `moe_gemm_q` à 972 tok | 1 717 ms | **1 319 ms (1,30×)** |
| préfill 3 252 tok | 6 037 / 6 092 | 6 066 / 6 104 (neutre) |
| préfill 92 tok | 801 / 803 / 800 | 801 / 800 / 802 (neutre) |
| décodage, ctx 92 tok | **30,5 tok/s** | 28,3 tok/s (**−7 %**) |
| décodage, ctx 3 252 tok | **19,35 tok/s** | 18,4 tok/s (**−5 %**) |

Texte identique aux deux contextes (le texte de W31 à 32 tokens est le préfixe
exact de celui de W35 à 64 tokens) ; régression reproductible à 1 ms près sur
deux runs par arbre.

**Mécanisme du coût, trouvé et corrigé (W35).** Le prêt de la dalle n'était
**jamais rendu** : un chunk stagé rétrécissait la part de la dalle pour réserver
le ring, et `arena_step` ne faisait que la rétrécir davantage. Après 64 pas de
décodage la dalle tenait **16 441 experts au lieu de 16 802** (arbre sans
stager), et le décodage relisait les 361 manquants sur l'hôte : −5 à −7 %,
reproductible. Le correctif rend le prêt plein (`set_loan`) au **premier pas
dont la passe ne stagère pas**, ce qui restaure exactement la résidence
(16 802) et le décodage au débit sans stager.

| Flash-Next | sans stager (W34) | stager seul (W35) | stager + prêt rendu (W36) |
|---|---:|---:|---:|
| préfill 972 tok | 2 497 ms | 2 105 ms | **2 120–2 133 ms (1,18×)** |
| `moe_gemm_q` 972 tok | 1 717 ms | 1 316 ms | **1 326 ms (1,30×)** |
| préfill 3 252 tok | 6 037 / 6 092 | 5 936 / 5 982 | 5 960 / 5 990 |
| préfill 92 tok | 801 / 803 / 800 | 804 | 816 / 825 |
| décodage ctx 3 252 | **19,35 tok/s** | 18,31 | **19,25 (−0,5 %)** |
| décodage ctx 92 | **30,5 tok/s** | 28,20 | **30,13 (−1,2 %)** |

Texte identique à 3 252×64 (même texte que W34) et aux autres tailles. Le
levier rentre donc dans la règle R43 : gain de TTFT de 1,18× pour **≤ 1,2 %** de
décodage, sous la barre des 2 %. Fusionné (`52fd19d`), puis les documents
corrigés (`5d67900` + cette entrée).

À noter pour la lecture des chiffres : le préfill de 3 252 tokens est **neutre**
ici ; le 8 186 → 5 964 ms observé entre W31 et W35 vient de la **tuile trellis
IQR** (W33), pas du stager.

**Chunk de queue : proposition examinée, non retenue (W36).** Le découpage de
972 tokens est **896 + 76** (le 704 + 99 est le cas 803, correction du chiffre
précédent). Il naît de `--prefix-resume exact` : `tail_checkpoint`
(`core/sched/scheduler.cpp:1292`, ancre 1297–1305, `kShortTailMin` 256 à
`scheduler.h:507`) pose une ancre à 896, et `plan_chunk` (1104, coupe 1131/1133)
s'arrête là. Le 76 relit **77 % des experts** en place, soit 14,0 GiB et
0,65–0,75 s des 2,04 s.

Fusionner la queue demanderait un snapshot linéaire **dans** le chunk (l'état
n'est capturé qu'en fin de chunk aujourd'hui), ce qui casse l'invariant « les
chunks du run froid » : un hit `exact` à 896 recalculerait 76 tokens dans une
forme jamais exécutée à froid, donc le texte ne serait plus garanti identique.
Relever `kShortTailMin` change la politique pour tous les prompts sous le seuil
et supprime l'ancre courte ; aucun signal « pas de reprise possible » n'existe,
le cache de préfixe VRAM restant actif même à `prefix-cache-mib: 0`.

**La mesure tranche contre l'arithmétique.** Budget 4096 et
`checkpoint-interval` 4096 sur le prompt de 3 252 (un chunk au lieu de
2 048 + 1 204) : **5 453 → 6 486 ms (+19 %)**, texte identique, décodage
inchangé. Un chunk plus gros ne réduit donc pas le TTFT ici : le coût d'arène
et de résidence dépasse la copie d'experts épargnée, comme le GUIDE §4.5
l'annonce. Le gain projeté de la fusion (~0,65 s) est du même ordre que ce que
cette mesure détruit, et il s'achète en cassant R22.

**Décision** : on ne touche pas au planificateur. Si le produit veut un jour le
gain *single-shot*, la seule forme acceptable est un drapeau
`--tail-anchor off` (cas court de `tail_checkpoint` rendant -1), jamais le
défaut, et il faut d'abord écrire trois tests (ancre courte linéaire à I=2048
dans `sched_test`, e2e froid/hit/tour appendu à I=2048, fixtures `--strict`).
R44.

## 2026-11-18 — W37 : `--tail-anchor off` (livré, opt-in)

**Ce qui est livré.** Le drapeau `--tail-anchor on|off` (défaut `on`) coupe
l'ancre courte de `Scheduler::tail_checkpoint` : avec `off`, un prompt sous le
premier intervalle s'exécute en **un seul chunk** et n'écrit **aucun snapshot
hors grille** (`scheduler.cpp`, `geometry.{h,cpp}`, `config.cpp`,
`rad_core.h`). Rien ne change à l'intervalle ni au-delà : le checkpoint de fin
d'intervalle est une fin de chunk, pas une ancre de queue. La ligne de
démarrage le dit quand le drapeau est actif.

**Mesure servi** (Flash-Next, même build, `flock`, `profile-ops` désactivé,
texte glouton) :

| cas | défaut | `--tail-anchor off` | |
|---|---:|---:|---|
| préfill 972 tok | 2 068 ms | **1 488 ms** | **1,39×** |
| décodage ctx 972 | 16,09 tok/s | 15,94 | −0,9 % (dans le bruit) |
| préfill 3 252 tok | 5 438 ms | 5 472 ms | neutre |
| décodage ctx 3 252 | 11,10 tok/s | 11,10 | identique |
| texte | — | — | **identique** aux deux tailles |

Le gain est exactement la queue de 76 tokens : un balayage du plan d'experts
en moins. C'est le seul levier de préfill qui restait, et il est cohérent avec
la proposition de W36 (fusion de la queue) sans en payer le prix : l'ancre
n'existe plus, donc rien ne peut être « recalculé dans une forme jamais
exécutée ». Le prix est explicite et mesuré : la conversation courte
re-préfille son prompt au tour suivant, ce que le e2e pin (0 snapshot,
`no_snapshot` compté).

**Tests** (les trois manquants de la proposition W36, adaptés) :
- `sched_test` `a_prompt_under_the_interval_runs_one_chunk_with_the_tail_anchor_off` :
  972 tokens / I=2048 → 896+76 et snapshot 896 par défaut, **un chunk** et
  aucun snapshot avec le drapeau ; 3 000 tokens identiques dans les deux cas.
  **548 964 checks** (+12).
- `engine_e2e_test`
  `a_conversation_under_one_interval_keeps_no_tail_snapshot_with_the_tail_anchor_off` :
  600 tokens, aucun snapshot écrit, tour suivant sans reprise
  (`reason="no_snapshot"`). **1 087 checks** (+11).
- `config_test` : parse `on|off`, refus de `later`, ligne de rapport.
  **2 927 checks**.

Suite complète (`build-w37`, build neuf) : `engine_e2e_test` 1 087 (3 skips),
`fixture_test` 109 671, `iqr_device_test` 76/11, `ple_hc_device_test` 58/10,
`moe_iqr_device_test` 15/15, `vision_rows_device_test` 66/10,
`attn_paged_device_test` 8/9, tous les autres device verts, `r3d_selftest`
**195 rows, 0 échec, 16 skips**.

**Quand l'utiliser** : serveur à prompt unique ou orienté préfill (batch
d'évaluation, génération longue) sur un modèle MoE où chaque chunk balaie le
plan d'experts. Le laisser **off** sur un serveur conversationnel : c'est là
que l'ancre courte fait son travail (M6, R22).

## 2026-11-18 — W38 : jalon validation (kbench complet, replay strict, `--tail-anchor` à 1 536/2 047/2 048)

**Arbre et build.** `6d9084b` rsyncé vers `w29-rebase`, build **neuf**
`build-w38` (Release, `-DRAD_GPU_TARGETS=gfx1100 -DRAD_WITH_FFMPEG=ON`), cache
de tune de `build-clean` recopié (**36 rows** chargées au démarrage). Aucun
changement de code : ce lot mesure, il ne livre rien.

**Suite de tests de la §5.5** — identique au bit aux compteurs de W37 sur un
build neuf : `sched_test` **548 964 / 213**, `engine_e2e_test` **1 087 / 52**
(3 skips), `fixture_test` **109 671 / 59**, `config_test` **2 927 / 71**,
`iqr_device_test` 76/11, `ple_hc_device_test` 58/10, `moe_iqr_device_test`
15/15, `vision_rows_device_test` 66/10, `attn_paged_device_test` 8/9,
`logits_bf16_device_test` 8/4, `gather_rows_u8_device_test` 33/3,
`mtp_enter_e4m3_device_test` 22/3, `logit_rerank_iqr_device_test` 36/5,
`act_i8g_group256_device_test` 103/6, `tp2_collectives_device_test` 12/2,
`r3d_selftest` **195 rows, 0 échec, 16 skips**.

**kbench complet, trois conteneurs** (`--bench`, un conteneur à la fois sous
`flock`, `build-w38`) :

| conteneur | cas | échecs | divergences | pire `rel_l2` | mur |
|---|---:|---:|---:|---:|---:|
| 27B (iq4r-l) | **3 181 / 3 181** | 0 | 0 | 3,619e-03 | 542 s |
| MiniCPM5-2B (IQ8R) | **1 802 / 1 802** | 0 | 0 | 3,485e-03 | 368 s |
| Flash-Next (IQR v4) | **4 786 / 4 786** | 0 | 0 | 1,520e-02 | 2 310 s |

Comptes **identiques** à la référence W29 d'`AGENTS.md`, pires `rel_l2`
identiques : `hc_read` (W31/W32), `gemm_nt_bias` (W31) et la tuile trellis de
`gemm_nt_q` (W33) ne dégradent rien. Les `skipped` restent ceux des oracles
absents (`gemm_nt_q`/`gemm_nt_q_gated` décrits par `libref` lui-même,
`gated_had_quant_i8g`, `rmsnorm_had_quant_i8g`, `ngram_hint` x8).

**Replay strict des fixtures** (la forme du ctest `kbench_ref` :
`rad-kbench --kernels libref --max-cases 1 --strict --coverage data/kbench_rows
--fixture …/kernels.rkb --fixture …/kernels_moe.rkb`) : **1 134 / 1 134**,
0 échec, 0 stale, 0 divergence, 211 capped, pire `rel_l2` 3,442e-03, 80 s.
`fixture_test` n'a **pas** d'option `--strict` (usage : `[pattern...] [--list]
[--shard i/n]`) : la ligne de la §6.A.3 du handover W37 est corrigée.

**`--tail-anchor off` au-delà de 972 tokens.** Protocole : cache de préfixe
**actif** (le drapeau n'agit que si `prefix_cache` est vrai,
`scheduler.cpp:1306`), mais chaque requête porte un **nonce de tête neuf** :
`cache_n = 0` et `prompt_n` exact sont vérifiés sur chaque point ; la longueur
en caractères est calibrée par **bissection** sur `prompt_n` (les pas de 5
caractères sautent de 2–3 tokens aux frontières BPE, 2 048 était inatteignable
par pas fixes). Le payload gagnant du serveur `on` est **rejoué à
l'identique** sur le serveur `off` (`temperature 0`, `max_tokens 8`), donc les
textes sont comparables. Deux réplicats indépendants (nonces différents,
serveurs relancés) :

| modèle | prompt | `on` (ms) | `off` (ms) | gain | texte |
|---|---:|---:|---:|---:|---|
| Flash-Next | 1 536 | 2 646 / 2 704 | 2 282 / 2 298 | **1,16–1,18×** | identique |
| Flash-Next | 2 047 | 3 395 / 3 434 | 2 938 / 2 968 | **1,16×** | identique |
| Flash-Next | 2 048 | 3 000 / 3 025 | 2 952 / 2 978 | 1,02× (neutre) | identique |
| 27B | 1 536 | 3 368 / 3 394 | 3 154 / 3 147 | **1,07–1,08×** | identique |
| 27B | 2 047 | 4 810 / 4 830 | 4 554 / 4 554 | **1,06×** | identique |
| 27B | 2 048 | 4 616 / 4 618 | 4 576 / 4 571 | 1,01× (neutre) | identique |

Le décodage (`NS=0`, 8 tokens générés : **indicatif**) ne bouge pas
(13,2–14,6 tok/s Flash-Next, 17,1–19,3 27B), et les textes sont identiques sur
les 12 paires.

- **R46** — *règle (W38)* : le gain de `--tail-anchor off` vit dans
  `(kShortTailMin, I)` quand le dernier chunk est **partiel** : 1,39× à 972,
  1,16–1,18× à 1 536, 1,16× à 2 047, puis **1,01–1,02× à exactement I**. À
  un intervalle plein il n'y a pas de queue partielle à ancrer : `on` et `off`
  convergent. Attendre un gain au-delà de I est une erreur d'arithmétique,
  pas une régression.

Artefacts : `<work-root>/work/w38-results/{flashnext,dense27b,minicpm}.{md,log}`,
`strict.log`, `ta4/`, `ta5/`.

## 2026-11-18 — W38b : 27B en `NS=3` — bande KV corrigée, blocage de placement mesuré

**But** : le levier §6.B.1 (« 27B en NS=3, ~1,5× sur le décodage, effort
faible »). Mesuré d'abord sans toucher au code, `build-w38`, prompt de
2 047 tokens, 64 tokens générés, une requête par serveur neuf :

| configuration | démarrage | décodage |
|---|---|---:|
| KV i8, NS=0 | oui | **15,35–15,39 tok/s** |
| KV i8, NS=3 | **non** | — |
| KV bf16, NS=0 | **non** | — |
| KV bf16, NS=3 | **non** | — |

- **i8 + NS=3** : `attn_paged`/`kv_store` du **drafter** (head_dim 128, gqa 4,
  fenêtre 2 048) ne sont servis qu'en **bf16/fp8_e4m3** ; les rows int8 sont
  celles du tronc (head_dim 256). Trois bandes device non résolues → refus.
- **bf16 + NS=0** : symétrique — le **tronc** h256 n'a pas de row bf16 dans
  `libr3d`. Un seul `--kv-cache-dtype` global ne peut donc pas servir les
  deux : les deux groupes ont des formes de tête différentes.

**Correction livrée** (`ee291d2`) : `drafter_kv_group` accepte une dtype propre
au groupe ; `qwen35_fp8` garde le cache du drafter en **bf16** quand le tronc
est en i8 (note au démarrage). `tests/arch_test.cpp` porte le nouveau contrat
(`qwen35_i8_trunk_keeps_the_int8_group_and_the_drafter_its_bf16_cache`) : tronc
i8 h256, drafter bf16 h128, chaque op sur sa dtype. Suite complète sur
`build-w39` verte : `arch_test` (après mise à jour), `sched_test` 548 964,
`engine_e2e_test` 1 087 (3 skips), `fixture_test` 109 671, `config_test`
2 927, tous les device (dont `attn_paged_device_test` 8/9), selftest
**195/0/16** ; kbench ciblé 27B (`attn_paged`, `gemm_nt_q`, `mtp_enter`,
`logit_rerank`, `dflash_conv`) **854/854**, 0 divergence, pire `rel_l2`
2,976e-03.

**Blocage suivant, mesuré** : le démarrage échoue désormais plus loin, sur
`'dflash.blk.0.attn_norm.weight' was placed on the mapped tier ... (widened at
load)`. `--debug-placement` donne la cause : **`dflash_conv` ne résout qu'un
kernel hôte** dans `libr3d`, donc tout le groupe de poids du drafter est épinglé
sur le *mapped tier* (lecture en place du conteneur) ; les normes du drafter
sont déclarées f32 alors que le conteneur les stocke en bf16, ce qui exige un
élargissement au chargement — incompatible avec la lecture en place. Forcer
`--placement all_vram` ne change rien (le groupe hôte reste épinglé).

- **R47** — *constat (W38)* : `NS=3` sur le 27B **n'est pas une recette**. Il
  faut au minimum une **row device `dflash_conv`** dans `libr3d` ; sans elle le
  bloc drafter est hôte, ses poids vont sur le *mapped tier*, et ses normes
  élargies font échouer le chargement. Le prérequis KV (cache du drafter en
  bf16) est livré ; le reste est du travail de noyau, donc hors du périmètre
  sans feu vert. Tant que la row manque, le 27B reste servi à `NS=0`
  (**15,4 tok/s** à 2 047 tokens, 64 tokens générés, KV i8).

Artefacts : `<work-root>/work/w38-results/ns27b/` (refus W38),
`ns27b39/`, `ns27b39v/` (build-w39), `w39-27b-targeted.{md,log}`.

**Contrôles du changement** : texte 27B `NS=0` **identique** entre `build-w38`
et `build-w39` (même prompt 2 047 tokens, 64 tokens gloutons, 309 caractères
au bit) ; sweep 27B **entier** sur `build-w39` — **3 181/3 181**, 0 échec,
0 divergence, pire `rel_l2` 3,619e-03 (identique à W38) ; ciblé
(`attn_paged`, `gemm_nt_q`, `mtp_enter`, `logit_rerank`, `dflash_conv`)
**854/854**.

**Vision re-mesurée** sur l'arbre de référence (`build-w38`, sans
`profile-ops`) : image 1 024² → 1 045 tokens, **TTFT 3,28 s** (contre 4,05 s
daté W32) ; 2 048² → 4 117 tokens, **TTFT 16,33 s** (contre 16,97 s). Le gain
de 1 024² vient des lots W33–W34 (tuile trellis, stager dès 256 rangées, ancre
de queue par défaut), pas d'un changement de ce lot. Les textes restent
cohérents.

**Flash-Next, décodage re-mesuré** sur `build-w38` (prompt 3 252 tokens,
64 tokens générés, un serveur neuf par configuration) : `NS=0`
**11,03 tok/s** (préfill 5 473 ms), `NS=3` **19,29 tok/s** (préfill
6 003 ms, **+9,7 %** : la passe de contexte du drafter se paie au préfill),
texte **identique** (313 caractères). Les chiffres de §5.2 d'`AGENTS.md`
(11,10 / 19,25) sont donc toujours justes ; le coût de préfill du MTP, lui,
n'était pas chiffré.

## 2026-11-18 — W38c : `dflash_conv` servi sur la carte, 27B en `NS=3`

**Feu vert** pour la row device que R47 exigeait. Livré `f63dffd` :

- `libr3d/r3dh_dflash_conv_body.h` — calcul dans un en-tête sans HIP (revue ISA
  au clang nu), transcrit de `ref_dflash_conv` : `out[t][h]` = somme des taps
  `j` avec `j == 0` ou `t % block >= j` de `(base[j][h] + delta[t][j][h/16]) *
  x[t-j][h]`, **ordre des taps croissant, un arrondi bf16 au store**, ce qui est
  bit-exact parce que toute la bibliothèque est compilée `-ffp-contract=off`.
  Un thread = 8 canaux (16 octets en lecture et en écriture) ; le groupe 16 est
  un multiple de 8, donc l'indice de groupe et les deux gains delta sont
  uniformes.
- `libr3d/r3d_dflash_conv.hip` — grille `(H/8/128, T)`, masque de bloc par
  **modulo** (donc tout `block_size >= 1` est servi, alors que libr4d se limite
  aux puissances de deux) ; `delta` lu **au pas de la projection** `2*taps*NG`
  (un pas compacté refusé : il lirait l'autre moitié des coefficients) ; `out`
  ne peut pas aliéner `x` (le tap 1 lit la ligne `t-1`).
- Contrat : `taps == 2`, `group_size == 16`, `hidden_size % 128 == 0`, bf16 ;
  la forme décrit `delta` au rang de la projection `[T, 2, taps, NG]` pour que
  le pas soit porté. `r3d_selftest` passe à **196 rows, 0 échec, 16 skips** et
  `tests/dflash_conv_device_test.cpp` (**37 checks, 3 cas**) compare au bit
  8 géométries : bande 27B (H 5 120, NG 320, bloc 4), les deux moitiés de la
  projection, `T = 10` avec bloc 4 (rangs 0 et 4 sans tap décalé), bloc 3 (non
  puissance de deux), bloc 1, bloc 32 sur `T = 8`, `T = 1`, `H = 128` ; plus les
  deux refus (alias, pas compacté). Suite complète verte sur `build-w39`
  (sched 548 964, e2e 1 087/3 skips, fixture 109 671, config 2 927, tous les
  device, selftest 196/0/16).

**27B servi** (`build-w39`, KV i8, prompt 2 047 et 135 tokens, 64 tokens
générés, deux serveurs indépendants par point) :

| prompt | `NS=0` | `NS=3` | gain | acceptation | texte |
|---:|---:|---:|---:|---:|---|
| 2 047 tokens | 15,36 / 15,34 | 15,79 / 15,75 | **1,03×** | 70 % | identique |
| 135 tokens | 27,89 / 27,69 | 34,42 / 33,97 | **1,23×** | 64 % | identique |

Le préfill paie 1,6–2,3 % au profit du drafter (4,72 → 4,82 s à 2 047). `NS=2`
est refusé par l'arch lui-même (`rad_arch_declare` → unsupported) : le bloc
entraîné est 4.

- **R47 clos** : la row `dflash_conv` lève le blocage de placement ; le prérequis
  KV (`ee291d2`) et celui-ci étaient les deux pièces manquantes.
- **R48** — *mesure (W38)* : `NS=3` sur le 27B vaut **1,23× à 135 tokens mais
  1,03× à 2 047**. L'arithmétique des pas (64 tokens à 70 % d'acceptation
  ≈ 21 verify + 21 passes de drafter en 4,04 s, contre 65 ms par token à
  `NS=0`) situe le verify à quatre rangées autour de 2,3× un pas à une rangée
  et le drafter autour de 45 ms : le « 1,5× » estimé en W37 supposait un verify
  quasi gratuit, faux sur un tronc dense compute-bound. Servir le 27B à `NS=0`
  pour un contexte long ; `NS=3` n'a d'intérêt que quand la cible est
  memory-bound (contexte court), et il ne change pas le texte glouton.

Artefacts : `<work-root>/work/w38-results/{w39-dflashconv.md,ns27b40/,
ns27b_scan/,ns27b_final/}`.

**Couverture kbench** : le sweep 27B complet reste **3 181/3 181**, 0 échec,
0 divergence, pire `rel_l2` 3,619e-03, mais les **112 cas `dflash_conv`** y sont
« skipped » : le graphe déclare `base` avec la plane entière (20 480 éléments,
les deux moitiés) et le site d'issue la tranche au runtime
(`wrow_slice(w_acb[i], 0|1, 1, taps*H)`), ce que l'outil ne sait pas
reconstituer depuis la déclaration — il refuse d'inventer la moitié plutôt que
de vérifier contre la mauvaise. La row est couverte par
`dflash_conv_device_test` (les deux moitiés, pas de projection compris), et le
service la valide par le texte glouton identique à `NS=0`.

## 2026-11-19 — W40a : `attn_dense` vision, les deux sorties essayées et rejetées

Le poste était dit « 53 % de sa borne VALU » (§6.B) : **faux**, et c'est la
première chose que la mesure a corrigée. Le noyau est borné par la **latence**
de la chaîne de 72 ops du dot, que 4 clés en parallèle ne masquent pas.

**FMA** (437 → 221 VALU/clé, 2 339 → 1 238 instructions flottantes, VGPR et
LDS inchangés) : **241,3 ms au lieu de 233,6 ms** à 16 384 patches × 16 têtes
(3 runs entrelacés, 5 127 contre 5 296 GFLOP/s) — **3,3 % plus lent**, aucun
écart numérique à mesurer puisqu'il perd sur la vitesse.

**Softmax en une passe, scores de tuile en LDS** (8 KiB transposés, max courant
par tuile, rescale de `sum` et des 72 `acc[d]` par `exp(m_old − m_new)`) :

| forme | avant | une passe |
|---|---:|---:|
| 16 384 × 16 têtes | 237,1 ms | **162,9 ms (1,46×)** |
| 4 096 × 16 | 17,07 ms | 12,05 ms (1,42×) |
| 512 × 16 | 397,9 µs | 285,7 µs (1,39×) |
| TTFT 1 024² (1 045 tokens) | 3,33 s | 3,19 s |
| TTFT 2 048² (4 117 tokens) | 16,79 s | **14,63 s (1,15×)** |

Écart numérique (16 384 × 16, q/k/v aléatoires) : 24 197 / 18 874 368 codes
bf16 diffèrent (0,128 %, dont 92,8 % d'un seul code), `rel_l2 = 8,84e-5`,
`max |a−b| = 1,22e-4` ; kbench reste à `rel_l2 ≤ 4,47e-6` vs libref (40/40) et
le cas uniforme d'`iqr_device_test` reste exact au bit.

**Mais le texte servi change** sur les deux images (1 024² et 2 048², deux runs
par variante): la variante est **rejetée par le gate**. Livré `9577eda3` :
commentaire seul dans `libr3d/r3d_vision_bf16.hip`, avec la preuve que le flux
d'instructions device du `.so` reconstruit est **identique** aux 7 278
instructions de référence. **R40 reste valide** ; rouvrir demande un store
exact qui tient, ou un nouveau contrat de texte servi.

Les diagnostics jetables qui donnent le plafond : supprimer le dot de la marche
2 = **1,45×**, supprimer la marche 1 = **1,48×**, supprimer le staging K de la
marche 2 = 2 %.

## 2026-11-19 — W40b : `hc_read` gagne 5–9 % sur la plage servie, la marge 1,7× est hors contrat

Profil (`rocprofv3 --kernel-trace`, `RAD_HC_BENCH_M=512`) : `down` 866 µs
(46 %), `up` 707 µs (39 %), `norm` 218 µs (12 %), `quant` 49 µs — le levier est
le down. Livré `a4ed7281` + `8401e6fd` (fusionnés) : tuile down **2×2×64**
(`hc_down_kernel_p2q2c64`) au-delà de 256 rangées, la tuile 1×2×128 restant
sous ce seuil ; deux cas d'oracle à `M=257` (E4M3 `lr=128`, bf16 `lr=20`)
couvrent le stage partiel, `ple_hc_device_test` passe de 58 à **76 checks**.
Bit-exact vs libref, textes servis identiques aux trois points.

| M | avant 1×2×128 (µs) | après 2×2×64 (µs) | gain |
|---:|---:|---:|---:|
| 128 | 734 | 840 | tuile étroite conservée |
| 512 | 1 953 | 1 849 | 5,3 % |
| 896 | 3 589 | 3 292 | 8,3 % |
| 2 048 | 7 197 | 6 610 | 8,2 % |
| 4 096 | 13 576 | 12 396 | 8,7 % |

Servi (NS=0, KV i8) : texte 3 252 tokens 5 443 → 5 397 ms (−0,85 %), image
1 024² 3 193 → 3 182 ms, 2 048² 16 474 → 16 451 ms ; décodage 11,10 → 11,08
tok/s (dans le bruit, chemin inchangé) ; textes identiques au bit à la
référence W38.

**Rejeté, sur mesure** : plans f32 pré-décodés (perdent 26–30 % à M=128–512,
le gain VALU est annulé par 4× d'octets de poids relus en L2), prefetch des
codes en registres (68→83 VGPR, 2 workgroups/CU au lieu de 3, neutre à −1 %),
tuiles P2Q4/P4Q2 (occupation à 2 blocs pour ≤2 % d'instructions en moins),
WMMA / `v_dot2` (ordre d'accumulation différent → interdit par le contrat),
découpage du dot entre lanes (casserait la somme ascendante).

**Conclusion** : le dot est à 2,75 instructions/MAC contre un plancher de 2,0 ;
le reste est de la latence à 24 vagues. La marge de **1,7×** de §6.B n'est pas
atteignable par vectorisation, largeur de lane ou staging sous contrainte
bit-exacte : c'est un constat, pas un reste à faire.

## 2026-11-19 — W40c : budget d'échanges du moteur de chaleur — −21 % d'octets de promotion, −1,75 % de décodage

Le « 63 Go/token » de §5.2 est la **taille du plan d'experts** (24 576 ×
2,56 Mo), pas les octets lus : la mesure donne **1,20 Gio d'experts routés par
token** à 3 252. Le lien réellement sollicité, lui, était dominé par les
**promotions** : 64 échanges par étape et par classe recopiaient des experts que
les étapes suivantes auraient touchés, sur un plan déjà résident à 66 %.

Livré (`49f9317a`, `ab38e620`, fusionnés) : `heat_params_for(plan, moves)`
abaisse le budget à **16** quand ≥ 60 % des **octets** d'experts mobiles sont en
VRAM, **après un broad pass** (histogramme ≥ 4 096 sélections, ~410 rangées à
top_k 10) et pour **512 étapes**, puis revient à 64 ; un prompt court garde 64
dès la première étape ; `--expert-moves N` (0 = auto, plafond 1 024) force le
budget et s'applique dès la première étape. Le partage est une décision
**d'union sur les classes** (documentée) ; un prompt chunké sous ~410 rangées
n'arme pas et retombe sur 64.

| décodage 3 252, NS=0, 64 tokens | avant | après |
|---|---:|---:|
| lus en place (non résidents) | 376,9 Mio/token | 365,8 Mio/token |
| promotions h2d | 160,4 Mio/token | **57,5 Mio/token** |
| **total sur le lien** | **537 Mio/token** | **423 Mio/token (−21 %)** |
| hit rate VRAM | 76,3 % | 77,0 % |

| point servi | NS=0 | NS=3 |
|---|---:|---:|
| TTFT 972 (8 tok) | 2 108 → 2 091 ms | 2 318 → 2 306 ms |
| TTFT 3 252 (8 tok) | 5 463 → 5 383 ms | 5 725 → 5 680 ms |
| décodage 3 252 (64 tok) | 5 949 → **5 845 ms (−1,75 %)** | 3 771 → 3 745 ms (−0,7 %) |
| décodage 972 (17 tok) | 1 285 → 1 274 ms | 541 → 535 ms |
| décodage 92 (64 tok, 3 reps) | 4 192 → 4 192 ms | 1 740 → 1 740 ms |

Textes identiques au bit partout ; suite complète verte (`config_test`
2 952 checks, `place_test` 52 018/101 cas), kbench Flash-Next **4 786/4 786**,
0 divergence. **Rejeté, sur mesure** : `kXpSlots 2→4` (décodage −1,4 %, bruit),
budget 16 **global** (+6,3 % à 92 tokens, hit 75,2 → 70,6 %), budget 48 global
(+0,85 % à 92), sans chaleur/XP (`--deterministic`, 6 746 contre 5 910 ms),
changement de précision d'un plan de poids (texte).

**Ce que ça dit** : à 3 252 le décodage est **latency-bound** (`hc_read`
28,6 ms, `attn_paged` 25,4, `moe_gemm_q` 20,7 sur 92,4 ms ; lien à ~31 % à
NS=3). Les octets du MoE ne sont plus le levier : −21 % d'octets de promotion
ne rendent que 1,75 %. Ne pas rouvrir un chantier « octets/token » sans
profiler d'abord.

## 2026-11-19 — W40d : l'aréna de scratch n'était jamais allouée — 27B `NS=3` à 1,74×

Le profil du pas à 4 rangées (2 047 tokens) désigne le coupable : `NS=0` passe
68,0 ms dont `attn_paged` h256 34,1 ; `NS=3` passe 194,6 ms dont **88,2 ms dans
l'attention du drafter** (5 couches × 17,6 ms), à `splits=1`. Cause :
`attn_scratch` (h128) et `attn_h256_scratch` lisaient `a->t[0]`, or le builder
ne passe **jamais** de tenseurs à un hook de scratch
(`core/build/rad_builder.cpp:583`) : les deux hooks répondaient 0 et le
split-KV était **silencieusement désactivé en service** depuis toujours.

Livré (`1943d0a2`, garde `276348c2`, fusionnés `b81e6269`) :

- `attn_scratch` (h128) dimensionne l'aréna depuis la géométrie gelée
  (`q_len × max_seqs`), 16 tranches ; `pick_splits` vise **128 workgroups**
  (le drafter a 8 têtes KV et 1 tuile, 32 workgroups laissaient 96 CU au
  tiers d'occupation) : **17,6 → 1,7 ms** par couche, **88 → 8,5 ms** par pas.
- Le noyau m16 reproduit, au-delà de M=1, le **bracketing par tranches** du
  GEMV split-K (mêmes tranches de 128-K, même ordre, une seule échelle) : les
  rangées du verify sont **bit-identiques** au pas à une rangée.
- **Garde `causal == 0`** : seuls les blocs spéculatifs prennent le split. Un
  tronc causal (`causal == 1`) répond 0, reste à `splits=1`, et sa sortie est
  bit-identique **par construction** — un texte servi qui bouge est un refus,
  même dans la tolérance documentée de 1 bf16 ULP.

Mesures servies (`build-w40f`, KV i8, 64 tokens générés, serveur neuf par point) :

| modèle / prompt | NS=0 | NS=3 | gain | acceptation | texte |
|---|---:|---:|---:|---:|---|
| 27B, 2 047 | 15,44 | **26,94** | **1,74×** | 70 % | identique |
| 27B, 139 | 27,3 | **39,3** | **1,44×** | 72 % | identique |
| Flash-Next, 3 252 | 11,31 | **18,99** | 1,68× | 54 % | identique |

Préfill : 27B 4 612 → 4 712 ms (+2,2 %), Flash-Next 5 360 → 5 876 ms
(+9,6 %, la passe de contexte du drafter, cohérent avec W38).

**R48 est caduc** : le « verify M=4 à 2,3× » n'était pas la cause — le coût
était l'attention du drafter privée de split. À 2 047, `NS=3` vaut 1,74×, et le
verify bracketté est bit-identique.

**Écarté, sur mesure** : le split-KV du **tronc h256** en service. Corriger
l'aréna fait passer le pas `NS=0` de 68 à **39 ms (1,73×)** et `NS=3` à
35,7 tok/s, mais le texte glouton de `NS=0` change sur ce prompt
(réassociation à 1 ulp bf16). Refusé : `NS=0` doit rester identique.

**R49 — le tronc h128 de MiniCPM était lent, et c'est un bug préexistant.**
Mesuré (même binaire, `libr3d` pristine de `fa7e2823` contre le nouveau) :
prompt 969 tokens, 60 générés, `NS=0` — pristine **2,48 tok/s**, nouveau
28,15 ; prompt 999 — 2,41 contre 25,82 ; prompt 59 — 22,18 contre 63,01. La
cause est la même (aréna à 0, `splits=1` sur une seule tuile). Le « 40,2 tok/s »
d'`AGENTS.md` n'est pas reproduit au point documenté (22,18 en pristine,
63,01 avec le split). Avec la garde, le tronc reste à `splits=1` et les textes
sont **identiques au pristine** aux trois prompts (`NS=0` et `NS=7`,
`prompt_n` 59/781/804) : **porte PASS**. Décision produit à prendre : accepter
le re-baseline du texte MiniCPM (tolérance 1 bf16 ULP documentée) pour gagner
10–11× sur son tronc, ou garder le texte actuel et traiter sa lenteur
autrement. Rien n'est changé en attendant.

**Validation du build fusionné.** `w40f` = `1943d0a2` + `276348c2` + les W40a–c
(`b81e6269`) : suite complète verte (`sched_test` 548 964, `engine_e2e_test`
1 087/3 skips, `fixture_test` 109 671, `config_test` 2 952, `ple_hc_device_test`
76, `dflash_conv_device_test` 37, selftest 196/0/16), kbench **27B 3 181/3 181**
(3,619e-03), **MiniCPM 1 802/1 802** (3,485e-03), **Flash-Next 4 786/4 786**
(1,520e-02), 0 échec, 0 divergence — pires `rel_l2` identiques à W38. Gates
servis : 27B **1,74×** et Flash-Next **1,68×** texte identique, MiniCPM textes
identiques au pristine à 59/781/804 en `NS=0` et `NS=7`. Artefacts :
`<work-root>/work/w40-results-b/{27b,minicpm,flashnext}.{md,log}` et les
textes/gates servis du même dossier.

**Bilan de la vague W40, pour le tableau des gains** : `hc_read` **+5,3–8,7 %**
bit-exact sur la plage servie (TTFT −0,1 à −0,9 %, `f498bc06`) ; budget de
chaleur **−21 % d'octets** sur le lien et décodage 3 252 **−1,75 %**
(`d8a28952`) ; 27B `NS=3` **1,74×** à 2 047 et 1,44× à 139 (`b81e6269`) ;
Flash-Next `NS=3` 18,99 contre 11,31 (1,68×) ; vision `attn_dense` **rejetée
sur mesure** (1,46× au noyau, texte servi changé).

## 2026-11-19 — W40e : le « 52–67 tok/s agrégés » du 27B est un chiffre de contexte court

Mesure du multi-séquence sur `w40f` (4 requêtes distinctes, 64 tokens générées
chacune, KV i8, `max-num-seqs 4`), avec un second tour à préfixes cachés quand
le cache mord :

| point | agrégé | par flux | `cache_n` |
|---|---:|---|---|
| 4 flux, ~256 tokens (tour 2) | **51,3 tok/s** | 13,0–15,3 | 256 (chaud) |
| 4 flux, ~256 tokens (tour 1, préfills entrelacés) | **61,2 tok/s** | — | — |
| 4 flux, 2 047 tokens, `NS=0` | 8,5–10,8 tok/s | 4,8–15,1 (inégal) | 0 |
| 4 flux, 2 047 tokens, `NS=3` | **15,1–15,6 tok/s** | 7,7–19,3 | 0 |
| 1 flux, 2 047 tokens | 15,4 (`NS=0`) / 23,7–26,9 (`NS=3`) | — | — |

Deux constats :

1. Le chiffre documenté (§5.4, « 4 séquences ≈ 52,3 tok/s agrégés ») est un
   chiffre de **contexte court** (256 tokens ici : 51–61 tok/s). Il ne se
   transporte pas à 2 k.
2. À 2 047 tokens, quatre flux **ne multiplient pas** le débit : 8,5–10,8 tok/s
   agrégés en `NS=0`, soit **moins qu'un seul flux** (15,4), et le partage est
   inégal (deux flux à ~13–15, deux à ~5–8). Le préfixe n'est pas réutilisé à
   cette longueur (`cache_n = 0`), donc la fenêtre inclut les préfills des
   autres flux : c'est un chiffre **de service**, pas un débit de décodage
   pur. `NS=3` remonte l'agrégé à **15,1–15,6**, c'est-à-dire au niveau du
   simple flux.
3. Conséquence pratique : sur le 27B à contexte long, la spéculation est le
   levier qui rend le multi-flux supportable ; augmenter `max-num-seqs` ne
   l'est pas. À contexte court, l'inverse (quatre flux à 51–61 tok/s pour des
   flux à 13–15 chacun).

Limite de la mesure : à 2 k, `cache_n = 0` au second tour, donc l'agrégat mêle
préfill et décodage ; ne pas comparer ces chiffres à un débit de décodage pur.
Artefacts : `<work-root>/work/w40-results-b/{multiseq,batch}/`.

## 2026-11-19 — W41a : décodage 27B — l'attention est la moitié du pas, le GEMM à 76 % du plafond

Profil par op du pas servi (delta entre deux blocs consécutifs, 68,2 ms
cumulés ≈ 65 ms réels, 2 047 tokens, NS=0, une rangée) :

| op | ms | part | appels |
|---|---:|---:|---:|
| ttn_paged h256 | 34,42 | **50,5 %** | 16 |
| gemm_nt_q + gemm_nt_q_gated | 27,52 | **40,4 %** | 288 |
| 
msnorm_had_quant_i8g | 2,21 | 3,2 % | 128 |
| logits_gemm_vocab, gdn_conv, gemm_nt | 2,80 | 4,1 % | 97 |

Le « 267 Go/s » de W40e décrivait le **pas entier** ; le GEMM seul lit
**15,25 Gio à ~555 Go/s**, contre un plafond pratique mesuré de **726–734 Go/s**
(
eadbw) et 611–676 Go/s sur le motif de tuile réel. Le noyau trellis n'est
pas le problème : décodage des codes 4–8 % du temps, barrières ~0, les charges
seules 90–96 %. Sur les 34,4 ms d'attention, le split-KV h256 est **interdit en
service** par le contrat de texte (W40d) : c'est la moitié du pas qui n'est pas
déplaçable sans changer l'arithmétique.

Livré (651d0bfc, fusionné) : à M ∈ {2,4}, le dot i32 de 128 K est coupé en
**quatre chaînes indépendantes** (somme i32 exacte, bracket et ordre f32
inchangés), et la grille split-K passe **tranche-major** (lockIdx.x =
tranche) pour la localité des poids. Bit-exact.

| point servi (A/B alterné, 3 paires) | avant | après | gain |
|---|---:|---:|---:|
| NS=0, 2 047 | 4 192,3 ms / 15,03 tok/s | 4 163,3 / 15,13 | +0,7 % |
| NS=3, 2 047 | 2 419,4 ms / 26,45 tok/s | 2 398,0 / 26,69 | +0,9 % |
| NS=0, 139 | 2 330,3 ms / 27,04 tok/s | 2 298,5 / 27,41 | +1,4 % |
| NS=3, 139 | 2 041,8 ms / 30,85 tok/s | 1 994,2 / 31,59 | +2,3 % |

kbench : gemm_nt_q **1,044×** à M=1 et **1,058×** à M=2/4 en moyenne, cible
5 120×6 144 M=1 **47,96 → 40,28 µs (1,19×)**, aucune row > 2 % plus lente ;
27B **3 181/3 181**, 0 divergence, pire 
el_l2 3,619e-03 ; suite complète
verte, iqr_device_test 76 (split-K down : 0 ulp bf16). Textes identiques au
bit aux références W40 (2 047, NS=0/NS=3) et à 139.

**Écarté, mesuré** : continuation coalescée (neutre), __syncwarp dans le fold
(neutre), MAD trellis simplifié (4–8 %, pas un levier), split-KV h256 en
service (1,73× mais texte changé), regroupement des tranches (le swap de grille
capte déjà la localité).

- **R50** — *constat (W41a)* : le pas de décodage du 27B est à **50,5 %
  d'attention** et 40,4 % de GEMM, ce dernier à **76 % du plafond de bande
  passante de la carte**. Sous contrat bit-exact, il n'y a plus de levier
  supérieur à ~1,06× sur le GEMM ; les 34 ms d'attention ne se déplacent qu'en
  changeant l'arithmétique (le split-KV du tronc, refusé pour le texte). Ne pas
  rouvrir un chantier GEMM de décodage 27B sans accepter ce constat.

## 2026-11-19 — W41b : le tronc h128 de MiniCPM par bande de têtes — 11× bit-exact, R49 clos

**Diagnostic** (ablations sur le noyau large, 969 clés) : la marche KV est
**gratuite** (chargements sur adresse chaude : 9,0 ms inchangé) ; score_of en
passe 1 = 310 µs/couche ; **PV = 8,25 ms des 9,0 ms**, dont 7,65 ms pour le
seul 16_to_f32 de V. Cause : cc[128] + qf[64] saturent les 256 VGPR, le
widening spille, et les broadcasts LDS se scalarisent en _readlane. Le
split-KV ne faisait que paralléliser ce coût — d'où son 28,15 tok/s
non bit-identique.

**Solution** (9e701396, fusionné) : découper la **dimension de sortie**, pas
les clés. Un workgroup = une tuile de R lignes × une tête, 32 lanes sur les
lignes, un lane possède 128/(32/R) dims. La marche, l'ordre dot2 du QK, la
chaîne cc[d] += w·v[d] sur j croissant, mx, sum += exp(sc−mx) et
f16x2_rne sont **inchangés** ; seule la lane qui porte un élément change. À
R=1 le score est calculé une fois et gardé en LDS (plan dynamique 4·jcap,
jusqu'à 12 288 clés). 	ests/attn_paged_device_test.cpp passe à **29 checks /
21 cas h128** (q_len 1→16, causal et non causal, fenêtres, gqa 1/4/8, 1→5
séquences, rectangulaire et cu_seqlens) comparés **au bit** au noyau large.

| noyau (µs/couche, 1 séquence) | large splits=1 | bande | × |
|---|---:|---:|---:|
| ctx 969, q=1 | 9 011 | **637** | 14,1 |
| ctx 8 208 / 12 288, q=1 | 77 001 / 115 988 | **5 489 / 8 070** | 14,0 / 14,4 |
| q=4 (verify) | 8 938 | 1 508 | 5,9 |
| q=8 (verify NS=7) | 9 014 | 2 104 | 4,3 |

**Servi MiniCPM5-2B** (tp1, KV bf16, 60 tokens, w40f → w41f, textes
**identiques au bit** partout) :

| prompt | NS=0 | NS=7 |
|---|---:|---:|
| 59 | 22,6 → **92,0** (4,1×) | 34,1 → 72,1 (2,1×) |
| 781 | 3,08 → **33,9** (11,0×) | 6,86 → 23,8 (3,5×) |
| 804 | 2,99 → **33,2** (11,1×) | 7,99 → 27,8 (3,5×) |
| 7 694 (agent) | 0,3 → **4,5** (15×) | — |

- **R49 clos sans re-baseline** : le split-KV n'était pas la seule voie, et il
  est même **dominé** (28,15 tok/s, non bit-identique) par une découpe qui ne
  touche à aucune parenthèse. Le plafond bit-exact est la chaîne de 64 dot2
  (~310 µs/couche et par passe) ; la bande tient 637 µs/couche à 969 clés,
  soit ~15 ms/token d'attention et **~44 tok/s de plafond servi** pour ce
  modèle à 969 clés.
- **R51** — *règle (W41b)* : pour paralléliser une attention **bit-exacte**,
  découper la **dimension de sortie** (une tuile de lignes × une tête par
  workgroup, un lane par dims) et jamais les **clés** : la réduction sur les
  clés est l'ordre de sommation, la découper change les parenthèses.

## 2026-11-19 — W41c : pas Flash-Next — staging KV préchargé, −6,5 %

Mesure (RADIANCE_AQL_STAMP, 1 589 dispatchs par pas, 6,6 % d'inactivité) :
l'attention h256 coûtait 2,08 ms/couche dont **85 % de staging KV sérialisé**
(318 µs sans staging) ; hc_read 27,2 ms dont hc_norm 10,3 ms pour
96 lancements.

Livré (92722381, fusionné) : le run de page de 16 o est préchargé **quatre
pages en avance** dans des registres (une page par lane) puis rangé par
put_page_run — mêmes clés, mêmes octets, même ordre ; hc_norm lit sa
somme double par vecteurs de 16 o (16 en vol) ; hc_up passe de 32 à 64 avec
des additions LDS par 8.

| point servi (même payload) | avant | après | Δ |
|---|---:|---:|---:|
| 3 252 NS=0 | 11,29 tok/s | **12,07** | **+6,5 %** (le journal a mesuré 11,98 après fusion) |
| 3 252 NS=3 | 18,95 | **19,79** | +4,2 % |
| 972 NS=0 | 4 904,1 ms | 4 639,5 ms | −5,4 % |
| 972 NS=3 | 2 986,8 ms | 2 889,4 ms | −3,3 % |
| préfill 3 252 / 972 | 5 412 / 2 100 ms | 5 409 / 2 093 ms | neutre |
| TTFT vision 1 024² / 2 048² | 3 309 / 16 400 ms | 3 276 / 16 316 ms | −1,0 / −0,5 % |

Par noyau : ttn 2 082 → 1 855 µs/couche, hc_norm 107 → 91, hc_up
45 → 28, hc_down 126 → 125. Textes identiques aux références W40 partout ;
kbench Flash-Next 4 786/4 786. **Écarté** : split-KV causal h256 (texte),
batch de 4 pages via stage_i8 (tableaux indexés partis en mémoire locale,
plus lent), batching des dots down/attn (neutre).

**Validation du build fusionné W41** (w41f = 651d0bfc + 92722381 +
9e701396, merges d77bff28/7c9dab0f) : suite complète verte — sched_test
548 964, engine_e2e_test 1 087 (3 skips), ixture_test 109 671,
config_test 2 952, ttn_paged_device_test **29/10** (21 cas h128 au bit),
ple_hc_device_test 76, dflash_conv_device_test 37, selftest **196/0/16** —
et kbench **27B 3 181/3 181** (3,619e-03), **MiniCPM 1 802/1 802**
(3,485e-03), **Flash-Next 4 786/4 786** (1,520e-02), 0 échec, 0 divergence.
Gates servis sur w41f : 27B NS=3 **27,45 vs 15,48 tok/s (1,77×)**,
Flash-Next **19,77 vs 11,98 (1,65×)**, MiniCPM textes identiques à w40f aux
trois prompts et deux NS (débits 92,0/33,9/33,2 en NS=0).
**Replay strict des fixtures rejoué au HEAD** (libref seul, forme du ctest
kbench_ref) : **1 134/1 134**, 0 échec, 0 stale, 0 divergence, 211 capped,
pire 
el_l2 3,442e-03 — la référence de W38 tient après les fusions.
Artefacts : <work-root>/work/w41-results/.
