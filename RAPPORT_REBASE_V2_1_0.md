# Rapport — rebase de `rdna3-iqr` sur `main` v2.1.0

**Date** : 2026-11-17
**Branche** : `rdna3-iqr`, rebasée sur `main` = `3045979` (v2.1.0, 2026-11-17)
**État du code** : `ad19972` — 149 commits au-dessus de `main` ; le présent
rapport est le commit de documentation qui suit.
**Sauvegarde avant rebase** : `rdna3-iqr-pre-rebase` = `20c5947`
**Hôte de validation** : `<gpu-user>@<gpu-host>`, W7900 `gfx1100`, 48 GiB,
ROCm/HIP 7.16.26385, build neuf `<work-root>/work/w29-rebase`

> **Objet.** Ce document trace le rebase, les conflits et leur résolution, les
> choix « code entrant d'abord », les deux défauts découverts (un côté
> `main`, un introduit par le rebase lui-même), et l'état de validation. Il ne
> remplace pas `RAPPORT_COMPARAISON_PERF_RDNA3.md` : les chiffres de
> performance de ce dernier sont **antérieurs au rebase** et devront être
> re-mesurés (le graphe QSA de Flash-Next a changé, cf. §7).

---

## 0. Résumé

- **159 commits** de la branche ont été rejoués ; les 15 commits de fusion ont
  été aplatis, laissant **144 commits rejoués** et une histoire linéaire.
  Cinq commits de finition ont été ajoutés après le rebase.
- **Quatre conflits manuels**, tous additifs (les deux côtés ajoutent des rows
  ou des schémas au même endroit) : `libr3d/CMakeLists.txt` et trois blocs de
  `libr3d/r3d_rows.cpp`. Un conflit répété a été résolu automatiquement par
  `rerere`. **Aucun code de noyau n'a été perdu.**
- **`ple_conv`** était implémenté des deux côtés. Conformément à la consigne,
  **le code entrant a été gardé** : `r3d_ple_conv.hip` de `main`, sa row, son
  schéma, ses contraintes, sa déclaration de plugin et son entrée CMake ;
  notre `r3d_ple_conv_bf16.hip` et ses doublons ont été supprimés. `ple_gate`
  reste le nôtre : `main` n'a pas de row RDNA3 pour cette op.
- **Un défaut de `main`** empêchait la compilation :
  `arch/common/rad_block_moe_fp8.h:1100` utilise `down_w5_fp8`, supprimé par
  le renommage de `d1b5a45` en `down_codebook_fp8`. Corrigé par la condition
  W5 d'origine (`a8 && fmt[0] == 11 && fmt[1] == 7`).
- **Un défaut introduit par le rebase** empêchait Flash-Next de démarrer : la
  row `ngram_ids_i32` avait disparu (le rebase a accepté sa suppression par
  `main`). Le graphe Flash-Next déclare `ngram_ids` en device à M=2048 ; sans
  row, le moteur s'arrête sur « no kernel resolved ». La row a été restaurée
  (`c11c451`), et Flash-Next sert de nouveau.
- Validation **correction** : build **EXIT=0**, **195 rows, 0 échec, 16
  skips**, `sched_test` **548 952 checks**, `engine_e2e_test` **1 076 checks**
  (3 skips), kbench **27B 3 181/3 181**, **Flash-Next 4 776/4 776**, **MiniCPM
  1 802/1 802**, 0 échec, 0 divergence. Les deux modèles servent (27B :
  192,5 tok/s préfill, 28,9 tok/s décodage ; Flash-Next : 22,5 tok/s décodage
  avec MTP à NS=3 ; 3 252 tokens de contexte sparse QSA à 234,8 tok/s).
- **Performance : le rebase est neutre.** Un build neuf de la branche
  **pre-rebase** (donc avec W27) donne **250,1/248,4 tok/s** et
  **4 486/4 521 µs** ; le post-rebase donne **249,2/249,0 tok/s** et
  **4 517/4 523 µs**. L'écart avec l'ancien binaire `final-all`
  (**285,7/284,0 tok/s**, **3 836/3 856 µs**) est une **dette de W27** : la
  généralisation `GS=1/2` du noyau IQR pour IQ8R coûte 17 % au 27B (GS=1). Le
  GEMV est inchangé. Le noyau, le layout, l'alignement, le flush HDP et
  l'horloge ont été éliminés (§5.5). **P0 fait** (`faa89fd`) : la
  spécialisation `GS=1` restaure **3 839 µs** et **288/287 tok/s** (16,8
  TFLOP/s), texte 27B identique, IQ8R inchangé.
- **Rien n'a été poussé.**

---

## 1. Périmètre et état avant rebase

| Élément | Valeur |
|---|---|
| Base commune ancienne | `ddcb9c6` (2.0.0) |
| Nouvelle base | `3045979` (v2.1.0, 86 commits depuis `ddcb9c6`) |
| Branche avant rebase | `20c5947`, 159 commits, avec 15 commits de fusion |
| Branche après rebase | `ad19972`, 149 commits linéaires (+ le présent rapport) |
| Fichiers changés vs `main` | 142 fichiers, +42 668 / −575 |
| Conflits textuels | 4 manuels + 1 auto (`rerere`) |
| Commits de finition | 5 (`5e98eb9`, `8137998`, `38fa364`, `c11c451`, `ad19972`) |

Méthode : `git rebase main` avec `rerere.enabled=true` (en option de commande,
pas de configuration persistante). Le rebase a été fait sur l'arbre de travail
propre ; les documents non suivis (`API_DISTANTE.md`, `HANDOVER_RDNA3.md`,
`PLAN_IMPLEMENTATION_RDNA3.md`) n'ont pas été touchés.

---

## 2. Conflits et résolutions

Tous les conflits venaient de **deux lignes d'ajout au même point d'insertion**
(rows/schémas de `libr3d`). Aucune divergence de contrat n'était en jeu : les
deux côtés ajoutaient des ops différentes.

| Commit rejoué | Fichier | Conflit | Résolution |
|---|---|---|---|
| `aa86e04` (h256 paged attention) | `libr3d/CMakeLists.txt` | entrées `r3d_attn_paged_h256` vs `r3d_mla_project`/`r3d_mla_attn_paged` | **les deux** |
| `b2f7398` (vision dense attention) | `libr3d/r3d_rows.cpp` | schémas `mla_*` vs `attn_dense` (`pAttnDense`/`oAttnDense`) | **les deux** |
| `06fc8e2` (router fusionné) | `libr3d/r3d_rows.cpp` | `router_sigmoid_group_topk` (branche) vs `router_gemm_topk_scatter` (`main`) | **les deux** |
| `5188c2d` (vision gemm_nt_bias) | `libr3d/r3d_rows.cpp` | `gemm_nt_f32` (`main`) vs `gemm_nt_bias` (branche) | **les deux** |
| étape 70 (répétition) | `libr3d/r3d_rows.cpp` | même bloc d'insertion qu'un conflit précédent | **rerere** (automatique) |

Les autres fichiers partagés (`core/sched/scheduler.*`, `geometry.*`,
`arch/common/rad_arch.h`, `tests/CMakeLists.txt`, `docs/OPS.md`) se sont
fusionnés sans conflit : la branche ajoute des lignes, `main` en ajoute
d'autres, et les contextes ne se recouvrent pas.

---

## 3. Ce qui a été pris du code entrant

### 3.1 `ple_conv` : doublon supprimé

Les deux lignes de développement avaient implémenté PLE (`Qwen4-Exp`) :

- `main` : `libr3d/r3d_ple_conv.hip`, row `ple_conv_bf16`, schéma `ple_conv`,
  paramètres/opérandes/contrainte, déclaration `r3d_rad_ple_conv`, CMake.
- branche : `libr3d/r3d_ple_conv_bf16.hip`, les mêmes noms et le même symbole
  `r3d_rad_ple_conv`, plus une API publique `r3d_ple_conv(...)` dans `r3d.h`
  (sans appelant).

Le rebase a produit les deux fichiers et des définitions dupliquées
(`redefinition of pPleConv/oPleConv/cPleConv`, puis double symbole). Résolution :

- garder **`main`** : `r3d_ple_conv.hip`, son schéma, sa row, sa contrainte,
  sa déclaration et son entrée CMake ;
- supprimer notre fichier, nos définitions, notre schéma, notre row, notre
  déclaration de plugin, notre entrée CMake et l'API morte de `r3d.h` ;
- garder **`ple_gate`** (notre `r3d_ple_gate_bf16.hip`) : `main` n'a pas de row
  `libr3d` pour `ple_gate`, seulement la row `libr4d`.

Vérification : `ple_hc_device_test` passe (48 checks), la row est unique, le
symbole est unique, et Flash-Next sert avec PLE.

### 3.2 `moe_gather` : garde de capacité de `main` adoptée

`main` corrige le test `T < M * top_k` en `T > M * top_k` (M = capacité de lot ;
la table de slots peut être plus courte que la capacité, jamais plus longue).
Notre branche avait déjà remplacé `M` par le nombre de rangées de `y` et
ajouté `bandM < M`. Le merge garde les deux :

- `M` = rangées réelles de `y`, `bandM` = étendue de bande ;
- `T > M * top_k` (garde de `main`, adoptée) ;
- le commentaire a été corrigé (`38fa364`) pour dire que `M` est le nombre de
  rangées du pas et `bandM` la bande.

La variante inverse (`moe_gather_inv`) conserve `T < M * top_k` comme dans
`main` ; ce n'est pas modifié.

### 3.3 Changements partagés de `main` conservés tels quels

- QSA `kQsaUnread`/`kQsaOnce`, `qsa_table_rows`, `qsa_exact_host` et
  `pass_key` (cf. §7) ;
- `rad_fp8.h` : clamp → `had_quant_act_i8g` (notre row existe déjà) ;
- `qwen4exp_fp8` : `expert_w6` hors `exp8`, calibration MTP ;
- AQL : flush du chemin hôte avant doorbell, cache scalaire derrière le wait
  chaîné, et ses tests device ;
- nouvelles rows RDNA3 : `gemm_nt_f32`, `moe_router_sigmoid_group_topk`,
  `kda_update`, `mla_project`, `mla_attn_paged` ;
- `ngram_ids_i32` retiré côté `main` (voir §4.2).

---

## 4. Deux défauts corrigés

### 4.1 Défaut de `main` : `down_w5_fp8` non déclaré

À `HEAD` de `main`, `arch/common/rad_block_moe_fp8.h:1100` contient :

```cpp
const bool clamp_w5 = down_w5_fp8 && c.swiglu_limit > 0;
```

`down_w5_fp8` a été renommé en `down_codebook_fp8` par `d1b5a45`, mais cette
utilisation est restée. Tous les plugins d'architecture qui incluent cet
en-tête (`bailing_hybrid*`, `flint`, `fishomni`, etc.) ne compilent pas.
Correction (`5e98eb9`) : rétablir la condition W5 d'origine, qui est ce que le
commentaire décrit,

```cpp
const bool clamp_w5 = a8 && fmt[0] == 11 && fmt[1] == 7 && c.swiglu_limit > 0;
```

Ce n'est pas un choix de contrat : c'est la définition d'avant le renommage.
À signaler en amont ; si `main` corrige autrement, prendre sa version.

### 4.2 Régression du rebase : `ngram_ids_i32` perdue

Notre branche avait la row `ngram_ids_i32` (`RAD_DOMAIN_DEVICE`, M=2048 sur
Flash-Next). `main` l'a retirée ; le rebase a accepté la suppression. Or
`arch/qwen4exp_fp8` déclare toujours `ngram_ids` en device à M=2048 pour le
préfill. Résultat observé au démarrage de Flash-Next :

```
---- unresolved: 406 op/band/domain combination(s)
  ngram_ids ... M=2048 [device] -> no device kernel for op 'ngram_ids' in any plugin
E startup stopped: no kernel resolved
```

Le noyau `libr3d/r3d_ngram_ids.hip` et son entrée CMake étaient toujours là ;
seule la row manquait. Restauration (`c11c451`) : la row est réinsérée avec son
schéma, sa contrainte `cNgramIds` et son symbole. L'unresolved passe de 406 à
405 (les 405 restants sont des bandes host, attendues), et Flash-Next démarre.

C'est le seul endroit où **le code entrant n'a pas été préféré** : `main` n'a
pas d'équivalent, et sa suppression casse le service RDNA3. À proposer en amont
si `main` veut servir Flash-Next sur `libr3d`.

---

## 5. Validation

### 5.1 Build

Arbre neuf `<work-root>/work/w29-rebase`, `rsync` sans `external/`,
`external-docs/`, `.git` ni `build/` :

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DRAD_GPU_TARGETS=gfx1100 -DRAD_WITH_FFMPEG=ON
cmake --build build -j 24
```

**EXIT=0.** `libr4d` et `libhalo` ne sont pas construits (cibles gfx12xx /
gfx1151), comme attendu. Aucun doublon de row ou de schéma après le rebase
(174 rows, 109 schémas uniques dans `r3d_rows.cpp`).

### 5.2 Tests

| Test | Résultat | Avant le rebase |
|---|---|---|
| `r3d_selftest` | **195 rows, 0 échec, 16 skips** | 190 rows, 0 échec, 16 skips |
| `sched_test` | **548 952 checks, 212 cas, 0 échec** | 548 782 checks |
| `engine_e2e_test` | **1 076 checks, 51 cas (3 skips), 0 échec** | 909 checks |
| `fixture_test` | 92 385 checks, 59 cas (7 skips) | — |
| `build_test` | 969 checks, 65 cas | — |
| `shipped_plugins_test` | 1 010 checks, 6 cas | — |
| `abi_layout_test` | 16 checks, 4 cas | — |
| `arch_test` | 183 527 checks, 237 cas | — |
| `device_test` | 61 808 checks, 68 cas (6 skips) | — |
| `iqr_device_test` | 0 ulp, 0/5120 divergence (U4) | — |
| `vision_rows_device_test` | 66 checks, 8 cas | — |
| `mtp_enter_e4m3_device_test` | 22 checks, 3 cas | — |
| `logit_rerank_iqr_device_test` | 36 checks, 5 cas | — |
| `act_i8g_group256_device_test` | 103 checks, 6 cas | — |
| `ple_hc_device_test` | 48 checks, 8 cas | — |
| `tp2_collectives_device_test` | 12 checks, 2 cas | — |
| `moe_iqr_device_test` / `moe_w4nl_device_test` | exact (bf16) | — |
| `qsa_*_device_test` | block_key 640 valeurs ; score 72 checks ; select 781 checks ; work 819 checks | — |
| `router_fused_device_test` | 38 checks, 8 cas | — |
| `gdn_conv_recurrent_had_device_test` | 46 checks, 5 cas | — |
| `r3d_rope_imrope_h128_device_test` | exact, M=4096 | — |

`build_test` et `shipped_plugins_test` exigent
`RADIANCE_HOME=<build>/radiance_home` (comme ctest) ; sans la variable, le
premier échoue sur un chemin de plugins absent, ce n'est pas une régression.
`dspark_device_test` est un test `libr4d` (RDNA4) et n'est pas applicable à
`gfx1100` ; il sort sur son message d'usage.

### 5.3 kbench

Exécuté avant la restauration de `ngram_ids_i32` (le kbench ne rend pas le
graphe du modèle ; il ne voit pas cette bande). La restauration ajoute une row
et ne touche aucun noyau existant ; le selftest final à `c11c451` la couvre
(le dernier commit `ad19972` ne change que l'indentation de cette row).

| Conteneur | Résultat | Pire rel_l2 |
|---|---|---|
| 27B `qwen3.8-27b-iq4r-l-v4.rad` | **3 181 / 3 181**, 0 échec, 0 divergence | 3,619e-03 |
| Flash-Next `qwen3.8-next-flash-iqr-v4.rad` | **4 776 / 4 776**, 0 échec, 0 divergence | 1,520e-02 |
| MiniCPM5-2B `minicpm5-2b-iq8r.rad` | **1 802 / 1 802**, 0 échec, 0 divergence | 3,485e-03 |

Le pire `rel_l2` Flash-Next passe de 1,206e-02 (référence W28) à 1,520e-02 ;
aucun cas n'échoue, la tolérance est respectée, mais la valeur mérite d'être
attribuée (nouvelles rows `main` ou graphe QSA) si l'on veut la ramener.

### 5.4 Service réel

| Modèle | Prompt | Génération | Préfill | Décodage |
|---|---:|---:|---:|---:|
| 27B, IQR, KV i8, NS=0 | 58 tok | 16 tok | **192,5 tok/s** | **28,9 tok/s** |
| Flash-Next, IQR, KV i8, NS=3 | 58 tok | 8 tok | 66,7 tok/s (petit prompt) | **22,5 tok/s** |

Flash-Next : 9 drafts, 5 acceptés (55,6 %) sur cette requête courte. Le prompt
de 58 tokens est **sous la borne QSA de 2 051 tokens** : le chemin dense
`by_table` de `main` est donc exercé. Le chemin sparse a été testé ensuite avec
un prompt de **3 252 tokens** : préfill **234,8 tok/s** (13,85 s, TTFT 13,85 s),
4 tokens générés, et la sélection de noyaux journalise bien
`qsa_block_key`, `qsa_score`, `qsa_select`, `qsa_tail_store` et `qsa_work` en
libr3d comme en avx. Aucun échec, aucune divergence signalée.

Flash-Next exige une configuration explicite sur cette carte :

```yaml
tp: 1
placement: expert_tiered
host-pool-mib: 24576
kv-cache-dtype: i8
num-speculative-tokens: 3
max-num-seqs: 1
max-model-len: 8192
```

Sans `host-pool-mib`, 20,81 GiB d'experts ne tiennent pas et le démarrage
s'arrête sur « out of memory » ; c'est un réglage de placement, pas un défaut
du rebase.

### 5.5 Performance : le rebase est neutre, l'écart vient de W27

**Correction d'une première conclusion.** L'A/B initial comparait le binaire
`final-all` (arbre **W26**, antérieur à W27) au binaire rebasé. Un **build
neuf de `rdna3-iqr-pre-rebase`** (donc avec W27, sans rebase) a été ajouté :

| Mesure (carte froide) | `final-all` (W26) | pre-rebase rebuild (W27) | post-rebase (W27) |
|---|---:|---:|---:|
| 27B, prompt 532 tok, préfill (2 runs) | **285,7 / 284,0 tok/s** | **250,1 / 248,4 tok/s** | **249,2 / 249,0 tok/s** |
| kbench `gemm_iqr_tiled` 12288×5120 M=512 | **3 836 / 3 856 µs** | **4 486 / 4 521 µs** | **4 517 / 4 523 µs** |
| kbench `gemm_iqr_nt_m64` 34816×5120 M=1 | 177,1 µs | — | 176,0 µs |

Conclusion : **le rebase est neutre**. Le rebuild pre-rebase (W27) et le
post-rebase donnent les mêmes temps ; l'écart de 12–17 % vient de la
**différence entre l'arbre `final-all` et la branche**, pas de `main`.

La cause est dans notre propre travail W27 (IQ8R) : `final-all` porte un
`r3d_gemm_iqr_m16.hip` **sans** le paramètre de groupe, avec une boucle `kb`
simple ; la branche porte la version W27 **généralisée GS=1/2**, avec une boucle
`g` imbriquée et des champs d'exécution `c.ngrp`/`c.gs`. Sur le 27B (groupe 128,
GS=1), cette généralisation coûte **17 %** au préfill IQR. Le noyau, le layout,
l'alignement (256 octets), le flush HDP de l'AQL (toggle et revert de `8f2e511`)
et l'horloge (post-rebase plus haute) ont été éliminés comme causes.

C'est donc une **dette de performance de W27**, pas une régression du rebase :

- le préfill IQR du 27B est passé de **17 TFLOP/s (W26)** à **14,3 TFLOP/s
  (W27)** ;
- l'IQ8R (MiniCPM5-2B, groupe 256) a gagné son service, mais au prix de ce
  chemin générique ;
- les chiffres de `RAPPORT_COMPARAISON_PERF_RDNA3.md` (17 TFLOP/s) décrivent
  **W26**, pas la branche actuelle.

**P0 fait (commit `faa89fd`)** : le noyau resident `GS=1` a été restauré depuis
la version W26 (`gemm_iqr_resident_gs1_kernel`), et `launch_tiled_form` y
dispatche pour `GS == 1` ; la boucle groupée reste pour `GS == 2` (IQ8R).
Résultats mesurés sur le build propre :

| Mesure | W27 avant | **GS=1 restauré** | W26 (`final-all`) |
|---|---:|---:|---:|
| kbench `gemm_iqr_tiled` 12288×5120 M=512 | 4 486 / 4 517 µs | **3 839 µs** | 3 836 / 3 856 µs |
| 27B préfill 532 tok | 251,5 / 250,5 tok/s | **288,4 / 287,3 tok/s** | 285,7 / 284,0 tok/s |
| TFLOP/s (12288×5120 M=512) | 14,3 | **16,8** | 16,8 |

Correction validée : `iqr_device_test`, `iqr_parity_test` et
`shipped_plugins_test` passent ; kbench **27B 3 181/3 181**, **Flash-Next
4 786/4 786**, **MiniCPM 1 802/1 802**, 0 échec, pires rel_l2 3,619e-03 /
1,520e-02 / 3,485e-03 ; le texte 27B (prompt « sky is blue », 32 tokens,
température 0) est **identique** entre l'arbre W27 sans GS=1 et l'arbre
patché (`choices` égales). L'IQ8R (groupe 256) garde le chemin `GS=2`, inchangé.

Le rebase est donc validé en **correction et en performance**, et la dette de
préfill W27 est soldée.

---

## 6. Différences restantes par rapport à `main`

La branche reste volontairement très au-dessus de `main` : **142 fichiers,
+42 668 / −575**. L'inventaire utile :

- **Rows RDNA3 ajoutées par `main`** : `gemm_nt_f32_bf16`,
  `moe_router_sigmoid_group_topk_f32`, `kda_update_k128_v128_bf16`,
  `mla_project_bf16`, `mla_attn_paged_bf16` (+ `ple_conv_bf16`, désormais
  servie par le noyau de `main`). Total rows déclarées : 174 (contre 168
  avant), selftest 195 (contre 190).
- **Notre travail conservé** : IQR (GEMV, préfill, U4/U5, relayout, IQ8R
  groupe 256), HC (read/enter/write/project), QSA (score/select/block_key/work),
  MTP (enter, draft head, rerank), vision (attn_dense h72, rope axial,
  gemm_nt_bias, grid_embed), TP2 (18 collectives), R22 (snapshot de queue),
  R23 (tables de pointeurs), kbench/tuning, tests device.
- **Rows `main` retirées de notre RDNA3** : `ngram_ids_i32` — restaurée
  (§4.2). Aucune autre.
- **Code de `main` modifié** : `arch/common/rad_block_moe_fp8.h` (une ligne,
  §4.1) et `libr3d/r3d_moe_gather.hip` (commentaire, §3.2).
- **Code de notre branche supprimé** : `r3d_ple_conv_bf16.hip` et son API
  morte (§3.1).

---

## 7. Risques et points de vigilance

1. **QSA change le graphe servi.** Sous la borne (2 051 tokens sur Flash-Next),
   `main` n'émet plus `qsa_score`/`qsa_select` sauf à la première couche
   full-attention ; l'attention lit la block-table directement
   (`by_table`). Au-delà, le chemin sparse reste. Nos rows QSA restent
   utilisées au-delà de la borne ; le kbench ne rend pas ce graphe. Il faut
   **servir un prompt > 2 051 tokens** pour valider le chemin sparse et
   re-mesurer `qsa_score`/`qsa_select` (le rapport de comparaison est
   antérieur).
2. **Le kbench n'est pas un test de graphe.** Les 4 776 cas Flash-Next
   passaient alors que le moteur ne démarrait pas (`ngram_ids`). La leçon est
   déjà dans AGENTS.md ; elle s'est vérifiée ici. Garder le service réel dans
   la validation de tout merge.
3. **`down_w5_fp8` et `ngram_ids_i32` sont deux régressions amont.** La
   première est un simple renommage inachevé ; la seconde est une suppression
   qui casse RDNA3. Les deux méritent un rapport amont, avec le test
   `flint`/`bailing_hybrid` pour la première et Flash-Next M=2048 pour la
   seconde.
4. **Le rebase a aplati 15 commits de fusion.** L'arbre final est validé, mais
   l'historique ne montre plus les branches latérales (`rdna3-w21-hc`,
   `rdna3-tp2`, etc. restent des branches locales). Pour une revue par lot, se
   référer aux branches, pas à la linéarisation.
5. **Les nouveaux modèles de `main` ne sont pas tous servables sur RDNA3.**
   La vision Muse (`grid_embed_pad`, `window_order`, attn h96), LFM3
   (`causal_conv1d`, `norm_add_norm`), l'audio/encodeur f32 et DFlash2
   (`dflash_conv`, `attn_paged_sink`, `qkv_norm_rope_kv_store`) n'ont pas de
   row `libr3d` ; ils restent RDNA4-only. Les rows ajoutées à `libr3d`
   (MLA/KDA/PLE/f32/router) couvrent Ling/Bailing, Flint et Qwen4-Exp.
6. **`ple_gate` n'existe que chez nous** sur RDNA3. Si `main` en ajoute une,
   la préférer comme pour `ple_conv`.
7. **Le pire `rel_l2` Flash-Next a augmenté** (1,206e-02 → 1,520e-02) sans
   échec. À attribuer avant de figer une référence de performance.

---

## 8. Suites recommandées

1. **Fait** : Flash-Next servi à 3 252 tokens (> 2 051) ; préfill 234,8 tok/s,
   TTFT 13,85 s, rows QSA sparse (`qsa_block_key`, `qsa_score`, `qsa_select`,
   `qsa_tail_store`, `qsa_work`) sélectionnées en libr3d et en avx. Rester à
   faire : comparer le texte au run d'avant rebase sur le même prompt.
2. **Fait** : les noyaux ont été re-mesurés (§5.5). Le GEMV est inchangé ; le
   rebuild **pre-rebase (W27)** donne 249 tok/s et 4,49 ms, identique au
   post-rebase. L'écart avec `final-all` (285 tok/s, 3,84 ms) était la dette
   W27, pas le rebase ; le patch GS=1 restaure 3 839 µs et 288 tok/s. Reste à
   re-mesurer bf16/vision, MoE, `hc_read`, `mtp_enter`, QSA et l'attention sur
   le build patché.
3. **Fait (`faa89fd`)** : spécialisation `GS == 1` (noyau resident restauré de
   W26, `GS=2` conservé pour IQ8R). kbench **27B 3 181/3 181**, **Flash-Next
   4 786/4 786**, **MiniCPM 1 802/1 802**, 0 échec ; texte 27B identique ;
   `iqr_device_test`, `iqr_parity_test`, `shipped_plugins_test` verts. Cible
   3,84 ms / 285 tok/s atteinte (3 839 µs / 288 tok/s).
4. Proposer en amont les deux correctifs (`down_w5_fp8`, `ngram_ids_i32`) avec
   leurs tests.
5. **Fait pour le 27B** : `rad-tune` rejoué sur le build propre
   (`build-clean`) confirme les chiffres pré-rebase — `gemm_nt_bf16_tiled`
   N=256 K=5120 M=1792 : `impl=0` 789,0 µs contre `impl=3` **199,5 µs**
   (3,95×, rel 4,9e-06) ; `gemm_nt_bf16_m16` : 21,58 µs contre **15,27 µs**
   (1,41×). Le cache reste absent du home servi (`build-clean/radiance_home`),
   donc le constat V3 du rapport de comparaison tient toujours : à livrer. Le
   tune Flash-Next n'a pas été rejoué.
6. Si `main` corrige `down_w5_fp8` ou `ngram_ids` différemment, prendre sa
   version au prochain rebase (les commits `5e98eb9` et `c11c451` sont
   identifiables et peuvent être abandonnés).
7. **Après GS=1** : re-mesurer bf16/vision, MoE, `hc_read`, `mtp_enter`, QSA et
   l'attention sur `build-clean` patché ; rejouer `rad-tune` sur Flash-Next ;
   mettre à jour `AGENTS.md`, `TABLEAU_REEL_IDEAL_RDNA3.md`,
   `JOURNAL_RDNA3.md` et `SUIVI_LOTS_RDNA3.md` (195 rows, `main` 2.1.0,
   GS=1, état W29).

---

## 9. Reproduction

```bash
# Rebase (déjà fait ; sauvegarde disponible)
git branch rdna3-iqr-pre-rebase 20c5947      # existante
git -c rerere.enabled=true rebase main

# Build neuf
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DRAD_GPU_TARGETS=gfx1100 -DRAD_WITH_FFMPEG=ON
cmake --build build -j 24

# Validation
flock -w 7200 /tmp/radiance-gpu.lock -c 'cd <tree> && \
  ./build/tests/sched_test && ./build/tests/engine_e2e_test && \
  ./build/tests/fixture_test && ./build/tests/iqr_device_test && \
  ./build/tests/vision_rows_device_test && ./build/tests/ple_hc_device_test && \
  ./build/bin/r3d_selftest ./build/radiance_home/kernels/libr3d.so'

# kbench (compte les cas, rend le graphe des rows seulement)
./build/bin/rad-kbench -m <model>.rad --home build/radiance_home \
  --report /tmp/kb.md

# Service réel (obligatoire pour le graphe)
./build/bin/radiance --config <config>.yaml
curl -s localhost:<port>/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
```

**État final** : branche `rdna3-iqr`, code à `ad19972` (149 commits au-dessus
de `main`) plus le commit de documentation de ce rapport ; arbre de travail
propre (hors documents non suivis préexistants), aucun push effectué.
**Correction et performance du rebase validées** ; la dette de W27
(`GS=1/2` du noyau IQR) est soldée par la spécialisation `GS == 1`
(`faa89fd`) : 3 839 µs, 288 tok/s, texte identique, IQ8R inchangé.
