# Audit du conteneur Flash-Next (Qwen3.8-Next-Flash-IQR-V4)

Branche : `rdna3-w14-audit` (base `rdna3-iqr` @ `46f2ffc`).
Date : 2026-11-16.
Hôte : `<gpu-user>@<gpu-host>`, une Radeon Pro W7900 (gfx1100, 48 Go) plus un iGPU
gfx1036. Modèle : `<work-root>/models/qwen3.8-next-flash-iqr-v4.rad`,
121 678 372 864 octets (113,3 Gio), mtime 2026-11-15 11:34.

Ce document est un audit. Aucun code produit n'a été ajouté au dépôt, le modèle et
`data/` n'ont pas été modifiés, et aucun benchmark de performance n'a été lancé.

> **Mise à jour du périmètre (2026-11-16, ADR-7).** Les conclusions de périmètre
> de cet audit ne sont plus celles du programme : la **vision** et le **MTP de
> Flash-Next** entrent dans la cible. Les constats de forme restent exacts (au
> moment de l'audit, `mtp_enter` et `logit_rerank` n'avaient aucune row
> `libr3d`), mais ils deviennent le travail du lot W25 (deux rows MTP +
> couverture kbench), et non une exclusion. Côté vision, `gemm_nt_bias_bf16` et
> `grid_embed_bf16` ont été livrées depuis (W15) ; il ne manque que le décodage
> média (build FFMPEG, lot W26).

## 0. Résumé exécutif

- Le conteneur est bien un Flash-Next : 48 couches dont 12 `full_attention` (3, 7, …,
  47) et 36 `linear_attention` (GDN), plus une couche MTP 48, 512 experts routés,
  `top_k=10`, 1 expert partagé de 640 colonnes, hidden 2560.
- La recette est **mixte** : les experts `gate_up` des 48 couches du tronc sont en
  treillis IQR `u4` (`tcq=16`, `tcq_code=mad`, `tcq_rows=128`), les experts `down`
  sont en codebook `w4nl` (`u4`, groupe 64, échelle E4M3 + f32), et 10 experts de
  4 couches sont laissés en bf16. Le MTP 48 est `u3` en q/k/v/qg et `u4` en
  `attn_output` (échelle bf16), ses experts sont `w4nl`.
- Le tronc est **IQ8R** en i8 tourné (`i8*bf16[1x128]/fwht128`) pour k/v/output,
  `ssm_out`, les experts partagés et PLE ; `qsa_qg`/`ssm_inz` sont IQR `u5`.
- Avec `libr3d` seul, le déclaratif ne va pas au bout : `gdn_conv_recurrent_update_had`
  n'a aucune row, puis le poids MoE IQR a trois plans et aucun op ne les lit
  (`moe_gemm_iqr` absent). Le graphe complet n'a donc été obtenu qu'avec deux
  contournements de **debug** décrits au §2.
- Les cinq ops QSA sont demandées : `qsa_tail_store` a une row device, les quatre
  autres non (`qsa_work`, `qsa_block_key`, `qsa_score`, `qsa_select`).
- Le conteneur ne peut pas être servi correctement par `libr3d` aujourd'hui, même
  avec `libavx` : le chemin IQR MoE et le chemin IQ8R dense tombent sur l'hôte ou
  échouent. Les rows à écrire sont listées au §7, dans l'ordre.

## 1. Méthode et commandes

Lecture des tables seulement, sans chargement VRAM :

```bash
# recette, métadonnées, encodages, plans
cd <work-root>/work/w13-clean
./build/bin/rad-info --recipe <work-root>/models/qwen3.8-next-flash-iqr-v4.rad
./build/bin/rad-info --meta   <work-root>/models/qwen3.8-next-flash-iqr-v4.rad
./build/bin/rad-info -v       <work-root>/models/qwen3.8-next-flash-iqr-v4.rad \
  | grep -E 'blk\.(0|3|34|44|46|47|48)\.'
./build/bin/rad-info --planes 'blk.0.ffn_gate_up_exps.0.weight' \
  <work-root>/models/qwen3.8-next-flash-iqr-v4.rad
```

Les plans et les blocs de plans ont été relus directement dans le répertoire du
conteneur par un petit lecteur d'audit (hors dépôt) :

```python
# /tmp/rad_dump.py, /tmp/rad_plain.py
# RadFileHeader: <IIQ5Q16Q8Q (248 octets)
# RadFileEntry : 136 octets, shape[6] à +32, layer/expert/slot/nplanes à +80
# RadEncoding : 568 octets, 24 scheme + 24 transform + nplanes + 8 plans de 64
# RadEncPlane : role[24], dtype, kind, block[2], extent[2]
```

Le graphe a été produit par l'engine, pas par rad-kbench :

```bash
# /tmp/fn-debug.yaml
server:
  host: 127.0.0.1
  port: 8199
  log: info
  radiance-home: <work-root>/work/w14-audit/build/radiance_home
  kernels: libr3d,libavx,libref
models:
  flashnext:
    model: <work-root>/models/qwen3.8-next-flash-iqr-v4.rad
    tp: 1
    debug-graph: true
    max-num-seqs: 4
    max-model-len: 200000
    placement: expert_tiered
    host-pool-mib: 43000
    kv-cache-dtype: i8
    prefix-cache-mib: 1024
    num-speculative-tokens: 3
    max-num-batched-tokens: 4096
    checkpoint-interval: 4096
```

```bash
flock -w 7200 /tmp/radiance-gpu.lock -c \
  'RAD_AUDIT_ALLOW_UNJOINED=1 timeout 1200 \
   ./build/bin/radiance --config /tmp/fn-debug.yaml > /tmp/fn-debug.log 2>&1'
```

`--debug-graph` s'arrête après le déclaratif, avant l'allocation des buffers et
avant tout calcul. Aucun essai GPU n'a été lancé en parallèle.

## 2. Contournements de debug (non commités, non produit)

Avec `libr3d` seul, le déclaratif échoue avant d'imprimer le graphe :

1. `arch/qwen35_fp8/qwen35_fp8.cpp` demande `gdn_conv_recurrent_update_had` dès que
   `m.trunk_rot` est vrai. Cette op n'existe que dans `libr4d` ; `libr3d` et `libref`
   ne la déclarent pas. Dans la copie d'audit, `m.a_go.q8_fed` et `m.a_go.q8_rot` ont
   été forcés à `false` pour prendre la paire non fusionnée `gdn_conv_update` +
   `gdn_conv_recurrent_update` (qui, elle, existe).
2. Le poids `blk.0.ffn_gate_up_exps.0.weight` est un poids à trois plans
   (`codes`, `trellis`, `trellis.mad`) plus `scale`/`scale.col`. Aucun op résolu ne le
   lit (`moe_gemm_iqr` absent), donc `core/build/rad_builder.cpp` rend une erreur
   fatale. Dans la copie d'audit seulement, `RAD_AUDIT_ALLOW_UNJOINED=1` saute cette
   erreur pour laisser le graphe s'imprimer. Ce n'est pas un correctif : le poids
   reste non servi dans cette exécution.

Conséquence : les 97 rows device listées ci-dessous sont celles que `libr3d` résout
réellement pour les ops qui ne dépendent pas des deux contournements. Les bandes MoE
IQR sont explicitement des trous, pas des résolutions cachées. Un démarrage réel
avec `libr3d` seul échoue sur (1) puis (2), et ne sert pas le modèle.

## 3. Ce que contient le conteneur

### 3.1 Métadonnées

| champ | valeur |
|---|---|
| `n_layers` | 48 (+ couche MTP 48) |
| `layer_types` | 36 `linear_attention`, 12 `full_attention` (3, 7, …, 47) |
| `hidden_size` | 2560 |
| `n_head` / `n_head_kv` / `head_dim` | 24 / 2 / 256 |
| `n_expert` / `n_expert_used` / `n_expert_shared` | 512 / 10 / 1 |
| `shared_expert_intermediate_size` | 640 (5 blocs de 128) |
| `n_vocab` | 248320 |
| `max_position_embeddings` | 262144 |
| QSA `indexer_heads` / `head_dim` | 4 / 128 |
| QSA `indexer_compress_ratio` / `budget` | 4 / 2048 |
| QSA `topk` | 2048 / 4 = 512 |
| rope | theta 1e7, partiel 0.25 (64 voies), M-RoPE sections 11 11 10 |
| MTP | 1 couche, `full_attention`, theta 1e7 |

### 3.2 Règles de quantification réellement appliquées

Extrait de `rad-info --recipe` (l'ordre est significatif : la règle la plus
spécifique passe avant `blk.*`) :

```text
blk.48.attn_qg.weight       gptq codes=u3 tcq=16 tcq_code=mad tcq_rows=256 block=1x* scale=f32 transform=fwht128
blk.48.attn_k.weight        gptq codes=u3 tcq=16 tcq_code=mad tcq_rows=256 block=1x* scale=f32 transform=fwht128
blk.48.attn_v.weight        gptq codes=u3 tcq=16 tcq_code=mad tcq_rows=256 block=1x* scale=f32 transform=fwht128
blk.48.attn_output.weight   gptq codes=u4 tcq=16 tcq_code=mad tcq_rows=256 group=128 scale=bf16 dyn=1 transform=fwht128
blk.48.ffn_gate_up_shexp    gptq codes=u3 tcq=16 tcq_code=mad tcq_rows=128 block=1x* scale=f32 transform=fwht128
blk.48.ffn_*_exps.*.weight  gptq table=w4nl group=64 scale=fp8_e4m3 scale2=f32 block2=*x* scale2_value=1.220703125e-4 transform=fwht128 cd=3
blk.*.attn_qg.weight        gptq codes=u5 tcq=16 tcq_code=mad tcq_rows=256 block=1x* scale=f32 transform=fwht128
blk.*.attn_k.weight         rtn codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym
blk.*.attn_v.weight         rtn codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym
blk.*.attn_output.weight    rtn codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym
blk.*.ssm_inz.weight        gptq codes=u5 tcq=16 tcq_code=mad tcq_rows=256 block=1x* scale=f32 transform=fwht128
blk.*.ssm_out.weight        rtn codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym
blk.*.ffn_gate_up_shexp     rtn codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym
blk.*.ffn_down_shexp        rtn codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym
blk.*.ffn_gate_up_exps.*    gptq codes=u4 tcq=16 tcq_code=mad tcq_rows=128 block=1x* scale=f32 transform=fwht128
blk.*.ffn_down_exps.*       gptq table=w4nl group=64 scale=fp8_e4m3 scale2=f32 block2=*x* transform=fwht128 cd=3
blk.*.ple_key/ple_value     rtn codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym
*_hc_down/*_hc_up           rtn codes=fp8_e4m3 ...
mtp.fc_hidden/fc_embedding  rtn codes=fp8_e4m3 ...
output.weight               rtn codes=i8 group=128 scale=bf16 clamp=sym
blk.*.ple_ngram.weight      rtn codes=fp8_e4m3 block=*x* scale=bf16 fixed
mtp.draft_head.weight       rtn codes=u2 zero=u8 group=128 scale=f16
```

Les experts bf16 protégés sont les dix `cast dtype=bf16` de la recette, listés au
§5.

### 3.3 Volumétrie des poids experts

| classe | nombre de poids | forme |
|---|---:|---|
| `ffn_gate_up_exps`, tronc 0–47 | 24 566 | IQR treillis `u4` |
| `ffn_down_exps`, tronc 0–47 | 24 566 | `w4nl` `u4` |
| `ffn_gate_up_exps` + `ffn_down_exps`, MTP 48 | 1 024 | `w4nl` `u4` |
| experts bf16 protégés (2+2+2+4) × 2 projections | 20 | bf16 |
| total experts | 50 176 | — |

Les 24 566 = 48 × 512 − 10. Les down `w4nl` font 845 120 octets par expert
(2560×640), les gate_up IQR `u4` environ 1,64 Mio par expert, les gate_up `w4nl`
du MTP 1 709 392 octets.

## 4. Ops du conteneur et rows résolues par `libr3d`

Le tableau est le résultat de `--debug-graph` avec les bibliothèques ci-dessus. La
colonne « rows device » est la row `libr3d` effectivement choisie ; `—` signifie
qu'aucune row device n'existe. Les ops marquées `host` ont une row `libavx` mais
pas de row `libr3d` : c'est un fallback hôte, interdit pour cet audit.

### 4.1 Cœur, normalisation, données

| op | rows device `libr3d` |
|---|---|
| `add` | `add_bf16` |
| `cast` | `cast_bf16_bf16` (3 bandes) |
| `mul` | `mul_bf16` |
| `sigmoid` | `sigmoid_bf16` |
| `silu_mul` | `silu_mul_bf16` |
| `rmsnorm` | `rmsnorm_bf16` |
| `layernorm` | `layernorm_bf16` |
| `gather_rows` | `gather_rows_bf16` |
| `scatter_rows` | `scatter_rows_bf16` |
| `embed_lookup` | `embed_lookup_bf16` |
| `embed_lookup_q` | `embed_lookup_q_bf16` |
| `ngram_ids` | `ngram_ids_i32` |
| `ngram_hint` | `ngram_hint_i32` |
| `row_topk` | `row_topk_bf16` |

### 4.2 GEMM denses

| op | rows device `libr3d` |
|---|---|
| `gemm_nt` | `gemm_nt_bf16_m16`, `gemm_nt_bf16_tiled` |
| `gemm_nt_q` | `gemm_iqr_nt_m64`, `gemm_iqr_tiled`, `gemm_nt_q_w2a8_m16` |
| `gemm_nt_q_gated` | `gemm_iqr_gated_i8a8h` |
| `logits_gemm` | `logits_gemm_bf16` |

Détail des bandes `gemm_nt_q` :

- IQR `u5` de `attn_qg` (`N=12288 K=2560`) et `ssm_inz` (`N=16384 K=2560`) :
  `gemm_iqr_nt_m64` / `gemm_iqr_tiled`.
- IQR `u4` de `blk.48.attn_output` (`N=2560 K=6144`, échelle bf16) : la forme est
  acceptée par la row, mais la bande est **refusée globalement** parce qu'elle est
  partagée avec 48 poids i8 simples de même géométrie (`ssm_out`/`attn_output`).
- IQR `u3` de `blk.48.attn_qg` (`N=12288 K=2560`) : `gemm_iqr_nt_m64`.
- IQR `u3` de `blk.48.attn_k/v` (`N=512 K=2560`) : même problème de bande partagée
  avec les `attn_k/v` i8 du tronc.
- w2a8 de `mtp.draft_head` (`N=248320 K=2560`) : `gemm_nt_q_w2a8_m16`.
- i8 simple tourné (`i8*bf16[1x128]/fwht128`) de `attn_k`, `attn_v`,
  `attn_output`, `ssm_out`, `ffn_*_shexp`, `ple_key/value`, `blk.48.ffn_down_shexp` :
  **aucune row device**. Le hôte `libavx` les exécute (`avx_gemm_nt_q`), mais c'est
  exactement le fallback à supprimer.

### 4.3 GDN

| op | rows device `libr3d` |
|---|---|
| `gdn_conv_prep` | `gdn_conv_prep_w4_h128` |
| `gdn_conv_update` | `gdn_conv_update_w4_h128` |
| `gdn_recurrent_update` | `gdn_recurrent_update_k128_v128` |
| `gdn_conv_recurrent_update` | `gdn_conv_recurrent_update_k128_v128` |
| `gdn_chunk_scan` | `gdn_chunk_scan_k128_v128_c64` |
| `gdn_kkt_solve` | `gdn_kkt_solve_c64` |
| `gdn_gated_rmsnorm` | `gdn_gated_rmsnorm_h128` |
| `gdn_conv_recurrent_update_had` | **aucune** (seulement `libr4d`) |

`gdn_conv_recurrent_update_had` est la version fusionnée qui écrit les codes i8
tournés de `ssm_out`. Sans elle, le plugin doit désactiver `q8_fed`/`q8_rot` et
`ssm_out` retombe sur le chemin i8 non tourné (lui aussi sans row device).

### 4.4 Attention et KV

| op | rows device `libr3d` |
|---|---|
| `attn_dense` | `attn_dense_bf16_h72` |
| `attn_paged` | `attn_paged_i8_h256_gqa12_decode`, `attn_paged_i8_h256_gqa12_prefill` |
| `rope` | `rope_imrope_h256`, `rope_axial_bf16_h72` |
| `rope_table` | `rope_table_f32` |
| `kv_store` | rows `block_size=16` seulement ; **pas de `block_size=4` i8** |
| `qsa_tail_store` | `qsa_tail_store_bf16` |
| `qsa_work` | **aucune** |
| `qsa_block_key` | **aucune** (hôte `avx_qsa_block_key`) |
| `qsa_score` | **aucune** |
| `qsa_select` | **aucune** |

Les 13 couches d'attention (12 du tronc + MTP 48) utilisent `block_size=4` et
`kv_dtype=i8` ; la row `kv_store` existante refuse `block_size 4` (elle n'a que 16).
`rope` refuse `head_dim=128`, `mode=imrope` (la QSA query) ; `rope_imrope_h256`
couvre l'attention principale.

### 4.5 MoE

| op | rows device `libr3d` |
|---|---|
| `moe_gemm` (experts bf16) | `moe_gemm_bf16` (4 bandes : gate_up/down × 2/4 experts) |
| `moe_gemm_q` IQR `i8a8h` | **aucune** (seules les rows `fp8a8f32`, `mxfp4a8`, `w4a8`, `w4a4` existent) |
| `moe_gemm_q` `w4nl64a8i8h` | **aucune** en device ; hôte `avx_moe_gemm_q` |
| `router_topk` | `moe_router_topk_bf16` |
| `router_topk_scatter` | `moe_router_topk_scatter` |
| `router_gemm_topk_scatter` | **aucune** (hôte `avx_router_gemm_topk_scatter`) |
| `moe_scatter` | `moe_scatter_small` |
| `moe_gather` | `moe_gather` |

Les rows MoE existantes sont :
`moe_gemm_q_e4m3` (`fp8a8f32`), `moe_gemm_mxfp4a8`, `moe_gemm_w4a8`
(`w4a8`), `moe_gemm_w4a4` (`w4a4`) et `moe_gemm_bf16` (`bf16`). Aucune ne déclare
`i8a8h` ou `w4nl64a8i8h`.

### 4.6 HC, PLE, vision, MTP, sampler

| op | rows device `libr3d` |
|---|---|
| `hc_enter` | `hc_enter_bf16` |
| `hc_read` | `hc_read_e4m3` |
| `hc_write` | `hc_write_bf16` |
| `ple_gate` | `ple_gate_bf16` |
| `ple_conv` | `ple_conv_bf16` |
| `quant_act_fp8` | `quant_act_fp8_bf16` |
| `quant_act_i8` | `quant_act_i8_bf16` |
| `had_quant_act_i8g` | `had_quant_act_i8g_bf16` |
| `gated_had_quant_i8g` | `gated_had_quant_i8g_bf16` |
| `gemm_nt_bias` | **aucune** (hôte) |
| `grid_embed` | **aucune** (hôte) |
| `mtp_enter` | **aucune** (hôte) |
| `logit_rerank` | **aucune** (hôte) |
| `sample_argmax`, `sample_dry`, `sample_logprobs`, `sample_mask`, `sample_minp`, `sample_ngram_ban`, `sample_penalties`, `sample_pick`, `sample_temp`, `sample_topk`, `sample_topp`, `sample_typical`, `sample_xtc` | rows `*_f32` |

`gemm_nt_bias` et `grid_embed` sont les ops de la tour vision ; elles n'ont pas de row
device dans ce build, même si W10 annonce 142 rows vision vertes. `mtp_enter` et
`logit_rerank` n'avaient pas de row device au moment de l'audit ; l'ADR-7
(2026-11-16) les remet dans la cible, et leur écriture est le lot W25.

## 5. MoE IQR : formes, tailles, experts protégés

### 5.1 Formes réellement stockées (lecture des plans)

Toutes les formes IQR du conteneur sont des **treillis calculés**
`tcq_code=mad` :

| poids | kind | B | L | lrun | lrb | scale | couches |
|---|---|---:|---:|---:|---:|---|---|
| `ffn_gate_up_exps` (tronc) | `kTrellisMad` | 4 | 16 | 7 (128 lignes) | 0 | `kRowF32` (f32/row + f32 `scale.col`/128) | 0–47 |
| `attn_qg` (tronc) | `kTrellisMad` | 5 | 16 | 8 (256 lignes) | 0 | `kRowF32` | 3, 7, …, 47 |
| `ssm_inz` (GDN) | `kTrellisMad` | 5 | 16 | 8 (256 lignes) | 0 | `kRowF32` | 36 couches |
| `blk.48.attn_qg/k/v` | `kTrellisMad` | 3 | 16 | 8 (256 lignes) | 0 | `kRowF32` | 48 |
| `blk.48.attn_output` | `kTrellisMad` | 4 | 16 | 8 (256 lignes) | 0 | `kBf16` (bf16/row, 1/128) | 48 |
| `blk.48.ffn_gate_up_shexp` | `kTrellisMad` | 3 | 16 | 7 (128 lignes) | 0 | `kRowF32` | 48 |
| `ffn_down_exps` (tronc) et `blk.48.ffn_*_exps` | `kTable`/`w4nl` | 4 | 0 | — | 0 | `kE4m3Tensor` (E4M3/64 + f32 tensor, table f32[16]) | 0–48 |

Les plans confirment `L=16` (`trellis` de 65 536 entrées) et `lrb=0`
(`trellis` de bloc `[rows x 1]`). `lrun` vient du bloc du plan `trellis` :
`128 -> 7`, `256 -> 8`. Le `scale` IQR est stocké en f32 par ligne
(`scale=f32[1x*]`) plus un `scale.col=f32[*x128]` ; le `blk.48.attn_output` est
le seul en bf16 par (ligne, 128), sans `scale.col`.

Le `w4nl` n'est **pas** la forme `kTable` de `R3D_IQR_FORMS` (celle-ci est un
tableau i8 de 256 entrées sous une échelle bf16). C'est une forme à part :
`codes=u4`, `table=f32[1x16]`, `scale=fp8_e4m3[1x64]`, `scale.1=f32[1x1]`
(valeur 1,220703125e-4), `group=64`, `cd=3`, `transform=fwht128`.

### 5.2 Tailles et classes

Configuration de référence : `tp=1`, `M` par passe = `moecfg.rows = 2048`
(`max_tok=4096` est traité en deux passes), `top_k=10`, `n_expert_all=512`,
`n_shared=1`, largeur experte 640 (`n_ff_exp=640`).

| projection | N | K | parts | `a_order` | dtype op | `N_odd` | `K_odd` | n_expert |
|---|---:|---:|---:|---|---|---:|---:|---|
| gate_up routé, tronc | 1280 (2×640) | 2560 | 2 | `token` | `i8a8h` | 1280 | 2560 | 512 / 510 / 508 |
| down routé, tronc | 2560 | 640 | 1 | `sorted` | `w4nl64a8i8h` | 2560 | 640 | 512 / 510 / 508 |
| gate_up routé, MTP 48 | 1280 | 2560 | 2 | `token` | `w4nl64a8i8h` | 1280 | 2560 | 512 |
| down routé, MTP 48 | 2560 | 640 | 1 | `sorted` | `w4nl64a8i8h` | 2560 | 640 | 512 |
| experts bf16 protégés, gate_up | 1280 | 2560 | 2 | `token` | `bf16` | 1280 | 2560 | 2 ou 4 |
| experts bf16 protégés, down | 2560 | 640 | 1 | `sorted` | `bf16` | 2560 | 640 | 2 ou 4 |
| expert partagé, gate_up | 1280 | 2560 | 2 | `token` | `i8a8h` (sauf MTP 48 : IQR `u3`) | 1280 | 2560 | 1 |
| expert partagé, down | 2560 | 640 | 1 | `sorted` | `i8a8h` | 2560 | 640 | 1 |

À `tp=1` les deux classes paires/impaires ont la même largeur. Les comptes
d'experts quantifiés sont :

| couches | `n_expert` routés | classe paire | classe impaire | protégés |
|---|---:|---:|---:|---|
| 0–33, 35–43, 45 | 512 | 256 | 256 | 0 |
| 34, 44, 46 | 510 | 255 | 255 | 2 |
| 47 | 508 | 254 | 254 | 4 |

À `tp=2`, la même couche a `wq[0]=3` blocs (384) et `wq[1]=2` blocs (256) :
gate_up `N=768`, `N_odd=512`, `K=K_odd=2560` ; down `N=N_odd=2560`,
`K=384`, `K_odd=256`. À `tp=4`, l'op devient `moe_gemm_q_mod4` (classes modulo 4),
`N=512`, `K=256`, `N_1..N_3=512`, `K_1..K_3=256`.

**Lignes par expert** : le poids d'un expert a `N` lignes (1280 gate_up, 2560
down) et `K` colonnes. Dans une passe, le GEMM trie `M * top_k = 2048 × 10 =
20 480` lignes, soit 40 lignes par expert en moyenne ; la run map distribue ces
lignes par expert, chaque tuile fait 16 lignes, et un run de treillis fait 128
lignes (gate_up) ou 256 (qg/ssm_inz/attn_output). Pour un préfill de 4096 tokens,
la passe est coupée en deux `M=2048`, soit 40 960 lignes triées au total.

### 5.3 Experts bf16 protégés

| couche | experts bf16 | poids |
|---|---|---|
| 34 | 407, 496 | gate_up [1280,2560] + down [2560,640] |
| 44 | 292, 350 | idem |
| 46 | 290, 392 | idem |
| 47 | 122, 143, 399, 445 | idem |

Ils sont retirés des tables quantifiées (`n_reg = 512 − p`) et servis par
`moe_gemm_bf16` avec `n_expert = p` (2 ou 4), après `moe_scatter`/`router_*_scatter`
avec `protect`. La row `moe_gemm_bf16` existe et résout les quatre bandes.

### 5.4 Mélange de formes dans une même couche

| couche | gate_up routé | down routé | partagé | attention |
|---|---|---|---|---|
| GDN 0–2, 4–6, … | IQR `u4` | `w4nl` | i8 | `ssm_inz` IQR `u5`, `ssm_out` i8 |
| full 3, 7, …, 47 | IQR `u4` | `w4nl` | i8 | `attn_qg` IQR `u5`, k/v/output i8, KV i8 block 4 |
| 34, 44, 46 | IQR `u4` (510) | `w4nl` (510) | i8 | + 2 experts bf16 |
| 47 | IQR `u4` (508) | `w4nl` (508) | i8 | + 4 experts bf16 |
| MTP 48 | `w4nl` (512) | `w4nl` (512) | gate_up IQR `u3`, down i8 | q/k/v IQR `u3`, output IQR `u4` bf16 |

Une couche 34/44/46/47 émet donc trois familles de GEMM MoE : `moe_gemm_q`
`i8a8h` pour gate_up, `moe_gemm_q` `w4nl64a8i8h` pour down, et `moe_gemm` bf16
pour les protégés. Le MTP 48 n'a que du `w4nl` pour ses experts, mais son
`attn_output` est IQR bf16 et son gate_up partagé est IQR `u3`.

## 6. QSA : shapes et dtype

Configuration mesurée : `max_model_len=200000`, `max_tok=4096`, `max_seqs=4`,
`num_spec=3`, `ratio=4`, `budget=2048`, `topk=512`, `ring=ratio+num_spec=7`,
`max_work=ceil(4096/4)+4=1028`, `max_blocks=ceil(200000/4)=50000`. Le conteneur
naturel est 262144 → `max_blocks=65536` ; voir le risque §8.2.

| op | M (range) | n | ratio | ring | blocks | topk | seqs | dtype / sorties | row `libr3d` |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `qsa_work` | 1..4096 | 128 | 4 | 7 | — | — | 4 | in bf16 ; out stage bf16, page/bpos/nc i32 | **aucune** |
| `qsa_block_key` | 1..1028 | 128 | 4 | — | — | — | — | in bf16 ; out bf16 ; `rotary_dim=64`, `block_size=4`, `eps=1e-6`, `theta=1e7`, `scale=1`, `wadd=1`, `mode=imrope`, `sections=11 11 10` | **aucune** (hôte) |
| `qsa_score` | 1..4096 | 128 | — | — | 50000 | — | 4 | in q bf16, nc i32 ; out score f32 | **aucune** (hôte) |
| `qsa_select` | 1..4096 | — | 4 | — | 50000 | 512 | 4 | in score f32, nc/pos/cu i32 ; out sel/nsel/sequ i32 ; `signed` absent (non signé) | **aucune** (hôte) |
| `qsa_tail_store` | 1..4096 | 128 | 4 | 7 | — | — | 4 | bf16 | `qsa_tail_store_bf16` |

Détails :

- `qsa_work` est demandée `M=4096`, `work=1028`, `seqs=4`, `dtype=bf16` ; ses
  sorties sont `stage` bf16 `[work*ratio, n]`, `page` i32 `[work]`,
  `bpos` i32 `[work]` (ou `[work,3]` en M-RoPE), `nc` i32 `[seqs]`.
- `qsa_block_key` a `M=1028` (blocs), `k` est `ratio` fois plus haut, `out` est une
  page KV bf16 de `n = 128` valeurs (1 tête, `block_size=ratio`). Le `M` de la
  bande est bien 1028, pas 4096.
- `qsa_score` a `heads=4`, `blocks=50000`, `seqs=4` ; `score` est f32
  `[rows, blocks]` (la bande imprime `dtype=bf16` pour l'entrée q).
- `qsa_select` remplit `sel` i32 `[rows, topk+1]` (512+1), `nsel` i32 `[rows]`,
  `sequ` i32 `[rows]`. Le conteneur ne demande pas la variante `signed=1` (GLM),
  seulement la non signée.
- Le chemin QSA utilise aussi `kv_store` block 4 i8 et `rope` h128 imrope ; les
  deux manquent en device (voir §7).
- Le plan `score` fait `2048 × 50000 × 4 = 409,6 Mio` dans la configuration
  mesurée. `qsa_rows` vaut 2048 quand `span_on` et `max_mixed_tok <= 2048` ; avec
  4096 lignes et 50000 blocs il ferait 819,2 Mio et le plugin refuse au-delà de
  512 Mio. C'est un point de dimensionnement à surveiller.

## 7. Rows et formes manquantes (exhaustif)

### 7.1 Formes IQR : ce que les rows refusent aujourd'hui

| forme | bits | L | lrb | scale | row dense `gemm_nt_q` | row MoE `moe_gemm_q` |
|---|---:|---:|---:|---|---|---|
| `kTrellisMad` | 5 | 16 | 0 | `kRowF32` | **acceptée** (`gemm_iqr_nt_m64`/`gemm_iqr_tiled`) | **refusée** (aucune row `i8a8h`) |
| `kTrellisMad` | 4 | 16 | 0 | `kRowF32` | **acceptée** | **refusée** |
| `kTrellisMad` | 3 | 16 | 0 | `kRowF32` | **acceptée** | **refusée** |
| `kTrellisMad` | 4 | 16 | 0 | `kBf16` | **acceptée** | **refusée** |
| `kSigned` (i8 tourné) | 8 | 0 | 0 | `kBf16` | **refusée** : `signed_bits()` ne reconnaît pas `RAD_I8`, et `R3D_IQR_FORMS`/`form_supported` n'ont pas `kSigned` 8 | sans objet |
| `kTable` `w4nl` | 4 | 0 | 0 | `kE4m3Tensor` + f32 | sans objet (aucun poids dense w4nl) | **refusée** (aucune row `w4nl64a8i8h` ; `moe_gemm_w4a8` ne lit que le w4a8 empaqueté) |
| `plain` bf16 | — | — | — | — | acceptée (`gemm_nt_bf16`) | acceptée (`moe_gemm_bf16`) |

Points importants :

- Le refus `kSigned` n'est pas un refus de **row** au sens du nom : la row
  `gemm_iqr_nt_m64` déclare déjà `dtype=i8a8h`. Le refus est dans le hook de
  layout (`r3d_iqr_layout.cpp`) et dans `form_supported` des noyaux. Il faut
  ajouter `RAD_I8 → kSigned, B=8` et la forme
  `X(kSigned, 8, 0, 0, 0, kBf16)` à `R3D_IQR_FORMS`, puis accepter cette forme
  dans `form_supported` (`r3d_gemm_iqr_m16.hip:382`,
  `r3d_gemm_iqr_gated.hip:214`, éventuellement `r3d_iqr_logits.hip:286`).
- Ce seul ajout débloque aussi les bandes mixtes : `blk.48.attn_k/v` IQR `u3`
  (`N=512 K=2560`) et `blk.48.attn_output` IQR `u4` bf16 (`N=2560 K=6144`) sont
  aujourd'hui agrégées avec des poids i8 simples de même géométrie, donc la
  bande entière est refusée. La row IQR accepte la forme, mais pas les instances
  i8 de la même bande.
- Le `w4nl` n'est pas une forme de `R3D_IQR_FORMS`. Il faut une forme dédiée
  (`kW4NL`) avec table f32[16], échelle E4M3/64 + f32 tensor, `group=64`, et un
  noyau de décodage ; le noyau MoE `w4a8` actuel ne lit que l'int4 empaqueté.

### 7.2 Rows device manquantes, par op

| op | bande(s) demandée(s) | état | row à écrire |
|---|---|---|---|
| `moe_gemm_q` | `N=1280 K=2560`, `i8a8h`, `parts=2`, `N_odd=1280 K_odd=2560`, `n_expert=512/510/508`, `top_k=10`, `M=2048` | aucune row device, aucun fallback hôte | `moe_gemm_iqr` |
| `moe_gemm_q` | `N=2560 K=640`, `w4nl64a8i8h`, `parts=1`, `n_expert=512/510/508`, `a_order=sorted` | aucune row device ; hôte `avx_moe_gemm_q` | `moe_gemm_w4nl` |
| `moe_gemm_q` | `N=1280 K=2560`, `w4nl64a8i8h`, `parts=2`, `n_expert=512`, MTP 48 | aucune row device ; hôte | `moe_gemm_w4nl` |
| `moe_gemm_q_mod4` | `i8a8h`/`w4nl64a8i8h`, 4 classes, tp4 | aucune row device adaptée | `moe_gemm_iqr_mod4` / `moe_gemm_w4nl_mod4` (si tp4) |
| `qsa_work` | `n=128 ratio=4 ring=7 seqs=4 work=1028 bf16 M=4096` | aucune row device | `qsa_work_bf16` |
| `qsa_block_key` | `n=128 ratio=4 rotary=64 block=4 eps=1e-6 theta=1e7 mode=imrope sections=11 11 10 M=1028` | hôte seulement | `qsa_block_key_bf16` (mrope+imrope) |
| `qsa_score` | `n=128 heads=4 blocks=50000 seqs=4 bf16 M=4096` | aucune row device | `qsa_score_bf16` (sortie f32) |
| `qsa_select` | `blocks=50000 topk=512 ratio=4 seqs=4 M=4096`, non signé | aucune row device | `qsa_select_i32` (+ variante `signed`) |
| `kv_store` | `head_dim=256 n_head_kv=2 block_size=4 kv_dtype=i8 M=4096` | rows block 16 seulement | `kv_store_i8` block 4 |
| `rope` | `head_dim=128 n_head=4 n_head_kv=0 mode=imrope rotary=64 sections=11 11 10 M=4096` | rows h256/h72 seulement | `rope_imrope_h128` (ou replier dans `qsa_score`) |
| `gdn_conv_recurrent_update_had` | `k128 v128`, codes i8 tournés | aucune row hors `libr4d` | `gdn_conv_recurrent_update_had` |
| `router_gemm_topk_scatter` | `M<=64`, `n_expert=512`, `top_k=10`, `protect`, bf16 | hôte seulement | row device fusionnée, ou garder la paire `router_gemm` + `router_topk_scatter` |
| `gemm_nt` | `N=1`, `K=2560`, `M=2048` (shared gate, préfill) | hôte seulement | row GEMV préfill, ou padding N |
| `gemm_nt_q` | i8 simple tourné, bandes `N=512/1280/2560/…` | hôte seulement | aucune nouvelle row : étendre `gemm_iqr_*` à `kSigned` 8 |
| `gemm_nt_bias` | vision | aucune row à l'audit (livrée W15 : `gemm_nt_bias_bf16`) | — (reste le décodage média, W26) |
| `grid_embed` | vision | aucune row à l'audit (livrée W15 : `grid_embed_bf16`) | — (reste le décodage média, W26) |
| `mtp_enter` | MTP | hôte seulement | row device à écrire (ADR-7, W25) |
| `logit_rerank` | MTP | hôte seulement | row device à écrire (ADR-7, W25) |

### 7.3 Ce qui est déjà servi

- Toutes les ops du GDN sauf `_had`.
- `attn_paged` h256 gqa12 en décodage et préfill, `attn_dense` vision h72.
- `rope_imrope_h256`, `rope_table_f32`.
- `moe_gemm_bf16` pour les experts protégés.
- `router_topk_scatter`, `moe_scatter_small`, `moe_gather`.
- Le dense IQR `qg`/`ssm_inz`/`blk48.qg` (`gemm_iqr_nt_m64`/`tiled`).
- HC, PLE, n-gram, sampler, tête vocabulaire (`logits_gemm`).

## 8. Incertitudes et risques

1. **Graphe obtenu avec deux contournements de debug.** Le graphe complet exige
   `RAD_AUDIT_ALLOW_UNJOINED=1` et la paire GDN non fusionnée. Les rows listées
   sont celles des binaires `libr3d` non modifiés, mais un démarrage réel avec
   `libr3d` seul échoue avant le graphe. Ne pas conclure du graphe qu'un chemin
   MoE IQR serait servi : il est explicitement sauté.
2. **QSA et contexte natif 262144.** Le conteneur déclare 262144. Avec
   `max_blocks=65536`, le plan `score` à 2048 lignes fait exactement 512 Mio ;
   à 4096 lignes il fait 1 Gio et le plugin refuse. Le déploiement doit garder
   `max_mixed_tok <= 2048` ou baisser `max-model-len`. La mesure a utilisé 200000
   (50000 blocs) et 4096 tokens, avec `qsa_rows=2048`.
3. **`w4nl` hors périmètre v1 (checklist §7.2).** Le conteneur a 25 590 poids
   experts `w4nl` (tous les `down`, plus le MTP). Sans la row `w4nl`, ces 25 590
   poids tombent sur `libavx` ou ne se déclarent pas. La case §7.2 doit être
   rouverte pour Flash-Next, sinon le modèle ne peut pas être servi par `libr3d`.
4. **MTP — dans la cible depuis l'ADR-7.** `mtp_enter` et `logit_rerank` n'ont
   pas de row device. Le conteneur demande 3 tokens de spéculation ; tant que
   les deux rows manquent, le run garde `num-speculative-tokens: 0`, ce qui
   retire aussi la couche 48 et ses poids `w4nl`/IQR. Les experts du MTP 48
   sont déjà servis (`moe_gemm_w4nl`).
5. **Vision — dans la cible depuis l'ADR-7.** `gemm_nt_bias` et `grid_embed`
   n'avaient pas de row device à l'audit ; elles sont servies depuis W15
   (`gemm_nt_bias_bf16`, `grid_embed_bf16`), comme `attn_dense` hd72. Ce qui
   manque est le décodage média : rebuild avec FFMPEG et un tour image (W26).
6. **`moe_gemm_iqr` n'a pas de référence hôte** (R4) : `libref` n'implémente pas
   la row. Il faut la référence avant de valider le noyau device, comme prévu en
   W5.
7. **Agrégation par bande.** Le graphe agrège les instances de même géométrie.
   Une row qui accepte à la fois `kSigned` et `kTrellisMad` est nécessaire pour
   les bandes mixtes `N=512 K=2560` et `N=2560 K=6144`. Sans elle, les poids IQR
   du MTP 48 restent non résolus même si leur forme est acceptée.
8. **`gdn_conv_recurrent_update_had`.** La paire non fusionnée est un repli
   correct mais coûteux ; le forcer dans le plugin pour l'audit a aussi forcé
   `q8_fed=false`, ce qui a fait apparaître des trous `ssm_out`/shared/PLE
   i8. Ces trous sont réels (forme `kSigned`), mais leur mesure exacte en
   performance n'a pas été faite.
9. **Pas de hash du modèle.** Le fichier n'a pas été haché (121 Go) ; l'audit
   s'appuie sur la taille, la date et les tables. Si le conteneur est remplacé,
   refaire `--recipe` et les plans.

## 9. Ordre de marche recommandé

1. **Forme `kSigned` 8 dense** (pas une nouvelle row) : `RAD_I8` dans
   `signed_bits`, `X(kSigned,8,0,0,0,kBf16)` dans `R3D_IQR_FORMS`, acceptation
   dans `form_supported` de `r3d_gemm_iqr_m16.hip` et
   `r3d_gemm_iqr_gated.hip`. Teste `attn_k/v/output`, `ssm_out`, `shexp`,
   `ple_*`, et débloque les bandes mixtes du MTP 48.
2. **`moe_gemm_iqr`** (`moe_gemm_q`, `i8a8h`, `kTrellisMad` L16 lrb0
   `kRowF32` u4/u5/u3, `lrun` 7/8, `parts` 1/2, classes paires/impaires,
   `top_k=10`, `n_expert=512/510/508`) : layout, run map, noyau 16 lignes et
   préfill, scratch, refus explicites. C'est le chemin critique M2.
3. **`moe_gemm_w4nl`** (`moe_gemm_q`, `w4nl64a8i8h`, table f32[16], E4M3/64 +
   f32, `group=64`, `a_order` token/sorted, `N=1280/2560`, `K=2560/640`) :
   décodage des niveaux `kW4nlI8`, noyau MoE, cas MTP 48.
4. **`moe_gemm_iqr_mod4` / `moe_gemm_w4nl_mod4`** seulement si `tp=4` est ciblé ;
   le déploiement actuel est `tp=1` (une W7900, 121 Go tiered).
5. **QSA** : `qsa_work_bf16`, `qsa_block_key_bf16` (imrope/mrope),
   `qsa_score_bf16` (f32 out), `qsa_select_i32` (non signé + signé). Partir de
   `libref/ref_qsa.cpp` et de `libr4d/r4d_qsa_*` ; `qsa_tail_store` existe déjà.
6. **`kv_store_i8` block 4** et **`rope_imrope_h128`**, ou replier le rope de la
   query dans `qsa_work`/`qsa_score` pour éviter une row de plus.
7. **`gdn_conv_recurrent_update_had`** : porter la row `libr4d` ou garder la
   paire non fusionnée comme repli documenté ; la version fusionnée écrit les
   codes tournés attendus par `ssm_out`.
8. **`router_gemm_topk_scatter`** device avec `protect` (décodage), ou accepter
   deux lancements `router_gemm` + `router_topk_scatter` déjà servis.
9. **`gemm_nt` N=1 préfill** (shared gate) : row GEMV ou padding N.
10. **Vision** : `gemm_nt_bias`/`grid_embed` servies depuis W15, reste le
    décodage média (W26) ; **MTP** `mtp_enter` / `logit_rerank` (W25) — les deux
    sont dans la cible depuis l'ADR-7.

## 10. Réponse courte

1. Le conteneur est mixte : 48 couches + MTP 48, 512 experts, top-k 10, 1 partagé,
   hidden 2560.
2. `moe_gemm_q` IQR `i8a8h` : aucune row `libr3d` ; `moe_gemm_iqr` à écrire.
3. `moe_gemm_q` `w4nl64a8i8h` (down + MTP) : aucune row device ; hôte seulement.
4. Le dense IQ8R (`kSigned` B=8) n'est pas reconnu par le layout/`form_supported`
   des rows `gemm_iqr_*` ; c'est le trou le plus large (k/v/output, ssm_out,
   shexp, PLE).
5. Les IQR `u5` qg/ssm_inz et `u3` qg MTP sont déjà servis par `gemm_iqr_*`.
6. `qsa_tail_store` est servi ; `qsa_work`, `qsa_block_key`, `qsa_score`,
   `qsa_select` manquent.
7. `kv_store` block 4 i8 et `rope` h128 imrope manquent aussi.
8. `gdn_conv_recurrent_update_had` n'existe que dans `libr4d`.
9. 10 experts bf16 protégés (34: 407/496 ; 44: 292/350 ; 46: 290/392 ;
   47: 122/143/399/445) sont servis par `moe_gemm_bf16`, mais leur layer émet
   aussi IQR + w4nl.
10. Risques : plan QSA 512 Mio au contexte natif, graphe obtenu avec deux
    contournements de debug ; `w4nl`, MTP et vision ne sont plus des exclusions
    (ADR-7) mais du travail (W25, W26, checklist §7.2).
