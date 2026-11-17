# Rapport : faire tourner Qwen3.8-Flash-Next (IQR-V4) sur RDNA3

Date : 2026-11-15 — hôte `ai` (Radeon PRO W7900, `gfx1100`), radiance 2.0.0 (commit
`ddcb9c6`), ROCm 10.1. Objet : évaluer ce qu'il manque à `libr3d` pour servir le
conteneur `qwen3.8-next-flash-iqr-v4.rad`, estimer l'effort, puis définir comment
mesurer les noyaux obtenus par rapport à un noyau idéal et par quel plan
d'optimisation s'en rapprocher.

> **Mise à jour du 2026-11-16 — l'analyse ci-dessous est historique.** Le portage
> a été fait et mesuré ; les prévisions de ce document sont remplacées par les
> chiffres réels (détail : `TABLEAU_REEL_IDEAL_RDNA3.md`, `JOURNAL_RDNA3.md`,
> `HANDOVER_RDNA3_PHASE7.md`). Lire les sections 1, 7.3 et 9 avec ce correctif :
>
> - Le conteneur **démarre et sert** (HTTP) sur la W7900 avec `libr3d` seul :
>   `r3d_selftest` **190/0/16** (169 avant W28), kbench Flash-Next **948/948**,
>   2B **384/384** et 27B **653/653**, 0 échec, 0 divergence ; MiniCPM5-2B IQ8R
>   sert à `NS=0` et `NS=3` (§7.6). État de branche `rdna3-iqr`, code `a8bc412`.
> - **Décodage mesuré : 217–250 ms/pas** (4,0–4,6 tok/s à `NS=0`), et non
>   30–45 ms/pas. Le poste dominant n'est ni le lien ni le MoE mais **`hc_read`** :
>   1,77 ms par appel à M=1 pour 6,55 Mo de poids (η 0,6 % de la borne bande),
>   ×48 appels par pas = ~93 ms/token, plus `hc_enter`/`hc_write` au plancher de
>   lancement. Avec le MTP (W25) le décodage passe à **8,21 tok/s** (1,82×).
> - **Préfill mesuré : 42 tok/s** (2 290 et 4 570 tokens), et non ~2,5 K tok/s.
>   `hc_read` y prend 47 % (470 ms par appel à M=2 048) ; `qsa_score` ~8 %
>   (83,4 ms à M=512, 103 GFLOP/s) ; `ple_gate` tient 69 TFLOP/s.
> - **η lien = 1,7 %** de la borne de 2 545 tok/s, donc le critère M2 (≥80 %)
>   n'est pas tenu ; ce n'est pas le MoE qui le fixe : R23 a mesuré le temps par
>   row MoE (IQR 70,5 % de la bande à M=32, w4nl 51 % à M=16, 10–12 % à M=1,
>   borné par le décodage treillis).
> - **Multi-tour** : cache de préfixe ×12 sur un prompt identique (4 096 tokens
>   servis, TTFT 108 → 9 s) ; le rejeu multi-agents servait 0 %, il sert
>   **52,6 %** depuis R22, et le tier KV hôte est **actif** depuis R21 (copie et
>   restauration vérifiées par hash).
> - **27B multi-séquence** : cassé par une row qui lisait le paramètre rangé
>   comme le nombre de lignes (au lieu d'une capacité de bande), arrêtant le
>   modèle dès un pas plus étroit que sa bande. Corrigé, avec la même faute dans
>   trois rows router ; 27B `--max-num-seqs 4` sert 4 requêtes à 67,3 tok/s
>   agrégés. Mode de test ajouté : `rad-kbench --band-capacity`.
> - La recommandation « RDNA4 plutôt que 20–25 semaines de portage » n'a plus
>   d'objet : le portage est livré ; le travail restant est l'optimisation de
>   `hc_read`, pas la faisabilité.
> - **Périmètre révisé le 2026-11-16 (ADR-7)** : la **vision** et le **MTP de
>   Flash-Next** sont dans la cible et **tous deux livrés**. Vision (W26) :
>   build `RAD_WITH_FFMPEG=ON`, `capabilities.image=true`, image 512²/1024²
>   servie. MTP (W25) : `mtp_enter`/`logit_rerank` en device, `NS=3`, **1,82×**
>   le décodage à texte identique (acceptation 40,3 %) ; les tables MTP 48 et
>   le draft head sont couverts par kbench (R23).

## 1. Résumé exécutif

- Le conteneur ne démarre pas sur la W7900 :
  `E qwen4exp_fp8: rad_arch_declare returned invalid argument` (engine_bringup.cpp:4794).
  Une cause certaine est l'absence, dans `libr3d`, des hooks de layout et des rows
  `gemm_iqr_*` / `moe_gemm_iqr` : le builder rejette un poids dont aucun noyau
  candidat ne sait lire l'encodage (`core/build/rad_builder.cpp:527`). L'op
  exact qui échoue le premier sera nommé par un dry-run instrumenté (§4).
- L'ISA RDNA3 ne bloque pas le décodage IQR : nous avons compilé et vérifié sur
  `gfx1100` `v_dot4_i32_iu8`, `v_wmma_i32_16x16x16_iu8`, `v_perm_b32`,
  `v_alignbit_b32`, `v_readlane_b32`, `v_permlanex16_b32`. Seules les charges
  transposées (`global_load_tr_b128`) sont `gfx12`-only, et le chemin IQR ne les
  utilise pas.
- Mais l'IQR n'est **qu'une** brique manquante. L'architecture `qwen4exp` exige
  aussi les connexions résiduelles à porte (HC), la couche PLE/n-gram, l'indexeur
  QSA, l'attention `head_dim=256` / GQA 12 / KV int8, et la tour vision. `libr3d`
  n'en fournit aucune (hors `qsa_tail_store`). Le portage réel est donc celui de
  l'architecture Qwen4-Exp, pas un simple ajout de noyaux de quantification.
- Effort estimé : **20 à 25 semaines-ingénieur** à un développeur pour un service
  texte + vision sans MTP ; 12 à 16 semaines à deux développeurs en parallèle.
  La tour vision coûte 3 à 4 semaines ; le MTP, 2 semaines de plus, est désactivable
  (`num-speculative-tokens: 0`) alors que la vision ne l'est pas pour ce conteneur.
- Performance attendue : le préfill d'un pas de 4096 tokens copie les 42 GiB du
  pool d'experts sur le lien, soit ≈1,6 s à 28 GB/s → ≈2,5 K tok/s, valeur de la
  référence RDNA4 mesurée (2545 tok/s). Le décodage, lui, dépend de la fraction
  d'experts résidents et du lien : environ **30 à 45 ms/pas** à une séquence contre
  28 ms pour la référence RDNA4, la W7900 compensant son PCIe 4.0 par 16 GiB de
  VRAM en plus.
- Recommandation : ne pas lancer le portage complet d'emblée. Faire d'abord un
  **spike IQR de 3 semaines** sur Qwen3.8-27B IQR-V4 (même quantification, architecture
  déjà servie par `libr3d`) pour mesurer les noyaux IQR réels sur `gfx1100`, puis
  décider. Si l'objectif est seulement d'obtenir des chiffres Flash-Next, une carte
  RDNA4 reste le chemin court.

## 2. Constat mesuré

| Élément | Valeur |
|---|---|
| Carte | Radeon PRO W7900, 48 GiB, `gfx1100` (RDNA3), 96 CU, 864 GB/s, PCIe 4.0 x16 |
| CPU / RAM | Ryzen 9 9950X (32 threads) / 122 GiB |
| ROCm / compilateur | 10.1, `hipcc` dans `/opt/rocm/bin`, cmake 4.2.3, ninja 1.13.2 |
| Modèle | `qwen3.8-next-flash-iqr-v4.rad`, 113 GiB (121 678 372 864 octets) |
| Réglage visé | `tp: 1`, 4 séquences, 200K, `expert_tiered`, pool 43 000 MiB, KV i8 |
| Échec | `rad_arch_declare` → `RAD_E_INVAL` ; seules libavx, libr3d, spechub et libref sont chargées |

Encodages réellement présents dans le conteneur (`rad-info --recipe`) :

| Tenseur | Encodage | Brique requise |
|---|---|---|
| `attn_qg` | `gptq codes=u5 tcq=16 tcq_code=mad … scale=f32` | IQR treillis, échelle f32 par ligne |
| `attn_output` | `gptq codes=u4 tcq=16 … group=128 scale=bf16 dyn=1` | IQR treillis, échelle par (ligne, 128) |
| `ssm_inz` | `gptq codes=u5 tcq=16 … scale=f32` | IQR treillis |
| `ssm_out`, `*_shexp`, `output` | `rtn codes=i8 group=128 scale=bf16` | int8 classique (déjà servi) |
| `ffn_gate_up_exps` | `gptq codes=u4 tcq=16 … scale=f32` | IQR treillis (MoE) |
| `ffn_down_exps` | `gptq table=w4nl group=64 scale=fp8_e4m3 scale2=f32 block2=*x*` | IQR table 4 bits + échelles E4M3 par 64 |
| 10 experts protégés | `cast dtype=bf16` | `moe_gemm` bf16 (déjà servi) |

Conclusion : le conteneur mélange, dans une même couche MoE, des experts IQR
treillis (`gate_up`) et des experts IQR table `w4nl` (`down`), plus quelques experts
bf16. Le noyau `moe_gemm_iqr` doit donc accepter plusieurs formes par couche, ce que
fait libr4d via le mot de forme porté par le layout de chaque plan.

## 3. Ce que l'ISA RDNA3 (PDF, février 2023) garantit

Instructions pertinentes, vérifiées dans `external-docs/rdna3-shader-instruction-set-architecture-feb-2023.pdf` :

| Besoin | Instruction ISA | Présence `gfx1100` | Preuve |
|---|---|---|---|
| Produit matrice int8 | `V_WMMA_I32_16X16X16_IU8` (opcode 68, p. 359) | oui | compilation + `v_wmma_i32_16x16x16_iu8` dans l'assembleur généré |
| Produit matrice int4 | `V_WMMA_I32_16X16X16_IU4` (opcode 69) | oui (documenté) | non utilisé par le chemin IQR |
| Somme de produits d'octets | `V_DOT4_I32_IU8` (opcode 22, §16.10) | oui | `__builtin_amdgcn_sudot4` compile et émet `v_dot4_i32_iu8` |
| Permutation d'octets | `V_PERM_B32` | oui | `__builtin_amdgcn_perm` |
| Fenêtre de bits | `V_ALIGNBIT_B32` | oui | `__builtin_amdgcn_alignbit` |
| Lecture de voie | `V_READLANE_B32` | oui | `__builtin_amdgcn_readlane` |
| Échange inter-voies | `V_PERMLANE16_B32` | oui | `v_permlanex16_b32` émis pour `gfx1100` |
| Produit 24 bits | `V_MUL_U32_U24` | oui | `__umul24` |
| Double émission VALU | `VOPD` (§16.11, p. 360) | oui (ISA) | non exploité : ROCm 10.1 désactive l'appairage |

Absences confirmées et leurs substituts :

- `global_load_tr_b128` : refusé pour `gfx1100` (« needs target feature
  `gfx12-insts,wavefrontsize32` ») ; `libr3d` le remplace par des transposes en
  registres, et le chemin IQR ne l'utilise pas.
- Pas de VGPR 16 bits ni de `v_cvt_pk_bf16_f32` : le décodage IQR travaille en
  dwords/octets et n'en dépend pas.
- `v_cvt_f32_fp8` (gfx12) : un repli logiciel de 11 opérations existe déjà
  (`libr4d/r4d_fp8_wmma.h:100-111`) pour les échelles E4M3 du `w4nl`.

Différence structurante : les fragments WMMA ne se répartissent pas de la même
façon. Sur `gfx11`, une voie porte 16 octets d'opérande (répliqué sur les deux
moitiés de la vague), contre 8 sur `gfx12` (`libr3d/r3dh_wmma.h:1-40` ;
`/opt/rocm/include/ck/utility/amd_wmma.hpp` : `int8x16` vs `int8x8`). C'est la
principale taxe de portage : registres d'opérandes doublés et lectures LDS
redondantes, donc un plafond WMMA plus bas à géométrie égale.

## 4. Périmètre réel du manque (au-delà de l'IQR)

Ops nécessaires à `qwen4exp` absents de `libr3d` (liste extraite des blocs
déclarés par le plugin ; `libr3d` couvre déjà GDN (h128), norms, quantiseurs
int8, router, scatter, KV store, `gemm_nt_q` int8/fp8, `moe_gemm`) :

| Famille | Ops manquantes | Présent libr3d | Obligatoire ici | Source libr4d | Risque |
|---|---|---|---|---|---|
| Résidu à porte (HC) | `hc_read`, `hc_write`, `hc_project`, `ar_hc_write`, `ar_gather_hc_write` (`hc_enter` = MTP) | non | oui | `r4d_hc_bf16.hip` (5 009 l.), `r4d_mhc.hip` (624 l.), `r4d_ar_hc.h` | élevé (volume, sémantique) |
| PLE / n-gram | `ple_conv`, `ple_gate`, `ngram_hint`, `embed_lookup_q` | `ngram_ids` seulement | oui | `r4d_ple_conv_bf16.hip`, `r4d_ple_gate_bf16.hip` | moyen |
| Indexeur QSA | `qsa_work`, `qsa_block_key`, `qsa_score`, `qsa_select` | `qsa_tail_store` seulement | oui | `r4d_qsa_work_bf16.hip`, `r4d_qsa_block_key_bf16.hip`, `r4d_qsa_score_bf16.hip`, `r4d_qsa_select_bf16.hip` (≈2 250 l.) | moyen |
| Attention pleine | `attn_paged` h256/GQA12/KV i8 (décodage + préfill) | h128, KV bf16 ou fp8 uniquement (`r3d_rows.cpp:1437-1446`) | oui | `r4d_attn_decode_h256_gqa6.hip`, `r4d_attn_paged_h256_gqa12.hip`, `r4d_attn_prefill_h256_gqa6.hip`, `r4d_attn_gen.hip` | élevé |
| Tour vision | encodeur ViT, `attn_dense`, `grid_embed` | non | oui (conteneur `vision`) | `r4d_encoder_bf16.hip` (1 621 l.), `r4d_vit_bf16.hip` (752 l.), `r4d_attn_vit_h72_bf16.hip` | moyen-élevé |
| IQR | hooks layout/relayout + rows `gemm_iqr_nt_m64`, `gemm_iqr_tiled`, `moe_gemm_iqr` | non | oui | `r4d_iqr_dec.h`, `r4d_gemm_iqr.hip`, `r4d_moe_iqr.hip`, `r4d_iqr_layout.cpp` (≈3 450 l.) | moyen |
| MTP | `mtp_enter`, `mtp_couple_propose`, `logit_rerank` | non | non si `num-speculative-tokens: 0` | dans `r4d_hc_bf16.hip`, `r4d_rowtopk_bf16.hip` | faible (désactivable) |

Non requis pour ce conteneur :

- Les formes codebook `w4nl64a8h` du noyau `w4a8` (`libr3d/r3d_moe_gemm_w4a8.hip:22-25`
  les liste comme « remaining ») : le conteneur IQR-V4 les porte dans le chemin IQR
  table, pas dans `moe_gemm_q` w4a8. Cette piste ne sert qu'au conteneur de
  production `w4nl64-i8lm.rad`.
- `gemm_iqr_gated`, `qk_norm_rope_gate*`, `kv_store_streamed`, `router_*_topk_scatter`,
  `moe_scatter_inv` : déclarés conditionnellement (`rad_op_declared`) et remplacés
  par les ops séparés déjà servis lorsqu'ils manquent.

Point de méthode : l'échec à `declare` masque la liste complète, car le premier op
non résolu arrête le plugin. La liste ci-dessus a été obtenue statiquement
(`RAD_OP(b, …)` dans les blocs inclus) ; un dry-run instrumenté du builder la
confirmera.

## 5. Faisabilité technique, brique par brique

**IQR (densité et MoE).** Le décodeur libr4d est déjà portable : il n'utilise que
`perm`, `alignbit`, `readlane`, `sudot4`, `__umul24` et du LDS, tous disponibles
(`libr4d/r4d_iqr_dec.h`). Ce qui est `gfx12`-spécifique, ce sont l'assemblage des
fragments WMMA (`r4d_fp8_wmma.h:47-71`) et les tuiles de préfill ; `libr3d` possède
déjà l'équivalent `gfx11` (`r3d_common.h:239` : `wmma_i8` en `int8x16`), et un noyau
MoE w4a8 opérationnel (`r3d_moe_gemm_w4a8.hip`). Le portage consiste donc à découpler
les décodeurs de l'en-tête WMMA `gfx12`, à fournir l'adaptateur de fragments
`gfx11`, à recopier les hooks de layout (code hôte pur) et à brancher les rows.
C'est le morceau le mieux borné du projet.

**Attention h256/GQA12/KV i8.** `libr3d` ne sert que `head_dim=128` et deux types de
KV (`bf16`, `fp8_e4m3`), pour décodage et préfill. Les couches pleines de Qwen4Exp
demandent 24 têtes Q, 2 têtes KV, 256 dims, KV int8. C'est un noyau nouveau, pas une
variante de paramètre : le split de préfill et la boucle de décodage doivent être
réécrits pour `gfx11` (la version libr4d est en `r4d_attn_gen.hip`, 999 l., plus les
spécialisations h256).

**QSA.** Les quatre noyaux manquants s'appuient sur des GEMM bf16 et des top-k :
ils réutilisent l'infrastructure existante, mais leur logique de blocs, de ring de
queue et de sélection doit être portée à l'identique. Le risque est la parité
numérique avec la sélection dense (exigence : reproduire le chemin dense au bit près
sous le seuil de budget).

**HC et PLE.** HC est le plus gros morceau (5 000 lignes) : norme groupée sur un flux
de 10 240, porte de lecture bas rang, quatre scalaires de réécriture, variantes de
réduction distribuée. PLE est plus petit mais touche au n-gram (table de 47,7 GiB
mmapée) et au embedding quantifié. Tous deux sont memory-bound, donc peu sensibles
à l'ISA : le portage est surtout du volume et de la parité.

**Vision.** La tour ViT (≈3 000 lignes) n'est pas indispensable au texte mais le
conteneur la déclare (`m.media`). Deux options : la porter, ou reconvertir un
conteneur sans branche média ; la seconde réduit l'effort mais change l'artefact
testé.

**MTP.** Déclarable seulement si `max_spec > 0` ; `--num-speculative-tokens 0`
l'écarte. À exclure du premier jalon.

## 6. Effort et plan

Hypothèses : un développeur senior connaissant HIP/ROCm, à temps plein, avec accès
à la W7900 ; parité numérique exigée ; aucune optimisation fine avant validation.

| Lot | Contenu | Semaines |
|---|---|---|
| A0 | Spike IQR : décodeur + layouts + `gemm_iqr_nt_m64` (GEMV/m16/tuile 16) sur une forme treillis et une forme table ; parité sur les fixtures `data/kernels.rkb` | 3 |
| A1 | IQR dense complet : m64, tuile préfill, variantes d'échelles (f32 ligne, bf16 par 128, E4M3 par 64 + second niveau) | 3 |
| A2 | `moe_gemm_iqr` : run map, préfill map, mélange treillis/table/bf16 protégés, expertise hétérogène | 4 |
| B1 | Attention h256/GQA12/KV i8 : décodage, préfill, split | 4 |
| B2 | QSA complet | 3 |
| B3 | HC + PLE | 5 |
| B4 | Tour vision (ou 0 si conteneur sans média) | 3-4 |
| C | Intégration, tests de bout en bout, durcissement | 2 |
| **Total** | texte + vision, hors MTP | **20-25** (1 dev) ; **12-16** (2 devs sur A et B) |

Jalons utiles :

1. **Spike A0 sur 27B IQR-V4** : les ops dont dépend l'architecture Qwen3.8-27B
   (GDN, attention h128, int8) existent déjà dans `libr3d`, et son conteneur IQR-V4
   n'ajoute que la quantification IQR ; le démarrage sur `gfx1100` reste à valider.
   C'est le moyen le moins cher de mesurer les noyaux IQR sur `gfx1100` et de
   valider la stratégie de fragments. Livrable : tableau réel/idéal des noyaux IQR,
   avant tout engagement sur Flash-Next.
2. **Décision** : si l'efficacité mesurée des noyaux IQR atteint les cibles de la
   §7, poursuivre ; sinon, s'arrêter et servir Flash-Next sur RDNA4.

## 7. Performance réelle vs noyau idéal

### 7.1 Définition du noyau idéal

Pour chaque classe, le « noyau idéal » est la borne inférieure physique sur la
W7900, avec : une lecture de chaque octet de poids par tuile, zéro conflit de banc
LDS, WMMA/VALU à plein débit, recouvrement parfait des chargements et du calcul,
aucun surcoût de lancement. L'efficacité est `η = t_idéal / t_mesuré`.

Constantes de la carte (constructeur ; à confirmer par mesure) :

| Ressource | Valeur |
|---|---|
| CU / horloge soutenue | 96 CU / ≈2,3 GHz |
| Bande passante VRAM | 864 GB/s crête, ≈650-730 GB/s utiles |
| VALU int32 | ≈15 Tops (`__umul24`, `perm`, `alignbit`, `sudot4` mélangés) |
| WMMA IU8 | ≈300 TMAC/s (≈600 TOPS), estimé à 1 WMMA 16×16×16 par ~3 cycles/CU |
| Lien hôte | PCIe 4.0 x16 : 31,5 GB/s crête, ≈26-28 GB/s utiles |
| WMMA `gfx11` | opérandes 16 octets/voie (×2 vs `gfx12`) → lectures LDS redondantes |

### 7.2 Noyaux IQR, bornes et cibles

Formes prises dans le conteneur (hidden 2 560, 48 couches, 12 blocs [3 GDN + 1 QSA]) :

| Noyau | Forme | Borne | Idéal | η visée | Commentaire |
|---|---|---|---|---|---|
| `gemm_iqr_nt_m64`, M=1 | `q\|gate` 12 288×2 560, u5 → 19,7 Mo | VRAM 864 GB/s | 22,8 µs | ≥65 % (≤35 µs) | décodage ≈10 µs, dominé par la VRAM |
| `gemm_iqr_tiled`, M=4096 | même forme, 257,7 GFLOP | WMMA ≈600 TOPS | 0,43 ms | ≥65 % (≤0,66 ms) | décodage amorti sur 4096 lignes |
| `moe_gemm_iqr`, 1 pas | 48 couches × top-10 × 4,92 M poids × 0,5 o ≈ 1,18 Go | lien ou VRAM selon résidence | voir 7.3 | ≥80 % de la borne lien | le calcul de décodage (~1,5 ms) est négligeable |
| `moe_gemm_iqr`, préfill 4096 | 42 GiB de pool copiés une fois | lien 28 GB/s | 1,61 s | ≥90 % | soit ≈2,5 K tok/s |
| `attn_paged` h256/KV i8, 1 pas | 12 couches pleines, 2 048 tokens sélectionnés | KV i8 ≈2 Mo/couche → 25 Mo | ≈30 µs + lancements | ≥60 % | QSA réduit déjà le KV lu |
| `qsa_select` | 512 blocs retenus | ALU + lancement | ≈20-50 µs | ≥50 % | sensible au surcoût de lancement |

### 7.3 Bout en bout : le lien, pas l'ALU

> Prévision **non confirmée** : le décodage mesuré est 5 à 8 fois plus lent et le
> préfill 60 fois plus lent que ce modèle, parce que `hc_read` domine (voir §7.5).
> Les chiffres ci-dessous restent utiles comme borne du MoE seul.

Le pas de décodage lit environ 1,18 Go de poids d'experts (4 bits). Si `r` est la
fraction résidente sur la carte, le temps incompressible est :

```
t_pas ≈ (1 - r) × 1,18 Go / BW_lien + t_calcul + t_lancements
```

avec `t_calcul ≈ 1,5-2,5 ms` (décodage IQR) et `t_lancements` dominé par HC/QSA
(des dizaines de petits noyaux par pas). Scénarios :

| BW lien effectif | r = 0,20 | r = 0,35 | r = 0,50 |
|---|---|---|---|
| 28 GB/s (PCIe 4.0) | ≈35 ms | ≈28 ms | ≈21 ms |
| 56 GB/s (PCIe 5.0) | ≈19 ms | ≈15 ms | ≈12 ms |

La référence RDNA4 `tp: 1` mesurée dans `docs/GUIDE.md` est de 28,0 ms/pas à une
séquence (95,3 ms à quatre). La W7900 ajoute 16 GiB de VRAM (donc un `r` plus
grand) mais divise le lien par deux : le résultat attendu se situe autour de
**30-45 ms/pas**, à confirmer par la mesure. Le préfill, lui, est déjà lié au lien
sur la référence : 42 GiB / 1,609 s = 28 GB/s, exactement la valeur mesurée de
2545 tok/s à 4096 tokens. Sur la W7900, le plafond est le même ; l'objectif
réaliste est **2,3-2,6 K tok/s** si le noyau MoE sature le lien.

Conséquences pour l'optimisation :

- Optimiser le décodage IQR au-delà de ~1,5 ms/pas ne change pas le débit : la
  borne est le lien. Le levier est la résidence des experts (pool, prefetch,
  carte plus large), pas la micro-optimisation du décodeur.
- Les noyaux **denses** (trunk, poids en VRAM) sont, eux, sensibles au noyau :
  c'est là que l'écart au roofline se paie. La W7900 dispose de 96 CU et de
  864 GB/s, ce qui laisse de la marge pour le trunk ; la mesure le confirmera.
- Le préfill est le poste où un noyau inefficace coûte le plus cher, car il doit
  soutenir 28 GB/s de copie tout en décodant ; viser ≥90 % de la borne lien.

### 7.4 Protocole de mesure (à exécuter dès le spike A0)

1. **Parité d'abord.** Les fixtures `data/kernels.rkb` contiennent
   `gemm_iqr_nt_m64` et `gemm_iqr_tiled` (lignes listées dans
   `data/kbench_rows/kbench_rows_kernels_libr4d.txt`) : rejouer ces cas sur
   `gfx1100` et comparer aux sorties enregistrées. Pour `moe_gemm_iqr`, aucune
   fixture n'existe : produire une référence hôte (décodage IQR vers int8, puis
   `gemm_i8a8`) et comparer, en documentant les écarts d'arrondi f32.
2. **Micro-benchmarks.** `rad-kbench --home "$RADIANCE_HOME" --kernels
   libref:libr3d --op <op>` sur les formes exactes du modèle, 30 répétitions,
   médianes et p95, horloges et températures relevées par `rocm-smi` avant/après.
3. **Bout en bout.** `/stats` du serveur (ms/pas, tok/s de préfill), 1 et 4
   séquences, préfixe froid puis chaud ; comparer au modèle de la §7.3.
4. **Remplir le tableau réel/idéal** de la §7.2 ; c'est le document de sortie du
   spike, avant toute décision de poursuite.

Critères d'acceptation proposés :

- Parité bit-à-bit sur les rows IQR denses rejouées ; écart documenté (≤1 ulp f32)
  pour le MoE.
- `η ≥ 65 %` sur les noyaux denses IQR, `≥ 80 %` de la borne lien sur le MoE,
  `≥ 90 %` sur le préfill de bout en bout.
- Pas de régression sur les lignes `kbench` existantes de `libr3d`.

### 7.5 Mesures réelles sur W7900 (2026-11-16)

Toutes à froid, arbre `w15-main` (code `a772d70`), `rad-kbench --kernels
libr3d:libref --bench`, horloge 2 882–2 894 MHz sans throttler (sauf la passe
QSA à 2 789 MHz, PPT0). Cas vérifiés : 182 (HC/PLE) + 147 (QSA), 0 échec.

| poste | M=1 | M=2 048 | η |
|---|---|---|---|
| `hc_read_e4m3` (n 2 560, hc 4, lr 320, 6,55 Mo) | **1,77 ms** | **470 ms** | **0,6 %** (borne 10,5 µs) |
| `hc_enter` / `hc_write` | 13,3 / 8,0 µs | 155 / 98 µs | plancher de lancement |
| `ple_gate` | 23,1 µs | 1 557 µs | 69 TFLOP/s à M=2 048 (56 % du pic) |
| `ple_conv` | 5,8 µs | 1 045 µs | 160 GB/s |
| `qsa_score` | 119 µs | 83,4 ms à M=512 | 103 GFLOP/s (2 % du pic) |
| `qsa_select` | 73,1 µs | 946 µs à M=512 | table/ALU |

Ce que cela change par rapport aux prévisions :

1. **Le décodage n'est pas lié au lien.** Le MoE n'est pas le poste mesurable
   dominant : `hc_read` coûte 0,23 ms par ligne au-delà de M=8 (la paire de
   matrices E4M3 est relue par ligne à ~28 GB/s), soit ~93 ms des 217–250 ms d'un
   pas à 4,0–4,6 tok/s. Les 30–45 ms/pas de la §7.3 supposaient les experts seuls
   et un `hc_read` à 18,7 µs (valeur du commentaire de la row).
2. **Le préfill n'atteint pas 2,5 K tok/s** : 42 tok/s mesurés, `hc_read` à 47 %
   (48 × 470 ms = 22,9 s pour 2 048 tokens) et `qsa_score` à ~8 %. La borne lien
   du MoE (1,61 s pour 4 096 tokens) reste donc très loin : η lien bout en bout
   = **1,7 %**.
3. **A/B de la forme de décodage** : les deux noyaux deux-grilles
   (`g_hc_decode_form = 1`) donnent les mêmes temps que le fused à tickets
   (1 782 vs 1 777 µs à M=1 ; 471 vs 470 ms à M=2 048), donc l'attente par
   tickets n'est pas la cause. Chaud et froid donnent aussi le même temps : ni la
   bande passante ni la résidence n'expliquent le coût, qui est dans le corps du
   read (lecture/décodage des matrices). C'est la mesure à instrumenter ensuite.
4. **Bout en bout** : préfill 38,7–41,9 tok/s (prompts distincts de 71–113
   tokens), décodage 4,41 tok/s à une séquence, 15,5 à quatre (3,86–3,93 chacune),
   22,1 à six. Contention : 4 décodeurs passent de 12,6 s seuls à 87,6 s à côté
   d'un préfill de 3 000 tokens (+590 %), le préfill tombant à 35 tok/s.
5. **Ce qui reste non mesuré** : rien de bloquant sur le périmètre v1. Le temps
   par row MoE est **mesuré** depuis R23 : IQR 70,5 % de la bande à M=32,
   w4nl 51 % à M=16, 10–12 % à M=1 (borné par le décodage treillis) ; kbench
   Flash-Next **968/968** et 27B **683/683** (0 échec, 0 divergence). L'image
   (W26) et le MTP (W25, `NS=3`, 4,51 → 8,21 tok/s, texte identique,
   acceptation 40,3 %) sont livrés ; les tables MTP 48 et le draft head sont
   couverts. Restent hors v1 : 10 cas `gemm_nt_q_gated` i8a8h sans référence
   libref, 8 `ngram_hint` sans oracle, et les 16 schemas sans oracle libre
   (`selftest`), dont `gate_had_quant_i8g` (couvert par un test device).

### 7.6 W24–W27 : `qsa_score` et IQ8R (2026-11-17)

**`qsa_score` à M=512** (Flash-Next, `--op qsa_score --bench`, trois passes
alternées base/candidat, arbre `w27-merge` puis `w27-qsa`) : médianes device
**79,0 / 82,7 / 83,3 ms**, horloge 2 842–2 852 MHz, **aucun throttler noté**,
258,5 MiB d'opérandes — cohérent avec les 83,4 ms de W18. La row hôte AVX du
même cas fait **9,0 ms** sur 32 threads, soit 9× moins. La cause est le motif
d'accès, pas le calcul : un thread traite un bloc, donc une warp de 32 lanes lit
**32 pages distinctes au même offset** ; chaque lane n'utilise que 2 octets d'un
secteur de 32 octets (amplification ×16), et la row est bornée par L2
(103 GFLOP/s, 2 % du pic). L'essai d'ILP — quatre blocs par thread, quatre
chaînes f32 indépendantes, arithmétique par (row, bloc) inchangée — a donné
**163,5 / 163,0 / 163,0 ms, 2× plus lent** (davantage de pages distinctes par
instruction) : il a été retiré. La suite utile est une réécriture **coalescée et
tuilée** (pages de blocs et lignes de `q` en LDS, un thread par (row, bloc)
conservé), avec ce protocole d'A/B.

**W28 (2026-11-17).** La réécriture coalescée est livrée : un workgroup stage
les pages de blocs et les lignes de `q` d'une tuile 32×64 en LDS, chaque thread
gardant ses chaînes f32 ascendantes. **M=512 : 82,27 → 2,23–2,30 ms (36–37×)**,
**bit-identique** à libref (rel_l2 0,00e+00), 72 checks exacts au test device,
4 946/4 946 en `--bench` dans les deux modes.

**`hc_read` (W28, R20).** Le profil engine (96 sites, 1,93 ms/appel, poids en
VRAM) a montré que le coût était dans le noyau : une lane par ligne au même
offset, 32× d'amplification de secteur en `down` (16 lanes actives sur 256) et
8× en `up`. Les deux noyaux stagent maintenant leur tranche de poids en LDS
(16 × 256 en `down`, 64 × 32 k en `up`), l'arithmétique restant identique.
À froid : **M=1 1 941 → 428 µs (4,5×)**, M=512 **114 → 12,4 ms (9,2×)**,
`ple_hc_device_test` 48 checks verts. En service, décodage Flash-Next
**4,48 → 12,73 tok/s (2,84×)**, texte cohérent.

**DSpark (W28, D2).** Le 2B à `NS=3` : **32,20 vs 20,64 tok/s (1,56×)**, texte
identique au bit, 33/210 drafts acceptés (15,7 %). Le cas selftest `w2a8`
déclarait son plan joint `U8` au lieu de `F16` et était toujours sauté ; un cas
numérique au bande du drafter (M=7, plan en mémoire device) le couvre.

**Collectifs tp>1 (W28, D3).** 18 rows `all_reduce`/`all_gather` portées de
libhalo (2/3/4/8 rangs, wires exact/wht6/ti8, gathers), schémas libr4d,
`device.p2p` requis ; test à deux rangs sur une carte (12 checks). Les jonctions
`ar_*` fusionnées restent des stubs.

**Trouvé à l'intégration.** Déclarer `gate_had_quant_i8g` fait prendre à
Flash-Next la branche fusionnée, émise sur **toute la projection** : `q`/`x` en
`[T, heads*head_dim]` alors que `n` nomme la largeur de tête, et la porte est une
vue `[T, heads, head_dim]` de pas de tête 512. La row refusait la géométrie ; le
pas de ligne et de tête est maintenant porté, un bloc de rotation reste dans une
tête, et un cas device dédié la couvre (103 checks sur 6 cas). Le 27B replie 30
cas `sigmoid`/`mul`/`quant_act_fp8` dans la même branche mais **répond au bit
près** comme avant (29,63 vs 29,50 tok/s).

**IQ8R groupe 256 (MiniCPM5-2B)** : le conteneur (3,04 Gio, 42 couches,
`n_embd` 2 048, `n_ff` 6 144, 16/2 têtes) déclare **523 ops / 423 poids** et
sert à `NS=0`. kbench ciblé sur les quatre ops IQ8R : **110/110**, pire rel_l2
**4,018e-05**, GEMM **bit-exact** (0,00e+00 sur 96 cas N×K×M) ; les deux
producteurs fusionnés n'ont pas d'oracle `libref` (« Not covered ») et sont
couverts par `act_i8g_group256_device_test` (67 checks sur 4 cas). Service en
**KV bf16** (les rows `attn_paged`/`kv_store` de cet arch n'acceptent pas i8 à
`head_dim=128`) : réponses cohérentes, **40,2 tok/s** sur 60 tokens générés
(69,0 tok/s sur 16), préfill 184–367 tok/s sur ces prompts courts. Le conteneur
2B **rejoué en entier** (`--kernels libr3d:libref --max-cases 1`) donne
**384/384**, 0 échec, 0 divergence, pire rel_l2 **1,810e-03** (30 rows device
exercées) : `attn_paged` gqa=8, `rope` neox h128, `kv_store`, `embed_lookup`,
la tête à 130 560 tokens et les sampleurs inclus.

## 8. Étape d'optimisation (après validation)

Ordre de priorité, gains attendus et garde-fous. Chaque étape doit battre la
précédente d'au moins 5 % et rester numériquement identique.

| # | Action | Cible | Gain attendu | Mesure |
|---|---|---|---|---|
| 1 | Tuile LDS du décodeur : vérifier et retuner le pas 136 pour les 32 bancs de `gfx11`, éviter les conflits sur la transposée 16×128 | dense IQR m16/tiled | 10-20 % | kbench, compteur de conflits (si exposé) |
| 2 | Double-buffer LDS et recouvrement charge/décodage/WMMA ; réutiliser le fragment B sur 2-4 tuiles de lignes | tiled, MoE préfill | 10-15 % | kbench |
| 3 | Fragments `gfx11` : construire les 16 octets par voie sans passage LDS redondant ; réutiliser A entre tuiles de colonnes | dense tiled, attention | 5-15 % | kbench |
| 4 | `V_DOT4` exact plutôt que widening f16 pour les sommes int8 (disponible sur gfx1100) | décodeur GEMV | 5-10 % | kbench + parité |
| 5 | MoE : blocks 8 vagues adaptés à 96 CU, ordre expert-majeur, réutilisation des tuiles décodées sur plusieurs tuiles de tokens, map de runs sans relecture | MoE décodage/préfill | 10-20 % | ms/pas, tok/s |
| 6 | Prefetch inter-couches et résidence : régler `host-pool-mib`, vérifier `expert prefetch on`, saturer le DMA ; mesurer le taux de copies utiles | décodage | 10-25 % | ms/pas, journal |
| 7 | Attention h256 : split-K de préfill, chargements KV i8 vectorisés, fusion `qk_norm_rope_gate` si le noyau existe | attention pleine | 10 % | kbench |
| 8 | QSA : regrouper les petits noyaux de l'indexeur par pas (12 couches pleines), réduire les lancements | QSA | 5-15 % de l'indexeur | ms/pas |
| 9 | Réglages système : lien x16 effectif, IOMMU `pt`, pool épinglé, proximité NUMA, limite de puissance et clocks soutenus | bout en bout | 5-15 % | rocm-smi, /stats |

Le tableau réel/idéal doit être mis à jour après chaque étape. On s'arrête lorsque
les cibles de la §7.4 sont atteintes, ou lorsque le gain marginal tombe sous 5 %
pendant deux itérations.

## 9. Décision recommandée

> **Décision révisée (2026-11-16)** : le portage est livré et Flash-Next sert le
> texte (voir l'encadré en tête et §7.5). La décision restante est l'ordre
> d'optimisation : (1) instrumenter puis corriger `hc_read` (1,77 ms/appel au lieu
> des 18,7 µs documentés ; ~40 % du décodage et 47 % du préfill) ; (2) les
> snapshots en rejeu multi-agents (0 % de cache) — **R22 livré** (52,6 % de
> tokens servis sur le rejeu 9 tours) et tier KV hôte livré (R21) ;
> (3) seulement ensuite `qsa_score` à M=512 et le MoE, dont le temps par
> row reste à mesurer. **Le périmètre élargi du 2026-11-16 (ADR-7) est
> livré** : la vision (W26) et le MTP de Flash-Next (W25, 1,82× le décodage).
> La faisabilité n'est plus en question, et « RDNA4 plutôt que porter » n'est
> plus la bonne alternative.

Plan d'implémentation détaillé, lot par lot, avec tests, jalons et critères de
sortie : `PLAN_IMPLEMENTATION_RDNA3.md`.

1. **Court terme (3 semaines)** : spike A0 sur Qwen3.8-27B IQR-V4 pour mesurer les
   noyaux IQR sur `gfx1100`. C'est le test le moins coûteux de la thèse « RDNA3 peut
   servir IQR ».
2. **Si le spike est concluant** : engager A1/A2 (IQR dense + MoE), puis B1-B3
   (attention h256, QSA, HC/PLE). Repousser vision et MTP au-delà du premier
   service texte (recommandation historique : l'ADR-7 du 2026-11-16 les ramène
   dans la cible, lots W25/W26 — **tous deux livrés**).
3. **Si l'objectif est uniquement de chiffrer Flash-Next** : une RDNA4 (R9700 ou
   équivalent) est plus rapide et moins risquée que 20-25 semaines de portage.
4. **À éviter** : commencer par la tour vision ou le MTP, ou micro-optimiser le
   décodeur MoE avant d'avoir mesuré que le lien n'est pas la borne. (Les deux
   premiers sont désormais des livrables de la cible — ADR-7, W25/W26,
   **tous deux livrés** — mais ils restent derrière `hc_read` dans l'ordre de
   valeur ; le conseil sur le MoE est toujours valable.)

## Annexe A — Preuves et références

- Échec : `E qwen4exp_fp8: rad_arch_declare returned invalid argument (near
  engine_bringup.cpp:4794)`, noyaux chargés : libavx, libr3d, spechub, libref.
- Compilation `gfx1100` (ROCm 10.1) : `v_dot4_i32_iu8`, `v_wmma_i32_16x16x16_iu8`,
  `v_perm_b32`, `v_alignbit_b32`, `v_readlane_b32`, `v_permlanex16_b32` ;
  `global_load_tr_b128` refusé (`gfx12-insts`).
- Contrat de sélection : `core/build/rad_builder.cpp:527` (un noyau dont le hook de
  layout ne lit pas l'encodage n'est pas candidat).
- IQR libr4d uniquement : `docs/OPS.md:2245` ; `libr3d/r3d_moe_gemm_w4a8.hip:22-25`
  (formes codebook w4nl « remaining »).
- Fragments : `libr3d/r3dh_wmma.h:1-40`, `libr3d/r3d_common.h:239`,
  `libr4d/r4d_fp8_wmma.h:47-71`.
- Attention h128 seule : `libr3d/r3d_rows.cpp:1437-1446`.
- Performances de référence RDNA4 one-card : `README.md:227-228`, `README.md:258-260`,
  `docs/GUIDE.md:850-865` (28,0 → 95,3 ms/pas avec prefetch).
- ISA : `external-docs/rdna3-shader-instruction-set-architecture-feb-2023.pdf`,
  §16.10 (opcodes 22, 68, 69 ; p. 358-359), §16.11 (VOPD, p. 360).
- Hôte : `API_DISTANTE.md` (SSH, specs, `<work-root>`, 424 Go libres sur `/opt`).

## Annexe B — Inventaire des sources à porter (lignes)

`r4d_hc_bf16.hip` 5 009 · `r4d_mhc.hip` 624 · `r4d_ar_hc.h` 106 ·
`r4d_ple_conv_bf16.hip` 250 · `r4d_ple_gate_bf16.hip` 176 ·
`r4d_qsa_work_bf16.hip` 223 · `r4d_qsa_block_key_bf16.hip` 228 ·
`r4d_qsa_score_bf16.hip` 459 · `r4d_qsa_select_bf16.hip` 1 215 ·
`r4d_attn_gen.hip` 999 · `r4d_attn_decode_h256_gqa6.hip` 1 195 ·
`r4d_attn_prefill_h256_gqa6.hip` 712 · `r4d_attn_vit_h72_bf16.hip` 446 ·
`r4d_vit_bf16.hip` 752 · `r4d_encoder_bf16.hip` 1 621 ·
`r4d_iqr_dec.h` 425 · `r4d_gemm_iqr.hip` 1 269 · `r4d_moe_iqr.hip` 711 ·
`r4d_iqr_layout.cpp` 918.

Total ≈ 17 500 lignes spécifiques, hors MTP et hors portions de `r4d_moe.hip`
(3 649 l.) réutilisées par le run map.

## 10. Écart RDNA4 → RDNA3 : kernels à porter ou recréer (2026-11-17)

Mesure faite sur l'état `de81ffe` en comparant les tables de rows :
`libr4d/r4d_rows.cpp` (**457 rows, 274 ops**) et `libr3d/r3d_rows.cpp`
(**164 rows, 101 ops**). **238 rows / 179 ops** n'ont pas d'équivalent RDNA3.
Depuis W27, `libr3d` en a **169** (les cinq rows IQ8R du tronc) : l'écart est
de **233 rows / 176 ops**. Ce chiffre est celui de la couverture *de toutes les
architectures que `arch/` sait déclarer* ; il ne signifie pas que la cible v1
est incomplète.

### 10.1 Ce qui est réellement requis

- **Cible v1 actuelle (Flash-Next, 27B, MiniCPM5-2B)** : aucun op manquant sur
  les chemins exercés — kbench **968/968** et **683/683**, selftest 169/0/15.
  Les seuls écarts conditionnels sont ci-dessous.
- **IQ8R (MiniCPM5-2B, artefact non-IQR)** : le tronc rotated int8 à
  **groupe 256** **charge et sert depuis W27** (producteurs groupe 256,
  `gemm_nt_q` `i8a8h` groupe 256). Restent, sans besoin pour `NS=0` : les
  variantes `ar_*`/non fusionnées (`tp > 1`), `gate_had_quant_i8g` et le fold
  `gemm_nt_q_gated` 256. Le drafter **DSpark** du même conteneur demanderait en
  plus les ops `dflash_*` (seulement si on sert `NS > 0`).
- **tp > 1** : `all_reduce`, `all_gather` et les fusions `ar_*` (18 rows) sont
  absents ; sans eux, `tp=2/4` n'est pas servable. Sans objet sur une carte.
- **Streaming KV** (`kv_store_streamed`, `kv_resolve`) : absent ; requis
  seulement si `--kv-resident`/le tier streamé est activé (le tier hôte actuel
  passe par `gather_rows_u8`/`scatter_rows_u8`).
- **Autres architectures** (`arch/` : qwen3.5, gemma, bailing, glm5, gemma4,
  ltx2, flux3, sam, qwen3asr, etc.) : les 238 rows restants. Ils ne sont pas
  requis par la cible v1, mais le sont si l'une de ces architectures devient
  une cible.

### 10.2 Écart complet par famille

| famille | rows | ops manquants |
|---|---:|---|
| `ar` | 18 | `all_gather`, `all_reduce` |
| `attn` | 58 | 30 ops : `attn_cross`, `attn_local_rel`, `attn_na3d`, `attn_paged_gate_had_quant`, `attn_paged_gate_quant`, `attn_paged_qrow`, `attn_paged_qrow_tree`, `attn_prefix_self`, `attn_vsa`, `dsa_pool`, `dsa_query`, `dsa_score`, `embed_attn`, `kv_resolve`, `kv_store_streamed`, `ministral_rope`, `mla_attn`, `mla_attn_paged`, `mla_project`, `pixtral_rope`, `position_scale`, `qk_norm_rope`, `qk_norm_rope_gate`, `qk_norm_rope_gate_kv_store`, `qk_norm_rope_gate_kv_store_streamed`, `qkv_norm_rope`, `qkv_norm_rope_kv_store`, `rope_rows`, `sam_attn`, `vsa_select` |
| `conv` | 8 | `causal_conv1d`, `conv1d`, `conv2d_nhwc`, `conv3d_ndhwc`, `conv_t1d`, `tconv_dw` |
| `dflash` | 4 | `dflash_conv`, `dflash_markov_greedy`, `dflash_markov_greedy_f32`, `dflash_tree` |
| `draft` | 4 | `lili_attn`, `lili_cand`, `lili_factor`, `lili_walk` |
| `elem` | 91 | 85 ops : `bwe_up`, `chan_affine`, `chan_split`, `chan_stats`, `clamp`, `cond_aug`, `conv1d_dw`, `conv2d`, `conv2d_unfold`, `cu_repeat`, `d2s3d`, `d2s_add`, `end_status`, `flow_euler`, `flow_renoise`, `flow_res_multistep`, `flow_x0_step`, `fourier_embed`, `frame_mix_store`, `frame_store`, `gate_add`, `gather_rows2`, `gelu_mul`, `gelu_quick`, `gelu_tanh`, `grid_embed_aa`, `grid_embed_cpe`, `grid_pos3`, `head_gate`, `im2col_nhwc`, `int_affine`, `interleave2`, `iota`, `kd_input`, `kd_step`, `lane_gather`, `lane_store`, `logit_softcap`, `ltx_guide`, `mix_rows`, `mul_weight`, `patch_fold`, `patch_unfold`, `pix_patch`, `point_fourier`, `pool_cols`, `pool_rows`, `randn_lanes`, `ras_pick`, `rel_sinusoid`, `relu`, `resample_nearest`, `resize_bilinear`, `rope_split`, `rvq_embed`, `s2d3d`, `s2d_avg_add`, `scale_rows`, `scatter_add_rows`, `seg_mean`, `seg_mean_bf16_sum`, `seg_sinusoid`, `seg_softmax_pool`, `snake`, `snake_aa`, `stft_logmel`, `sub_i32`, `swiglu_clamped`, `swiglu_step`, `table_add`, `tile_crop`, `tile_grid`, `tile_stitch`, `timestep_embed`, `to_rgba8`, `vol_index`, `vq_encode`, `vtile_crop`, `vtile_gather`, `vtile_join`, `vtile_plan`, `vtile_stitch`, `win_finish`, `win_input`, `win_prep` |
| `gdn` | 2 | `kda_update`, `ssd_update` |
| `gemm` | 7 | `gemm_nt_f32`, `lora_expand`, `lora_expand_mm`, `lora_shrink` |
| `misc` | 1 | `prefetch` |
| `moe` | 4 | `moe_gather_inv`, `moe_scatter_inv`, `router_sigmoid_group_topk`, `router_sigmoid_topk_scatter` |
| `norm` | 21 | `add_rmsnorm`, `group_norm`, `join_add_scale`, `l2norm_chan`, `ln_mod`, `ln_mod_had_quant_i8g`, `mhc_contract`, `mhc_post`, `mhc_pre`, `mhc_pre_finish`, `mhc_pre_partial`, `norm_add_norm`, `norm_add_norm_route`, `norm_add_norm_route_i8g`, `pixel_norm`, `ple_conv_bias`, `rms_mod`, `rms_mod_had_quant_i8g`, `rms_mod_quant_i8`, `rmsnorm_p`, `rmsnorm_rows` |
| `quant` | 15 | 11 ops : `ar_ln_had_quant_i8`, `ar_rmsnorm_had_quant_i8g`, `ar_rmsnorm_quant_fp8`, `ar_rmsnorm_quant_i8`, `ar_rmsnorm_quant_i8g`, `gate_had_quant_i8g`, `gate_quant_fp8`, `gelu_mul_quant_i8g`, `gram_accum`, `rmsnorm_had_quant_i8`, `rmsnorm_quant_i8g` |
| `sample` | 3 | `cfg_topk_sample`, `sample_chain`, `sample_merge_topk` |
| `select` | 1 | `pair_argmax` |
| `vision` | 1 | `pixtral_merge` |

### 10.3 Rows RDNA4-only à recréer pour gfx11

Ces rows reposent sur des instructions absentes ou différentes sur RDNA3
(WMMA gfx12, mma à échelle de bloc, fp4/fp6). Elles ne se « portent » pas :
il faut réécrire le noyau avec `v_dot4` / `v_mfma_f32_16x16x16_bf16` ou
déclarer la forme non servie.

- `lora_expand_mm_wmma`, `lora_shrink_wmma`, `lora_shrink_wmma_q` ;
- `mla_attn_rows_bf16`, `mla_attn_matrix_bf16`, `mla_attn_*_fp8kv` ;
- `dsa_score_fp8` (FP8 WMMA ; `dsa_pool`/`dsa_query` sont des noyaux
  wave-local probablement portables, à vérifier) ;
- `dequant_w4_bf16` (ordre des fragments WMMA) ;
- `moe_gemm_w4a8`, `moe_gemm_i8a8`, `moe_gemm_w4a8i8`, leurs variantes
  `_mod4` (le RDNA3 a des équivalents IQR/`w4nl`, pas ces formats ; ce n'est
  pas une recréation 1:1) ;
- `lora_expand_tiled`/`lora_shrink_scalar` sont portables ; `lora_expand_mm` et
  les variantes `_wmma` sont à recréer ;
- `gram_accum` et `dequant_w4` hors WMMA restent portables.

Cette liste est celle des rows dont la **description** nomme WMMA ou un ordre
de fragments WMMA ; elle n'est pas une preuve par noyau. Pour toute row
manquante, vérifier le fichier `libr4d/r4d_*.hip` avant de conclure « portable »
ou « à recréer ».

### 10.4 Variantes absentes d'ops déjà présents

Le tableau §10.2 ne voit que les **ops** ; un op présent peut n'avoir, en
RDNA3, que certaines variantes. Les écarts qui comptent pour une cible :

| op | variante RDNA4 | état RDNA3 | usage |
|---|---|---|---|
| `gemm_nt_q` | `gemm_iq8r_nt_m16`, `gemm_iq8r_tiled` (**i8a8h groupe 256**) | **livré (W27)** : `gemm_iqr_nt_m16_w256`, `gemm_iqr_tiled_w256` | **IQ8R** (MiniCPM5-2B) |
| `gemm_nt_q_gated` | `gemm_iqr_gated` à groupe 256 | absent (non déclaré par `llama_dense_fp8` pour un poids i8) | IQ8R/MoE gated |
| `gemm_nt_q` | `gemm_i8a8_nt_m16`, `gemm_i8a8_tiled` (int8 **non rotated**, groupe 256) | absent | conteneur i8a8 non-IQR |
| `rmsnorm_had_quant_i8g` | `rmsnorm_had_quant_iq8r` | **livré (W27)** : row `rmsnorm_had_quant_i8g_256_bf16` | IQ8R |
| `gated_had_quant_i8g` | `gated_had_quant_iq8r` | **livré (W27)** : row `gated_had_quant_i8g_256_bf16` | IQ8R gated |
| `had_quant_act_i8g` | `had_quant_act_iq8r` | **livré (W27)** : row `had_quant_act_i8g_256_bf16` | IQ8R |
| `mtp_enter` | `mtp_enter_bf16` | absent (`libr3d` : e4m3 seulement) | MTP bf16 |
| `moe_gemm_q` | `moe_gemm_i8a8`, `moe_gemm_w4a8i8` | absent (RDNA3 : IQR/`w4nl`/e4m3) | autres conteneurs MoE |
| `attn_paged` | `attn_decode_h64/h512`, gqa1/2/4/8/16 | absent (RDNA3 : h128/h256 ciblés) | autres modèles |
| `attn_dense` | h64/h128/h256/h384/h512 et gqa2/4/8 | partiel (hd72 vision, cible) | autres modèles |
| `logits_gemm` | `logits_gemm_i8`, `logits_gemm_iqr` | vérifier les rows `logits_*` RDNA3 | têtes non ciblées |
| `gemm_nt_bias` | `gemm_bf16_nt_bias_tiled`, `gemm_f32_nt_bias` | cible couverte (vision), variantes absentes | autres modèles |
| `qsa_select` / `kv_len_trim` | `qsa_select_i32`, `kv_len_trim` | équivalents RDNA3 à vérifier par le schéma | cible QSA |

Le cas **IQ8R** était prouvé par l'artefact MiniCPM5-2B : `radiance` s'arrêtait
sur `llama_dense_fp8: rad_arch_declare -> invalid argument`, la trace
`rad_block_attn_fp8.h:110` → `nq_h.declare` montrant que le producteur reçoit
**groupe 256** alors que `rmsnorm_had_quant_i8g` n'avait que 128 ; le GEMM
`i8a8h` groupe 256 manquait aussi. Ce n'était pas un défaut du fichier.

**W27 (2026-11-17) a fermé ce cas pour le tronc à un rang.** Le layout accepte
l'encodage IQ8R (`affine`, 3 plans, `fwht128`, codes i8 tiled, scale u8 tiled
`[1,256]`, scale.1 bf16 tiled `[1,0]`) et range l'échelle comme libr4d : un
bf16 par (row,256), produit arrondi une fois du niveau u8 et du niveau bf16 de
la ligne. Les noyaux m16/résident/WMMA sont templatés sur le groupe : deux
blocs de 128 s'accumulent dans **un seul** `sub` i32 exact, puis un seul fold
`(sub × a_scale) × w_scale`, l'ordre de `libref` (`ref_gemm.h`). Split-K est
refusé à 256. Les producteurs font deux rotations 128 indépendantes par groupe
et un seul amax/une seule échelle sur 256 (le facteur 1/128 reste sur le poids).
Les deux producteurs fusionnés restent sans oracle libref (« Not covered »
dans kbench) ; `act_i8g_group256_device_test` les compare à une référence hôte
indépendante.

Résultat : le 2B déclare **523 ops / 423 poids** et sert à `NS=0` ; kbench
ciblé **110/110** (pire rel_l2 4,018e-05 ; GEMM **bit-exact**), selftest
**169/0/15**, kbench Flash-Next **968/968** et 27B **683/683** dans les deux
modes. Le service utilise **KV bf16** : les rows `attn_paged`/`kv_store` de cet
arch ne prennent pas i8 à `head_dim=128`. Restent pour un usage complet les
variantes `ar_*`/non fusionnées (tp > 1), le fold gated 256 et DSpark.



### 10.5 Ordre de reprise proposé

1. **P0 — IQ8R groupe 256** (MiniCPM5-2B) — **fait (W27)** pour le tronc à un
   rang : producteurs groupe 256, `gemm_nt_q` `i8a8h` groupe 256, 2B servi à
   `NS=0`. Restent `ar_*`/non fusionnés (tp > 1), `gemm_nt_q_gated` 256, et
   `dflash_*` si le drafter est servi.
2. **P0 — tp > 1** : `all_reduce`, `all_gather`, `ar_*` (18 rows) si le
   multi-carte revient dans la cible.
3. **P1 — architectures déjà déclarables** : attention/KV (`qk_norm_rope*`,
   `qkv_norm_rope*`, `attn_paged_qrow*`, `attn_prefix_self`, MLA/DSA),
   normes/quant (`norm_add_norm*`, `rms_mod*`, `gate_*`, `rmsnorm_*`),
   MoE/router inverses (`moe_gather_inv`, `moe_scatter_inv`,
   `router_sigmoid_*`), sampler (`sample_chain`, `sample_merge_topk`,
   `cfg_topk_sample`, `pair_argmax`).
4. **P2 — média/génération** : `elem` (85), `conv` (6), `vision`
   (`pixtral_merge`), `draft`/`dflash` des autres familles — uniquement si
   l'architecture correspondante entre dans la cible.
5. **P3 — RDNA4-only** : recréer ou marquer non supporté (§10.3).

**Rappel de périmètre** : la v1 actuelle (Flash-Next + 27B, texte/vision/MTP)
est complète pour ses chemins exercés ; l'écart de 238 rows est celui de la
couverture multi-architectures, pas une régression de la v1.
