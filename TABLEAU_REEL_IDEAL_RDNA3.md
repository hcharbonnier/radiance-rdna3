# Tableau réel / idéal — RDNA3

Mesures sur la Radeon PRO W7900 (gfx1100), 48 GiB, ROCm 10.1, carte au repos sauf
mention. Médiane de 5 lots ; chaque ligne cite la configuration et l'état de
l'arbre. Les cibles « idéal » viennent de `RAPPORT_RDNA3.md` §7.

## W0 — GEMM IQR dense (commit `c9f7d51`, 2026-11-15)

Forme `q|gate` 12 288 × 2 560 u5, `kRowF32`, arbre `rdna3-iqr`, 50–100 lancements
par lot, 25–41 °C, pas d'autre processus GPU.

| M | Réel | Idéal | η | Note |
|---|---|---|---|---|
| 1 (GEMV) | 26,41 µs, 744,5 GB/s | 22,8 µs, 864 GB/s | **86,3 %** | cible ≥65 % tenue |
| 16 | 239,1 µs, 82,2 GB/s, 4,2 TFLOP/s | borne WMMA ≈ 0,43 ms pour M=4096 | — | échafaudage `v_dot4`, W3 doit le porter |

Forme `down` 5 120 × 17 408 u4, `kRowF32` :

| M | Réel | Idéal | η | Note |
|---|---|---|---|---|
| 1 | 100,2 µs, 444,8 GB/s | 864 GB/s | 51,5 % | 320 tuiles seulement ; split-K en W3/W4 |
| 16 | 710,6 µs, 62,7 GB/s | — | — | idem |

## W12 — fusion et préfill (commit `d8418c1`, 2026-11-15)

27B `qwen3.8-27b-iq4r-l-v4`, 64 tokens greedy, 3 lots, serveur `max-num-seqs 1`,
`max-num-batched-tokens 512`, KV i8. Base = phase 4 (`57e7bcf`) ; W12 = même carte,
mêmes requêtes, mêmes scripts.

| prompt | base préfill | W12 préfill | base décodage | W12 décodage | base TTFT | W12 TTFT |
|---|---|---|---|---|---|---|
| 16 tok | 71,6 tok/s | 75,9 tok/s | 27,3 tok/s | 27,2 tok/s | 224 ms | 211 ms |
| 33 tok | 69,6 tok/s | 123,5 tok/s | 27,0 tok/s | 27,0 tok/s | 474 ms | 267 ms |
| 819 tok | 79,9 tok/s | **303,2 tok/s** | 20,4 tok/s | **20,8 tok/s** | 10,25 s | **2,70 s** |

Attribution par A/B sur la même carte (décodage 16 tok / préfill 819 tok), chaque
variante ne retirant qu'un commit :

| variante | décodage | préfill |
|---|---|---|
| base phase 4 | 27,3 | 79,9 |
| W12 complet (tête avant pitch) | 26,3 | 306 |
| sans gated fusionné | 25,9 | 303 |
| sans tête 16 bits | 27,0 | 300 |
| W12 final (pitch 33) | 27,2 | 303 |

Lecture : le **GEMM IQR résident** porte le préfill (+223 tok/s, −74 % de TTFT à
819) ; le **gated fusionné** gagne ~1,5 % en décodage ; la **tête 16 bits** coûtait
3 % à cause d'un conflit de bancs LDS (stride 32 dwords), supprimé par un pitch de
33 dwords — elle est désormais à parité, avec l'accumulation f32.

## W13 — split-K, tête, h256, vision (commit `f6be390`, 2026-11-15)

27B, **builds propres** (répertoires neufs ; un build incrémental après rsync peut
garder des objets périmés et fausser l'A/B), 64 tokens greedy, 3 lots, mêmes
requêtes que W12. W12 = `9558549` (les chiffres W12 de la section précédente ont
été re-mesurés ici sur build propre ; écarts < 1,5 %).

| prompt | préfill W12 | préfill W13 | décodage W12 | décodage W13 | TTFT W12 | TTFT W13 |
|---|---|---|---|---|---|---|
| 16 tok | 103,1 | 103,4 | 27,0 | **28,4** | 150 ms | 155 ms |
| 33 tok | 122,0 | 122,1 | 26,8 | **28,2** | 271 ms | 270 ms |
| 819 tok | 299,7 | 301,0 | 20,7 | **21,2** | 2,73 s | 2,72 s |

Noyaux (mesures propres) :

| noyau | forme | W12 | W13 |
|---|---|---|---|
| GEMV `down` | 5120×17408, M=1, split-K 8 | 104,0 µs, 428,5 GB/s | **67,3 µs, 662,4 GB/s** (49,5 → 76,6 % de 864) |
| `attn_dense` hd72 | M=512 | 684,7 µs | **350,7 µs (−49 %)** |
| `attn_dense` hd72 | M=256 / 128 / 64 / 32 | 342,5 / 175,6 / 92,1 / 48,4 µs | 179,6 / 94,3 / 48,7 / 26,7 µs |
| `gemm_iqr_tiled` résident | 12288×2560, M=512 / 4096 | 2083 / 17 410 µs (15,5 / 14,8 TFLOP/s) | idem |
| tuile WMMA (construite, éteinte) | mêmes formes | — | 2801 / 23 110 µs (11,5 / 11,2 TFLOP/s) |

Lecture : le **split-K** porte le décodage (+5,2 % à 16 et 33 tokens, +2,4 % à
819) et 1,55× sur le GEMV `down` ; la **vision** gagne 45–49 % de M=32 à M=512
(les petites tailles restent au plancher de lancement) ; la **tête** (un lancement,
pitch 33, bit-exact M=1/M=2) et **h256** (cache de scores LDS, pitch 272) sont
validés et neutres en préfill ; la **tuile WMMA** est 25–32 % plus lente que le
`v_dot4` et le préfill 27B tombe de 301 à 231 tok/s avec elle — construite,
testée, **désactivée** (`kIqrWmmaMinM` au-delà de tout M), à reprendre en W14.

## Flash-Next (W16, 2026-11-16)

Conteneur `qwen3.8-next-flash-iqr-v4` (121 GB), `placement: expert_tiered`,
`host-pool-mib: 43000`, KV i8, `num-speculative-tokens: 0`,
`max-model-len: 8192`, `max-num-batched-tokens: 2048`. Serveur réel, 1 séquence,
greedy, 64 tokens générés :

| prompt | préfill | décodage | TTFT |
|---|---|---|---|
| 30 tok | 20,6 tok/s | 4,6 tok/s | 1,46 s |
| 2290 tok | **42,1 tok/s** | **4,0 tok/s** | 54,3 s |
| 4570 tok | **41,5 tok/s** | **4,0 tok/s** | 110,2 s |
| 4 × 30 tok concurrents | 17,8 tok/s (par requête) | 4,0–4,5 tok/s | 1,68–3,38 s |

Lecture : premier **bring-up fonctionnel** du 121 GB, tous les ops résolus en
device (0 bande device manquante sur 5 074 questions). Le décodage à 4 tok/s
reflète le tiering des experts (SSD/host), l'absence de MTP et un premier chemin
sans optimisation ; le préfill tient 42 tok/s jusqu'à 4 570 tokens.
`kbench` sur ce conteneur : après W17, **856/856 cas corrects, 0 échec,
0 divergence** (`--kernels libr3d:libref`, pire rel_l2 1,2e-2). Les deux défauts
trouvés par le premier run sont corrigés : la tête `logits_gemm_bf16` indexait
son échelle par ligne (1×128) comme une grille 128×128 (rel_l2 0,53 → 1,7e-3),
et trois sorties QSA divergeaient de libref. Robustesse : un prompt de
12 001 tokens est refusé proprement (HTTP 400, message explicite) et le serveur
reste vivant ; les redémarrages sont propres.

Contrôle final de non-régression 27B sur la pointe W17 : préfill **303,3 tok/s**
à 819 tokens (médiane de 3), décodage **28,6 tok/s** à 16 tokens et **21,4** à
819, sorties identiques — les rows ajoutées ne changent pas le chemin dense.

**Rejeu 27B (W19, 2026-11-16)** — chargement **15,87 GiB en 2,7 s** (6,2 GB/s,
dont 1,7 s de disque à 9,8 GB/s), modèle prêt en 3,3 s, 17,57 GiB tenus, KV
1,58 GiB à 4 séquences. Configuration : `--kv-cache-dtype i8
--num-speculative-tokens 0` (le drafter h128 fenêtré réclame bf16 ; laisser la
spéculation active casse le déclaratif).

| cas (max-num-seqs 1) | préfill | décodage |
|---|---|---|
| 844 tokens, 16 générés | **318,8 tok/s** | 23,2 tok/s |
| 29 tokens, 64 générés | 113,7 tok/s | **29,4 tok/s** |
| 4 requêtes de 32 tokens | — | sérialisées, 5,41 s au total |

**`max-num-seqs 4` : corrigé en W20.** Le 27B échouait au premier pas parce que
`logits_gemm_vocab_iqr` lisait le paramètre rangé (`M`) comme le nombre de lignes
de `x`, alors que le moteur y laisse le **plafond de bande** (4) et porte
l'étendue du pas sur les tenseurs. La row lit maintenant l'étendue sur `x` (comme
`libr4d/r4d_logits_vocab.hip:408-412`), avec la capacité comme plancher. L'audit
`rad-kbench --band-capacity` (nouveau mode, voir `AGENTS.md`) a trouvé la même
faute dans `router_topk`, `router_topk_scatter` et `router_sigmoid_topk`, corrigés
aussi. Résultat à `SEQS=4` :

| cas (max-num-seqs 4) | préfill | décodage |
|---|---|---|
| 844 tokens, 16 générés | **317,5 tok/s** | 23,1 tok/s |
| 29 tokens, 64 générés | 113,4 tok/s | **29,3 tok/s** |
| 4 requêtes de 32 tokens en parallèle | — | **13,6–17,9 tok/s chacune, 67,3 tok/s agrégés** (2,67 s) |

Soit ×2,3 sur le débit agrégé par rapport aux 4 requêtes sérialisées à `SEQS=1`
(29,4 tok/s). Non-régression : `r3d_selftest` 159/0/15, `iqr_device_test` 67
checks verts, kbench **27B 605/605** et **Flash-Next 856/856** par défaut (pire
rel_l2 3,619e-3 et 1,206e-2, identiques) **et** dans les mêmes chiffres en
`--band-capacity` (0 débordement).

Les timings par row du premier run kbench ne sont **pas exploitables** : la
carte a throttlé (PPT0/PPT1) pendant les 8 minutes de mesure ; HC/PLE, QSA et
le MoE par row sont repris à froid dans la section suivante. Les 78 cas non
couverts par kbench étaient hors v1 à cette date :
MTP 48 (`mtp.draft_head`, `blk.48`), `ngram_hint` sans oracle libref.
Depuis, R23 (tables MoE) et W25 (MTP) les couvrent : 968/968 ; il reste 10 cas
`gemm_nt_q_gated` i8a8h sans référence libref et 8 `ngram_hint`.

## Flash-Next — rows isolées (W17, 2026-11-16, carte froide)

Arbre `<work-root>/work/w15-main` (code `a772d70`), carte au repos (32 °C), lots de 1 ms,
`--kernels libr3d:libref --bench`, deux passes : `--op hc --op ple` (M ≤ 512, puis
`--max-tok 2048`) et `--op qsa`. Horloge moyenne 2 882–2 894 MHz, **aucun throttler** sur les
passes HC/PLE (bord 32 → 59 °C, jonction 67 °C) ; la passe QSA a tourné à 2 789 MHz (PPT0
pendant le warm-up), donc ~3–4 % pessimiste. Tous les cas sont vérifiés contre `libref` :
HC/PLE 110 + 72 cas, 0 échec (pire rel_l2 1,5e-2) ; QSA 147 cas, 0 échec (rel_l2 0).

### HC — `hc_read_e4m3` (n 2 560, hc 4, lowrank 320, matrices E4M3)

6,55 Mo de poids par appel (W_down 320 × 10 240 + W_up 10 240 × 320, E4M3). Plancher à la
bande passante froide mesurée (622 GB/s) : **10,5 µs** ; le commentaire de la row
(`libr4d/r4d_hc_bf16.hip:948`, `:1129`) annonce 14,9 µs pour W_up et 18,7 µs pour le read au
pas de décodage.

| M | `hc_read` (sans inject) | `hc_read` (inject + rotate) | débit effectif |
|---|---|---|---|
| 1 | **1 771 µs** | 1 934 µs | 3,7–3,9 GB/s (η ≈ 0,6 %) |
| 2 | 1 771 | 1 924 | |
| 4 | 1 816 | 1 954 | |
| 8 | 3 392 | 3 516 | |
| 32 | 9 502 | 9 485 | |
| 128 | 27 685 | 27 969 | |
| 512 | 113 746 | 115 405 | |
| 1 024 | 229 909 | 233 276 | |
| 2 048 | **470 160** | 476 922 | |

Lecture : au-delà de M=8 le temps croît de **0,23 ms par ligne** ajoutée, soit la relecture de
la paire de matrices par ligne (13,4 Go à M=2 048) à ~28 GB/s, et non une lecture amortie par
tuile de 128 lignes comme le décrit la row. À 48 `hc_read` par pas (96 matrices), cela fait
**93 ms par token en décodage** (≈40 % du pas de 217–250 ms à 4,0–4,6 tok/s) et **22,9 s d'un
préfill de 2 048 tokens** (47 % des 48,6 s mesurés, plafond 89 tok/s). C'est le premier poste
de Flash-Next, devant le tiering des experts ; l'écart à la borne de la row (~18,7 µs) est
d'un facteur ~95 et demande un profil engine pour être clos (kbench seul ne le tranche pas).

**Profil engine (2026-11-17, préparation W21).** Un serveur Flash-Next
(`--profile-ops`, `NS=0`, `MLP=8192`, `SEQS=1`, `expert_tiered`, pool hôte 43 000,
KV i8) sur une requête greedy de 4 tokens donne : **96 sites `hc_read`
(attn + ffn par couche), 1,93 ms/appel**, `wait` 3–8 µs ; le reste du pas est
dominé par `embed_lookup` (9,5 ms/appel, depuis le pool hôte) et les GEMM
`gdn`/`ssm` (2–5 ms). Le même (op, kernel, géométrie) mesure **1,77 ms** dans
kbench (`M=1`, `n=2560 hc=4 lowrank=320 mix=e4m3`). Le plan de placement donne
les poids `hc_norm` **en VRAM** (`blk.*.{attn,ffn}_hc_norm.weight`, 1,88 Mio × 48
chacun), donc ni le lien ni le tiering ne sont en cause. Conclusion : le coût
est **dans le noyau RDNA3** (relecture par ligne au lieu de l'amortissement par
tuile de 128 lignes que la row décrit) ; l'engine ne fait que l'exécuter.
À 6,55 Mo par appel (chiffre W18) en 1,77 ms, le débit est ~3,7 GB/s, très
loin du plancher de bande (10,5 µs) et de la référence libr4d. La correction
est un travail de noyau (W21) et attend le feu vert.

### HC — `hc_enter`, `hc_write`

| M | `hc_enter` | `hc_write` |
|---|---|---|
| 1 | 13,3 µs | 8,0 µs |
| 8 | 13,4 | 8,1 |
| 64 | 14,6 | 9,0 |
| 512 | 40,0 | 29,0 |
| 2 048 | 155,1 | 97,9 |

Les deux sont au plancher de lancement en décodage (3,07 µs par lancement) : ~4 lancements
pour `hc_enter`, ~2,6 pour `hc_write`. `hc_project` n'a pas de bande dans ce conteneur :
aucune mesure.

### PLE

| M | `ple_gate` | `ple_conv` |
|---|---|---|
| 1 | 23,1 µs | 5,8 µs |
| 32 | 32,1 | 42,8 |
| 128 | 94,8 | 93,8 |
| 512 | 336,3 | 259,4 |
| 1 024 | 685,6 | 530,1 |
| 2 048 | 1 556,9 | 1 044,9 |

`ple_gate` à M=2 048 : 107 GFLOP en 1,56 ms ≈ **69 TFLOP/s** (η ≈ 56 % du pic bf16) ;
`ple_conv` est lié aux activations (160 Mio en 1,04 ms ≈ 160 GB/s). En décodage, M=1 est au
plancher de lancement (7,5 lancements pour `ple_gate`, 1,9 pour `ple_conv`). `ngram_hint`
reste sans oracle.

### QSA

| op | M=1 | M=512 | opérandes (M=512) |
|---|---|---|---|
| `qsa_score` | 119,3 µs | **83,4 ms** (±9,6 %) | 258,5 Mio |
| `qsa_select` | 73,1 µs | 945,7 µs | 131,0 Mio |
| `qsa_block_key` | 11,5 µs | — | 1,75 Kio |
| `qsa_work` | 3,7 µs | 5,0 µs | 281 Kio |
| `qsa_tail_store` | 3,1 µs | 75,2 µs | 1,01 Mio |

`qsa_score` à M=512 : 4,3 GMAC en 83 ms ≈ **103 GFLOP/s** (2 % du pic) ; c'est le poste QSA
à optimiser (12 couches × 4 sous-lots de 512 = ~4 s par préfill de 2 048 tokens, 8 % du temps
mesuré). `qsa_select` reste ALU/table. Les autres rows sont au plancher.

### MoE — mesuré (W23, 2026-11-16)

kbench pose désormais les tables du conteneur : la vue complète des 5 plans
(sans `fwht128`), l'échelle IQR lue comme `scale[ligne] × scale.col[bloc 128]`
et la **table de pointeurs** du moteur (slab + pointeurs hôte/device, conteneur
et replay). Le run passe de 856 à **968/968** cas (0 échec, 0 divergence) avec
W25 ; η = experts lus × octets / temps / 603 GB/s (copie froide 256 MiB),
experts lus = borne haute `min(n_expert, M·top_k)`.

| M | `moe_gemm_iqr` µs | η IQR | `moe_gemm_w4nl` µs | η w4nl |
|---:|---:|---:|---:|---:|
| 1 | 230,2 | 0,118 | 292,6 | 0,096 |
| 4 | 382,8 | 0,283 | 499,2 | 0,225 |
| 16 | 696,0 | 0,625 | 876,7 | **0,512** |
| 32 | 1 229,4 | **0,705** | 1 801,6 | 0,498 |
| 64 | 2 311,2 | 0,601 | 2 929,7 | 0,491 |
| 512 | 3 989,3 | 0,350 | 8 757,0 | 0,164 |

`gemm_iqr` dense : N=16384/K=2560 M=1 → 57,19 µs, 460 GB/s (η 0,763) ; M=16 →
70,61 µs (η 0,159). Lecture : à M=1 le MoE n'est pas borné par la bande
(10–12 %) mais par le décodage treillis et le parallélisme par expert ; l'IQR
atteint 425 GB/s (70 %) à M=32 puis redevient compute-bound (35 % à M=512) ;
le w4nl plafonne à 309 GB/s (51 %) à M=16.

### η lien (M2)

Borne du rapport §7.3 : 42 Gio copiés une fois à 28 GB/s = 1,61 s pour 4 096 tokens, soit
2 545 tok/s. Flash-Next mesuré : **42 tok/s** de préfill (2 290 tokens en 54,3 s), donc
**η ≈ 1,7 %**, très loin des 80 % du critère. L'attribution par row montre que le plafond
vient de `hc_read` (89 tok/s à lui seul, 3,5 % de la borne), pas du MoE : à M=1 le MoE
tourne à 10–12 % de la bande (borné par le décodage treillis) et à 70,5 % (IQR) / 51 %
(w4nl) à M=32 / M=16.

## M6 — démo serveur Flash-Next (2026-11-16)

Serveur `w15-main` (`a772d70`) avec la configuration v1 : `TP=1`, `NS=0` (MTP hors v1),
`MLEN=8192`, `HOSTPOOL=43000`, `KVHOST=1024`, `MNBT=2048`, `WIRE=wht6` (défaut). Démarrage :
poids 41,12 GiB/carte, KV 1,67 GiB (65 584 tokens, 8 × 8192), 20,36 GiB de poids experts dans
le pool hôte. Prompts de prose distincts et non répétitifs (aucun préfixe commun).

### Débit, 1 séquence (64 tokens gloutons)

| prompt | tokens | TTFT | préfill | décodage |
|---|---|---|---|---|
| 1 (MoE expliqué) | 113 | 2,69 s | 41,9 tok/s | 4,58 tok/s |
| 2 (plan de mesure) | 113 | 2,70 s | 41,9 tok/s | 4,56 tok/s |
| 3 (note d'estran) | 71 | 1,83 s | 38,7 tok/s | 4,58 tok/s |

Le TTFT suit exactement le préfill à 38,7–41,9 tok/s (113 tokens → 2,7 s) : aucun
coût fixe visible à ces tailles, contrairement à l'hypothèse de tiering à froid.

### 1, 4 et 6 séquences (48 tokens chacune)

| séquences | mur | par requête | total |
|---|---|---|---|
| 1 | 10,89 s | 4,41 tok/s | 4,41 tok/s |
| 4 | 12,43 s | 3,86–3,93 tok/s | 15,52 tok/s |
| 6 | 13,10 s | 3,66–3,74 tok/s | 22,05 tok/s |

Le débit agrégé suit le nombre de séquences (×3,5 à 4, ×5 à 6) ; le décodage par requête perd
~14 % à 4 et ~16 % à 6.

### Cache de préfixe

| cas | tokens | servis par le cache | TTFT |
|---|---|---|---|
| prompt identique, 1<sup>re</sup> fois | 4 468 | 0 | 108,4 s |
| prompt identique, 2<sup>e</sup> fois | 4 468 | **4 096** | 8,97 s |
| même préfixe + 1 354 tokens | 5 822 | 4 096 | 42,4 s |
| conversation qui grandit, tour 1 / 2 / 3 | 2 455 / 2 858 / 3 260 | 0 / **2 048** / **2 048** | 58,9 / 19,7 / 29,6 s |
| rejeu « agent » (5 tours, sous-agents) | 8 543 | **0** (0 %) | 29,6–81,3 s |

Trois constats :
1. Le cache fonctionne (×12 sur un prompt identique, demi-préfill sur un tour qui grandit) mais
   la reprise est **exacte** : elle ne repart qu'à une frontière de chunk de 2 048 tokens
   (`prefix resume: exact`). Un tour de moins de 2 048 tokens ne peut donc rien réutiliser.
2. Le rejeu multi-agents n'a **rien** réutilisé : `prefix_hit_cuts_total{reason="no_snapshot"}`
   = 4, `linear_checkpoints_written_total` = 1 pour 5 tours, alors que l'attention détenait
   4 268 tokens (`attn_cached_tokens_total = prefix_hit_cut_tokens_total = 4 268`). Les
   snapshots d'état linéaire ne sont pas écrits quand les conversations s'entrelacent.
3. Le tier KV hôte était **désactivé** par le moteur malgré `--prefix-cache-mib 1024` :
   `W kv tiers: no device gather/scatter for group 1 at a 4128-byte fragment, so a finished
   conversation cannot leave VRAM. The tiers are off.` **Résolu en W22a** : les rows u8
   `gather_rows`/`scatter_rows` déplacent les fragments opaques. Sur Flash-Next le tier se
   configure avec **2 groupes paginés** (51,38 Kio par entrée, lots de 318) ; sur le 27B,
   1 groupe (516 Kio, lots de 64). Rejeu 27B de 3 tours avec éviction : `to_host` 471,
   `promoted` 245, `restore_cuts` 0, `failures` 0, 2 048 tokens cachés par tour ; hash de
   vérification actif (`RADIANCE_DEBUG_TIER_VERIFY=1`).

### Contention préfill / décodage (`scripts/fnconc.sh`, 4 décodeurs × 48 tokens, préfill ~3 000 tokens)

| passe | décodeurs seuls | à côté du préfill | coût | préfill |
|---|---|---|---|---|
| warm | 12,64 s | 87,69 s | +593,8 % | 87,68 s = 35 tok/s |
| 1 | 12,64 s | 87,63 s | +593,3 % | 87,62 s = 35 tok/s |
| 2 | 12,72 s | 87,59 s | +588,6 % | 87,58 s = 35 tok/s |

Les décodeurs ne terminent qu'avec le préfill (mur identique) : le préfill perd ~17 %
(35 vs 42 tok/s) en leur laissant du device, mais leur latence est celle du préfill entier.

### Robustesse

| cas | résultat |
|---|---|
| prompt de 12 002 tokens | HTTP 400 en 0,00 s, message utile (« shorten the prompt by at least 3 811 tokens ») |
| `max_tokens: 20000` | **pas de 400** : le serveur plafonne à 8 191 (journal) et génère ; à la déconnexion du client, `finish=abort` et `num_requests_running` retombe à 0 |
| abandon en cours de génération (client part à 3 s) | requête suivante servie en 3,57 s, aucun slot fuit |
| 6 admissions simultanées (`--max-num-seqs 8`) | 6/6 servies, 3,66–3,74 tok/s chacune |
| après tout cela | `/health` 200, `num_requests_running` 0 |

### Contextes > 8 192 (16k)

Même configuration, `MLEN=16384`, `SEQS=4` : KV 1,23 GiB (65 596 tokens, 4 × 16 384),
poids 41,55 GiB, 24 snapshots d'état linéaire (6 par séquence).

| cas | résultat |
|---|---|
| prompt de 13 421 tokens | **accepté**, TTFT 328,5 s, préfill **40,9 tok/s**, 8 tokens générés |
| prompt de 18 756 tokens | HTTP 400 en 0,00 s (« maximum context length is 16384 … ») |

Le pool KV tient 4 × 16k exactement (65 596 ≥ 65 536) ; le débit de préfill ne bouge pas
(40,9 vs 41,9 tok/s à 113 tokens), donc le coût est bien proportionnel aux tokens. 200k
reste hors démonstration : il faudrait ~10 × plus de pool (et le tier KV hôte, cf. ci-dessous).

## Vision (W26, 2026-11-16)

Build `-DRAD_WITH_FFMPEG=ON` (libavformat 62.3.100, avcodec 62.11.100,
swscale 9.1.100), même configuration que ci-dessus. Démarrage :
`images and video through 'vision'`, tour de 27 blocs **856,68 Mio en VRAM**,
staging média 48,27 Mio pour 16 384 patches, `capabilities.image=true`.
Serveur réel, 1 séquence, greedy, `enable_thinking: false` :

| tour | tokens de prompt | dont image | TTFT | réponse |
|---|---|---|---|---|
| texte (~234 tok) | 234 | — | 5,99 s | référence |
| image 512² | 281 | 256 | **8,19 s** | trois carrés rouge/vert/bleu, couleurs et disposition correctes |
| image 1024² | 1 045 | 1 024 | **28,73 s** | dégradé bleu nuit → orangé, exact |

Lecture : le préfill texte tourne à ~39 tok/s ; au même débit, 281 tokens
vaudraient ~7,2 s et 1 045 ~26,8 s, donc la tour (décodage et redimensionnement
compris) coûte **~1,0 s pour 1 024 patches** et **~1,9 s pour 4 096** (estimé
par différence de TTFT, pas profilé par op). `kbench` ciblé vision :
**480/480 cas**, 0 échec, 0 divergence ; `--kernels libr3d:libref
--max-cases 1` reste **856/856** dans les deux modes.
La vidéo est annoncée par le processeur mais n'a pas été testée.

### Robustesse du chemin média (W26, 18 cas)

Même serveur, même build, une seule session `flock` ; 18 cas, tous aboutis,
`/health` à 200 après chacun, **VRAM constante à 45 896 MiB** (aucune croissance
par image, rendue à l'arrêt), 0 panic/abort/OOM/HIP error, texte de contrôle
identique avant/après.

| cas | HTTP | tokens image | TTFT | note |
|---|---|---|---|---|
| 2048² (16 384 patches, un pass) | 200 | 4 098 | 121,5 s | plafond du pass, 48 tokens générés |
| 3000² (9 Mpx) | 200 | 4 098 | 121,9 s | **redimensionnée** à 2048², pas refusée |
| 2 × 512² | 200 | 516 | 15,1 s | les deux images distinguées |
| 2 × 1024² | 200 | 2 052 | 58,5 s | 2 × 4 096 patches |
| texte→image / image→texte | 200 | 258 | ~8,0 s | même prompt, même TTFT |
| PNG alpha (RGBA) / JPEG | 200 | 258 | ~7,9 s | décodés, descriptions correctes |
| base64 invalide, `data:text/plain`, PNG tronqué, part inconnu | 400 | — | 0,0 s | message explicite et champ fautif, aucun travail GPU |
| requête valide après les 4 erreurs | 200 | 258 | 7,8 s | récupération immédiate |

Tokens image : 512² → 256 (+2 marqueurs), 1024² → 1 024, 2048² → 4 096,
2 × 512² → 512, 2 × 1024² → 2 048.

Limites observées, à connaître pour l'exploitation :
- **Redimensionnement silencieux** : au-delà de `image-max-pixels` (4 194 304
  px ici) l'image est réduite à 2048², jamais refusée ; le client ne voit que
  le nombre de tokens image.
- **Watchdog bruyant** : 12 avertissements « nothing has been logged for
  20/40 s » (`core/engine.cpp:1307`) pendant les passes longues, toutes
  abouties ; à traiter comme du bruit pour les passes média.
- **Coût d'un pass complet** : TTFT ~121 s (≈34 tok/s de préfill, ≈135
  patches/s), puis ~4 tok/s de décodage à `max-num-seqs 1` ; ce n'est pas une
  recette interactive.
- **Type MIME non vérifié** : `decode_media_url` reconnaît le format aux
  octets ; une `data:text/plain` portant un PNG valide serait acceptée.
- **`THROTTLE_STATUS` d'`amd-smi`** vaut `THROTTLED` même au repos sur cet
  hôte : suivre la puissance et la fréquence, pas ce champ.
- **Non couvert** : 2 × 2048² (32 768 patches au total) ; le staging média est
  dimensionné pour un pass de 16 384 patches, chaque image étant préparée
  séparément.

## MTP Flash-Next (W25, 2026-11-16)

Rows `mtp_enter_e4m3` (norme groupée du flux large + deux produits E4M3) et
`logit_rerank_iqr`/`_i8` (rescore exact des 8 candidats du draft head),
plus deux corrections de `gemm_nt_q_w2a8_m16` (l'en-tête d'un poids joint en
VRAM n'est plus lu sur l'hôte ; le MT est choisi pour que son chunk divise K,
cas M=1/K=2560). Serveur Flash-Next, même prompt, greedy, `max_tokens 256`,
1 séquence, 8k, KV i8, host pool 43 000, build FFmpeg :

| | NS=0 | NS=3 |
|---|---|---|
| décodage | 4,508 tok/s (255 pas) | **8,213 tok/s** (115 pas) |
| TTFT | 1,92 s | 2,03 s |
| texte | 1 322 car. | **identique** (1 322 car.) |
| acceptation | — | **40,3 %** (66,1 / 60,5 / 37,0 % aux profondeurs 1–3) |

**1,82×** le décodage à texte identique, sans toucher au préfill. Les 14 cas
`mtp.draft_head` et les tables MTP 48 sont couverts par kbench (R23 + W25,
968/968).

### Ce que M6 ne démontre pas

- Le préfixe multi-tour **au-delà de 2 048 tokens** : le tier KV hôte est actif
  depuis R21 (copie et restauration vérifiées) mais le corpus de démo n'a pas
  exercé une session de 200k ; il faudrait ~10 × plus de pool.
- L'image/vision : **levé en W26** (voir § Vision) ; la vidéo et l'audio ne sont pas
  testés.
- Le MTP de Flash-Next : **levé en W25** (voir § MTP) ; la couverture kbench
  des tables MTP 48 est faite en R23 (968/968).

## MiniCPM5-2B IQ8R (W27, 2026-11-17)

Premier conteneur non-IQR servi : 3,04 Gio, 42 couches, `n_embd` 2 048,
`n_ff` 6 144, 16 têtes / 2 KV, tête vocabulaire i8 groupe 128. `NS=0`,
`--tp 1`, `--max-model-len 4096`, `--max-num-seqs 1`, **KV bf16** (les rows
`attn_paged`/`kv_store` de cet arch refusent i8 à `head_dim=128`), build neuf
`w27-merge`.

| | valeur |
|---|---|
| déclaration | **523 ops, 423 poids**, 2 groupes KV |
| résolution IQ8R | `gemm_nt_q` groupe 256 → `gemm_iqr_nt_m16_w256` (M ≤ 16) / `gemm_iqr_tiled_w256` ; producteurs → `*_256_bf16` |
| kbench ciblé (4 ops) | **110/110**, pire rel_l2 4,018e-05, GEMM bit-exact |
| décodage | **40,2 tok/s** (60 tokens) ; 69,0 tok/s (16 tokens) |
| préfill | 184–367 tok/s (prompts de 5–10 tokens) |
| texte | cohérent (capitales, listes) |

Les deux producteurs fusionnés (`rmsnorm_had_quant_i8g`,
`gated_had_quant_i8g`) n'ont pas d'oracle `libref` : kbench les laisse « Not
covered », le test `act_i8g_group256_device_test` les compare à une référence
hôte indépendante (67 checks). `had_quant_act_i8g` a un repli AVX.

## `qsa_score` à M=512 (W24, 2026-11-17)

Flash-Next, `--op qsa_score --bench --max-cases 1`, trois passes alternées sur
`w27-merge` (base) et `w27-qsa` (essai), carte à 65 W au repos avant la série :

| passe | M=512 device | horloge | throttle |
|---|---:|---:|---|
| base 1 | 78,98 ms | 2 852 MHz | aucun |
| base 2 | 82,68 ms | 2 842 MHz | aucun |
| base 3 | 83,33 ms | 2 842 MHz | aucun |
| essai ILP 4 blocs | 163,5 / 163,0 / 163,0 ms | 2 450–2 464 MHz | aucun |
| row hôte AVX (même cas) | **9,00 ms** (32 threads) | — | — |

Le débit base est 103 GFLOP/s (2 % du pic) sur 258,5 MiB d'opérandes. Cause :
une warp lit 32 pages de blocs **distinctes** au même offset (un thread = un
bloc), 2 octets utiles par secteur de 32 ; la row est bornée par L2. L'essai
ILP (quatre chaînes f32 indépendantes par thread, arithmétique inchangée,
exacte au test) double le nombre de pages par instruction et **perd 2×** : il
est retiré. La suite est une réécriture coalescée/tuilée, pas un déroulage.

## Lecture

- Le GEMV large atteint 86 % ; la forme `down` était à 49,5 %, le split-K la
  porte à 76,6 % (662 GB/s sur 864).
- Le chemin M≥64 reste un `v_dot4` : la matrice 16×16×16 IU8 construite en W13
  est 25–32 % **plus lente** et reste éteinte. La cible « WMMA » du rapport n'est
  pas atteinte par cette forme ; c'est la piste W14 (émission/staging, pas la
  résidence).
- La vision `attn_dense` hd72 gagne 45–49 % de M=32 à M=512 ; la mesure à
  M=16384 reste à faire (kbench s'arrête à 512 sur ce conteneur).
- h256 et la tête sont mesurés de bout en bout (préfill 301, décodage 21,2 à 819),
  pas isolément. HC/PLE et MoE restent non mesurés. Une seule carte, benchmarks
  séquentiels, builds propres.

## À remplir / différé

| Lot | Forme | Réel | Idéal | η |
|---|---|---|---|---|
| W3 | GEMV 27B, split-K | 662,4 GB/s (76,6 %) sur `down` | 864 GB/s | **fait** |
| W4 | `gemm_iqr_tiled` M=512/4096 | 15,5 / 14,8 TFLOP/s (résident) ; WMMA 11,5 / 11,2 | borne WMMA | WMMA plus lente, éteinte |
| W5 | `moe_gemm_iqr`, `moe_gemm_w4nl` | — | | **non mesurable** : kbench refuse le format des tables du conteneur (110 cas sautés) |
| W6 | `attn_paged` h256 decode/prefill | inclus 27B : préfill 301, décodage 21,2 | | à isoler |
| W8/W9 | `hc_read`, `ple_gate`, `ple_conv` | `hc_read` 1,77 ms à M=1 (η 0,6 %) ; `ple_gate` 69 TFLOP/s à M=2 048 | 10,5 µs (borne bande) | **fait** — `hc_read` est le poste dominant |
| W10 | `attn_dense` hd72 | −49 % à M=512 | | M=16384 non mesuré |
| M1 | 27B tokens/s (préfill, décodage) | 301,0 / 21,2–28,4 (SEQS=1) ; **SEQS=4** : 317,5 tok/s et 67,3 tok/s agrégés sur 4 requêtes | | **dépassé** |
| M2 | η lien Flash-Next, préfill 2 048 | **1,7 %** (42 tok/s) ; plafond `hc_read` 89 tok/s | 80 % de 2 545 tok/s | **non tenu**, cause identifiée |
| M4 | QSA par row | `qsa_score` 83,4 ms à M=512 (103 GFLOP/s) ; `qsa_select` 946 µs | | **fait** |
| M5 | Flash-Next tokens/s | préfill 42,1 / 41,5 tok/s ; décodage 4,0–4,6 | | **fait**, limité par `hc_read` |
| M6 | Démo serveur (prompts, multi-tour, contention, robustesse) | décodeurs ×7,0 à côté d'un préfill ; cache de préfixe ×12 sur prompt identique, 0 % en rejeu agent | | **fait**, deux limites (tier KV off, snapshots interleavés) |

## Optimisations différées (W14+)

Ces noyaux sont livrés et validés, mais pas à leur optimum ; ils restent dans
l'arbre et sont marqués à optimiser, pas à supprimer.

- **`hc_read_e4m3` (priorité 0, Flash-Next)** : 1,77 ms à M=1 pour 6,55 Mo de
  poids (η 0,6 % de la borne bande) et 0,23 ms par ligne au-delà de M=8 : la
  paire de matrices est relue par ligne à ~28 GB/s au lieu d'être amortie par
  tuile. À 48 appels par pas, c'est ~93 ms/token en décodage et 47 % du préfill
  de 2 048 tokens. Le commentaire de la row annonce 18,7 µs au décodage : l'écart
  (×95) doit être profilé côté engine avant de toucher au noyau.
- **Tuile WMMA du préfill** : construite, bit-identique, couverte par les tests,
  **éteinte** car 25–32 % plus lente que le `v_dot4` à toutes les formes mesurées.
  Le problème est l'émission et le staging (decode par une seule wave, fragments),
  pas la résidence : W14 doit la reprendre et la rallumer si elle gagne.
- **`gemm_iqr_gated_i8a8h` fusionné** : 512 threads, 136 blocs pour 96 CU (1,4
  bloc/CU) ; essayer un découpage à 256 threads ou un kernel persistant. Gain
  actuel ~1,5 % en décodage, à ne pas régresser.
- **`attn_dense` hd72 à M=16384** : le gain est mesuré jusqu'à M=512 ; ajouter un
  cas kbench à 16384 avant d'aller plus loin.
- **`attn_paged` h256** : le cache de scores et le pitch 272 sont livrés ; le gain
  isolé n'est pas mesuré et le split du préfill reste interdit par l'ordre exact
  que les tests exigent.
- **`qsa_score` à M=512** : 83,4 ms (103 GFLOP/s, 2 % du pic) pour 258 Mio
  d'opérandes ; ~4 s par préfill de 2 048 tokens. Le plan refuse 2 048/4 096
  lignes (512 Mio/1 Gio), d'où `max-num-batched-tokens: 2048`.
- **PLE** : mesuré à froid ; `ple_gate` tient 69 TFLOP/s à M=2 048 et M=1 est au
  plancher de lancement, rien à optimiser avant le reste.
- **Tier KV hôte (M6)** : **actif** depuis R21 (W22a). Les fragments KV sont
  des octets opaques (3,25 KiB par bloc de 4 tokens pour `kv_qsa_bkey`,
  fragment 4 128 octets) et passent par les rows u8 dédiées, sans codec E4M3.
  Flash-Next : 2 groupes paginés, 51,38 Kio/entrée ; 27B : 1 groupe,
  516 Kio/entrée. Copie et restauration mesurées (27B : `to_host` 471,
  `promoted` 245, 0 échec), hash de vérification actif.
- **Snapshots d'état linéaire en rejeu multi-agents (M6)** : **résolu en W22b**.
  Sous `--prefix-resume exact`, un prompt plus court qu'un intervalle n'avait
  ni point de grille ni snapshot, donc le tour suivant reprenait zéro état
  linéaire (`no_snapshot`). Il ancre maintenant un snapshot au dernier
  frontière de quantum (un quantum plus bas hors grille), dès 256 tokens
  d'ancre. Rejeu 9 tours : tokens servis 24 % → **52,6 %**, préfill recalculé
  −37 %, `no_snapshot` 6 → 2 (premiers tours de sous-agents), checkpoints
  écrits 1 → 7. Compromis : sous le premier intervalle la reprise est hors
  grille, comme `nearest` ; l'accord bit à bit n'y est plus garanti pour un
  tour appendu (un prompt répété reste exact), au-delà `exact` reste strict.
- **E4M3/rotated** : aucun modèle non-IQR sur l'hôte.


