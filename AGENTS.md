# AGENTS.md — Radiance (RDNA3, `rdna3-iqr`)

## Contexte

- Branche de travail : `rdna3-iqr`.
- Tout travail GPU (tests, serveur, benchmark) se fait sous
  `flock -w 7200 /tmp/radiance-gpu.lock -c '...'`, un seul processus à la fois.
- Builds : `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
  -DRAD_GPU_TARGETS=gfx1100 -DRAD_WITH_FFMPEG=ON` puis
  `cmake --build build -j 16`.
- Les `.so` sont écrits **directement** dans `build/radiance_home/kernels` ;
  arrêter le serveur avant de reconstruire. Pour un A/B, utiliser un
  **répertoire de build neuf** (un build incrémental après `rsync -a` peut
  garder des objets périmés).
- Ne pas modifier `core/`, `arch/`, `libr4d/` ni les docs partagés sans besoin.

## Consignes de travail

- **Autonomie** : continuer jusqu'à un jalon (si vraiment important), une décision à prendre ou une
  autorisation à obtenir. Ne pas s'arrêter pour un progrès routinier, et ne pas
  demander confirmation d'une action déjà autorisée.
- **Outils** : pour lire, éditer ou créer des fichiers **locaux**, utiliser les
  outils natifs (`read`, `edit`, `write`, `glob`, `grep`) plutôt que le shell.
  Le shell sert aux builds, tests, git, SSH et au GPU. Les scripts temporaires
  vont dans `%TEMP%\kilo` (Windows) ou `/tmp` (hôte), pas dans le dépôt.
- **Sous-agents et parallélisme** :
  - **Implémentation de code parallélisable** → **Agent Manager**
    (`agent_manager`, mode `worktree`), à privilégier. **Aucune limite** sur le
    nombre d'agents : découper selon les zones de code indépendantes (pas de
    plafond à 3 ou 4).
  - **Tout travail parallélisable qui n'est pas de l'implémentation** (validation,
    campagnes de tests, mesures, exploration, doc) **sur la branche courante** →
    outil `task` uniquement, **jamais** Agent Manager.
  - **Chaque agent Agent Manager relit son propre travail** (`/review worktree`)
    avant de conclure, puis **corrige lui-même** les bugs qu'il a trouvés et
    revérifie. Un agent qui délègue sans relecture n'a pas terminé.
  - **Fusion** : l'agent principal est **responsable de la fusion** du code dans
    la branche courante dès que le travail des sous-agents est terminé, validé
    et corrigé (build + tests + kbench, cf. `## Validation`). Ne pas laisser un
    worktree validé non fusionné.
  - Ne pas créer de session Agent Manager pour du non-code ; une réponse à un
    message inter-agents se fait via `agent_manager` avec le `replyTo` fourni.
  - Un seul travail GPU à la fois sur l'hôte (voir `flock`) : le parallélisme
    porte sur le code et les tests hôte, pas sur des runs carte concurrents.
- **Décisions de contrat** : avant de durcir ou d'assouplir le contrat d'une row,
  lire l'implémentation de référence RDNA4 (`libr4d/`) : c'est elle qui a servi
  à fabriquer les conteneurs. Un contrôle qui n'est pas dans le schéma de la row
  est un bug de la row, pas du moteur.
- **Optimisation noyau** : ne pas démarrer de travail d'optimisation sans feu
  vert explicite de l'utilisateur. Une correction de bug de contrat (comme R24)
  n'est pas de l'optimisation et peut être faite sans attendre.
- **Commit** : commiter **systématiquement** les modifications **relues et
  validées** (relecture faite, build + tests + kbench quand applicable), sans
  attendre une demande ; une modification non relue ou non validée reste dans
  l'arbre de travail. Séparer code et docs (style du dépôt : `libr3d: ...`,
  `docs: ...`), sans `.gitignore` ni documents hors dépôt. Un worktree Agent
  Manager validé est fusionné puis commité dans la foulée. Jamais de push sans
  demande.
- **Mesures** : donner le chiffre, la commande et l'état de la carte ; un chiffre
  throttlé (PPT0/PPT1) est invalide. Comparer à la borne (plancher de lancement,
  débit mémoire) et pas seulement entre versions.
- **Arbres hôte** : `w15-main` est l'état `a772d70`, `w20-r24` l'état
  corrigé (W20), `w26-vision` l'état FFmpeg/vision (R21 + R26), **`final-all`
  l'état de référence courant** (R21 + R22 + R23 + R25 + R26). Tout A/B se
  fait dans un **nouveau** répertoire avec un build neuf.
- **Documents** : chacun a un rôle distinct, ne pas les fusionner ni en écrire
  un sixième qui les répète.
  - `JOURNAL_RDNA3.md` — journal horodaté, une entrée par jour travaillé : les
    mesures **à leur date**, avec commande, horloges et état de l'arbre. Les
    chiffres anciens y restent (c'est la trace), même corrigés depuis.
  - `TABLEAU_REEL_IDEAL_RDNA3.md` — tableau réel/idéal par noyau et par poste
    (M, forme, η, cause de l'écart). C'est la vue chiffrée de référence ; les
    sections historiques y sont suivies de l'état courant.
  - `SUIVI_LOTS_RDNA3.md` — suivi des lots W0→W28 et des jalons M0→M7 (état,
    reste à faire), plus le registre des risques R1→R28.
  - `RAPPORT_RDNA3.md` — rapport d'analyse : faisabilité, ISA, bornes, plan
    d'optimisation, écart RDNA4→RDNA3 par famille (§10). §1–§9 sont
    historiques et préfixés comme tels ; §7.5/§7.6 et §10 sont vivants.
  - `MEMO_M0_RDNA3.md` — pièce de la décision M0 (Go/No-Go du spike IQR) ;
    **document clos**, ne pas mettre à jour, il sert de trace de la décision.
- **Langue** : documentation et rapports en français ; code et commentaires de
  code en anglais.

## Tester un kernel et ses performances : `rad-kbench`

`rad-kbench` (`tools/rad_kbench.cpp`) est l'outil de référence. Il charge les
plugins kernel, résout chaque row comme le moteur, vérifie les sorties contre
`libref`, puis chronomètre chaque **(op, kernel, bibliothèque, géométrie)** :
µs device médiane / min / p90, avec warm-up, cache froid et contrôle des
throttlers. Le rapport markdown contient une ligne par couple
(op, kernel, bibliothèque) — c'est la vue de comparaison.

### Commandes de base

```bash
# Toutes les rows du conteneur, vérification + chronométrage
rad-kbench -m MODEL.rad --home build/radiance_home --bench --report /tmp/kbench.md

# Cibler une op (préfixe, répétable) et plusieurs lots par cas
rad-kbench -m MODEL.rad --home build/radiance_home \
  --op gemm_nt_q --op moe_gemm_q --bench --max-cases 3 --report /tmp/kbench.md

# Deux implémentations ensemble, même protocole
rad-kbench -m MODEL.rad --home build/radiance_home \
  --kernels libr3d:libr3d_naive --op gemm_nt_q --bench --report /tmp/ab.md

# Chronométrage seul, après un run vérifié (pas de référence, plus rapide)
rad-kbench -m MODEL.rad --home build/radiance_home \
  --kernels libr3d:libr3d_naive --op gemm_nt_q --timing-only --report /tmp/ab.md
```

Options utiles :

| option | effet |
|---|---|
| `--op NAME` | ne traite que cette op (préfixe, répétable) |
| `--kernels A:B` | bibliothèques chargées, la première gagne pour la résolution du moteur ; **kbench vérifie et chronomètre les rows de chacune** |
| `--bench` | chronomètre chaque cas vérifié |
| `--timing-only` | chronomètre les rows device sans vérifier (à réserver à un arbre déjà validé) |
| `--bench-flush MIB` | empreinte de rotation (cache froid) ; défaut 256 sur carte |
| `--bench-iters N` / `--bench-reps N` / `--bench-warm-ms N` | taille de lot, nombre de lots, warm-up |
| `--max-cases N` | répétitions par (op, kernel, bibliothèque, géométrie) |
| `--band-capacity` | issue chaque cas avec le paramètre rangé au **plafond de sa bande** et les opérandes à la taille du point : la condition du moteur (un pas plus étroit que sa bande). Un kernel qui lit ce paramètre comme un nombre de lignes passe les cas par défaut et échoue ici |
| `--report FILE` | rapport markdown (Correctness, Speed, Divergences, Not covered) |
| `--strict` | replay : un fixture périmé fait échouer le run |

### Comparer optimisé vs non optimisé

1. **Le baseline doit être un plugin GPU**, pas `libref`. `libref` est
   host/CPU : il apparaît dans une colonne séparée et ne se compare pas aux µs
   device. Pour un vrai A/B, empaqueter le kernel naïf en plugin
   (`rad_add_plugin(... kernels ...)`), exporter le même symbole de row
   (`r3d_rad_<op>`), avec le **même schéma, les mêmes contraintes et une
   priorité inférieure**. Charger `--kernels libr3d:libr3d_naive` : le premier
   gagne la résolution du moteur, mais le rapport chronomètre les rows des deux.
2. **Cibler l'op** avec `--op` pour éviter de payer tout le conteneur.
3. Comparer des lignes de **même géométrie**. La colonne `floor` donne le
   plancher de lancement : un kernel proche du plancher est limité par le
   lancement, pas par son calcul.
4. Il n'y a **pas de colonne « speedup »** : elle se post-traite. Exemple
   (adapte les indices si le format du rapport change ; vérifie l'en-tête de la
   table Speed) :

```python
# usage: python3 speedup.py /tmp/ab.md libr3d libr3d_naive
import sys
path, A, B = sys.argv[1:4]
rows = {}
for line in open(path, encoding='utf-8'):
    if not line.startswith('| `'):
        continue
    c = [x.strip().strip('`') for x in line.strip().strip('|').split('|')]
    if len(c) < 12:
        continue
    op, kernel, lib, geom = c[0], c[1], c[2], c[3]
    med = c[4] if c[4] != '-' else c[5]     # device µs, sinon host µs
    try:
        med = float(med)
    except ValueError:
        continue
    rows.setdefault((op, geom), {})[lib] = med
for (op, geom), d in sorted(rows.items()):
    if A in d and B in d:
        print(f"{op:28s} {geom:60s} {A}={d[A]:.1f}us {B}={d[B]:.1f}us "
              f"speedup={d[B]/d[A]:.2f}x")
```

### Lire et valider un run

- Le rapport a quatre parties utiles : **Correctness** (une ligne par cas),
  **Speed** (op, kernel, bibliothèque, géométrie, médiane device, médiane
  host, min, p90, lots, dispersion, plancher, octets, notes), **Divergences**
  (deux bibliothèques qui décrivent la même sortie différemment : l'une des
  deux a un contrat faux) et **Not covered** (cas non exécutés : forme refusée,
  oracle absent).
- Une note `throttled (PPT…)` invalide la ligne : la carte a baissé ses
  fréquences. Laisser refroidir et refaire un petit lot `--op`.
- Un `FAIL` est une erreur de correction, pas de performance : le corriger
  avant de conclure un A/B (un kernel plus rapide mais faux ne compte pas).
- `--bench` sur un conteneur de 121 GB prend plusieurs minutes et chauffe la
  carte ; préférer `--op` et mesurer sur carte froide.

## Validation

```bash
flock -w 7200 /tmp/radiance-gpu.lock -c 'cd <work-root>/work/<tree> && \
  ./build/tests/sched_test && ./build/tests/engine_e2e_test && ./build/tests/fixture_test && \
  ./build/tests/iqr_device_test && ./build/tests/logits_bf16_device_test && \
  ./build/tests/gather_rows_u8_device_test && ./build/tests/vision_rows_device_test && \
  ./build/tests/mtp_enter_e4m3_device_test && ./build/tests/logit_rerank_iqr_device_test && \
  ./build/tests/act_i8g_group256_device_test && ./build/tests/ple_hc_device_test && \
  ./build/tests/tp2_collectives_device_test && \
  ./build/tests/attn_paged_device_test && ./build/tests/dflash_conv_device_test && \
  ./build/bin/r3d_selftest ./build/radiance_home/kernels/libr3d.so'
```

État de référence de la branche (R21 + R22 + R23 + R25 + R26 + W27 + W28) :
**190 rows, 0 échec, 16 skips** ; `sched_test` **548 782 checks, 0 échec** et
`engine_e2e_test` **909 checks, 0 échec** (3 skips) ; `kbench` Flash-Next
**948/948** (4 946/4 946 en `--bench` dans les deux modes, W24), MiniCPM5-2B
**384/384** et 27B **653/653**, 0 échec, 0 divergence (pires rel_l2 1,206e-02,
1,810e-03, 3,619e-03) ; le nombre de cas suit les rows résolues, pas la
couverture. **Sweep complet rejoué en W38** sur un build neuf (`build-w38`,
`6d9084b`) : 27B **3 181/3 181**, Flash-Next **4 786/4 786**, MiniCPM
**1 802/1 802**, 0 échec, 0 divergence, mêmes pires `rel_l2` ; replay
`--strict` des fixtures (`rad-kbench --kernels libref`, forme du ctest
`kbench_ref`) **1 134/1 134** — `fixture_test` n'a pas d'option `--strict`.
Sur `build-w39` (W38c), `r3d_selftest` passe à **196 rows, 0 échec, 16 skips**
(row `dflash_conv` ajoutée) et `dflash_conv_device_test` **37 checks / 3 cas**
la compare au bit à libref sur huit géométries, les deux moitiés de la
projection de coefficients comprises ; kbench laisse ses 112 cas du 27B en
« skipped » (la déclaration porte la plane entière, le site d'issue la tranche
par `wrow_slice`).
Test de contrat : `iqr_device_test` → `iqr_vocab_head_accepts_a_narrow_step_in_a_wide_band` ;
`gather_rows_u8_device_test` couvre les rows u8 du tier KV (fragment 4 128 octets,
octets opaques, pitches larges) ; `mtp_enter_e4m3_device_test` et
`logit_rerank_iqr_device_test` couvrent le MTP (22 + 36 checks). MTP servi à
`NS=3` : 4,51 → 8,21 tok/s (1,82×), texte identique, acceptation 40,3 %. Vision : build **`-DRAD_WITH_FFMPEG=ON`**
(libavformat 62.3), `capabilities.image=true`, image 512² → 256 tokens,
1024² → 1 024, 2048² → 4 096 (TTFT servi 3,3 s / 16,3 s, mesuré W38 sur
`build-w38`) ; au-delà de `image-max-pixels` l'image est réduite à 2048², pas
refusée. La tour
(856,68 Mio) est en VRAM, le staging média 48,27 Mio.

État **rebasé** (2026-11-17, `main` v2.1.0 `3045979`, branche `rdna3-iqr`) :
**195 rows, 0 échec, 16 skips** ; `sched_test` **548 952**, `engine_e2e_test`
**1 076** (3 skips) ; kbench **27B 3 181/3 181**, **Flash-Next 4 786/4 786**,
MiniCPM **1 802/1 802**, 0 échec (pires rel_l2 3,619e-03 / 1,520e-02 /
3,485e-03) ; préfill 27B **288 tok/s** (spécialisation `GS=1`, `faa89fd`),
**450 tok/s depuis W33** (tuile registre 2×4 sur la forme trellis, TTFT
3 252 tokens 11,2 → **7,2 s**, décodage inchangé, texte identique),
Flash-Next long contexte 3 252 tokens **234,8 tok/s** ; cache de tune livré
dans `build-clean/radiance_home/tune` (bf16 `impl=3`, **5,9×** sur les rows
tiled) ; `attn_paged_device_test` **8 checks/9 cas** (h256 i8 decode/préfill,
gqa6 bs16 et gqa12 bs4, court + split ; le 9e est le mode temps
`RAD_ATTN_BENCH=1`) à **1 bf16 ULP** de libref — écart documenté (polynôme
`exp` à 1 f32 ulp, parenthèses du split-KV) ; le mode temps, **scratch/split-KV
alloué**, donne **4,06 ms** (gqa6) / **2,37 ms** (gqa12) à 32K q=1 (sans
scratch, repli `splits=1` non servi : 29,1 / 33,3 ms) ; le **préfill** est
aussi en split-KV depuis W30 (cible 128 workgroups, repli `splits=1` sans
aréna) : q=17 à 8K **2 376 µs** (gqa6) / **1 795** (gqa12) contre 12 122 /
13 635, et 8 508 / 6 364 à 32K, décodage inchangé ; les multi-rangées
doivent préserver l'ordre par rangée, mais le **verify M=4** n'est pas un
levier (W30 : q=4 ≈ q=1 en gqa6, +23–31 % en gqa12, R36). **W31** : `hc_read`
**2,6–2,9×** au préfill (bit-exact, M=1 inchangé) et `gemm_nt_bias` de la tour
vision **8,7–11×** (0 ulp) ; servi, Flash-Next texte 3 252 tokens 13,56 →
**8,19 s**, image 2 048² 39,2 → **19,1 s**, texte identique au bit. `attn_dense`
n'est **pas** effondré (5,30 TFLOP/s à 16 384 patches × 16 têtes, 53 % de sa
borne VALU ; le « 19 GFLOP/s » de W30 était un comptage faux, R37), et **W32**
porte `hc_read` à **1,23× à M=1, 1,78× à 128, 2,94× à 512** (bit-exact, 7,7–7,9×
cumulé depuis W30) : servi, Flash-Next texte 972 tok 3,13 → **2,56 s**, 3 252
tok 8,19 → **6,32 s**, image 1 024² 4,66 → **4,05 s**, texte identique au bit.
`attn_dense` est **clos sous contrat** (R40 : lane=clé +23 %, kStage 64 ×9,9).
Le run map MoE à tuile étroite est **rejeté** : kbench 1,27× à M=128 mais
**+8,5 % en service** à M=1204 (R39). **W34** : le stager de préfill stagère
aussi les chunks de 256–1 024 rangées — Flash-Next texte 972 tokens
2 497 → **2 120 ms** (1,18×, `moe_gemm_q` 1,30×) — et le prêt de la dalle est
rendu au premier pas non stagé (`set_loan`), sans quoi le décodage perdait
5–7 % (16 441 experts résidents au lieu de 16 802) : après correctif,
**19,25 tok/s** à ctx 3 252 et **30,13** à 92 (≤ 1,2 % de l'arbre sans
stager). R43 (> 2 % de décodage = refus) reste la règle. **W37** : le drapeau
`--tail-anchor off` (défaut `on`) fait courir à un prompt sous le premier
intervalle son dernier chunk entier et supprime l'ancre courte — Flash-Next
972 tokens **2 068 → 1 488 ms (1,39×)**, texte identique, décodage inchangé,
3 252 tokens neutre — au prix de la reprise courte du tour suivant (0
snapshot). À laisser `on` sur un serveur conversationnel. `final-all` est
l'arbre **W26** (ne pas l'utiliser comme baseline de perf sans rebuild) ;
détails dans `RAPPORT_REBASE_V2_1_0.md`.

Recette 27B (dense, IQR) : `--kv-cache-dtype i8`, `--max-num-seqs 4` — le
multi-séquence est servi depuis W20 (4 requêtes ≈ 67 tok/s agrégés, préfill
≈ 318 tok/s). Le drafter sert depuis W38c (`dflash_conv` sur la carte, cache
du drafter en bf16) : `--num-speculative-tokens 3` donne **1,23×** à 135 tokens
de contexte (34,2 contre 27,8 tok/s, acceptation 64 %) mais **1,03×** à 2 047
(15,8 contre 15,4, 70 %) ; texte glouton identique. `NS=2` est refusé par
l'arch (bloc entraîné = 4). Défaut `NS=0` en contexte long, `NS=3` en contexte
court (R48).

Recette MiniCPM5-2B (IQ8R, groupe 256) : `--tp 1 --num-speculative-tokens 0`
en **KV bf16** (les rows `attn_paged`/`kv_store` de cet arch refusent i8 à
`head_dim=128`) ; 523 ops / 423 poids, décodage 40,2 tok/s à 60 tokens. À
`--num-speculative-tokens 3` (drafter DSpark) : **32,2 tok/s contre 20,6**
(1,56×), texte identique au bit, 33 drafts acceptés sur 210 (15,7 %).

## Pièges connus

- Builds périmés après `rsync -a` (mtimes) → répertoire de build neuf pour tout A/B.
- Un `.so` remplacé pendant qu'un serveur tourne : tuer le serveur avant de reconstruire.
- Vérifier que le processus qui répond sur le port est bien le binaire visé
  (`readlink /proc/<pid>/exe`).
- Un build sans `-DRAD_WITH_FFMPEG=ON` refuse toute image
  (`capabilities.image=false`) ; l'hôte exige les paquets `libav*-dev` et
  `pkg-config --modversion libavformat` doit répondre. `<gpu-user>` n'a pas
  `sudo` : l'installation des paquets est une action administrateur.
- **Passes média** : une image au-delà de `image-max-pixels` (4 194 304 px) est
  redimensionnée à 2048², pas refusée ; le watchdog peut journaliser
  « nothing has been logged for 20/40 s » (`core/engine.cpp:1307`) pendant une
  passe longue (121 s à l'arrivée de la vision, 16 s depuis W32), c'est du
  bruit. `THROTTLE_STATUS` d'`amd-smi` vaut
  `THROTTLED` même au repos ici : suivre puissance et fréquence.
- Les cas `ngram_hint` restent sans oracle : leur « Not covered » n'est pas une
  régression. Les cas **MTP 48** sont en revanche exigibles depuis l'ADR-7
  (2026-11-16, vision et MTP de Flash-Next dans la cible) : un MTP non couvert
  est un manque à combler (lot W25), pas une exclusion.
- **IQ8R (groupe 256)** : servi (W27). Le GEMM `i8a8h` groupe 256 lit une
  échelle bf16 par (row, 256) = produit u8×bf16, un seul `sub` i32 par 256 ;
  split-K est refusé à cette largeur. Les producteurs fusionnés
  `rmsnorm_had_quant_i8g` et `gated_had_quant_i8g` n'ont **pas d'oracle libref**
  (kbench les laisse « Not covered ») : leur correctness passe par
  `act_i8g_group256_device_test`. Le 2B se sert en **KV bf16** (l'arch refuse i8
  à `head_dim=128`).
- **Déclarer une op change le graphe de l'arch.** Dès qu'une row existe,
  `rad_op_declared` peut faire prendre la branche fusionnée ; la row doit servir
  la géométrie réellement émise, souvent la ligne entière
  (`[T, heads*head_dim]`) alors que le paramètre `n` nomme la largeur de tête,
  avec une porte en vue `[T, heads, head_dim]` de pas propre (W28 :
  `gate_had_quant_i8g` sur Flash-Next, sinon le moteur s'arrête à l'étape 0).
  Un kbench vert ne prouve pas cela — il ne rend pas le graphe du modèle :
  couvrir la vue par un cas device **et** servir le modèle après le merge.
  Inversement, une op déclarée peut replier des cas élémentaires (27B :
  30 cas `sigmoid`/`mul`/`quant_act_fp8`), d'où un compte kbench qui baisse
  sans échec : comparer les textes servis, pas seulement les compteurs.
- **MTP** : Flash-Next se sert à `NS=3` (1,82× le décodage, texte identique,
  acceptation ~40 %) ; le 27B sert `NS=3` depuis W38c, mais n'y gagne que
  1,23× en contexte court et 1,03× à 2 k (R48) — d'où `NS=0` par défaut au-delà
  de ~1 k tokens. Un poids
  joint `rad.planes` d'une row device peut vivre en VRAM : un shim ne doit pas
  lire son en-tête sur l'hôte (SIGSEGV, corrigé dans `gemm_nt_q_w2a8_m16`),
  et le MT du bras `w2a8` doit avoir un chunk qui divise K (M=1/K=2560 → MT=4).
  `mtp_enter` est déjà parallèle en rangées (grille `(n/8, M)`, un bloc par
  8 canaux et rangée) : sa linéarité en M est le travail d'un GEMV par rangée,
  pas un défaut ; ne pas planifier un batch 2–4× sans profiler le graphe.
  Les tables MTP 48 et le draft head sont couverts par kbench (**948/948** au
  HEAD W28 ; 968 avant que des ops déclarées ne replient des cas
  élémentaires) ; restent sans oracle libref des cas `gemm_nt_q_gated` i8a8h,
  `gate_had_quant_i8g`/`gated_had_quant_i8g` (couverts par les tests device) et
  8 `ngram_hint`.
- **27B en `NS>0`** : servi depuis W38c. Le blocage était double, et les deux
  pièces sont livrées : le drafter h128 prend sa **propre** dtype de cache
  (bf16 sous tronc i8, `ee291d2`), et `dflash_conv` a une **row device**
  (`f63dffd`) — sans elle le groupe de poids du drafter restait épinglé sur le
  *mapped tier* et ses normes élargies échouaient (« widened at load »).
  `NS=3` rend 1,23× à 135 tokens et 1,03× à 2 047 (R48) ; `NS=2` est refusé par
  l'arch.
- Le paramètre rangé d'une row (`M`) est un **plafond de bande**, jamais le
  nombre de lignes du pas : l'étendue est sur les tenseurs. Une row qui compare
  les deux casse le multi-séquence du moteur (bug R24, corrigé en W20) ;
  `--band-capacity` couvre cette classe.
- **`--prefix-resume exact`** : un prompt plus court que le premier intervalle
  ancre un snapshot de queue hors grille (la dernière frontière de quantum, un
  quantum plus bas si elle est hors grille), dès 256 tokens d'ancre. La reprise
  d'un tour appendu court n'est donc plus garantie bit à bit (comme sous
  `nearest`) ; un prompt répété reste exact et, au-delà de l'intervalle,
  `exact` reste strict. Ne pas « corriger » cette ancre sans relire R22
  (`core/sched/scheduler.cpp`, `ChunkGeometry`).
- Ne pas lancer deux benchmarks ou un benchmark pendant qu'un agent teste.
