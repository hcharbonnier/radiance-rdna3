# Handover — phase 2 du portage RDNA3 (gfx1100)

Destinataire : nouvel agent autonome. Auteur : coordinateur. Date : 2026-11-15 15:20Z.
Statut : W0/W6/W8/W9 fusionnés et vérifiés ; W10-lite livré par un sous-agent sur une
branche, **pas encore fusionné** ; la tête vocabulaire IQR est écrite mais se refuse.

Ce document remplace le handover initial pour l'état courant. Il est autosuffisant ;
lis-le en entier, puis `PLAN_IMPLEMENTATION_RDNA3.md`, `RAPPORT_RDNA3.md`,
`API_DISTANTE.md`, `JOURNAL_RDNA3.md`, `SUIVI_LOTS_RDNA3.md`,
`TABLEAU_REEL_IDEAL_RDNA3.md` et `docs/adr/`.

## 1. Où en est le programme

- Dépôt local : `C:\dev\git\radiance`, branche **`rdna3-iqr`**, HEAD `769c1c8`.
  Les commits locaux sont autorisés (sans push) ; **ne pas réécrire l'historique**.
  Les sujets de certains commits sont tronqués à « docs: » / « libr3d: » (artefact
  de quoting) ; c'est cosmétique, ne pas amender.
- Hôte : `ssh <gpu-user>@<gpu-host>`. W7900 gfx1100 48 GiB, ROCm 10.1.
  Répertoire de build de référence : `<work-root>/work/radiance` (synchronisé
  depuis le dépôt local ; `/build` exclu, pas `core/build`).
- Modèles : `<work-root>/models/qwen3.8-27b-iq4r-l-v4.rad` (18,7 Go),
  `qwen3.8-next-flash-iqr-v4.rad` (113 GiB, intact). `/opt` : 277 Go libres.
- **M0 = GO** (mémo `MEMO_M0_RDNA3.md`).

### Fusionné et vérifié sur la carte

| Lot | État | Preuve |
|---|---|---|
| W0 IQR dense | **fait** | GEMV 26,41 µs / 744,5 GB/s / η 86,3 % sur q\|gate 12 288×2 560 u5 ; parité layout libr4d ; sortie GPU bit-exacte vs référence hôte |
| W6 attention h256 + KV i8 | **fait, revu** | 15/15 cas h256 bit-exacts vs libref (gqa12/gqa6, decode/prefill, split-KV) ; `kv_store` i8, `rope_table`, `imrope` |
| W8 HC / W9 PLE | **fait, revu** | 8 cas / 48 vérifications vs libref, `hc_layout_test` 26 checks |
| Selftest complet | **vert** | 140 rows, 0 échec, 13 skips de shims |

Le 27B, après W6, **déclare** et ne butait plus que sur **54 bandes vision** :
`attn_dense` hd72 (gqa 1, causal 0) et `rope` axial hd72 (sections 18 18).

### Livré, non fusionné : W10-lite

Branche **`rdna3-w10-vision-lite`**, commit **`b2f7398`** :
`libr3d/r3d_vision_bf16.hip` (attn_dense hd72, rope axial hd72), rows
`attn_dense_bf16_h72` (prio 50) et `rope_axial_bf16_h72` (prio 80), schéma
`attn_dense` identique à libref, 6 cas selftest, `--vision-only`.

Rapport du sous-agent (non revérifié par le coordinateur) : `--static` rc 0,
142 rows, 77/90 schémas identiques ; `--vision-only` 6/6 verts ; attn_dense
uniforme bit-exact, aléatoire ≤ 3,05e-05 (1 code bf16) ; rope axial 0 code.
Le run complet annoncé : 142 rows, 0 échec.

**Première action : revoir et fusionner cette branche** (voir §3).

## 2. Ce qui reste avant M1

1. Fusionner W10, rebuild, selftest complet (attendu : 142 rows, 0 échec).
2. Relancer le declare du 27B : il ne doit plus rester **aucune** bande device.
3. Charger et servir le 27B sur 8100, génération courte. Point d'attention : la
   tête vocabulaire IQR (`logits_gemm_vocab_iqr`) **refuse** encore au run. Le 27B
   a `output.k4..k7` ; si le moteur l'utilise, il échouera au premier pas. Vérifier
   s'il prend ce chemin ou le `logits_gemm` de repli ; si c'est la tête IQR, la
   corriger (§4) avant de conclure M1.
4. Mesurer (un benchmark à la fois, via `task`) : IQR GEMV/m16 sur l'arbre
   fusionné, attention h256 decode/prefill, `attn_dense` hd72 M=16384, puis
   tokens/s du 27B. Remplir `TABLEAU_REEL_IDEAL_RDNA3.md`.
5. Si le 27B génère : **M1**, mettre à jour mémo/journal/suivi, puis W5 (MoE),
   W7 (QSA), W11 (intégration), W12 (optimisation).

## 3. Procédure de revue et de fusion (à faire en premier)

```bash
# local, WSL
cd /mnt/c/dev/git/radiance
git log --oneline rdna3-iqr..rdna3-w10-vision-lite   # attendu : b2f7398
git diff --stat rdna3-iqr...rdna3-w10-vision-lite
git merge --no-ff rdna3-w10-vision-lite
```

Puis, sur l'hôte :

```bash
rsync -a --delete --exclude /.git --exclude /build --exclude /external-docs \
  --exclude /.kilo /mnt/c/dev/git/radiance/ <gpu-user>@<gpu-host>:<work-root>/work/radiance/
ssh <gpu-user>@<gpu-host> 'cd <work-root>/work/radiance && \
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAD_GPU_TARGETS=gfx1100 && \
  cmake --build build --target libr3d libref r3d_selftest radiance -j 16'
ssh <gpu-user>@<gpu-host> 'cd <work-root>/work/radiance && \
  ./build/bin/r3d_selftest ./build/radiance_home/kernels/libr3d.so'
```

`libref.so` doit être à côté de `libr3d.so` (le build le produit). Attendu :
142 rows, 0 échec. Si un cas h256/PLE/HC casse, comparer au run de référence
`/tmp/selftest-merged2.log` (140 rows vertes).

Ensuite, générer la config 27B et déclarer :

```bash
ssh <gpu-user>@<gpu-host> 'cd <work-root>/work/radiance && \
  ./build/bin/rad-yaml --model <work-root>/models/qwen3.8-27b-iq4r-l-v4.rad \
    --tp 1 --max-num-seqs 1 --max-model-len 200000 --kv-cache-dtype i8 \
    --num-speculative-tokens 0 --max-num-batched-tokens 512 \
    --host 127.0.0.1 --port 8100 > /tmp/27b.yaml'
ssh <gpu-user>@<gpu-host> 'cd <work-root>/work/radiance && \
  RADIANCE_HOME=$PWD/build/radiance_home GPU_MAX_ALLOC_PERCENT=100 timeout 300 \
  ./build/bin/radiance --config /tmp/27b.yaml > /tmp/27b-m3.log 2>&1; \
  echo rc=$?; grep -e "^E " /tmp/27b-m3.log | cut -c1-150 | sort -u | head'
```

Attendu : plus de ligne « device band(s) resolved to no kernel ». Si une bande
manque, elle s'affiche avec ses paramètres ; c'est le prochain lot.

Pour servir : même commande sans `timeout`, en tâche de fond, puis une requête
courte sur `http://127.0.0.1:8100/v1/completions` (voir le README et
`deploy/compose/tp1.radiance.yaml`). Ne pas lancer deux modèles sur la carte.

## 4. Tête vocabulaire IQR : état exact et piste

`libr3d/r3d_iqr_logits.hip` implémente quantisation FWHT/int8 par groupe, un
GEMM IQR par classe via le shim `r3d_rad_gemm_nt_q_iqr_m16` (validé), puis le
scatter des ids. La row `logits_gemm_vocab_iqr` **refuse** au run sauf si
`RAD_IQR_VOCAB` est posé (mode test), car elle diverge de la référence hôte :

- poids constant (codes 0, c=10), A=q réel, AS=~0,0032, K=128, N=256 :
  device **10,56**, référence **462** ; avec A=1 et AS=1, device **1280** =
  référence (donc décodeur, poids et noyau sont bons) ;
- conclusion partielle : le device ne lit pas le même A/AS que la référence, ou
  l'échelle appliquée n'est pas celle lue ; le chemin par le shim n'a pas changé
  le résultat. Piste : instrumenter un noyau qui recopie A/AS tels que le GEMM
  les voit, comparer à `q`/`s` relus du scratch ; vérifier l'offset `s`
  (`align256(M*n_embd)`), le pitch `nkb`, et l'ordre des écritures quantizer/GEMM.
- Le test est `tests/iqr_device_test.cpp`, cas `iqr_vocab_head_matches_its_reference`,
  activé par `RAD_IQR_VOCAB=1`. Il doit devenir vert avant de lever le refus.

Si le 27B utilise bien cette op, c'est le seul bloqueur M1 restant.

## 5. Benchmarks : règle de sérialisation

Une seule carte. Les benchmarks doivent être lancés **via l'outil `task`**, un
seul sous-agent à la fois, et **jamais** pendant qu'un autre agent exécute un
test GPU (`r3d_selftest`, serveur, modèle). Avant de lancer : vérifier
`pgrep -af 'r3d_selftest|bin/radiance'` et `rocm-smi --showuse` ; noter horloges
et températures avant/après. Le benchmark IQR existe :
`RAD_IQR_BENCH=1 ./build/tests/iqr_device_test` (médiane de 5 lots). Pour
l'attention et la vision, ajouter des cas de mesure ou un petit binaire dédié au
sous-agent ; ne pas modifier les kernels pendant la mesure.

## 6. Contrat d'autonomie

Autorisé : lire/écrire le dépôt, créer tests/ADR/journaux, synchroniser vers
`<work-root>/work/radiance`, compiler, exécuter tests et benchmarks (sériels),
télécharger des conteneurs de travail, committer localement sur `rdna3-iqr`
(sans push).

Interdit sans accord : commit/push/PR sur un dépôt distant, réécriture
d'historique, suppression/déplacement d'un modèle (surtout les 113 GiB), de
conteneurs Docker ou de données de `/opt` hors du répertoire de travail, création
de conteneurs/images, arrêt de `open-webui`/`cloudflared`, sudo hors APT,
modification de libr4d, du plugin `qwen4exp_fp8`, des schémas d'ops, du format
du conteneur, ou des réglages réseau/sécurité, changement de périmètre
(MTP, w4nl w4a8, tp>1, RDNA3.5), contournement d'un jalon.

S'arrêter et rapporter si : un jalon échoue sans correction crédible ; un lot
dépasse son budget de 20 % ; une décision interdite est nécessaire ; blocage
> 2 jours sur un inconnu ; les mesures montrent que les cibles du rapport sont
hors d'atteinte.

## 7. Pièges connus (gagne du temps)

- PowerShell mange les `$` et les pipes dans les arguments SSH. Utilise WSL pour
  rsync et des scripts distants simples ; évite `a | b` dans une commande SSH
  (le pipe peut être interprété localement). `ssh hote 'cmd'` puis traitement local
  du fichier est plus sûr.
- `rsync --exclude /build`, pas `--exclude build` (sinon `core/build/radhead.h`
  disparaît et la compilation échoue).
- Le serveur ne prend pas d'options en ligne de commande : générer un YAML avec
  `rad-yaml`. `RADIANCE_HOME` doit pointer sur `build/radiance_home`.
- `pgrep -f motif` se reconnaît lui-même ; utiliser `pgrep -af` et filtrer.
- Les 3 tests média échouent sans FFmpeg : attendu.
- `r3d_selftest` a besoin de `libref.so` à côté du plugin, sinon il saute les
  comparaisons.
- Un seul GPU : pas de benchmark pendant un test d'un autre agent.
- Le relayout IQR hôte est scalaire (parallélisé par tuiles) ; le chargement du
  27B peut être long, à mesurer.

## 8. Carte des fichiers ajoutés

- IQR : `libr3d/r3d_iqr.h`, `r3d_iqr_layout.cpp`, `r3d_iqr_dec.h`,
  `r3d_gemm_iqr_m16.hip`, `r3d_iqr_embed.hip`, `r3d_iqr_logits.hip`.
- Attention : `r3d_attn_paged_h256.hip`, `r3d_vision_bf16.hip` (W10).
- HC/PLE : `r3d_hc_bf16.hip`, `r3d_hc_ple_common.h`, `r3d_ar_hc_write.hip`,
  `r3d_embed_lookup_q.hip`, `r3d_ngram_hint.hip`, `r3d_ple_gate_bf16.hip`,
  `r3d_ple_conv_bf16.hip`.
- Tests : `iqr_parity_test`, `iqr_device_test`, `ple_hc_device_test`,
  `hc_layout_test`, cas h256/vision dans `r3d_selftest.cpp`.
- Docs : `JOURNAL_RDNA3.md`, `SUIVI_LOTS_RDNA3.md`, `MEMO_M0_RDNA3.md`,
  `TABLEAU_REEL_IDEAL_RDNA3.md`, `docs/adr/ADR-0001..0006`.

## 9. Décisions et dettes ouvertes

- ADR-1 duplication IQR (parité octets avec libr4d), ADR-2 tags privés
  IQR `0x60000000`/`0x70000000`, ADR-3 noms/priorités, ADR-4 parité MoE,
  ADR-5 conteneur sans média autorisé, ADR-6 MTP hors périmètre.
- Tag HC E4M3 `r4d.hc.e4m3.*` partagé avec libr4d (décision du 2026-11-15).
- La row `gemm_iqr_tiled` est un lancement m16 par tranches de 16 (re-lecture du
  poids) : correct, pas performant. W4 doit la remplacer par une tuile résidente
  / WMMA.
- Formes table/grid du décodeur dense non implémentées (le 27B n'en a pas ; les
  experts `ffn_down` du Flash-Next les portent, W5).
- MoE sans fixture : référence hôte à produire (ADR-4).
- Vision : `attn_dense` correct-first, ~4,3e9 triplets et ~1,2 To de K/V par
  appel à M=16384 (secondes à minutes par image) ; `rope` axial fait ~19 M pow
  doubles. Optimisations en W12.
