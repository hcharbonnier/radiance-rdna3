# Handover — phase 3 (RDNA3, gfx1100)

Destinataire : nouvel agent autonome. Date : 2026-11-15 16:20Z. Branche
`rdna3-iqr`. Ce document complète `HANDOVER_RDNA3_PHASE2.md` (environnement,
contrat d'autonomie, pièges) ; il donne l'état exact après W10 et le seul
bloqueur de M1.

## 1. État vérifié

- W10-lite fusionné : merge `9ca90f7` puis suivi `efa1361`. Le worktree local
  avait été ramené à l'état pré-W10 par un checkout inattendu ; il a été restauré
  depuis HEAD, puis le suivi a été fusionné la propreté.
- Build propre puis `r3d_selftest` avec `libref.so` : **142 rows, 0 échec,
  13 skips de shims**. Les cas vision (`attn_dense` hd72, `rope` axial) sont
  couverts par les cas manuels du selftest.
- Le 27B **déclare sans aucune bande device manquante**, charge en **3,0 s**
  (21,16 GiB sur la carte), écoute sur 8100 ; `/v1/models` répond.
- `HANDOVER_RDNA3_PHASE2.md` reste valable pour : hôte, répertoires, build,
  limites d'autonomie, benchmarks sériels, pièges PowerShell/SSH.

## 2. Bloqueur M1 : la norme int8 du trunk

La première complétion échoue au premier pas et **arrête le moteur** :

```
E kernel rmsnorm_quant_fp8_bf16 (libr3d, device) refused op 'rmsnorm_quant_fp8'
  at M=5 [band M<=512] geometry {n=5120 eps=1e-06 group=128 dtype=bf16 wadd=1 M=512}:
  invalid argument
E step 0 failed: invalid argument -- the engine cannot continue
```

Cause : le trunk du 27B est int8. `qwen35_fp8` demande
`rmsnorm_had_quant_i8g` si l'op est déclarée, sinon retombe sur
`rmsnorm_quant_fp8` (E4M3). libr3d ne déclare pas la première ; la seconde est
sélectionnée puis refuse un `q` int8. Log complet : `/tmp/27b-serve.log`.

### Tentative déjà faite (à ne pas répéter telle quelle)

Un composite `rmsnorm_had_quant_i8g` (row + schéma + scratch, appelant
`rmsnorm_quant_fp8` puis `had_quant_act_i8g`) a été ajouté, puis **annulé** : le
startup échouait avec « called in the wrong phase » (`RAD_E_STATE`, plan de
buffers), sans trace `RAD_ARCH_TRACE`/`RAD_BUILD_TRACE`. Probablement un
désaccord entre la declare de dimensionnement (probe) et la declare réelle
(présence/absence de l'op, ou un buffer décrit avec un dtype différent).
L'arbre est revenu à l'état vérifié.

### Plan recommandé

Deux routes, dans cet ordre :

1. **Noyau fusionné, sans changer le nombre d'ops du graphe.** Déclarer
   `rmsnorm_had_quant_i8g` avec le schéma de `rmsnorm_quant_fp8`
   (`pRmsQuantFp8`/`oRmsQuantFp8` : x, residual?, w, q, scale, out_bf16?), une
   shape hook qui décrit `q` en **I8** et `scale` en f32, et un noyau unique :
   add résiduel (si `wadd`), norme RMS sur la somme arrondie bf16 (comme la row
   fp8), puis Hadamard 128 et codes int8 row-major, échelle
   `amax/sqrt(128)/127` par (ligne, 128). Réutiliser `r3d_rmsnorm_quant_fp8_kernel`
   pour la norme et `r3d_fwht_wave<128>` + `i8_codes4` de
   `r3d_had_quant_act_i8g.hip`. Un bloc par ligne, une wave par groupe de 128
   après la réduction de norme, valeur de norme en LDS.
2. **Si le plan de buffers refuse encore un op supplémentaire** : ne pas
   déclarer de nouvel op ; laisser l'architecture retomber sur
   `rmsnorm_quant_fp8`, et **dispatcher dans le shim existant sur le dtype de
   `q`** : si `q` est int8, exécuter le composite (norme → bf16 normalisé en
   scratch, puis `had_quant_act_i8g`) avec un **scratch hook ajouté à la row
   fp8** ; sinon, chemin E4M3 actuel. L'architecture choisit déjà ce fallback
   quand l'op fusionnée est absente.

Contraintes de la cible : `n=5120`, `group=128`, `wadd=1`, bande `M<=512`
(le pas réel peut être 5). `q` int8, `scale` f32 [M, n/128], `out_bf16`
optionnel. La norme doit être prise sur la somme **arrondie bf16** pour rester
identique à la chaîne de libref.

Validation : cas device comparés à libref (déclare la référence de l'op), dans
`r3d_selftest` ou un test dédié ; M=1/5/512, avec et sans résidu, n=5120.
Puis relancer le serveur et une complétion courte.

## 3. Après la norme

Ordre probable des prochains blocages :

1. **Tête vocabulaire IQR** : `logits_gemm_vocab_iqr` refuse au run (sauf
   `RAD_IQR_VOCAB=1`). Symptôme mesuré : poids constant, `A=q`, `AS≈0,0032`,
   K=128 → device **10,56** vs référence **462** ; avec A=1/AS=1 → 1280 exact.
   Décodeur, poids et GEMM sont bons ; le device ne lit pas le même A/AS que la
   référence. Instrumenter A/AS tels que le GEMM les voit, vérifier l'offset
   `s` (`align256(M*n_embd)`), le pitch `nkb` et l'ordre quantizer/GEMM.
2. `gemm_iqr_tiled` reste un m16 par tranches (correct, lent) ; le préfill 27B
   sera lent mais fonctionnel. W4/W12.
3. Benchmarks : un seul à la fois, via l'outil `task`, jamais pendant un test
   GPU d'un autre agent.

## 4. Commandes utiles

```bash
# état et synchro
cd /mnt/c/dev/git/radiance && git log --oneline -5
rsync -a --delete --exclude /.git --exclude /build --exclude /external-docs \
  --exclude /.kilo /mnt/c/dev/git/radiance/ <gpu-user>@<gpu-host>:<work-root>/work/radiance/

# build + selftest
ssh <gpu-user>@<gpu-host> 'cd <work-root>/work/radiance && \
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAD_GPU_TARGETS=gfx1100 && \
  cmake --build build --target libr3d libref r3d_selftest radiance -j 16 && \
  ./build/bin/r3d_selftest ./build/radiance_home/kernels/libr3d.so'

# serveur 27B (YAML déjà généré dans /tmp/27b.yaml)
ssh <gpu-user>@<gpu-host> 'cd <work-root>/work/radiance && \
  RADIANCE_HOME=$PWD/build/radiance_home GPU_MAX_ALLOC_PERCENT=100 \
  ./build/bin/radiance --config /tmp/27b.yaml > /tmp/27b-serve.log 2>&1'
# requête : voir /tmp/req27b.sh sur l'hôte (curl /v1/completions)
```

Le serveur **s'arrête** sur une erreur de pas : le relancer après chaque
correctif. `RAD_ARCH_TRACE=1`, `RAD_BUILD_TRACE=1`, `RAD_R3D_TRACE=1` donnent
les refus détaillés. `r3d_selftest` exige `libref.so` à côté du plugin.

## 5. Limites inchangées

Commits locaux sur `rdna3-iqr` autorisés, **pas de push/PR/réécriture**.
Interdit sans accord : supprimer/déplacer un modèle (surtout les 113 GiB), un
conteneur Docker ou des données de `/opt` hors du répertoire de travail ;
créer un conteneur ; arrêter `open-webui`/`cloudflared` ; sudo hors APT ;
modifier libr4d, le plugin qwen4exp, les schémas, le conteneur ou le réseau ;
changer de périmètre (MTP, w4a8, tp>1, RDNA3.5) ; contourner un jalon. S'arrêter
et rapporter aux jalons et aux blocages de fond ; benchmarks sériels
obligatoires (une seule carte).
