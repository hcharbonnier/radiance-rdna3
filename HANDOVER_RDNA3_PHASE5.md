# Handover — RDNA3, phase 5 (W12 livré)

2026-11-15. Branche `rdna3-iqr`, pointe `d8418c1`. Ce document remplace
`HANDOVER_RDNA3_PHASE4.md` pour l'état courant ; l'historique des phases 2–4
reste valable pour l'environnement et les conventions.

## 1. État vérifié

| Vérification | Résultat |
|---|---|
| `r3d_selftest` complet | **144 rows, 0 échec, 14 skips** (shims sans opérandes) |
| `iqr_device_test` | **4 cas, 11 checks verts** (GEMM, gated bit-identique, tête, bench conditionnel) |
| 27B texte | démarre, mêmes sorties greedy que la phase 4 |
| Préfill 819 tokens | **303,2 tok/s**, TTFT **2,70 s** (base : 79,9 tok/s, 10,25 s) |
| Décodage 16 / 33 / 819 | **27,2 / 27,0 / 20,8 tok/s** (base : 27,3 / 27,0 / 20,4) |
| h256 | 15/15 cas identiques à libref |
| Vision hd72 | 142 rows vertes, sorties identiques ou dans la tolérance libref |

Aucun modèle MoE ni non-IQR n'est présent sur l'hôte : W5 et la vérification
E4M3 sur un modèle dense restent non mesurées.

## 2. Ce que W12 a livré

1. **`gemm_iqr_tiled` résident** (`d4de0ca`, `2dc7658`) — la tuile de poids
   16×128 est décodée une fois par (tuile, bloc K) et réutilisée pour toutes les
   tranches de 16 tokens ; l'itération M est à l'intérieur de la boucle K.
   Bit-identique au chemin précédent (M=33/160/512 au test device). C'est le
   gain de préfill.
2. **Épilogue gated fusionné** (`119377b`) — `gemm_iqr_gated_i8a8h` fait le
   décodage gate/up, le produit gaté, l'arrondi bf16, le Hadamard 128 et les
   codes int8 en **un launch**, 16 waves par groupe de 128 colonnes, **sans
   scratch `[M, 2n]`**. Codes, échelles et `out_bf16` identiques octet pour octet
   à la paire m16 + `gated_had_quant_i8g`.
3. **Tête IQR 16 bits** (`6daca29`, `c4c7bdd`, `050f94c`, `d8418c1`) —
   activation `q = 128·hi + lo` sur deux passes int8, sommes entières exactes,
   repli f32 unique, **aucun intermédiaire bf16** pour les logits. Le pitch LDS
   de 33 dwords supprime le conflit de bancs qui coûtait 3 % au décodage ; la
   tête est désormais à parité.
4. **`attn_paged` h256** (`e0d3d64`) — page i8 staged en blocs de 16 octets,
   slots visibles seuls, passe max sans V. 15/15 identiques à libref.
5. **`attn_dense` hd72** (`aa5ba99`, `20b364f`, `331b916`) — K/V tuilés en LDS,
   chargements bf16 16 octets, même ordre de dots ; gardes `grid.y`/`grid.z` et
   index de ligne 64 bits.

## 3. Mesures et attribution

Requêtes : `/tmp/req-16.json` (16 tokens), `/tmp/req-long.json` (33),
`/tmp/req-300.json` (819), toutes en `max_tokens=64`, `temperature=0`, 3 lots,
serveur `/tmp/27b.yaml` (`max-num-seqs 1`, `max-num-batched-tokens 512`, KV i8).
Les champs viennent de `timings` de la réponse (`prompt_per_second`,
`predicted_per_second`, `ttft_ms`).

| variante | décodage 16 tok | préfill 819 tok |
|---|---|---|
| base phase 4 (`<work-root>/work/radiance`) | 27,3 | 79,9 |
| W12 complet, tête avant pitch | 26,3 | 306 |
| sans gated fusionné | 25,9 | 303 |
| sans tête 16 bits | 27,0 | 300 |
| **W12 final** | **27,2** | **303** |

Conclusion : le préfill vient du **résident**, le gated fusionné est un **gain**
(~1,5 %), la tête 16 bits est **neutre** après le pitch. Le détail par prompt et
les optimisations différées sont dans `TABLEAU_REEL_IDEAL_RDNA3.md`.

## 4. Commandes

Hôte : `<gpu-user>@<gpu-host>` (W7900 gfx1100, 32 threads), modèles
`<work-root>/models`. Un seul GPU : **tout test GPU sous**
`flock -w 7200 /tmp/radiance-gpu.lock -c '...'`, un benchmark à la fois.

Sync + build (depuis WSL, `SRC` = worktree, `DST` = répertoire distant) :

```bash
#!/bin/bash
set -e
SRC=/mnt/c/dev/git/radiance
DST=<work-root>/work/w12-main
H=<gpu-user>@<gpu-host>
ssh -o BatchMode=yes $H "mkdir -p $DST"
rsync -a --delete --exclude /.git --exclude /build --exclude /external-docs --exclude /.kilo "$SRC/" "$H:$DST/"
ssh -o BatchMode=yes $H "cd $DST && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAD_GPU_TARGETS=gfx1100 > /tmp/w12-cmake.log 2>&1 && cmake --build build --target libr3d libref r3d_selftest iqr_device_test -j 16 > /tmp/w12-build.log 2>&1 && echo BUILD_OK"
```

Pour servir le 27B, un `build` neuf doit aussi contenir les plugins
d'architecture : construire la cible par défaut (`cmake --build build -j 16`),
sinon le serveur échoue avec « no architecture plugin claims 'qwen35' ».

Tests :

```bash
ssh -o BatchMode=yes <gpu-user>@<gpu-host> \
  "flock -w 7200 /tmp/radiance-gpu.lock -c 'cd <work-root>/work/w12-main && ./build/tests/iqr_device_test && ./build/bin/r3d_selftest ./build/radiance_home/kernels/libr3d.so'"
```

Bout en bout (une répétition) :

```bash
cd <work-root>/work/w12-main
RADIANCE_HOME=$PWD/build/radiance_home GPU_MAX_ALLOC_PERCENT=100 \
  setsid ./build/bin/radiance --config /tmp/27b.yaml > /tmp/w12-serve.log 2>&1 &
# attendre /v1/models
curl -s -H 'Content-Type: application/json' -d @/tmp/w12-b819.json \
  http://127.0.0.1:8100/v1/completions | python3 -c 'import json,sys; print(json.load(sys.stdin)["timings"])'
pkill -f "$PWD/build/bin/radiance"
```

`/tmp/w12-b15.json`, `-b33`, `-b819` sont les requêtes ci-dessus avec
`max_tokens=64`. Comparer toujours deux arbres avec les mêmes requêtes, le même
script et des lots séquentiels.

## 5. Pièges

- **Bancs LDS** : une tuile `[16][128]` de dwords lue par les 16 lignes au même
  `c4` a un stride de 32 dwords, donc un conflit 16 voies. Pitcher à **33
  dwords** (`sW[q*33 + lane]`). C'est ce qui a annulé le coût de la tête 16 bits.
- **Scratch gated** : la row répond **0** ; le scratch `[M, 2n]` n'existe plus.
  Ne pas le réintroduire sans raison.
- **Référence des tests** : la tête et le gated se comparent à `reference()` /
  à la paire existante, pas à une réimplémentation ad hoc.
- **`M` de la bande** : le gated reçoit `M` = plus grand pas (16), les tenseurs
  portent les lignes du pas courant ; le noyau n'utilise que `qm`.
- **Plugins** : un `build` partiel ne peuple pas `radiance_home/architectures/`.
- **Un seul serveur 8100** : `pkill -f "$BIN"` avant/après, et vérifier
  `ss -ltn` avant de relancer.
- **Benchmarks** : jamais en parallèle d'un test d'un agent ; le verrou `flock`
  sérialise les tests, mais les benchmarks se font après fusion, un arbre à la
  fois.

## 6. À optimiser plus tard (W13+) — noyaux conservés

Rien de ce qui suit n'est à supprimer : ce sont des noyaux validés, livrés, dont
le coût ou le gain n'est pas encore à l'optimum.

1. **Préfill** : `gemm_iqr_tiled` reste un échafaudage `v_dot4` ; porter le cœur
   en IU8/WMMA 16×16×16. Premier levier du préfill.
2. **`down` 5 120 × 17 408** : 320 tuiles / 96 CU, 51,5 % en W0 ; split-K 4–8.
3. **Gated fusionné** : 512 threads, 136 blocs → 1,4 bloc/CU ; essayer deux
   groupes par bloc ou 256 threads (gain actuel ~1,5 %).
4. **Tête 16 bits** : classes lancées séparément ; paralléliser les classes ou
   recouvrir le décodage de la tuile.
5. **`attn_paged` h256** : split du préfill non fait ; gain isolé non mesuré.
6. **`attn_dense` hd72** : mesurer à M=16384 avant d'aller plus loin.
7. **HC/PLE** (W8/W9) : aucune mesure.
8. **E4M3/rotated** : vérifier sur un modèle dense non-IQR (aucun sur l'hôte).

## 7. Non ajouté volontairement

`API_DISTANTE.md`, `HANDOVER_RDNA3.md`, `PLAN_IMPLEMENTATION_RDNA3.md`,
`RAPPORT_RDNA3.md` et `external-docs/` restent hors dépôt (déjà untracked avant
W12 ; PDF ISA AMD sous copyright).
