# Handover — RDNA3, phase 6 (W13 livré)

2026-11-15. Branche `rdna3-iqr`, pointe `f6be390`. Ce document remplace
`HANDOVER_RDNA3_PHASE5.md` pour l'état courant ; les phases 2–5 restent valables
pour l'environnement et les conventions.

## 1. État vérifié

| Vérification | Résultat |
|---|---|
| `r3d_selftest` complet | **144 rows, 0 échec, 14 skips** |
| `iqr_device_test` | **6 cas, 27 checks verts** |
| 27B texte | démarre, mêmes sorties greedy |
| Préfill 819 tokens | **301,0 tok/s**, TTFT **2,72 s** |
| Décodage 16 / 33 / 819 | **28,4 / 28,2 / 21,2 tok/s** |
| h256 | 15/15 cas identiques à libref |
| Vision hd72 | 142 rows vertes ; M=16384 exact sur segments uniformes |

Toutes les mesures ci-dessus viennent de **builds propres** (répertoires neufs),
mêmes requêtes que W12, 3 lots médians. Un build incrémental contaminé avait
d'abord annoncé 229 tok/s : voir §5.

## 2. Ce que W13 a livré

1. **Split-K du GEMV étroit** (`cc07477`, `e3ace2a`, `1cdc0a9`, `954f0d7`,
   `e91adb1`) — huit tranches de K pour M=1 et N ≤ 8192, partiels i32,
   réduction déterministe, scratch 256-aligné fourni par le hook de la row
   `gemm_iqr_nt_m64` ; repli sur le chemin non découpé sans scratch aligné.
   Le GEMV `down` passe de **104,0 à 67,3 µs** (428,5 → **662,4 GB/s**,
   49,5 → 76,6 % de 864), le décodage 27B gagne **+5,2 %**.
2. **Tête IQR 16 bits, un lancement** (`9fc05a8`, `5f8f2f8`) — les quatre classes
   dans un grid (`blockIdx.z`), repli f32 factorisé, hi/lo sur 32 lanes au M=1,
   pitch LDS 33 ; test **bit-exact M=1 vs M=2** et classes à comptes inégaux.
3. **`attn_paged` h256** (`aca776f`, `f1ff175` pour la revue) — cache LDS des
   scores de la passe max (5120 f32) et pitch de 272 octets ; la passe deux ne
   stage que V si le cache tient, sinon elle recalcule le dot dans le même ordre.
   Aucune somme réassociée : 15/15 identiques.
4. **`attn_dense` hd72** (`838d1f2`, `f1ff175`) — quatre clés par pas, chaînes
   indépendantes, ordre inchangé ; **−49 % à M=512** (684,7 → 350,7 µs),
   −48 % à M=256, −47 % à M=128/64, −45 % à M=32.
5. **Tuile WMMA du préfill** (`c711c8b`) — construite, bit-identique, mais
   **25–32 % plus lente** que le `v_dot4` résident ; **éteinte** par
   `kIqrWmmaMinM = 1 << 30` (`f6be390`), à reprendre en W14. Le préfill 27B
   tombe de 301 à 231 tok/s quand on la rallume.
6. **Revue W12** (`9558549`) — pitch du gated MT>1, gardes mortes, store V de la
   passe max h256, borne `max_seqlen` de la vision, description `n_embd`.

## 3. Mesures

27B `qwen3.8-27b-iq4r-l-v4`, 64 tokens greedy, 3 lots, `max-num-seqs 1`,
`max-num-batched-tokens 512`, KV i8, builds neufs.

| prompt | préfill W12 | préfill W13 | décodage W12 | décodage W13 | TTFT W12 | TTFT W13 |
|---|---|---|---|---|---|---|
| 16 tok | 103,1 | 103,4 | 27,0 | 28,4 | 150 ms | 155 ms |
| 33 tok | 122,0 | 122,1 | 26,8 | 28,2 | 271 ms | 270 ms |
| 819 tok | 299,7 | 301,0 | 20,7 | 21,2 | 2,73 s | 2,72 s |

Noyaux (propres) :

| noyau | forme | W12 | W13 |
|---|---|---|---|
| GEMV `down` | 5120×17408 M=1 split 8 | 104,0 µs / 428,5 GB/s | 67,3 µs / 662,4 GB/s |
| `attn_dense` hd72 | M=512 | 684,7 µs | 350,7 µs |
| `attn_dense` hd72 | M=256 / 128 / 64 / 32 | 342,5 / 175,6 / 92,1 / 48,4 | 179,6 / 94,3 / 48,7 / 26,7 |
| `gemm_iqr_tiled` résident | 12288×2560 M=512 / 4096 | 2083 / 17 410 µs | idem |
| tuile WMMA (éteinte) | mêmes formes | — | 2801 / 23 110 µs |

`attn_paged` h256 et la tête ne sont pas isolés : seuls les chiffres 27B les
incluent. `rad-kbench -m 27b --bench` ne les atteint pas non plus (les rows
`gemm_iqr_*`, `attn_paged_i8_h256_*` restent dans « Rows nothing ran »), il ne
sert donc que pour la vision et les autres rows du conteneur.

## 4. Commandes

Hôte : `<gpu-user>@<gpu-host>` (W7900 gfx1100, ROCm 10.1, 32 threads). Un seul
GPU : tests et benchmarks sous `flock -w 7200 /tmp/radiance-gpu.lock -c '...'`.

Build **neuf** (obligatoire pour un A/B ; ne pas réutiliser un build existant) :

```bash
#!/bin/bash
set -e
SRC=/mnt/c/dev/git/radiance
DST=<work-root>/work/w13-clean
H=<gpu-user>@<gpu-host>
ssh -o BatchMode=yes $H "rm -rf $DST; mkdir -p $DST"
rsync -a --exclude /.git --exclude /build --exclude /external-docs --exclude /.kilo "$SRC/" "$H:$DST/"
ssh -o BatchMode=yes $H "cd $DST && cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DRAD_GPU_TARGETS=gfx1100 > /tmp/clean-cmake.log 2>&1 && \
  cmake --build build -j 16 > /tmp/clean-build.log 2>&1 && echo BUILD_OK"
```

Tests :

```bash
ssh -o BatchMode=yes <gpu-user>@<gpu-host> \
  "flock -w 7200 /tmp/radiance-gpu.lock -c 'cd <work-root>/work/w13-clean && \
   ./build/tests/iqr_device_test && \
   ./build/bin/r3d_selftest ./build/radiance_home/kernels/libr3d.so'"
```

Benchmark 27B : démarrer le serveur **après** avoir libéré le port, puis vérifier
`readlink /proc/<pid>/exe` avant d'envoyer les requêtes (un ancien serveur sur
8100 fausse tout). Requêtes `/tmp/w12-b15.json`, `-b33`, `-b819` (`max_tokens=64`).

Mesure noyau : `RAD_IQR_BENCH=1 ./build/tests/iqr_device_test` (GEMV, tiled,
split) et `./build/bin/rad-kbench -m <27b.rad> --home <home> --bench --report`.
`kbench` est lent (~210 s) et ne couvre pas les rows IQR/h256 sur ce conteneur.

## 5. Pièges

- **Objets périmés** : `rsync -a` préserve les mtimes. Un rebuild incrémental
  dans un répertoire existant peut ne rien reconstruire (ou un mélange), d'où un
  faux préfill à 229 tok/s. A/B = répertoire de build neuf ; comparer aussi le
  md5 du `libr3d.so` du home.
- **Home vivant** : les `.so` sont écrits directement dans
  `radiance_home/kernels` (`LIBRARY_OUTPUT_DIRECTORY`). Un serveur lancé pendant
  un build charge un mélange. Tuer le serveur avant de reconstruire.
- **Port 8100** : vérifier qu'il est libre avant le lancement, et que le processus
  qui répond est bien `$BIN` (`/proc/<pid>/exe`). Le script `pkill` par chemin ne
  tue que son arbre.
- **WMMA** : éteinte volontairement. Ne pas baisser `kIqrWmmaMinM` sans mesurer ;
  elle est plus lente et le préfill s'effondre.
- **Split-K** : la row `gemm_iqr_nt_m64` devient non concurrente quand elle
  déclare un scratch (N ≤ 8192), y compris aux M > 1 où le split n'est pas pris.
  `num-speculative-tokens 0` neutralise le coût aujourd'hui.
- **VOPD** : `RAD_HIP_VOPD_REPEATABLE=OFF`, donc `-amdgpu-enable-vopd=false` ;
  les builds sont reproductibles sur la carte. Ne pas changer ce cache sans
  re-mesurer.
- **kbench** : ne pas conclure depuis `Rows nothing ran` ; les rows IQR et h256
  ne sont pas atteintes sur ce conteneur.

## 6. À optimiser plus tard (W14+) — noyaux conservés

1. **Tuile WMMA du préfill** : éteinte, 25–32 % plus lente ; reprendre
   l'émission et le staging (décodage par une seule wave, fragments 16 octets)
   avant de la rallumer. Le `v_dot4` résident reste le chemin par défaut.
2. **Gated fusionné** : 1,4 bloc/CU (136 groupes, 96 CU). L'agent W13 n'a livré
   aucune variante ; kernel persistant ou découpage à 256 threads à explorer.
3. **`attn_dense` hd72 à M=16384** : gain mesuré jusqu'à M=512 ; ajouter un cas
   kbench à 16384.
4. **`attn_paged` h256** : mesure isolée ; split du préfill interdit par l'ordre
   exact exigé par les tests.
5. **HC/PLE** : aucune mesure.
6. **E4M3/rotated** : aucun modèle non-IQR sur l'hôte.

## 7. Non ajouté volontairement

`API_DISTANTE.md`, `HANDOVER_RDNA3.md`, `PLAN_IMPLEMENTATION_RDNA3.md`,
`RAPPORT_RDNA3.md` et `external-docs/` restent hors dépôt (déjà untracked avant
W12 ; PDF ISA AMD sous copyright).
