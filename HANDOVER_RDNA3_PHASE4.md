# Handover — phase 4 (RDNA3, gfx1100)

Destinataire : nouvel agent autonome. Date : 2026-11-15 18:50Z. Branche
`rdna3-iqr`. Ce document complète `HANDOVER_RDNA3_PHASE2.md` (environnement,
contrat d'autonomie, pièges) et remplace `HANDOVER_RDNA3_PHASE3.md` pour l'état
courant.

## 1. État vérifié

- HEAD avant ce document : code `1c66232` (`libr3d: int8 norm fusion, i8a8h
  gated fold, IQR head and prefill fixes`), docs `8eaf1be` (journal + suivi).
  Ce handover est le commit suivant.
- Build propre (`gfx1100`) et `r3d_selftest` avec `libref.so` :
  **144 rows, 0 échec, 14 skips de shims**. Les 14 skips sont les shims sans
  schéma libref, dont le nouvel op `rmsnorm_had_quant_i8g`.
- `./build/bin/r3d_selftest … --norm-only` : **7/7** cas identiques au bit
  (M = 1, 5, 512 ; résidu absent, présent, et présent avec `wadd=0`).
- `./build/tests/iqr_device_test` : **3/3** (GEMM m16 + tiled M=33, tête
  vocabulaire, bench optionnel).
- 27B : charge **21,17 GiB en 3,3 s**, écoute sur 8100, `/v1/models` répond,
  **sans variable d'environnement**. Génération validée jusqu'à **819 tokens
  de prompt** :

| prompt | TTFT | préfill | décodage |
|---|---|---|---|
| 15 tok | 0,22 s | 69,1 tok/s | 28,5 tok/s |
| 16 tok | 0,23 s | 70,2 tok/s | 28,4 tok/s |
| 33 tok | 0,48 s | 69,2 tok/s | 27,9 tok/s |
| 83 tok | 1,08 s | ~77 tok/s | 27,3 tok/s |
| 819 tok | 10,3 s | 79,4 tok/s | 21,0 tok/s |

Exemples de sortie : « Paris. The quick », « city of Paris. », « Lisbon. The
capital », « the Cook Islands is ».

- **M1 est atteint** pour le trunk int8 du 27B. W11 est terminé.

## 2. Ce qui a été fait dans cette phase

- **`rmsnorm_had_quant_i8g`** (`libr3d/r3d_rmsnorm_had_quant_i8g.hip`) :
  add résiduel → variance sur la somme **écrite en bf16** → gain → arrondi bf16
  → Hadamard 128 → int8 row-major, échelle `amax/127` par (ligne, 128). Un
  workgroup par ligne, une wave par groupe après la réduction. Le FWHT passe
  par une tranche LDS par wave (`fwht128_lds`), pas par DPP dans la boucle :
  même réseau et même ordre d'opérandes que `r3d_fwht_wave`, donc bit-identique,
  et plus de motif mal compilé sur gfx1102.
- **`gemm_iqr_gated_i8a8h`** (`libr3d/r3d_gemm_iqr_gated.hip`) : gate_up IQR
  via `r3d_rad_gemm_nt_q_iqr_m16` (ou le tiled au-delà de 16 lignes), en
  scratch bf16, puis `gated_had_quant_i8g`. C'est **deux noyaux device**, pas
  un seul : la fusion d'épilogue est le travail W12 (voir §5). Le fold FFN
  débloque le préfill et garde le trunk int8.
- **Convention d'échelle corrigée** : libr3d pliait `1/sqrt(128)` dans les
  quantiseurs tournés. `docs/OPS.md`, `ARCHITECTURES.md`, libref et libr4d
  disent que le 1/128 est porté par le plan de poids. Les formes `i8g` écrivent
  `amax/127`, les formes E4M3 tournées `amax/448` ; groupe nul → échelle 1 ;
  le produit gaté est arrondi bf16 avant la rotation, comme la paire non
  fusionnée. Fichiers : `r3d_had_quant_act_i8g.hip`,
  `r3d_gated_had_quant_i8g.hip`, `r3d_had_quant_act_fp8.hip`,
  `r3d_gated_had_quant_fp8.hip`, `r3d_iqr_logits.hip`, et le nouveau noyau.
- **`b_scale` IQR** : le plan d'une forme `block=1x*` a l'étendue logique
  `N*K`, pas `N*(K/128)`. Le shim m16 puis le shim tiled n'exigent plus que sa
  présence ; le layout alloue ses octets. Un cas M=33 a été ajouté à
  `iqr_device_test`.
- **Préfill GDN** : `rad_gemm_run` (`libr3d/r3d_vocab_bf16.hip`) lisait les
  premiers octets de **tout** poids comme un en-tête `rad.planes` depuis
  l'hôte. Pour `ssm_ab` (bf16 [96, 5120]), le poids est en VRAM : lecture
  invalide, SIGSEGV dans `GdnFP8::step` sur tout prompt > 16 tokens. Le peek
  d'en-tête ne s'applique plus qu'aux poids privés (`dtype != RAD_BF16`).
- **Tête vocabulaire** : le noyau et le GEMM étaient corrects. La référence
  locale du test lisait la queue des échelles au mauvais offset
  (`align256(rows*4)` au lieu de `cf_off`) et appliquait le facteur de ligne
  après l'accumulation. Le test utilise maintenant `reference()`, la référence
  GEMM validée du même fichier ; il est vert. **`RAD_IQR_VOCAB` a été
  supprimé** : la row est servie par défaut. `r3d_logits_gemm_vocab_iqr_scratch`
  réserve toujours le scratch (M·n_embd + échelles + logits bf16).
- **Correctifs de revue** : contrôle `N % 256` dans le shape hook gated,
  `out_bf16` écrit en deux dwords (contrôle d'alignement), test du résidu
  corrigé (slot absent quand `residual=false`, cas `wadd=0` ajouté), description
  de row gated honnête sur les deux launches/scratch.

## 3. Environnement et commandes

Hôte : `ssh <gpu-user>@<gpu-host>`, W7900 gfx1100 48 GiB, ROCm 10.1.
Répertoire : `<work-root>/work/radiance` (synchronisé depuis
`C:\dev\git\radiance` ; exclure `/build`, pas `build`).
Modèle : `<work-root>/models/qwen3.8-27b-iq4r-l-v4.rad` (18,7 Go).
Config existante : `/tmp/27b.yaml` (tp1, max-num-seqs 1, max-model-len 200000,
kv-cache i8, speculative 0, max-num-batched-tokens 512, port 8100).

**Toujours mettre les commandes WSL dans un script bash temporaire** : PowerShell
mange `$`, `|` et les quotes imbriquées. Exemple de script de synchronisation
et build, à écrire sous
`C:\Users\<user>\AppData\Local\Temp\kilo\syncbuild.sh` puis à lancer avec
`wsl.exe bash /mnt/c/Users/<user>/AppData/Local/Temp/kilo/syncbuild.sh` :

```bash
#!/bin/bash
set -e
rsync -a --delete --exclude /.git --exclude /build --exclude /external-docs --exclude /.kilo \
  /mnt/c/dev/git/radiance/ <gpu-user>@<gpu-host>:<work-root>/work/radiance/
ssh -o BatchMode=yes <gpu-user>@<gpu-host> 'cd <work-root>/work/radiance && \
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAD_GPU_TARGETS=gfx1100 > /tmp/cmake.log 2>&1 && \
  cmake --build build --target libr3d libref r3d_selftest radiance -j 32 > /tmp/build.log 2>&1 && echo BUILD_OK || (echo BUILD_FAIL; tail -40 /tmp/build.log)'
```

Tests :

```bash
# selftest complet
ssh <gpu-user>@<gpu-host> 'cd <work-root>/work/radiance && \
  ./build/bin/r3d_selftest ./build/radiance_home/kernels/libr3d.so'

# norme seule (7 cas)
ssh <gpu-user>@<gpu-host> 'cd <work-root>/work/radiance && \
  ./build/bin/r3d_selftest ./build/radiance_home/kernels/libr3d.so --norm-only'

# GEMM IQR + tête (le binaire est dans build/tests, pas build/bin)
ssh <gpu-user>@<gpu-host> 'cd <work-root>/work/radiance && \
  ./build/tests/iqr_device_test'
```

Serveur (le lancer en tâche de fond, puis requête) :

```bash
ssh <gpu-user>@<gpu-host> 'cd <work-root>/work/radiance && \
  (RADIANCE_HOME=<work-root>/work/radiance/build/radiance_home GPU_MAX_ALLOC_PERCENT=100 \
   setsid ./build/bin/radiance --config /tmp/27b.yaml > /tmp/27b-serve.log 2>&1 &) ; sleep 15'
ssh <gpu-user>@<gpu-host> 'curl -s --max-time 300 -H "Content-Type: application/json" \
  -d @/tmp/req-long.json http://127.0.0.1:8100/v1/completions'
ssh <gpu-user>@<gpu-host> 'pkill -f bin/radiance'
```

`/tmp/req27b.json` (8 tokens), `/tmp/req-long.json` (33 tokens),
`/tmp/req-300.json` (819 tokens) existent sur l'hôte. Le serveur **s'arrête sur
une erreur de pas** : relire `/tmp/27b-serve.log` et relancer après correctif.
`RAD_ARCH_TRACE=1`, `RAD_BUILD_TRACE=1`, `RAD_R3D_TRACE=1` donnent les refus
détaillés. `rocgdb -batch -ex run -ex "thread apply all bt" --args …` a été
utilisé pour le SIGSEGV GDN.

## 4. Pièges et points de vigilance

- Le test de la tête **doit** utiliser `reference()` ; ne pas réintroduire un
  décodage manuel de `scale.col` (c'était le faux écart 10,56 vs 462).
- Le plan `b_scale` d'une forme IQR `block=1x*` est privé et large (N×K
  logique) : ne pas le valider par `N*(K/128)`. Le m16 et le tiled sont
  désormais alignés ; tout nouveau shim doit l'être aussi.
- La correction d'échelle touche aussi les formes **E4M3 tournées**
  (`had_quant_act_fp8`, `gated_had_quant_fp8`). Le 27B est int8 et a été
  validé ; aucun modèle non-IQR (2B/4B) n'a été rejoué. À vérifier avant de
  considérer ces rows comme non régressives.
- Le peek `rad.planes` est maintenant conditionné à `dtype != RAD_BF16` : un
  poids bf16 en VRAM n'est plus lu par l'hôte. Si une tête quantisée devenait
  stockée autrement, vérifier qu'elle est toujours mappée/lisible par l'hôte.
- `r3d_selftest` exige `libref.so` à côté de `libr3d.so` (le build le produit).
- `pgrep -f` se reconnaît lui-même : filtrer. Ne pas lancer deux modèles ni un
  benchmark pendant un test GPU.
- Le préfill IQR utilise encore `gemm_iqr_tiled`, un m16 par tranches de 16 qui
  relit le poids ; c'est correct mais lent. W12 doit le remplacer.
- La liaison PCIe est à 16 GT/s x16 au lieu de 32 (warning au démarrage) ; à
  investiguer si les mesures hôte↔carte sont loin des cibles.

## 5. Travail suivant (W12, par impact)

1. **Fusionner l'épilogue gated en un seul noyau** : `gemm_iqr_gated_i8a8h`
   lance aujourd'hui m16/tiled puis `gated_had_quant_i8g` et matérialise
   `[M, 2n]` en scratch. Un noyau unique (gate/up lus ensemble, silu, Hadamard
   128, codes) supprime le scratch et le second launch. C'est le principal
   écart SOTA restant sur le décodage.
2. **`gemm_iqr_tiled` résident** : remplacer le m16 par tranches (re-lecture du
   poids) par une tuile résidente M ou WMMA. Le préfill 27B est à ~79 tok/s ;
   c'est le plafond le plus visible.
3. **Tête IQR** : la row fait un GEMM m16 par classe puis un scatter. La
   remplacer par la matrice 16 bits de la tête pour garder l'accumulation f32
   (le commentaire d'en-tête le prévoyait pour W11 ; c'est maintenant W12).
4. **Vision / attention** : `attn_dense` hd72 est correct-first ; mesurer à
   M=16384 et optimiser (`TABLEAU_REEL_IDEAL_RDNA3.md`).
5. **Mesures** : remplir le tableau réel (IQR GEMV/m16, attention, vision,
   27B decode/prefill). Un benchmark à la fois, via `task`, jamais pendant un
   test GPU. Le décodage 27B à 27,9 tok/s correspond à ~522 GB/s de lecture de
   poids (18,7 Go/token) ; la carte peut viser plus haut.

## 6. Fichiers touchés dans cette phase

- Nouveaux : `libr3d/r3d_rmsnorm_had_quant_i8g.hip`,
  `libr3d/r3d_gemm_iqr_gated.hip`.
- Modifiés : `libr3d/CMakeLists.txt`, `r3d_plugin.h`, `r3d_rows.cpp`,
  `r3d_shapes.cpp`, `r3d_selftest.cpp`, `r3d_gemm_iqr_m16.hip`,
  `r3d_iqr_logits.hip`, `r3d_had_quant_act_i8g.hip`,
  `r3d_gated_had_quant_i8g.hip`, `r3d_had_quant_act_fp8.hip`,
  `r3d_gated_had_quant_fp8.hip`, `r3d_vocab_bf16.hip`,
  `tests/iqr_device_test.cpp`, `JOURNAL_RDNA3.md`, `SUIVI_LOTS_RDNA3.md`.
- Commits : `1c66232` (code), `8eaf1be` (journal/suivi), ce handover.

## 7. Limites inchangées

Commits locaux sur `rdna3-iqr` autorisés, **pas de push/PR/réécriture**.
Interdit sans accord : supprimer/déplacer un modèle (surtout les 113 GiB) ou des
données de `/opt` hors du répertoire de travail ; créer/arrêter des conteneurs ;
arrêter `open-webui`/`cloudflared` ; sudo hors APT ; modifier libr4d, le plugin
`qwen4exp`, les schémas, le conteneur ou le réseau ; changer de périmètre
(MTP, w4a8, tp>1, RDNA3.5) ; contourner un jalon. S'arrêter et rapporter aux
jalons et aux blocages de fond ; benchmarks sériels obligatoires (une carte).
