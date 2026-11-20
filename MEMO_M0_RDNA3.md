# Mémo M0 — Go/No-Go du spike IQR (W0)

> **Document historique (2026-11-15).** C'est la pièce de la décision M0 : elle
> reste telle quelle pour la traçabilité. L'état courant est dans
> `HANDOVER_RDNA3_PHASE7.md` (190 rows, W28), les chiffres dans
> `TABLEAU_REEL_IDEAL_RDNA3.md` et `JOURNAL_RDNA3.md`.

Date : 2026-11-15. Périmètre : W0 sur Qwen3.8-27B IQR-V4, gfx1100.
État de l'arbre : branche `rdna3-iqr`, non commité au moment de la rédaction.

## Décision

**GO.** Les critères du plan §6.2 sont remplis pour le chemin dense du 27B :

1. **Parité des formes** : le relayout libr3d est identique octet pour octet à celui de
   libr4d sur les 24 formes communes testées (`tests/iqr_parity_test.cpp`, 285
   vérifications, aller et retour). Les formes refusées par libr3d sont refusées par
   libr4d de la même façon.
2. **Parité des rows** : `gemm_iqr_nt_m64` produit exactement la réponse de la référence
   hôte (`tests/iqr_device_test.cpp`, 4 cas, M=1/5/16, u4/u5, `kRowF32`/`kBf16`,
   comparaison bf16 exacte). Le décodeur a été vérifié séparément : 0 valeur fausse sur
   65 536 (16×128 par unité, 32 unités).
3. **Performance GEMV** : 26,41 µs sur la forme cible `q|gate` 12 288×2 560 u5,
   soit **744,5 GB/s et η = 86,3 %** de la crête VRAM (864 GB/s), au-delà de la cible
   de 65 % (≤35 µs).
4. **Aucun blocage structurel** : les instructions nécessaires (`v_dot4_i32_iu8`,
   `v_perm`, `v_alignbit`, `v_readlane`, `__umul24`) compilent et s'exécutent sur
   gfx1100 ; `global_load_tr_b128` n'est pas utilisé.

## Mesures

| Forme | M | Temps | Débit poids | η VRAM |
|---|---|---|---|---|
| 12 288 × 2 560 u5 (q\|gate) | 1 | 26,41 µs | 744,5 GB/s | **86,3 %** |
| 12 288 × 2 560 u5 | 16 | 239,1 µs | 82,2 GB/s | 9,5 % |
| 5 120 × 17 408 u4 (down) | 1 | 100,2 µs | 444,8 GB/s | 51,5 % |
| 5 120 × 17 408 u4 | 16 | 710,6 µs | 62,7 GB/s | 7,3 % |

- Médiane de 5 lots de 50–100 lancements, carte à 25–41 °C, sclk non pilotée
  (le modèle n'occupe pas la carte), aucun autre processus GPU.
- Le chemin M=16 utilise `v_dot4` et n'est qu'un échafaudage de W3 ; sa borne est
  l'ALU, pas la VRAM. Il ne fait pas partie du critère M0. W3 devra le porter vers
  16×16×16 IU8 / LDS avant M1.

## Écarts et corrections au plan

1. **Aucune fixture IQR n'existe** dans `data/kernels.rkb` : `data/kbench_rows/
   kbench_rows_kernels_libr4d.txt` cite des rows atteintes sur une autre machine, mais
   le fichier versionné ne contient ni `gemm_iqr` ni `trellis` (vérifié par `strings`).
   La référence libref ne sert pas les codes IQR. Le critère « rejouer les fixtures »
   est donc remplacé par un test à référence hôte indépendante, plus fort pour le
   décodeur et le layout (Go/No-Go accepté sur cette base).
2. **Le 27B n'est pas servable à M1 avec W1–W4 seuls** : `head_dim=256`, GQA 6, et
   `libr3d` ne sert que h128. Il faut W6 (attention h256) ou un repli libref. De plus,
   l'embedding à table L12 et la tête vocabulaire IQR manquent. W6 doit démarrer dès
   maintenant, en parallèle de W3/W4.
3. Les formes table (`w4nl`) du 27B n'existent pas dans ses GEMM ; elles restent
   propres aux experts `ffn_down` du Flash-Next et à W5.
4. Le transpose treillis hôte est scalaire (correct, parallélisé par tuiles). Il n'a pas
   été mesuré sur un chargement complet ; à vectoriser en W1 si le chargement du 27B
   dépasse quelques minutes.

## Conditions du GO

- W1/W2 : étendre les formes (table, grid, `kE4m3`, `tcq_rb`) et refuser proprement ce
  qui n'est pas encore lu.
- W3 : porter M=16 vers la matrice 16×16×16 IU8 ou une tuile LDS, puis viser la cible
  du rapport ; mesurer aussi le down (N=5 120, 445 GB/s) et corriger par split-K si
  nécessaire.
- W4 : m64 et préfill.
- W6 en parallèle dès maintenant (attention h256/GQA/KV i8), puis embed et tête
  vocabulaire avant M1.
- Aucun benchmark concurrent : une seule carte.

## Fichiers livrés

- `libr3d/r3d_iqr.h`, `libr3d/r3d_iqr_layout.cpp`, `libr3d/r3d_iqr_dec.h`,
  `libr3d/r3d_gemm_iqr_m16.hip`
- `libr3d/r3d_rows.cpp` (row `gemm_iqr_nt_m64`), `libr3d/r3d_plugin.h`,
  `libr3d/CMakeLists.txt`
- `tests/iqr_parity_test.cpp`, `tests/iqr_device_test.cpp`, `tests/CMakeLists.txt`
- `docs/adr/ADR-0001..0006`, `JOURNAL_RDNA3.md`, `SUIVI_LOTS_RDNA3.md`, ce mémo.
