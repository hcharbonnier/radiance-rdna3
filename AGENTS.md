# AGENTS.md — Radiance (RDNA3, `rdna3-iqr`)

## Contexte

- Branche de travail : `rdna3-iqr`. Hôte GPU : `<gpu-user>@<gpu-host>`
  (W7900 gfx1100, 48 GiB, 32 threads), **une seule carte**.
- Tout travail GPU (tests, serveur, benchmark) se fait sous
  `flock -w 7200 /tmp/radiance-gpu.lock -c '...'`, un seul processus à la fois.
- Builds : `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
  -DRAD_GPU_TARGETS=gfx1100` puis `cmake --build build -j 16`.
- Les `.so` sont écrits **directement** dans `build/radiance_home/kernels` ;
  arrêter le serveur avant de reconstruire. Pour un A/B, utiliser un
  **répertoire de build neuf** (un build incrémental après `rsync -a` peut
  garder des objets périmés).
- Ne pas modifier `core/`, `arch/`, `libr4d/` ni les docs partagés sans besoin.
- Accès SSH, droits, specs de l'hôte et contraintes d'inférence distante :
  voir `API_DISTANTE.md` (référence pour tout travail GPU distant).

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
  ./build/tests/iqr_device_test && ./build/tests/logits_bf16_device_test && \
  ./build/bin/r3d_selftest ./build/radiance_home/kernels/libr3d.so'
```

État de référence de la branche : **159 rows, 0 échec, 15 skips** ;
`kbench` Flash-Next : **856/856, 0 échec, 0 divergence**.

## Pièges connus

- Builds périmés après `rsync -a` (mtimes) → répertoire de build neuf pour tout A/B.
- Un `.so` remplacé pendant qu'un serveur tourne : tuer le serveur avant de reconstruire.
- Vérifier que le processus qui répond sur le port est bien le binaire visé
  (`readlink /proc/<pid>/exe`).
- Les rows MTP 48 et `ngram_hint` sont hors v1 : leurs cas « Not covered » ne
  sont pas des régressions.
- Ne pas lancer deux benchmarks ou un benchmark pendant qu'un agent teste.
