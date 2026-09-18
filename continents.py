"""Continents from plate-topology growth, not noise thresholds.

No value-noise (or any noise) field defines land here. The recipe is standard
practice for tectonic planet generation: farthest-point continent seeds,
round-robin flood-fill growth with per-continent area targets, and trapped-sea
absorption. Close published references for the same recipe are Red Blob Games'
planet generation notes (Amit Patel, MIT-licensed text) and the World Orogen
generator's ocean-land assignment (its code is GPL-3.0 and is NOT reused here;
this implementation is written independently for the SphereMesh structure).

The returned mask is a compositional field in [0,1]: 1 inside grown
landmasses, blended margins a few mesh cells wide. Tectonic plates then
transport this crust exactly like before; plates keep containing both
continental and oceanic crust, as on Earth.
"""
from __future__ import annotations
import numpy as np


def _neighbours(indptr, indices, i):
    return indices[indptr[i]:indptr[i + 1]]


def grow_continents(mesh, seed, count, size_variety, budget_fraction,
                    elongation=0.6, margin_smooth=2):
    """Grow `count` landmasses over `mesh` until `budget_fraction` of the
    sphere area is claimed. Returns (cont_mask, info). Deterministic per seed.
    """
    rng = np.random.default_rng(int(seed) + 7919)
    n = mesh.n
    area = np.asarray(mesh.area, dtype=float)
    total = float(area.sum())
    indptr = np.asarray(mesh.adjacency.indptr, dtype=np.int64)
    indices = np.asarray(mesh.adjacency.indices, dtype=np.int64)
    pts = np.asarray(mesh.p, dtype=float)

    count = max(1, int(count))
    budget = float(budget_fraction) * total

    # 1. Farthest-point seeds, top-3 jitter so neighbouring seeds differ.
    mind2 = np.full(n, np.inf)
    seeds = []
    first = int(rng.integers(n))
    seeds.append(first)
    d2 = np.sum((pts - pts[first]) ** 2, axis=1)
    mind2 = np.minimum(mind2, d2)
    for _ in range(1, count):
        order = np.argsort(mind2)
        # Jitter: random pick among the three most isolated cells.
        pick = order[max(0, len(order) - 3):]
        s = int(pick[rng.integers(len(pick))])
        seeds.append(s)
        d2 = np.sum((pts - pts[s]) ** 2, axis=1)
        mind2 = np.minimum(mind2, d2)

    # 2. Per-continent area targets (log-normal spread by size_variety).
    w = np.exp(rng.normal(0.0, 1.0, len(seeds)) * float(size_variety) * 1.25)
    targets = budget * w / w.sum()

    # 3. Trim seeds if seed cells alone exceed the budget (keep >= 1).
    seed_order = sorted(range(len(seeds)), key=lambda i: -area[seeds[i]])
    kept = list(range(len(seeds)))
    while len(kept) > 1 and sum(area[seeds[i]] for i in kept) > budget:
        kept.remove(seed_order.pop(0))
    seeds = [seeds[i] for i in kept]
    targets = budget * (w[kept] / w[kept].sum())
    ncont = len(seeds)

    claimed = np.full(n, -1, dtype=np.int32)
    acc = np.zeros(ncont)
    for c, s in enumerate(seeds):
        claimed[s] = c
        acc[c] += area[s]
    axes = pts[rng.integers(n, size=ncont)]  # elongation directions
    seedpos = pts[np.array(seeds)]

    frontiers = [set() for _ in range(ncont)]
    for c, s in enumerate(seeds):
        for nb in _neighbours(indptr, indices, s):
            if claimed[nb] < 0:
                frontiers[c].add(int(nb))

    def score(c, x):
        same = 0
        ok = True
        for nb in _neighbours(indptr, indices, x):
            o = claimed[nb]
            if o == c:
                same += 1
            elif o >= 0:
                ok = False
                break
        if not ok:
            return None
        v = pts[x] - seedpos[c]
        nv = float(np.linalg.norm(v))
        elong = abs(float(v @ axes[c]) / nv) if nv > 1e-12 else 0.0
        return same + rng.random() * 0.75 + elongation * elong

    # 4. Round-robin growth; one cell per continent per round.
    progress = True
    while progress:
        progress = False
        for c in range(ncont):
            if acc[c] >= targets[c] or not frontiers[c]:
                continue
            best = []
            for x in frontiers[c]:
                s = score(c, x)
                if s is not None:
                    best.append((s, x))
            if not best:
                continue
            best.sort(key=lambda t: -t[0])
            x = best[rng.integers(min(3, len(best)))][1]
            claimed[x] = c
            acc[c] += area[x]
            for f in frontiers:
                f.discard(x)
            for nb in _neighbours(indptr, indices, x):
                nb = int(nb)
                if claimed[nb] < 0:
                    frontiers[c].add(nb)
            progress = True
        if acc.sum() >= budget:
            break

    # 5. Absorb trapped interior seas bordered by a single continent.
    land = float(acc.sum())
    cap = budget * 1.1
    seen = np.zeros(n, dtype=bool)
    for i in range(n):
        if claimed[i] >= 0 or seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        bordering = set()
        while stack:
            y = stack.pop()
            comp.append(y)
            for nb in _neighbours(indptr, indices, y):
                nb = int(nb)
                o = claimed[nb]
                if o >= 0:
                    bordering.add(o)
                elif not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        if len(bordering) == 1:
            comp_area = float(sum(area[y] for y in comp))
            if land + comp_area <= cap:
                c = bordering.pop()
                for y in comp:
                    claimed[y] = c
                acc[c] += comp_area
                land += comp_area

    mask = (claimed >= 0).astype(float)
    cont = mesh.smooth(mask, margin_smooth)
    info = {'claimed_fraction': float(mask @ area / total),
            'continent_areas_km2': [float(a) for a in acc],
            'continent_count': ncont}
    return np.clip(cont, 0, 1), info
