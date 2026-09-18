"""Map-stage fluvial geomorphology: steady-state stream-power erosion.

No noise defines landforms here. Given tectonic uplift and runoff, the
detachment-limited stream-power law

    dz/dt = U - K * Q^m * S          (S = downstream slope, Q = discharge)

at steady state (dz/dt = 0) gives graded longitudinal profiles

    S = (U/K) * Q^-m

which are integrated upstream from coastal outlets in a single
topologically-ordered pass (the converged limit of an iterative implicit
Braun-Willett solve, without the iterations). Valleys carve toward their
graded profile; lowlands aggrade toward it (deposition); large rivers
overdeepen near the coast into estuaries, so coasts become drowned,
dendritic shorelines instead of smooth contours. A light hillslope
diffusion rounds interfluves. Method family: Cordonnier et al. 2016
(tectonic uplift + fluvial erosion), Tzathas et al. 2024 (analytical
stream-power solutions).
"""
from __future__ import annotations
import numpy as np
from numba import njit


@njit(cache=True)
def _graded_profile(order, rec, step, Unorm, Q, grade, Q0, mexp, h, z):
    n = len(rec)
    for k in range(len(order)):
        i = order[k]
        j = rec[i]
        if j == i or j < 0 or j >= n:
            z[i] = h[i]
            continue
        q = Q[i] if Q[i] > 1.0 else 1.0
        s = grade * (0.25 + Unorm[i]) * (Q0 / q) ** mexp
        if s > 1.5:
            s = 1.5
        zj = z[j]
        zi = zj + s * step[i]
        # Anchor coastal outlets at their terrain height (never below 0.5 m).
        if h[j] < 0.0 and zi < 0.5:
            zi = 0.5 if h[i] > 0.5 else h[i]
        z[i] = zi


@njit(cache=True)
def _diffuse(h, land, passes, blend, ny, nx):
    out = h.copy()
    tmp = h.copy()
    for _ in range(passes):
        for y in range(ny):
            yu = y - 1 if y > 0 else 0
            yd = y + 1 if y < ny - 1 else ny - 1
            for x in range(nx):
                if not land[y, x]:
                    tmp[y, x] = out[y, x]
                    continue
                xl = x - 1 if x > 0 else nx - 1
                xr = x + 1 if x < nx - 1 else 0
                avg = (out[yu, xl] + out[yu, x] + out[yu, xr] +
                       out[y, xl] + out[y, x] * 2.0 + out[y, xr] +
                       out[yd, xl] + out[yd, x] + out[yd, xr]) / 10.0
                tmp[y, x] = out[y, x] + blend * (avg - out[y, x])
        out, tmp = tmp, out
    return out


def apply_fluvial(h, uplift, runoff, lat, R, pos, strength, grade, Q0, mexp,
                  deposition_m, estuary_m, diffusion_passes, trunk_Q):
    """Carve graded valleys into `h` (metres, 2-D). Returns new height field.

    Routing and profiling run on lightly smoothed inputs so mesh-sampling
    jitter cannot imprint honeycomb chatter into the carve; the carve delta
    itself is diffused so valleys are wider than one raster cell and free of
    staircase streaks. The original heights are carved, never replaced.
    """
    from planet import iterative_route  # local import: planet imports this module
    ny, nx = h.shape
    N = nx * ny
    land = h > 0
    # Smooth working copies for topology only.
    hs = _diffuse(h, np.ones_like(h, dtype=bool), 2, 0.35, ny, nx)
    us = _diffuse(np.clip(uplift, 0, 3.0), np.ones_like(h, dtype=bool), 2, 0.4, ny, nx)
    iters = 3
    fill, rec, order, area = iterative_route(hs, lat, R,
                                            pos=pos.reshape(-1, 3),
                                            iterations=iters)
    Q = np.maximum(area.ravel() * np.maximum(runoff.ravel(), 0.05), 1.0)
    Unorm = np.clip(us.ravel(), 0, 3.0)
    step = np.zeros(N)
    pf = pos.reshape(-1, 3)
    for i in np.asarray(order):
        j = int(rec[int(i)])
        if j != int(i):
            step[int(i)] = float(np.linalg.norm(pf[int(i)] - pf[j])) * R / 1000.0
    z = np.empty(N)
    _graded_profile(np.asarray(order, dtype=np.int64),
                    np.asarray(rec, dtype=np.int64),
                    step, Unorm, Q, float(grade), float(Q0), float(mexp),
                    h.ravel(), z)
    z = z.reshape(ny, nx)
    land = h > 0
    # Carve valleys toward the graded profile; diffuse the delta first so
    # valleys span several cells and raster staircases never show.
    carve = np.clip(h - z, 0, None)
    carve = _diffuse(carve, np.ones_like(h, dtype=bool), 2, 0.45, ny, nx)
    fill_dep = np.clip(z - h, 0, float(deposition_m))
    h2 = h - float(strength) * carve * land
    h2 = np.where(land & (h < 400.0), h2 + 0.5 * fill_dep, h2)
    # Estuaries: only large trunks overdeepen near the coast, as single-cell
    # channels without dilation (dilation turns raster staircases into
    # fishbones). The final diffusion softens them into drowned valleys.
    near_coast = land & (h2 < 450.0)
    trunk = (area.reshape(ny, nx) > float(trunk_Q)) & near_coast
    h2 = np.where(trunk, np.minimum(h2, z - float(estuary_m)), h2)
    # Hillslope diffusion rounds interfluves, keeps channels.
    if diffusion_passes > 0:
        h2 = _diffuse(h2, land, int(diffusion_passes), 0.2, ny, nx)
    h2 = np.where(~land, np.minimum(h, -0.5), h2)
    info = {'carved_mean_m': float((h - h2)[land].mean()) if land.any() else 0.0,
            'trunk_cells': int(trunk.sum())}
    return h2, info
