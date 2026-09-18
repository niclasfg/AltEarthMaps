# Temporary diagnostics: compare candidate channel-field definitions (deleted after use).
import sys
import numpy as np
sys.path.insert(0, '/home/niclas/terrain_explorer')
import _port_waterways as P
from PIL import Image

OUT = '/mnt/c/Users/Niclas/.lmstudio/scratchpads/p6/'

def hillshade(shown, S):
    hh = shown.reshape(S, S)
    dx = (np.roll(hh, -1, 1) - np.roll(hh, 1, 1))
    dy = (np.roll(hh, -1, 0) - np.roll(hh, 1, 0))
    nrm = np.stack([-dx, dy, np.full_like(hh, .004)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    sun = np.array([.5, .6, .62]); sun /= np.linalg.norm(sun)
    return np.clip(nrm @ sun, 0, 1)[..., None] * np.array([.5, .49, .45])

for label, fp, exact in (('fp12m', .012, False), ('exact', None, True)):
    r = P.render('diag_' + label, 20.25, 10.4, 24.0, 420, fp=fp, exact=exact)
    f = r['fields']; S = r['size']
    print('=== %s ===' % label)
    for k in 'ABCD':
        q = np.quantile(f[k], [.5, .9, .99, .999])
        print('  field %s: p50=%.3f p90=%.3f p99=%.3f p99.9=%.3f max=%.3f' % (k, q[0], q[1], q[2], q[3], f[k].max()))
    tiles = []
    for k in 'ABCD':
        s = float(np.quantile(f[k], .99)) + 1e-9
        g = np.clip(f[k] / s, 0, 1).reshape(S, S)
        tiles.append(np.repeat(g[..., None], 3, -1))
    tiles.append(hillshade(r['shown'], S))
    w = r['cover'].reshape(S, S)
    tiles.append(np.repeat(w[..., None], 3, -1) * np.array([.16, .38, .47]) + (1 - np.repeat(w[..., None], 3, -1)) * .15)
    row1 = np.concatenate(tiles[:3], 1); row2 = np.concatenate(tiles[3:], 1)
    img = np.concatenate([row1, row2], 0)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(OUT + 'diag_' + label + '.png')
    print('  saved diag_%s.png  (row1: A B C, row2: D hillshade current-water)' % label)
