# Temporary: compare smoothing variants for the waterway field (deleted after use).
import sys
import numpy as np
sys.path.insert(0, '/home/niclas/terrain_explorer')
import _port_waterways as P
from PIL import Image

OUT = '/mnt/c/Users/Niclas/.lmstudio/scratchpads/p6/'

def panel(tag, lon, lat, fp, name):
    r = P.render('tmp2_' + tag, lon, lat, 24.0, 300, fp=fp, exact=False)
    f = r['fields']; S = r['size']; tiles = []
    for k in ('G', 'H', 'I'):
        v = f[k]
        qs = np.quantile(v, [.5, .9, .99])
        print('  %s: p50=%.2f p90=%.2f p99=%.2f max=%.2f' % (k, qs[0], qs[1], qs[2], v.max()))
        s = float(np.quantile(v, .985)) + 1e-9
        g = np.clip(v / s, 0, 1).reshape(S, S)
        tiles.append(np.repeat(g[..., None], 3, -1))
    hh = r['shown'].reshape(S, S)
    nrm = np.stack([-(np.roll(hh, -1, 1) - np.roll(hh, 1, 1)), (np.roll(hh, -1, 0) - np.roll(hh, 1, 0)), np.full_like(hh, .004)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    sun = np.array([.5, .6, .62]); sun /= np.linalg.norm(sun)
    tiles.append(np.clip(nrm @ sun, 0, 1)[..., None] * np.array([.5, .49, .45]))
    img = np.concatenate(tiles, 1)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(OUT + name + '.png')
    print('  saved %s   (G | H coarse-biased | I soft-mask | hillshade)' % name)

print('lowland fp=12m')
panel('lowland', 20.25, 10.4, .012, 'smooth_lowland')
print('focus fp=12m')
panel('focus', 72.16, 14.5, .012, 'smooth_focus')
