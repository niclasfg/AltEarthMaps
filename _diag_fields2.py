# Temporary: channel-field statistics across regions (deleted after use).
import sys
import numpy as np
sys.path.insert(0, '/home/niclas/terrain_explorer')
import _port_waterways as P
from PIL import Image

OUT = '/mnt/c/Users/Niclas/.lmstudio/scratchpads/p6/'
patches = [(20.25, 10.4, 'lowland'), (72.16, 14.5, 'focus'), (20.0, -30.0, 'south'), (140.0, 45.0, 'east')]

def show(tag, lon, lat, fp, exact):
    r = P.render('tmp_' + tag, lon, lat, 24.0, 300, fp=fp, exact=exact)
    f = r['fields']
    parts = []
    for k in ('B', 'E', 'G'):
        v = f[k]
        qs = np.quantile(v, [.5, .9, .95, .99])
        parts.append('%s[p50=%.2f p90=%.2f p95=%.2f p99=%.2f max=%.2f]' % (k, qs[0], qs[1], qs[2], qs[3], v.max()))
    print('%-8s %-6s %s' % (tag, 'exact' if exact else ('%.3f' % fp), ' '.join(parts)))
    return r

for lon, lat, tag in patches:
    for fp, exact in ((.06, False), (.012, False), (None, True)):
        show(tag, lon, lat, fp, exact)

def panel(tag, lon, lat, fp, name):
    r = show(tag, lon, lat, fp, False)
    f = r['fields']; S = r['size']; tiles = []
    for k in ('B', 'E', 'G'):
        s = float(np.quantile(f[k], .985)) + 1e-9
        g = np.clip(f[k] / s, 0, 1).reshape(S, S)
        tiles.append(np.repeat(g[..., None], 3, -1))
    hh = r['shown'].reshape(S, S)
    nrm = np.stack([-(np.roll(hh, -1, 1) - np.roll(hh, 1, 1)), (np.roll(hh, -1, 0) - np.roll(hh, 1, 0)), np.full_like(hh, .004)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    sun = np.array([.5, .6, .62]); sun /= np.linalg.norm(sun)
    tiles.append(np.clip(nrm @ sun, 0, 1)[..., None] * np.array([.5, .49, .45]))
    img = np.concatenate(tiles, 1)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(OUT + name + '.png')
    print('  saved ' + name)

panel('lowland', 20.25, 10.4, .012, 'cand_lowland')
panel('focus', 72.16, 14.5, .012, 'cand_focus')
