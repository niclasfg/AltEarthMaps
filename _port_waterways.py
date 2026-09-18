# Temporary reference port of the modified height shader (deleted after use).
# Mirrors shaders/common.glsl + shaders/terrain.glsl in float64 to preview the
# gradient-stepped waterway field and tune the hydrology thresholds.
import sys, math
import numpy as np
sys.path.insert(0, '/home/niclas/terrain_explorer')
import terrain
from PIL import Image
from scipy.spatial import cKDTree

ROOT = '/home/niclas/terrain_explorer/'
CACHE = ROOT + 'cache/4e513673dc15a50b/'
OUT = '/mnt/c/Users/Niclas/.lmstudio/scratchpads/p6/'

c = terrain.checked_config()
R = c['planet']['radius_metres'] / 1000.0
RP = c['relief']['plain_fraction']
TAU = 2 * math.pi
WORLD_SEED = c['planet']['seed']
rel, ero, app = c['relief'], c['erosion'], c['appearance']
R_WL = np.array(rel['wavelength_metres']) / 1000.0
R_AMP = np.array(rel['amplitude_metres']) / 1000.0
R_COL = np.array(rel['collision_gain']); R_RIFT = np.array(rel['rift_gain']); R_TR = np.array(rel['transform_gain'])
E_WL = np.array(ero['wavelength_metres']) / 1000.0
E_AMP = np.array(ero['amplitude_metres']) / 1000.0
E_GW = np.array(ero['gully_weight']); E_DET = np.array(ero['detail']); E_ONS = np.array(ero['onset'])
E_CS = np.array(ero['cell_scale']); E_NORM = np.array(ero['normalization'])
E_CR = np.array(ero['crease_rounding']); E_RR = np.array(ero['ridge_rounding'])
E_COL = np.array(ero['collision_gain']); E_RIFT = np.array(ero['rift_gain']); E_TR = np.array(ero['transform_gain'])
E_ASSUMED = ero['assumed_slope']; E_IRM = ero['initial_rounding_multiplier']; E_OBRM = ero['old_belt_rounding_multiplier']
HY_ONSET = c['hydrology']['gully_stream_onset']; HY_SOFT = c['hydrology']['gully_stream_softness']
HY_TWL = c['hydrology']['trunk_display_wavelength_metres'] / 1000.0
BANK_KM = c['hydrology']['valley_width_metres'] / 1000.0
E_N = len(E_WL)

# ---- bank / layer offsets exactly as terrain.bundle builds them ----
bank = []
def add(scale_m):
    i = len(bank); bank.append((scale_m / 1000.0, [i * 7.127, -i * 3.719, i * 11.23])); return i
R_IDS = np.array([add(w) for w in rel['wavelength_metres']])
E_IDS = np.array([add(w * E_CS[i]) for i, w in enumerate(ero['wavelength_metres'])])
D_IDS = np.array([add(w) for w in c['dunes']['wavelength_metres']])
S_IDS = np.array([add(w) for w in app['surface_wavelength_metres']])
TREE_ID = add(app['canopy_cell_metres']); MICRO_ID = add(0.6)
L_SCALE = np.array([b[0] for b in bank])
UCELL = np.zeros((len(bank), 3), np.int64); UFRAC = np.zeros((len(bank), 3))

def set_view(lon_deg, lat_deg):
    lon, lat = math.radians(lon_deg), math.radians(lat_deg)
    ctr = np.array([math.cos(lat) * math.cos(lon), math.sin(lat), math.cos(lat) * math.sin(lon)])
    for i, (sc, off) in enumerate(bank):
        for j in range(3):
            q = ctr[j] * R / sc + off[j]; k = math.floor(q)
            UCELL[i, j] = k; UFRAC[i, j] = q - k

def u32(a):
    return (np.asarray(a, dtype=np.int64) & np.int64(0xFFFFFFFF)).astype(np.uint32)

def mixbits(h):
    h = h ^ (h >> np.uint32(16)); h = h * np.uint32(0x7feb352d)
    h = h ^ (h >> np.uint32(15)); h = h * np.uint32(0x846ca68b)
    return h ^ (h >> np.uint32(16))

def ih(x, y, z):
    return mixbits(u32(x) ^ mixbits(u32(y) + np.uint32(0x9e3779b9)) ^ mixbits(u32(z) + u32(WORLD_SEED)))

def rand(x, y, z):
    return (ih(x, y, z) >> np.uint32(8)).astype(np.float64) / 16777216.0

def hash2(x, y, plane):
    h = ih(x, y, np.full_like(np.asarray(x), plane * 7919))
    a = (h >> np.uint32(8)).astype(np.float64) / 16777216.0 * 2 - 1
    b = (mixbits(h ^ np.uint32(0x68bc21eb)) >> np.uint32(8)).astype(np.float64) / 16777216.0 * 2 - 1
    return a, b

def ramp(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)

def footprint_weight(wl, fp):
    return 1 - ramp(.12, .48, fp / wl)

def proj(v, chart):
    return np.stack([v[:, 1], v[:, 2]], 1) if chart == 0 else (np.stack([v[:, 0], v[:, 2]], 1) if chart == 1 else np.stack([v[:, 0], v[:, 1]], 1))

def projI(v, chart):
    return np.array([v[1], v[2]]) if chart == 0 else (np.array([v[0], v[2]]) if chart == 1 else np.array([v[0], v[1]]))

def nd3(P, lid):
    sc = L_SCALE[lid]
    q = P / sc + UFRAC[lid]
    c0 = np.floor(q); f = q - c0; cell = c0.astype(np.int64) + UCELL[lid]
    u = f * f * f * (f * (f * 6 - 15) + 10); du = 30 * f * f * (f * (f - 2) + 1)
    h = np.zeros(len(P)); g = np.zeros((len(P), 3))
    for z in (0, 1):
        for y in (0, 1):
            for x in (0, 1):
                o = np.array([x, y, z], float)
                w = np.where(o == 0, 1 - u, u)
                a = rand(cell[:, 0] + x, cell[:, 1] + y, cell[:, 2] + z) * 2 - 1
                h += a * w[:, 0] * w[:, 1] * w[:, 2]
                sg = 2 * o - 1
                g += a[:, None] * sg[None, :] * du * np.stack([w[:, 1] * w[:, 2], w[:, 0] * w[:, 2], w[:, 0] * w[:, 1]], 1)
    return h, g / sc

def phacelle(P, lid, chart, dx, dy, cs, norm):
    q = proj(P / L_SCALE[lid] + UFRAC[lid], chart)
    c0 = np.floor(q); f = q - c0
    cell = c0.astype(np.int64) + projI(UCELL[lid], chart)
    side = np.stack([-dy, dx], 1) * cs * TAU
    sx = np.zeros(len(P)); sy = np.zeros(len(P)); tot = np.zeros(len(P))
    for i in (-1, 0, 1, 2):
        for j in (-1, 0, 1, 2):
            hx, hy = hash2(cell[:, 0] + i, cell[:, 1] + j, chart)
            d0 = f[:, 0] - i - hx * .5; d1 = f[:, 1] - j - hy * .5
            w = np.maximum(0, np.exp(-(d0 * d0 + d1 * d1) * 2) - .01111)
            t = d0 * side[:, 0] + d1 * side[:, 1] + .25 * TAU
            sx += np.cos(t) * w; sy += np.sin(t) * w; tot += w
    z = np.stack([sx, sy], 1) / np.maximum(tot, 1e-8)[:, None]
    ln = np.linalg.norm(z, axis=1)
    return z / np.maximum(1 - norm, ln)[:, None], side

def ease_out(t):
    v = 1 - np.clip(t, 0, 1); return 1 - v * v

def smooth_start(t, s):
    return np.where(t >= s, t - .5 * s, .5 * t * t / np.maximum(s, 1e-10))

def chart_weights(n):
    w = np.abs(n) ** 8; return w / w.sum(1, keepdims=True)

def erode(P, n, g, fade, tamp, fp, hist, rug, exact=False):
    N = len(P); wts = chart_weights(n)
    ex = np.zeros(N); dis = np.zeros(N)
    fields = {key: np.zeros(N) for key in 'ABCDEFGHIJ'}
    for chart in range(3):
        weight = wts[:, chart]
        g0 = proj(g, chart) - g[:, chart, None] * proj(n, chart) / n[:, chart, None]
        slope = np.maximum(np.linalg.norm(g0, axis=1), 1e-10)
        gs = g0 / slope[:, None] * E_ASSUMED
        f = np.clip(fade, -1, 1).copy()
        rounding = (E_CR[0] + (E_RR[0] - E_CR[0]) * np.clip(f + .5, 0, 1)) * E_IRM
        mask = ease_out(smooth_start(slope * 1.25, rounding * 1.25))
        he = np.zeros(N); hd = np.zeros(N)
        acc = {key: np.zeros(N) for key in 'ABCDEFGHIJ'}
        for k in range(E_N):
            vis = 1.0 if exact else float(footprint_weight(E_WL[k], fp))
            if (not exact) and vis == 0: break
            ph, side = phacelle(P, E_IDS[k], chart, gs[:, 0], gs[:, 1], E_CS[k], E_NORM[k])
            sw = side * (-1.0 / L_SCALE[E_IDS[k]])
            response = RP + (1 - RP) * np.clip(rug * E_COL[k] + hist[:, 1] * E_RIFT[k] + hist[:, 2] * E_TR[k], 0, 1)
            amp = E_AMP[k] * tamp * response
            gs = gs + np.sign(ph[:, 1])[:, None] * sw * amp[:, None] * E_GW[k]
            f_prev = f
            faded = f + (ph[:, 0] * E_GW[k] - f) * mask
            delta = (faded - .2) * amp; he += delta; hd += delta * vis; f = faded
            vis_eff = 1.0 if exact else vis
            acc['A'] += np.clip(-ph[:, 0], 0, 1) * mask * vis_eff      # phase x activity
            acc['B'] += np.clip(.2 - faded, 0, 1) * vis_eff            # cut state
            acc['C'] += np.clip(0., 0, 1) * vis_eff                    # (was raw phase)
            acc['D'] += mask * vis_eff                                  # activity alone
            acc['E'] += np.clip(f_prev - faded, 0, 1) * mask * vis_eff  # pull-down vs baseline
            acc['F'] += np.clip(.2 - faded, 0, 1) * amp * vis_eff       # cut depth (km)
            acc['G'] += np.clip(.2 - faded, 0, 1) * mask * vis_eff      # masked cut state
            acc['H'] += np.clip(.2 - faded, 0, 1) * mask * vis_eff * (E_AMP[k] / E_AMP[0]) ** .3   # coarse-biased
            acc['J'] += np.clip(f_prev - faded, 0, 1) * mask * vis_eff * (E_AMP[k] / E_AMP[0]) ** .3  # pull-down, coarse-biased
            acc['I'] += np.clip(.2 - faded, 0, 1) * np.sqrt(mask) * vis_eff                        # soft mask
            r = (E_CR[k] + (E_RR[k] - E_CR[k]) * np.clip(ph[:, 0] + .5, 0, 1)) * (E_OBRM + (1 - E_OBRM) * np.clip(hist[:, 0], 0, 1))
            nxt = ease_out(smooth_start(np.abs(ph[:, 1]) * E_ONS[k], r * E_ONS[k]))
            mask = (1 - np.power(1 - np.clip(mask, 0, 1), E_DET[k])) * nxt
        ex += he * weight; dis += hd * weight
        for key in 'ABCDEFGHIJ': fields[key] += acc[key] * weight
    return ex, dis, fields

def sphere_map(tex, n):
    H, W = tex.shape[:2]
    u = np.arctan2(n[:, 2], n[:, 0]) / TAU + .5
    v = np.arcsin(np.clip(n[:, 1], -1, 1)) / np.pi + .5
    qx = u * W - .5; qy = v * H - .5
    i0 = np.floor(qx).astype(np.int64); j0 = np.floor(qy).astype(np.int64); fx = qx - i0; fy = qy - j0
    i1 = (i0 + 1) % W; i0 = i0 % W; j1 = np.clip(j0 + 1, 0, H - 1); j0 = np.clip(j0, 0, H - 1)
    a = tex[j0, i0]; b = tex[j0, i1]; c2 = tex[j1, i0]; d2 = tex[j1, i1]
    return (a * (1 - fx)[:, None] + b * fx[:, None]) * (1 - fy)[:, None] + (c2 * (1 - fx)[:, None] + d2 * fx[:, None]) * fy[:, None]

macro = np.fromfile(CACHE + 'macro.bin', '<f4').reshape(1024, 2048, 4).astype(np.float64)
tect = np.fromfile(CACHE + 'tectonics.bin', '<f4').reshape(1024, 2048, 4).astype(np.float64)
rivers = np.fromfile(CACHE + 'rivers.bin', '<f4').reshape(-1, 12).astype(np.float64)
segA = rivers[:, 0:3]; segB = rivers[:, 4:7]
tt = np.linspace(0, 1, 3)[None, :, None]
samples = (segA[:, None, :] * (1 - tt) + segB[:, None, :] * tt).reshape(-1, 3)
sample_h = np.repeat((rivers[:, 3] + rivers[:, 7]) / 2, 3)
sample_w = np.repeat((rivers[:, 8] + rivers[:, 9]) / 2, 3)
rtree = cKDTree(samples)

def render(name, lon_deg, lat_deg, span_km, size, fp=None, exact=False):
    fp = span_km / size if fp is None else fp
    set_view(lon_deg, lat_deg)
    lon, lat = math.radians(lon_deg), math.radians(lat_deg)
    ctr = np.array([math.cos(lat) * math.cos(lon), math.sin(lat), math.cos(lat) * math.sin(lon)])
    east = np.array([-math.sin(lon), 0, math.cos(lon)]); north = np.cross(east, ctr); north /= np.linalg.norm(north)
    yy, xx = np.mgrid[0:size, 0:size]
    dE = (((xx + .5) / size - .5) * span_km / R).reshape(-1)
    dN = ((.5 - (yy + .5) / size) * span_km / R).reshape(-1)
    n = ctr[None, :] + dE[:, None] * east[None, :] + dN[:, None] * north[None, :]
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    P = n * R
    m = sphere_map(macro, n); tc = sphere_map(tect, n)
    land = ramp(-.03, .15, m[:, 0]); rug = m[:, 1]
    h = m[:, 0].copy(); shown = h.copy()
    rough = RP + (1 - RP) * rug
    ea = np.where(np.abs(n[:, 1:2]) < .99, np.cross(np.array([0., 1., 0.]), n), np.cross(np.array([1., 0., 0.]), n))
    ea = ea / np.linalg.norm(ea, axis=1, keepdims=True); no = np.cross(n, ea)
    dd = .00025
    def unit(v): return v / np.linalg.norm(v, axis=1, keepdims=True)
    a1 = sphere_map(macro, unit(n + ea * dd)); a0 = sphere_map(macro, unit(n - ea * dd))
    b1 = sphere_map(macro, unit(n + no * dd)); b0 = sphere_map(macro, unit(n - no * dd))
    g = (ea * (a1[:, 0] - a0[:, 0])[:, None] + no * (b1[:, 0] - b0[:, 0])[:, None]) / (2 * dd * R)
    rg = np.zeros((len(n), 3)); localRelief = np.zeros(len(n))
    for i in range(len(R_WL)):
        hh, gg = nd3(P, R_IDS[i])
        lr = RP + (1 - RP) * np.clip(rug * R_COL[i] + tc[:, 1] * R_RIFT[i] + tc[:, 2] * R_TR[i], 0, 1)
        amp = R_AMP[i] * lr * land
        h += hh * amp; shown += hh * amp * (1.0 if exact else footprint_weight(R_WL[i], fp)); rg += gg * amp[:, None]; localRelief += hh * amp
    g = g + rg; g = g - n * np.sum(n * g, 1)[:, None]
    erosionAmp = land * (.22 + .78 * ramp(.003, .05, np.linalg.norm(g, axis=1)))
    maxDrop = np.zeros(len(n))
    for i in range(E_N):
        resp = RP + (1 - RP) * np.clip(rug * E_COL[i] + tc[:, 1] * E_RIFT[i] + tc[:, 2] * E_TR[i], 0, 1)
        maxDrop += E_AMP[i] * 1.2 * resp
    erosionAmp = np.minimum(erosionAmp, np.maximum(0, h - .0003) / np.maximum(maxDrop, 1e-5))
    ex, dis, fields = erode(P, n, g, np.clip(localRelief / np.maximum(.05, rough), -1, 1), erosionAmp, fp, tc, rug, exact)
    h += ex; shown += dis
    st = fields['J']
    streamCover = np.where((st > HY_ONSET) & (land > .05) & (shown > -.001), ramp(HY_ONSET, HY_ONSET + HY_SOFT, st), 0.0)
    dist, idx = rtree.query(n)
    dkm = 2 * np.arcsin(np.clip(dist / 2, 0, 1)) * R
    w = sample_w[idx]; elev = sample_h[idx]
    tv = 1.0 if exact else float(footprint_weight(HY_TWL, fp))
    trunkCover = (1 - ramp(-max(fp * .6, .0005), max(fp * .6, .0005), dkm - w)) * tv
    infl = 1 - ramp(0, BANK_KM, dkm)
    target = elev - .003 + np.power(np.clip(dkm / BANK_KM, 0, None), 1.6) * np.maximum(.015, m[:, 0] - elev)
    shownT = shown * (1 - infl) + np.minimum(shown, target) * infl
    cover = np.maximum(trunkCover, streamCover)
    wet = cover > .01
    print('%-11s fp=%7.1f m/px  wet=%6.3f%%  (stream %.3f%%, trunk %.3f%%)  max st=%.2f' % (
        name, fp * 1000, 100 * wet.mean(), 100 * ((streamCover > .01) & (trunkCover <= .01)).mean(),
        100 * ((trunkCover > .01) & (streamCover <= .01)).mean(), st.max()))
    S = size; sp = span_km / size
    hh = shownT.reshape(S, S)
    dx = (np.roll(hh, -1, 1) - np.roll(hh, 1, 1)) * sp
    dy = (np.roll(hh, -1, 0) - np.roll(hh, 1, 0)) * sp
    nrm = np.stack([-dx, dy, np.full_like(hh, sp * 2)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    sun = np.array([.5, .6, .62]); sun /= np.linalg.norm(sun)
    light = np.clip(nrm @ sun, .10, None)
    col = np.array([.42, .41, .34]) * light[..., None]
    wc = cover.reshape(S, S)[..., None]
    col = col * (1 - wc) + np.stack([np.full((S, S), .16), np.full((S, S), .38), np.full((S, S), .47)], -1) * wc
    col[land.reshape(S, S) < .05] = np.array([.07, .17, .31])
    Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8)).save(OUT + name + '.png')
    return {'fields': fields, 'size': S, 'land': land, 'shown': shown, 'cover': cover}

if __name__ == '__main__':
    render('port_far', 20.25, 10.4, 96.0, 420, fp=.24)
    render('port_mid', 20.25, 10.4, 24.0, 420, fp=.06)
    render('port_near', 20.25, 10.4, 24.0, 420, fp=.012)
    render('port_exact', 20.25, 10.4, 24.0, 420, exact=True)
    print('done')
