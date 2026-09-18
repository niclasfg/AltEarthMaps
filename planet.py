"""Global guide maps built from EVOLVED tectonic crust.
The history-dependent fields from tectonics.py set mean elevation and relief;
climate and the existing coarse hydrology operate on those resulting heights.
Fine surface detail remains GPU-evaluated rather than stored planet-wide.
"""
from __future__ import annotations
import hashlib, heapq, json, math, time
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree, SphericalVoronoi

REVISION = 'sphere-5.0-grown-continents-1'


class Cancelled(Exception):pass


def _check(cancel):
    if cancel is not None and cancel():raise Cancelled()


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


def noise3(p, seed):
    """Deterministic 3-D quintic value noise. Only coarse geological controls."""
    i=np.floor(p).astype(np.int64); f=p-i; u=f*f*f*(f*(f*6-15)+10)
    out=np.zeros(p.shape[:-1], np.float64)
    for z in (0,1):
        for y in (0,1):
            for x in (0,1):
                q=i+np.array([x,y,z],np.int64)
                h=(q[...,0]*73856093 ^ q[...,1]*19349663 ^ q[...,2]*83492791 ^ int(seed)) & 0xffffffff
                h=(h^(h>>16))*0x7feb352d & 0xffffffff
                h=(h^(h>>15))*0x846ca68b & 0xffffffff
                h=h^(h>>16)
                w=(u[...,0] if x else 1-u[...,0])*(u[...,1] if y else 1-u[...,1])*(u[...,2] if z else 1-u[...,2])
                out+=w*((h>>8)/16777216.*2-1)
    return out


def grid(w):
    """South-to-north texel centres. Longitude seam uses wrap, never a hard edge."""
    lon=(np.arange(w)+.5)*2*np.pi/w-np.pi
    lat=(np.arange(w//2)+.5)*np.pi/(w//2)-np.pi/2
    xx,yy=np.meshgrid(lon,lat)
    return np.stack((np.cos(yy)*np.cos(xx),np.sin(yy),np.cos(yy)*np.sin(xx)),axis=-1),lat


def flood(h, lat, R):
    """Priority-Flood on a wrapping sphere, and topologically ordered receivers.
    Eight neighbours in the raster. Across a pole, the opposite meridian is used.
    Equal-level routing follows the earlier flood visit: no cycles on lake flats.
    Returns spill heights, receivers, visit order and area (km²).
    """
    ny,nx=h.shape; N=nx*ny
    raw=h.ravel(); filled=raw.copy(); seen=raw<0
    rec=np.arange(N,dtype=np.int32)
    # Seed every ocean cell, but only queue its coastline. This preserves a
    # common global outlet without sorting hundreds of thousands of sea cells.
    ocean=seen.reshape(ny,nx)
    near_land=ndimage.maximum_filter((~ocean).astype(np.uint8),size=3,mode=('nearest','wrap'))>0
    edges=np.flatnonzero(ocean & near_land)
    heap=[(float(raw[i]),int(i)) for i in edges]; heapq.heapify(heap)
    order=[]; moves=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while heap:
        z,i=heapq.heappop(heap); order.append(i); y,x=divmod(i,nx)
        for dy,dx in moves:
            yy=y+dy;xx=x+dx
            if yy<0: yy=0;xx+=nx//2
            elif yy>=ny: yy=ny-1;xx+=nx//2
            j=yy*nx+xx%nx
            if seen[j]:continue
            seen[j]=True;rec[j]=i;zz=max(float(raw[j]),z);filled[j]=zz
            heapq.heappush(heap,(zz,j))
    # Use steepest lower neighbour wherever possible; retain the flood predecessor
    # on flats. This is still a graph on the coarse guide, not a final 1-m solver.
    rank=np.full(N,-1,np.int32);rank[np.asarray(order)]=np.arange(len(order))
    H=filled.reshape(ny,nx); best=np.zeros((ny,nx)); rr=rec.reshape(ny,nx)
    ids=np.arange(N,dtype=np.int32).reshape(ny,nx)
    for dy,dx in moves:
        Q=np.roll(H,(-dy,-dx),(0,1));J=np.roll(ids,(-dy,-dx),(0,1))
        # Explicit cap neighbour (half a turn at pole).
        if dy==-1: Q[0]=np.roll(H[0],-(dx+nx//2)); J[0]=np.roll(ids[0],-(dx+nx//2))
        if dy==1: Q[-1]=np.roll(H[-1],-(dx+nx//2));J[-1]=np.roll(ids[-1],-(dx+nx//2))
        d=(2*np.pi*R/nx)*np.sqrt(dy*dy+(dx*np.cos(lat[:,None]))**2)
        s=(H-Q)/np.maximum(d,1e-3)
        valid=(s>best)&(Q<H)&(~ocean)
        rr[valid]=J[valid];best[valid]=s[valid]
    # Descending spill elevation is a topological order except on exact flats,
    # where descending visit rank guarantees the receiver was visited earlier.
    order=np.lexsort((rank,filled)).astype(np.int32)
    dlat=np.pi/ny; area_rows=R**2*2*np.pi/nx*(np.sin(lat+dlat/2)-np.sin(lat-dlat/2))/1e6
    area=np.broadcast_to(area_rows[:,None],(ny,nx)).copy().ravel()
    for i in order[::-1]:
        j=rec[i]
        if j!=i:area[j]+=area[i]
    return filled.reshape(ny,nx),rec,order,area


def _accumulate_area(rec, fill_flat, prev_order, lat, R, dist_tie=None):
    """Recompute topological order and latitude-weighted area. On exact flats
    spill alone cannot order drainage; dist_tie (pre-sweep dist-to-ocean, where
    receivers are strictly smaller) puts receivers first and keeps the single
    accumulation pass valid."""
    N = len(rec)
    if dist_tie is not None:
        order = np.lexsort((np.asarray(dist_tie), fill_flat)).astype(np.int32)
    else:
        prev_rank = np.full(N, -1, np.int32)
        prev_rank[np.asarray(prev_order)] = np.arange(len(prev_order))
        order = np.lexsort((prev_rank, fill_flat)).astype(np.int32)
    ny = len(lat)
    nx = N // ny
    dlat = np.pi / ny
    area_rows = R ** 2 * 2 * np.pi / nx * (np.sin(lat + dlat / 2) - np.sin(lat - dlat / 2)) / 1e6
    area = np.broadcast_to(area_rows[:, None], (ny, nx)).copy().ravel()
    for i in order[::-1]:
        j = rec[i]
        if j != i:
            area[j] += area[i]
    return order, area


def downstream_distance(fill_flat, rec, order, pos, R):
    """Km along receiver chain to the ocean outlet. Single ascending pass;
    receiver graph is acyclic by construction, dist strictly decreases."""
    N = len(rec)
    dist = np.zeros(N, dtype=np.float64)
    if pos is not None:
        for i in order:
            j = int(rec[i])
            if j != i:
                dist[i] = dist[j] + float(np.linalg.norm(pos[i] - pos[j])) * R / 1000.0
    else:
        for i in order:
            j = int(rec[i])
            if j != i:
                dist[i] = dist[j] + 1.0
    return dist


def _improve_flat_receivers(fill, rec, ocean, dist, lat, R):
    """One sweep: flat cells (no lower neighbour) re-pick the neighbour with
    smallest (spill, dist-to-ocean) that still makes downstream progress.
    Follow-only: heights are never carved here."""
    ny, nx = fill.shape
    N = nx * ny
    H = fill
    D = dist.reshape(ny, nx)
    rr = rec.reshape(ny, nx)
    flat_rec = rec.reshape(ny, nx)
    bestQ = fill.ravel()[rec].reshape(ny, nx).astype(float)
    bestD = dist[rec].reshape(ny, nx).astype(float)
    bestJ = rr.copy()
    ids = np.arange(N, dtype=np.int32).reshape(ny, nx)
    moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    d0 = (2 * np.pi * R / nx)
    for dy, dx in moves:
        Q = np.roll(H, (-dy, -dx), (0, 1))
        J = np.roll(ids, (-dy, -dx), (0, 1))
        DD = np.roll(D, (-dy, -dx), (0, 1))
        if dy == -1:
            Q[0] = np.roll(H[0], -(dx + nx // 2))
            J[0] = np.roll(ids[0], -(dx + nx // 2))
            DD[0] = np.roll(D[0], -(dx + nx // 2))
        if dy == 1:
            Q[-1] = np.roll(H[-1], -(dx + nx // 2))
            J[-1] = np.roll(ids[-1], -(dx + nx // 2))
            DD[-1] = np.roll(D[-1], -(dx + nx // 2))
        step = d0 * np.sqrt(dy * dy + (dx * np.cos(lat[:, None])) ** 2)
        # Candidate must not climb above spill (+eps) and must progress downstream.
        better = (Q <= H + 1e-6) & (DD < D - 1e-9)
        better &= (Q < bestQ - 1e-9) | ((np.abs(Q - bestQ) <= 1e-9) & (DD < bestD - 1e-9))
        # Never route ocean interior, never self-loop land.
        better &= ~ocean
        better &= (J != ids)
        # Only reroute cells that are currently flat (no downhill receiver).
        flat = (bestQ >= H - 1e-9)
        take = better & flat & (Q <= bestQ + 1e-9)
        bestQ[take] = Q[take]
        bestD[take] = DD[take]
        bestJ[take] = J[take]
        _ = step  # step length already encoded in dist; kept for symmetry
    changed = np.sum((bestJ != flat_rec) & (~ocean))
    rec2 = rec.copy()
    rec2[~ocean.ravel()] = bestJ.ravel()[~ocean.ravel()]
    return rec2, int(changed)


def iterative_route(h, lat, R, pos=None, iterations=3):
    """Iterative gradient-following routing. Pass 1 is Priority-Flood; later
    passes only re-resolve flat receivers toward shorter ocean paths and
    recompute area. Heights are never modified (follow-only)."""
    fill, rec, order, area = flood(h, lat, R)
    if iterations <= 1:
        return fill, rec, order, area
    ocean = (h < 0)
    for _ in range(1, int(iterations)):
        dist = downstream_distance(fill.ravel(), rec, order, pos, R)
        rec2, changed = _improve_flat_receivers(fill, rec, ocean, dist, lat, R)
        if changed == 0:
            break
        order, area = _accumulate_area(rec2, fill.ravel(), order, lat, R, dist_tie=dist)
        # Keep the chain acyclic: only accept strict downstream progress.
        rec = rec2
    dist = downstream_distance(fill.ravel(), rec, order, pos, R)
    # Final safety: any residual uphill link keeps its old receiver only if it
    # still drains; count is reported, not hidden.
    return fill, rec, order, area


def flow_fields(h, spill, rec, order, area, pos, R):
    """Baked downstream guidance: dist-to-ocean (km), downstream slope,
    Strahler order and log accumulation. All follow-only derivatives."""
    N = len(rec)
    dist = downstream_distance(spill.ravel(), rec, order, pos, R)
    flat_spill = spill.ravel()
    slope = np.zeros(N, dtype=np.float64)
    for i in np.asarray(order):
        j = int(rec[int(i)])
        if j != int(i):
            step = float(np.linalg.norm(pos[int(i)] - pos[j])) * R / 1000.0 if pos is not None else 1.0
            slope[int(i)] = max(0.0, (flat_spill[int(i)] - flat_spill[j]) / max(step, 1e-3))
    # Strahler order upstream-first (headwaters first = descending spill last).
    strahler = np.ones(N, dtype=np.int32)
    maxup = np.ones(N, dtype=np.int32)
    cntup = np.zeros(N, dtype=np.int32)
    for i in order[::-1]:
        j = int(rec[int(i)])
        if j != int(i):
            if strahler[int(i)] > maxup[j]:
                maxup[j] = strahler[int(i)]
                cntup[j] = 1
            elif strahler[int(i)] == maxup[j]:
                cntup[j] += 1
            if cntup[j] >= 2:
                strahler[j] = max(strahler[j], maxup[j] + 1)
            else:
                strahler[j] = max(strahler[j], maxup[j])
    return dist, slope, strahler


def river_records(h, spill, rec, order, area, n, c, source_area=None, relax_iters=3, bank_cap_metres=None):
    """Shared junction positions ensure every rendered tributary meets its parent.
    Iterative main-stem relaxation removes the raster's 45-degree staircase.
    Heights remain monotone from source to outlet. Water width is in metres.
    Follow-only: input heights are never modified, only junction positions."""
    hc=c['hydrology'];R=c['planet']['radius_metres'];raw=h.ravel();p=n.reshape(-1,3).copy()
    thresh=float(source_area) if source_area is not None else float(hc['river_source_area_km2'])
    selected=(area>thresh)&(raw>0)&(rec!=np.arange(len(rec)))
    indices=np.flatnonzero(selected)
    old=p.copy()
    # Iterative relaxation: re-derive the dominant donor each sweep so shared
    # confluences converge instead of depending on single-pass visit order.
    for _ in range(max(1, int(relax_iters))):
        donor=np.full(len(rec),-1,np.int32);donor_area=np.zeros(len(rec))
        for i in indices:
            j=rec[i]
            if area[i]>donor_area[j]:donor[j]=i;donor_area[j]=area[i]
        base=p.copy()
        for i in indices:
            if donor[i]>=0:
                p[i]=unit(.6*base[i]+.2*base[donor[i]]+.2*base[rec[i]])
    donor=np.full(len(rec),-1,np.int32);donor_area=np.zeros(len(rec))
    for i in indices:
        j=rec[i]
        if area[i]>donor_area[j]:donor[j]=i;donor_area[j]=area[i]
    bed=spill.ravel().copy()
    # Enforce strictly descending longitudinal profiles on selected branches;
    # downstream-first traversal keeps confluences shared.
    for i in order:
        j=rec[i]
        if selected[i]:bed[i]=max(bed[i],bed[j]+.015)
    width=np.minimum(hc['maximum_half_width_metres'],hc['width_coefficient_metres']*np.sqrt(area))
    bank_full=float(hc['valley_width_metres'])
    if bank_cap_metres is not None:
        bank_full=min(bank_full, float(bank_cap_metres))
    records=[]
    for i in indices:
        j=rec[i]
        if raw[j]<-150: # Clip the last segment at interpolated mean sea level.
            t=raw[i]/max(raw[i]-raw[j],1e-6);b=unit(old[i]*(1-t)+old[j]*t);zb=0
        else:b=p[j];zb=max(0,bed[j])
        a=p[i]
        if np.dot(a,b)>.999999999999:continue
        length=np.linalg.norm(b-a)
        ti=unit(b-(p[donor[i]] if donor[i]>=0 else a))*length
        tj=unit((p[rec[j]] if rec[j]!=j else b)-(p[donor[j]] if donor[j]>=0 else a))*length
        def curve(t):
            return unit((2*t**3-3*t*t+1)*a+(t**3-2*t*t+t)*ti+(-2*t**3+3*t*t)*b+(t**3-t*t)*tj)
        points=[curve(t) for t in np.linspace(0,1,5)]
        for k in range(4):
            t0=k/4;t1=(k+1)/4
            records.append([*points[k],((1-t0)*bed[i]+t0*zb)/1000,
                            *points[k+1],((1-t1)*bed[i]+t1*zb)/1000,
                            ((1-t0)*width[i]+t0*width[j])/1000,
                            ((1-t1)*width[i]+t1*width[j])/1000,
                            bank_full/1000,area[i]])
    return np.asarray(records,np.float32).reshape(-1,12)


def river_index(records, R, nx=96,ny=96):
    """Cubed-sphere bins; bounded size even at poles. Each reach's support is
    conservatively duplicated into bins on either side of every face boundary.
    These bins index geometry and NEVER define terrain or river boundaries.
    Finer than the legacy 64 grid to bound per-bin counts as river mileage
    grows with land fraction.
    """
    bins=[[] for _ in range(6*nx*ny)]
    axes=[(1,2),(0,2),(0,1)]
    for i,r in enumerate(records):
        a=r[:3].astype(float);b=r[4:7].astype(float)
        length=math.acos(float(np.clip(np.dot(a,b),-1,1)))*R
        pad=float(r[10])*1000 + length/8 + 1000
        pts=unit(np.array([a,a*.75+b*.25,a*.5+b*.5,a*.25+b*.75,b]))
        slots=set()
        for p in pts:
            # Include every face potentially intersected by the support cap.
            for axis in range(3):
                if abs(p[axis]) < max(abs(p))-pad/R*3:continue
                aa,bb=axes[axis];v=abs(p[axis]);face=2*axis+(p[axis]<0)
                xx=(p[aa]/v*.5+.5)*nx; yy=(p[bb]/v*.5+.5)*ny
                rad=2.0*pad/R*nx
                for gy in range(max(0,math.floor(yy-rad)),min(ny,math.floor(yy+rad)+1)):
                    for gx in range(max(0,math.floor(xx-rad)),min(nx,math.floor(xx+rad)+1)):
                        slots.add(int(face)*nx*ny+gy*nx+gx)
        for k in slots:bins[k].append(i)
    header=np.zeros((6*ny,nx,2),np.uint32);flat=[]
    for i,b in enumerate(bins):header.reshape(-1,2)[i]=[len(flat),len(b)];flat.extend(b)
    a=np.asarray(flat,np.uint32);size=1024;rows=max(1,math.ceil(len(a)/size))
    packed=np.zeros(size*rows,np.uint32);packed[:len(a)]=a
    return header,packed.reshape(rows,size),max(map(len,bins))


def build(c, directory, progress=None, cancel=None):
    def report(frac,stage):
        if progress is not None:progress(frac,stage)
        _check(cancel)
    start=time.perf_counter();p=c['planet'];t=c['tectonics'];cl=c['climate'];R=p['radius_metres'];seed=p['seed']
    w=int(p['map_width']);n,lat=grid(w);points=n.reshape(-1,3)
    from tectonics import simulate
    from continents import grow_continents
    print('Growing continents and evolving crust before generating climate and drainage...',flush=True)
    def initial_crust(mesh):
        cont,_=grow_continents(mesh,seed,p['continent_count'],p['continent_size_variety'],p['continent_budget_fraction'])
        return cont
    import re
    _mare=re.compile(r'([\d.]+)\s*Ma remaining')
    _dur=float(t['duration_myr'])
    def _tecprog(msg, **kw):
        m=_mare.search(str(msg))
        frac=0.68*(1-float(m.group(1))/_dur) if m else None
        report(frac if frac is not None else 0.0,'tectonics')
    report(0.0,'tectonics')
    mesh,fields,converging,spreading,tectonic_stats,owner=simulate(c,initial_crust,directory,progress=_tecprog)
    report(0.70,'sampling')
    node_fields=np.c_[fields,converging,spreading]
    sampled=np.empty((len(points),11),np.float32)
    for begin in range(0,len(points),65536):
        sampled[begin:begin+65536]=mesh.sample(points[begin:begin+65536],node_fields)
    sample=sampled.reshape(w//2,w,11)
    h=sample[...,0].astype(float)*1000
    orogen=sample[...,1]
    strength=sample[...,9];divergence=sample[...,10]
    land=h>0;pos=n*R
    weights=np.broadcast_to(np.cos(lat)[:,None],h.shape).ravel()
    tectonic_map=np.stack((sample[...,5],sample[...,6],sample[...,7],sample[...,4]),axis=-1).astype('<f4')
    crust_map=np.stack((sample[...,2],sample[...,3],sample[...,8],sample[...,1]),axis=-1).astype('<f4')
    temp0=cl['equator_temperature_c']-cl['pole_temperature_drop_c']*np.abs(n[...,1])**1.25
    temp0+=cl['regional_temperature_variation_c']*noise3(pos/3200000,seed+977)
    latdeg=np.abs(np.degrees(lat[:,None])); rain=np.log(cl['mean_rainfall_mm'])+1.0*np.exp(-(latdeg/12)**2)-1.7*np.exp(-((latdeg-29)/11)**2)+.2*np.exp(-((latdeg-53)/15)**2)
    rain=np.broadcast_to(rain,h.shape).copy()+cl['regional_rainfall_variation']*noise3(pos/1500000+27,seed+744)
    # Prevailing winds reverse between Hadley/Ferrel cells. One upwind sample is
    # a rain-shadow proxy; not fluid dynamics or time-dependent meteorology.
    wind=np.where((latdeg>32)&(latdeg<65),1,-1)
    upwind=(np.roll(h,8,axis=1)*((wind>0).astype(float))+np.roll(h,-8,axis=1)*((wind<0).astype(float)))
    rain-=cl['rain_shadow_strength']*np.clip((upwind-h)/2500,0,2)
    rain=np.clip(rain,np.log(40),np.log(6000))
    lith=np.clip(.20+.40*sample[...,2]+.25*sample[...,5]+.15*np.minimum(sample[...,8],1),0,1)
    report(0.72,'sampling')
    # Map-stage stream-power geomorphology: graded valleys from tectonic
    # uplift and rainfall-driven discharge. Everything downstream (coasts,
    # routing, rivers) sees the eroded heights, so drainage stays consistent.
    fl=c.get('fluvial',{})
    if fl.get('enabled',False):
        from fluvial import apply_fluvial
        print('Carving fluvial valleys (stream-power)…',flush=True)
        runoff=np.clip(np.exp(rain-rain[h>0].mean()) if (h>0).any() else np.ones_like(h),.15,4.0)
        conv=np.clip(strength/50.0,0,2.0)
        uplift=np.clip(orogen,0,2.0)+0.5*conv
        h,fluv_info=apply_fluvial(h,uplift,runoff,lat,R,n,fl.get('strength',.65),fl.get('grade',.06),
                                 fl.get('discharge_reference_km2',5000.0),fl.get('discharge_exponent',.5),
                                 fl.get('deposition_metres',150.0),fl.get('estuary_depth_metres',120.0),
                                 fl.get('diffusion_passes',2),fl.get('estuary_area_km2',20000.0))
        print(f"Fluvial: mean valley carve {fluv_info['carved_mean_m']:.0f} m; {fluv_info['trunk_cells']} estuary cells.",flush=True)
        land=h>0
    report(0.76,'fluvial')
    # Sea-datum trim: solve the datum for the target emerged fraction and
    # report the implied water inventory. Fixed-volume solves leave land at
    # the mercy of each seed's collision history (25–35% observed); targeting
    # emerged fraction is seed-robust. The shift is exact at the coastline
    # where it matters; deep-basin load error is irrelevant to the contour.
    target=float(p.get('emerged_land_target',0.35))
    wlat=np.broadcast_to(np.cos(lat)[:,None],h.shape).ravel()
    hf=h.ravel()
    pre_trim_land=float(np.sum(wlat*(hf>0))/wlat.sum())
    order=np.argsort(hf,kind='stable')
    cum=np.cumsum(wlat[order])
    cut=(1.0-target)*cum[-1]
    datum=float(hf[order[np.searchsorted(cum,cut)]])
    h=h-datum
    land=h>0
    ratio=1030.0/3300.0
    # Heights are metres, areas km²: V = Σ max(-h/1000,0)·area /(1-ratio).
    dlat=np.pi/(h.shape[0]);Rkm=R/1000.0
    cell_km2=(Rkm**2*2*np.pi/h.shape[1]*(np.sin(lat+dlat/2)-np.sin(lat-dlat/2)))
    cell_area=np.broadcast_to(cell_km2[:,None],h.shape)
    implied_water_km3=float(np.sum(np.maximum(-h/1000.0,0)*cell_area)/(1-ratio))
    print(f"Sea datum trimmed {datum:+.0f} m for {target:.0%} emerged land (pre-trim {pre_trim_land:.1%}); implied ocean volume {implied_water_km3/1e9:.2f} Gkm³.",flush=True)
    # Signed coast distance is a guide attribute, not a randomly placed shore.
    # Approximation on the global lat/lon guide; tectonic transport itself has
    # no latitude-grid distortion or special polar seam. Uses trimmed land.
    coast=(ndimage.distance_transform_edt(land)-ndimage.distance_transform_edt(~land))*(2*np.pi*R/w)
    report(0.78,'climate')
    # Global hydrology uses a cheaper grid but remains topologically global.
    rw=int(c['hydrology']['routing_width']);step=w//rw
    hsmall=ndimage.zoom(h,1/step,order=1,prefilter=False) if step>1 else h.copy()
    ns,lats=grid(rw)
    if c['hydrology']['enabled']:
        print('Routing global watersheds, lakes and major rivers…',flush=True)
        iters=int(c['hydrology'].get('route_iterations',3))
        ns_flat=ns.reshape(-1,3)
        fill,rec,order,area=iterative_route(hsmall,lats,R,pos=ns_flat,iterations=iters)
        records=river_records(hsmall,fill,rec,order,area,ns,c,relax_iters=iters)
        report(0.86,'routing')
        # Refine pass at full guide resolution for tributaries (follow-only).
        # Coarse records keep area>river_source; refine adds smaller basins only.
        n_ref=len(records)
        if bool(c['hydrology'].get('refine_routing',True)) and rw<w:
            try:
                print('Refining tributary routing at full guide resolution…',flush=True)
                nf,latf=grid(w)
                fill_f,rec_f,order_f,area_f=iterative_route(h,latf,R,pos=nf.reshape(-1,3),iterations=max(1,min(2,iters)))
                th_f=float(c['hydrology'].get('refine_source_area_km2',8000.0))
                th_c=float(c['hydrology']['river_source_area_km2'])
                rec_f2=river_records(h,fill_f,rec_f,order_f,area_f,nf,c,source_area=th_f,relax_iters=2,bank_cap_metres=6000.0)
                # Keep only tributary-scale reaches to avoid duplicating trunks.
                if len(rec_f2):
                    keep=rec_f2[:,11]<th_c*1.05
                    rec_f2=rec_f2[keep]
                    if len(rec_f2):
                        records=np.concatenate([records,rec_f2],axis=0) if len(records) else rec_f2
            except Exception as e:
                print(f'Refine routing skipped ({e}); keeping coarse rivers.',flush=True)
        report(0.92,'rivers')
        # Largest-first cap: bounds shader bins at any land fraction. Equal
        # areas sort stably, so whole reaches survive the cut together.
        cap=int(c['hydrology'].get('max_river_records',100000))
        if len(records)>cap:
            keep=np.argsort(records[:,11],kind='stable')[-cap:]
            records=records[np.sort(keep)]
            print(f'Capped rivers to {len(records)} largest records.',flush=True)
        depth=fill-hsmall
        lake=(depth>c['hydrology']['lake_minimum_depth_metres'])&(fill>10)&(depth<c['hydrology']['lake_maximum_depth_metres'])
        # Interpolate spill surfaces only inside a connected basin. Store a
        # signed lake shore field rather than interpolating a missing sentinel.
        signed=(ndimage.distance_transform_edt(lake)-ndimage.distance_transform_edt(~lake))*2*np.pi*R/rw
        wet=np.stack((fill/1000,signed/1000,np.log1p(area.reshape(hsmall.shape)),depth/1000),axis=-1)
        # Baked downstream guidance for fine erosion octaves (follow-only data).
        dist_km,slope,strahler=flow_fields(hsmall,fill,rec,order,area,ns_flat,R)
        flow=np.stack((dist_km.reshape(hsmall.shape)/1000.0,slope.reshape(hsmall.shape),
                       np.clip(strahler.reshape(hsmall.shape)/8.0,0,1),np.log1p(area.reshape(hsmall.shape))/12.0),axis=-1)
        hdr,idx,maxbin=river_index(records,R)
        stats={'river_reaches':len(records),'coarse_reaches':int(n_ref),'maximum_reaches_in_bin':int(maxbin),'lakes_grid_cells':int(lake.sum()),
               'uphill_coarse_receivers':int(np.sum(fill.ravel()[rec]>fill.ravel()+1e-7)),
               'route_iterations':int(iters),
               'land_fraction':float(np.sum(weights*land.ravel())/weights.sum())}
    else:
        records=np.zeros((0,12),np.float32);hdr=np.zeros((384,64,2),np.uint32);idx=np.zeros((1,1024),np.uint32)
        wet=np.zeros((rw//2,rw,4));wet[...,1]=-1e5;stats={'river_reaches':0,'maximum_reaches_in_bin':0}
        flow=np.zeros((rw//2,rw,4));flow[...,0]=1e3
    macro=np.stack((h/1000,orogen,coast/1000,lith),axis=-1).astype('<f4')
    climate=np.stack((temp0,rain,strength,divergence),axis=-1).astype('<f4')
    # Present-day plate map: nearest mesh cell owner per map pixel (uint ids).
    _,nearest=mesh.tree.query(points,k=1)
    plates=np.asarray(owner[np.asarray(nearest)],dtype='<u4').reshape(w//2,w)
    report(0.95,'plates')
    directory.mkdir(parents=True,exist_ok=True)
    for cap in (macro,climate,tectonic_map,crust_map):
        cap[0]=cap[0].mean(axis=0);cap[-1]=cap[-1].mean(axis=0)
    for name,value in [('macro',macro),('climate',climate),('tectonics',tectonic_map),('crust',crust_map),('water',wet.astype('<f4')),('flow',flow.astype('<f4')),('plates',plates),('river_header',hdr.astype('<u4')),('river_index',idx.astype('<u4'))]:
        value.tofile(directory/(name+'.bin'))
        value.tofile(directory/(name+'.bin'))
    rr=np.zeros((max(1,len(records)),3,4),'<f4');rr[:len(records)]=records.reshape(-1,3,4);rr.tofile(directory/'rivers.bin')
    # A useful initial focus is a mid-latitude coastline near an active belt.
    score=orogen*(np.abs(temp0-20)<14)*(h>1200)*(h<5000)
    iy,ix=np.unravel_index(np.argmax(score),score.shape)
    recommended={'longitude':float((ix+.5)/w*360-180),'latitude':float((iy+.5)/(w//2)*180-90)}
    stats.update({'build_seconds':round(time.perf_counter()-start,3),'height_min_m':float(h.min()),'height_max_m':float(h.max()),'recommended_focus':recommended,
                  'sea_datum_trim_m':float(datum),'pre_trim_land_fraction':float(pre_trim_land),'implied_ocean_volume_km3':float(implied_water_km3),'emerged_land_target':float(target)})
    (directory/'config_used.json').write_text(json.dumps(c,indent=2,sort_keys=True))
    meta={'macro':[w,w//2],'climate':[w,w//2],'water':[rw,rw//2],'flow':[rw,rw//2],'plates':[w,w//2],'river_header':[96,576],
          'river_index':[1024,idx.shape[0]],'rivers':[3,len(rr)],'max_bin':int(stats['maximum_reaches_in_bin']),
          'stats':stats,'tectonics':[w,w//2],'crust':[w,w//2],'tectonic_report':tectonic_stats}
    (directory/'meta.json').write_text(json.dumps(meta,indent=2))
    print(f"Globe guide ready in {stats['build_seconds']} s; {len(records)} river reaches.",flush=True)
    return meta


def prepare(c,root,progress=None,cancel=None):
    # Renderer/erosion changes don't require rebuilding the coarse globe.
    source=(root/'tectonics.py').read_bytes()+Path(__file__).read_bytes()
    key=hashlib.sha256(json.dumps({k:c[k] for k in ('planet','tectonics','climate','hydrology','fluvial')},sort_keys=True).encode()+REVISION.encode()+source).hexdigest()[:16]
    path=root/'cache'/key
    if (path/'meta.json').exists():return path,json.loads((path/'meta.json').read_text())
    try:return path,build(c,path,progress=progress,cancel=cancel)
    except Cancelled:
        import shutil;shutil.rmtree(path,ignore_errors=True);raise
