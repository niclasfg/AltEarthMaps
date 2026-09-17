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

REVISION = 'sphere-4.0-tectonic-1'


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


def warped(n, R, seed, amount, wavelength):
    p=n*(R/wavelength)
    d=np.stack([noise3(p+np.array([j*13.19, j*31.17, j*7.19]),seed+91*j) for j in range(3)],axis=-1)
    return unit(n+d*(amount/R))


def spherical_plates(count, seed):
    rng=np.random.default_rng(seed+11)
    # Quasi-uniform initial seeds followed by jitter; no gridded island domains.
    k=np.arange(count)+.5; y=1-2*k/count; a=k*np.pi*(3-np.sqrt(5))
    centres=unit(np.stack((np.sqrt(1-y*y)*np.cos(a), y, np.sqrt(1-y*y)*np.sin(a)),axis=-1)+rng.normal(0,.13,(count,3)))
    # Every plate is a rigid Euler rotation; tangential velocity is omega cross n.
    omega=rng.normal(size=(count,3))*.55
    return centres,omega


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


def river_records(h, spill, rec, order, area, n, c):
    """Shared junction positions ensure every rendered tributary meets its parent.
    A mild main-stem relaxation removes the raster's 45-degree staircase.
    Heights remain monotone from source to outlet. Water width is in metres.
    """
    hc=c['hydrology'];R=c['planet']['radius_metres'];raw=h.ravel();p=n.reshape(-1,3).copy()
    selected=(area>hc['river_source_area_km2'])&(raw>0)&(rec!=np.arange(len(rec)))
    indices=np.flatnonzero(selected)
    donor=np.full(len(rec),-1,np.int32);donor_area=np.zeros(len(rec))
    for i in indices:
        j=rec[i]
        if area[i]>donor_area[j]:donor[j]=i;donor_area[j]=area[i]
    old=p.copy()
    for i in indices:
        if donor[i]>=0:
            p[i]=unit(.6*old[i]+.2*old[donor[i]]+.2*old[rec[i]])
    bed=spill.ravel().copy()
    # Enforce strictly descending longitudinal profiles on selected branches;
    # downstream-first traversal keeps confluences shared.
    for i in order:
        j=rec[i]
        if selected[i]:bed[i]=max(bed[i],bed[j]+.015)
    width=np.minimum(hc['maximum_half_width_metres'],hc['width_coefficient_metres']*np.sqrt(area))
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
                            hc['valley_width_metres']/1000,area[i]])
    return np.asarray(records,np.float32).reshape(-1,12)


def river_index(records, R, nx=64,ny=64):
    """Cubed-sphere bins; bounded size even at poles. Each reach's support is
    conservatively duplicated into bins on either side of every face boundary.
    These bins index geometry and NEVER define terrain or river boundaries.
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


def build(c, directory):
    start=time.perf_counter();p=c['planet'];t=c['tectonics'];cl=c['climate'];R=p['radius_metres'];seed=p['seed']
    w=int(p['map_width']);n,lat=grid(w);points=n.reshape(-1,3)
    from tectonics import simulate
    print('Evolving crust and tectonic history before generating climate and drainage...',flush=True)
    def initial_field(directions):
        cp=warped(directions,R,seed+366,540000,3800000)*R
        value=np.zeros(directions.shape[:-1])
        for i,(wav,amp) in enumerate(zip(p['continent_wavelength_metres'],p['continent_amplitude'])):
            value+=amp*noise3(cp/wav+np.array([i*17.17,5.71*i,-19.31*i]),seed+19*i)
        return value
    mesh,fields,converging,spreading,tectonic_stats=simulate(c,initial_field,directory)
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
    # Signed coast distance is a guide attribute, not a randomly placed shore.
    # Approximation on the global lat/lon guide; tectonic transport itself has
    # no latitude-grid distortion or special polar seam.
    coast=(ndimage.distance_transform_edt(land)-ndimage.distance_transform_edt(~land))*(2*np.pi*R/w)
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
    # Global hydrology uses a cheaper grid but remains topologically global.
    rw=int(c['hydrology']['routing_width']);step=w//rw
    hsmall=ndimage.zoom(h,1/step,order=1,prefilter=False) if step>1 else h.copy()
    ns,lats=grid(rw)
    if c['hydrology']['enabled']:
        print('Routing global watersheds, lakes and major rivers…',flush=True)
        fill,rec,order,area=flood(hsmall,lats,R)
        records=river_records(hsmall,fill,rec,order,area,ns,c)
        depth=fill-hsmall
        lake=(depth>c['hydrology']['lake_minimum_depth_metres'])&(fill>10)&(depth<c['hydrology']['lake_maximum_depth_metres'])
        # Interpolate spill surfaces only inside a connected basin. Store a
        # signed lake shore field rather than interpolating a missing sentinel.
        signed=(ndimage.distance_transform_edt(lake)-ndimage.distance_transform_edt(~lake))*2*np.pi*R/rw
        wet=np.stack((fill/1000,signed/1000,np.log1p(area.reshape(hsmall.shape)),depth/1000),axis=-1)
        hdr,idx,maxbin=river_index(records,R)
        stats={'river_reaches':len(records),'maximum_reaches_in_bin':int(maxbin),'lakes_grid_cells':int(lake.sum()),
               'uphill_coarse_receivers':int(np.sum(fill.ravel()[rec]>fill.ravel()+1e-7)),
               'land_fraction':float(np.sum(weights*land.ravel())/weights.sum())}
    else:
        records=np.zeros((0,12),np.float32);hdr=np.zeros((384,64,2),np.uint32);idx=np.zeros((1,1024),np.uint32)
        wet=np.zeros((rw//2,rw,4));wet[...,1]=-1e5;stats={'river_reaches':0,'maximum_reaches_in_bin':0}
    macro=np.stack((h/1000,orogen,coast/1000,lith),axis=-1).astype('<f4')
    climate=np.stack((temp0,rain,strength,divergence),axis=-1).astype('<f4')
    directory.mkdir(parents=True,exist_ok=True)
    for cap in (macro,climate,tectonic_map,crust_map):
        cap[0]=cap[0].mean(axis=0);cap[-1]=cap[-1].mean(axis=0)
    for name,value in [('macro',macro),('climate',climate),('tectonics',tectonic_map),('crust',crust_map),('water',wet.astype('<f4')),('river_header',hdr.astype('<u4')),('river_index',idx.astype('<u4'))]:
        value.tofile(directory/(name+'.bin'))
    rr=np.zeros((max(1,len(records)),3,4),'<f4');rr[:len(records)]=records.reshape(-1,3,4);rr.tofile(directory/'rivers.bin')
    # A useful initial focus is a mid-latitude coastline near an active belt.
    score=orogen*(np.abs(temp0-20)<14)*(h>1200)*(h<5000)
    iy,ix=np.unravel_index(np.argmax(score),score.shape)
    recommended={'longitude':float((ix+.5)/w*360-180),'latitude':float((iy+.5)/(w//2)*180-90)}
    stats.update({'build_seconds':round(time.perf_counter()-start,3),'height_min_m':float(h.min()),'height_max_m':float(h.max()),'recommended_focus':recommended})
    meta={'macro':[w,w//2],'climate':[w,w//2],'water':[rw,rw//2],'river_header':[64,384],
          'river_index':[1024,idx.shape[0]],'rivers':[3,len(rr)],'max_bin':int(stats['maximum_reaches_in_bin']),
          'stats':stats,'tectonics':[w,w//2],'crust':[w,w//2],'tectonic_report':tectonic_stats}
    (directory/'meta.json').write_text(json.dumps(meta,indent=2))
    print(f"Globe guide ready in {stats['build_seconds']} s; {len(records)} river reaches.",flush=True)
    return meta


def prepare(c,root):
    # Renderer/erosion changes don't require rebuilding the coarse globe.
    source=(root/'tectonics.py').read_bytes()+Path(__file__).read_bytes()
    key=hashlib.sha256(json.dumps({k:c[k] for k in ('planet','tectonics','climate','hydrology')},sort_keys=True).encode()+REVISION.encode()+source).hexdigest()[:16]
    path=root/'cache'/key
    if (path/'meta.json').exists():return path,json.loads((path/'meta.json').read_text())
    return path,build(c,path)
