"""One global coarse geological/drainage guide for a finite spherical planet.

The mountain envelope is distance to converging spherical Voronoi BOUNDARIES.
This is a static kinematic construction, not Cortial et al.'s time evolution.
Priority-Flood provides an acyclic coarse receiver graph and spill-level lakes.
Fine ground is evaluated by the GPU, never materialised for the whole planet.
"""
from __future__ import annotations
import hashlib, heapq, json, math, time
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree, SphericalVoronoi

REVISION = 'sphere-3.0-3'


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


def boundary_fields(n, centres, omega, R, settings):
    """Sample CONNECTED Voronoi edges, not nearest-two plate masks. Interpolated
    edge data avoids discontinuous uplift where the second-nearest seed changes.
    """
    sv=SphericalVoronoi(centres);sv.sort_vertices_of_regions(); edges={}
    for i,region in enumerate(sv.regions):
        for a,b in zip(region,region[1:]+region[:1]):edges.setdefault(tuple(sorted((a,b))),[]).append(i)
    samples=[]; convergence=[]
    for (a,b),plates in edges.items():
        if len(plates)!=2:continue
        i,j=plates;A=sv.vertices[a];B=sv.vertices[b]
        angle=math.acos(float(np.clip(A@B,-1,1)))
        t=np.linspace(0,1,max(3,math.ceil(angle*R/35000)+1))
        q=unit(np.sin((1-t[:,None])*angle)*A+np.sin(t[:,None]*angle)*B)
        normal=unit(centres[j]-centres[i]);tangent=unit(normal-q*(q@normal)[:,None])
        conv=np.sum((np.cross(omega[i],q)-np.cross(omega[j],q))*tangent,axis=1)
        samples.extend(q);convergence.extend(conv)
    samples=np.asarray(samples);convergence=np.asarray(convergence)
    dist,near=cKDTree(samples).query(n.reshape(-1,3),k=4,workers=1)
    weights=1/np.maximum(dist,0.002)**2
    conv=np.sum(weights*convergence[near],axis=1)/weights.sum(axis=1)
    d=2*np.arcsin(np.clip(dist[:,0]/2,0,1))*R
    return d.reshape(n.shape[:-1]),conv.reshape(n.shape[:-1])


def build(c, directory):
    start=time.perf_counter();p=c['planet'];t=c['tectonics'];cl=c['climate'];R=p['radius_metres'];seed=p['seed']
    w=int(p['map_width']);n,lat=grid(w);points=n.reshape(-1,3)
    print('Generating spherical continents and boundary-following mountain belts…',flush=True)
    base=np.zeros(n.shape[:2]);pos=n*R
    continent_pos=warped(n,R,seed+366,540000,3800000)*R
    for i,(wav,amp) in enumerate(zip(p['continent_wavelength_metres'],p['continent_amplitude'])):
        base+=amp*noise3(continent_pos/wav+np.array([i*17.17,5.71*i,-19.31*i]),seed+19*i)
    weights=np.broadcast_to(np.cos(lat)[:,None],base.shape).ravel()
    sort=np.argsort(base.ravel());cdf=np.cumsum(weights[sort]);threshold=base.ravel()[sort[np.searchsorted(cdf,cdf[-1]*(1-p['land_fraction']))]]
    crust=base-threshold
    # Positive continental crust gives lowlands, not mountains everywhere.
    land=crust>0;coast=crust*R/2.4
    h=np.where(land,p['lowland_height_metres']*(1-np.exp(-crust*4)),
               -p['abyss_depth_metres']*(1-np.exp(coast/p['shelf_width_metres'])))
    centres,omega=spherical_plates(int(t['plate_count']),seed)
    q=warped(n,R,seed+789,t['boundary_warp_metres'],t['boundary_warp_wavelength_metres'])
    d,convergence=boundary_fields(q,centres,omega,R,t)
    strength=np.clip((convergence-t['convergence_threshold'])/.65,0,1)
    along=.68+.45*noise3(pos/t['range_along_variation_metres']+11,seed+942)
    belt=np.zeros_like(h)
    for width,height in zip(t['belt_width_metres'],t['belt_height_metres']):
        belt+=height*np.exp(-(d/width)**2)
    # Only the continental side becomes a large orogen. Ocean-ocean arcs get a
    # smaller lift and normally remain submerged. No uplift of whole provinces.
    landfade=np.clip(coast/60000,0,1);landfade=landfade*landfade*(3-2*landfade)
    uplift=belt*strength*along*landfade
    orogen=np.clip(uplift/3300,0,1)
    h+=uplift
    divergence=np.clip((-convergence-.1)/.7,0,1).reshape(base.shape)
    h+=t['ridge_height_metres']*divergence*np.exp(-(d/170000)**2)*(~land)
    h-=t['trench_depth_metres']*strength*np.exp(-(d/70000)**2)*(~land)
    # Fold-parallel macro relief gives the coarse overview a structural grain.
    folded=(.55+.45*noise3(pos/190000+31,seed+887))*np.cos(d/17000+noise3(pos/280000,seed+345)*4)
    h+=orogen*folded*160
    temp0=cl['equator_temperature_c']-cl['pole_temperature_drop_c']*np.abs(n[...,1])**1.25
    temp0+=cl['regional_temperature_variation_c']*noise3(pos/3200000,seed+977)
    latdeg=np.abs(np.degrees(lat[:,None])); rain=np.log(cl['mean_rainfall_mm'])+1.0*np.exp(-(latdeg/12)**2)-1.7*np.exp(-((latdeg-29)/11)**2)+.2*np.exp(-((latdeg-53)/15)**2)
    rain=np.broadcast_to(rain,base.shape).copy()+cl['regional_rainfall_variation']*noise3(pos/1500000+27,seed+744)
    # Prevailing winds reverse between Hadley/Ferrel cells. One upwind sample is
    # a rain-shadow proxy; not fluid dynamics or time-dependent meteorology.
    wind=np.where((latdeg>32)&(latdeg<65),1,-1)
    upwind=(np.roll(h,8,axis=1)*((wind>0).astype(float))+np.roll(h,-8,axis=1)*((wind<0).astype(float)))
    rain-=cl['rain_shadow_strength']*np.clip((upwind-h)/2500,0,2)
    rain=np.clip(rain,np.log(40),np.log(6000))
    lith=.5+.5*noise3(pos/270000+47,seed+918)
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
        records=np.zeros((0,12),np.float32);hdr=np.zeros((64,128,2),np.uint32);idx=np.zeros((1,1024),np.uint32)
        wet=np.zeros((rw//2,rw,4));wet[...,1]=-1e5;stats={'river_reaches':0,'maximum_reaches_in_bin':0}
    macro=np.stack((h/1000,orogen,coast/1000,lith),axis=-1).astype('<f4')
    climate=np.stack((temp0,rain,strength,divergence),axis=-1).astype('<f4')
    directory.mkdir(parents=True,exist_ok=True)
    for cap in (macro,climate):
        cap[0]=cap[0].mean(axis=0);cap[-1]=cap[-1].mean(axis=0)
    for name,value in [('macro',macro),('climate',climate),('water',wet.astype('<f4')),('river_header',hdr.astype('<u4')),('river_index',idx.astype('<u4'))]:
        value.tofile(directory/(name+'.bin'))
    rr=np.zeros((max(1,len(records)),3,4),'<f4');rr[:len(records)]=records.reshape(-1,3,4);rr.tofile(directory/'rivers.bin')
    # A useful initial focus is a mid-latitude coastline near an active belt.
    score=orogen*(np.abs(temp0-20)<14)*(h>1200)*(h<5000)
    iy,ix=np.unravel_index(np.argmax(score),score.shape)
    recommended={'longitude':float((ix+.5)/w*360-180),'latitude':float((iy+.5)/(w//2)*180-90)}
    stats.update({'build_seconds':round(time.perf_counter()-start,3),'height_min_m':float(h.min()),'height_max_m':float(h.max()),'recommended_focus':recommended})
    meta={'macro':[w,w//2],'climate':[w,w//2],'water':[rw,rw//2],'river_header':[64,384],
          'river_index':[1024,idx.shape[0]],'rivers':[3,len(rr)],'max_bin':int(stats['maximum_reaches_in_bin']),
          'stats':stats,'plates':centres.tolist(),'omega':omega.tolist()}
    (directory/'meta.json').write_text(json.dumps(meta,indent=2))
    print(f"Globe guide ready in {stats['build_seconds']} s; {len(records)} river reaches.",flush=True)
    return meta


def prepare(c,root):
    # Renderer/erosion changes don't require rebuilding the coarse globe.
    key=hashlib.sha256(json.dumps({k:c[k] for k in ('planet','tectonics','climate','hydrology')},sort_keys=True).encode()+REVISION.encode()).hexdigest()[:16]
    path=root/'cache'/key
    if (path/'meta.json').exists():return path,json.loads((path/'meta.json').read_text())
    return path,build(c,path)
