"""Kinematic, time-dependent lithosphere on a spherical finite-volume mesh.

Not mantle convection, and not an Earth reconstruction. Plate velocities and
rift times are prescribed/seeded. Crust, age and strain are transported, rather
than re-evaluated from a new Voronoi diagram every step. See METHOD.md for the
numerical method, units, source references, budgets and modelling limitations.

Units here: km, Myr, km^2, km^3. 1 cm/yr = 10 km/Myr.
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull, SphericalVoronoi, cKDTree
from scipy import sparse
from numba import njit

# Surface-area densities for each independently moving plate. C+O is coverage.
C, O, VC, OA, COMP, EXT, SHEAR, ARC = range(8)
FIELDS = ['continental_area_fraction', 'oceanic_area_fraction',
          'continental_crust_volume_per_area_km', 'oceanic_age_moment_myr',
          'compression_memory', 'extension_memory', 'shear_memory',
          'arc_crust_volume_per_area_km']
REVISION = 'finite-volume-tectonics-4-relaxation'


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-30)


def icosphere(level):
    g=(1+math.sqrt(5))/2
    p=unit(np.array([(0,a,b*g) for a in (-1,1) for b in (-1,1)] +
                    [(a,b*g,0) for a in (-1,1) for b in (-1,1)] +
                    [(b*g,0,a) for a in (-1,1) for b in (-1,1)],float))
    tri=ConvexHull(p).simplices
    for _ in range(level):
        points=list(p); mid={}; out=[]
        def midpoint(a,b):
            key=(min(a,b),max(a,b))
            if key not in mid:
                mid[key]=len(points);points.append(unit(p[a]+p[b]))
            return mid[key]
        for a,b,c in tri:
            ab=midpoint(a,b);bc=midpoint(b,c);ca=midpoint(c,a)
            out.extend([(a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)])
        p=np.array(points);tri=np.asarray(out,np.int32)
    return p


class SphereMesh:
    """Voronoi control volumes dual to an icosphere. No polar/seam special cases."""
    def __init__(self,level,radius_km):
        self.p=icosphere(level);self.R=radius_km;self.tree=cKDTree(self.p)
        hull=ConvexHull(self.p);tri=hull.simplices
        dual=unit(hull.equations[:,:3])
        edge_faces={}
        for f,t in enumerate(tri):
            for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
                edge_faces.setdefault((min(a,b),max(a,b)),[]).append(f)
        self.edges=np.array(list(edge_faces),np.int32)
        faces=np.array(list(edge_faces.values()),np.int32)
        a,b=self.edges.T;mid=unit(self.p[a]+self.p[b]);outward=unit(self.p[b]-self.p[a])
        dv=dual[faces[:,1]]-dual[faces[:,0]]
        # Integral of normal velocity along a spherical dual edge is exactly
        # R^2 * omega.dot(dual_end - dual_start), with consistent orientation.
        # It telescopes around every polygon: solid rotation has zero divergence.
        dv*=np.where(np.sum(dv*np.cross(mid,outward),axis=1)>=0,1.,-1.)[:,None]
        self.flux_coeff=dv*radius_km**2
        self.length=np.arccos(np.clip(np.sum(dual[faces[:,0]]*dual[faces[:,1]],axis=1),-1,1))*radius_km
        self.primal_length=np.arccos(np.clip(np.sum(self.p[a]*self.p[b],axis=1),-1,1))*radius_km
        self.mid,self.outward=mid,outward
        self.area=SphericalVoronoi(self.p).calculate_areas()*radius_km**2
        self.n=len(self.p);self.spacing=math.sqrt(4*math.pi*radius_km**2/self.n)
        neighbours=[[] for _ in range(self.n)]
        for i,j in self.edges:neighbours[i].append(j);neighbours[j].append(i)
        self.upstream=np.empty((len(a),2),np.int32);self.backscale=np.empty((len(a),2))
        for e,(i,j) in enumerate(self.edges):
            for side,(s,d) in enumerate(((i,j),(j,i))):
                f=self.p[d]-self.p[s];f-=self.p[s]*np.dot(f,self.p[s]);f=unit(f)
                candidates=np.array(neighbours[s]);v=self.p[candidates]-self.p[s]
                projection=v@f;k=np.argmin(projection)
                self.upstream[e,side]=candidates[k]
                self.backscale[e,side]=np.linalg.norm(self.p[d]-self.p[s])/max(-projection[k],1e-12)
        row=np.r_[a,b];col=np.r_[b,a]
        w=np.r_[self.length/self.primal_length,self.length/self.primal_length]
        self.adjacency=sparse.csr_matrix((w,(row,col)),shape=(self.n,self.n))
        self.degree=np.asarray(self.adjacency.sum(axis=1)).ravel()

    def smooth(self,a,iterations=3):
        """Local spherical smoothing for masks; never transports plate ownership."""
        out=a.copy()
        for _ in range(iterations):
            out=.55*out+.45*(self.adjacency@out)/self.degree.reshape((-1,)+(1,)*(out.ndim-1))
        return out

    def sample(self,points,values):
        """Continuous inverse-distance interpolation of coarse simulation outputs.
        Eight nearest samples, blended continuously with compact support at k=8.
        No claim that this invents finer-scale tectonic information.
        """
        dst,idx=self.tree.query(points,k=8,workers=1)
        radius=np.maximum(dst[:,-1:],1e-12)
        w=np.maximum(0,1-dst/radius)**2/np.maximum(dst,1e-9)**2
        w/=w.sum(axis=1,keepdims=True)
        return np.einsum('ij,ijk->ik',w,values[idx])


@njit(cache=True)
def _minmod(a,b,c):
    if a>0 and b>0 and c>0:return min(a,b,c)
    if a<0 and b<0 and c<0:return max(a,b,c)
    return 0.


@njit(cache=True)
def _euler(u,flux,edges,upstream,backscale,area,dt,active):
    """Conservative MUSCL flux with a monotonized-central limiter.
    Equal/opposite volume crosses each dual edge. The half-edge extrapolation
    uses an approximately collinear upstream neighbour on the unstructured mesh.
    CFL <= .45 keeps all reconstructed nonnegative densities positive.
    """
    out=u.copy()
    for p in range(u.shape[0]):
        if not active[p]:continue
        for e in range(len(edges)):
            f=flux[p,e]
            if f==0.:continue
            if f>0:s,d=edges[e,0],edges[e,1];side=0
            else:s,d=edges[e,1],edges[e,0];side=1;f=-f
            b=upstream[e,side];r=backscale[e,side]
            ds=dt*f/area[s];dd=dt*f/area[d]
            # Reconstruct the area carriers. Transport extensive tracers with
            # those SAME mass fluxes, not independent reconstructions. Otherwise
            # a cell may retain crust volume with zero continental area, or age
            # with zero ocean area. Tracer concentrations use donor-cell values:
            # first order for tracers, second order for the area carriers.
            faces=np.empty(2)
            for k in (C,O):
                val=u[p,s,k]
                forward=u[p,d,k]-val;backward=(val-u[p,b,k])*r
                face=max(0.,val+.5*_minmod(2*backward,.5*(backward+forward),2*forward))
                faces[k]=face
                out[p,s,k]-=face*ds;out[p,d,k]+=face*dd
            for k in (VC,OA,COMP,EXT,SHEAR,ARC):
                if k==VC:carrier=u[p,s,C];face=faces[C]
                elif k==OA:carrier=u[p,s,O];face=faces[O]
                else:carrier=u[p,s,C]+u[p,s,O];face=faces[C]+faces[O]
                amount=face*u[p,s,k]/carrier if carrier>1e-30 else 0.
                out[p,s,k]-=amount*ds;out[p,d,k]+=amount*dd
    return out


@njit(cache=True)
def advect(u,flux,edges,upstream,backscale,area,dt,active):
    # SSP-RK2: two conservative stages, second-order in time for fixed velocities.
    a=_euler(u,flux,edges,upstream,backscale,area,dt,active)
    b=_euler(a,flux,edges,upstream,backscale,area,dt,active)
    return .5*(u+b)


@njit(cache=True)
def courant_limit(flux,edges,area,active,cfl):
    worst=0.
    for p in range(len(active)):
        if not active[p]:continue
        outgoing=np.zeros(len(area))
        for e in range(len(edges)):
            f=flux[p,e];outgoing[edges[e,0] if f>0 else edges[e,1]]+=abs(f)
        for i in range(len(area)):worst=max(worst,outgoing[i]/area[i])
    return cfl/max(worst,1e-20)


@njit(cache=True)
def interact(u,area,dt,settings,contact):
    """Resolve superposed plate coverages. Account for every crust-volume sink.
    Oceanic material is removed first (oldest first). Continental volume is NOT
    subducted: consumed area transfers its volume to the overriding continent.
    Uncovered area stretches a still-thick continental margin or creates new
    zero-age ocean floor. These are reduced-order lithosphere interaction rules.
    """
    nplate,N,_=u.shape
    min_rift,base_thick,root_tau,arc_fraction,arc_tau,comp_tau,ext_tau,shear_tau,root_power=settings
    budgets=np.zeros(6) # created ocean area, removed ocean area, eroded C vol,
                        # new arc vol, subducted arc vol, eroded arc vol
    for i in range(N):
        # Clipping numerical roundoff only; appreciable negative densities are
        # checked before this kernel by the caller.
        for p in range(nplate):
            for k in range(8):
                if u[p,i,k]<0.:u[p,i,k]=0.
            u[p,i,OA]+=u[p,i,O]*dt
            u[p,i,COMP]*=math.exp(-dt/comp_tau)
            u[p,i,EXT]*=math.exp(-dt/ext_tau)
            u[p,i,SHEAR]*=math.exp(-dt/shear_tau)
        total=0.
        for p in range(nplate):total+=u[p,i,C]+u[p,i,O]
        if total>1.+1e-13:
            excess=total-1.
            # Subduct dense oceanic lithosphere before buoyant continental crust.
            while excess>1e-13:
                loser=-1;oldest=-1e30
                for p in range(nplate):
                    if u[p,i,O]>1e-13:
                        score=u[p,i,OA]/u[p,i,O]+p*1e-8
                        if score>oldest:oldest=score;loser=p
                if loser<0:break
                removed=min(excess,u[loser,i,O]);old_o=u[loser,i,O]
                old_f=old_o+u[loser,i,C]
                u[loser,i,OA]*=1.-removed/old_o;u[loser,i,O]-=removed
                basalt=u[loser,i,ARC]*removed/max(old_f,1e-20)
                u[loser,i,ARC]-=basalt;budgets[4]+=basalt*area[i]
                # Strain memory belongs to surviving surface crust. It must not
                # remain at a location after its oceanic carrier is subducted.
                for k in (COMP,EXT,SHEAR):
                    u[loser,i,k]*=1.-removed/max(old_f,1e-20)
                budgets[1]+=removed*area[i];excess-=removed
                upper=-1;buoy=-1e30
                for p in range(nplate):
                    f=u[p,i,C]+u[p,i,O]
                    if p!=loser and f>1e-12:
                        age=u[p,i,OA]/max(u[p,i,O],1e-20)
                        # Area-weighted buoyancy priority. An infinitesimal
                        # mixed-cell continental trace cannot capture all magma.
                        score=2.*u[p,i,C]+u[p,i,O]/(1.+age/80.)
                        if score>buoy:buoy=score;upper=p
                if upper<0:
                    for p in range(nplate):
                        if u[p,i,C]+u[p,i,O]>1e-12:upper=p;break
                if upper>=0:
                    magma=removed*7.*arc_fraction
                    u[upper,i,ARC]+=magma;budgets[3]+=magma*area[i]
                    u[upper,i,COMP]+=.45*removed
            # Continental collision removes horizontal area but retains volume.
            while excess>1e-13:
                loser=-1;least=1e30;upper=-1;most=-1e30
                # With one continental density there is no resolved polarity
                # law. Use the dominant local continental carrier as the upper
                # plate, not a vanishing sliver with an enormous ratio V/C.
                for p in range(nplate):
                    if u[p,i,C]>most and u[p,i,C]>1e-13:
                        most=u[p,i,C];upper=p
                for p in range(nplate):
                    if p!=upper and u[p,i,C]>1e-13:
                        h=u[p,i,VC]/u[p,i,C]
                        if h+p*1e-9<least:least=h+p*1e-9;loser=p
                if loser<0:loser=upper
                if loser<0:break
                if loser==upper:
                    # Same plate has compressed material under its own local
                    # discretization. Preserve volume while reducing coverage.
                    u[loser,i,C]-=excess;u[loser,i,COMP]+=excess;excess=0.;break
                removed=min(excess,u[loser,i,C]);ratio=removed/u[loser,i,C]
                volume=u[loser,i,VC]*ratio
                u[loser,i,C]-=removed;u[loser,i,VC]-=volume;u[upper,i,VC]+=volume
                u[upper,i,COMP]+=removed
                for k in (COMP,EXT,SHEAR,ARC):
                    moved=u[loser,i,k]*ratio;u[loser,i,k]-=moved;u[upper,i,k]+=moved
                contact[loser,upper]+=removed*area[i];excess-=removed
        elif total<1.-1e-13:
            gap=1.-total
            # dt is limited so an interior cell cannot become completely empty.
            for p in range(nplate):
                f=u[p,i,C]+u[p,i,O]
                if f<=1e-14:continue
                new=gap*f/max(total,1e-20)
                h=u[p,i,VC]/max(u[p,i,C],1e-14)
                stretch=new*u[p,i,C]/f if h>min_rift else 0.
                u[p,i,C]+=stretch;u[p,i,O]+=new-stretch
                u[p,i,EXT]+=new
                budgets[0]+=(new-stretch)*area[i]
        # Long-term erosion/collapse: remove excess root volume into an explicit
        # unresolved sediment reservoir. It is NOT silent mass loss or smoothing
        # that creates crust. Isostatic elevation is evaluated from surviving VC.
        for p in range(nplate):
            excess=max(0.,u[p,i,VC]-base_thick*u[p,i,C])
            thickness=u[p,i,VC]/max(u[p,i,C],1e-20)
            rate=(max(1.,thickness/base_thick)**root_power)/root_tau
            loss=excess*(-math.expm1(-dt*rate))
            u[p,i,VC]-=loss;budgets[2]+=loss*area[i]
            loss=u[p,i,ARC]*(-math.expm1(-dt/arc_tau))
            u[p,i,ARC]-=loss;budgets[5]+=loss*area[i]
    return budgets


@njit(cache=True)
def spread_roots(u,edges,conductance,area,dt,diffusivity):
    """Conservative pairwise gravitational exchange; symmetric edge splitting.
    For each pair, integrate the frozen-coefficient thickness diffusion EXACTLY:
      d(hi-hj)/dt = -K * (1/mi + 1/mj) * (hi-hj).
    Transfer the corresponding equal/opposite volume. Each pair is a convex
    mixing, even when the receiving carrier is tiny. Forward and reverse
    half-sweeps reduce order bias. This is an operator-split relaxation, not
    a full thin-viscous-sheet solution; coefficient/time splitting remains.
    """
    out=u.copy()
    for p in range(u.shape[0]):
        for sweep in range(2):
            for step in range(len(edges)):
                e=step if sweep==0 else len(edges)-1-step
                i,j=edges[e]
                for k in (VC,ARC):
                    fi=out[p,i,C] if k==VC else out[p,i,C]+out[p,i,O]
                    fj=out[p,j,C] if k==VC else out[p,j,C]+out[p,j,O]
                    if fi<1e-8 or fj<1e-8:continue
                    hi=out[p,i,k]/fi;hj=out[p,j,k]/fj
                    mobility=min(16.,max(1.,max(hi,hj)/35.)**3) if k==VC else 1.
                    mi=fi*area[i];mj=fj*area[j]
                    inverse=1./mi+1./mj
                    rate=diffusivity*mobility*conductance[e]*min(fi,fj)*inverse
                    amount=(hi-hj)*(-math.expm1(-rate*.5*dt))/inverse
                    out[p,i,k]-=amount/area[i];out[p,j,k]+=amount/area[j]
    return out


def plate_cooling_depth(age,t):
    """Isostatic plate-cooling Fourier solution (not an age-independent abyss).
    Finite terms are normalized so depth(0) equals ridge_depth exactly. The
    exponential time scale is L^2/(pi^2*kappa), with consistent km/Myr units.
    """
    tau=t['thermal_plate_thickness_km']**2/(math.pi**2*t['thermal_diffusivity_km2_per_myr'])
    total=np.zeros_like(age,dtype=float);zero=0.
    for k in range(1,64,2):
        w=1./k**2;zero+=w;total+=w*np.exp(-k*k*np.maximum(age,0)/tau)
    cool=1-total/zero
    return t['ridge_depth_metres']/1000+t['thermal_subsidence_metres']/1000*cool


def state_fields(u,t):
    c=u[:,:,C].sum(0);o=u[:,:,O].sum(0)
    thick=np.divide(u[:,:,VC].sum(0),c,out=np.zeros_like(c),where=c>1e-8)
    age=np.divide(u[:,:,OA].sum(0),o,out=np.zeros_like(o),where=o>1e-8)
    comp=1-np.exp(-u[:,:,COMP].sum(0)/t['compression_mask_strain'])
    ext=1-np.exp(-u[:,:,EXT].sum(0)/t['extension_mask_strain'])
    shear=1-np.exp(-u[:,:,SHEAR].sum(0)/t['shear_mask_strain'])
    rho_m=t['mantle_density_kg_m3'];rho_c=t['continental_density_kg_m3']
    gain=1-rho_c/rho_m
    ch=t['reference_continent_elevation_metres']/1000+gain*(thick-t['reference_crust_thickness_km'])
    # Unloaded freeboard relative to a common mantle datum. The global sea-level
    # solve below adds the water load consistently rather than forcing a target
    # land fraction after evolution. Ocean plate depths are calibrated at datum 0.
    ratio=t['water_density_kg_m3']/rho_m
    ocean=-(1-ratio)*plate_cooling_depth(age,t)
    basalt=u[:,:,ARC].sum(0)
    igneous=basalt*(1-t['oceanic_density_kg_m3']/rho_m)
    # Use volume directly for buoyancy, including tiny mixed coast fractions.
    h=c*t['reference_continent_elevation_metres']/1000+gain*(u[:,:,VC].sum(0)-t['reference_crust_thickness_km']*c)+o*ocean+igneous
    young=np.clip(np.maximum(thick-t['reference_crust_thickness_km'],0)/30,0,1)
    rug=np.clip(.7*young+.4*comp+.18*shear,0,1)*np.clip(c*1.5,0,1)
    return np.stack([h,rug,c,thick,age,comp,ext,shear,igneous],axis=-1)


def apply_ocean_load(values,area,volume_km3,t):
    """A fixed water inventory, not a prescribed final land percentage.
    For unloaded freeboard b and sea datum L, submerged height z obeys
    z=b-rho_water/rho_mantle*(L-z). Hence water depth=(L-b)/(1-ratio).
    """
    b=values[:,0];r=t['water_density_kg_m3']/t['mantle_density_kg_m3']
    lo=float(b.min()-1);hi=float(b.max()+volume_km3/area.sum()+1)
    for _ in range(56):
        mid=(lo+hi)*.5;volume=float(np.sum(np.maximum(mid-b,0)*area)/(1-r))
        if volume<volume_km3:lo=mid
        else:hi=mid
    sea=(lo+hi)*.5
    out=values.copy();out[:,0]=np.where(b>=sea,b-sea,(b-sea)/(1-r))
    return out,sea


def active_boundaries(mesh,u,omega,t):
    """Present-day signed normal and tangential relative velocities, km/Myr.
    Classification is at edges between DIFFERENT plate owners. Transforms do
    not acquire collision uplift simply because their speed is large.
    """
    a,b=mesh.edges.T;owner=np.argmax(u[:,:,C]+u[:,:,O],axis=0)
    valid=owner[a]!=owner[b]
    rel=np.cross(omega[owner[a]]-omega[owner[b]],mesh.mid)*mesh.R
    normal=np.sum(rel*mesh.outward,axis=1)
    tangent=np.cross(mesh.mid,mesh.outward);slide=np.abs(np.sum(rel*tangent,axis=1))
    rate=np.stack((np.maximum(normal,0),np.maximum(-normal,0),slide),axis=-1)*valid[:,None]
    value=np.zeros((mesh.n,3));den=np.zeros(mesh.n)
    for k in range(3):
        np.add.at(value[:,k],a,rate[:,k]*mesh.length)
        np.add.at(value[:,k],b,rate[:,k]*mesh.length)
    np.add.at(den,a,mesh.length*valid);np.add.at(den,b,mesh.length*valid)
    value/=np.maximum(den[:,None],1e-20)
    return mesh.smooth(value,2),owner,valid,normal,slide


def _vector_setting(t,name,size):
    v=np.asarray(t[name],float)
    if len(v)==1:return np.full(size,v[0])
    if len(v)!=size:raise ValueError(f'tectonics.{name}: one value or {size} values required')
    return v.copy()


def simulate(config,initial_crust,directory,progress=print):
    """Forward evolution of a synthetic 1-Ga-old state to synthetic present day."""
    t=config['tectonics'];R=config['planet']['radius_metres']/1000;start=time.perf_counter()
    mesh=SphereMesh(int(t['mesh_subdivisions']),R);seed=int(config['planet']['seed']);rng=np.random.default_rng(seed+51883)
    count=int(t['plate_count']);P=count+len(t['rift_ages_myr'])+4
    idx=np.arange(count)+.5;y=1-2*idx/count;angle=idx*math.pi*(3-math.sqrt(5))
    centres=unit(np.c_[np.sqrt(1-y*y)*np.cos(angle),y,np.sqrt(1-y*y)*np.sin(angle)]+rng.normal(0,.12,(count,3)))
    # Voronoi ONLY INITIALIZES plates; future labels are transported fractions.
    labels=np.argmax(mesh.p@centres.T,axis=1)
    # Continental crust comes from grown landmass regions, never a noise field.
    # Margins are already blended over a few mesh cells by the growth step.
    cont=np.clip(np.asarray(initial_crust(mesh),dtype=float).ravel(),0,1)
    u=np.zeros((P,mesh.n,8),float)
    thickness=_vector_setting(t,'initial_crust_thickness_km',count)
    speeds=_vector_setting(t,'plate_speed_cm_per_year',count)
    poles=unit(rng.normal(size=(P,3)));omega=np.zeros((P,3));target=np.zeros_like(omega)
    active=np.zeros(P,np.bool_);active[:count]=True
    for p in range(count):
        owned=labels==p;u[p,owned,C]=cont[owned];u[p,owned,O]=1-cont[owned]
        u[p,owned,VC]=cont[owned]*thickness[p]
        u[p,owned,OA]=(1-cont[owned])*(t['initial_ocean_age_myr'][0]+(t['initial_ocean_age_myr'][1]-t['initial_ocean_age_myr'][0])*(.5+.5*np.sin(mesh.p[owned]@poles[p]*7)))
        omega[p]=poles[p]*speeds[p]*10/R
    drive_speed=np.linalg.norm(omega,axis=1)
    target[:]=omega
    # A common whole-globe rotation is a frame choice, not tectonics.
    weights=np.bincount(labels,weights=mesh.area,minlength=P)
    spin=np.sum(omega*weights[:,None],axis=0)/weights.sum();omega[active]-=spin;target[:]=omega
    duration=float(t['duration_myr']);elapsed=0.;steps=0;next_report=0.;last_motion=0.;next_epoch=float(t['motion_epoch_myr'])
    events=[];budgets=np.zeros(6);contact=np.zeros((P,P));history=[];snaps={};next_snapshot=0.
    settings=np.array([t['breakup_crust_thickness_km'],t['reference_crust_thickness_km'],t['root_relaxation_myr'],t['arc_accretion_fraction'],t['arc_relaxation_myr'],t['compression_memory_myr'],t['extension_memory_myr'],t['shear_memory_myr'],t['root_relaxation_thickness_power']])
    initial_volume=float(np.sum(u[:,:,VC]*mesh.area));min_seen=0.;max_cover_error=0.;merge_time=0.
    rift_times=sorted(duration-float(age) for age in t['rift_ages_myr'] if 0<float(age)<duration);rift_index=0
    flux=omega@mesh.flux_coeff.T;dt_cfl=courant_limit(flux,mesh.edges,mesh.area,active,t['cfl'])
    progress(f'Tectonics: {duration:g} Myr, {mesh.n:,} spherical cells (~{mesh.spacing:.0f} km), {count} initial plates.',flush=True)
    while True:
        if elapsed>=next_snapshot-1e-7 or elapsed>=duration-1e-7:
            sf,sea_datum=apply_ocean_load(state_fields(u,t),mesh.area,config['planet']['ocean_volume_km3'],t);tag=f'{int(round(duration-elapsed))}Ma'
            snaps['height_'+tag]=sf[:,0].astype(np.float32);snaps['plates_'+tag]=np.argmax(u[:,:,C]+u[:,:,O],0).astype(np.int16)
            snaps['continental_'+tag]=sf[:,2].astype(np.float32)
            area_plate=np.sum((u[:,:,C]+u[:,:,O])*mesh.area[None,:],axis=1)
            history.append({'sea_datum_km':sea_datum,'emerged_land_fraction':float(np.sum(mesh.area[sf[:,0]>0])/mesh.area.sum()),'age_myr':duration-elapsed,'active_plates':int(active.sum()),'continental_fraction':float(np.sum(sf[:,2]*mesh.area)/mesh.area.sum()),'height_min_km':float(sf[:,0].min()),'height_max_km':float(sf[:,0].max()),'continental_volume_km3':float(np.sum(u[:,:,VC]*mesh.area)), 'mean_ocean_age_myr':float(np.sum(u[:,:,OA]*mesh.area)/max(np.sum(u[:,:,O]*mesh.area),1e-20))})
            next_snapshot=elapsed+t['snapshot_interval_myr']
        if elapsed>=duration-1e-7:break
        changed=False
        if elapsed>=next_epoch-1e-7:
            # Prescribed slow changes of plate motion, not a mantle-force solver.
            new=unit(rng.normal(size=(P,3)))
            speed=drive_speed.copy()
            target=unit(.7*unit(target)+.3*new)*speed[:,None]
            events.append({'age_myr':duration-elapsed,'type':'prescribed_motion_epoch'})
            next_epoch+=t['motion_epoch_myr'];changed=True
        if rift_index<len(rift_times) and elapsed>=rift_times[rift_index]-1e-7:
            platearea=np.sum((u[:,:,C]+u[:,:,O])*mesh.area,axis=1)
            carea=np.sum(u[:,:,C]*mesh.area,axis=1);score=platearea*(.3+carea/np.maximum(platearea,1))
            parent=int(np.argmax(score));free=np.flatnonzero(~active)
            if len(free):
                child=int(free[0]);centre=unit(np.sum(mesh.p*(u[parent,:,C]+u[parent,:,O])[:,None]*mesh.area[:,None],axis=0))
                direction=rng.normal(size=3);direction=unit(direction-centre*(direction@centre))
                split=mesh.p@direction>0
                shares=np.array([platearea[parent]-np.sum((u[parent,split,C]+u[parent,split,O])*mesh.area[split]),np.sum((u[parent,split,C]+u[parent,split,O])*mesh.area[split])])
                if min(shares)>.08*max(shares):
                    u[child,split]=u[parent,split];u[parent,split]=0;active[child]=True
                    delta=np.cross(centre,direction)*(t['rift_full_speed_cm_per_year']*10/R*.5)
                    old=omega[parent].copy();omega[parent]=old-delta;omega[child]=old+delta
                    target[parent]=omega[parent];target[child]=omega[child]
                    drive_speed[parent]=np.linalg.norm(omega[parent]);drive_speed[child]=np.linalg.norm(omega[child])
                    events.append({'age_myr':duration-elapsed,'type':'rift','parent':parent,'daughter':child});changed=True
            rift_index+=1
        # Integrate angular-velocity transitions on a geological time scale.
        if elapsed-last_motion>=t['velocity_update_myr']-1e-7 or changed:
            delta=elapsed-last_motion;omega=target+(omega-target)*math.exp(-delta/t['motion_response_myr']);last_motion=elapsed
            flux=omega@mesh.flux_coeff.T;dt_cfl=courant_limit(flux,mesh.edges,mesh.area,active,t['cfl'])
        dt=min(float(t['max_timestep_myr']),dt_cfl,duration-elapsed,next_snapshot-elapsed,next_epoch-elapsed)
        if rift_index<len(rift_times):dt=min(dt,rift_times[rift_index]-elapsed)
        if dt<1e-9:raise RuntimeError('Tectonic time scheduling did not advance.')
        u=advect(u,flux,mesh.edges,mesh.upstream,mesh.backscale,mesh.area,dt,active)
        if steps%20==0:
            low=float(u.min());min_seen=min(min_seen,low)
            if not np.isfinite(u).all() or low<-1e-7:raise RuntimeError(f'Unstable tectonic transport (minimum {low}). Reduce tectonics.cfl.')
        budgets+=interact(u,mesh.area,dt,settings,contact)
        u=spread_roots(u,mesh.edges,mesh.length/mesh.primal_length,mesh.area,dt,t['root_spreading_km2_per_myr'])
        # Shear history follows the crust; transforms do not shorten the crust.
        if steps%8==0:
            rates,owner,valid,normal,slide=active_boundaries(mesh,u,omega,t)
            mask=rates[:,2]*dt*8/max(t['deformation_width_km'],mesh.spacing)
            for p in np.flatnonzero(active):u[p,:,SHEAR]+=(u[p,:,C]+u[p,:,O])*mask
        elapsed+=dt;steps+=1
        if elapsed-merge_time>=t['suture_check_interval_myr']:
            merge_time=elapsed
            platearea=np.sum((u[:,:,C]+u[:,:,O])*mesh.area,axis=1)
            pair=contact+contact.T;np.fill_diagonal(pair,0)
            scaled=pair/np.maximum(np.minimum(platearea[:,None],platearea[None,:]),1)
            i,j=np.unravel_index(np.argmax(scaled),scaled.shape)
            if scaled[i,j]>t['suture_consumed_area_fraction'] and active[i] and active[j]:
                if platearea[i]<platearea[j]:i,j=j,i
                omega[i]=(omega[i]*platearea[i]+omega[j]*platearea[j])/(platearea[i]+platearea[j]);target[i]=omega[i]
                drive_speed[i]=(drive_speed[i]*platearea[i]+drive_speed[j]*platearea[j])/(platearea[i]+platearea[j]);drive_speed[j]=0.
                u[i]+=u[j];u[j]=0;active[j]=False;omega[j]=0;target[j]=0
                contact[i]+=contact[j];contact[:,i]+=contact[:,j];contact[j]=0;contact[:,j]=0;contact[i,i]=0
                events.append({'age_myr':duration-elapsed,'type':'continental_suture','survivor':int(i),'absorbed':int(j)})
                flux=omega@mesh.flux_coeff.T;dt_cfl=courant_limit(flux,mesh.edges,mesh.area,active,t['cfl'])
            contact*=math.exp(-t['suture_check_interval_myr']/t['suture_memory_myr'])
        if elapsed>=next_report or elapsed>=duration:
            cover=(u[:,:,C]+u[:,:,O]).sum(0);err=float(np.max(np.abs(cover-1)));max_cover_error=max(max_cover_error,err)
            progress(f'  {duration-elapsed:6.1f} Ma remaining | {steps} steps | {int(active.sum())} plates | {time.perf_counter()-start:.1f} s',flush=True)
            next_report=elapsed+100
    values=state_fields(u,t);rates,owner,valid,normal,slide=active_boundaries(mesh,u,omega,t)
    # Present trenches are on the oceanic side only; sea-floor ages already
    # generate spreading-ridge buoyancy without a second random ridge mask.
    trench=(1-values[:,2])*np.clip(rates[:,0]/(t['boundary_reference_speed_cm_per_year']*10),0,1)
    values[:,0]-=trench*t['trench_depth_metres']/1000*(1-t['water_density_kg_m3']/t['mantle_density_kg_m3'])
    values,sea_datum=apply_ocean_load(values,mesh.area,config['planet']['ocean_volume_km3'],t)
    snaps['height_0Ma']=values[:,0].astype(np.float32)
    history[-1]['sea_datum_km']=sea_datum
    history[-1]['emerged_land_fraction']=float(mesh.area[values[:,0]>0].sum()/mesh.area.sum())
    history[-1]['height_min_km']=float(values[:,0].min());history[-1]['height_max_km']=float(values[:,0].max())
    conv=np.clip(rates[:,0]/(t['boundary_reference_speed_cm_per_year']*10),0,1)
    div=np.clip(rates[:,1]/(t['boundary_reference_speed_cm_per_year']*10),0,1)
    # Unaltered raw node fields are retained for numerical audits and reuse.
    final_volume=float(np.sum(u[:,:,VC]*mesh.area));expected=initial_volume-budgets[2]
    plates=[]
    for p in np.flatnonzero(active):
        area=(u[p,:,C]+u[p,:,O])*mesh.area;ar=area.sum()
        if ar<1:continue
        plates.append({'id':int(p),'area_km2':float(ar),'continental_fraction':float(np.sum(u[p,:,C]*mesh.area)/ar),
                       'mean_elevation_m':float(np.sum(area*values[:,0])/ar*1000),'euler_vector_rad_per_myr':omega[p].tolist(),
                       'maximum_speed_cm_per_year':float(np.linalg.norm(omega[p])*R/10)})
    stats={'model':'conservative kinematic finite-volume lithosphere; NOT mantle convection',
           'sea_datum_km':sea_datum,'ocean_volume_target_km3':config['planet']['ocean_volume_km3'],'ocean_volume_actual_km3':float(np.sum(np.maximum(-values[:,0],0)*mesh.area)),'duration_myr':elapsed,'steps':steps,'mesh_cells':mesh.n,'nominal_cell_spacing_km':mesh.spacing,
           'initial_plates':count,'final_plates':int(active.sum()),'rift_events':sum(e['type']=='rift' for e in events),
           'suture_events':sum(e['type']=='continental_suture' for e in events),
           'ocean_area_created_km2':budgets[0],'ocean_area_subducted_km2':budgets[1],
           'initial_continental_volume_km3':initial_volume,'final_continental_volume_km3':final_volume,
           'eroded_continental_volume_km3':budgets[2], 'continental_volume_budget_relative_error':(final_volume-expected)/max(initial_volume,1),
           'arc_volume_generated_km3':budgets[3],'arc_volume_subducted_km3':budgets[4],'arc_volume_eroded_km3':budgets[5],
           'max_coverage_error':max_cover_error,'minimum_transport_density':min_seen,
           'simulation_seconds':time.perf_counter()-start,'history':history,'events':events,'plates':plates}
    directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/'tectonics_state.npz',unit_positions=mesh.p.astype(np.float32),cell_area_km2=mesh.area,
                        plate_ids=owner,continental_fraction=values[:,2],crust_thickness_km=values[:,3],
                        ocean_age_myr=values[:,4],height_km=values[:,0],compression_memory=values[:,5],extension_memory=values[:,6],
                        shear_memory=values[:,7],igneous_height_km=values[:,8],convergence_km_per_myr=rates[:,0],divergence_km_per_myr=rates[:,1],
                        transform_km_per_myr=rates[:,2],omega_rad_per_myr=omega)
    np.savez_compressed(directory/'tectonics_history.npz',unit_positions=mesh.p.astype(np.float32),**snaps)
    (directory/'tectonics_report.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    return mesh,values,conv,div,stats,owner


def validate_config(c):
    t=c['tectonics']
    if not 0<c['planet']['ocean_volume_km3']<1e12:raise ValueError('planet.ocean_volume_km3 must be positive and less than 1e12.')
    if not isinstance(t['mesh_subdivisions'],int) or not 3<=t['mesh_subdivisions']<=6:
        raise ValueError('tectonics.mesh_subdivisions: use 3–6 (higher uses much more time/memory).')
    if not 0<t['duration_myr']<=2000:raise ValueError('tectonics.duration_myr: use >0 and <=2000.')
    if not .05<=t['cfl']<=.45:raise ValueError('tectonics.cfl: use .05–.45.')
    if not 0<t['max_timestep_myr']<=10:raise ValueError('tectonics.max_timestep_myr: use >0 and <=10.')
    for name in ('plate_speed_cm_per_year','initial_crust_thickness_km'):
        v=_vector_setting(t,name,t['plate_count'])
        if not np.isfinite(v).all() or np.min(v)<=0:raise ValueError(f'tectonics.{name}: positive finite values required.')
    if np.max(t['plate_speed_cm_per_year'])>20:raise ValueError('Plate speeds over 20 cm/yr are not supported by this configuration.')
    if len(t['rift_ages_myr'])>16 or any(not 0<a<t['duration_myr'] for a in t['rift_ages_myr']):
        raise ValueError('rift_ages_myr: at most 16 distinct ages strictly between 0 and duration_myr.')
    if len(set(t['rift_ages_myr']))!=len(t['rift_ages_myr']):raise ValueError('rift_ages_myr must not contain duplicates.')
    if not (t['mantle_density_kg_m3']>t['oceanic_density_kg_m3']>t['continental_density_kg_m3']>t['water_density_kg_m3']>0):
        raise ValueError('Densities must satisfy mantle > oceanic > continental > water > 0.')
    for name in ('motion_epoch_myr','motion_response_myr','velocity_update_myr','suture_check_interval_myr',
                 'suture_memory_myr','root_relaxation_myr','arc_relaxation_myr','compression_memory_myr',
                 'extension_memory_myr','shear_memory_myr','thermal_plate_thickness_km','thermal_diffusivity_km2_per_myr',
                 'compression_mask_strain','extension_mask_strain','shear_mask_strain','snapshot_interval_myr',
                 'boundary_reference_speed_cm_per_year','reference_crust_thickness_km','root_relaxation_thickness_power'):
        if not math.isfinite(t[name]) or t[name]<=0:raise ValueError(f'tectonics.{name}: positive finite number required.')
