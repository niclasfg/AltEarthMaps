#!/usr/bin/env python3
"""Run with Python 3.11+: python terrain.py. Local files only; no online maps."""
from __future__ import annotations
import sys,os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
if sys.version_info<(3,11):raise SystemExit('Python 3.11 or newer is required.')
import json,math,mimetypes,tomllib,webbrowser
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path(__file__).resolve().parent
VERSION='Tectonic Globe 4.0'


# Settings schema for the control panel. Tier: 'instant' (JS re-render only),
# 'rebundle' (shader defines recompiled, no science rerun), 'rebuild' (maps
# regenerated, ~1 min). Kind: float/int/bool/enum/floatlist/textlist/color.
# floatlist len: int, 'plate' (== tectonics.plate_count) or 'any'.
def _S(tab,section,key,kind,tier,label,help,extra=None):
    e={'tab':tab,'section':section,'key':key,'kind':kind,'tier':tier,'label':label,'help':help}
    if extra:e.update(extra)
    return e
def _F(tab,section,key,tier,label,help,lo,hi,step):
    return _S(tab,section,key,'float',tier,label,help,{'min':lo,'max':hi,'step':step})
def _I(tab,section,key,tier,label,help,lo,hi,step=1):
    return _S(tab,section,key,'int',tier,label,help,{'min':lo,'max':hi,'step':step})
SETTINGS_SCHEMA=[
 _F('View','viewer','start_longitude_degrees','instant','Start longitude','Initial view centre.',-180,180,0.5),
 _F('View','viewer','start_latitude_degrees','instant','Start latitude','Initial view centre.',-90,90,0.5),
 _F('View','viewer','view_width_metres','instant','View width (m)','Orthographic span at load.',1e3,5e7,1e5),
 _F('View','viewer','min_view_width_metres','instant','Min zoom (m)','Closest zoom allowed.',10,1e6,10),
 _I('View','viewer','max_render_width','instant','Render width','Pixels across; lower if GPU struggles.',256,3840,64),
 _F('View','viewer','interactive_scale','instant','Drag quality','Resolution fraction while moving.',0.2,1,0.05),
 _I('View','viewer','settle_ms','instant','Settle delay (ms)','Full-quality render after drag stops.',0,2000,10),
 _S('View','viewer','supersampling','enum','instant','Supersampling','2 renders 4x pixels when still.',{'options':[1,2]}),
 _I('World','planet','seed','rebuild','Seed','New value builds a new world.',0,999999),
 _F('World','planet','radius_metres','rebuild','Radius (m)','Spherical mean radius.',1e5,1e8,1e5),
 _F('World','planet','emerged_land_target','rebuild','Land fraction','Sea datum solved to hit this.',0.05,0.80,0.01),
 _F('World','planet','ocean_volume_km3','rebuild','Ocean volume (km³)','Tectonic-run inventory; sets operating point.',1e8,3e9,1e7),
 _S('World','planet','map_width','enum','rebuild','Guide width','Global map resolution (rebuilds everything).',{'options':[256,512,1024,2048,4096]}),
 _I('World','planet','continent_count','rebuild','Continents','Target landmass count.',1,12),
 _F('World','planet','continent_size_variety','rebuild','Size variety','0 equal, 1 isles-to-supercontinents.',0,1,0.05),
 _F('World','planet','continent_budget_fraction','rebuild','Crust budget','Sphere share claimed by continents.',0.05,0.90,0.01),
 _F('World','tectonics','duration_myr','rebuild','Duration (Myr)','Forward-simulated deep time.',100,2000,50),
 _S('World','tectonics','mesh_subdivisions','enum','rebuild','Mesh level','4: 2.5k cells, 5: 10k, 6: 41k (slow).',{'options':[3,4,5,6]}),
 _F('World','tectonics','max_timestep_myr','rebuild','Max timestep (Myr)','Reduced to satisfy CFL.',0.1,10,0.1),
 _F('World','tectonics','cfl','rebuild','CFL number','Smaller is slower, not more detailed.',0.05,0.45,0.01),
 _I('World','tectonics','plate_count','rebuild','Plates','Initial plate count.',4,100),
 _S('World','tectonics','plate_speed_cm_per_year','floatlist','rebuild','Plate speeds (cm/yr)','One value broadcasts; else one per plate.',{'len':'plate','min':0.1,'max':20,'step':0.1}),
 _S('World','tectonics','initial_crust_thickness_km','floatlist','rebuild','Crust thickness (km)','Per-plate initial thickness.',{'len':'plate','min':5,'max':80,'step':0.5}),
 _S('World','tectonics','initial_ocean_age_myr','floatlist','rebuild','Ocean age range (Myr)','[young, old] new-to-old seafloor.',{'len':2,'min':0,'max':500,'step':1}),
 _F('World','tectonics','motion_epoch_myr','rebuild','Motion epoch (Myr)','Prescribed drift retarget interval.',10,1000,10),
 _F('World','tectonics','motion_response_myr','rebuild','Motion response (Myr)','Finite response time.',1,500,1),
 _F('World','tectonics','velocity_update_myr','rebuild','Velocity update (Myr)','Angular-velocity refresh.',0.5,50,0.5),
 _S('World','tectonics','rift_ages_myr','textlist','rebuild','Rift ages (Myr BP)','Comma list; empty disables rifting.',{'min':0,'max':2000}),
 _F('World','tectonics','rift_full_speed_cm_per_year','rebuild','Rift speed (cm/yr)','Divergence after a split.',0.5,10,0.1),
 _F('World','tectonics','breakup_crust_thickness_km','rebuild','Breakup thickness (km)','Below this, openings make ocean crust.',3,30,0.5),
 _F('World','tectonics','suture_check_interval_myr','rebuild','Suture check (Myr)','How often plates may merge.',1,200,1),
 _F('World','tectonics','suture_consumed_area_fraction','rebuild','Suture fraction','Contact needed to merge.',0.05,0.9,0.01),
 _F('World','tectonics','suture_memory_myr','rebuild','Suture memory (Myr)','Contact integration time.',10,1000,10),
 _F('World','tectonics','mantle_density_kg_m3','rebuild','Mantle density','Must exceed crust densities.',2500,4500,10),
 _F('World','tectonics','continental_density_kg_m3','rebuild','Continent density','Below mantle, above water.',2000,3200,10),
 _F('World','tectonics','oceanic_density_kg_m3','rebuild','Oceanic density','Between continent and mantle.',2000,3400,10),
 _F('World','tectonics','water_density_kg_m3','rebuild','Water density','Lowest of all.',500,1500,5),
 _F('World','tectonics','reference_crust_thickness_km','rebuild','Reference crust (km)','Airy reference column.',10,80,0.5),
 _F('World','tectonics','reference_continent_elevation_metres','rebuild','Reference elevation (m)','Freeboard of the reference column.',0,3000,10),
 _F('World','tectonics','thermal_plate_thickness_km','rebuild','Thermal plate (km)','Plate-cooling thickness.',30,300,1),
 _F('World','tectonics','thermal_diffusivity_km2_per_myr','rebuild','Thermal diffusivity','1e-6 m²/s ≈ 31.6.',1,200,0.1),
 _F('World','tectonics','ridge_depth_metres','rebuild','Ridge depth (m)','Newborn seafloor depth.',1000,6000,50),
 _F('World','tectonics','thermal_subsidence_metres','rebuild','Subsidence (m)','Old-floor deepening.',0,7000,50),
 _F('World','tectonics','trench_depth_metres','rebuild','Trench depth (m)','Subduction freeboard correction.',0,8000,50),
 _F('World','tectonics','boundary_reference_speed_cm_per_year','rebuild','Reference speed (cm/yr)','Normalizes boundary rates.',1,20,0.5),
 _F('World','tectonics','arc_accretion_fraction','rebuild','Arc accretion','Consumed-crust magma fraction.',0,1,0.01),
 _F('World','tectonics','arc_relaxation_myr','rebuild','Arc relaxation (Myr)','Arc-root decay time.',10,1000,10),
 _F('World','tectonics','root_spreading_km2_per_myr','rebuild','Root spreading','Lateral crust relaxation.',0,50000,100),
 _F('World','tectonics','root_relaxation_myr','rebuild','Root relaxation (Myr)','Denudation timescale.',10,1000,10),
 _F('World','tectonics','root_relaxation_thickness_power','rebuild','Relaxation power','Thickness dependence.',1,6,0.1),
 _F('World','tectonics','compression_memory_myr','rebuild','Compression memory','Strain-history lifetime.',10,1000,10),
 _F('World','tectonics','extension_memory_myr','rebuild','Extension memory','Strain-history lifetime.',10,1000,10),
 _F('World','tectonics','shear_memory_myr','rebuild','Shear memory','Strain-history lifetime.',10,1000,10),
 _F('World','tectonics','compression_mask_strain','rebuild','Compression strain','Mask saturation scale.',0.01,5,0.01),
 _F('World','tectonics','extension_mask_strain','rebuild','Extension strain','Mask saturation scale.',0.01,5,0.01),
 _F('World','tectonics','shear_mask_strain','rebuild','Shear strain','Mask saturation scale.',0.01,5,0.01),
 _F('World','tectonics','deformation_width_km','rebuild','Deformation width (km)','Boundary zone width.',10,1000,10),
 _F('World','tectonics','snapshot_interval_myr','rebuild','Snapshot interval','Saved data only.',10,1000,10),
 _F('World','climate','equator_temperature_c','rebuild','Equator temp (°C)','Sea-level equatorial temperature.',-10,40,0.5),
 _F('World','climate','pole_temperature_drop_c','rebuild','Pole drop (°C)','Equator-to-pole difference.',0,80,1),
 _F('World','climate','lapse_c_per_km','rebuild','Lapse rate (°C/km)','Cooling with height.',0,10,0.1),
 _F('World','climate','mean_rainfall_mm','rebuild','Mean rainfall (mm)','Global moisture baseline.',100,5000,50),
 _F('World','climate','regional_temperature_variation_c','rebuild','Regional temp var (°C)','Large-scale anomalies.',0,15,0.5),
 _F('World','climate','regional_rainfall_variation','rebuild','Regional rain var','Log-rainfall anomalies.',0,3,0.1),
 _F('World','climate','rain_shadow_strength','rebuild','Rain shadow','Lee-side drying strength.',0,3,0.1),
 _S('Water','hydrology','enabled','bool','rebuild','Hydrology on','Routing, rivers, lakes, flow guide.',{}),
 _S('Water','hydrology','routing_width','enum','rebuild','Routing width','Coarse watershed grid.',{'options':[256,512,1024,2048,4096]}),
 _F('Water','hydrology','river_source_area_km2','rebuild','Trunk area (km²)','Major-river threshold.',1000,500000,1000),
 _I('Water','hydrology','route_iterations','rebuild','Route passes','Flat-receiver refinement.',1,10),
 _S('Water','hydrology','refine_routing','bool','rebuild','Refine pass','Full-resolution tributaries.',{}),
 _F('Water','hydrology','refine_source_area_km2','rebuild','Tributary area (km²)','Refine-pass threshold.',500,200000,500),
 _I('Water','hydrology','max_river_records','rebuild','Max river records','Largest-first cap.',1000,500000,1000),
 _F('Water','hydrology','width_coefficient_metres','rebuild','Width coefficient','Half-width = c·√area.',0.1,5,0.05),
 _F('Water','hydrology','maximum_half_width_metres','rebuild','Max half-width (m)','River width cap.',100,5000,50),
 _F('Water','hydrology','valley_width_metres','rebuild','Valley width (m)','Carve influence radius.',1000,40000,500),
 _F('Water','hydrology','lake_minimum_depth_metres','rebuild','Lake min depth (m)','Depression filter floor.',1,200,1),
 _F('Water','hydrology','lake_maximum_depth_metres','rebuild','Lake max depth (m)','Depression filter ceiling.',50,2000,10),
 _F('Water','hydrology','erosion_flow_gain','rebundle','Flow gain','Fine-octave downstream steering.',0,2,0.05),
 _I('Water','hydrology','coarse_follow_octaves','rebundle','Follow octaves','First N octaves follow only.',0,13),
 _F('Water','hydrology','gully_stream_onset','rebundle','Stream onset','Channel strength for water.',0,13,0.05),
 _F('Water','hydrology','gully_stream_softness','rebundle','Stream softness','Ramp above onset.',0.05,13,0.05),
 _F('Water','hydrology','trunk_display_wavelength_metres','rebundle','Trunk wavelength (m)','Trunk fade-in scale.',1000,400000,1000),
 _F('Terrain','fluvial','strength','rebuild','Fluvial strength','0 original, 1 graded profiles.',0,1,0.05),
 _S('Terrain','fluvial','enabled','bool','rebuild','Fluvial on','Stream-power carving pass.',{}),
 _F('Terrain','fluvial','grade','rebuild','Reference grade','Downstream slope scale.',0.001,1,0.005),
 _F('Terrain','fluvial','discharge_reference_km2','rebuild','Reference discharge','Q scale for grade.',100,1e7,100),
 _F('Terrain','fluvial','discharge_exponent','rebuild','Discharge exponent','Theory ≈ 0.5.',0.1,1.5,0.05),
 _F('Terrain','fluvial','deposition_metres','rebuild','Deposition cap (m)','Lowland aggradation limit.',0,1000,10),
 _F('Terrain','fluvial','estuary_area_km2','rebuild','Estuary area (km²)','Trunks that overdeepen.',1000,500000,1000),
 _F('Terrain','fluvial','estuary_depth_metres','rebuild','Estuary depth (m)','Mouth overdeepening.',0,1000,10),
 _I('Terrain','fluvial','diffusion_passes','rebuild','Diffusion passes','Hillslope rounding.',0,10),
 _S('Terrain','relief','wavelength_metres','textlist','rebundle','Wavelengths (m)','Decreasing landform scales.',{'min':1,'max':1e8}),
 _S('Terrain','relief','amplitude_metres','textlist','rebundle','Amplitudes (m)','Per-layer relief.',{'min':0,'max':20000}),
 _F('Terrain','relief','plain_fraction','rebundle','Plain fraction','Quiet-interior relief share.',0,1,0.005),
 _S('Terrain','relief','collision_gain','textlist','rebundle','Collision gain','Per-layer tectonic response.',{'min':0,'max':5}),
 _S('Terrain','relief','rift_gain','textlist','rebundle','Rift gain','Per-layer tectonic response.',{'min':0,'max':5}),
 _S('Terrain','relief','transform_gain','textlist','rebundle','Transform gain','Per-layer tectonic response.',{'min':0,'max':5}),
 _S('Terrain','erosion','wavelength_metres','textlist','rebundle','Wavelengths (m)','Decreasing gully scales.',{'min':0.1,'max':1e7}),
 _S('Terrain','erosion','amplitude_metres','textlist','rebundle','Amplitudes (m)','Per-octave cut depth.',{'min':0,'max':5000}),
 _S('Terrain','erosion','gully_weight','textlist','rebundle','Gully weight','Direction-steering share.',{'min':0,'max':2}),
 _S('Terrain','erosion','detail','textlist','rebundle','Detail','Mask sharpening.',{'min':0.1,'max':5}),
 _S('Terrain','erosion','ridge_rounding','textlist','rebundle','Ridge rounding','Crest softness.',{'min':0,'max':1}),
 _S('Terrain','erosion','crease_rounding','textlist','rebundle','Crease rounding','Gully softness.',{'min':0,'max':1}),
 _S('Terrain','erosion','onset','textlist','rebundle','Onset','Gully trigger.',{'min':0.1,'max':5}),
 _S('Terrain','erosion','cell_scale','textlist','rebundle','Cell scale','Phacelle frequency.',{'min':0.05,'max':1.9}),
 _S('Terrain','erosion','normalization','textlist','rebundle','Normalization','Vector limiting (<0.95).',{'min':0.05,'max':0.94}),
 _S('Terrain','erosion','collision_gain','textlist','rebundle','Collision gain','Per-octave response.',{'min':0,'max':5}),
 _S('Terrain','erosion','rift_gain','textlist','rebundle','Rift gain','Per-octave response.',{'min':0,'max':5}),
 _S('Terrain','erosion','transform_gain','textlist','rebundle','Transform gain','Per-octave response.',{'min':0,'max':5}),
 _F('Terrain','erosion','old_belt_rounding_multiplier','rebundle','Old-belt rounding','Relaxed-range softness.',1,5,0.1),
 _F('Terrain','erosion','assumed_slope','rebundle','Assumed slope','Steering slope.',0.1,3,0.05),
 _F('Terrain','erosion','initial_rounding_multiplier','rebundle','Initial rounding','First-octave softness.',0,1,0.05),
 _S('Terrain','dunes','wavelength_metres','textlist','rebundle','Wavelengths (m)','Dune scales.',{'min':0.1,'max':1e6}),
 _S('Terrain','dunes','amplitude_metres','textlist','rebundle','Amplitudes (m)','Dune heights.',{'min':0,'max':500}),
 _F('Terrain','dunes','wind_degrees','rebundle','Wind (deg)','Dune orientation.',0,360,1),
 _F('Terrain','dunes','maximum_slope_degrees','rebundle','Max slope (deg)','Dune cutoff slope.',1,45,0.5),
 _S('Look','ecology','tree_slope_degrees','floatlist','rebundle','Tree slopes (deg)','[full forest, none].',{'len':2,'min':0,'max':90,'step':0.5}),
 _S('Look','ecology','rock_slope_degrees','floatlist','rebundle','Rock slopes (deg)','[start, full rock].',{'len':2,'min':0,'max':90,'step':0.5}),
 _S('Look','ecology','snow_temperature_c','floatlist','rebundle','Snow temps (°C)','[full, none].',{'len':2,'min':-60,'max':30,'step':0.5}),
 _S('Look','appearance','ocean_deep','color','rebundle','Deep ocean','Abyss reflectance.',{}),
 _S('Look','appearance','ocean_shallow','color','rebundle','Shallow ocean','Shelf reflectance.',{}),
 _S('Look','appearance','forest','color','rebundle','Forest','Canopy reflectance.',{}),
 _S('Look','appearance','grass','color','rebundle','Grass','Grassland reflectance.',{}),
 _S('Look','appearance','desert','color','rebundle','Desert','Arid reflectance.',{}),
 _S('Look','appearance','rock','color','rebundle','Rock','Bare-rock reflectance.',{}),
 _S('Look','appearance','snow','color','rebundle','Snow','Snow reflectance.',{}),
 _S('Look','appearance','lake','color','rebundle','Lake water','Freshwater reflectance.',{}),
 _S('Look','appearance','bathymetry','bool','rebundle','Bathymetry','Seafloor shading.',{}),
 _F('Look','appearance','hillshade_strength','rebundle','Hillshade','Relief shading strength.',0,1,0.05),
 _F('Look','appearance','canopy_cell_metres','rebundle','Canopy cell (m)','Tree texture scale.',1,100,0.5),
 _F('Look','appearance','canopy_height_metres','rebundle','Canopy height (m)','Tree display height.',0,50,0.5),
 _S('Look','appearance','surface_wavelength_metres','textlist','rebundle','Pattern scales (m)','Reflectance bands.',{'min':0.1,'max':1e7}),
 _S('Look','appearance','surface_colour_strength','textlist','rebundle','Pattern strength','Per-band strength.',{'min':0,'max':1}),
]
SCHEMA_INDEX={(e['section'],e['key']):e for e in SETTINGS_SCHEMA}


def checked_config(overrides=None):
    # Layer validation; tectonic widths naturally run core -> foreland instead.
    c=tomllib.loads((ROOT/'config.toml').read_text())
    if overrides is None and (ROOT/'overrides.json').is_file():
        try:overrides=json.loads((ROOT/'overrides.json').read_text())
        except (OSError,ValueError):overrides=None
    if overrides:
        for s,kv in overrides.items():
            if not isinstance(kv,dict):raise ValueError(f'overrides: section {s!r} must be an object.')
            if s not in c or not isinstance(c[s],dict):c[s]={}
            for k,v in kv.items():c[s][k]=v
    specs=[('relief','wavelength_metres',['amplitude_metres','collision_gain','rift_gain','transform_gain']),
           ('erosion','wavelength_metres',['amplitude_metres','gully_weight','detail','ridge_rounding','crease_rounding','onset','cell_scale','normalization','collision_gain','rift_gain','transform_gain']),
           ('dunes','wavelength_metres',['amplitude_metres']),
           ('appearance','surface_wavelength_metres',['surface_colour_strength'])]
    for s,root,fields in specs:
        n=len(c[s][root])
        if not 1<=n<=24:raise ValueError(f'{s}: need 1–24 layers.')
        for k in [root]+fields:
            a=c[s][k]
            if not isinstance(a,list) or len(a) not in (1,n):raise ValueError(f'{s}.{k}: need one value or {n} values.')
            if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in a):raise ValueError(f'{s}.{k}: finite numbers required.')
            c[s][k]=[float(x) for x in (a*n if len(a)==1 else a)]
        if any(x<=0 for x in c[s][root]):raise ValueError(f'{s}: wavelengths must be positive.')
        if any(a<=b for a,b in zip(c[s][root],c[s][root][1:])):raise ValueError(f'{s}: wavelengths must decrease.')
    if not 4<=c['tectonics']['plate_count']<=100:raise ValueError('plate_count must be 4–100.')
    from tectonics import validate_config
    validate_config(c)
    if not 1e5<=c['planet']['radius_metres']<=1e8:raise ValueError('radius_metres must be between 100 km and 100,000 km.')
    if not isinstance(c['planet'].get('continent_count'),int) or not 1<=c['planet']['continent_count']<=12:
        raise ValueError('planet.continent_count must be an integer 1–12.')
    if not isinstance(c['planet'].get('seed'),int) or isinstance(c['planet'].get('seed'),bool) or not 0<=c['planet']['seed']<=99999999:
        raise ValueError('planet.seed must be an integer 0–99999999.')
    if not isinstance(c['planet'].get('emerged_land_target'),(int,float)) or isinstance(c['planet'].get('emerged_land_target'),bool) or not .05<=c['planet']['emerged_land_target']<=.80:
        raise ValueError('planet.emerged_land_target must be .05–.80.')
    if not isinstance(c['planet'].get('continent_size_variety'),(int,float)) or isinstance(c['planet'].get('continent_size_variety'),bool) or not 0<=c['planet']['continent_size_variety']<=1:
        raise ValueError('planet.continent_size_variety must be 0–1.')
    if not isinstance(c['planet'].get('continent_budget_fraction'),(int,float)) or isinstance(c['planet'].get('continent_budget_fraction'),bool) or not .05<=c['planet']['continent_budget_fraction']<=.90:
        raise ValueError('planet.continent_budget_fraction must be .05–.90.')
    for s,k in [('planet','map_width'),('hydrology','routing_width')]:
        v=c[s][k]
        if v not in (256,512,1024,2048,4096):raise ValueError(f'{s}.{k}: use 256,512,1024,2048 or 4096.')
    if c['planet']['map_width']%c['hydrology']['routing_width']:raise ValueError('routing_width must divide map_width.')
    hy=c['hydrology'];steps=len(c['erosion']['wavelength_metres'])
    if not all(isinstance(hy[k],(int,float)) and not isinstance(hy[k],bool) and math.isfinite(hy[k]) for k in ('gully_stream_onset','gully_stream_softness','trunk_display_wavelength_metres','refine_source_area_km2','erosion_flow_gain')):raise ValueError('hydrology: finite numbers required for the stream settings.')
    if not 0<=hy['gully_stream_onset']<=steps:raise ValueError(f'gully_stream_onset must be 0–{steps} (one unit per displayed erosion step).')
    if not 0<hy['gully_stream_softness']<=steps:raise ValueError(f'gully_stream_softness must be above 0 and at most {steps}.')
    if not 1000<=hy['trunk_display_wavelength_metres']<=400000:raise ValueError('trunk_display_wavelength_metres must be between 1 km and 400 km.')
    if not isinstance(hy.get('route_iterations',3),(int,float)) or not 1<=int(hy.get('route_iterations',3))<=10:raise ValueError('hydrology.route_iterations must be 1–10.')
    hy['route_iterations']=int(hy.get('route_iterations',3))
    if not isinstance(hy.get('refine_routing',True),bool):raise ValueError('hydrology.refine_routing must be true/false.')
    if not 0<hy['refine_source_area_km2']<hy['river_source_area_km2']:raise ValueError('hydrology.refine_source_area_km2 must be positive and below river_source_area_km2.')
    if not isinstance(hy.get('max_river_records'),int) or not 1000<=hy.get('max_river_records',100000)<=500000:raise ValueError('hydrology.max_river_records must be 1000–500000.')
    fl=c.get('fluvial',{})
    if not isinstance(fl.get('enabled',True),bool):raise ValueError('fluvial.enabled must be true/false.')
    for k,lo,hi in (('strength',0,1),('grade',.001,1),('discharge_reference_km2',1,1e9),('discharge_exponent',.1,1.5),('deposition_metres',0,1000),('estuary_area_km2',1,1e9),('estuary_depth_metres',0,1000)):
        v=fl.get(k)
        if not isinstance(v,(int,float)) or isinstance(v,bool) or not lo<=v<=hi:raise ValueError(f'fluvial.{k} must be {lo}–{hi}.')
    if not isinstance(fl.get('diffusion_passes'),int) or not 0<=fl.get('diffusion_passes',2)<=10:raise ValueError('fluvial.diffusion_passes must be 0–10.')
    hy.setdefault('max_river_records',100000)
    c.setdefault('fluvial',{})
    for k,v in (('enabled',True),('strength',.65),('grade',.06),('discharge_reference_km2',5000.0),('discharge_exponent',.5),('deposition_metres',150.0),('estuary_area_km2',20000.0),('estuary_depth_metres',120.0),('diffusion_passes',2)):
        c['fluvial'].setdefault(k,v)
    if not 0<=hy['erosion_flow_gain']<=2:raise ValueError('hydrology.erosion_flow_gain must be 0–2.')
    if not isinstance(hy.get('coarse_follow_octaves',3),(int,float)) or not 0<=int(hy.get('coarse_follow_octaves',3))<=steps:raise ValueError('hydrology.coarse_follow_octaves must be 0–erosion steps.')
    hy['coarse_follow_octaves']=int(hy.get('coarse_follow_octaves',3))
    if c['viewer']['supersampling'] not in (1,2):raise ValueError('supersampling must be 1 or 2.')
    if not 256<=c['viewer']['max_render_width']<=3840:raise ValueError('max_render_width must be 256–3840.')
    for k in ('normalization','cell_scale'):
        if any(not 0<x<1.9 for x in c['erosion'][k]):raise ValueError(f'erosion.{k}: invalid value.')
    if any(x>=.95 for x in c['erosion']['normalization']):raise ValueError('normalization must be below 0.95.')
    return c


def bundle(c,meta):
    lines=['#version 300 es','precision highp float;','precision highp int;','precision highp sampler2D;','precision highp usampler2D;',
           '#define PI 3.141592653589793','#define TAU 6.283185307179586','#define clamp01(x) clamp(x,0.0,1.0)']
    def literal(x):
        if isinstance(x,bool):return 'true' if x else 'false'
        if isinstance(x,list):return 'vec'+str(len(x))+'('+','.join(literal(v) for v in x)+')'
        return repr(float(x))
    def define(n,v):lines.append(f'#define {n} {literal(v)}')
    def arr(n,v,typ='float'):lines.append(f'const {typ} {n}[{len(v)}]={typ}[{len(v)}]('+','.join(str(int(x)) if typ=='int' else literal(x) for x in v)+');')
    bank=[]
    def add(w):
        i=len(bank);bank.append({'scale':w/1000,'offset':[i*7.127,-i*3.719,i*11.23]});return i
    for s,pref in [('relief','R'),('erosion','E'),('dunes','D')]:
        a=c[s];lines.append(f'#define {pref}_N {len(a["wavelength_metres"])}')
        arr(pref+'_IDS',[add(w*(a['cell_scale'][i] if s=='erosion' else 1)) for i,w in enumerate(a['wavelength_metres'])],'int')
        if s=='erosion':
            # Cut-state weight per erosion step for the refined waterways: the
            # coarsest step (largest amplitude) counts fully, finer steps add
            # detail without letting their noise dominate the channel network.
            a0=a['amplitude_metres'][0]
            arr('E_CUT_WEIGHT',[(x/a0)**.3 for x in a['amplitude_metres']])
        for k,v in a.items():
            if isinstance(v,list):arr(pref+'_'+k.upper(),[x/1000 if k.endswith('_metres') else x for x in v])
            else:define(pref+'_'+k.upper(),v)
    a=c['appearance'];arr('S_IDS',[add(w) for w in a['surface_wavelength_metres']],'int');arr('S_AMP',a['surface_colour_strength']);lines.append(f'#define S_N {len(a["surface_wavelength_metres"])}')
    lines.append(f'#define TREE_ID {add(a["canopy_cell_metres"])}')
    lines.append(f'#define MICRO_ID {add(.6)}')
    arr('L_SCALE',[x['scale'] for x in bank]);lines.append(f'#define L_N {len(bank)}')
    for k,v in a.items():
        if not isinstance(v,list) or k not in ('surface_colour_strength','surface_wavelength_metres'):define('A_'+k.upper(),v/1000 if k.endswith('_metres') else v)
    for k,v in c['ecology'].items():define('EC_'+k.upper(),v)
    hy=c['hydrology']
    define('HY_STREAM_ONSET',hy['gully_stream_onset']);define('HY_STREAM_SOFTNESS',hy['gully_stream_softness'])
    define('HY_TRUNK_WAVELENGTH',hy['trunk_display_wavelength_metres']/1000)
    define('HY_FLOW_GAIN',hy['erosion_flow_gain']);define('HY_COARSE_FOLLOW',hy['coarse_follow_octaves'])
    # Per-octave steering ramp: 0 for the first coarse_follow_octaves (follow
    # only), then smooth ramp to erosion_flow_gain at the finest octave so fine
    # gullies are increasingly drawn toward the baked downstream field.
    _n=len(c['erosion']['wavelength_metres']);_cf=hy['coarse_follow_octaves'];_g=hy['erosion_flow_gain']
    _infl=[]
    for _k in range(_n):
        if _k<_cf or _n<=_cf:_infl.append(0.0)
        else:
            _t=(_k-_cf+1)/max(1,_n-_cf);_s=_t*_t*(3-2*_t);_infl.append(_g*_s)
    arr('E_FLOW_INFLUENCE',_infl)
    define('RADIUS',c['planet']['radius_metres']/1000);define('LAPSE',c['climate']['lapse_c_per_km'])
    lines.append(f'const uint WORLD_SEED={int(c["planet"]["seed"])&0xffffffff}u;')
    lines.append(f'#define MAX_RIVER_BIN {max(1,int(meta["max_bin"]))}')
    lines.append(f'#define RIVER_BINS {int(meta["river_header"][0])}')
    common='\n'.join(lines)+'\n'+(ROOT/'shaders/common.glsl').read_text()
    vertex='#version 300 es\nvoid main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.-1.,0.,1.);}'
    return {'config':c,'meta':meta,'bank':bank,'vertex':vertex,
            'height':common+(ROOT/'shaders/terrain.glsl').read_text(),
            'display':common+(ROOT/'shaders/display.glsl').read_text()}


STATE={'folder':None,'meta':None,'data':b'','lock':None,'build_lock':None,'jobs':{},'job_seq':0}


def _file_config():
    return tomllib.loads((ROOT/'config.toml').read_text())


def _read_overrides():
    try:return json.loads((ROOT/'overrides.json').read_text())
    except (OSError,ValueError):return {}


def _dirty_paths():
    file=_file_config();ov=_read_overrides();out=[]
    for s,kv in ov.items():
        for k,v in (kv.items() if isinstance(kv,dict) else []):
            if file.get(s,{}).get(k,'__missing__')!=v:out.append(f'{s}.{k}')
    return out


def _run_job(job_id,config):
    import traceback
    from planet import prepare, Cancelled
    job=STATE['jobs'][job_id]
    cancel=job['cancel']
    def progress(frac,stage):
        if cancel.is_set():raise Cancelled()
        job['progress']=max(0.0,min(1.0,frac));job['stage']=stage
    try:
        folder,meta=prepare(config,ROOT,progress=progress,cancel=cancel.is_set)
        data=json.dumps(bundle(config,meta)).encode()
        with STATE['lock']:
            STATE['folder']=folder;STATE['meta']=meta;STATE['data']=data
        job.update(state='done',progress=1.0,stage='ready')
    except Cancelled:job.update(state='cancelled',stage='cancelled')
    except Exception as e:job.update(state='error',stage='failed',error=f'{type(e).__name__}: {e}')


def _merge_as_overrides(merged):
    file=_file_config();ov={}
    for s,kv in merged.items():
        if not isinstance(kv,dict):continue
        for k,v in kv.items():
            if file.get(s,{}).get(k,'__missing__')!=v:ov.setdefault(s,{})[k]=v
    return ov


def _job_wrapper(jid,candidate):
    STATE['current_job']=jid
    try:_run_job(jid,candidate)
    finally:
        STATE.pop('current_job',None)
        try:STATE['build_lock'].release()
        except RuntimeError:pass


def _send_json(handler,obj,code=200):
    body=json.dumps(obj).encode()
    handler.send_response(code);handler.send_header('Content-Type','application/json')
    handler.send_header('Content-Length',str(len(body)));handler.send_header('Cache-Control','no-cache');handler.end_headers()
    try:handler.wfile.write(body)
    except (BrokenPipeError,ConnectionResetError):pass


def main():
    import threading
    try:
        c=checked_config()
        from planet import prepare
        folder,meta=prepare(c,ROOT)
        data=json.dumps(bundle(c,meta)).encode()
    except ImportError as e:raise SystemExit(f'{e}\nInstall dependencies: python -m pip install -r requirements.txt')
    except (OSError,ValueError,KeyError,TypeError,tomllib.TOMLDecodeError) as e:raise SystemExit(f'Cannot start: {e}\nCheck config.toml.')
    STATE.update(folder=folder,meta=meta,data=data,lock=threading.Lock(),build_lock=threading.Lock())
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            name=self.path.split('?',1)[0]
            if name=='/bundle.json':
                with STATE['lock']:body=STATE['data']
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-cache');self.end_headers()
                try:self.wfile.write(body)
                except (BrokenPipeError,ConnectionResetError):pass
                return
            if name=='/api/schema':_send_json(self,{'schema':SETTINGS_SCHEMA,'version':VERSION});return
            if name=='/api/settings':
                _send_json(self,{'values':checked_config(),'dirty':_dirty_paths()});return
            if name.startswith('/api/jobs/'):
                job=STATE['jobs'].get(name.rsplit('/',1)[-1])
                if job is None:self.send_error(404);return
                _send_json(self,{'state':job['state'],'progress':job['progress'],'stage':job['stage'],'error':job.get('error')});return
            if name.startswith('/data/'):
                with STATE['lock']:folder=STATE['folder']
                path=folder/name.removeprefix('/data/')
            else:path=ROOT/('index.html' if name=='/' else name.lstrip('/'))
            path=path.resolve()
            if name.startswith('/data/'):
                ok=path.is_file()
            else:ok=path.is_relative_to(ROOT) and path.is_file()
            if not ok:self.send_error(404);return
            body=path.read_bytes();mime=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-cache');self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_POST(self):
            name=self.path.split('?',1)[0]
            if name!='/api/settings':self.send_error(404);return
            try:length=int(self.headers.get('Content-Length',0))
            except ValueError:length=0
            try:req=json.loads(self.rfile.read(length) or b'{}')
            except ValueError:_send_json(self,{'error':'Invalid JSON.'},400);return
            if not isinstance(req.get('settings',{}),dict):_send_json(self,{'error':'settings must be an object.'},400);return
            for s,kv in req['settings'].items():
                if not isinstance(kv,dict):_send_json(self,{'error':f'Section {s!r} must be an object.'},400);return
                for k in kv:
                    if (s,k) not in SCHEMA_INDEX:_send_json(self,{'error':f'Unknown setting {s}.{k}.'},400);return
            current=checked_config()
            merged=json.loads(json.dumps(current))
            for s,kv in req['settings'].items():merged.setdefault(s,{}).update(kv)
            try:candidate=checked_config(_merge_as_overrides(merged))
            except (ValueError,KeyError,TypeError,tomllib.TOMLDecodeError) as e:_send_json(self,{'error':str(e)},400);return
            changed=[(s,k) for s,kv in req['settings'].items() for k in kv
                     if current.get(s,{}).get(k)!=candidate.get(s,{}).get(k)]
            tier='instant'
            for s,k in changed:
                t=SCHEMA_INDEX[(s,k)]['tier']
                if t=='rebuild':tier='rebuild';break
                if t=='rebundle':tier='rebundle'
            if req.get('rebuild'):tier='rebuild'
            try:
                file=_file_config();ov={}
                for s,kv in req['settings'].items():
                    for k in kv:
                        if k in candidate.get(s,{}) and file.get(s,{}).get(k,'__missing__')!=candidate[s][k]:ov.setdefault(s,{})[k]=candidate[s][k]
                if ov:(ROOT/'overrides.json').write_text(json.dumps(ov,indent=2,sort_keys=True))
            except OSError as e:_send_json(self,{'error':f'Cannot persist overrides: {e}'},500);return
            if tier in ('instant','rebundle'):
                # Rebundle is milliseconds: refresh the embedded config even
                # when only instant (viewer-side) values changed.
                try:
                    with STATE['lock']:meta=STATE['meta']
                    data=json.dumps(bundle(candidate,meta)).encode()
                except (ValueError,KeyError) as e:_send_json(self,{'error':str(e)},400);return
                with STATE['lock']:STATE['data']=data
                _send_json(self,{'action':'rebundled','tier':tier});return
            if not STATE['build_lock'].acquire(blocking=False):
                _send_json(self,{'error':'A build is already running.','job':STATE.get('current_job')},409);return
            STATE['job_seq']+=1;jid=str(STATE['job_seq'])
            import threading as _th
            STATE['jobs'][jid]={'state':'running','progress':0.0,'stage':'queued','cancel':_th.Event()}
            th=_th.Thread(target=_job_wrapper,args=(jid,candidate),daemon=True);th.start()
            _send_json(self,{'action':'rebuild','tier':tier,'job':jid})
        def do_DELETE(self):
            name=self.path.split('?',1)[0]
            if name.startswith('/api/jobs/'):
                job=STATE['jobs'].get(name.rsplit('/',1)[-1])
                if job is None:self.send_error(404);return
                job['cancel'].set()
                _send_json(self,{'state':job['state']});return
            if name=='/api/settings':
                ov=_read_overrides();tier='instant'
                for s,kv in ov.items():
                    for k in (kv if isinstance(kv,dict) else {}):
                        t=SCHEMA_INDEX.get((s,k),{}).get('tier','rebuild')
                        if t=='rebuild':tier='rebuild';break
                        if t=='rebundle':tier='rebundle'
                    if tier=='rebuild':break
                try:(ROOT/'overrides.json').unlink(missing_ok=True)
                except OSError as e:_send_json(self,{'error':str(e)},500);return
                _send_json(self,{'action':'reset','tier':tier});return
            self.send_error(404)
        def log_message(self,*args):pass
    try:server=ThreadingHTTPServer(('127.0.0.1',int(c['viewer']['port'])),Handler)
    except OSError as e:raise SystemExit(f'{e}\nClose the other copy or change viewer.port.')
    url=f'http://127.0.0.1:{server.server_port}'
    print(f'{VERSION}\n{url}\nDrag to rotate; scroll to zoom. Ctrl+C to stop.',flush=True)
    if c['viewer']['open_browser']:webbrowser.open(url)
    try:server.serve_forever(.2)
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
