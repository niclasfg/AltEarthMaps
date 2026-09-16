#!/usr/bin/env python3
"""Local configuration/file server. All terrain evaluation runs in WebGL2.
Run with Python >= 3.11: python terrain.py. Standard library only.
"""
from __future__ import annotations
import json, math, re, sys, tomllib, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSION = 'World Terrain 2.0'
VERTEX = '''#version 300 es
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);
gl_Position=vec4(p*2.0-1.0,0.0,1.0);}'''


def vector(section: dict, name: str, n: int, lo=-1e10, hi=1e10) -> list[float]:
    a = section[name]
    if not isinstance(a, list) or len(a) not in (1, n):
        raise ValueError(f'{name}: use one value or {n} values, one for every wavelength.')
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
               and math.isfinite(x) and lo <= x <= hi for x in a):
        raise ValueError(f'{name}: values must be finite and between {lo} and {hi}.')
    return [float(x) for x in (a * n if len(a) == 1 else a)]


def load_config() -> dict:
    with (ROOT/'config.toml').open('rb') as f: c=tomllib.load(f)
    def num(s,k,lo,hi):
        x=c[s][k]
        if not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) or not lo<=x<=hi:
            raise ValueError(f'{s}.{k} must be a number in [{lo}, {hi}].')
    specs={
        'continents': {'amplitude_metres':(0,1e7),'rotation_degrees':(-360,360)},
        'relief': {'amplitude_metres':(0,1e7),'rotation_degrees':(-360,360)},
        'erosion': {'amplitude_metres':(0,1e6),'gully_weight':(0,1),'detail':(.01,10),
                    'ridge_rounding':(0,10),'crease_rounding':(0,10),'onset':(.01,20),
                    'cell_scale':(.15,2),'normalization':(0,.9)},
        'dunes': {'amplitude_metres':(0,1e5),'cell_scale':(.2,2),'crest_fraction':(.5,.95)},
        'coast': {'amplitude_metres':(0,10000)}}
    for s,fields in specs.items():
        n=len(c[s]['wavelength_metres'])
        if not 1<=n<=24: raise ValueError(f'{s}: supply 1–24 wavelengths.')
        c[s]['wavelength_metres']=vector(c[s],'wavelength_metres',n,.125,1e9)
        if any(a<=b for a,b in zip(c[s]['wavelength_metres'],c[s]['wavelength_metres'][1:])):
            raise ValueError(f'{s}: wavelengths must go from LARGE to SMALL.')
        for k,(lo,hi) in fields.items(): c[s][k]=vector(c[s],k,n,lo,hi)
    for w,a in [('temperature_wavelength_metres','temperature_amplitude_c'),('rain_wavelength_metres','rain_log_amplitude')]:
        n=len(c['climate'][w]);c['climate'][w]=vector(c['climate'],w,n,1000,1e9)
        c['climate'][a]=vector(c['climate'],a,n,0,200)
    n=len(c['appearance']['surface_wavelength_metres'])
    for k,lo,hi in [('surface_wavelength_metres',.01,1000),('surface_height_metres',0,50),('surface_colour_strength',0,1)]:
        c['appearance'][k]=vector(c['appearance'],k,n,lo,hi)
    for k in ('relief_multiplier','uplift_metres','ridge_rounding'):
        c['provinces'][k]=vector(c['provinces'],k,5,0,1e7)
    for k in ('radius_metres','depth_metres'):
        a=vector(c['lakes'],k,2,.01,1e6)
        if a[0]>a[1]: raise ValueError(f'lakes.{k}: minimum exceeds maximum.')
        c['lakes'][k]=a
    for s,k,lo,hi in [('viewer','port',0,65535),('viewer','max_render_width',256,2560),
        ('viewer','interactive_scale',.2,1),('viewer','settle_ms',20,2000),
        ('viewer','min_view_width_metres',1,1e7),('viewer','max_view_width_metres',1000,1e9),
        ('provinces','cell_metres',10000,1e7),('provinces','blend_sharpness',.5,30),
        ('provinces','parameter_jitter',0,.8),('climate','lapse_c_per_km',0,20),
        ('climate','mean_rainfall_mm',1,20000),('lakes','cell_metres',10000,1e6),
        ('lakes','probability',0,1),('appearance','canopy_cell_metres',1,100),
        ('appearance','canopy_height_metres',0,100),('world','coast_uplift_fade_metres',10,10000),
        ('continents','warp_wavelength_metres',1000,1e9),('coast','vertical_band_metres',1,10000)]: num(s,k,lo,hi)
    if c['lakes']['radius_metres'][1]>.2*c['lakes']['cell_metres']:
        raise ValueError('lakes.radius_metres maximum must be <= 0.2 * lakes.cell_metres.')
    if c['viewer']['supersampling'] not in (1,2): raise ValueError('supersampling must be 1 or 2.')
    v=c['viewer']
    if not v['min_view_width_metres']<=v['view_width_metres']<=v['max_view_width_metres']:
        raise ValueError('Initial view width must lie between min and max.')
    if sum(x*x for x in c['appearance']['sun_direction'])==0: raise ValueError('Sun direction cannot be zero.')
    for k,a in c['appearance'].items():
        if isinstance(a,list) and len(a)==3 and not k.startswith('surface_') and k!='sun_direction':
            vector(c['appearance'],k,3,0,1)
    return c


def lit(v):
    if isinstance(v,bool): return 'true' if v else 'false'
    if isinstance(v,list): return f'vec{len(v)}('+','.join(lit(x) for x in v)+')'
    return repr(float(v))


def bundle(c:dict)->dict:
    """Compile configuration into constants and an origin-rebased noise bank.
    JavaScript computes each lattice's integer origin + fractional remainder in
    double precision. Fine detail never hashes a huge imprecise float coordinate.
    """
    lines=['#version 300 es','precision highp float;','precision highp int;',
           'precision highp sampler2D;','#define PI 3.14159265358979',
           '#define TAU 6.28318530717959','#define clamp01(x) clamp(x,0.0,1.0)']
    bank=[]
    def define(n,x): lines.append(f'#define {n} {lit(x)}')
    def arr(n,a,typ='float'):
        lines.append(f'const {typ} {n}[{len(a)}]={typ}[{len(a)}]('+','.join(str(int(x)) if typ=='int' else lit(x) for x in a)+');')
    def add(w,rot=0,offset=None):
        i=len(bank)
        # Distinct salted domains; fractional offsets stay modest and deterministic.
        if offset is None: offset=[(i*17.317)%91.0-43.1,(i*-29.719)%83.0-39.7]
        bank.append({'wavelength':w/1000,'angle':math.radians(rot),'offset':offset})
        return i
    def block(s,prefix,rotate=False,cell=False):
        sec=c[s];n=len(sec['wavelength_metres'])
        lines.append(f'#define {prefix}_N {n}')
        ids=[]
        for i,w in enumerate(sec['wavelength_metres']):
            ids.append(add(w*(sec['cell_scale'][i] if cell else 1),sec['rotation_degrees'][i] if rotate else 0))
        arr(prefix+'_IDS',ids,'int')
        for k,a in sec.items():
            if isinstance(a,list):
                arr(prefix+'_'+k.upper(),[v/1000 if k in ('wavelength_metres','amplitude_metres') else v for v in a])
    block('continents','C',True);block('relief','R',True);block('erosion','E',cell=True);block('dunes','D',cell=True);block('coast','B')
    for w,a,pref in [('temperature_wavelength_metres','temperature_amplitude_c','T'),('rain_wavelength_metres','rain_log_amplitude','P')]:
        ids=[add(w,31+i*47) for i,w in enumerate(c['climate'][w])]
        lines.append(f'#define {pref}_N {len(ids)}');arr(pref+'_IDS',ids,'int');arr(pref+'_AMP',c['climate'][a])
    ids=[add(w,29+i*31) for i,w in enumerate(c['appearance']['surface_wavelength_metres'])]
    lines.append(f'#define S_N {len(ids)}');arr('S_IDS',ids,'int')
    arr('S_HEIGHT',[x/1000 for x in c['appearance']['surface_height_metres']]);arr('S_COLOUR',c['appearance']['surface_colour_strength'])
    singles={'WARP_X':(c['continents']['warp_wavelength_metres'],13),
             'WARP_Y':(c['continents']['warp_wavelength_metres'],71),
             'PROVINCE':(c['provinces']['cell_metres'],0),
             'OROGENY':(c['provinces']['orogeny_wavelength_metres'],38),
             'WIND':(300000,22),'LAKE':(c['lakes']['cell_metres'],0),
             'LAKE_WARP1':(7000,19),'LAKE_WARP2':(2000,67),
             'CANOPY':(c['appearance']['canopy_cell_metres'],0),'LEAF':(2.5,37),
             'MATERIAL':(1200,37)}
    for name,(w,r) in singles.items(): lines.append(f'#define {name}_ID {add(w,r,[0,0] if name in ("PROVINCE","LAKE","CANOPY") else None)}')
    lines.append(f'#define LATTICE_N {len(bank)}')
    arr('L_SCALE',[x['wavelength'] for x in bank]);arr('L_COS',[math.cos(x['angle']) for x in bank]);arr('L_SIN',[math.sin(x['angle']) for x in bank])
    lines+=['uniform ivec2 uCell[LATTICE_N];','uniform vec2 uFraction[LATTICE_N];']
    for k in ('relief_multiplier','ridge_rounding'): arr('PROF_'+k.upper(),c['provinces'][k])
    arr('PROF_UPLIFT',[v/1000 for v in c['provinces']['uplift_metres']])
    for sec,pref in [('world','W_'),('climate','CL_'),('ecology','EC_'),('provinces','PR_'),('lakes','LK_')]:
        for k,v in c[sec].items():
            if k in ('seed','names','relief_multiplier','uplift_metres','ridge_rounding') or isinstance(v,list) and k not in ('radius_metres','depth_metres'): continue
            if isinstance(v,(bool,int,float,list)):
                if k.endswith('_metres'): v=[x/1000 for x in v] if isinstance(v,list) else v/1000
                define(pref+k.upper(),v)
    define('SEED_UINT',float(c['world']['seed']&0xffffffff))
    lines.append(f'const uint WORLD_SEED={c["world"]["seed"]&0xffffffff}u;')
    for k,v in c['erosion'].items():
        if not isinstance(v,list): define('E_'+k.upper(),v)
    for k,v in c['dunes'].items():
        if not isinstance(v,list): define('D_'+k.upper(),v/1000 if k.endswith('_metres') else v)
    define('WARP_AMOUNT',c['continents']['warp_metres']/1000)
    define('COAST_BAND',c['coast']['vertical_band_metres']/1000)
    for k,v in c['appearance'].items():
        if k.startswith('surface_'): continue
        if k in ('shadow_steps',): lines.append(f'#define A_{k.upper()} {int(v)}')
        else: define('A_'+k.upper(),v/1000 if k.endswith('_metres') else v)
    pre='\n'.join(lines)+'\n'+(ROOT/'shaders/common.glsl').read_text()
    return {'config':c,'bank':bank,'vertex':VERTEX,
            'height':pre+(ROOT/'shaders/world.glsl').read_text()+(ROOT/'shaders/height.glsl').read_text(),
            'display':pre+(ROOT/'shaders/display.glsl').read_text()}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path=self.path.split('?',1)[0]
        try:
            if path=='/bundle.json': payload=json.dumps(bundle(load_config()),allow_nan=False).encode();mime='application/json'
            elif path in ('/','/index.html','/viewer.js'):
                name='index.html' if path in ('/','/index.html') else 'viewer.js'
                payload=(ROOT/name).read_bytes();mime='text/html' if name.endswith('.html') else 'text/javascript'
            elif path=='/favicon.ico': self.send_response(204);self.end_headers();return
            else: self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mime+'; charset=utf-8')
            self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store')
            self.end_headers();self.wfile.write(payload)
        except (BrokenPipeError,ConnectionResetError): pass
        except Exception as e:
            print(f'Configuration error: {e}',file=sys.stderr)
            data=json.dumps({'error':str(e)}).encode();self.send_response(500)
            self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(data)
    def log_message(self,*args): pass


def main():
    try:
        c=load_config();bundle(c)
        server=ThreadingHTTPServer(('127.0.0.1',int(c['viewer']['port'])),Handler)
    except (ValueError,KeyError,TypeError,OSError,tomllib.TOMLDecodeError) as e:
        raise SystemExit(f'Cannot start: {e}\nCheck config.toml and whether another copy is using the port.')
    url=f'http://127.0.0.1:{server.server_port}'
    print(f'{VERSION}\n{url}\nDrag to pan; wheel to zoom. Edit config.toml and refresh. Ctrl+C to stop.',flush=True)
    if c['viewer']['open_browser']:webbrowser.open(url)
    try:server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
