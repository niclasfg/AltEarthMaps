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


def checked_config():
    # Layer validation; tectonic widths naturally run core -> foreland instead.
    c=tomllib.loads((ROOT/'config.toml').read_text())
    specs=[('planet','continent_wavelength_metres',['continent_amplitude']),
           ('relief','wavelength_metres',['amplitude_metres','collision_gain','rift_gain','transform_gain']),
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
    for s,k in [('planet','map_width'),('hydrology','routing_width')]:
        v=c[s][k]
        if v not in (256,512,1024,2048,4096):raise ValueError(f'{s}.{k}: use 256,512,1024,2048 or 4096.')
    if c['planet']['map_width']%c['hydrology']['routing_width']:raise ValueError('routing_width must divide map_width.')
    hy=c['hydrology'];steps=len(c['erosion']['wavelength_metres'])
    if not all(isinstance(hy[k],(int,float)) and not isinstance(hy[k],bool) and math.isfinite(hy[k]) for k in ('gully_stream_onset','gully_stream_softness','trunk_display_wavelength_metres')):raise ValueError('hydrology: finite numbers required for the stream settings.')
    if not 0<=hy['gully_stream_onset']<=steps:raise ValueError(f'gully_stream_onset must be 0–{steps} (one unit per displayed erosion step).')
    if not 0<hy['gully_stream_softness']<=steps:raise ValueError(f'gully_stream_softness must be above 0 and at most {steps}.')
    if not 1000<=hy['trunk_display_wavelength_metres']<=400000:raise ValueError('trunk_display_wavelength_metres must be between 1 km and 400 km.')
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
    define('RADIUS',c['planet']['radius_metres']/1000);define('LAPSE',c['climate']['lapse_c_per_km'])
    lines.append(f'const uint WORLD_SEED={int(c["planet"]["seed"])&0xffffffff}u;')
    lines.append(f'#define MAX_RIVER_BIN {max(1,int(meta["max_bin"]))}')
    common='\n'.join(lines)+'\n'+(ROOT/'shaders/common.glsl').read_text()
    vertex='#version 300 es\nvoid main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.-1.,0.,1.);}'
    return {'config':c,'meta':meta,'bank':bank,'vertex':vertex,
            'height':common+(ROOT/'shaders/terrain.glsl').read_text(),
            'display':common+(ROOT/'shaders/display.glsl').read_text()}


def main():
    try:
        c=checked_config()
        from planet import prepare
        folder,meta=prepare(c,ROOT)
        data=json.dumps(bundle(c,meta)).encode()
    except ImportError as e:raise SystemExit(f'{e}\nInstall dependencies: python -m pip install -r requirements.txt')
    except (OSError,ValueError,KeyError,TypeError,tomllib.TOMLDecodeError) as e:raise SystemExit(f'Cannot start: {e}\nCheck config.toml.')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            name=self.path.split('?',1)[0]
            if name=='/bundle.json':body=data;mime='application/json'
            else:
                if name.startswith('/data/'):path=folder/name.removeprefix('/data/')
                else:path=ROOT/('index.html' if name=='/' else name.lstrip('/'))
                path=path.resolve()
                if not path.is_relative_to(ROOT) or not path.is_file():self.send_error(404);return
                body=path.read_bytes();mime=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-cache');self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
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
