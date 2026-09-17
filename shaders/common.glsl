/* Spherical-coordinate and precision adapter; units inside shaders are km.
   Every detail lattice is rebased using the JavaScript camera's double-precision
   ECEF origin. The globe has no longitude seam in its procedural noise. */
uniform vec3 uCentre,uEast,uNorth;
uniform vec2 uSize;
uniform float uSpan,uPixel;
uniform ivec3 uCell[L_N];
uniform vec3 uFraction[L_N];
uniform sampler2D uMacro,uClimate,uWater,uTectonics,uCrust;
uniform highp usampler2D uRiverHeader,uRiverIndex;
uniform sampler2D uRivers;
uint mixbits(uint h){h^=h>>16u;h*=0x7feb352du;h^=h>>15u;h*=0x846ca68bu;return h^(h>>16u);}
uint ih(ivec3 p){uvec3 q=uvec3(p);return mixbits(q.x^mixbits(q.y+0x9e3779b9u)^mixbits(q.z+WORLD_SEED));}
float rand(ivec3 p){return float(ih(p)>>8u)/16777216.0;}
vec2 hash2(ivec2 p,int plane){uint h=ih(ivec3(p,plane*7919));return vec2(h>>8u,mixbits(h^0x68bc21ebu)>>8u)/16777216.*2.-1.;}
float ramp(float a,float b,float x){float t=clamp01((x-a)/(b-a));return t*t*(3.-2.*t);}
float footprintWeight(float wavelength,float footprint){return 1.-ramp(.12,.48,footprint/wavelength);}
vec2 sphereUV(vec3 n){return vec2(atan(n.z,n.x)/TAU+.5,asin(clamp(n.y,-1.,1.))/PI+.5);}
ivec2 texWrap(ivec2 p,ivec2 s){return ivec2((p.x%s.x+s.x)%s.x,clamp(p.y,0,s.y-1));}
vec4 sphereMap(sampler2D tex,vec3 n){
 ivec2 sz=textureSize(tex,0);vec2 q=sphereUV(n)*vec2(sz)-.5;ivec2 i=ivec2(floor(q));vec2 f=fract(q);
 vec4 a=texelFetch(tex,texWrap(i,sz),0),b=texelFetch(tex,texWrap(i+ivec2(1,0),sz),0);
 vec4 c=texelFetch(tex,texWrap(i+ivec2(0,1),sz),0),d=texelFetch(tex,texWrap(i+ivec2(1),sz),0);
 return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);
}
// Stable orthographic sphere intersection. Avoid subtracting two 6371-km
// values when the visible patch is only a few metres across.
bool locate(vec2 screen,out vec3 n,out vec3 local,out float foreshorten){
 vec2 xy=(screen/uSize-.5)*vec2(uSpan,uSpan*uSize.y/uSize.x);
 float r2=dot(xy,xy),z2=RADIUS*RADIUS-r2;
 if(z2<=0.){n=vec3(0);local=vec3(0);foreshorten=0.;return false;}
 float z=sqrt(z2);float dz=-r2/(RADIUS+z);
 local=uEast*xy.x+uNorth*xy.y+uCentre*dz;
 n=normalize(uCentre+local/RADIUS);foreshorten=z/RADIUS;return true;
}
vec4 nd3(vec3 p,int id){
 vec3 q=p/L_SCALE[id]+uFraction[id];ivec3 cell=ivec3(floor(q))+uCell[id];vec3 f=fract(q);
 vec3 u=f*f*f*(f*(f*6.-15.)+10.),du=30.*f*f*(f*(f-2.)+1.);
 float h=0.;vec3 g=vec3(0);
 for(int z=0;z<2;z++)for(int y=0;y<2;y++)for(int x=0;x<2;x++){
  ivec3 o=ivec3(x,y,z);vec3 w=mix(1.-u,u,vec3(o));float a=rand(cell+o)*2.-1.;
  h+=a*w.x*w.y*w.z;vec3 sg=vec3(o)*2.-1.;
  g+=a*sg*du*vec3(w.y*w.z,w.x*w.z,w.x*w.y);
 }
 return vec4(h,g/L_SCALE[id]);
}
vec2 proj(vec3 v,int chart){return chart==0?v.yz:(chart==1?v.xz:v.xy);}
ivec2 projI(ivec3 v,int chart){return chart==0?v.yz:(chart==1?v.xz:v.xy);}
vec3 unproj(vec2 v,int chart){return chart==0?vec3(0.,v):(chart==1?vec3(v.x,0.,v.y):vec3(v,0.));}
// Phacelle Noise / Advanced Terrain Erosion Filter:
// copyright (c) 2025 Rune Skovbo Johansen. MPL-2.0; see licenses/ and
// runevision_reference.glsl. Changes here are input chart, hash and scale adapter.
vec4 phacelle(vec3 p,int id,int chart,vec2 direction,float cellScale,float normalization){
 vec2 q=proj(p/L_SCALE[id]+uFraction[id],chart),f=fract(q);
 ivec2 cell=ivec2(floor(q))+projI(uCell[id],chart);
 vec2 side=direction.yx*vec2(-1.,1.)*cellScale*TAU,sum=vec2(0);float total=0.;
 for(int i=-1;i<=2;i++)for(int j=-1;j<=2;j++){
  ivec2 o=ivec2(i,j);vec2 d=f-vec2(o)-hash2(cell+o,chart)*.5;
  float w=max(0.,exp(-dot(d,d)*2.)-.01111);
  float t=dot(d,side)+.25*TAU;sum+=vec2(cos(t),sin(t))*w;total+=w;
 }
 vec2 z=sum/max(total,1e-8);z/=max(1.-normalization,length(z));return vec4(z,side);
}
float ease_out(float t){float v=1.-clamp01(t);return 1.-v*v;}
float smooth_start(float t,float s){return t>=s?t-.5*s:.5*t*t/max(s,1e-10);}
vec2 safe_normalize(vec2 v){float l=length(v);return l>1e-10?v/l:v;}
vec3 chartWeights(vec3 n){vec3 w=pow(abs(n),vec3(8));return w/(w.x+w.y+w.z);}
