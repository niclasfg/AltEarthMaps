/* Origin-rebased noise adapter. The noised/Phacelle expressions derive from
   Rune Skovbo Johansen's MPL-2.0 code in runevision.glsl (retained unmodified).
   Changes: integer hashing; split lattice origin; gradients rotated to world.
   Cells are hashed as INTEGERS, never converted to huge imprecise float keys. */
uint mixbits(uint x){x^=x>>16u;x*=0x7feb352du;x^=x>>15u;x*=0x846ca68bu;x^=x>>16u;return x;}
vec2 ihash(ivec2 p){uvec2 q=uvec2(p);uint h=mixbits(q.x^mixbits(q.y+0x9e3779b9u)^WORLD_SEED);
return vec2(h>>8u,mixbits(h^0x68bc21ebu)>>8u)/16777216.0;}
vec2 gradHash(ivec2 p){return ihash(p)*2.0-1.0;}
mat2 rot(int i){return mat2(L_COS[i],L_SIN[i],-L_SIN[i],L_COS[i]);}
vec2 coord(vec2 p,int i){return rot(i)*p/L_SCALE[i]+uFraction[i];}
float ramp(float a,float b,float x){float t=clamp01((x-a)/(b-a));return t*t*(3.0-2.0*t);}
float rampDerivative(float a,float b,float x){float t=clamp01((x-a)/(b-a));return 6.0*t*(1.0-t)/(b-a);}
vec2 safe_normalize(vec2 n){float l=length(n);return l>1e-10?n/l:n;}
float pow_inv(float t,float power){return 1.0-pow(1.0-clamp01(t),power);}
float ease_out(float t){float v=1.0-clamp01(t);return 1.0-v*v;}
float smooth_start(float t,float s){return t>=s?t-.5*s:.5*t*t/max(s,1e-12);}

// Value and exact derivative of the INPUT gradient noise, in world km units.
vec3 nd(vec2 p,int id){
 vec2 q=coord(p,id);ivec2 cell=ivec2(floor(q))+uCell[id];vec2 f=fract(q);
 vec2 u=f*f*f*(f*(f*6.0-15.0)+10.0),du=30.0*f*f*(f*(f-2.0)+1.0);
 vec2 ga=gradHash(cell),gb=gradHash(cell+ivec2(1,0));
 vec2 gc=gradHash(cell+ivec2(0,1)),gd=gradHash(cell+ivec2(1,1));
 float va=dot(ga,f),vb=dot(gb,f-vec2(1,0)),vc=dot(gc,f-vec2(0,1)),vd=dot(gd,f-1.0);
 float n=va+u.x*(vb-va)+u.y*(vc-va)+u.x*u.y*(va-vb-vc+vd);
 vec2 d=ga+u.x*(gb-ga)+u.y*(gc-ga)+u.x*u.y*(ga-gb-gc+gd)
      +du*(u.yx*(va-vb-vc+vd)+vec2(vb,vc)-va);
 return vec3(n,transpose(rot(id))*d/L_SCALE[id]);
}
// The 4x4 compact Gaussian support and partial phase normalization are retained.
vec4 phacelle(vec2 p,int id,vec2 direction,float cellScale,float normalization){
 vec2 q=coord(p,id),f=fract(q);ivec2 cell=ivec2(floor(q))+uCell[id];
 vec2 side=direction.yx*vec2(-1,1)*cellScale*TAU;
 vec2 sum=vec2(0);float total=0.0;
 for(int i=-1;i<=2;i++)for(int j=-1;j<=2;j++){
   ivec2 o=ivec2(i,j);vec2 d=f-vec2(o)-gradHash(cell+o)*.5;
   float w=max(0.0,exp(-dot(d,d)*2.0)-.01111);
   float phase=dot(d,side)+.25*TAU;
   sum+=vec2(cos(phase),sin(phase))*w;total+=w;
 }
 vec2 z=sum/max(total,1e-8);z/=max(1.0-normalization,length(z));
 return vec4(z,side);
}

// Finite pixel-footprint approximation used only by the image renderer.
// Set to ZERO to evaluate the fixed full-resolution world for point queries.
uniform float uFilterPixel;
float band(float wavelength){return 1.0-ramp(.18,.55,uFilterPixel/wavelength);}
vec3 filteredNd(vec2 p,int id){float w=band(L_SCALE[id]);return w>0.?nd(p,id)*w:vec3(0);}
