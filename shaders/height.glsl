// Three render targets: terrain surface/bare earth; climate/water; vegetation.
uniform vec2 uOrigin;   // Relative to the CPU-rebased noise-bank anchor, in km.
uniform float uPixel;
layout(location=0) out vec4 outHeight;
layout(location=1) out vec4 outClimate;
layout(location=2) out vec4 outCover;

float canopy(vec2 p,float forest){
 vec2 q=coord(p,CANOPY_ID),f=fract(q);ivec2 cell=ivec2(floor(q))+uCell[CANOPY_ID];
 float result=0.;
 for(int i=-1;i<=1;i++)for(int j=-1;j<=1;j++)for(int k=0;k<2;k++){
  ivec2 key=cell+ivec2(i,j)+ivec2(k*7241,-k*3943);
  vec2 h=ihash(key),v=ihash(key+ivec2(2713,871));
  if(v.x>forest*.78)continue;
  vec2 d=f-(vec2(i,j)+h);float radius=mix(.20,.46,v.y);
  float rr=dot(d,d)/(radius*radius);if(rr>=1.)continue;
  result=max(result,pow(1.-rr,.65)*mix(.5,1.,h.y));
 }
 return result;
}
void main(){
 vec2 p=uOrigin+gl_FragCoord.xy*uPixel;
 LandData land=landAt(p);WaterData water=lakesAt(p,land.z,land.slope);
 if(water.ground<W_SEA_LEVEL_METRES)water.level=max(water.level,W_SEA_LEVEL_METRES);
 vec4 eco=ecology(land,water);float trees=0.;
 // Canopy is an object layer, not bare terrain. It is area-averaged when crowns
 // become subpixel; the actual bare-earth function does not depend on uPixel.
 if(A_CANOPY&&eco.z>.001){
  float resolve=1.-ramp(.0025,.008,uPixel);
  trees=mix(eco.z*.38,resolve>.001?canopy(p,eco.z):0.,resolve);
 }
 float surface=water.ground+trees*A_CANOPY_HEIGHT_METRES;
 if(water.level>water.ground){surface=water.level;trees=0.;}
 outHeight=vec4(surface,water.ground,land.ridge,land.erosion);
 outClimate=vec4(eco.xy,land.dune,water.level);
 outCover=vec4(eco.z,trees,eco.w,water.proximity);
}
