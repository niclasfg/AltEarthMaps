// Physical-unit orthographic display. Biome appearance is a function of local
// climate, bare-earth height, estimated ecological slope, actual surface slope,
// aspect-adjusted moisture, water and sediment — not random polygon colours.
uniform sampler2D uHeight;uniform sampler2D uClimate;uniform sampler2D uCover;
uniform sampler2D uShadow;
uniform vec2 uSize;uniform vec2 uOrigin;uniform float uPixel;
uniform vec2 uShadowOrigin;uniform float uShadowPixel;uniform vec2 uShadowSize;
uniform float uBorder;
out vec4 outColor;
vec4 sampleMap(sampler2D t,vec2 q,vec2 size){
 vec2 v=q-.5;ivec2 i=ivec2(floor(v)),hi=ivec2(size)-1;vec2 f=fract(v);
 return mix(mix(texelFetch(t,clamp(i,ivec2(0),hi),0),texelFetch(t,clamp(i+ivec2(1,0),ivec2(0),hi),0),f.x),
            mix(texelFetch(t,clamp(i+ivec2(0,1),ivec2(0),hi),0),texelFetch(t,clamp(i+ivec2(1),ivec2(0),hi),0),f.x),f.y);
}
float contextHeight(vec2 p){
 vec2 uv=(p-uOrigin)/uPixel;
 if(all(greaterThanEqual(uv,vec2(1)))&&all(lessThan(uv,uSize-1.)))return sampleMap(uHeight,uv,uSize).x;
 return sampleMap(uShadow,(p-uShadowOrigin)/uShadowPixel,uShadowSize).x;
}
float shadow(vec2 p,float h,vec3 sun){
 if(!A_SHADOWS)return 1.;
 float v=1.,d=max(.003,uPixel*1.7);
 for(int i=0;i<A_SHADOW_STEPS;i++){
  float clear=h+sun.y*d-contextHeight(p+sun.xz*d);
  v=min(v,clamp01(16.*(clear+uPixel*.8)/d));d=d*1.27+.003;
  if(d>35.||v<=0.)break;
 }
 return v;
}
void main(){
 vec2 q=gl_FragCoord.xy+uBorder,p=uOrigin+q*uPixel;
 vec4 h=sampleMap(uHeight,q,uSize),cl=sampleMap(uClimate,q,uSize),cov=sampleMap(uCover,q,uSize);
 float dx=(sampleMap(uHeight,q+vec2(1,0),uSize).x-sampleMap(uHeight,q-vec2(1,0),uSize).x)/(2.*uPixel);
 float dy=(sampleMap(uHeight,q+vec2(0,1),uSize).x-sampleMap(uHeight,q-vec2(0,1),uSize).x)/(2.*uPixel);
 float ex=(sampleMap(uHeight,q+vec2(1,0),uSize).y-sampleMap(uHeight,q-vec2(1,0),uSize).y)/(2.*uPixel);
 float ey=(sampleMap(uHeight,q+vec2(0,1),uSize).y-sampleMap(uHeight,q-vec2(0,1),uSize).y)/(2.*uPixel);
 vec3 normal=normalize(vec3(-dx,1,-dy));
 float actualSlope=length(vec2(ex,ey));
 bool water=cl.w>h.y;
 vec3 colour;float specular=0.;
 if(water){
  float depth=cl.w-h.y;
  float shallow=exp(-depth/(cl.w>W_SEA_LEVEL_METRES+.001?.012:.040));
  colour=mix(cl.w>W_SEA_LEVEL_METRES+.001?A_LAKE_DEEP:A_OCEAN_DEEP,A_OCEAN_SHALLOW,shallow);
  float ice=ramp(-1.,-9.,cl.x);colour=mix(colour,A_SNOW*.82,ice);
  normal=vec3(0,1,0);specular=.06*(1.-ice);
 }else{
  float t=cl.x,a=cl.y,forest=cov.x;
  float dryness=1.-ramp(.25,.9,a),cold=1.-ramp(-4.,8.,t);
  float tropical=ramp(18.,27.,t)*ramp(.8,1.6,a);
  vec3 tree=mix(A_FOREST,A_TROPICAL_FOREST,tropical);
  tree=mix(tree,A_BOREAL_FOREST,ramp(13.,3.,t));
  vec3 base=mix(A_DRY_SOIL,A_GRASS,ramp(.20,.80,a));
  base=mix(base,A_TUNDRA,cold);
  base=mix(base,A_SAND,dryness*ramp(4.,18.,t));
  // Correct area-averaged forest colour at continental scale; actual crowns at
  // close scale. No tiny crowns sampled every kilometre and painted as giant trees.
  float resolved=1.-ramp(.0025,.008,uPixel);
  float cover=mix(forest,ramp(.02,.12,cov.y),resolved);
  base=mix(base,tree,cover*.92);
  float rock=ramp(tan(radians(EC_ROCK_SLOPE_START_DEGREES)),tan(radians(EC_ROCK_SLOPE_END_DEGREES)),max(actualSlope,cov.z));
  rock=max(rock,(1.-ramp(-8.,4.,t))*.6*(1.-forest));
  base=mix(base,A_ROCK,rock*(1.-cl.z));
  base=mix(base,A_SAND,cl.z);
  float grain=nd(p,MATERIAL_ID).x;
  base*=1.+.10*grain;
  vec3 micro=vec3(0);float variation=0.;
  for(int i=0;i<S_N;i++){
   float w=1.-ramp(.18,.5,uPixel/L_SCALE[S_IDS[i]]);
   if(w<=0.)continue;
   vec3 n=nd(p,S_IDS[i]);micro+=n*S_HEIGHT[i]*w;variation+=n.x*S_COLOUR[i]*w;
  }
  float materialBump=mix(.30,1.,rock)*(1.-cover*.7);
  normal=normalize(vec3(-dx-micro.y*materialBump,1.,-dy-micro.z*materialBump));
  base*=1.+variation;
  float snow=ramp(EC_SNOW_TEMPERATURE_NONE_C,EC_SNOW_TEMPERATURE_FULL_C,t);
  // Steep faces retain exposed rock through snow; no blanket snow paint cliffs.
  snow*=mix(1.,.45,ramp(.6,1.8,actualSlope));base=mix(base,A_SNOW,snow);
  float shore=exp(-max(0.,h.y-W_SEA_LEVEL_METRES)/.008)*(1.-ramp(.05,.5,actualSlope));
  base=mix(base,A_SAND,shore*(1.-snow));
  colour=base;
 }
 vec3 sun=normalize(A_SUN_DIRECTION);float sh=shadow(p,h.x,sun);
 float diffuse=max(0.,dot(normal,sun));
 // Matte, broad ambient illumination. Do not turn small gullies into white
 // 'rivers'; this version makes no claim of global flow-routed river networks.
 vec3 lit=colour*(vec3(.40,.44,.49)+vec3(1.30,1.24,1.15)*diffuse*mix(.18,1.,sh));
 if(water)lit+=specular*pow(max(0.,dot(reflect(-sun,normal),vec3(0,1,0))),60.);
 // Display transfer only. No local-contrast enhancement or sharpening.
 lit=pow(max(lit,vec3(0)),vec3(1./2.2));
 outColor=vec4(clamp(lit,0.,1.),1.);
}
