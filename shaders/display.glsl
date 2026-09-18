uniform sampler2D uGround,uEnvironment,uWet;
uniform vec2 uTargetSize;
uniform float uBorder;
uniform int uOverlay;
out vec4 outColor;
vec4 ground(ivec2 p){return texelFetch(uGround,clamp(p,ivec2(0),textureSize(uGround,0)-1),0);}
uint plateAt(vec3 n){
 ivec2 sz=textureSize(uPlates,0);vec2 q=sphereUV(n)*vec2(sz)-.5;
 return texelFetch(uPlates,texWrap(ivec2(floor(q)),sz),0).r;
}
vec3 plateColour(uint id){
 float f=fract(float(id)*0.377);
 return .45+.45*cos(6.28318*(f+vec3(0.,.33,.67)));
}
void main(){
 vec3 n,p;float facing;
 if(!locate(gl_FragCoord.xy,n,p,facing)){
  vec2 xy=(gl_FragCoord.xy/uSize-.5)*vec2(uSpan,uSpan*uSize.y/uSize.x);float r=length(xy)/RADIUS;
  vec3 space=vec3(.017,.026,.044);space+=vec3(.06,.14,.24)*exp(-max(0.,r-1.)*105.);
  outColor=vec4(space,1);return;
 }
 ivec2 q=ivec2(gl_FragCoord.xy)+ivec2(int(uBorder));vec4 a=ground(q);
 vec4 env=texelFetch(uEnvironment,q,0),wet=texelFetch(uWet,q,0);
 float fp=uPixel/max(.09,facing);float h=a.x;
 // Numerical normals from the actual displayed height, not the erosion filter's
 // approximate slope steering. Correct for orthographic foreshortening.
 vec3 nr,pr,nu,pu;float junk;
 locate(gl_FragCoord.xy+vec2(1,0),nr,pr,junk);locate(gl_FragCoord.xy+vec2(0,1),nu,pu,junk);
 float dx=(ground(q+ivec2(1,0)).x-ground(q-ivec2(1,0)).x)*.5;
 float dy=(ground(q+ivec2(0,1)).x-ground(q-ivec2(0,1)).x)*.5;
 vec3 tx=(pr-p)+n*dx,ty=(pu-p)+n*dy;
 vec3 normal=normalize(cross(tx,ty));if(dot(normal,n)<0.)normal=-normal;
 if(dot(normal,n)<.001)normal=n;
 float bx=dx-(ground(q+ivec2(1,0)).z-ground(q-ivec2(1,0)).z)*A_CANOPY_HEIGHT_METRES*.5;
 float by=dy-(ground(q+ivec2(0,1)).z-ground(q-ivec2(0,1)).z)*A_CANOPY_HEIGHT_METRES*.5;
 vec3 bareNormal=normalize(cross((pr-p)+n*bx,(pu-p)+n*by));if(dot(bareNormal,n)<0.)bareNormal=-bareNormal;
 float slope=sqrt(max(0.,1.-pow(dot(bareNormal,n),2.)))/max(.001,dot(bareNormal,n));
 float temp=env.x,rain=exp(env.y),arid=rain/(450.+max(0.,temp)*45.);
 vec3 colour;
 if(uOverlay==0){
 float moisture=ramp(.30,1.5,arid),forest=wet.w;
 float desert=(1.-ramp(.32,.80,arid))*ramp(2.,16.,temp);
 vec3 grass=mix(A_GRASS,A_FOREST,forest*.88);
 colour=mix(grass,A_DESERT,desert);
 colour=mix(colour,vec3(.35,.35,.28),ramp(4.,-7.,temp)*.40);
 // Distinguish bedrock and arid plateaus from ice: a high dry plateau isn't white.
 float rock=ramp(tan(radians(EC_ROCK_SLOPE_DEGREES.x)),tan(radians(EC_ROCK_SLOPE_DEGREES.y)),slope);
 rock=max(rock,desert*ramp(.8,2.3,h)*.7);rock=max(rock,env.z*ramp(2.5,4.8,h)*(1.-forest*.5)*.55);
 vec3 lith=mix(A_ROCK*vec3(.89,.95,1.02),A_ROCK*vec3(1.10,1.02,.88),env.w);
 colour=mix(colour,lith,rock);
 float snow=ramp(EC_SNOW_TEMPERATURE_C.y,EC_SNOW_TEMPERATURE_C.x,temp)*(1.-ramp(1.,2.,slope));
 colour=mix(colour,A_SNOW,snow);
 // Reflectance bands live in ECEF metres, not screen coordinates. No minimum
 // line width or fake crispness when a feature is smaller than a pixel.
 float material=0.;for(int i=0;i<S_N;i++){
  float b=footprintWeight(L_SCALE[S_IDS[i]],fp);if(b>0.)material+=nd3(p,S_IDS[i]).x*S_AMP[i]*b;
 }
 colour*=1.+material;
 if(a.z>.001){colour=mix(colour,A_FOREST*(.85+.4*a.z),clamp01(a.z*4.));}
 if(wet.x>.001){
  float depth=wet.y;float shallow=exp(-depth/.16);
  vec3 ocean=mix(A_OCEAN_DEEP,A_OCEAN_SHALLOW,shallow*.7);
  if(A_BATHYMETRY){
   // Cartographic seafloor shading, like a bathymetric map; not transparent water.
   float dhx=(sphereMap(uMacro,nr).x-sphereMap(uMacro,n).x)/max(length(pr-p),.001);
   float dhy=(sphereMap(uMacro,nu).x-sphereMap(uMacro,n).x)/max(length(pu-p),.001);
   float relief=clamp((-dhx+dhy)*4.,-.18,.18);
   ocean*=1.+relief;
  }
  vec3 water=mix(ocean,A_LAKE,clamp01(wet.z));
  colour=mix(colour,water,clamp01(wet.x));
 }
 // Cartographic overhead illumination. No specular term on soil or vegetation.
 // Camera-relative light keeps all inspected geography readable; modest limb
 // darkening provides the globe shape without a night-side hiding half the map.
 vec3 sun=normalize(uCentre*.90-uEast*.55+uNorth*.65);
 float baseLight=max(.2,dot(n,sun)),localLight=max(.0,dot(normal,sun));
 float hill=clamp(localLight/max(baseLight,.3),.30,1.6);
 float shade=mix(1.,hill,A_HILLSHADE_STRENGTH*(1.-wet.x));
 shade*=.60+.40*sqrt(max(0.,facing));colour*=shade;
 // Thin atmosphere at the limb only; not a fog filter over detailed maps.
 float haze=pow(1.-facing,5.)*.35;
 colour=mix(colour,vec3(.25,.40,.54),haze);
 }else{
  // Data overlays: flat cartographic colour, no hillshade. All inputs are
  // already-bound guide textures or field buffers; no extra passes.
  vec4 tect=sphereMap(uTectonics,n),crust=sphereMap(uCrust,n),flow=sphereMap(uFlow,n);
  if(uOverlay==1){float t=clamp01((h+5.)/11.);colour=vec3(t);}
  else if(uOverlay==2){colour=h>0.?vec3(.92):vec3(.02,.03,.05);colour=mix(colour,vec3(.1,.5,.55),clamp01(wet.z));}
  else if(uOverlay==3){float t=clamp01((temp+45.)/90.);vec3 c1=vec3(.25,.2,.6),c2=vec3(.9,.93,.95),c3=vec3(.9,.8,.2),c4=vec3(.8,.15,.1);colour=t<.33?mix(c1,c2,t/.33):(t<.66?mix(c2,c3,(t-.33)/.33):mix(c3,c4,(t-.66)/.34));}
  else if(uOverlay==4){float t=clamp01((env.y-3.7)/4.2);vec3 c1=vec3(.45,.3,.15),c2=vec3(.2,.5,.25),c3=vec3(.2,.4,.8);colour=t<.5?mix(c1,c2,t*2.):mix(c2,c3,t*2.-1.);}
  else if(uOverlay==5){colour=vec3(clamp01(env.z));}
  else if(uOverlay==6){colour=clamp01(vec3(tect.x,tect.y,tect.z)*1.2);}
  else if(uOverlay==7){float t=clamp01(tect.w/250.);colour=h>0.?vec3(.08):mix(vec3(.1,.7,.8),vec3(.02,.05,.35),t);}
  else if(uOverlay==8){float t=clamp01(crust.y/70.);colour=h>0.?vec3(t):vec3(.02,.03,.06);}
  else if(uOverlay==9){float wtr=max(wet.x,wet.z);vec3 base=h>0.?vec3(.12):vec3(.01,.02,.03);colour=mix(base,vec3(.2,.75,.9),clamp01(wtr));colour=mix(colour,vec3(.1,.3,.9),clamp01(flow.a)*.5*(1.-clamp01(wtr)));}
  else{uint id=plateAt(n);colour=plateColour(id)*(h>0.?1.:.4);}
 }
 outColor=vec4(clamp(colour,0.,1.),1.);
}
