uniform sampler2D uGround,uEnvironment,uWet;
uniform vec2 uTargetSize;
uniform float uBorder;
out vec4 outColor;
vec4 ground(ivec2 p){return texelFetch(uGround,clamp(p,ivec2(0),textureSize(uGround,0)-1),0);}
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
 float moisture=ramp(.30,1.5,arid),forest=wet.w;
 float desert=(1.-ramp(.32,.80,arid))*ramp(2.,16.,temp);
 vec3 grass=mix(A_GRASS,A_FOREST,forest*.88);
 vec3 colour=mix(grass,A_DESERT,desert);
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
 outColor=vec4(clamp(colour,0.,1.),1.);
}
