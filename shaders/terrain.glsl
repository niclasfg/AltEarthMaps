layout(location=0) out vec4 outGround; // display height, exact ground, canopy, coverage
layout(location=1) out vec4 outEnvironment; // T, log rainfall, land ruggedness, lithology
layout(location=2) out vec4 outWater; // water coverage, depth, freshwater, ridge mask
uniform bool uExact;

vec3 macroGradient(vec3 n){
 vec3 east=normalize(abs(n.y)<.99?cross(vec3(0,1,0),n):cross(vec3(1,0,0),n));vec3 north=cross(n,east);
 float d=.00025;
 float a=sphereMap(uMacro,normalize(n+east*d)).x-sphereMap(uMacro,normalize(n-east*d)).x;
 float b=sphereMap(uMacro,normalize(n+north*d)).x-sphereMap(uMacro,normalize(n-north*d)).x;
 return (east*a+north*b)/(2.*d*RADIUS);
}
// Full recurrence state never depends on pixel footprint. A separate accumulated
// height is filtered for display. This isn't exact integration of nonlinear
// erosion, but avoids changing the parent terrain and its gully directions.
vec3 erode(vec3 p,vec3 n,vec3 inputSlope,float fade,float terrainAmp,float footprint,vec3 history,float ruggedness){
 vec3 weights=chartWeights(n);float exact=0.,display=0.,ridge=0.;
 for(int chart=0;chart<3;chart++){
  float weight=weights[chart];if(weight<.0001)continue;
  float nr=n[chart];vec2 g0=proj(inputSlope,chart)-inputSlope[chart]*proj(n,chart)/nr;
  float slope=max(length(g0),1e-10);vec2 gs=safe_normalize(g0)*E_ASSUMED_SLOPE;
  float f=fade,rounding=mix(E_CREASE_ROUNDING[0],E_RIDGE_ROUNDING[0],clamp01(f+.5))*E_INITIAL_ROUNDING_MULTIPLIER;
  float mask=ease_out(smooth_start(slope*1.25,rounding*1.25));
  float rm=ease_out(slope*2.8),rf=fade,he=0.,hd=0.;
  for(int k=0;k<E_N;k++){
   // At an unresolved scale the displayed contribution vanishes, but we retain
   // every ancestor needed by later scales. Exact queries always run all levels.
   float vis=footprintWeight(E_WAVELENGTH_METRES[k],footprint);
   if(!uExact && vis==0.)break;
   vec4 ph=phacelle(p,E_IDS[k],chart,safe_normalize(gs),E_CELL_SCALE[k],E_NORMALIZATION[k]);
   ph.zw*=-1./L_SCALE[E_IDS[k]];
   float response=R_PLAIN_FRACTION+(1.-R_PLAIN_FRACTION)*clamp01(ruggedness*E_COLLISION_GAIN[k]+history.y*E_RIFT_GAIN[k]+history.z*E_TRANSFORM_GAIN[k]);
   float amp=E_AMPLITUDE_METRES[k]*terrainAmp*response;
   gs+=sign(ph.y)*ph.zw*amp*E_GULLY_WEIGHT[k];
   float faded=mix(f,ph.x*E_GULLY_WEIGHT[k],mask);
   float delta=(faded-.20)*amp;he+=delta;hd+=delta*vis;f=faded;
   float r=mix(E_CREASE_ROUNDING[k],E_RIDGE_ROUNDING[k],clamp01(ph.x+.5));
   r*=mix(E_OLD_BELT_ROUNDING_MULTIPLIER,1.,history.x);
   float next=ease_out(smooth_start(abs(ph.y)*E_ONSET[k],r*E_ONSET[k]));
   mask=(1.-pow(1.-clamp01(mask),E_DETAIL[k]))*next;
   rf=mix(rf,ph.x,rm);rm*=ease_out(abs(ph.y)*1.5);
  }
  exact+=he*weight;display+=hd*weight;ridge+=rf*(1.-rm)*weight;
 }
 return vec3(exact,display,ridge);
}

// Cube-face spatial lookup for the vector river skeleton. Every bin contains
// all reach supports touching it, including supports from neighbouring faces.
uvec2 riverBin(vec3 n){
 vec3 a=abs(n);int axis=a.x>a.y?(a.x>a.z?0:2):(a.y>a.z?1:2);
 int face=axis*2+(n[axis]<0.?1:0);vec2 uv=proj(n,axis)/a[axis]*.5+.5;
 ivec2 b=clamp(ivec2(uv*64.),ivec2(0),ivec2(63));b.y+=64*face;
 return texelFetch(uRiverHeader,b,0).rg;
}
vec4 riverRecord(int id,int part){int i=id*3+part;return texelFetch(uRivers,ivec2(i%1023,i/1023),0);}
vec4 river(vec3 n,float footprint){
 uvec2 bin=riverBin(n);float best=1e10,water=-1e4,width=0.,bank=0.;
 for(int j=0;j<MAX_RIVER_BIN;j++){
  if(uint(j)>=bin.y)break;uint k=bin.x+uint(j);
  int id=int(texelFetch(uRiverIndex,ivec2(int(k%1024u),int(k/1024u)),0).r);
  vec4 A=riverRecord(id,0),B=riverRecord(id,1),C=riverRecord(id,2);
  if(!uExact && C.x<footprint*.03)continue;
  vec3 ab=B.xyz-A.xyz;float t=clamp01(dot(n-A.xyz,ab)/max(dot(ab,ab),1e-15));
  vec3 q=normalize(mix(A.xyz,B.xyz,t));float d=length(n-q)*RADIUS;
  float w=mix(C.x,C.y,t);float relative=d-w;
  if(relative<best){best=relative;width=w;water=mix(A.w,B.w,t);bank=C.z;}
 }
 return vec4(best,water,width,bank);
}

float crown(vec3 p,vec3 n){
 vec3 weights=chartWeights(n);float result=0.;
 for(int chart=0;chart<3;chart++){
  if(weights[chart]<.0001)continue;
  vec2 q=proj(p/L_SCALE[TREE_ID]+uFraction[TREE_ID],chart),f=fract(q);
  ivec2 key=projI(uCell[TREE_ID],chart)+ivec2(floor(q));float tree=0.;
  for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){
   ivec2 o=ivec2(x,y);vec2 h=hash2(key+o,chart);vec2 d=f-vec2(o)-.5-h*.35;
   float radius=.29+.18*(h.x*.5+.5);float r2=dot(d,d)/(radius*radius);
   tree=max(tree,sqrt(max(0.,1.-r2))*(.55+.45*(h.y*.5+.5)));
  }
  result+=weights[chart]*tree;
 }
 return result;
}

void main(){
 vec3 n,p;float facing;
 if(!locate(gl_FragCoord.xy,n,p,facing)){outGround=vec4(0);outEnvironment=vec4(0);outWater=vec4(0);return;}
 float fp=uExact?0.:uPixel/max(.09,facing);
 vec4 m=sphereMap(uMacro,n),cl=sphereMap(uClimate,n),wet=sphereMap(uWater,n);
 vec4 tect=sphereMap(uTectonics,n),crust=sphereMap(uCrust,n);
 float land=ramp(-.03,.15,m.x),rug=m.y,h=m.x,shown=h;
 vec3 g=macroGradient(n),rg=vec3(0);float rough=R_PLAIN_FRACTION+(1.-R_PLAIN_FRACTION)*rug;
 float localRelief=0.;
 for(int i=0;i<R_N;i++){
  vec4 ns=nd3(p,R_IDS[i]);
  float layerRough=R_PLAIN_FRACTION+(1.-R_PLAIN_FRACTION)*clamp01(rug*R_COLLISION_GAIN[i]+tect.y*R_RIFT_GAIN[i]+tect.z*R_TRANSFORM_GAIN[i]);
  float a=R_AMPLITUDE_METRES[i]*layerRough*land;
  h+=ns.x*a;shown+=ns.x*a*footprintWeight(R_WAVELENGTH_METRES[i],fp);rg+=ns.yzw*a;localRelief+=ns.x*a;
 }
 g+=rg;g-=n*dot(n,g);
 float temp=cl.x-LAPSE*max(0.,h),aridity=exp(cl.y)/(450.+max(0.,temp)*45.);
 float erosionAmp=land*(.22+.78*ramp(.003,.05,length(g)));
 // Downward displacement is bounded on land, independent of render footprint.
 float maxDrop=0.;for(int i=0;i<E_N;i++){
  float response=R_PLAIN_FRACTION+(1.-R_PLAIN_FRACTION)*clamp01(rug*E_COLLISION_GAIN[i]+tect.y*E_RIFT_GAIN[i]+tect.z*E_TRANSFORM_GAIN[i]);
  maxDrop+=E_AMPLITUDE_METRES[i]*1.2*response;
 }
 erosionAmp=min(erosionAmp,max(0.,h-.0003)/max(maxDrop,1e-5));
 vec3 er=erode(p,n,g,clamp(localRelief/max(.05,rough),-1.,1.),erosionAmp,fp,tect.xyz,rug);
 h+=er.x;shown+=er.y;
 float slope=length(g);
 float desert=(1.-ramp(.30,.70,aridity))*ramp(6.,18.,temp);
 float dune=desert*(1.-ramp(tan(radians(D_MAXIMUM_SLOPE_DEGREES))*.4,tan(radians(D_MAXIMUM_SLOPE_DEGREES)),slope))*land;
 if(dune>.001){
  vec3 w=chartWeights(n);int chart=w.x>w.y?(w.x>w.z?0:2):(w.y>w.z?1:2);
  vec2 wind=vec2(cos(radians(D_WIND_DEGREES)),sin(radians(D_WIND_DEGREES)));
  for(int k=0;k<D_N;k++){
   float b=footprintWeight(D_WAVELENGTH_METRES[k],fp);if(!uExact&&b==0.)break;
   vec4 ph=phacelle(p,D_IDS[k],chart,wind,.9,.45);
   float v=pow(clamp01(.5+.5*ph.x),1.8)*D_AMPLITUDE_METRES[k]*dune;
   h+=v;shown+=v*b;
  }
 }
 // Lake water stays level. Its outline comes from coarse depressions, not a
 // separate lake in each random cell. Noise only roughens the shoreline locally.
 float freshwater=0.,surface=0.;float lakePresence=ramp(-15.,10.,wet.y);
 if(lakePresence>.001 && wet.x>0.01 && m.x>0.){
  float target=wet.x-max(.003,wet.w);h=mix(h,target,lakePresence);shown=mix(shown,target,lakePresence);
  surface=wet.x;freshwater=lakePresence;
 }
 // Only evaluate vector reaches on land and where they can affect the image.
 float riverCover=0.;
 if(land>.05 && (uExact||fp<10.)){
  vec4 rv=river(n,fp);
  if(rv.x<rv.w && rv.w>0.){
   float d=max(0.,rv.x),influence=1.-ramp(0.,rv.w,d);
   float target=rv.y-.003 + pow(d/max(rv.w,.001),1.6)*max(.015,m.x-rv.y);
   h=mix(h,min(h,target),influence);shown=mix(shown,min(shown,target),influence);
   riverCover=1.-ramp(-max(fp*.6,.0005),max(fp*.6,.0005),rv.x);
   if(rv.x<0.){h=min(h,rv.y-.003);shown=min(shown,rv.y-.003);}
   if(riverCover>0.){surface=max(surface,rv.y);freshwater=max(freshwater,riverCover);}
  }
 }
 temp=cl.x-LAPSE*max(0.,h);
 float trees=ramp(1.,10.,temp)*ramp(.4,1.25,aridity)*(1.-ramp(tan(radians(EC_TREE_SLOPE_DEGREES.x)),tan(radians(EC_TREE_SLOPE_DEGREES.y)),slope))*land;
 float canopy=0.;float treeWeight=footprintWeight(A_CANOPY_CELL_METRES,fp);
 if(treeWeight>0.&&trees>.03)canopy=crown(p,n)*A_CANOPY_HEIGHT_METRES*trees*treeWeight;
 float waterCover=max(riverCover,ramp(-max(fp*.006,.0003),max(fp*.006,.0003),surface-shown));
 if(freshwater<.01 && shown>0.)waterCover=riverCover;
 canopy*=1.-waterCover;
 outGround=vec4(shown+canopy,h,canopy/A_CANOPY_HEIGHT_METRES,1.);
 outEnvironment=vec4(temp,cl.y,rug,clamp01(.65*crust.x+.35*clamp01(crust.z)));
 outWater=vec4(waterCover,max(0.,surface-h),freshwater,trees);
}
