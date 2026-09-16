/* World adapter (MPL-2.0 where derived). Scientific/source boundaries in README.
   p is in kilometres RELATIVE to the view's rebased lattice origin.
   The fixed world is evaluated with uFilterPixel=0. The viewer can approximate
   its low-pass image at wider pixel footprints, without moving seeds or cells. */
struct Province {float relief;float uplift;float rounding;float tj;float pj;float sand;
                 vec2 dr;vec2 du;};
struct BaseData {vec3 h;float raw;float gate;Province province;vec2 climate;};
struct LandData {float z;vec2 slope;float ridge;float erosion;float dune;vec2 climate;};
struct WaterData {float ground;float level;float proximity;vec2 slope;};

vec2 climateAt(vec2 p){
 float t=CL_SEA_LEVEL_MEAN_C,pr=log(CL_MEAN_RAINFALL_MM);
 for(int i=0;i<T_N;i++)t+=nd(p,T_IDS[i]).x*T_AMP[i];
 for(int i=0;i<P_N;i++)pr+=nd(p,P_IDS[i]).x*P_AMP[i];
 return vec2(t,clamp(exp(pr),CL_MINIMUM_RAINFALL_MM,CL_MAXIMUM_RAINFALL_MM));
}

Province provinceAt(vec2 p){
 vec2 q=coord(p,PROVINCE_ID),f=fract(q);ivec2 cell=ivec2(floor(q))+uCell[PROVINCE_ID];
 float closest=1e10;
 for(int i=-2;i<=2;i++)for(int j=-2;j<=2;j++){
  vec2 r=ihash(cell+ivec2(i,j));vec2 d=f-vec2(i,j)-(.15+.70*r);
  closest=min(closest,dot(d,d));
 }
 Province a=Province(0.,0.,0.,0.,0.,0.,vec2(0),vec2(0));
 float total=0.;vec2 dwTotal=vec2(0);
 for(int i=-2;i<=2;i++)for(int j=-2;j<=2;j++){
  ivec2 index=cell+ivec2(i,j);vec2 rnd=ihash(index);
  vec2 d=f-vec2(i,j)-(.15+.70*rnd);float d2=dot(d,d);
  // Compact support avoids changing the field when the finite query window moves.
  float support=max(0.,1.-d2/3.61);
  if(support<=0.)continue;
  float w=exp(-PR_BLEND_SHARPNESS*(d2-closest))*support*support*support;
  vec2 dw=w*(-2.*PR_BLEND_SHARPNESS*d-6.*d/max(3.61-d2,1e-8))/PR_CELL_METRES;
  vec2 extra=ihash(index+ivec2(173,921)),extra2=ihash(index+ivec2(-713,47));
  vec2 site=p-d*PR_CELL_METRES;
  float oro=ramp(-.22,.25,nd(site,OROGENY_ID).x);
  float mountainChance=mix(.10,.66,oro);
  int type=0;
  if(extra.x<mountainChance)type=extra.y<.60?3:2;
  else type=extra.y<.52?0:(extra.y<.88?1:4);
  float variation=1.+PR_PARAMETER_JITTER*(extra2.x*2.-1.);
  float relief=PROF_RELIEF_MULTIPLIER[type]*variation;
  float uplift=PROF_UPLIFT[type]*variation;
  a.relief+=w*relief;a.uplift+=w*uplift;a.rounding+=w*PROF_RIDGE_ROUNDING[type];
  a.tj+=w*(extra2.y*2.-1.)*CL_CELL_TEMPERATURE_JITTER_C;
  a.pj+=w*(rnd.y*2.-1.)*CL_CELL_RAIN_LOG_JITTER;
  a.sand+=w*extra2.x;a.dr+=dw*relief;a.du+=dw*uplift;
  total+=w;dwTotal+=dw;
 }
 a.relief/=total;a.uplift/=total;a.rounding/=total;a.tj/=total;a.pj/=total;a.sand/=total;
 a.dr=(a.dr-a.relief*dwTotal)/total;a.du=(a.du-a.uplift*dwTotal)/total;
 return a;
}

vec3 continentAt(vec2 p,bool filtered){
 vec3 wx=nd(p,WARP_X_ID),wy=nd(p,WARP_Y_ID);
 vec2 warped=p+WARP_AMOUNT*vec2(wx.x,wy.x);
 mat2 J=mat2(1.)+WARP_AMOUNT*mat2(wx.y,wy.y,wx.z,wy.z);
 vec3 a=vec3(W_CONTINENTAL_BIAS_METRES,0,0);
 for(int i=0;i<C_N;i++)a+=(filtered?filteredNd(warped,C_IDS[i]):nd(warped,C_IDS[i]))*C_AMPLITUDE_METRES[i];
 a.yz=transpose(J)*a.yz;
 return a;
}

BaseData baseAt(vec2 p,bool filtered){
 vec3 c=continentAt(p,filtered);Province r=provinceAt(p);
 float raw=c.x;
 float h=raw*W_SEA_VERTICAL_SCALE,derivative=W_SEA_VERTICAL_SCALE;
 if(raw>0.){
  // Smooth derivative at the shoreline, without stretching all inland elevations.
  float a=.05*(W_SEA_VERTICAL_SCALE-W_LAND_VERTICAL_SCALE);
  h=raw*W_LAND_VERTICAL_SCALE+a*(1.-exp(-raw/.05));
  derivative=W_LAND_VERTICAL_SCALE+a/.05*exp(-raw/.05);
 }
 vec3 n=vec3(0);
 for(int i=0;i<R_N;i++)n+=(filtered?filteredNd(p,R_IDS[i]):nd(p,R_IDS[i]))*R_AMPLITUDE_METRES[i];
 float gate=ramp(0.,W_COAST_UPLIFT_FADE_METRES,raw);
 vec2 dg=rampDerivative(0.,W_COAST_UPLIFT_FADE_METRES,raw)*c.yz;
 float raised=r.uplift+r.relief*n.x;
 vec2 dr=r.du+r.relief*n.yz+n.x*r.dr;
 vec3 result=vec3(h+gate*raised,derivative*c.yz+gate*dr+raised*dg);
 return BaseData(result,raw,gate,r,climateAt(p)+vec2(r.tj,0));
}

// Vectorization of Runevision's ErosionFilter recurrence. Amplitude/wavelength
// arrays replace geometric strength*=gain, freq*=lacunarity. Crucially the masks
// and the gully steering vector persist from coarse to fine, as in his algorithm.
vec4 erode(vec2 p,vec3 inputH,float fade,Province r,float gate,out float ridge){
 vec3 h=inputH;float slope=max(length(h.yz),1e-10);
 float round0=mix(E_CREASE_ROUNDING[0],r.rounding,clamp01(fade+.5))*E_INITIAL_ROUNDING_MULTIPLIER;
 float mask=ease_out(smooth_start(slope*E_INITIAL_ONSET,round0*E_INITIAL_ONSET));
 float ridgeMask=ease_out(slope*E_RIDGE_MAP_ONSET),ridgeFade=fade;
 vec2 steer=mix(h.yz,h.yz/slope*E_ASSUMED_SLOPE,E_ASSUMED_SLOPE_WEIGHT);
 float magnitude=0.;
 for(int i=0;i<E_N;i++){
  float amp=E_AMPLITUDE_METRES[i]*r.relief*gate;
  float bandwidth=band(E_WAVELENGTH_METRES[i]);
  vec4 ph=bandwidth>0.?phacelle(p,E_IDS[i],safe_normalize(steer),E_CELL_SCALE[i],E_NORMALIZATION[i]):vec4(0.);
  ph.xy*=bandwidth;
  ph.zw*=-1./L_SCALE[E_IDS[i]];
  float sloping=abs(ph.y);
  steer+=sign(ph.y)*ph.zw*amp*E_GULLY_WEIGHT[i];
  vec3 gullies=vec3(ph.x,ph.y*ph.zw);
  vec3 faded=mix(vec3(fade,0,0),gullies*E_GULLY_WEIGHT[i],mask);
  h+=faded*amp;magnitude+=amp;fade=faded.x;
  float rounding=mix(E_CREASE_ROUNDING[i],E_RIDGE_ROUNDING[i]*r.rounding/.1,clamp01(ph.x+.5));
  float next=mix(.82,ease_out(smooth_start(sloping*E_ONSET[i],rounding*E_ONSET[i])),bandwidth);
  mask=pow_inv(mask,E_DETAIL[i])*next;
  ridgeFade=mix(ridgeFade,gullies.x,ridgeMask);
  ridgeMask*=mix(.82,ease_out(sloping*E_RIDGE_MAP_OCTAVE_ONSET),bandwidth);
 }
 ridge=ridgeFade*(1.-ridgeMask);
 return vec4(h-inputH,magnitude);
}

float aridity(float rain,float temperature){return rain/(500.+40.*max(temperature,0.));}
LandData landAt(vec2 p){
 BaseData b=baseAt(p,true);float fluct=b.h.x-b.province.uplift*b.gate;
 float fade=clamp(fluct/max(.45,b.province.relief*.8),-1.,1.);
 float ridge=0.;vec4 e=vec4(0);
 // No camera culling. True ocean needs no mountain erosion, but coastline
 // continuation is continuous because both amplitude and its derivative fade.
 if(b.gate>0.)e=erode(p,b.h,fade,b.province,b.gate,ridge);
 float z=b.h.x+e.x+E_HEIGHT_OFFSET*e.w;
 vec2 slope=b.h.yz+e.yz;
 // The internal slope is the author's APPROXIMATE steering/output derivative;
 // shading normals are independently derived from final height samples.
 float t=b.climate.x-CL_LAPSE_C_PER_KM*max(z,0.);
 float rain=b.climate.y*exp(b.province.pj);
 float a=aridity(rain,t);
 float dune=ramp(D_MINIMUM_TEMPERATURE_C-4.,D_MINIMUM_TEMPERATURE_C+5.,t)
     *ramp(D_MAXIMUM_ARIDITY+.18,D_MAXIMUM_ARIDITY-.12,a)
     *ramp(D_MAXIMUM_HEIGHT_METRES,D_MAXIMUM_HEIGHT_METRES*.7,z)
     *ramp(tan(radians(D_MAXIMUM_SLOPE_DEGREES)),.02,length(b.h.yz))
     *b.gate*ramp(.08,.35,b.province.sand)*D_SAND_COVERAGE;
 if(dune>.0001){
  float wind=radians(CL_WIND_DEGREES)+nd(p,WIND_ID).x*.7;
  vec2 direction=vec2(-sin(wind),cos(wind));
  for(int i=0;i<D_N;i++){
   float bandwidth=band(D_WAVELENGTH_METRES[i]);
   vec4 ph=bandwidth>0.?phacelle(p,D_IDS[i],direction,D_CELL_SCALE[i],.40):vec4(1.,0.,0.,0.);
   float phase=(atan(ph.y,ph.x)+PI)/TAU;
   float crest=D_CREST_FRACTION[i];
   float s=phase<crest?pow(phase/crest,1.25):(1.-phase)/(1.-crest);
   z+=dune*D_AMPLITUDE_METRES[i]*mix(.42,s*length(ph.xy),bandwidth);
  }
 }
 float coastWeight=exp(-abs(b.h.x-W_SEA_LEVEL_METRES)/COAST_BAND);
 if(coastWeight>.0001)for(int i=0;i<B_N;i++){
  vec3 n=filteredNd(p,B_IDS[i])*B_AMPLITUDE_METRES[i];z+=coastWeight*n.x;slope+=coastWeight*n.yz;
 }
 return LandData(z,slope,ridge,e.w>0.?e.x/e.w:0.,dune,vec2(b.climate.x,rain));
}

WaterData lakesAt(vec2 p,float ground,vec2 slope){
 WaterData outp=WaterData(ground,-1e20,0.,slope);
 if(!LK_ENABLED)return outp;
 vec2 q=coord(p,LAKE_ID),f=fract(q);ivec2 cell=ivec2(floor(q))+uCell[LAKE_ID];
 for(int i=-1;i<=1;i++)for(int j=-1;j<=1;j++){
  ivec2 key=cell+ivec2(i,j);vec2 r=ihash(key),v=ihash(key+ivec2(894,-713));
  if(v.x>LK_PROBABILITY)continue;
  vec2 d=(f-vec2(i,j)-(.25+.5*r))*LK_CELL_METRES;
  float radius=mix(LK_RADIUS_METRES.x,LK_RADIUS_METRES.y,v.y*v.y);
  if(length(d)>1.9*radius)continue;
  vec2 centre=p-d;BaseData b=baseAt(centre,false);
  float temp=b.climate.x-CL_LAPSE_C_PER_KM*max(b.h.x,0.);
  if(b.raw<.18||b.h.x<.02||b.h.x>LK_MAXIMUM_ELEVATION_METRES||
     length(b.h.yz)>tan(radians(LK_MAXIMUM_SLOPE_DEGREES))||
     aridity(b.climate.y,temp)<LK_MINIMUM_ARIDITY)continue;
  vec2 extra=ihash(key+ivec2(-501,843));float angle=extra.x*TAU;
  mat2 M=mat2(cos(angle),sin(angle),-sin(angle),cos(angle));
  float aspect=mix(1.25,2.7,extra.y);vec2 uv=M*d/radius;uv.y*=aspect;
  // Irregular analytic bowl, not a painted water polygon. Compact support
  // means another visit/tile computes exactly the same bed and shoreline.
  vec3 w1=nd(p,LAKE_WARP1_ID),w2=nd(p,LAKE_WARP2_ID);
  float rr=length(uv)+.30*w1.x+.13*w2.x;
  vec2 dr=transpose(M)*vec2(uv.x,uv.y*aspect)/max(length(uv)*radius,1e-6)+.30*w1.yz+.13*w2.yz;
  if(rr>=1.60)continue;
  float depth=mix(LK_DEPTH_METRES.x,LK_DEPTH_METRES.y,extra.x);
  float level=max(.008,b.h.x-.30*depth);
  float bowl=level+depth*(rr*rr-.62);
  float weight=1.-ramp(1.05,1.6,rr);
  outp.slope=mix(outp.slope,2.*depth*rr*dr,weight)-(bowl-outp.ground)*rampDerivative(1.05,1.60,rr)*dr;
  outp.ground=mix(outp.ground,bowl,weight);
  outp.proximity=max(outp.proximity,1.-ramp(.75,1.60,rr));
  if(rr<1.05&&outp.ground<level)outp.level=max(outp.level,level);
 }
 return outp;
}

// Temperature and moisture are corrected using final BARE-EARTH altitude,
// slope/aspect and nearby water. No tropical trees on an alpine or icy summit.
vec4 ecology(LandData land,WaterData water){
 float z=water.ground,t=land.climate.x-CL_LAPSE_C_PER_KM*max(0.,z-W_SEA_LEVEL_METRES);
 float slope=length(water.slope),north=-water.slope.y/max(slope,.01);
 vec2 wind=vec2(cos(radians(CL_WIND_DEGREES)),sin(radians(CL_WIND_DEGREES)));
 float orographic=clamp(dot(water.slope,wind)*3.,-1.,1.);
 float rain=land.climate.y*exp(CL_RAIN_SHADOW_STRENGTH*orographic);
 float a=aridity(rain,t);
 a*=1.+CL_ASPECT_MOISTURE_STRENGTH*north;
 a/=1.+CL_SLOPE_DRYING_STRENGTH*slope;
 a+=CL_WATER_MOISTURE_BOOST*max(water.proximity,exp(-abs(z-W_SEA_LEVEL_METRES)/.025));
 float forest=ramp(EC_TREE_TEMPERATURE_MIN_C,EC_TREE_TEMPERATURE_FULL_C,t)
   *ramp(EC_TREE_ARIDITY_MIN,EC_TREE_ARIDITY_FULL,a)
   *ramp(tan(radians(EC_TREE_SLOPE_END_DEGREES)),tan(radians(EC_TREE_SLOPE_START_DEGREES)),slope);
 if(z<W_SEA_LEVEL_METRES||water.level>z)forest=0.;
 return vec4(t,a,forest,slope);
}
