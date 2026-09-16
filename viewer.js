'use strict';
// Library-free orthographic globe. All fine terrain is a function of a fixed
// point on the sphere, not camera latitude or a wrapping rectangular tile.
(async()=>{
const canvas=document.getElementById('map'),status=document.getElementById('status');
const fail=e=>{document.getElementById('error').style.display='block';document.getElementById('error').textContent=String(e);document.getElementById('loading').style.display='none';status.textContent='Error';console.error(e);};
try{
 const b=await (await fetch('/bundle.json')).json(),c=b.config,v=c.viewer,R=c.planet.radius_metres/1000;
 const gl=canvas.getContext('webgl2',{alpha:false,antialias:false,preserveDrawingBuffer:true});
 if(!gl||!gl.getExtension('EXT_color_buffer_float'))throw Error('WebGL2 and floating-point render targets are required. Enable browser hardware acceleration.');
 canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();fail('Graphics context lost. Reduce max_render_width and restart/refresh.');});
 const norm=a=>{let s=Math.hypot(...a);return a.map(x=>x/s)},dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],add=(a,b,s=1)=>a.map((x,i)=>x+b[i]*s);
 let centre,east,north,span=v.view_width_metres/1000;
 function orient(lon,lat){lon*=Math.PI/180;lat*=Math.PI/180;centre=[Math.cos(lat)*Math.cos(lon),Math.sin(lat),Math.cos(lat)*Math.sin(lon)];east=[-Math.sin(lon),0,Math.cos(lon)];north=norm(cross(east,centre));}
 orient(v.start_longitude_degrees,v.start_latitude_degrees);
 function program(src){
  function shader(type,s){let a=gl.createShader(type);gl.shaderSource(a,s);gl.compileShader(a);if(!gl.getShaderParameter(a,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(a));return a;}
  const p=gl.createProgram();let vs=shader(gl.VERTEX_SHADER,b.vertex),fs=shader(gl.FRAGMENT_SHADER,src);gl.attachShader(p,vs);gl.attachShader(p,fs);gl.linkProgram(p);gl.deleteShader(vs);gl.deleteShader(fs);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));
  let u={};for(let i=0;i<gl.getProgramParameter(p,gl.ACTIVE_UNIFORMS);i++){const a=gl.getActiveUniform(p,i);u[a.name]=gl.getUniformLocation(p,a.name);}return{p,u};
 }
 const height=program(b.height),display=program(b.display),blit=program(`#version 300 es
 precision highp float;uniform sampler2D uColor;uniform vec2 uOutputSize;out vec4 outColor;void main(){outColor=texture(uColor,gl_FragCoord.xy/uOutputSize);}`);
 gl.bindVertexArray(gl.createVertexArray());
 const data={};
 for(const name of ['macro','climate','water','rivers','river_header','river_index']){
  const bytes=await(await fetch('/data/'+name+'.bin')).arrayBuffer(),dims=b.meta[name],uint=name.startsWith('river_'),channels=name==='river_header'?2:name==='river_index'?1:4;
  const tex=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,tex);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  // River reaches are packed in 1024-wide rows, not a texture taller than GPU limits.
  let w=dims[0],h=dims[1],array=uint?new Uint32Array(bytes):new Float32Array(bytes);
  if(name==='rivers'){const tw=1023,th=Math.ceil(array.length/(tw*4)),a=new Float32Array(tw*th*4);a.set(array);array=a;w=tw;h=th;}
  if(w>gl.getParameter(gl.MAX_TEXTURE_SIZE)||h>gl.getParameter(gl.MAX_TEXTURE_SIZE))throw Error('Global texture exceeds graphics limit. Reduce planet.map_width.');
  gl.texImage2D(gl.TEXTURE_2D,0,uint?(channels===2?gl.RG32UI:gl.R32UI):gl.RGBA32F,w,h,0,uint?(channels===2?gl.RG_INTEGER:gl.RED_INTEGER):gl.RGBA,uint?gl.UNSIGNED_INT:gl.FLOAT,array);
  data[name]=tex;
 }
 function bind(p,name,tex,unit){gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(gl.TEXTURE_2D,tex);gl.uniform1i(p.u[name],unit);}
 function maps(p){for(const [i,kv] of Object.entries([['uMacro','macro'],['uClimate','climate'],['uWater','water'],['uRivers','rivers'],['uRiverHeader','river_header'],['uRiverIndex','river_index']]))bind(p,kv[0],data[kv[1]],+i);}
 function uniforms(p,w,h,pixel,visibleSpan){gl.uniform3fv(p.u.uCentre,centre);gl.uniform3fv(p.u.uEast,east);gl.uniform3fv(p.u.uNorth,north);gl.uniform2f(p.u.uSize,w,h);gl.uniform1f(p.u.uPixel,pixel);gl.uniform1f(p.u.uSpan,visibleSpan);
  const cell=new Int32Array(b.bank.length*3),fract=new Float32Array(cell.length);
  b.bank.forEach((a,i)=>{for(let j=0;j<3;j++){let q=centre[j]*R/a.scale+a.offset[j],k=Math.floor(q);cell[3*i+j]=k|0;fract[3*i+j]=q-k;}});
  gl.uniform3iv(p.u['uCell[0]'],cell);gl.uniform3fv(p.u['uFraction[0]'],fract);
 }
 function target(count,float){return{textures:Array.from({length:count},()=>gl.createTexture()),fbo:gl.createFramebuffer(),float,w:0,h:0};}
 const field=target(3,true),colour=target(1,false);
 function resize(t,w,h){if(t.w===w&&t.h===h)return;t.w=w;t.h=h;gl.bindFramebuffer(gl.FRAMEBUFFER,t.fbo);
  t.textures.forEach((a,i)=>{gl.bindTexture(gl.TEXTURE_2D,a);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,t.float?gl.NEAREST:gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,t.float?gl.NEAREST:gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.texImage2D(gl.TEXTURE_2D,0,t.float?gl.RGBA32F:gl.RGBA8,w,h,0,gl.RGBA,t.float?gl.FLOAT:gl.UNSIGNED_BYTE,null);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0+i,gl.TEXTURE_2D,a,0);});gl.drawBuffers(t.textures.map((_,i)=>gl.COLOR_ATTACHMENT0+i));if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('Render-target allocation failed. Lower max_render_width.');
 }
 let moving=false,pending=false,settle,fence=null,frames=0,last={};
 function labels(){let mpp=span*1000/innerWidth,goal=mpp*95,p=10**Math.floor(Math.log10(goal)),metres=[1,2,5,10].map(x=>x*p).filter(x=>x<=goal).pop()||p;
  document.getElementById('scaleText').textContent=(metres>=1000?`${metres/1000} km`:`${metres} m`)+' · at centre';document.getElementById('bar').style.width=metres/mpp+'px';
  const lat=Math.asin(Math.max(-1,Math.min(1,centre[1])))*180/Math.PI,lon=Math.atan2(centre[2],centre[0])*180/Math.PI;
  status.textContent=`${lat.toFixed(2)}°, ${lon.toFixed(2)}° · ${mpp>=10?Math.round(mpp):mpp.toFixed(2)} m/pixel`;
 }
 function render(){pending=false;try{
  if(fence){let state=gl.clientWaitSync(fence,0,0);if(state===gl.TIMEOUT_EXPIRED){pending=true;requestAnimationFrame(render);return;}if(state===gl.WAIT_FAILED)throw Error('GPU synchronization failed.');gl.deleteSync(fence);fence=null;}
  const w=Math.max(128,Math.round(Math.min(innerWidth*devicePixelRatio,v.max_render_width)*(moving?v.interactive_scale:1))),h=Math.max(80,Math.round(w*innerHeight/innerWidth)),ss=moving?1:v.supersampling,sw=w*ss,sh=h*ss,pixel=span/sw,border=2;
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
  resize(field,sw+4,sh+4);gl.bindFramebuffer(gl.FRAMEBUFFER,field.fbo);gl.viewport(0,0,field.w,field.h);gl.useProgram(height.p);maps(height);uniforms(height,field.w,field.h,pixel,span+4*pixel);gl.uniform1i(height.u.uExact,0);gl.drawArrays(gl.TRIANGLES,0,3);
  resize(colour,sw,sh);gl.bindFramebuffer(gl.FRAMEBUFFER,colour.fbo);gl.viewport(0,0,sw,sh);gl.useProgram(display.p);maps(display);uniforms(display,sw,sh,pixel,span);
  bind(display,'uGround',field.textures[0],6);bind(display,'uEnvironment',field.textures[1],7);bind(display,'uWet',field.textures[2],8);gl.uniform1f(display.u.uBorder,border);gl.drawArrays(gl.TRIANGLES,0,3);
  gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.viewport(0,0,w,h);gl.useProgram(blit.p);bind(blit,'uColor',colour.textures[0],0);gl.uniform2f(blit.u.uOutputSize,w,h);gl.drawArrays(gl.TRIANGLES,0,3);
  fence=gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE,0);gl.flush();frames++;last={frames,span,centre:[...centre],w,h,pixel};window.globe.ready=true;labels();document.getElementById('loading').style.display='none';
 }catch(e){fail(e);}}
 function request(interactive=true){moving=interactive;clearTimeout(settle);if(!pending){pending=true;requestAnimationFrame(render);}if(interactive)settle=setTimeout(()=>request(false),v.settle_ms);}
 function point(x,y){const dx=(x/innerWidth-.5)*span,dy=(.5-y/innerHeight)*span*innerHeight/innerWidth,r2=dx*dx+dy*dy;if(r2>=R*R)return null;return norm(add(add(centre.map(z=>z*Math.sqrt(R*R-r2)),east,dx),north,dy));}
 function rotate(from,to){const axis=cross(from,to),s=Math.hypot(...axis),co=dot(from,to);if(s<1e-12)return;const k=axis.map(x=>x/s);function rot(v){return norm(add(add(v.map(x=>x*co),cross(k,v),s),k,dot(k,v)*(1-co)));}centre=rot(centre);east=rot(east);north=norm(cross(east,centre));}
 function zoom(factor,x=innerWidth/2,y=innerHeight/2){const anchor=point(x,y);span=Math.max(v.min_view_width_metres/1000,Math.min(R*5,span*factor));const after=point(x,y);if(anchor&&after)rotate(after,anchor);request();}
 let anchor=null;
 canvas.addEventListener('pointerdown',e=>{anchor=point(e.clientX,e.clientY);canvas.setPointerCapture(e.pointerId);});
 canvas.addEventListener('pointermove',e=>{if(!anchor)return;const under=point(e.clientX,e.clientY);if(under){rotate(under,anchor);request();}});
 for(const event of ['pointerup','pointercancel'])canvas.addEventListener(event,()=>{anchor=null;request(false);});
 canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(Math.max(-1,Math.min(1,e.deltaY*(e.deltaMode===1?16:1)*.0015))),e.clientX,e.clientY);},{passive:false});
 document.getElementById('plus').onclick=()=>zoom(.5);document.getElementById('minus').onclick=()=>zoom(2);document.getElementById('home').onclick=()=>{span=R*3.6;request(false);};addEventListener('resize',()=>request());
 // Small inspection hooks, no shipped test framework or extra interface panels.
 window.globe={ready:false,getView:()=>({...last}),setView:(lon,lat,widthKm)=>{orient(lon,lat);span=widthKm;request(false);},
  read:()=>{gl.finish();gl.bindFramebuffer(gl.FRAMEBUFFER,field.fbo);gl.readBuffer(gl.COLOR_ATTACHMENT0);let a=new Float32Array(field.w*field.h*4);gl.readPixels(0,0,field.w,field.h,gl.RGBA,gl.FLOAT,a);gl.bindFramebuffer(gl.FRAMEBUFFER,null);return{w:field.w,h:field.h,values:a};},gl,bundle:b};
 document.querySelector('#brand small').textContent=`${(R).toLocaleString()} km radius · drag to rotate · scroll to zoom`;request(false);
}catch(e){fail(e);}
})();
