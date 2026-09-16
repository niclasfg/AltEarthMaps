'use strict';
// Plain WebGL2 viewer. Three terrain outputs, one colour pass, one screen blit.
// No libraries, satellite-image service, asset download or image-tile queue.
(async function(){
const canvas=document.getElementById('map'),status=document.getElementById('status');
const fail=e=>{const b=document.getElementById('error');b.style.display='block';b.textContent=String(e);status.textContent='Error';};
try{
 const response=await fetch('/bundle.json'),bundle=await response.json();
 if(!response.ok||bundle.error)throw Error(bundle.error||'Cannot read configuration.');
 const cfg=bundle.config,v=cfg.viewer;
 const gl=canvas.getContext('webgl2',{alpha:false,antialias:false,preserveDrawingBuffer:true});
 if(!gl||!gl.getExtension('EXT_color_buffer_float'))throw Error('WebGL2 with floating-point render targets is required. Enable browser hardware acceleration and update your graphics driver.');
 canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();fail('Graphics context lost. Lower max_render_width or supersampling, then refresh.');});
 function program(fragment){
  function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
  const p=gl.createProgram(),vs=shader(gl.VERTEX_SHADER,bundle.vertex),fs=shader(gl.FRAGMENT_SHADER,fragment);
  gl.attachShader(p,vs);gl.attachShader(p,fs);gl.linkProgram(p);gl.deleteShader(vs);gl.deleteShader(fs);
  if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));
  const u={};for(let i=0;i<gl.getProgramParameter(p,gl.ACTIVE_UNIFORMS);i++){const a=gl.getActiveUniform(p,i);u[a.name]=gl.getUniformLocation(p,a.name);}
  return {p,u};
 }
 const height=program(bundle.height),display=program(bundle.display),blit=program(`#version 300 es
 precision highp float;uniform sampler2D uColor;uniform vec2 uSize;out vec4 outColor;
 void main(){outColor=texture(uColor,gl_FragCoord.xy/uSize);}`);
 gl.bindVertexArray(gl.createVertexArray());
 function target(float=true,count=3){return {textures:Array.from({length:count},()=>gl.createTexture()),fbo:gl.createFramebuffer(),w:0,h:0,float};}
 const main=target(),context=target(),colour=target(false,1);
 function resize(t,w,h){
  if(t.w===w&&t.h===h)return;
  if(Math.max(w,h)>gl.getParameter(gl.MAX_TEXTURE_SIZE))throw Error('Render target exceeds GPU limit. Lower max_render_width.');
  t.w=w;t.h=h;gl.bindFramebuffer(gl.FRAMEBUFFER,t.fbo);
  t.textures.forEach((tex,i)=>{gl.bindTexture(gl.TEXTURE_2D,tex);
   gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,t.float?gl.NEAREST:gl.LINEAR);
   gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,t.float?gl.NEAREST:gl.LINEAR);
   gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
   gl.texImage2D(gl.TEXTURE_2D,0,t.float?gl.RGBA32F:gl.RGBA8,w,h,0,gl.RGBA,t.float?gl.FLOAT:gl.UNSIGNED_BYTE,null);
   gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0+i,gl.TEXTURE_2D,tex,0);
  });
  gl.drawBuffers(t.textures.map((_,i)=>gl.COLOR_ATTACHMENT0+i));
  if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('Could not allocate terrain textures. Lower render resolution.');
 }
 function bindTexture(p,name,t,unit,index=0){gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(gl.TEXTURE_2D,t.textures[index]);gl.uniform1i(p.u[name],unit);}
 function bankAt(p,cx,cy){
  // Rebase every procedural lattice separately using JavaScript doubles. This
  // preserves metre detail far from the origin; integer hash keys can wrap at
  // 2^32 cells, but do NOT collapse at float's 24-bit mantissa limit.
  const cells=new Int32Array(bundle.bank.length*2),frac=new Float32Array(cells.length);
  bundle.bank.forEach((a,i)=>{const c=Math.cos(a.angle),s=Math.sin(a.angle);
   const q=[(c*cx-s*cy)/a.wavelength+a.offset[0],(s*cx+c*cy)/a.wavelength+a.offset[1]];
   for(let j=0;j<2;j++){const k=Math.floor(q[j]);cells[2*i+j]=k|0;frac[2*i+j]=q[j]-k;}
  });
  gl.uniform2iv(p.u['uCell[0]'],cells);gl.uniform2fv(p.u['uFraction[0]'],frac);
 }
 function generate(t,origin,pixel,cx,cy,filterPixel=pixel){
  gl.bindFramebuffer(gl.FRAMEBUFFER,t.fbo);gl.viewport(0,0,t.w,t.h);gl.useProgram(height.p);
  bankAt(height,cx,cy);gl.uniform1f(height.u.uFilterPixel,filterPixel);gl.uniform2fv(height.u.uOrigin,origin);gl.uniform1f(height.u.uPixel,pixel);gl.drawArrays(gl.TRIANGLES,0,3);
 }
 let x=v.start_x_metres/1000,y=v.start_y_metres/1000,span=v.view_width_metres/1000;
 let frame=0,pending=false,moving=false,settle,fence=null,last={},contextCache='';
 function scaleBar(){const mpp=span*1000/innerWidth;let goal=mpp*120,pow=10**Math.floor(Math.log10(goal));
  const metres=[1,2,5,10].map(n=>n*pow).filter(n=>n<=goal).pop()||pow;
  document.getElementById('scaleText').textContent=metres>=1000?`${metres/1000} km`:`${Number(metres.toPrecision(3))} m`;
  document.getElementById('bar').style.width=`${metres/mpp}px`;
  status.textContent=`${mpp>=10?Math.round(mpp):mpp.toFixed(2)} m / pixel · ${moving?'moving':'WebGL'}`;
 }
 function render(){pending=false;
  try{
   if(fence){const state=gl.clientWaitSync(fence,0,0);if(state===gl.WAIT_FAILED)throw Error('GPU synchronization failed.');
    if(state===gl.TIMEOUT_EXPIRED){pending=true;requestAnimationFrame(render);return;}gl.deleteSync(fence);fence=null;}
   const w=Math.max(128,Math.round(Math.min(innerWidth*devicePixelRatio,v.max_render_width)*(moving?v.interactive_scale:1))),h=Math.max(80,Math.round(w*innerHeight/innerWidth));
   const ss=moving?1:v.supersampling,sw=w*ss,sh=h*ss,border=2,pixel=span/sw;
   if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
   const origin=[-span/2-border*pixel,-span*h/w/2-border*pixel];
   resize(main,sw+2*border,sh+2*border);generate(main,origin,pixel,x,y);
   const cs=cfg.appearance.shadows?192:1,csSpan=Math.max(90,span*1.6,span*h/w*1.6),cp=csSpan/cs;
   const absolute=[Math.floor((x-csSpan/2)/cp)*cp,Math.floor((y-csSpan/2)/cp)*cp];
   const co=[absolute[0]-x,absolute[1]-y],key=absolute.join(',')+','+cp;
   if(key!==contextCache){resize(context,cs,cs);generate(context,co,cp,x,y);contextCache=key;}
   resize(colour,sw,sh);gl.bindFramebuffer(gl.FRAMEBUFFER,colour.fbo);gl.viewport(0,0,sw,sh);gl.useProgram(display.p);bankAt(display,x,y);
   bindTexture(display,'uHeight',main,0);bindTexture(display,'uClimate',main,1,1);bindTexture(display,'uCover',main,2,2);bindTexture(display,'uShadow',context,3);
   gl.uniform2f(display.u.uSize,main.w,main.h);gl.uniform2fv(display.u.uOrigin,origin);gl.uniform1f(display.u.uPixel,pixel);
   gl.uniform2fv(display.u.uShadowOrigin,co);gl.uniform1f(display.u.uShadowPixel,cp);gl.uniform2f(display.u.uShadowSize,cs,cs);gl.uniform1f(display.u.uBorder,border);gl.drawArrays(gl.TRIANGLES,0,3);
   gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.viewport(0,0,w,h);gl.useProgram(blit.p);bindTexture(blit,'uColor',colour,0);gl.uniform2f(blit.u.uSize,w,h);gl.drawArrays(gl.TRIANGLES,0,3);
   fence=gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE,0);gl.flush();
   last={x,y,span,pixel,origin,w:sw,h:sh,outputW:w,outputH:h,frame:++frame};scaleBar();window.terrain.ready=true;
  }catch(e){fail(e);}
 }
 function request(interactive=true){moving=interactive;clearTimeout(settle);if(!pending){pending=true;requestAnimationFrame(render);}
  if(interactive)settle=setTimeout(()=>{moving=false;if(!pending){pending=true;requestAnimationFrame(render);}},v.settle_ms);}
 function zoom(factor,px=innerWidth/2,py=innerHeight/2){const old=span;span=Math.min(v.max_view_width_metres/1000,Math.max(v.min_view_width_metres/1000,span*factor));
  x+=(px-innerWidth/2)/innerWidth*(old-span);y-=(py-innerHeight/2)/innerWidth*(old-span);request();}
 let drag=null;
 canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,id:e.pointerId};canvas.setPointerCapture(e.pointerId);});
 canvas.addEventListener('pointermove',e=>{if(!drag||drag.id!==e.pointerId)return;x-=(e.clientX-drag.x)/innerWidth*span;y+=(e.clientY-drag.y)/innerWidth*span;drag.x=e.clientX;drag.y=e.clientY;request();});
 canvas.addEventListener('pointerup',()=>{drag=null;request(false);});canvas.addEventListener('pointercancel',()=>{drag=null;request(false);});
 canvas.addEventListener('wheel',e=>{e.preventDefault();const d=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?innerHeight:1);zoom(Math.exp(Math.max(-1,Math.min(1,d*.0015))),e.clientX,e.clientY);},{passive:false});
 document.getElementById('plus').onclick=()=>zoom(.5);document.getElementById('minus').onclick=()=>zoom(2);addEventListener('resize',()=>request());
 // Inspection only, no additional user interface or test framework. Coordinates
 // for setView are kilometres; all editable config remains in METRES.
 function read(t,index=0){gl.bindFramebuffer(gl.FRAMEBUFFER,t.fbo);gl.readBuffer(gl.COLOR_ATTACHMENT0+index);const a=new Float32Array(t.w*t.h*4);gl.readPixels(0,0,t.w,t.h,gl.RGBA,gl.FLOAT,a);gl.readBuffer(gl.COLOR_ATTACHMENT0);gl.bindFramebuffer(gl.FRAMEBUFFER,null);return a;}
 window.terrain={ready:false,getView:()=>({...last}),setView:(cx,cy,width)=>{x=cx;y=cy;span=width;request(false);},
  readHeights:()=>({w:main.w,h:main.h,values:read(main)}),readClimate:()=>read(main,1),readCover:()=>read(main,2),
  gl,bundle,program,target,resize,generate,read,bankAt};request(false);
}catch(e){fail(e);}
})();
