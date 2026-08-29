(function(){
"use strict";
const D = JSON.parse(document.getElementById('benchdata').textContent);
const N = window.NARRATIVE || {};
const $ = (s,r)=>(r||document).querySelector(s);
/* The full benchmark report and the viewer report share this file. The viewer
   omits the authored-narrative sections entirely, so every block below that
   writes into one first checks that its host element exists. A missing section
   is a variant, not an error. */
const have = sel => !!document.querySelector(sel);
const put = (sel,html)=>{const n=document.querySelector(sel); if(n&&html!=null&&html!=='')n.innerHTML=html;};
const el=(t,a)=>{const n=document.createElement(t);if(a)for(const k in a){
  if(k==='html')n.innerHTML=a[k];else if(k==='text')n.textContent=a[k];else n.setAttribute(k,a[k]);}return n;};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt=(v,d)=>(v===null||v===undefined||Number.isNaN(v))?'—':(+v).toFixed(d===undefined?3:d);
const pct=v=>(v===null||v===undefined)?'—':(v*100).toFixed(1)+'%';

const OK=D.systems.filter(s=>s.status==='ok');
const byId=Object.fromEntries(D.systems.map(s=>[s.id,s]));
const CLS=D.classes, COL=D.colors;
const ENS=D.ensemble||{per_class:{},routing:{},per_page_class_counts:{}};

/* ---------- theme ---------- */
if(have('#themeBtn'))$('#themeBtn').addEventListener('click',()=>{
  const cur=document.documentElement.getAttribute('data-theme');
  const now=cur||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
  document.documentElement.setAttribute('data-theme',now==='dark'?'light':'dark');
  drawScatter();
});

/* ---------- hero ---------- */
if(have('#heroMeta')){
  const corpus=D.corpus||{},docs=corpus.documents||[];
  const hero = N.heroMeta || [
    ['Repositories', new Set(D.systems.map(s=>s.repo)).size],
    ['Configurations run', OK.length],
    ['Pages analysed', D.pages.length],
    ['Source documents', docs.length||1],
    ['Region classes', (D.classes||[]).length]];
  $('#heroMeta').innerHTML=hero.map(([k,v])=>`<div><b>${v}</b>${k}</div>`).join('');
}
if(have('#docTitle')&&(D.corpus||{}).documents&&D.corpus.documents.length===1){
  const d=D.corpus.documents[0];
  $('#docTitle').textContent=d.file||'Layout comparison';
}

/* ---------- static narrative blocks (full report only) ---------- */
put('#findingCards',(N.findings||[]).map((f,i)=>
  `<div class="finding"><span class="n">Finding ${i+1}</span><h3>${f.title}</h3>${f.body}</div>`).join(''));
put('#qaGrid',(N.answers||[]).map(([q,a])=>
  `<div class="qa-item"><div class="qa-q">${q}</div><div class="qa-a">${a}</div></div>`).join(''));
put('#methodBody',N.method); put('#ranksBody',N.ranks); put('#limitsBody',N.limits);
put('#footerBody',N.footer); put('#routingBody',N.routing);
put('#metricNote',N.metricNote); put('#countNote',N.countNote);

/* ================= ORBIT VIEWER ================= */
const V={page:D.pages[0].id,hidden:new Set(),cls:new Set(CLS),labels:false,conf:false,
         order:false,fill:true,ref:false,links:true,minConf:0};

/* Fixed slot assignment: a model's angular position never changes.  The order is
   persisted in benchmark/reports/orbit_slot_order.json and only ever appended to,
   so a model added later lands in the next free outer slot and every model already
   placed keeps the exact position a reader learned last time. */
// Ring capacity must exceed the number of scored systems: a system with no
// slot is silently dropped from the orbit view, which is the report's main
// artefact.
const RING_CAP=[6,10,14,18,22];
const SLOTS=(()=>{
  const ok=new Set(D.systems.filter(s=>s.status==='ok').map(s=>s.id));
  const persisted=(D.slot_order||[]).filter(id=>ok.has(id));
  const order=persisted.concat(D.systems.filter(s=>s.status==='ok'&&!persisted.includes(s.id)).map(s=>s.id));
  const out={};let i=0;
  for(let ring=0;ring<RING_CAP.length&&i<order.length;ring++){
    const cap=RING_CAP[ring];
    for(let k=0;k<cap&&i<order.length;k++,i++){
      out[order[i]]={ring,slot:k,cap,
        angle:(-Math.PI/2)+(2*Math.PI*k/cap)+(ring%2?Math.PI/cap:0)};
    }
  }
  return out;
})();
const RING_R=[580,940,1300,1660,2020];
const SAT_W=[240,250,250,250,250];
const HUB_W=380;
const WORLD=5200;

function shortName(s){
  const p=s.display.split('·');
  return {a:(p[0]||'').trim(),b:(p.slice(1).join('·')||'').trim()};
}

/* page strip */
(function(){
  const strip=$('#pageStrip');
  D.pages.forEach(p=>{
    const b=el('button',{class:'pagethumb','aria-pressed':String(p.id===V.page),
      'data-page':p.id,title:`${p.doc} · p${p.page} · ${p.stratum}`});
    b.innerHTML=`<img src="${p.img}" alt=""><b>${p.id.replace('page_','')}</b>`;
    b.addEventListener('click',()=>{V.page=p.id;
      strip.querySelectorAll('.pagethumb').forEach(x=>
        x.setAttribute('aria-pressed',String(x.dataset.page===V.page)));
      renderOrbit(); renderCounts();});
    strip.appendChild(b);
  });
})();

/* model chips */
(function(){
  const c=$('#sysChips');
  const all=el('button',{class:'chip','aria-pressed':'true',text:'all models'});
  all.addEventListener('click',()=>{
    const on=all.getAttribute('aria-pressed')==='true';
    V.hidden = on ? new Set(OK.map(s=>s.id)) : new Set();
    all.setAttribute('aria-pressed',String(!on));
    c.querySelectorAll('[data-sys]').forEach(x=>
      x.setAttribute('aria-pressed',String(!V.hidden.has(x.dataset.sys))));
    renderOrbit();});
  c.appendChild(all);
  OK.forEach(s=>{
    const nm=shortName(s);
    const b=el('button',{class:'chip','aria-pressed':'true','data-sys':s.id,
      title:s.display, text:nm.b||nm.a});
    b.addEventListener('click',()=>{
      if(V.hidden.has(s.id))V.hidden.delete(s.id);else V.hidden.add(s.id);
      b.setAttribute('aria-pressed',String(!V.hidden.has(s.id))); renderOrbit();});
    c.appendChild(b);
  });
})();

/* class chips + legend */
(function(){
  const c=$('#clsChips');
  const all=el('button',{class:'chip','aria-pressed':'true',text:'all classes'});
  all.addEventListener('click',()=>{
    const on=all.getAttribute('aria-pressed')==='true';
    V.cls=on?new Set():new Set(CLS);
    all.setAttribute('aria-pressed',String(!on));
    c.querySelectorAll('[data-cls]').forEach(x=>x.setAttribute('aria-pressed',String(!on)));
    renderOrbit();});
  c.appendChild(all);
  CLS.forEach(k=>{
    const b=el('button',{class:'chip','aria-pressed':'true','data-cls':k});
    b.innerHTML=`<i class="sw" style="background:${COL[k]}"></i>${k}`;
    b.addEventListener('click',()=>{
      if(V.cls.has(k))V.cls.delete(k);else V.cls.add(k);
      b.setAttribute('aria-pressed',String(V.cls.has(k))); renderOrbit();});
    c.appendChild(b);
  });
  $('#legend').innerHTML=CLS.map(k=>
    `<span><i style="background:${COL[k]}"></i>${k}</span>`).join('');
})();

[['tLabels','labels'],['tConf','conf'],['tOrder','order'],['tFill','fill'],
 ['tRef','ref'],['tLinks','links']].forEach(([id,key])=>{
  const b=$('#'+id);
  b.addEventListener('click',()=>{V[key]=!V[key];
    b.setAttribute('aria-pressed',String(V[key])); renderOrbit();});
});
$('#confSlider').addEventListener('input',e=>{
  V.minConf=+e.target.value;$('#confVal').textContent=V.minConf.toFixed(2);
  renderOrbit();renderCounts();});

function visibleRegions(pid,sid){
  const pr=(D.predictions[pid]||{})[sid];
  if(!pr)return [];
  return pr.r.filter(r=>V.cls.has(CLS[r[0]])&&!(r[1]>=0&&r[1]<V.minConf));
}

function overlaySVG(page,sid,scaleHint){
  const W=page.w,H=page.h;
  let s=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">`;
  if(V.ref){
    (page.ref.graphic_areas||[]).forEach(b=>{s+=`<rect x="${b[0]}" y="${b[1]}" width="${b[2]-b[0]}" height="${b[3]-b[1]}" fill="none" stroke="#7E858D" stroke-width="7" stroke-dasharray="24 16"/>`;});
    (page.ref.grid_candidates||[]).forEach(b=>{s+=`<rect x="${b[0]}" y="${b[1]}" width="${b[2]-b[0]}" height="${b[3]-b[1]}" fill="none" stroke="#15171A" stroke-width="7" stroke-dasharray="10 12"/>`;});
    (page.ref.column_bands||[]).forEach(b=>{s+=`<rect x="${b[0]}" y="0" width="${b[1]-b[0]}" height="${H}" fill="none" stroke="#9C6A22" stroke-width="5" stroke-dasharray="5 18"/>`;});
  }
  if(sid){
    const sw=Math.max(4,W/(scaleHint||300));
    const fs=Math.round(W/34);
    visibleRegions(page.id,sid).forEach(r=>{
      const [ci,cf,x1,y1,x2,y2,ord]=r,k=CLS[ci],c=COL[k];
      if(V.fill)s+=`<rect x="${x1}" y="${y1}" width="${x2-x1}" height="${y2-y1}" fill="${c}" opacity="0.15"/>`;
      s+=`<rect x="${x1}" y="${y1}" width="${x2-x1}" height="${y2-y1}" fill="none" stroke="${c}" stroke-width="${sw}"/>`;
      if(V.labels){
        let t=k;
        if(V.order&&ord>=0)t=ord+'·'+t;
        if(V.conf&&cf>=0)t+=' '+cf.toFixed(2);
        const w=t.length*fs*0.58+12,ly=Math.max(0,y1-fs*1.3);
        s+=`<rect x="${x1}" y="${ly}" width="${w}" height="${fs*1.25}" fill="${c}"/>`;
        s+=`<text x="${x1+6}" y="${ly+fs*0.98}" fill="#fff" font-family="IBM Plex Mono, monospace" font-size="${fs}">${esc(t)}</text>`;
      }
    });
  }
  return s+'</svg>';
}

const world=$('#world'), stage=$('#stage');
let view={x:0,y:0,k:1};

function applyView(){world.style.transform=`translate(${view.x}px,${view.y}px) scale(${view.k})`;}

function renderOrbit(){
  const page=D.pages.find(p=>p.id===V.page);
  const shown=OK.filter(s=>!V.hidden.has(s.id)&&SLOTS[s.id]);
  const cx=WORLD/2, cy=WORLD/2;
  let html='';

  if(V.links){
    let g=`<svg class="orbit-rings" width="${WORLD}" height="${WORLD}">`;
    const rings=[...new Set(shown.map(s=>SLOTS[s.id].ring))];
    rings.forEach(r=>{g+=`<circle cx="${cx}" cy="${cy}" r="${RING_R[r]}" fill="none" stroke="var(--rule-2)" stroke-width="1.5" stroke-dasharray="3 9" opacity="0.75"/>`;});
    shown.forEach(s=>{
      const sl=SLOTS[s.id];
      const x=cx+Math.cos(sl.angle)*RING_R[sl.ring], y=cy+Math.sin(sl.angle)*RING_R[sl.ring];
      const m=(page.metrics||{})[s.id]||{};
      const agree=Math.max(0.12,Math.min(1,(m.text_or_table_recall||0)));
      g+=`<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="var(--accent)" stroke-width="${(0.8+agree*2.6).toFixed(2)}" opacity="${(0.10+agree*0.30).toFixed(2)}"/>`;
    });
    html+=g+'</svg>';
  }

  // hub — the reference page
  const hubH=Math.round(HUB_W*page.h/page.w);
  html+=`<div class="hub" style="left:${cx-HUB_W/2}px;top:${cy-(hubH+96)/2}px;width:${HUB_W}px">
    <div class="hub-hd">Reference page · ${esc(page.id)}
      <small>${esc(page.doc)} · p${page.page} · ${esc(page.stratum.replace(/_/g,' '))} · ${page.w}×${page.h}px</small></div>
    <div class="hub-fig"><img src="${page.img}" alt="reference page">${V.ref?overlaySVG(page,null):''}</div>
    <div class="hub-mx"><span>body lines <b>${page.ref.n_body_lines}</b></span>
      <span>columns <b>${page.ref.columns}</b></span>
      <span>graphics <b>${page.ref.n_graphics}</b></span>
      <span>ruled grids <b>${page.ref.n_grids}</b></span></div></div>`;

  shown.forEach(s=>{
    const sl=SLOTS[s.id], w=SAT_W[sl.ring];
    const h=Math.round(w*page.h/page.w);
    const x=cx+Math.cos(sl.angle)*RING_R[sl.ring]-w/2;
    const y=cy+Math.sin(sl.angle)*RING_R[sl.ring]-(h+82)/2;
    const m=(page.metrics||{})[s.id]||{};
    const pr=(D.predictions[page.id]||{})[s.id];
    const n=visibleRegions(page.id,s.id).length;
    const nm=shortName(s);
    html+=`<div class="sat" data-sys="${s.id}" style="left:${x}px;top:${y}px;width:${w}px">
      <div class="sat-hd"><span class="t">${esc(nm.b||nm.a)}<small>${esc(nm.a)} · ring ${sl.ring+1}·${sl.slot+1}</small></span>
        <span class="n">${n}</span></div>
      <div class="sat-fig"><img src="${page.img}" alt="">${overlaySVG(page,s.id,w*1.2)}</div>
      <div class="sat-mx"><span>recall <b>${pct(m.text_or_table_recall)}</b></span>
        <span>prec <b>${pct(m.text_precision)}</b></span>
        <span>fig <b>${fmt(m.graphic_iou,2)}</b></span>
        <span>${pr?pr.t.toFixed(0):'—'} ms</span></div></div>`;
  });
  world.innerHTML=html;
  world.style.width=WORLD+'px';world.style.height=WORLD+'px';
  // Hover magnifies the tile to a readable size *regardless of current zoom*,
  // so the orbit stays an overview without forcing a click to read anything.
  world.querySelectorAll('.sat').forEach(node=>{
    node.addEventListener('click',()=>openFocus(node.dataset.sys));
    node.addEventListener('mouseenter',()=>{
      const k=Math.max(1,1.15/view.k);
      node.style.transform=`scale(${k.toFixed(3)})`;
      node.classList.add('mag');});
    node.addEventListener('mouseleave',()=>{
      node.style.transform='';node.classList.remove('mag');});
  });
}

function fitView(){
  const shown=OK.filter(s=>!V.hidden.has(s.id)&&SLOTS[s.id]);
  const maxRing=shown.length?Math.max(...shown.map(s=>SLOTS[s.id].ring)):0;
  const need=(RING_R[maxRing]+340)*2;
  const r=stage.getBoundingClientRect();
  view.k=Math.min(r.width/need,r.height/need);
  view.x=r.width/2-(WORLD/2)*view.k;
  view.y=r.height/2-(WORLD/2)*view.k;
  applyView();
}
$('#zIn').addEventListener('click',()=>{view.k=Math.min(2.5,view.k*1.25);applyView();});
$('#zOut').addEventListener('click',()=>{view.k=Math.max(0.08,view.k/1.25);applyView();});
$('#zFit').addEventListener('click',fitView);
stage.addEventListener('wheel',e=>{
  e.preventDefault();
  const r=stage.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  const f=e.deltaY<0?1.12:1/1.12,nk=Math.max(0.08,Math.min(2.5,view.k*f));
  view.x=mx-(mx-view.x)*(nk/view.k);view.y=my-(my-view.y)*(nk/view.k);view.k=nk;applyView();
},{passive:false});
let drag=null;
stage.addEventListener('pointerdown',e=>{
  if(e.target.closest('.sat')||e.target.closest('.zoombar'))return;
  drag={x:e.clientX-view.x,y:e.clientY-view.y};stage.classList.add('dragging');
  stage.setPointerCapture(e.pointerId);});
stage.addEventListener('pointermove',e=>{
  if(!drag)return;view.x=e.clientX-drag.x;view.y=e.clientY-drag.y;applyView();});
stage.addEventListener('pointerup',()=>{drag=null;stage.classList.remove('dragging');});
stage.addEventListener('pointercancel',()=>{drag=null;stage.classList.remove('dragging');});

/* focus overlay */
function openFocus(sid){
  const page=D.pages.find(p=>p.id===V.page),s=byId[sid];
  const m=(page.metrics||{})[sid]||{},pr=(D.predictions[page.id]||{})[sid];
  $('#focusTitle').textContent=`${s.display} — ${page.id}`;
  $('#focusGrid').innerHTML=`
    <div class="focus-pane"><div class="hd"><span>Reference page</span>
      <span class="tiny">${page.w}×${page.h}px · 300 dpi</span></div>
      <div class="focus-fig"><img src="${page.img}" alt="">${V.ref?overlaySVG(page,null):''}</div></div>
    <div class="focus-pane"><div class="hd"><span>${esc(s.display)}</span>
      <span class="tiny">${visibleRegions(page.id,sid).length} regions · ${pr?pr.t.toFixed(0):'—'} ms</span></div>
      <div class="focus-fig"><img src="${page.img}" alt="">${overlaySVG(page,sid,900)}</div></div>`;
  $('#focusMeta').innerHTML=
    `<strong>${esc(s.display)}</strong> on ${esc(page.doc)} p${page.page} — text recall
     ${pct(m.text_recall)}, text+table ${pct(m.text_or_table_recall)}, precision
     ${pct(m.text_precision)}, spill ${pct(m.text_spill)}, figure IoU ${fmt(m.graphic_iou,3)},
     overlap ${pct(m.overlap_ratio)}, ${fmt(m.n_regions,0)} regions.`;
  $('#focus').classList.add('on');
}
$('#focusClose').addEventListener('click',()=>$('#focus').classList.remove('on'));
$('#focus').addEventListener('click',e=>{if(e.target.id==='focus')$('#focus').classList.remove('on');});
document.addEventListener('keydown',e=>{if(e.key==='Escape')$('#focus').classList.remove('on');});

function renderRingMap(){
  const shown=OK.filter(s=>!V.hidden.has(s.id)&&SLOTS[s.id]);
  const rings={};
  shown.forEach(s=>{(rings[SLOTS[s.id].ring]=rings[SLOTS[s.id].ring]||[]).push(s);});
  $('#ringMap').innerHTML=Object.keys(rings).sort().map(r=>
    `<b>ring ${+r+1}</b>`+rings[r].sort((a,b)=>SLOTS[a.id].slot-SLOTS[b.id].slot)
      .map(s=>`<span><i>${SLOTS[s.id].slot+1}</i>${esc(shortName(s).b||shortName(s).a)}</span>`).join('')
  ).join('');
}
const _ro=renderOrbit;
renderOrbit=function(){_ro();renderRingMap();};

renderOrbit();
requestAnimationFrame(fitView);
addEventListener('resize',()=>{clearTimeout(window.__rt);window.__rt=setTimeout(fitView,180);});

/* ================= PER-PAGE COUNT MATRIX ================= */
function renderCounts(){
  if(!have('#countTable'))return;
  const pid=V.page;
  const counts=(ENS.per_page_class_counts||{})[pid]||{};
  const present=CLS.filter(k=>OK.some(s=>(counts[s.id]||{})[k]));
  const med={};
  present.forEach(k=>{
    const v=OK.map(s=>(counts[s.id]||{})[k]||0).sort((a,b)=>a-b);
    med[k]=v.length?(v.length%2?v[(v.length-1)/2]:Math.round((v[v.length/2-1]+v[v.length/2])/2)):0;
  });
  const head='<thead><tr><th>Model</th>'+present.map(k=>
    `<th style="cursor:default"><span style="display:inline-flex;align-items:center;gap:6px">
      <i style="width:9px;height:9px;border-radius:2px;background:${COL[k]};display:inline-block"></i>${k}</span></th>`).join('')
    +'<th style="cursor:default">total</th></tr></thead>';
  const rows=OK.slice().sort((a,b)=>{
    const ta=present.reduce((x,k)=>x+((counts[a.id]||{})[k]||0),0);
    const tb=present.reduce((x,k)=>x+((counts[b.id]||{})[k]||0),0);
    return ta-tb;}).map(s=>{
    const c=counts[s.id]||{};
    const tot=present.reduce((x,k)=>x+(c[k]||0),0);
    return `<tr><td class="name">${esc(s.display)}<span class="repo">${s.id}</span></td>`+
      present.map(k=>{
        const v=c[k]||0,dev=Math.abs(v-med[k]),rel=med[k]?dev/Math.max(med[k],1):(v?1:0);
        const op=v===0?0:Math.min(.5,.05+rel*.42);
        return `<td class="num heat ${v?'hi':'c0'}">
          <i style="background:${dev>0?'var(--bad)':'var(--good)'};opacity:${op.toFixed(2)}"></i>
          <span>${v||'·'}</span></td>`;}).join('')+
      `<td class="num">${tot}</td></tr>`;}).join('');
  const consRow=`<tr style="background:var(--accent-soft)"><td class="name">Consensus (median)</td>`+
    present.map(k=>`<td class="num" style="font-weight:700">${med[k]}</td>`).join('')+
    `<td class="num" style="font-weight:700">${present.reduce((x,k)=>x+med[k],0)}</td></tr>`;
  $('#countTable').innerHTML=head+'<tbody>'+consRow+rows+'</tbody>';
}
renderCounts();

/* ================= ROUTING ================= */
(function(){
  if(!have('#routeCards'))return;
  const R=ENS.routing||{},P=ENS.per_class||{};
  const badge=d=>({route:'good','tied at ceiling':'ok','no separable leader':'warn',
    'unstable leader':'warn','insufficient evidence':'bad'})[d]||'';
  const order=Object.keys(R).sort((a,b)=>
    (P[b]?.n_consensus_regions||0)-(P[a]?.n_consensus_regions||0));
  $('#routeCards').innerHTML=order.map(k=>{
    const r=R[k],p=P[k]||{};
    const eq=(r.equivalent_group||[]).map(x=>byId[x]?byId[x].display:x);
    return `<div class="route-card">
      <h4><i style="width:12px;height:12px;border-radius:3px;background:${COL[k]};display:inline-block"></i>
        ${k} <span class="badge ${badge(r.decision)}">${r.decision}</span></h4>
      <p class="why"><strong>${esc(byId[r.system]?byId[r.system].display:r.system)}</strong> — ${esc(r.rationale)}</p>
      <p class="tiny" style="margin:.5em 0 0">${p.n_consensus_regions||0} consensus regions across
        ${p.n_pages_with_class||0} pages${eq.length>1?` · equivalent: ${esc(eq.slice(0,4).join(', '))}${eq.length>4?' …':''}`:''}</p>
    </div>`;}).join('');

  const rows=order.map(k=>{
    const p=P[k]||{},r=R[k];
    const eq=(p.equivalent_group||[]).map(x=>byId[x]?byId[x].display:x);
    return `<tr><td><span class="badge" style="border-color:${COL[k]};color:${COL[k]}">${k}</span></td>
      <td class="num">${p.n_consensus_regions||0}</td>
      <td class="num">${p.n_pages_with_class||0}</td>
      <td class="name" style="min-width:200px">${esc(byId[p.leader]?byId[p.leader].display:p.leader||'—')}</td>
      <td class="num">${fmt(p.leader_f1,3)}</td>
      <td class="num">${fmt(p.margin,3)}</td>
      <td class="num">${pct(p.leader_page_win_rate)}</td>
      <td style="white-space:normal;min-width:260px;font-size:.8rem;color:var(--ink-2)">${esc(eq.join(', '))}</td>
      <td><span class="badge ${badge(r.decision)}">${r.decision}</span></td></tr>`;}).join('');
  $('#classTable').innerHTML=`<thead><tr><th style="cursor:default">Class</th>
    <th style="cursor:default">Consensus regions</th><th style="cursor:default">Pages</th>
    <th style="cursor:default">Leader</th><th style="cursor:default">F1</th>
    <th style="cursor:default">Margin</th><th style="cursor:default">Page win rate</th>
    <th style="cursor:default">Statistically equivalent</th>
    <th style="cursor:default">Decision</th></tr></thead><tbody>${rows}</tbody>`;
})();

/* ================= RATINGS ================= */
(function(){
  if(!have('#ratingTable'))return;
  const dims=Object.keys(D.rubric||{}),cols=[...dims,'Overall layout'];
  const rows=OK.filter(s=>s.ratings&&Object.keys(s.ratings).length)
    .sort((a,b)=>(b.ratings['Overall layout']||0)-(a.ratings['Overall layout']||0));
  const head='<thead><tr><th style="cursor:default">Model</th>'+cols.map(c=>
    `<th style="cursor:default" title="${esc((D.rubric||{})[c]||'weighted mean over rated dimensions')}">${c}</th>`).join('')+'</tr></thead>';
  const cell=v=>v===null||v===undefined
    ?'<td class="num" style="color:var(--ink-3)">N/A</td>'
    :`<td class="num heat"><i style="background:var(--accent);opacity:${(0.06+0.44*(v-1)/4).toFixed(3)}"></i><span>${(+v).toFixed(v%1?2:0)}</span></td>`;
  $('#ratingTable').innerHTML=head+'<tbody>'+rows.map(s=>
    `<tr><td class="name">${esc(s.display)}<span class="repo">${s.id}</span></td>`+
    cols.map(c=>cell(s.ratings[c])).join('')+'</tr>').join('')+'</tbody>';
})();

/* ================= METRIC TABLE ================= */
const MCOLS=[{k:'_name',t:'Model'},
 {k:'text_recall',t:'Text recall',hi:1},{k:'text_or_table_recall',t:'Text+table recall',hi:1},
 {k:'text_precision',t:'Text precision',hi:1},{k:'text_spill',t:'Spill',hi:0},
 {k:'line_capture',t:'Line capture',hi:1},{k:'any_region_capture',t:'Any-region capture',hi:1},
 {k:'graphic_iou',t:'Figure IoU',hi:1},{k:'gutter_cross_rate',t:'Gutter cross',hi:0,d:4},
 {k:'overlap_ratio',t:'Overlap',hi:0},{k:'class_diversity',t:'Classes/page',hi:1,d:1},
 {k:'n_regions',t:'Regions/page',hi:null,d:1},{k:'inference_s',t:'Infer s',hi:0,d:3}];
let sortKey='text_or_table_recall',sortDir=-1;
const mv=(s,k)=>k==='_name'?s.display:((s.metrics[k]||{}).median??null);
function buildMetricTable(){
  if(!have('#metricTable'))return;
  const rows=OK.slice().sort((a,b)=>{
    const x=mv(a,sortKey),y=mv(b,sortKey);
    if(sortKey==='_name')return sortDir*String(x).localeCompare(String(y));
    if(x===null)return 1;if(y===null)return -1;return sortDir*(x-y);});
  const rng={};MCOLS.forEach(c=>{if(c.k==='_name')return;
    const v=rows.map(r=>mv(r,c.k)).filter(x=>x!==null);rng[c.k]=[Math.min(...v),Math.max(...v)];});
  $('#metricTable').innerHTML='<thead><tr>'+MCOLS.map(c=>
    `<th data-k="${c.k}" class="${c.k===sortKey?(sortDir<0?'desc':'asc'):''}">${c.t}</th>`).join('')+
    '</tr></thead><tbody>'+rows.map(s=>'<tr>'+MCOLS.map(c=>{
      if(c.k==='_name'){
        // Systems a paired sign test cannot separate from the leader on this
        // metric are marked: within that group the table order is noise, not
        // evidence.  See D.separability.
        const sep=(D.separability||{})[sortKey];
        const tie=sep&&sep.group&&sep.group.indexOf(s.id)>=0;
        const lead=sep&&sep.leader===s.id;
        const tag=tie?`<span class="tiebadge" title="${lead?'Leader on this metric':'Not separable from the leader on this metric (paired sign test, p='+((sep.p_vs_leader||{})[s.id]??'?')+')'}">${lead?'leader':'tie'}</span>`:'';
        return `<td class="name">${esc(s.display)}${tag}<span class="repo">${s.id}</span></td>`;}
      const v=mv(s,c.k);if(v===null)return '<td class="num">—</td>';
      let sh='';
      if(c.hi!==null&&c.hi!==undefined){
        const [lo,hi]=rng[c.k],sp=(hi-lo)||1;let n=(v-lo)/sp;if(!c.hi)n=1-n;
        sh=`<i style="background:var(--accent);opacity:${(0.06+0.42*n).toFixed(3)}"></i>`;}
      return `<td class="num heat">${sh}<span>${fmt(v,c.d===undefined?3:c.d)}</span></td>`;
    }).join('')+'</tr>').join('')+'</tbody>';
  $('#metricTable').querySelectorAll('th[data-k]').forEach(th=>th.addEventListener('click',()=>{
    const k=th.dataset.k;
    if(k===sortKey)sortDir*=-1;else{sortKey=k;sortDir=(k==='_name')?1:-1;}
    buildMetricTable();}));
}
buildMetricTable();

/* ================= SCATTER + PERF ================= */
function drawScatter(){
  if(!have('#scatter'))return;
  const pts=OK.map(s=>({s,x:(s.metrics.inference_s||{}).median,
    y:(s.metrics.text_or_table_recall||{}).median,
    m:(s.resources||{}).cuda_peak_alloc_mb||0,
    cpu:!((s.resources||{}).cuda_peak_alloc_mb)})).filter(p=>p.x&&p.y);
  const W=900,H=460,PL=58,PR=20,PT=18,PB=48;
  const xs=pts.map(p=>Math.log10(p.x)),ys=pts.map(p=>p.y);
  const x0=Math.min(...xs)-.14,x1=Math.max(...xs)+.14;
  const y0=Math.max(0,Math.min(...ys)-.07),y1=Math.min(1,Math.max(...ys)+.07);
  const X=v=>PL+(Math.log10(v)-x0)/(x1-x0)*(W-PL-PR);
  const Y=v=>H-PB-(v-y0)/(y1-y0)*(H-PT-PB);
  const maxM=Math.max(1,...pts.map(p=>p.m));
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="quality versus speed">`;
  for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4,y=Y(v);
    s+=`<line x1="${PL}" y1="${y}" x2="${W-PR}" y2="${y}" stroke="var(--rule)"/>`;
    s+=`<text x="${PL-9}" y="${y+3}" text-anchor="end">${(v*100).toFixed(0)}%</text>`;}
  [0.02,0.05,0.1,0.25,0.5,1,2,4,10,20,40,80].filter(t=>Math.log10(t)>=x0&&Math.log10(t)<=x1).forEach(t=>{
    const x=X(t);
    s+=`<line x1="${x}" y1="${PT}" x2="${x}" y2="${H-PB}" stroke="var(--rule)"/>`;
    s+=`<text x="${x}" y="${H-PB+16}" text-anchor="middle">${t<1?t*1000+'ms':t+'s'}</text>`;});
  s+=`<text class="axl" x="${(W+PL)/2}" y="${H-8}" text-anchor="middle">median inference per page (log)</text>`;
  s+=`<text class="axl" x="14" y="${(H-PB+PT)/2}" text-anchor="middle" transform="rotate(-90 14 ${(H-PB+PT)/2})">text + table recall</text>`;
  pts.sort((a,b)=>b.m-a.m).forEach(p=>{
    const r=7+Math.sqrt(p.m/maxM)*17,cx=X(p.x),cy=Y(p.y);
    s+=`<circle cx="${cx}" cy="${cy}" r="${r.toFixed(1)}" fill="${p.cpu?'none':'var(--accent)'}" fill-opacity="0.2" stroke="var(--accent)" stroke-width="${p.cpu?2:1.4}" ${p.cpu?'stroke-dasharray="3 3"':''}/>`;
    s+=`<circle cx="${cx}" cy="${cy}" r="2.6" fill="var(--ink)"/>`;
    s+=`<text x="${cx+r+6}" y="${cy+3}" fill="var(--ink-2)">${esc(p.s.id)}</text>`;});
  $('#scatter').innerHTML=s+'</svg>';
}
drawScatter();
(function(){
  if(!have('#perfTableWrap'))return;
  const rows=OK.slice().sort((a,b)=>((a.metrics.inference_s||{}).median||9)-((b.metrics.inference_s||{}).median||9))
   .map(s=>{const r=s.resources||{},m=s.metrics||{},it=(m.inference_s||{}).median;
    return `<tr><td class="name">${esc(s.display)}<span class="repo">${s.id}</span></td>
      <td class="mono" style="font-size:.72rem">${esc(s.device||r.device||'—')}</td>
      <td class="num">${fmt(it,3)}</td><td class="num">${it?fmt(1/it,2):'—'}</td>
      <td class="num">${r.cuda_peak_alloc_mb?Math.round(r.cuda_peak_alloc_mb):'—'}</td>
      <td class="num">${r.peak_rss_mb?Math.round(r.peak_rss_mb):'—'}</td>
      <td class="num">${fmt(s.model_load_s,1)}</td></tr>`;}).join('');
  $('#perfTableWrap').innerHTML=`<div class="eyebrow" style="margin-bottom:14px">Measured cost</div>
    <div style="overflow-x:auto"><table><thead><tr><th style="cursor:default">Model</th>
    <th style="cursor:default">Device</th><th style="cursor:default">s/page</th>
    <th style="cursor:default">pages/s</th><th style="cursor:default">VRAM MB</th>
    <th style="cursor:default">RSS MB</th><th style="cursor:default">Load s</th>
    </tr></thead><tbody>${rows}</tbody></table></div>`;
})();

/* ================= CONSENSUS ================= */
(function(){
  if(!have('#consTable'))return;
  const rows=OK.slice().sort((a,b)=>(b.consensus.class_agreement||0)-(a.consensus.class_agreement||0))
   .map(s=>{const c=s.consensus||{};
    return `<tr><td class="name">${esc(s.display)}<span class="repo">${s.id}</span></td>
      <td class="num">${pct(c.consensus_recall)}</td><td class="num">${pct(c.class_agreement)}</td>
      <td class="num">${pct(c.consensus_precision)}</td><td class="num">${pct(c.solo_rate)}</td>
      <td class="num">${c.n_regions||'—'}</td>
      <td class="num">${s.device_agreement==null?'—':pct(s.device_agreement)}</td></tr>`;}).join('');
  put('#consensusBody',(N.consensus||'')+
   `<div class="tablewrap" style="margin-top:24px"><table><thead><tr>
    <th style="cursor:default">Model</th><th style="cursor:default">Consensus recall</th>
    <th style="cursor:default">Class agreement</th><th style="cursor:default">Backed regions</th>
    <th style="cursor:default">Solo regions</th><th style="cursor:default">Regions</th>
    <th style="cursor:default">GPU=CPU</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="small" style="margin-top:14px">${N.consensusNote||''}</p>`);
})();

/* ================= SYSTEM CARDS ================= */
(function(){
  if(!have('#sysCards'))return;
  const host=$('#sysCards');
  const order=[...OK].sort((a,b)=>((b.ratings||{})['Overall layout']||0)-((a.ratings||{})['Overall layout']||0));
  const rest=D.systems.filter(s=>s.status!=='ok');
  [...order,...rest].forEach(s=>{
    const n=(N.systems||{})[s.id]||{},m=s.metrics||{},r=s.resources||{},mod=s.model||{};
    const kv=[];const push=(k,v)=>{if(v!=null&&v!=='')kv.push(`<dt>${k}</dt><dd>${esc(String(v))}</dd>`);};
    push('repository',s.repo);
    push('checkpoint',mod.repo_id||mod.model_name||mod.catalog||mod.model||mod.checkpoint||mod.hf_repo);
    push('weights',mod.weights||mod.weights_dir);
    push('architecture',mod.architecture||mod.arch);
    push('runtime',mod.framework); push('device',s.device||mod.device||r.device);
    push('config',Object.keys(s.config||{}).length?JSON.stringify(s.config):null);
    const badges=[`<span class="badge ${s.status==='ok'?'ok':''}">${s.status}</span>`];
    if(mod.reading_order)badges.push('<span class="badge good">reading order</span>');
    if((m.has_polygons||{}).mean>0)badges.push('<span class="badge good">instance masks</span>');
    if((s.ratings||{})['Overall layout'])badges.push(`<span class="badge">rating ${s.ratings['Overall layout']}/5</span>`);
    host.appendChild(el('div',{class:'syscard',html:`
      <header><h3>${esc(s.display)}</h3><div class="chips" style="gap:6px">${badges.join('')}</div></header>
      <div class="body">
        ${s.status==='ok'?`<div class="statrow">
          <span>text+table <b>${pct((m.text_or_table_recall||{}).median)}</b></span>
          <span>precision <b>${pct((m.text_precision||{}).median)}</b></span>
          <span>figure IoU <b>${fmt((m.graphic_iou||{}).median,3)}</b></span>
          <span>s/page <b>${fmt((m.inference_s||{}).median,3)}</b></span>
          <span>regions/page <b>${fmt((m.n_regions||{}).median,1)}</b></span></div>`:''}
        ${n.verdict?`<p>${n.verdict}</p>`:''}
        ${n.strengths?`<div class="eyebrow" style="margin:0 0 6px">Strengths</div><ul class="tight">${n.strengths.map(x=>'<li>'+x+'</li>').join('')}</ul>`:''}
        ${n.weaknesses?`<div class="eyebrow" style="margin:0 0 6px">Weaknesses</div><ul class="tight">${n.weaknesses.map(x=>'<li>'+x+'</li>').join('')}</ul>`:''}
        ${n.prod?`<div class="eyebrow" style="margin:0 0 6px">Production</div><p class="small" style="margin:0 0 1em">${n.prod}</p>`:''}
        <dl class="kv">${kv.join('')}</dl>
      </div>`}));
  });
})();

/* ================= CORPUS ================= */
(function(){
  if(!have('#corpusTable'))return;
  const rows=D.corpus.documents.map(d=>`<tr><td class="name">${esc(d.file)}</td>
    <td class="num">${d.pages}</td><td class="num">${(d.size_bytes/1e6).toFixed(1)}</td>
    <td class="num">${(d.script.arabic_ratio*100).toFixed(1)}%</td>
    <td class="num">${d.total_chars.toLocaleString()}</td>
    <td class="mono" style="font-size:.74rem">${Object.entries(d.page_sizes).map(([k,v])=>k+'×'+v).join(', ')}</td></tr>`).join('');
  const strata={};D.pages.forEach(p=>strata[p.stratum]=(strata[p.stratum]||0)+1);
  $('#corpusBody').innerHTML=(N.corpus||'')+
    `<div class="tablewrap" style="margin:28px 0"><table><thead><tr>
      <th style="cursor:default">Document</th><th style="cursor:default">Pages</th>
      <th style="cursor:default">MB</th><th style="cursor:default">RTL script</th>
      <th style="cursor:default">Characters</th><th style="cursor:default">Page sizes (pt)</th>
      </tr></thead><tbody>${rows}</tbody></table></div>
    <div class="eyebrow" style="margin:32px 0 10px">${D.pages.length} pages</div>
    <div class="chips">${Object.entries(strata).sort((a,b)=>b[1]-a[1])
      .map(([k,v])=>`<span class="chip" style="cursor:default">${k.replace(/_/g,' ')} <span class="ct">${v}</span></span>`).join('')}</div>
    ${N.corpus2||''}`;
})();

/* ================= TAXONOMY ================= */
(function(){
  if(!have('#taxTable'))return;
  const taxes=[...new Set(D.taxonomy_map.map(r=>r.taxonomy))].sort();
  const active=new Set(taxes),f=$('#taxFilter');
  taxes.forEach(t=>{
    const b=el('button',{class:'chip','aria-pressed':'true',text:t});
    b.addEventListener('click',()=>{if(active.has(t))active.delete(t);else active.add(t);
      b.setAttribute('aria-pressed',String(active.has(t)));draw();});
    f.appendChild(b);});
  function draw(){
    $('#taxTable').innerHTML=`<thead><tr><th style="cursor:default">Taxonomy</th>
      <th style="cursor:default">Source class</th><th style="cursor:default">Canonical</th>
      <th style="cursor:default">Mapping</th><th style="cursor:default">Note</th></tr></thead><tbody>`+
      D.taxonomy_map.filter(r=>active.has(r.taxonomy)).map(r=>
      `<tr><td class="mono" style="font-size:.72rem">${r.taxonomy}</td>
       <td class="mono">${esc(r.source_class)}</td>
       <td><span class="chip" style="cursor:default"><i class="sw" style="background:${COL[r.canonical_class]||'#888'}"></i>${r.canonical_class}</span></td>
       <td><span class="badge ${r.confidence==='exact'?'good':(r.confidence==='ambiguous'?'warn':'')}">${r.confidence}</span></td>
       <td style="white-space:normal;font-size:.82rem;color:var(--ink-2);min-width:300px">${esc(r.notes||'')}</td></tr>`).join('')+'</tbody>';
  }
  draw();
})();
})();
