"""Build a single-file, self-contained HTML audit tool for 卢晓琳 (③).
Blind evaluation: each essay's two feedback versions are shown as 'Feedback A/B'
(true RAG arm hidden in data, revealed only on export). Auto-saves to localStorage.
"""
import json, random, os
BASE = "results"
bundle = json.load(open(os.path.join(BASE, "audit", "audit_items.json")))
essays = {e["essay_id"]: e for e in bundle["essays"]}
items = bundle["items"]

random.seed(42)
# per-essay blind mapping: assign RAGoff/RAGon -> A/B randomly
blind = {}
for eid in essays:
    arms = ["RAGoff", "RAGon"]
    random.shuffle(arms)
    blind[eid] = {arms[0]: "A", arms[1]: "B"}
for it in items:
    it["blind"] = blind[it["essay_id"]][it["arm"]]

data = {"essays": list(essays.values()), "items": items}

HTML = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>反馈审计工具 — LT6582</title>
<style>
*{box-sizing:border-box} body{font-family:-apple-system,"PingFang SC",Segoe UI,sans-serif;margin:0;background:#f4f5f7;color:#1c2024;line-height:1.5}
header{position:sticky;top:0;background:#1f2d3d;color:#fff;padding:12px 20px;z-index:10;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header h1{font-size:16px;margin:0}
header input{padding:6px 10px;border-radius:6px;border:1px solid #ccc}
.btn{background:#2f6feb;color:#fff;border:0;padding:8px 14px;border-radius:6px;cursor:pointer;font-size:14px}
.btn.alt{background:#3a4a5a} .btn:hover{opacity:.9}
.progress{font-size:13px;color:#cdd}
main{max-width:1000px;margin:0 auto;padding:20px}
.essay{background:#fff;border-radius:10px;margin-bottom:24px;box-shadow:0 1px 4px rgba(0,0,0,.08);overflow:hidden}
.essay>summary{cursor:pointer;padding:14px 18px;font-weight:600;background:#eef1f5;font-size:15px}
.etext{padding:12px 18px;white-space:pre-wrap;background:#fbfbfd;border-bottom:1px solid #eee;font-size:13px;color:#444;max-height:240px;overflow:auto}
.fb{padding:14px 18px;border-top:6px solid #f0f0f0}
.fb h3{margin:0 0 4px;font-size:15px}
.card{border:1px solid #e3e6ea;border-radius:8px;padding:12px;margin:10px 0;background:#fff}
.dim{display:inline-block;font-size:11px;background:#eaf0ff;color:#2f6feb;padding:2px 8px;border-radius:10px;margin-bottom:6px}
.quote{font-style:italic;color:#8a1f1f;background:#fdf3f3;padding:6px 8px;border-radius:5px;margin:4px 0;font-size:13px}
.issue{font-size:14px;margin:4px 0} .corr{font-size:13px;color:#2a6}
.controls{display:flex;gap:18px;flex-wrap:wrap;margin-top:8px;padding-top:8px;border-top:1px dashed #e3e6ea}
.ctl label{font-size:12px;color:#555;display:block;margin-bottom:3px}
.seg button{border:1px solid #ccc;background:#fff;padding:4px 9px;cursor:pointer;font-size:13px}
.seg button:first-child{border-radius:6px 0 0 6px} .seg button:last-child{border-radius:0 6px 6px 0}
.seg button.on{background:#2f6feb;color:#fff;border-color:#2f6feb}
.seg button.on.bad{background:#d9433f;border-color:#d9433f}
.fblevel{background:#f7f9fc;border-radius:8px;padding:12px;margin-top:8px}
.fblevel textarea{width:100%;min-height:60px;border:1px solid #ccc;border-radius:6px;padding:8px;font-size:13px;font-family:inherit}
.hint{font-size:12px;color:#777;margin:2px 0 6px}
</style></head><body>
<header>
  <h1>📝 反馈审计工具</h1>
  <span>审计员：</span><input id="coder" placeholder="你的名字" oninput="save()">
  <span class="progress" id="prog"></span>
  <button class="btn" onclick="exportJSON()">导出 JSON</button>
  <button class="btn alt" onclick="exportCSV()">导出 CSV</button>
</header>
<main id="app"></main>
<script>
const DATA = __DATA__;
const KEY = "lt6582_audit";
let state = JSON.parse(localStorage.getItem(KEY) || "{}");
function save(){ state.coder=document.getElementById('coder').value; localStorage.setItem(KEY, JSON.stringify(state)); prog(); }
function setItem(id,field,val,bad){ state['item_'+id]=state['item_'+id]||{}; state['item_'+id][field]=val; save();
  document.querySelectorAll('[data-grp="'+id+'_'+field+'"] button').forEach(b=>{b.classList.remove('on');b.classList.toggle('on', b.dataset.val==String(val)); if(bad&&b.dataset.val==String(val))b.classList.add('bad');}); }
function setFb(key,field,val){ state['fb_'+key]=state['fb_'+key]||{}; state['fb_'+key][field]=val; save();
  document.querySelectorAll('[data-grp="'+key+'_'+field+'"] button').forEach(b=>{b.classList.remove('on');b.classList.toggle('on', b.dataset.val==String(val));}); }
function setMissed(key,val){ state['fb_'+key]=state['fb_'+key]||{}; state['fb_'+key].missed=val; save(); }
function prog(){ const tot=DATA.items.length; let done=0; DATA.items.forEach(i=>{if(state['item_'+i.item_id]&&state['item_'+i.item_id].verdict)done++;});
  document.getElementById('prog').textContent=`已评 ${done}/${tot} 条`; }

const seg=(grp,opts)=>`<div class="seg" data-grp="${grp}">`+opts.map(o=>`<button data-val="${o.v}" onclick="${o.fn}">${o.t}</button>`).join('')+`</div>`;

function render(){
  const app=document.getElementById('app');
  DATA.essays.forEach(e=>{
    const blindArms=[...new Set(DATA.items.filter(i=>i.essay_id===e.essay_id).map(i=>i.blind))].sort();
    let html=`<details class="essay" open><summary>${e.essay_id} （proficiency: ${e.proficiency}）— 点击展开/收起作文原文</summary>`;
    html+=`<div class="etext">${e.text.replace(/</g,'&lt;')}</div>`;
    blindArms.forEach(bl=>{
      const its=DATA.items.filter(i=>i.essay_id===e.essay_id && i.blind===bl);
      html+=`<div class="fb"><h3>反馈 ${bl}</h3><div class="hint">逐条判断 AI 标记的这个“错误”是否成立。</div>`;
      its.forEach(it=>{
        html+=`<div class="card"><span class="dim">${it.dimension}</span>`;
        if(it.quote) html+=`<div class="quote">学生原句：${it.quote.replace(/</g,'&lt;')}</div>`;
        html+=`<div class="issue"><b>AI 指出：</b>${it.issue.replace(/</g,'&lt;')}</div>`;
        if(it.correction) html+=`<div class="corr">建议改为：${it.correction.replace(/</g,'&lt;')}</div>`;
        html+=`<div class="controls">
          <div class="ctl"><label>这是真错误吗？</label>${seg(it.item_id+'_verdict',[
            {v:'TP',t:'真错误',fn:`setItem(${it.item_id},'verdict','TP')`},
            {v:'FP',t:'假错误(误报)',fn:`setItem(${it.item_id},'verdict','FP',true)`},
            {v:'unsure',t:'不确定',fn:`setItem(${it.item_id},'verdict','unsure')`}])}</div>
          <div class="ctl"><label>维度归类对吗？</label>${seg(it.item_id+'_dim_ok',[
            {v:'yes',t:'对',fn:`setItem(${it.item_id},'dim_ok','yes')`},
            {v:'no',t:'错',fn:`setItem(${it.item_id},'dim_ok','no',true)`}])}</div>
          <div class="ctl"><label>对学生有用度 (1低-4高)</label>${seg(it.item_id+'_utility',[1,2,3,4].map(n=>(
            {v:n,t:n,fn:`setItem(${it.item_id},'utility',${n})`})))}</div>
        </div></div>`;
      });
      const key=e.essay_id+'_'+bl;
      html+=`<div class="fblevel"><label><b>反馈 ${bl} 整体语气</b></label>${seg(key+'_tone',[
        {v:'appropriate',t:'恰当',fn:`setFb('${key}','tone','appropriate')`},
        {v:'acceptable',t:'可接受',fn:`setFb('${key}','tone','acceptable')`},
        {v:'inappropriate',t:'不当',fn:`setFb('${key}','tone','inappropriate')`}])}
        <div class="hint" style="margin-top:8px">AI <b>漏掉</b>的真实错误(每行一个,用于算漏报 FN):</div>
        <textarea oninput="setMissed('${key}', this.value)" placeholder="例如：第二段 they're 拼成 their ……">${(state['fb_'+key]&&state['fb_'+key].missed)||''}</textarea>
      </div>`;
      html+=`</div>`;
    });
    html+=`</details>`;
    app.insertAdjacentHTML('beforeend',html);
  });
  // restore selections
  document.getElementById('coder').value=state.coder||'';
  DATA.items.forEach(i=>{const s=state['item_'+i.item_id]; if(s){if(s.verdict)setItem(i.item_id,'verdict',s.verdict,s.verdict==='FP');if(s.dim_ok)setItem(i.item_id,'dim_ok',s.dim_ok,s.dim_ok==='no');if(s.utility)setItem(i.item_id,'utility',s.utility);}});
  DATA.essays.forEach(e=>['A','B'].forEach(bl=>{const k=e.essay_id+'_'+bl;const s=state['fb_'+k];if(s&&s.tone)setFb(k,'tone',s.tone);}));
  prog();
}
function collect(){
  const rows=DATA.items.map(i=>{const s=state['item_'+i.item_id]||{};return {
    item_id:i.item_id,essay_id:i.essay_id,proficiency:i.proficiency,arm:i.arm,blind:i.blind,
    dimension:i.dimension,quote:i.quote,issue:i.issue,
    verdict:s.verdict||'',dim_ok:s.dim_ok||'',utility:s.utility||''};});
  const fbs=[];DATA.essays.forEach(e=>['A','B'].forEach(bl=>{const k=e.essay_id+'_'+bl;const s=state['fb_'+k]||{};
    const arm=(DATA.items.find(i=>i.essay_id===e.essay_id&&i.blind===bl)||{}).arm||'';
    if(arm)fbs.push({essay_id:e.essay_id,blind:bl,arm:arm,tone:s.tone||'',missed:(s.missed||'').trim()});}));
  return {coder:state.coder||'',items:rows,feedbacks:fbs};
}
function dl(name,txt,type){const b=new Blob([txt],{type});const u=URL.createObjectURL(b);const a=document.createElement('a');a.href=u;a.download=name;a.click();}
function exportJSON(){dl('audit_'+(state.coder||'coder')+'.json',JSON.stringify(collect(),null,2),'application/json');}
function exportCSV(){const c=collect();let csv='item_id,essay_id,arm,blind,dimension,verdict,dim_ok,utility\\n';
  c.items.forEach(r=>csv+=[r.item_id,r.essay_id,r.arm,r.blind,r.dimension,r.verdict,r.dim_ok,r.utility].join(',')+'\\n');
  csv+='\\nessay_id,blind,arm,tone,missed_count\\n';
  c.feedbacks.forEach(f=>csv+=[f.essay_id,f.blind,f.arm,f.tone,f.missed?f.missed.split('\\n').filter(x=>x.trim()).length:0].join(',')+'\\n');
  dl('audit_'+(state.coder||'coder')+'.csv',csv,'text/csv');}
render();
</script></body></html>"""

html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
for path in [os.path.join(BASE, "audit", "反馈审计工具.html"),
             "反馈审计工具_卢晓琳.html"]:
    open(path, "w", encoding="utf-8").write(html)
    print("saved:", path)
print("items:", len(items), "| blind map:", blind)
