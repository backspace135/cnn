const $ = id => document.getElementById(id);
let last = null;              // 上次的 state，用于判断某些部分是否需要重绘
let lastFmapsVer = -1;        // 特征图只在版本号变化时才重绘（省性能）

// ---------------- 工具 ----------------
const fmt = v => (v === null || v === undefined) ? '—' : (+v).toFixed(4);
const fmtTime = s => { if(!s) return '—'; const m=Math.floor(s/60),ss=Math.floor(s%60); return m+':'+(ss<10?'0':'')+ss; };
const STATUS = {running:'训练中', paused:'已暂停', idle:'未开始', done:'已完成', cancelled:'已取消', error:'训练出错', '连接中':'连接中'};

// Hash navigation keeps views linkable and works with browser back/forward.
function showView(){
  const target=location.hash.slice(1) || 'intro';
  const view=['intro','model','gloss'].includes(target)?'learn':target==='theory'?'theory':target==='predict'?'predict':target==='lab'?'lab':'live';
  document.querySelectorAll('[data-page]').forEach(el=>el.hidden=el.dataset.page!==view);
  document.querySelectorAll('[data-view]').forEach(el=>{
    if(el.dataset.view===view) el.setAttribute('aria-current','page');
    else el.removeAttribute('aria-current');
  });
  const copy={live:['从像素到预测，读懂神经网络','启动一次训练，观察模型如何在错误中调整参数。'],
    lab:['自己拧一拧旋钮','每次只改变一个参数，比较曲线和准确率的变化。'],
    theory:['亲手计算卷积','移动卷积核，逐格算出特征图。'],
    predict:['你的数字，模型的判断','画一个数字，看看输入预处理和十类概率。'],
    learn:['你的第一堂神经网络课','从一个神经元出发，用手写数字实验理解机器如何学习。']};
  $('pageTitle').textContent=copy[view][0]; $('pageDescription').textContent=copy[view][1];
  $('trainingControls').hidden=view!=='live';
  if(target==='model'||target==='gloss') $(target).scrollIntoView();
  else window.scrollTo(0,0);
}
window.addEventListener('hashchange',showView);
showView();

let controlPending=false;
async function control(action, hyper){
  if(controlPending) return false;
  controlPending=true;
  document.querySelectorAll('.ctrl button, #lab button').forEach(b=>b.disabled=true);
  $('actionNotice').hidden=true;
  try{
    const response=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(hyper?{action,...hyper}:{action})});
    const result=await response.json();
    if(!response.ok || !result.ok) throw new Error(result.msg || '操作未成功，请稍后重试。');
    return true;
  }catch(error){
    $('actionNotice').textContent='操作失败：'+error.message;
    $('actionNotice').hidden=false;
    return false;
  }finally{
    controlPending=false;
    document.querySelectorAll('.ctrl button, #lab button').forEach(b=>b.disabled=false);
    if(last) render(last); // 恢复状态约束，避免请求完成后短暂重新启用非法操作
  }
}
function resetQuick(){  // 顶部"恢复默认并重训"
  const cfg = {epochs:6, lr:0.001, batch_size:64, dropout:0.3, device_mode:last?.config?.device_mode||'auto'};
  $('inpDevice').value=cfg.device_mode;
  $('inpEpochs').value=6; $('inpLr').value=0.001; $('inpBatch').value=64; $('inpDrop').value=0.3;
  control('reset', cfg);
}
async function runLab(){  // 调参实验室"应用并重新训练"
  for(const input of document.querySelectorAll('#lab input')){ if(!input.reportValidity()) return; }
  const cfg = {
    device_mode: $('inpDevice').value,
    epochs: Math.max(1,+$('inpEpochs').value||6),
    lr: +$('inpLr').value||0.001,
    batch_size: Math.max(16,+$('inpBatch').value||64),
    dropout: Math.min(0.7,Math.max(0,+$('inpDrop').value||0)),
  };
  $('labMsg').textContent='正在停止旧训练并应用参数…';
  const ok=await control('reset', cfg);
  $('labMsg').textContent=ok?'已应用新配置，重新训练中。':'应用失败，请重试。';
}

// ---------------- 绘制 ----------------
function poly(el, pts, stroke, w){
  const d = pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(2)+' '+p[1].toFixed(2)).join(' ');
  const path = document.createElementNS('http://www.w3.org/2000/svg','path');
  path.setAttribute('d',d); path.setAttribute('fill','none'); path.setAttribute('stroke',stroke);
  path.setAttribute('stroke-width',w); path.setAttribute('stroke-linejoin','round');
  path.setAttribute('stroke-linecap','round'); el.appendChild(path);
}
function svgNode(el,tag,attrs,text){
  const node=document.createElementNS('http://www.w3.org/2000/svg',tag);
  Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,v));
  if(text!==undefined) node.textContent=text;
  el.appendChild(node); return node;
}
function drawChart(svgId, x, series, yZero=true){
  const el=$(svgId); el.innerHTML='';
  const W=480,H=210,left=43,right=12,top=15,bottom=29;
  const all=series.flatMap(s=>s.data).filter(v=>Number.isFinite(v));
  const empty=!x.length || !all.length;
  let ymin=empty?0:(yZero?0:Math.max(0,Math.min(...all)-2));
  let ymax=empty?(yZero?1:100):Math.max(...all);
  if(ymax<=ymin) ymax=ymin+1;
  if(!empty) ymax+= (ymax-ymin)*.08;
  if(!yZero) ymax=Math.min(100,ymax);
  const X=i=>left+(W-left-right)*(x.length===1?.5:(x[i]-x[0])/(x[x.length-1]-x[0]||1));
  const Y=v=>H-bottom-(H-top-bottom)*(v-ymin)/(ymax-ymin||1);
  for(let g=0;g<4;g++){
    const value=ymin+(ymax-ymin)*g/3,y=Y(value);
    svgNode(el,'line',{x1:left,x2:W-right,y1:y,y2:y,stroke:'#e8ebf0','stroke-width':1,'stroke-dasharray':'3 4'});
    svgNode(el,'text',{x:left-9,y:y+3,fill:'#667182','font-size':10,'text-anchor':'end'}, yZero?value.toFixed(value<1?2:1):value.toFixed(0)+'%');
  }
  if(empty){
    svgNode(el,'text',{x:W/2,y:H/2,fill:'#667182','font-size':12,'text-anchor':'middle'},svgId==='accChart'?'完成第一轮训练后显示准确率':'开始训练后显示损失曲线');
    return;
  }
  [...new Set([0,Math.floor((x.length-1)/2),x.length-1])].forEach(i=>
    svgNode(el,'text',{x:X(i),y:H-7,fill:'#667182','font-size':10,'text-anchor':'middle'},x[i]));
  series.forEach(s=>{
    let segment=[];
    const flush=()=>{
      if(segment.length===1) svgNode(el,'circle',{cx:segment[0][0],cy:segment[0][1],r:3,fill:s.color});
      else if(segment.length) poly(el,segment,s.color,s.w||2);
      segment=[];
    };
    x.forEach((_,i)=>{ if(Number.isFinite(s.data[i])) segment.push([X(i),Y(s.data[i])]); else flush(); });
    flush();
  });
}
function drawBars(svgId, values){
  const el=$(svgId); el.innerHTML='';
  const W=480,H=180,bw=44;
  if(!values.length){
    svgNode(el,'text',{x:W/2,y:H/2,fill:'#667182','font-size':12,'text-anchor':'middle'},'完成第一轮训练后显示各类准确率');
    return;
  }
  values.forEach((v,i)=>{
    const x=22+i*bw,bh=Math.max(0,Math.min(100,v))*1.2;
    svgNode(el,'rect',{x,y:H-26-bh,width:26,height:bh,rx:3,fill:'#7390d7'});
    svgNode(el,'text',{x:x+13,y:H-26-bh-7,fill:'#667182','font-size':10,'text-anchor':'middle'},Math.round(v)+'%');
    svgNode(el,'text',{x:x+13,y:H-8,fill:'#667182','font-size':11,'text-anchor':'middle'},i);
  });
}
function pixToCanvas(canvas, px){  // px: 28x28 嵌套数组
  const ctx=canvas.getContext('2d'); canvas.width=28; canvas.height=28;
  const img=ctx.createImageData(28,28);
  for(let r=0;r<28;r++)for(let c=0;c<28;c++){
    const v=Math.min(255,Math.max(0,px[r][c]));
    const i=(r*28+c)*4; img.data[i]=v;img.data[i+1]=v;img.data[i+2]=v;img.data[i+3]=255;
  }
  ctx.putImageData(img,0,0);
}
function fmToCanvas(canvas, fm){  // fm: 14x14 嵌套数组
  const ctx=canvas.getContext('2d'); canvas.width=14; canvas.height=14;
  const img=ctx.createImageData(14,14);
  for(let r=0;r<14;r++)for(let c=0;c<14;c++){
    const v=Math.min(255,Math.max(0,fm[r][c]));
    const i=(r*14+c)*4; img.data[i]=v;img.data[i+1]=v;img.data[i+2]=v;img.data[i+3]=255;
  }
  ctx.putImageData(img,0,0);
}

function renderConfusion(matrix){
  const box=$('confusionMatrix');
  if(!Array.isArray(matrix)||matrix.length!==10){box.textContent='完成第一轮训练后显示。';return;}
  const table=document.createElement('table'); table.className='confusion-table';
  const caption=table.createCaption(); caption.textContent='行：真实数字；列：预测数字（验证集样本数）';
  const header=table.insertRow(); header.appendChild(document.createElement('th'));
  for(let col=0;col<10;col++){const th=document.createElement('th');th.scope='col';th.textContent=col;header.appendChild(th);}
  const max=Math.max(1,...matrix.flat());
  matrix.forEach((row,r)=>{const tr=table.insertRow();const th=document.createElement('th');th.scope='row';th.textContent=r;tr.appendChild(th);
    row.forEach((value,c)=>{const td=tr.insertCell();td.textContent=value;
      td.title=`真实 ${r}，预测 ${c}：${value} 张`;
      td.style.setProperty('--heat',String(.06+.36*value/max));
      if(r===c)td.className='diagonal';});});
  box.replaceChildren(table);
}

async function loadModelLayers(){
  try{
    const response=await fetch('/api/model-info');
    const result=await response.json();
    if(!response.ok) throw new Error(result.msg||'无法读取模型结构');
    const table=document.createElement('table');table.className='experiment-table';
    const head=table.insertRow();['层','输出形状（第一维为批量）','可学习参数'].forEach(name=>{
      const cell=document.createElement('th');cell.scope='col';cell.textContent=name;head.appendChild(cell);
    });
    result.layers.forEach(layer=>{
      const row=table.insertRow();[layer.layer,layer.shape.join(' × '),Number(layer.parameters).toLocaleString()].forEach(value=>{
        const cell=row.insertCell();cell.textContent=value;
      });
    });
    const summary=document.createElement('p');summary.className='teach';summary.textContent=result.summary;
    $('modelLayers').replaceChildren(table,summary);
  }catch(error){$('modelLayers').textContent=error.message;}
}

async function loadExperiments(){
  try{
    const response=await fetch('/api/experiments');
    if(!response.ok) return;
    const entries=(await response.json()).experiments||[];
    const box=$('experimentHistory');
    if(!entries.length){box.textContent='暂无完整运行。';return;}
    const table=document.createElement('table');table.className='experiment-table';
    const heading=table.insertRow();
    ['运行','轮数','学习率','批大小','Dropout','验证准确率','验证 loss'].forEach(label=>{
      const cell=document.createElement('th');cell.scope='col';cell.textContent=label;heading.appendChild(cell);
    });
    entries.slice().reverse().forEach(entry=>{
      const row=table.insertRow();const cfg=entry.config||{};
      [entry.id,cfg.epochs,cfg.lr,cfg.batch_size,cfg.dropout,
        entry.validation_acc?.toFixed(2)+'%',entry.validation_loss?.toFixed(4)].forEach(value=>{
          const cell=row.insertCell();cell.textContent=value??'—';
        });
    });
    box.replaceChildren(table);
  }catch(_){ /* 实验历史是辅助内容，不覆盖训练请求错误。 */ }
}

// ---------------- 主渲染 ----------------
function render(s){
  const st=s.status||'idle';
  const hardware=s.hardware||{};
  const device=s.config?.device;
  $('deviceLabel').textContent=device==='cuda'?'CUDA · '+(hardware.gpu_name||'NVIDIA GPU'):device==='cpu'?'CPU':'检测中…';
  if(s.dataset_sizes){
    $('trainCount').textContent=Number(s.dataset_sizes.train).toLocaleString();
    $('valCount').textContent=Number(s.dataset_sizes.validation).toLocaleString();
    $('testCount').textContent=Number(s.dataset_sizes.test).toLocaleString();
  }
  $('cudaOption').disabled=!hardware.cuda_available;
  $('deviceHint').textContent=hardware.cuda_available
    ? '可用显卡：'+hardware.gpu_name+'。选择模式后点击下方按钮，模型将从头训练。'
    : (hardware.cuda_reason||'正在检测可用训练设备…');
  if(!last && s.config){
    $('inpDevice').value=s.config.device_mode||'auto';
    $('inpEpochs').value=s.config.epochs; $('inpLr').value=s.config.lr;
    $('inpBatch').value=s.config.batch_size; $('inpDrop').value=s.config.dropout;
  }
  if(st==='error'){
    $('actionNotice').textContent='训练出错：'+(s.error||'请查看训练日志。');
    $('actionNotice').hidden=false;
  }

  $('dot').className='dot '+(st==='running'?'running':st==='paused'?'paused':st==='done'?'done':'idle');
  $('statusTxt').textContent=STATUS[st]||st;
  $('btnStart').textContent=s.model_loaded?'已加载模型（重置再训练）':st==='paused'?'继续训练':st==='running'?'训练进行中':st==='done'?'训练已完成':'开始训练';
  if(!controlPending){ $('btnStart').disabled=s.model_loaded||st==='running'||st==='done'||st==='cancelled'||st==='error'; $('btnPause').disabled=st!=='running'; $('btnStop').disabled=!['running','paused'].includes(st); }

  // KPI + 进度
  $('kLoss').textContent=fmt(s.train_loss_cur);
  $('kEpoch').textContent=s.epoch_current+' / '+s.epoch_total;
  $('kStep').textContent=s.step;
  const acc=s.epochs_val_acc; $('kAcc').textContent=acc.length?acc[acc.length-1].toFixed(2)+'%':'—';
  $('kTime').textContent=fmtTime(s.elapsed_sec)+(s.eta_sec?' / '+fmtTime(s.eta_sec):'');
  const pct=s.epoch_total&&s.steps_per_epoch?100*Math.min(s.step/(s.epoch_total*s.steps_per_epoch),1):0;
  $('pbar').style.width=pct+'%';
  $('pEpoch').textContent=(s.epoch_current||0)+' / '+(s.epoch_total||0);
  $('pPct').textContent=pct.toFixed(0)+'%';

  // 正在发生什么的文字
  $('teachProgress').innerHTML = st==='running'
    ? `正在 <b>Epoch ${s.epoch_current||0}/${s.epoch_total||0}</b>，已走 <b>${s.step||0}</b> 步（每轮约 ${s.steps_per_epoch||'?'} 步）。
       正在把训练集第 ${s.epoch_current||0} 遍完整看一遍。`
    : st==='done'?'训练完成。看看最终准确率，或去「调参实验室」换个参数再玩。':STATUS[st];
  if(s.train_loss_cur!==null)
    $('teachLoss').innerHTML=`当前 loss=<b>${(+s.train_loss_cur).toFixed(4)}</b>。
        看它是不是整体<b>往下走</b>？往下走=模型在进步。数字越小，说明模型越"确定"自己的判断。`;
  else $('teachLoss').textContent='损失衡量预测与正确答案的差距，整体下降通常意味着模型正在进步。';

  // 曲线（loss 降采样）
  const N=600, stepX=Math.max(1,Math.floor(s.loss_history.length/Math.max(N,1)));
  const x=[],raw=[],smooth=[];
  for(let k=0;k<s.loss_history.length;k+=stepX){
    x.push(s.loss_steps[k]||k); raw.push(s.loss_history[k]);
    smooth.push(s.smooth_loss&&s.smooth_loss[k]!=null?s.smooth_loss[k]:null);
  }
  drawChart('lossChart', x.slice(-N), [
    {data:raw.slice(-N),color:'#bdc7d8',w:1.5},{data:smooth.slice(-N),color:'#355ed3',w:2.4}]);
  drawChart('accChart', acc.map((_,i)=>i+1), [
    {data:s.epochs_train_acc||[],color:'#8a97ad',w:2},{data:acc,color:'#355ed3',w:2.4}],false);
  drawBars('accBars',s.class_acc);
  if(!last || s.confusion!==last.confusion) renderConfusion(s.confusion);
  $('testResult').textContent=s.test_acc!=null
    ? `完整训练后独立测试：准确率 ${s.test_acc.toFixed(2)}%，loss ${s.test_loss.toFixed(4)}。请勿反复根据测试集分数选参数。`
    : '测试集不参与每轮观察或调参，只有完整训练结束后评估一次。';
  if(!s.fmaps?.length){
    $('fmaps').innerHTML='<p class="empty">完成第一轮训练后显示 32 张特征图</p>';
    lastFmapsVer=-1;
  }
  if(!s.samples?.length){
    $('smax').innerHTML='<p class="empty">完成第一轮训练后显示预测概率</p>';
    $('samples').innerHTML='<p class="empty">完成第一轮训练后显示验证样例</p>';
    $('smaxTitle').textContent=''; $('sampleNote').textContent='';
  }

  // 特征图（每轮才变，用版本号去抖动，避免每 0.7s 全量重绘）
  if(s.fmaps && s.fmaps.length && s.fmaps_version !== lastFmapsVer){
    const box=$('fmaps'); box.innerHTML='';
    (s.fmaps||[]).forEach((fm,k)=>{
      const d=document.createElement('div'); d.className='fcell';
      const cv=document.createElement('canvas'); cv.width=14; cv.height=14; fmToCanvas(cv,fm);
      const small=document.createElement('small'); small.textContent='核 '+(k+1);
      d.appendChild(cv); d.appendChild(small); box.appendChild(d);
    });
    lastFmapsVer = s.fmaps_version;
  }

  // Softmax 概率条（用第一张样例）
  if(s.samples && s.samples.length && s.samples[0].pred!==null){
    const sm=s.samples[0];
    $('smaxTitle').textContent = `· 这张图的真实数字是 ${sm.true}，模型 ${(sm.pred===sm.true)?'判断正确':'判断错了'}：猜成 ${sm.pred}（${(sm.prob*100).toFixed(1)}%）`;
    const box=$('smax'); box.innerHTML='';
    (Array.isArray(sm.probs)?sm.probs:[0,0,0,0,0,0,0,0,0,0]).forEach((p,k)=>{
      const row=document.createElement('div'); row.className='srow'+(k===sm.true?' true':'');
      const t=document.createElement('span'); t.className='tagk'; t.textContent=k;
      const bb=document.createElement('div'); bb.className='barbox';
      const b=document.createElement('div'); b.className='bar'; b.style.width=(p*100)+'%'; bb.appendChild(b);
      const pc=document.createElement('span'); pc.className='pct'; pc.textContent=(p*100).toFixed(1)+'%';
      row.appendChild(t); row.appendChild(bb); row.appendChild(pc); box.appendChild(row);
    });
  }

  // 预测样例网格
  if(s.samples && s.samples.length && s.samples[0].pred!==null){
    $('sampleNote').textContent='· 共 '+s.samples.length+' 张验证样例';
    const box=$('samples'); box.innerHTML='';
    s.samples.forEach(sm=>{
      const d=document.createElement('div'); d.className='sample '+((sm.pred===sm.true)?'correct':'wrong');
      const cv=document.createElement('canvas'); cv.width=28; cv.height=28; pixToCanvas(cv, sm.pixels);
      const lb=document.createElement('div'); lb.className='lab';
      const cls = (sm.pred===sm.true)?'c':'w';
      lb.innerHTML='实 <b>'+sm.true+'</b> · 预 <b class="'+cls+'">'+((sm.pred===sm.true)?sm.true:sm.pred)+'</b> '+
          '<span style="color:var(--muted)">'+(sm.prob*100).toFixed(0)+'%</span>'+
          '<span class="result-tag">'+((sm.pred===sm.true)?'识别正确':'识别错误')+'</span>';
      d.appendChild(cv); d.appendChild(lb); box.appendChild(d);
    });
  }

  // 日志（只在新内容时整段重绘）
  if(s.log){
    const key=s.log.join('|');
    if(key!==lastLogKey){ lastLogKey=key; $('log').innerHTML=(s.log||[]).map(l=>l.replace(/&/g,'&amp;').replace(/</g,'&lt;')).join('<br>'); }
    const lg=$('log'); lg.scrollTop=lg.scrollHeight;
  }
  // Fill the learning examples when samples first arrive, even after an idle load.
  if(s.samples?.length){
    document.querySelectorAll('#digits .cell').forEach((cell,i)=>{
      const sample=s.samples[i]; if(!sample) return;
      pixToCanvas(cell.querySelector('canvas'),sample.pixels);
      cell.lastElementChild.textContent=sample.true;
    });
  }
  if(st==='done' && (!last || last.status!=='done')) loadExperiments();
  last=s;
}
let lastLogKey='';

drawChart('lossChart',[],[]);
drawChart('accChart',[],[],false);
drawBars('accBars',[]);

// ---------------- 轮询：前一个请求完成后才发下一个 ----------------
async function pollState(){
  try{
    const response=await fetch('/api/state');
    const state=await response.json();
    if(!response.ok) throw new Error(state.msg||'无法读取训练数据。');
    render(state);
  }catch(error){
    $('statusTxt').textContent='连接中断，正在重试…'; $('dot').className='dot idle';
    $('actionNotice').textContent=error.message; $('actionNotice').hidden=false;
  }finally{setTimeout(pollState,700);}
}

// 初加载：样例由 state.samples 提供，不必等训练完成。
document.addEventListener('DOMContentLoaded',()=>{
  const box=$('digits');
  for(let n=0;n<10;n++){
    const cell=document.createElement('div');cell.className='cell';
    cell.innerHTML='<canvas width="28" height="28" role="img" aria-label="手写数字样例"></canvas><div></div>';
    box.appendChild(cell);
  }
  pollState();
  loadExperiments();
  loadModelLayers();
});
