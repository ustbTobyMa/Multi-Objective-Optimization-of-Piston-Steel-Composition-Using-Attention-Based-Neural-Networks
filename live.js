/* V2.3 optional real DeepSeek mode. No provider key in the browser.
   Backend access passphrase is kept in memory only. Files require consent. */
(function(){
  'use strict';
  const original={knowledge,ask,page,newChat:actions.new,reset:actions.reset,demo:actions.demo,about:actions.about};
  const L={enabled:false,endpoint:'',token:'',model:'',turns:[],busy:false,controller:null,seq:0,scope:'builtin',consent:null,tested:false};
  const C=window.PistonKnowledge.state;
  try{L.endpoint=localStorage.getItem('piston-gateway-url')||'';}catch(_){}
  function url(value){const u=new URL(value);if(!(u.protocol==='https:'||(u.protocol==='http:'&&['localhost','127.0.0.1'].includes(u.hostname)))||u.username||u.password||u.search||u.hash||u.pathname!=='/')throw Error('请填写后端的HTTPS根地址，不要填写DeepSeek官方地址、密钥或带参数的链接。');if(u.hostname==='api.deepseek.com')throw Error('这里填写你部署的后端地址，不是DeepSeek官方API地址。');return u.origin;}
  function textHTML(s){return esc(s).split('\n').map(line=>/^#{1,4} /.test(line)?'<h3>'+line.replace(/^#{1,4} /,'')+'</h3>':line.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>').replace(/`([^`]+)`/g,'<code>$1</code>')).join('\n');}
  const selected=()=>S.files.find(f=>!f.image&&f.cid===C.scope);
  function evidence(){
    const file=selected();
    if(C.scope!=='builtin'&&!file)throw Error('所选资料已移除，请重新选择。');
    if(file){const prefix=file.text.slice(0,14000),lines=prefix.split(/\r?\n/);return {source:file.cid,filename:file.name,context:lines.map((line,i)=>'[文件:L'+(i+1)+'] '+line).join('\n').slice(0,18000),limited:prefix.length<file.text.length};}
    const box=document.createElement('div');
    const info=docs.map(d=>{box.innerHTML=docText(d);return '['+d[0]+'] '+d[1]+'\n'+box.textContent;}).join('\n\n');
    return {source:'builtin',filename:'内置材料与工艺样例库',context:('以下全部为人为构造的演示数据，非企业实测。\n'+JSON.stringify(data)+'\n'+info).slice(0,18000)};
  }
  function cancel(){const pending=L.turns.at(-1);if(L.busy&&pending&&!pending.done&&!pending.error)pending.error='请求已停止，现有文字可能不完整。';L.seq++;L.controller?.abort();L.controller=null;L.busy=false;S.busy=false;}
  function clearChat(){cancel();L.turns=[];L.consent=null;L.scope=C.scope;}
  const panel=()=>`<div class="live-bar"><div><strong>DeepSeek ${L.enabled?'· 真实问答已启用':'· 接入配置已就绪'}</strong><p>${L.enabled?'通过你的受保护后端调用；工艺优化仍为规则演示。':'当前仍为规则演示，配置后端并测试通过后才启用真实模型。'}</p></div><button class="btn ${L.enabled?'':'primary'}" data-live="settings">${L.enabled?'连接设置':'连接 DeepSeek'} ↗</button></div>`;
  function turnHTML(t,i){return `<section class="live-answer"><div class="question">${esc(t.q)}</div><div class="agent">✧ 活塞知识助手 <span class="badge">DeepSeek · ${esc(t.model||L.model)}</span></div><div class="live-text" id="liveText${i}">${textHTML(t.a||'')}</div><div id="liveStatus${i}" class="live-status ${t.error?'error':''}">${esc(t.error||(t.done?'模型已回复 · 工程结论仍需验证':'正在请求模型…'))}</div><details class="live-sources"><summary>本轮发送的资料：${esc(t.file)}${t.limited?'（仅节选）':''}</summary><p>仅发送所选资料片段、最近对话和当前问题。资料并非经核验的事实；模型引用需要回到原文核对。</p>${t.scope==='builtin'?docs.map(d=>`<button class="link" data-doc="${d[0]}">[${d[0]}] ${esc(d[1])}</button>`).join(' · '):'<p>正文中的[文件:L数字]对应原文行号。</p>'}</details></section>`;}
  function view(){return head('DEEPSEEK KNOWLEDGE COPILOT','让资料与模型，一起回答你的问题。','真实模型回复与连续追问；先回答，再解释依据、假设和验证建议。',btn('＋ 新建对话','new')+btn('▣ 全屏演示','fullscreen'))+panel()+`<div class="source-selector card"><div><b>本次回答依据</b><span>自有资料发送前需单独确认</span></div><select id="liveSource" aria-label="选择模型参考资料"><option value="builtin">内置材料与工艺样例库</option>${S.files.filter(f=>!f.image&&f.cid).map(f=>`<option value="${esc(f.cid)}" ${C.scope===f.cid?'selected':''}>${esc(f.name)}</option>`).join('')}</select><button class="btn" data-action="upload">＋ 添加资料</button></div><div class="card live-chat"><div class="panel-head"><h2>✧ 材料与工艺知识助手</h2><span class="grow"></span><span class="badge green">DeepSeek · 在线模式</span><button class="btn" data-live="export">导出对话</button></div><div id="chatBody" class="chat-body copilot-body" aria-live="polite">${L.turns.map(turnHTML).join('')||'<div class="live-welcome"><h2>现在可以自由提问，也可以连续追问。</h2><p>例如：为什么材料在高温下会出现疲劳寿命差异？请把一般机理与样例证据分开说明。</p><p>会将当前问题、选定资料和最近对话发送至你的后端，再由后端调用DeepSeek。不会上传图片或PDF，也不会自动发送其他文件。</p></div>'}</div><div class="compose"><div class="chips"><button data-follow="请分析350°C疲劳样例，并区分数据事实和可能原因">分析疲劳样例</button><button data-follow="为什么这样判断？哪些解释还需要试验证据？">解释判断依据</button><button data-follow="帮我安排一个有对照的验证方案">安排验证方案</button></div><form id="chatForm" class="composer"><textarea id="questionInput" rows="2" maxlength="2000" placeholder="自由提问，或继续追问…" aria-label="输入问题"></textarea><button id="send" class="btn primary" type="submit" ${L.busy?'disabled':''}>发送 ↑</button><button class="btn" type="button" data-live="stop" ${L.busy?'':'hidden'}>停止</button></form><div class="compose-note"><span>真实回复需消耗DeepSeek API额度 · 错误不会替换为假回复</span><span>资料由你选择发送</span></div></div></div>`;}
  knowledge=function(){return L.enabled?view():panel()+original.knowledge();};
  page=function(v,write=true){if(L.enabled&&v!=='knowledge'&&L.busy)cancel();original.page(v,write);if(L.enabled&&S.page==='knowledge'&&$('liveSource'))$('liveSource').value=C.scope;};
  function settings(){
    modal('连接 DeepSeek · 密钥仅保存在后端',`<div class="live-form"><div class="live-mode-notice">${L.enabled?'真实问答已启用。':'当前尚未连接真实模型。'} GitHub Pages 继续承载网页，模型密钥应在后端的环境变量中配置。</div><label for="gatewayURL">后端服务地址</label><input id="gatewayURL" type="url" value="${esc(L.endpoint)}" placeholder="https://你的服务.onrender.com" autocomplete="off"><label for="gatewayToken">演示访问口令（不是 DeepSeek API Key）</label><input id="gatewayToken" type="password" value="" autocomplete="off" placeholder="${L.token?'本次会话已填写；留空沿用':'后端 DEMO_ACCESS_TOKEN'}"><p>口令只在当前页面内存中保留，刷新即清空；不要在这里填写 sk- 开头的API密钥。</p><label class="live-consent"><input id="gatewayConsent" type="checkbox"><span>同意把提问、所选内置样例和最近对话发送至该后端及DeepSeek；添加的自有资料会另行确认。模型请求会消耗账户API额度。</span></label><div class="warning">首次需要部署后端。未授权部署账户、未设置后端密钥或测试失败时，不会假装已经接入。测试连接会发送一次简短的“OK”请求，不发送企业资料。</div><p id="gatewayStatus" class="live-config-status" role="status"></p></div>`, '<button class="btn" data-live="offline">使用规则演示</button><div class="row"><button class="btn" data-live="test">测试连接</button><button class="btn primary" data-live="enable" disabled>启用真实问答</button></div>','DEEPSEEK INTEGRATION V2.3');
    L.tested=false;
  }
  async function test(){
    const status=$('gatewayStatus'),button=document.querySelector('[data-live=test]');
    try{
      if(!$('gatewayConsent').checked)throw Error('请先确认数据发送与API调用说明。');
      const endpoint=url($('gatewayURL').value.trim()),token=$('gatewayToken').value.trim()||L.token;
      if(token.startsWith('sk-')){ $('gatewayToken').value='';throw Error('这是API密钥。请将它配置在后端，不要交给浏览器。');}
      if(token.length<24||token.length>256)throw Error('请输入至少24字符的演示访问口令。');
      button.disabled=true;status.textContent='正在通过后端测试DeepSeek真实回复…';
      const response=await fetch(endpoint+'/api/check',{method:'POST',credentials:'omit',redirect:'error',referrerPolicy:'no-referrer',headers:{'Content-Type':'application/json',Authorization:'Bearer '+token},body:'{}',signal:AbortSignal.timeout(100000)});
      const r=await response.json();if(!response.ok||!r.ok)throw Error(r.message||'后端未通过模型测试。');
      L.endpoint=endpoint;L.token=token;L.model=String(r.model||'DeepSeek');L.tested=true;
      try{localStorage.setItem('piston-gateway-url',endpoint);}catch(_){}
      status.textContent='模型已实际返回测试正文，可以启用真实问答。';document.querySelector('[data-live=enable]').disabled=false;$('gatewayToken').value='';
    }catch(e){L.tested=false;status.textContent=e.name==='TypeError'?'无法连接后端，请检查部署状态、HTTPS和允许来源配置。':e.message;document.querySelector('[data-live=enable]').disabled=true;}finally{button.disabled=false;}
  }
  async function request(q,preset){
    q=String(q||'').trim().slice(0,2000);if(!q||L.busy)return;
    if(preset)C.scope='builtin';
    if(L.scope!==C.scope){clearChat();L.scope=C.scope;}
    let ev;try{ev=evidence();}catch(e){toast(e.message);return;}
    if(ev.source!=='builtin'&&L.consent!==ev.source){
      modal('发送所选资料前，请确认',`<h3>${esc(ev.filename)}</h3><p>本次只发送这份文本/CSV的${ev.limited?'开头节选':'内容'}（最多18,000字符）、你的问题和相关对话，经过你配置的后端转发至DeepSeek。不会发送其他文件、图片或PDF。</p><div class="warning">涉及企业机密或个人信息时，先脱敏并确认有权使用外部模型。当前未发送任何资料。</div>`,'<button class="btn" data-action="close">暂不发送</button><button id="approveLiveFile" class="btn primary">同意发送并提问</button>','EXTERNAL DATA TRANSFER');
      $('approveLiveFile').onclick=()=>{L.consent=ev.source;close();request(q,preset);};return;
    }
    const history=L.turns.filter(t=>t.done&&!t.error&&t.scope===ev.source).slice(-3).flatMap(t=>[{role:'user',content:t.q.slice(0,4000)},{role:'assistant',content:t.a.slice(0,4000)}]);
    const turn={q,a:'',file:ev.filename,scope:ev.source,limited:ev.limited,model:L.model,done:false,error:''};L.turns.push(turn);if(L.turns.length>16)L.turns.shift();const i=L.turns.length-1;
    L.busy=true;S.busy=true;L.controller=new AbortController();const seq=++L.seq;page('knowledge');S.busy=true;
    const timer=setTimeout(()=>L.controller?.abort(),100000);
    const update=()=>{if(S.page!=='knowledge'||seq!==L.seq)return;const el=$('liveText'+i);if(el)el.innerHTML=textHTML(turn.a);};
    try{
      const response=await fetch(L.endpoint+'/api/chat',{method:'POST',credentials:'omit',redirect:'error',referrerPolicy:'no-referrer',headers:{'Content-Type':'application/json',Authorization:'Bearer '+L.token},signal:L.controller.signal,body:JSON.stringify({question:q,history,context:ev.context,source:ev.source,fileConsent:ev.source==='builtin'||L.consent===ev.source})});
      if(!response.ok){const r=await response.json().catch(()=>({}));throw Error(r.message||'请求失败，请检查后端。');}
      if(!response.body)throw Error('当前浏览器没有收到响应数据流。');
      const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='',gotDone=false;
      function line(raw){if(!raw.startsWith('data:'))return;const msg=JSON.parse(raw.slice(5));if(msg.type==='error')throw Error(msg.message);if(msg.type==='delta')turn.a+=String(msg.text||'');if(msg.type==='done'){gotDone=true;turn.done=true;if(msg.finishReason==='length')turn.a+='\n\n（达到本轮长度上限，可继续追问。）';}}
      while(true){const chunk=await reader.read();if(chunk.done)break;if(seq!==L.seq){await reader.cancel();return;}buffer+=decoder.decode(chunk.value,{stream:true});let pos;while((pos=buffer.indexOf('\n'))>=0){line(buffer.slice(0,pos).trim());buffer=buffer.slice(pos+1);}update();}
      buffer+=decoder.decode();if(buffer.trim())line(buffer.trim());if(!gotDone)throw Error('响应连接中断，现有文字可能不完整。');
    }catch(e){turn.error=e.name==='AbortError'?'已停止或请求超时，现有文字可能不完整。':e.message;L.controller?.abort();}
    finally{clearTimeout(timer);if(seq===L.seq){L.busy=false;S.busy=false;L.controller=null;if(S.page==='knowledge'){page('knowledge');$('liveText'+i)?.scrollIntoView({block:'nearest'});}}}
  }
  ask=function(q,preset){return L.enabled?request(q,preset):original.ask(q,preset);};
  actions.new=function(){clearChat();original.newChat();};actions.reset=function(){clearChat();original.reset();};
  actions.demo=function(){if(L.enabled){cancel();L.enabled=false;toast('自动演示改用规则样例，不调用付费模型。');}return original.demo();};
  const oldStop=stop;stop=function(notify=true){cancel();return oldStop(notify);};
  function disconnect(){cancel();L.enabled=false;L.token='';L.tested=false;clearChat();close();page('knowledge');toast('已退出真实问答，访问口令已从页面内存清除。');}
  function exportTurns(){download('DeepSeek_活塞知识对话.md','# DeepSeek真实问答记录\n\n'+L.turns.map(t=>'## 问题\n'+t.q+'\n\n来源：'+t.file+'\n\n'+t.a+(t.error?'\n\n注意：'+t.error:'')).join('\n\n---\n\n'));}
  document.addEventListener('click',e=>{const el=e.target.closest('[data-live]');if(!el||el.disabled)return;const a=el.dataset.live;if(a==='settings')settings();if(a==='test')test();if(a==='enable'&&L.tested){clearChat();L.enabled=true;close();page('knowledge');}if(a==='offline')disconnect();if(a==='stop'){const t=L.turns.at(-1);if(t&&!t.done)t.error='已手动停止，现有文字可能不完整。';cancel();page('knowledge');}if(a==='export')exportTurns();});
  document.addEventListener('change',e=>{if(e.target.id==='liveSource'){C.scope=e.target.value;clearChat();page('knowledge');}});
  document.addEventListener('input',e=>{if(['gatewayURL','gatewayToken','gatewayConsent'].includes(e.target.id)){L.tested=false;const b=document.querySelector('[data-live=enable]');if(b)b.disabled=true;}});
  window.addEventListener('beforeunload',()=>{cancel();L.token='';});
  actions.about=function(){modal('活塞智研 · V2.3 模型接入版',`<h3>两种问答模式</h3><p>规则演示模式使用样例与确定性计算。DeepSeek模式通过受保护的后端调用真实模型，只有测试成功并明确启用后才会使用。</p><p>当前状态：${L.enabled?'DeepSeek在线模式':'规则演示，真实模型未启用'}。</p><h3>数据与密钥</h3><p>API密钥仅放在后端环境变量。当前网页不包含密钥；演示访问口令仅在当前页面内存中保存。自有文本与CSV需确认后才发送到后端及DeepSeek。</p><h3>演示边界</h3><p>21条内置数据仍是人工构造，铸造和机加工仍使用演示规则。模型回复可能出错，不能直接用于生产；没有PDF解析或图像识别。双品牌标识为演示重绘/文字排版，不代表企业官方背书。</p>`);};
  const status=document.querySelector('.local-status small:last-child');if(status)status.textContent='在线模式仅发送已选择并确认的资料';
  const top=document.querySelector('.top-actions');if(top){const b=document.createElement('button');b.className='ib';b.dataset.live='settings';b.title='DeepSeek连接设置';b.setAttribute('aria-label','DeepSeek连接设置');b.textContent='✧';top.prepend(b);}
  page(S.page,false);
})();
