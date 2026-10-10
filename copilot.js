/* V2.2.1: empty-first local chat, cancellable progress and progressive answers.
   This is a deterministic explanation layer, NOT an LLM. Uploaded documents are
   untrusted evidence: always escaped, never executed, never sent to a server. */
(function () {
  'use strict';
  const C = { turns: [], scope: 'builtin', subject: '', root: '', sequence: 0, metric: {}, pending: false, job: null, draft: '' };
  // A conversation starts with a user submission, never a seeded question.
  S.answered=false; S.question=''; S.intent='';
  const original = { page, docsView, addFiles, about: actions.about, settings: actions.settings, newChat: actions.new, reset: actions.reset };
  const titles = {fatigue:'350°C 高周疲劳',lcf:'低周疲劳',tmf:'热机疲劳',thermal:'热扩散与导热',modulus:'弹性模量',wear:'摩擦试验',dsc:'差热分析',casting:'铝合金铸造',machining:'钢活塞机加工'};
  const sourceNames = Object.fromEntries(docs.map(d => [d[0], d[1]]));
  const E = esc;
  const files = () => S.files.filter(f => !f.image && typeof f.text === 'string').map(f => { if(!f.cid) f.cid = 'F' + (++C.sequence); return f; });
  const chosen = () => files().find(f => f.cid === C.scope);
  const scopeLabel = () => chosen()?.name || '内置材料与工艺样例库';
  const cite = id => `<button class="link" data-doc="${id}">[${id}] 查看依据 ↗</button>`;
  const readRecord = id => `<button class="link" data-record="${id}">[${id}] ↗</button>`;
  const focusOf = q => /一句话|简短|简洁|汇报|总结一下/.test(q)?'brief':/为什么|原因|解释|怎么得|怎么算/.test(q)?'why':/验证|下一步|怎么做|怎么办|试验计划|实验计划|建议/.test(q)?'next':/依据|证据|原文|哪条|哪一条|来源/.test(q)?'evidence':'answer';
  function topicOf(q) {
    if(/热机|热机械|TMF|温度循环/i.test(q)) return 'tmf';
    if(/低周|LCF/i.test(q)) return 'lcf';
    if(/铸造|浇注|模温|气孔/.test(q)) return 'casting';
    if(/机加工|切削|粗糙|刀具|进给|车削|去除率/.test(q)) return 'machining';
    if(/导热|热扩散|比热/.test(q)) return 'thermal';
    if(/弹性模量|杨氏模量/.test(q)) return 'modulus';
    if(/摩擦|磨损/.test(q)) return 'wear';
    if(/差热|吸热峰|DSC/i.test(q)) return 'dsc';
    if(/疲劳|寿命|调质|热处理/.test(q)) return 'fatigue';
    return '';
  }
  function isFollow(q) { return /为什么|原因|解释|验证|下一步|那|这个|这个结论|上面|刚才|它|依据|证据|一句话|总结|更好|怎么算|计算|差多少|建议/.test(q); }
  function getTopic(q, preset) { return titles[preset] ? preset : topicOf(q) || (isFollow(q)?C.subject:''); }
  function sections(conclusion, explanations, evidence, next, refs, limit, extra={}) { return {conclusion, explanations, evidence, next, refs, limit, ...extra}; }
  function summarizeBuilt(topic, focus, q) {
    const deviceMap={fatigue:'HCF',lcf:'LCF',tmf:'TMF',thermal:'DIF',modulus:'MOD',wear:'WEAR',dsc:'DSC'};
    if(deviceMap[topic]) {
      const rr = data.filter(r => r.device === deviceMap[topic]), a=rr[0], b=rr[rr.length-1], change=b.value-a.value, relative=change/a.value*100;
      const context = q + ' ' + C.root;
      if(/42CrMo/i.test(context) && topic==='thermal' || /铝合金|AlSi/i.test(context) && ['fatigue','lcf','tmf','modulus','wear'].includes(topic)) return missing('材料牌号不匹配', '当前这一场景没有你指定材料的对应记录，不能把其他材料的数值当作答案。请添加对应材料台账，或选择已有的 42CrMo 疲劳样例。');
      const temps=[...context.matchAll(/(\d+)\s*(?:°\s*C|℃|摄氏度|度)/gi)].map(m=>+m[1]);
      if(['fatigue','lcf'].includes(topic) && temps.some(t=>t!==350)) return missing('指定温度缺少记录', '这里仅有 350°C 的疲劳样例，没有你指定温度的对应记录。不能把 350°C 的寿命直接外推；需要相应温度、载荷条件下的数据。');
      const isFatigue=['fatigue','lcf','tmf'].includes(topic);
      const first = isFatigue ? `在这组样例里，${E(b.state)}的记录寿命最高，为 ${fmt(b.value,0)} 次；相对${E(a.state)}的 ${fmt(a.value,0)} 次，高了 ${fmt(relative)}%。它值得优先复验，但现在还不能说“这个热处理一定更好”。` : ({
        thermal:`样例的热扩散系数从 ${a.temp} 的 ${fmt(a.value)} 增至 ${b.temp} 的 ${fmt(b.value)} mm²/s，变化为 +${fmt(relative)}%。但这些数值还不能直接回答“导热系数是多少”，因为缺少同温度下的密度和比热。`,
        modulus:`在这三条样例中，弹性模量由 ${a.temp} 的 ${a.value} GPa 变为 ${b.temp} 的 ${b.value} GPa，下降 ${fmt(-relative)}%。这是样例内的温度对比，不能据此直接推算零件变形或其他温度的模量。`,
        wear:`表面状态 C 的样例平均摩擦系数最低，为 ${b.value}；相对状态 A 的 ${a.value} 低 ${fmt(-relative)}%。但“摩擦系数更小”不等于“实际磨损量更少”，还需要磨损测量。`,
        dsc:`三条样例的吸热峰位置分别为 ${rr.map(r=>r.value+'°C').join('、')}。峰位发生差异是待分析的现象，单凭峰温还不能确认具体反应、相变或最优热处理。`
      })[topic];
      const condition=topic==='tmf'?'温度循环、相位和机械应变幅':topic==='fatigue'?'温度、应力幅、应力比和频率':topic==='lcf'?'温度、总应变幅和频率':'材料状态、测试温度与测试方法';
      const descriptions = isFatigue ? [
        ['这组数值说明什么',`记录中的${condition}一致，观察到的寿命按 A、B、C 递增。因此可以把 C 作为后续对比对象，而不是把三条记录全部丢给工程师自己判断。`],
        ['为什么还不能判定原因',`热处理状态和材料批次同时不同，而且每种状态只有一条记录。热处理、批次波动与试验离散性混在一起，现有台账无法区分谁导致了差异。热处理影响是待验证解释，不是已经证实的因果结论。`],
        ['差异怎样算出来',`(${fmt(b.value,0)} − ${fmt(a.value,0)}) ÷ ${fmt(a.value,0)} × 100% = ${fmt(relative)}%。这个百分比只描述两条样例的差值，不是模型预测的生产收益。`]
      ] : ({thermal:[['先区分两个指标','热扩散系数 α 和导热系数 λ 是不同指标。λ = α × ρ × cₚ；使用 SI 单位时，α 的 mm²/s 需要乘以 10⁻⁶ 转成 m²/s。'],['为什么暂时不给出导热系数','当前三条台账没有匹配温度的密度 ρ 和比热 cₚ。随意补常数会把假设伪装成企业数据，所以这里只对热扩散样例作比较。']],modulus:[['对比范围','这三条记录对应同一示例材料状态、不同温度；仅描述样例的变化。'],['不能直接推到零件','零件变形还取决于几何、载荷、约束和温度场。弹性模量表本身不足以判断某个活塞的变形是否合格。']],wear:[['现有资料测了什么','台账给出平均摩擦系数，并没有提供磨损体积、磨损率或表面形貌。'],['结论应该怎样说','可以比较这几条摩擦系数记录，不能将它改写为耐磨寿命提升。']],dsc:[['现有证据','记录给出峰位，但没有完整热分析曲线、峰面积和配套相分析。'],['解释边界','不能从单个峰温直接识别材料反应。下一步先调取原始曲线与测试条件，再与其他表征交叉验证。']]})[topic];
      const next = isFatigue?`把 A 与 C 作为对照，在同批次材料、同一${condition}下补充重复试验；同时补齐热处理曲线、组织与断口资料。重复数和失效判据由试验负责人确认。` : ({thermal:'补齐同一材料状态、同温度的密度和比热，统一单位后计算导热系数，并记录测量不确定度。',modulus:'补充重复测量与测试方法说明；实际零件分析应使用经确认的材料曲线，并结合其载荷与边界条件。',wear:'在相同载荷、速度、对偶件与润滑条件下，补测磨损量及表面形貌，再讨论耐磨表现。',dsc:'获取完整原始曲线、基线处理和重复测量记录，再结合其他表征判断峰的来源。'})[topic];
      const refs=topic==='fatigue'?['D01','D03']:topic==='tmf'?['D02','D03']:['D03','D06'];
      return sections(first,descriptions,docRows(rr),next,refs,'以上数值来自人工构造样例；没有重复样本，不构成显著性检验、因果证明或真实寿命预测。',{count:rr.length,records:rr.map(r=>r.id)});
    }
    if(topic==='casting'||topic==='machining') {
      const keep={page:S.page,result:S.result,stale:S.stale};let r;
      try { S.page=topic;r=calculate(); } finally { S.page=keep.page;S.result=keep.result;S.stale=keep.stale; }
      if(!r.best) return missing('当前约束下没有候选方案','演示候选集没有同时满足约束的组合，不能强行给出“最优参数”。先检查 Ra 上限与刀具寿命约束，或补充候选数据。');
      const a=r.base,b=r.best.m,p=r.best.p,c=topic==='casting';
      const conclusion=c?`不建议仅凭“气孔偏多”就认定是浇注温度的问题。先核查缺陷和工艺记录，再比较候选参数。按当前演示规则，可把 ${p.pour}°C 浇注、${p.mold}°C 模温、${p.cooling}作为待验证组合。`:`这里不是所有指标同时提高，而是先满足表面质量约束，再权衡去除率和刀具寿命。按“${goalNames[r.goal]}”目标，当前候选为 ${p.speed} m/min、${fmt(p.feed,2)} mm/r、切深 ${fmt(p.depth)} mm。`;
      const explanation=c?[['为什么这样回答','气孔问题需要关联缺陷分布、材料批次、熔体处理、排气和冷却记录。当前演示没有实际缺陷诊断证据，不能把某个参数当作已确认的根因。'],['规则计算了什么',`候选组合在演示公式中对应良品率 ${fmt(b.quality)}%，比当前输入的 ${fmt(a.quality)}% 高 ${fmt(b.quality-a.quality)} 个百分点。该数值来自人工设置的温度偏差与冷却惩罚项，不是实测收益。`],['为什么还要做对照','先保留当前参数作为基线，再检测建议组合。否则即使批次指标变化，也难以判断是否来自这次调整。']]:[['为什么不只追求速度','这版先过滤不满足 Ra 上限或模拟刀具寿命底线的组合，再按你选择的目标排序。更换目标，会改变推荐结果。'],['参数怎样对应指标',`候选的 Ra 为 ${fmt(b.ra,2)} μm，模拟寿命为 ${fmt(b.life)} min。切削段去除率按 Q = v × f × a 计算：${p.speed} × ${fmt(p.feed,2)} × ${fmt(p.depth)} = ${fmt(b.mrr)} cm³/min。`],['这个解释有哪些限制','Ra 和寿命是演示函数输出，不是模型训练结果；去除率不包含装夹、换刀等时间，也不能代表产线效率。']];
      const evidence=table(['比较项','当前输入','候选方案'],c?[['良品率（规则模拟）',fmt(a.quality)+'%',fmt(b.quality)+'%'],['气孔缺陷率（规则模拟）',fmt(a.defect)+'%',fmt(b.defect)+'%']]:[['Ra（模拟）',fmt(a.ra,2)+' μm',fmt(b.ra,2)+' μm'],['材料去除率（几何估算）',fmt(a.mrr)+' cm³/min',fmt(b.mrr)+' cm³/min'],['刀具寿命（模拟）',fmt(a.life)+' min',fmt(b.life)+' min']]);
      return sections(conclusion,explanation,evidence,'安排基线组、建议组和单参数邻近组，统一材料批次、检测口径与失效判据；真实试验通过后再讨论工艺导入。',[c?'D04':'D05','D06'],'上述方案与指标由演示规则生成，不是实际生产建议。',{count:r.all.length,process:topic});
    }
    return missing('先确认你想解决的问题','可以围绕材料疲劳、热机疲劳、导热、铸造气孔或机加工提出问题。也可以切换到你添加的资料，让我提取其中的结论和待核实项。当前尚未连接通用大模型，超出支持范围时不会编造回答。');
  }
  function missing(title,body){return sections(body,[[title,'需要先补齐对应资料，再给出有依据的分析。']],'<p>没有符合当前问题的证据记录。</p>','请补充材料牌号、测试条件和关注的性能指标，或添加对应文本 / CSV 台账。',[],'未找到对应证据，不把其他场景的数据硬套过来。',{missing:true,count:0});}
  function parseCSV(text) {
    const raw=text.replace(/^\uFEFF/,'').slice(0,1000000), line=raw.split(/\r?\n/,1)[0], delimiter=line.includes('\t')?'\t':line.includes(',')?',':';';
    let rr=[],row=[],cell='',quote=false;
    for(let i=0;i<raw.length;i++){const ch=raw[i];if(ch==='"'){if(quote&&raw[i+1]==='"'){cell+='"';i++;}else quote=!quote;}else if(ch===delimiter&&!quote){row.push(cell.trim());cell='';}else if((ch==='\n'||ch==='\r')&&!quote){if(ch==='\r'&&raw[i+1]==='\n')i++;row.push(cell.trim());if(row.some(Boolean))rr.push(row);row=[];cell='';if(rr.length>5000)break;}else cell+=ch;}
    if(cell||row.length){row.push(cell.trim());if(row.some(Boolean))rr.push(row);}
    if(quote||rr.length<2||rr[0].length<2||rr[0].length>40)return null;
    const headers=rr.shift(), bad=rr.filter(r=>r.length!==headers.length).length;rr=rr.filter(r=>r.length===headers.length);
    return {headers,rows:rr.slice(0,5000),bad,limited:text.length>1000000||rr.length>=5000};
  }
  function number(v) { const s=String(v).replace(/,/g,'').trim();return /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?%?$/i.test(s)?Number(s.replace('%','')):null; }
  function numericColumns(csv){return csv.headers.map((h,j)=>({h,j,nums:csv.rows.map(r=>number(r[j])).filter(n=>n!==null&&Number.isFinite(n))})).filter(c=>c.nums.length>=Math.max(1,csv.rows.length*.65)&&!/(?:编号|批次|序号|^id$)/i.test(c.h));}
  function csvAnswer(f,q,focus){
    const d=f.csv||(f.csv=parseCSV(f.text));if(!d)return null;
    const columns=numericColumns(d);if(!columns.length)return null;
    const normalized=s=>s.toLowerCase().replace(/[^\p{L}\p{N}]/gu,'');
    const qn=normalized(q), requested=columns.find(c=>qn.includes(normalized(c.h))&&normalized(c.h).length>1);
    let col=requested||columns.find(c=>String(c.j)===String(C.metric[f.cid]));
    if(!col)col=columns.find(c=>/寿命|强度|粗糙|良品|缺陷|导热|扩散|模量|数值|磨损|硬度/.test(c.h))||columns[0];
    C.metric[f.cid]=col.j;
    const compareQ=C.root||q;let selected=d.rows.map((r,i)=>({r,i})),filters=[];
    for(let j=0;j<d.headers.length;j++){
      if(/材料|牌号/.test(d.headers[j])){const options=[...new Set(d.rows.map(r=>r[j]))].filter(s=>s.length>1&&compareQ.toLowerCase().includes(s.toLowerCase()));if(options.length){selected=selected.filter(x=>options.includes(x.r[j]));filters.push(d.headers[j]+': '+options.join(' / '));}}
      if(/温度/.test(d.headers[j])){const t=compareQ.match(/(\d+)\s*(?:°\s*C|℃|摄氏度)/i);if(t){selected=selected.filter(x=>Number.parseFloat(x.r[j])===+t[1]);filters.push(d.headers[j]+': '+t[1]);}}
    }
    const valid=selected.map(x=>({...x,n:number(x.r[col.j])})).filter(x=>x.n!==null&&Number.isFinite(x.n));
    if(!valid.length)return missing('筛选后没有可计算的记录',`在《${E(f.name)}》中没有找到满足 ${E(filters.join('；')||'当前筛选')} 的有效数值。请确认字段与单位，不能拿其他温度或材料代替。`);
    const sorted=[...valid].sort((x,y)=>x.n-y.n),lo=sorted[0],hi=sorted[sorted.length-1],avg=valid.reduce((s,x)=>s+x.n,0)/valid.length;
    const labelIndex=d.headers.findIndex(h=>/批次|编号|样品|样本|方案|试样/.test(h)),label=x=>labelIndex<0?'第 '+(x.i+1)+' 条记录':x.r[labelIndex];
    const rel=lo.n>0?`，相对最小值的差异为 ${fmt((hi.n-lo.n)/lo.n*100)}%`:'';
    const unit=/%|百分/.test(col.h)?' 个百分点':'（原表同列单位）';
    const conclusion=`我对《${E(f.name)}》中“${E(col.h)}”的 ${valid.length} 条有效记录做了比较${filters.length?'，范围为 '+E(filters.join('；')):''}。最大值为 ${fmt(hi.n,2)}（${E(label(hi))}），最小值为 ${fmt(lo.n,2)}（${E(label(lo))}），差 ${fmt(hi.n-lo.n,2)}${unit}${rel}。这是数值对比，不是“越大越好”的自动判定。`;
    const explanations=[['为什么得到这个结论',`从所选列读取数值后排序；差值 = ${fmt(hi.n,2)} − ${fmt(lo.n,2)} = ${fmt(hi.n-lo.n,2)}。${lo.n>0?'相对差值以最小值为分母。':'最小值非正数，不计算相对百分比。'}未填或无法识别的值不当作零。`],['为什么不能直接认定原因','不同材料、温度、载荷或批次可能让这些数值不具可比性。当前程序只做所列筛选，不会自动消除混杂因素，也不会把差异解释成已经证实的热处理效果。'],['还可以怎么读这张表',`当前列的算术平均值为 ${fmt(avg,2)}；这只是 ${valid.length} 条有效记录的描述，不是模型预测。可在下方改选指标，再追问它的差异与验证方法。`]];
    const pick=`<label class="metric-selector">分析指标 <select id="copilotMetric">${columns.map(x=>`<option value="${x.j}" ${x.j===col.j?'selected':''}>${E(x.h)}</option>`).join('')}</select></label>`;
    const selectedRows=[...new Set([hi,lo,...valid])].slice(0,8);
    const evidence=table(['原表数据记录序号',...d.headers.map(E)],selectedRows.map(x=>[String(x.i+1),...x.r.map(E)]))+'<p>序号不含表头；只显示前 8 条相关记录，完整内容可打开源文件。</p>';
    return sections(conclusion,explanations,evidence,'先确认指标方向和单位，再按同材料、同测试条件分组对比；检查重复样本、原始曲线和异常记录，之后再判断是否值得复验。',[],`${valid.length} 条参与计算；${selected.length-valid.length} 条该列非数值；${d.bad} 条格式不齐已跳过。${d.limited?'为保证浏览器响应，最多读取前 1,000,000 字符 / 5,000 条记录。':''}来源为本次添加的资料，未核验真实性。`,{file:f,metricUI:pick,count:valid.length});
  }
  function terms(q){
    let words=[];if(typeof Intl.Segmenter==='function'){const sg=new Intl.Segmenter('zh',{granularity:'word'});words=[...sg.segment(q.toLowerCase())].filter(x=>x.isWordLike).map(x=>x.segment);}
    else words=q.toLowerCase().match(/[a-z0-9]{2,}|[\u4e00-\u9fff]{2}/g)||[];
    return [...new Set(words.filter(w=>w.length>1&&!/^(请问|请你|帮我|为什么|什么|这个|那个|资料|说明|解释|分析|一下|内容|总结|怎么|如何|根据|上传)$/.test(w)))];
  }
  function textAnswer(f,q,focus){
    const raw=f.text.slice(0,80000), lines=raw.split(/\r?\n/).map((text,i)=>({text:text.trim(),line:i+1})).filter(x=>x.text.length>5&&!/^[-=#*\s]+$/.test(x.text));
    if(!lines.length)return missing('资料暂时没有足够正文','当前文本没有可用于解释的正文内容；请添加试验描述、结果和条件，或使用 CSV 台账。');
    const generic=/概括|总结|主要|讲了|这份|这篇|这段|这张/.test(q)||isFollow(q), query=terms(isFollow(q)?C.root+' '+q:q);
    const ranked=lines.map(x=>({...x,score:query.reduce((s,t)=>s+(x.text.toLowerCase().includes(t)?1:0),0)})).sort((a,b)=>b.score-a.score||a.line-b.line);
    if(!generic&&query.length&&ranked[0].score===0)return missing('这份资料没有匹配内容',`没有在《${E(f.name)}》找到与当前问题匹配的语句。这是关键词匹配的边界，并不表示现实中不存在答案；请补充资料或换一种表述。`);
    const hits=(ranked[0].score?ranked.filter(x=>x.score>0):lines).slice(0,4).sort((a,b)=>a.line-b.line);
    const short=s=>E(s.replace(/^#+\s*|^[-*]\s*/g,'').slice(0,240));
    const observed=hits.find(x=>!/^#/.test(x.text)&&/结果|发现|记录|测得|增加|降低|寿命|缺陷|偏高/.test(x.text))||hits.find(x=>!/^#/.test(x.text))||hits[0];
    const recommendations=hits.find(x=>/建议|应当|应该|复验|验证|需要|下一步/.test(x.text));
    const uncertainty=hits.find(x=>/可能|尚未|不能|不确定|未排除|缺少|不足/.test(x.text));
    let conclusion=`这份资料给出的关键信息是：“${short(observed.text)}” [L${observed.line}]。${recommendations?'资料还提出：“'+short(recommendations.text)+'” [L'+recommendations.line+']。':'资料目前没有提供可直接执行的验证安排，需要再补齐条件和判据。'}`;
    const rates=[...observed.text.matchAll(/(\d+(?:\.\d+)?)\s*%/g)].map(x=>+x[1]), kinds=[...new Set(observed.text.match(/缺陷率|良品率|合格率|气孔率/g)||[])];
    if(rates.length===2&&kinds.length===1&&(observed.text.match(/样品|方案|试样|批次/g)||[]).length>=2){const delta=rates[1]-rates[0];conclusion=`这份资料的${kinds[0]}对比是 ${rates[0]}% → ${rates[1]}%，相差 ${fmt(Math.abs(delta))} 个百分点${rates[0]>0?'，相对前一个数值'+(delta>=0?'增加':'减少')+fmt(Math.abs(delta)/rates[0]*100)+'%':''}。[L${observed.line}] ${uncertainty?'但原文保留了条件限制，所以这里能确认的是记录差异，不能直接证明某个参数导致了改善。':'这只是在复述并计算该段的记录差异，还不是原因判断。'}`;}
    const why=uncertainty?`原文已经保留了不确定性：“${short(uncertainty.text)}” [L${uncertainty.line}]。所以应将观察现象与原因判断分开，不能把“可能”改成“已经证实”。`:'这些段落描述了现象或处理线索，但仅凭文本命中无法验证因果。程序不会把资料外的解释当作原文事实。';
    const explanations=[['换成工程师能用的话','先把原文的观察结果作为待解释的问题，再把建议作为待验证的行动。资料中没有的温度、指标提升或失效原因，不替它补成确定结论。'],['为什么这样解读',why],['哪些地方还需要确认',`这次只用了《${E(f.name)}》的 ${hits.length} 段内容，没有把内置的 42CrMo 样例混进去。应回到完整原文核对材料、试验条件和上下文，避免截取段落造成误读。`]];
    const evidence=hits.map(x=>`<div class="excerpt"><b>L${x.line}</b><p>${E(x.text)}</p></div>`).join('');
    return sections(conclusion,explanations,evidence,recommendations?'把原文建议整理成验证任务，逐项补充负责人、测试条件、对照组和验收标准；原文没有给出的数值仍需确认。':'请补充对应材料、试验条件、原始结果和对照记录，再形成验证任务。',[],`抽取式归纳与规则解释，不是大模型语义理解；不核验原文真实性。${f.text.length>80000?'当前只分析前 80,000 字符。':''}`,{file:f,count:hits.length});
  }
  function uploadAnswer(f,q,focus){return /\.csv$/i.test(f.name)?csvAnswer(f,q,focus)||textAnswer(f,q,focus):textAnswer(f,q,focus);}
  function followups(topic,mode){return mode==='file'?['为什么得到这个结论？','这份资料缺少哪些证据？','下一步该怎么验证？']:topic==='casting'?['为什么不能只调浇注温度？','具体怎么安排对照验证？','用一句话概括建议']:topic==='machining'?['材料去除率是怎么算的？','为什么不能只追求速度？','下一步怎么验证？']:['为什么不能直接下结论？','这个结论的依据是哪几条？','下一步怎么安排验证？'];}
  function renderAnswer(t,latest=true){
    if(t.cancelled)return `<div class="cancelled-answer">${t.partial?'<div class="copilot-answer">'+t.partial+'</div>':''}<p>已停止。${t.partial?'以上仅为部分内容，回答尚未完成。':'本次没有生成回答。'}</p><button class="btn soft" data-draft="${E(t.q)}">重新编辑这个问题</button></div>`;
    const r=t.result, focus=t.focus;
    let intro=r.conclusion;
    if(focus==='why'&&r.explanations.length){const pattern=/怎么算|怎么得|计算|差值|去除率/.test(t.q)?/差异怎样|参数怎样|为什么得到/:/为什么还不能|为什么不能|为什么不只|为什么这样|为什么暂时/;intro=(r.explanations.find(x=>pattern.test(x[0]))||r.explanations[0])[1];}
    if(focus==='next')intro=r.next;
    if(focus==='evidence')intro=`这个回答使用了 ${r.count||0} 条记录或相关段落。下面可以打开具体依据；观察结果与解释是分开呈现的。`;
    const brief=focus==='brief';
    return `<div class="copilot-answer"><div class="copilot-answer-top"><span class="copilot-spark">✧</span><b>活塞知识助手</b>${t.elapsed?`<small class="response-time">处理与展示 ${t.elapsed} 秒</small>`:''}<span class="badge">${t.scope==='builtin'?'样例分析':'资料解读'} · 规则演示</span></div>${t.follow?`<div class="context-hint">接着“${E(t.contextTitle)}”继续回答 · ${E(t.sourceName)}</div>`:''}<section class="direct-answer"><div class="answer-kicker">${({why:'解释给你听',next:'建议这样验证',evidence:'依据在这里',brief:'给汇报用的一句话'})[focus]||'先说结论'}</div><p>${intro}</p></section>${brief?'':`<div class="explanation-points">${r.explanations.map((x,i)=>`<section><span class="explain-number">0${i+1}</span><div><h3>${x[0]}</h3><p>${x[1]}</p></div></section>`).join('')}</div><section class="next-answer"><h3>下一步可以这样做</h3><p>${r.next}</p>${r.process?`<button class="btn soft" data-page="${r.process}">带着当前问题比较参数 →</button>`:''}</section>`}${latest&&r.metricUI?r.metricUI:''}<details class="answer-evidence" ${focus==='evidence'?'open':''}><summary>查看原始依据与数据 <span>${r.count||0} 条 / 段 · 点击展开</span></summary>${r.evidence}${r.file?`<button class="link" data-open-source="${r.file.cid}">打开完整原文 ↗</button>`:r.refs.map(cite).join(' ')}</details><p class="answer-limit">ⓘ ${r.limit}</p>${latest?`<div class="continue-panel"><b>接着问，不用重新描述背景</b><div class="chips">${followups(t.topic,t.scope==='builtin'?'builtin':'file').map(q=>`<button data-follow="${E(q)}">${q} →</button>`).join('')}</div></div><div class="answer-tools"><button class="link" data-co="copy">复制回答</button><button class="link" data-co="export">保存问答摘要</button><button class="link" data-co="capabilities">查看能力边界</button></div>`:''}</div>`;
  }
  function createTurn(q,topic,follow){
    const focus=/分析|比较|概括|对比/.test(q)?'answer':focusOf(q),f=chosen(),result=f?uploadAnswer(f,q,focus):summarizeBuilt(topic,focus,q);
    return {q,topic,focus,result,follow,scope:C.scope,sourceName:scopeLabel(),contextTitle:f?f.name:titles[topic]||'前一个问题'};
  }
  function evidenceAside(t){
    const r=t&&!t.cancelled?t.result:null;
    return `<aside class="copilot-aside"><div class="card evidence"><span class="eyebrow">ANSWER WITH EVIDENCE</span><h3>${r?'本轮回答依据':'资料工作台'}</h3><p class="muted">${r?'先读解释，需要时再查看原文。':'提交问题后，这里显示关联资料与原始记录。'}</p>${r?.file?`<button class="docitem" data-open-source="${r.file.cid}"><span class="fileicon">FILE</span><span><b>${E(r.file.name)}</b><small>本次添加 · 未上传服务器</small></span></button>`:r?r.refs.map(id=>`<button class="docitem" data-doc="${id}"><span class="fileicon">${id}</span><span><b>${sourceNames[id]}</b><small>内置示例资料</small></span><span>↗</span></button>`).join(''):`<div class="evidence-empty"><span aria-hidden="true">▤</span><b>${C.pending?'正在整理关联资料':'等待你的问题'}</b><p>${C.pending?'正在处理所选资料，完成后可在这里核对来源。':'先选资料，再提问。不会预先替你作出结论。'}</p></div>`}<button class="btn soft full" data-page="docs">管理资料与问答来源 →</button></div><div class="note-box"><h3>先提问，再看分析</h3><p>理解问题 → 关联资料 → 核对条件 → 整理回答</p><p>处理进度与分段输出由本地代码组织，不是在线模型思考。</p><button class="link" data-co="capabilities">查看本地演示能力 ↗</button></div></aside>`;
  }
  const prompts=[['材料研发','350°C 疲劳寿命差异',questions.fatigue],['铸造工艺','气孔问题先检查什么？',questions.casting],['机加工艺','质量与加工效率如何权衡？',questions.machining]];
  function welcome(){return `<section class="question-welcome"><div class="welcome-symbol" aria-hidden="true">✧</div><h2>今天想解决哪个材料或工艺问题？</h2><p>输入问题，点击发送后开始分析。也可以先选一个示例，再补充你的要求。</p><div class="welcome-questions">${(C.scope==='builtin'?prompts:[['资料解读','这份资料说明了什么？','请概括这份资料，并解释主要结论'],['数据对比','这份台账的差异有多大？','对比这份台账的数值，并解释差异']]).map(x=>`<button type="button" data-draft="${E(x[2])}"><small>${x[0]}</small><b>${x[1]}</b><span>填入问题 ↗</span></button>`).join('')}</div><div class="welcome-foot">本地演示 · 提交后才生成回答 · 无需密钥</div></section>`;}
  function progressHTML(job){
    const labels=['理解问题','关联资料','核对条件','整理回答'];
    const phase=job.phase==='output'?'正在呈现回答':labels[job.stage];
    const steps=labels.map((label,i)=>`<div class="process-step ${job.phase==='output'||i<job.stage?'done':i===job.stage?'current':'waiting'}"><span>${job.phase==='output'||i<job.stage?'✓':i+1}</span><b>${label}</b></div>`).join('');
    const hints=[`正在识别问题主题与关注点。`,`正在读取：${E(job.sourceName)}。`,`正在核对材料、温度与指标的可比较条件。`,job.turn?.result?.missing?'当前资料不足，将解释缺少什么，不强行生成结论。':'正在整理结论、解释与可追溯的依据。'];
    return `<section class="kb-process ${job.phase==='output'?'outputting':''}" aria-label="本地处理进度"><div class="process-heading"><div class="process-title"><span class="process-orbit" aria-hidden="true"></span><b>${phase}</b><span class="thinking-dots" aria-hidden="true"><i></i><i></i><i></i></span></div><span class="badge">本地流程演示</span></div><div class="process-steps">${steps}</div><p class="process-message" role="status">${job.phase==='output'?'分析已整理，正在分段显示。你可以随时停止。':hints[job.stage]}</p><div class="process-bottom"><small>资料处理与展示节奏，不代表大模型内部思考。</small><button class="btn" type="button" data-co="stop-answer">■ 停止回答</button></div></section>`;
  }
  function pendingHTML(job){return `<section class="conversation-stage" id="activeQuestion"><div class="question">${E(job.q)}</div>${progressHTML(job)}<div class="copilot-answer answer-stream" aria-busy="true">${job.fragments.slice(0,job.revealed).join('')}</div></section>`;}
  function chatHTML(){
    const history=C.turns.map((turn,i)=>!C.pending&&i===C.turns.length-1?`<section class="conversation-stage" id="activeQuestion"><div class="question">${E(turn.q)}</div>${renderAnswer(turn)}`:`<details class="previous-turn"><summary><span>第 ${i+1} 轮${turn.cancelled?' · 已停止':''}</span>${E(turn.q)}</summary>${renderAnswer(turn,false)}</details>`).join('');
    return history+(C.job?pendingHTML(C.job):C.turns.length?'':welcome());
  }
  knowledge=function(){
    const ff=files();if(C.scope!=='builtin'&&!chosen())C.scope='builtin';
    const t=C.turns.at(-1), status=C.pending?'正在处理':t?(t.cancelled?'已停止':'回答已完成'):'等待提问';
    return head('MATERIALS & PROCESS COPILOT','从一个问题开始，讲清数据与工艺。','先输入问题，再看处理进度、回答解释与原始依据。',btn('＋ 新建对话','new')+btn('▣ 全屏演示','fullscreen'))+`<div class="source-selector card"><div><b>本次使用的资料</b><span>内置样例与添加的资料分开使用</span></div><select id="copilotSource" aria-label="选择本次问答资料" ${C.pending?'disabled':''}><option value="builtin" ${C.scope==='builtin'?'selected':''}>内置材料与工艺样例库</option>${ff.map(f=>`<option value="${f.cid}" ${C.scope===f.cid?'selected':''}>${E(f.name)}</option>`).join('')}</select><button class="btn" data-action="upload">＋ 添加资料</button></div><div class="kb copilot-kb"><div class="card copilot-chat"><div class="panel-head"><h2>✧ 材料与工艺知识助手</h2><span class="grow"></span><span class="badge ${C.pending?'':'green'}" id="chatStatus" role="status">${status} · 纯本地</span><button class="ib" data-co="transcript" aria-label="查看对话记录">◷</button></div><div id="chatBody" class="chat-body copilot-body">${chatHTML()}</div><div class="compose"><div class="chips"><span class="draft-hint">点击填入，发送后分析</span>${(C.scope==='builtin'?prompts.map(x=>[x[1],x[2]]):[['概括这份资料','请概括这份资料，并解释主要结论'],['比较台账数值','对比这份台账的数值，并解释差异']]).map(x=>`<button type="button" data-draft="${E(x[1])}" ${C.pending?'disabled':''}>${x[0]}</button>`).join('')}</div><form id="chatForm" class="composer"><textarea id="questionInput" rows="2" maxlength="800" placeholder="例如：请比较 350°C 疲劳样例，说明哪些差异值得复验…" aria-label="输入问题" ${C.pending?'disabled':''}>${E(C.draft)}</textarea><button id="send" type="submit" class="btn primary" ${C.pending||!C.draft.trim()?'disabled':''}>${C.pending?'处理中…':'发送 ↑'}</button>${C.pending?'<button class="btn stop-compose" type="button" data-co="stop-answer">停止</button>':''}</form><div class="compose-note"><span>资料归纳、数值计算与规则解释 · 非通用大模型</span><span><span id="draftCount">${C.draft.length} / 800</span> · Enter 发送</span></div></div></div>${evidenceAside(t)}</div>`;
  };
  function refreshPending(job){
    if(C.job!==job||job.token!==S.chatToken||S.page!=='knowledge')return false;
    const target=$('activeQuestion');if(target)target.outerHTML=pendingHTML(job);
    if($('chatStatus'))$('chatStatus').textContent=job.phase==='output'?'正在呈现回答 · 纯本地':'正在处理 · 纯本地';
    return true;
  }
  function fragmentsFor(turn){
    // Split complete DOM blocks, never slice raw HTML or expose unfinished tags.
    const box=document.createElement('div');box.innerHTML=renderAnswer(turn);
    const answer=box.firstElementChild,result=[];
    for(const child of answer.children){if(child.classList.contains('explanation-points'))for(const section of child.children)result.push('<div class="explanation-points">'+section.outerHTML+'</div>');else result.push(child.outerHTML);}
    return result;
  }
  function keepTurn(turn){C.turns.push(turn);if(C.turns.length>16)C.turns.shift();}
  function cancelKnowledge(retain=true){
    const job=C.job;if(!job)return;
    S.chatToken++;C.job=null;C.pending=false;S.busy=false;
    if(retain)keepTurn({q:job.q,topic:job.topic,scope:job.scope,sourceName:job.sourceName,cancelled:true,partial:job.fragments.slice(0,job.revealed).filter(x=>!x.includes('continue-panel')&&!x.includes('answer-tools')&&!x.includes('metric-selector')).join('')});
    C.subject=job.previousSubject;C.root=job.previousRoot;S.answered=C.turns.length>0;
  }
  function prepareQuestion(q){
    if(C.pending)return;
    C.draft=String(q||'').trim().slice(0,800);S.answered=C.turns.length>0;page('knowledge');
    const input=$('questionInput');input?.focus({preventScroll:true});input?.scrollIntoView({block:'center',behavior:S.motion?'instant':'smooth'});
  }
  ask=async function(q,preset){
    q=String(q||'').trim().slice(0,800);if(!q||C.pending)return;
    if(titles[preset])C.scope='builtin';
    const previous=C.turns.findLast(t=>!t.cancelled),topic=getTopic(q,preset),follow=!titles[preset]&&!!previous&&previous.scope===C.scope&&isFollow(q)&&(!topicOf(q)||topicOf(q)===C.subject);
    const previousSubject=C.subject,previousRoot=C.root;
    if(!follow){C.root=q;C.subject=topic||'unsupported';}else if(topic)C.subject=topic;
    S.question=q;S.intent=C.subject;S.answered=false;
    const job={token:++S.chatToken,q,topic:C.subject,scope:C.scope,sourceName:scopeLabel(),previousSubject,previousRoot,stage:0,phase:'processing',fragments:[],revealed:0,turn:null,started:performance.now()};
    C.job=job;C.pending=true;C.draft='';S.busy=true;page('knowledge');
    const active=$('activeQuestion'),rect=active?.getBoundingClientRect();
    if(rect&&(rect.top<95||rect.top>window.innerHeight*.65))active.scrollIntoView({block:'start',behavior:'instant'});
    const alive=()=>C.job===job&&job.token===S.chatToken&&S.page==='knowledge';
    const reduced=S.motion||window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    try{
      for(let i=0;i<4;i++){
        if(!alive())return;
        job.stage=i;
        if(i===3)job.turn=createTurn(q,job.topic,follow);
        refreshPending(job);await wait(reduced?120:[550,650,650,500][i]);
      }
      if(!alive())return;
      job.fragments=fragmentsFor(job.turn);job.phase='output';refreshPending(job);
      for(let i=1;i<=job.fragments.length;i++){
        await wait(reduced?0:175);if(!alive())return;
        job.revealed=i;refreshPending(job);
      }
      if(!alive())return;
      job.turn.elapsed=((performance.now()-job.started)/1000).toFixed(1);keepTurn(job.turn);
      S.history.unshift({q,intent:job.topic,time:new Date().toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'})});S.history=S.history.slice(0,16);
      C.job=null;C.pending=false;S.busy=false;S.answered=true;page('knowledge');
    }catch(error){
      if(!alive())return;
      cancelKnowledge(true);page('knowledge');toast('本次未完成，请检查资料格式后重新提问。');
    }finally{
      // A cancelled timer must not unlock or overwrite a newer task.
      if(C.job===job){C.job=null;C.pending=false;S.busy=false;if(S.page==='knowledge')page('knowledge');}
    }
  };
  const oldStop=stop;
  stop=function(notify=true){cancelKnowledge(true);oldStop(notify);};
  page=function(v,write=true){
    const same=S.page===v,y=window.scrollY,x=window.scrollX;
    if(v!=='knowledge')cancelKnowledge(true);
    original.page(v,write);if(S.page==='docs')enhanceDocs();
    if(same)window.scrollTo({top:y,left:x,behavior:'instant'});
  };
  actions.new=function(){cancelKnowledge(false);C.turns=[];C.root='';C.draft='';C.subject='';S.history=[];original.newChat();};
  actions.reset=function(){cancelKnowledge(false);C.turns=[];C.root='';C.subject='';C.draft='';C.scope='builtin';original.reset();S.answered=false;};
  function capability(){modal('资料在哪里，与回答是否智能，是两件事',`<h3>这版已经能做</h3><p>依据样例数值形成结论、解释计算方法和判断边界，保留当前对话上下文。添加的 TXT / MD 可做段落归纳，CSV 可选列、计算差值并引用来源。</p><h3>仍然不能冒充的能力</h3><p>当前没有连接通用大模型。文本解释采用抽取与规则，无法可靠理解任意问题；不含 PDF 解析、图片识别、语义向量检索或复杂统计推断。</p><h3>处理进度代表什么</h3><p>提交问题后展示资料整理、条件核对与分段输出流程。步骤由本地代码组织，不是通用大模型的思维过程；新增的展示节奏不会改变数据或计算结果。</p><p>本次添加的资料只留在浏览器会话内，刷新即清空，不会被发布到 GitHub。</p>`,'<span>演示解读，不伪称在线模型</span>'+btn('明白了','close'),'KNOWLEDGE ENGINE V2.2');}
  actions.about=function(){original.about();$('dialogBody').insertAdjacentHTML('afterbegin','<div class="callout"><b>V2.2 问答层已更新：</b>支持添加文本 / CSV 参与本次资料问答，提供规则解释和追问；仍无通用大模型、PDF 解析或图片识别。</div>');$('dialogBody').innerHTML=$('dialogBody').innerHTML.replace('新添加资料不会参与问答，不具备PDF解析与图片识别。','添加的文本 / CSV 可用于本次资料问答；不具备 PDF 解析与图片识别。');};
  actions.settings=function(){original.settings();$('dialogBody').insertAdjacentHTML('beforeend','<p>知识助手使用 V2.2 规则解读引擎；添加文本可参与本次问答，不自动上传。</p>');};
  function enhanceDocs(){
    const screen=$('screen');if(!screen||$('knowledgeFiles'))return;
    screen.querySelector('.page-head p').textContent='添加文本或 CSV，选择“基于此资料提问”，查看结论、解释和原文依据。';
    screen.querySelectorAll('p.tiny').forEach(el=>{if(el.textContent.includes('不自动参与问答'))el.textContent='文本 / CSV 可用于本次资料问答；图片仅预览。资料不上传服务器，刷新后清空。';});
    const list=files();screen.insertAdjacentHTML('beforeend',`<section class="card pad knowledge-files" id="knowledgeFiles"><div class="row between"><h3>让资料参与回答，而不只是查看</h3><button class="btn soft" data-co="sample">载入演示台账，准备提问</button></div><p>先选资料，再问“差异有多大”“为什么不能直接下结论”“还需补什么证据”。</p>${list.map(f=>`<div class="docitem"><div class="grow"><b>${E(f.name)}</b><small>${/\.csv$/i.test(f.name)?'可选数值列并解释差异':'段落归纳与原文引用'} · 本次会话</small></div><button class="btn primary" data-query-file="${f.cid}">基于此资料提问 →</button></div>`).join('')}</section>`);
  }
  addFiles=async function(ff){await original.addFiles(ff);files();if(S.page==='docs'){$('knowledgeFiles')?.remove();enhanceDocs();}if(ff.some(f=>/\.(?:txt|md|csv)$/i.test(f.name)))toast('资料已添加，点击“基于此资料提问”即可得到解读；不上传服务器。');};
  function openSource(id){const i=S.files.findIndex(f=>f.cid===id);if(i>=0)openLocal(i);else toast('这份会话资料已移除，回答仍保留当时的证据快照。');}
  function askFile(id){if(!files().some(f=>f.cid===id))return;cancelKnowledge(false);C.scope=id;C.root='';C.subject='';C.turns=[];S.history=[];S.answered=false;C.draft='';prepareQuestion('请概括这份资料，并解释主要结论');}
  function transcript(){modal('本次连续对话',C.turns.map((t,i)=>`<details class="previous-turn" ${i===C.turns.length-1?'open':''}><summary>${i+1}. ${E(t.q)}</summary>${renderAnswer(t,false)}</details>`).join('')||'<p>尚未提问。</p>','<span>仅当前页面保存，刷新后清空。</span>'+btn('关闭','close'));}
  function answerText(){const t=C.turns.at(-1);if(!t)return '';const box=document.createElement('div');box.innerHTML=renderAnswer(t,false);box.querySelectorAll('button,select').forEach(e=>e.remove());box.querySelectorAll('h3,p,section,td').forEach(e=>e.append(document.createTextNode('\n')));return '# '+t.q+'\n\n来源：'+t.sourceName+'\n规则解读演示，未连接通用大模型。\n\n'+box.textContent.trim();}
  document.addEventListener('click',async event=>{
    const el=event.target.closest('[data-follow],[data-co],[data-open-source],[data-query-file]');if(!el||el.disabled)return;
    if(el.dataset.follow){if(S.demo)stop(false);ask(el.dataset.follow);}else if(el.dataset.openSource)openSource(el.dataset.openSource);else if(el.dataset.queryFile)askFile(el.dataset.queryFile);else{
      const action=el.dataset.co;
      if(action==='capabilities')capability();if(action==='transcript')transcript();if(action==='stop-answer'){cancelKnowledge(true);page('knowledge');toast('已停止，本次任务不会在后台继续输出');}
      if(action==='export')download('活塞知识助手_问答解释.md',answerText());
      if(action==='copy'){try{await navigator.clipboard.writeText(answerText());toast('已复制回答、解释与来源');}catch(_){modal('复制本次回答','<textarea id="copyAnswer" style="width:100%;min-height:240px" readonly>'+E(answerText())+'</textarea>');$('copyAnswer').select();}}
      if(action==='sample'){const text='样品编号,材料牌号,热处理状态,测试温度(°C),应力幅(MPa),疲劳寿命(次),数据说明\n例-A,42CrMo,调质 A,350,300,120000,人工构造示例\n例-B,42CrMo,调质 B,350,300,150000,人工构造示例\n例-C,42CrMo,调质 C,350,300,180000,人工构造示例';let f=files().find(f=>f.name==='演示用_材料疲劳对比.csv');if(!f){if(S.files.length>=12){toast('请先移除一份资料');return;}f={name:'演示用_材料疲劳对比.csv',size:new Blob([text]).size,image:false,text};S.files.push(f);files();}askFile(f.cid);}
    }
  });
  document.addEventListener('change',event=>{if(event.target.id==='copilotSource'){cancelKnowledge(false);C.scope=event.target.value;C.turns=[];C.root='';C.subject='';C.draft='';S.history=[];S.answered=false;page('knowledge');}if(event.target.id==='copilotMetric'){const f=chosen();if(f){C.metric[f.cid]=+event.target.value;prepareQuestion('解释这个指标的差异');}}});
  // Suggestions prepare a draft. Only Send / Enter (or explicit auto-demo) runs a task.
  document.addEventListener('click',event=>{const el=event.target.closest('[data-draft],[data-ask],[data-follow]');if(!el)return;event.preventDefault();event.stopImmediatePropagation();if(C.pending){toast('请先停止当前回答，再编辑下一个问题');return;}if(S.demo)stop(false);if(el.dataset.ask){C.scope='builtin';prepareQuestion(questions[el.dataset.ask]||'');}else prepareQuestion(el.dataset.draft||el.dataset.follow||'');},true);
  document.addEventListener('input',event=>{if(event.target.id==='questionInput'){C.draft=event.target.value;if($('draftCount'))$('draftCount').textContent=C.draft.length+' / 800';if($('send'))$('send').disabled=C.pending||!C.draft.trim();}});
  document.addEventListener('keydown',event=>{if(event.key==='Escape'&&C.pending&&$('overlay').hidden){cancelKnowledge(true);page('knowledge');toast('已停止当前回答');}});

  window.PistonKnowledge={parseCSV,numericColumns,topicOf,focusOf,answerText,state:C};
  actions.history=transcript;
  page(S.page,false);
  document.documentElement.classList.remove('knowledge-boot');
})();
