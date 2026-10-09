(() => {
  'use strict';
  function mount(root, d, options) {
  const pageDocument=globalThis.document;
  const themeTarget=root.host;
  themeTarget.dataset.embedded='true';
  const systemTheme=window.matchMedia('(prefers-color-scheme: dark)');
  const syncTheme=()=>{
    const theme=pageDocument.documentElement.dataset.theme;
    themeTarget.dataset.theme=theme==='dark'||theme==='light'?theme:systemTheme.matches?'dark':'light';
    const palette=d.palette[themeTarget.dataset.theme];
    themeTarget.style.setProperty('--best',palette.best);
    themeTarget.style.setProperty('--blue',palette.rpent);
    themeTarget.style.setProperty('--baseline',palette.reference);
  };
  syncTheme();
  systemTheme.addEventListener('change',syncTheme);
  new MutationObserver(syncTheme).observe(pageDocument.documentElement,{attributes:true,attributeFilter:['data-theme']});
  const document = root;
  const query = new URLSearchParams(location.search);
  let lang = options.language ?? (query.get('lang') === 'en' ? 'en' : 'zh');
  const views = new Map(d.benchmarks.flatMap(b=>b.views.map(v=>[v.id,{...v,benchmark_id:b.id}])));
  const rowsByView = new Map([...views.keys()].map(id=>[id,[]]));
  d.results.forEach(r=>rowsByView.get(r.view_id).push(r));
  const configs = new Map(d.configurations.map(c => [c.id, c]));
  const selection = new Map(d.benchmarks.map(s => [s.id, s.default_view]));
  const costGroups=d.benchmarks.filter(b=>d.cost_results.some(r=>r.benchmark_id===b.id)).map(b=>[b.id,b.name]);
  const collapsedModules = new Set([...d.benchmarks.map(s=>s.id),...costGroups.map(([id])=>'costs-'+id)]);
  const normalizeAnchor=hash=>hash.replace(/^#/, '').replace(/^native-/, '');
  const sectionForAnchor=id=>id==='time-token-costs'||id.startsWith('costs-')?'time-token-costs':id&&id!=='top'?'performance':null;
  let activeSection=options.section??sectionForAnchor(normalizeAnchor(location.hash))??(query.get('section')==='time-token-costs'?'time-token-costs':'performance');
  const activeModules=new Map();
  if (views.has(query.get('view'))) {
    const s = d.benchmarks.find(s => s.views.some(v=>v.id===query.get('view')));
    if (s) selection.set(s.id, query.get('view'));
  }
  const copy = {
  "en": {
    "paper": "Paper",
    "title": "RPent Leaderboard",
    "performance": "Performance",
    "costs": "Time & Token Costs",
    "backTop": "Back to top ↑",
    "view": "Evaluation",
    "rate": "Success rate",
    "method": "Method",
    "meanTime": "Mean time / episode (s)",
    "mean_total_tokens": "Mean input + output tokens / episode",
    "total_output_tokens": "Total output tokens",
    "unreported": "Not reported",
    "modelUnknown": "Model not reported",
    "noReasoning": "no-reasoning",
    "table": "Results table",
    "download": "Download CSV",
    "print": "Print page",
    "menu": "Chart options",
    "countsOnly": "Evaluated",
    "saved": "CSV downloaded",
    "allScores": "All methods & reported scores",
    "environments": "Environments"
  },
  "zh": {
    "paper": "论文",
    "title": "RPent 排行榜",
    "performance": "评测成绩",
    "costs": "耗时与 Token 开销",
    "backTop": "返回顶部 ↑",
    "view": "评测范围",
    "rate": "成功率",
    "method": "方法",
    "meanTime": "平均每回合耗时（秒）",
    "mean_total_tokens": "平均每回合输入＋输出 token",
    "total_output_tokens": "总输出 token",
    "unreported": "未报告",
    "modelUnknown": "未报告模型",
    "noReasoning": "no-reasoning",
    "table": "结果明细表",
    "download": "下载 CSV",
    "print": "打印页面",
    "menu": "图表选项",
    "countsOnly": "已评测",
    "saved": "CSV 已下载",
    "allScores": "完整方法与分项成绩",
    "environments": "环境"
  }
};
  const t = key => copy[lang][key] ?? key;
  const tr = value => value && typeof value === 'object' ? value[lang] ?? value.en : value;
  const h = value => String(value ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
  const asset = name => new URL(`assets/${name}`,options.assetBase).href;
  function label(r) {
    const c=configs.get(r.configuration_id);
    if (c.display_name) return c.display_name;
    if (c.kind==='external') return c.model;
    return c.model?`RPent / ${c.model}`:'RPent / '+c.backend;
  }
  function detailLabel(r) {
    const c=configs.get(r.configuration_id);
    if(c.perception_model) return `${c.perception_model} · ${lang==='en'?'visual localization':'视觉定位'}`;
    if(r.evaluation_note) return tr(r.evaluation_note);
    const reasoning=c.effort==='max'&&c.reasoning===true?['max.reasoning']:[c.effort,c.reasoning===false?t('noReasoning'):c.reasoning===true?'reasoning':null];
    return [c.kind==='external'?null:c.backend, ...reasoning,
      !c.model && c.kind!=='external'?t('modelUnknown'):null].filter(Boolean).join(' · ');
  }
  function color(r, bestRate) {
    const c=configs.get(r.configuration_id);
    if(Number(r.rate)===bestRate)return 'var(--best)';
    return c.kind==='external'?'var(--baseline)':'var(--blue)';
  }
  const allRows = view => [...rowsByView.get(view.id)].sort((a,b)=>
    a.rate===null?b.rate===null?a.id.localeCompare(b.id):1:b.rate===null?-1:Number(b.rate)-Number(a.rate)||a.id.localeCompare(b.id));
  const count = r => r.successes!==null?`${r.successes}/${r.episodes}`:r.episodes!==null?`${t('countsOnly')}: ${r.episodes}`:'—';
  function headline(id) {
    const view=views.get(id);
    const benchmark=d.benchmarks.find(b=>b.id===view.benchmark_id);
    return benchmark.name+' · '+tr(view.label)+' · '+t('rate');
  }
  function noteReference(record) {
    const benchmark=d.benchmarks.find(b=>b.id===views.get(record.view_id).benchmark_id);
    return (benchmark.notes??[]).map((note,index)=>{
      if(!note.configuration_ids.includes(record.configuration_id))return '';
      const marker=index+1;
      const description=lang==='en'?`Evaluation note ${marker} for ${label(record)}`:`${label(record)}的评测说明 ${marker}`;
      return `<sup class="note-reference"><a href="#${h(note.id)}" role="doc-noteref" aria-label="${h(description)}">[${marker}]</a></sup>`;
    }).join('');
  }
  function noteParagraph(note,index) {
    return `<p id="${h(note.id)}" class="scope-note" tabindex="-1" role="note"><span class="note-number">[${index+1}]</span> ${h(tr(note.text))}</p>`;
  }
  function chart(view, kind='primary') {
    const rows=allRows(view).filter(r=>r.rate!==null);
    const bestRate=Number(rows[0]?.rate);
    return `<div class="chart ${kind}" aria-label="${h(headline(view.id))}"><div class="chart-grid" aria-hidden="true">${Array.from({length:6},()=>'<i></i>').join('')}</div><div class="chart-rows">${rows.map(r=>
      `<div class="chart-row" data-record="${r.id}" aria-label="${h(label(r)+', '+detailLabel(r)+', '+r.rate+'%, '+count(r))}"><div class="method-label"><span class="method-text"><span class="method-name">${h(label(r))}${noteReference(r)}</span><span class="method-description">${h(detailLabel(r))}</span></span></div><div class="bar-track"><div class="bar${Number(r.rate)===0?' zero':''}" style="--value:${Number(r.rate)}%;--bar-color:${color(r,bestRate)}"><span class="bar-value">${r.rate}%</span></div></div></div>`).join('')}</div><div class="axis" aria-hidden="true">${[0,20,40,60,80,100].map(x=>`<span>${x}</span>`).join('')}</div></div>`;
  }
  function comparisonMatrix(s) {
    const columns=s.views;
    const rows=d.configurations.filter(c=>columns.some(v=>rowsByView.get(v.id).some(r=>r.configuration_id===c.id&&r.rate!==null)));
    return `<details class="comparison-matrix" data-details="${s.id}"><summary>${t('allScores')} <span>${rows.length} ${lang==='en'?'configurations':'配置'}</span></summary><div class="table-wrap" tabindex="0" role="region" aria-label="${h(s.name+' '+t('allScores'))}"><table><thead><tr><th>${t('method')}</th>${columns.map(v=>`<th class="rate">${h(tr(v.label))}</th>`).join('')}</tr></thead><tbody>${rows.map(c=>{
      const sample=columns.flatMap(allRows).find(r=>r.configuration_id===c.id&&r.rate!==null);
      return `<tr data-method="${c.id}"><th scope="row"><div class="matrix-label"><span>${h(label(sample))}${noteReference(sample)}<small> ${h(detailLabel(sample))}</small></span></div></th>${columns.map(v=>{
        const r=allRows(v).find(r=>r.configuration_id===c.id);
        return `<td class="rate"${r?` data-matrix-record="${r.id}"`:''} title="${h(tr(r?.evaluation_note)||tr(v.label))}">${r?.rate!=null?r.rate+'%':`<span class="not-reported" aria-label="${t('unreported')}">—</span>`}</td>`;
      }).join('')}</tr>`;
    }).join('')}</tbody></table></div></details>`;
  }
  function sectionBody(s) {
    const view=views.get(selection.get(s.id));
    return `<div class="subsection-heading"><h4>${h(tr(s.heading))}</h4>${s.summary?`<p>${h(tr(s.summary))}</p>`:''}</div><div class="chart-tools"><div class="view-controls"><label for="view-${s.id}">${t('view')}</label><select id="view-${s.id}" data-section="${s.id}" aria-label="${h(s.name+' '+t('view'))}">${s.views.map(v=>`<option value="${v.id}"${view.id===v.id?' selected':''}>${h(tr(v.label))}</option>`).join('')}</select></div><div class="chart-menu"><button type="button" class="icon-button" data-menu="${s.id}" aria-haspopup="menu" aria-expanded="false" aria-controls="menu-${s.id}" aria-label="${t('menu')}" title="${t('menu')}"><img src="${asset('ellipsis.svg')}" alt=""></button><div class="menu-options" id="menu-${s.id}" role="menu" hidden><button role="menuitem" data-action="csv" data-section="${s.id}">${t('download')}</button><button role="menuitem" data-action="table" data-section="${s.id}">${t('table')}</button><button role="menuitem" data-action="print">${t('print')}</button></div></div></div>${chart(view)}${comparisonMatrix(s)}`;
  }
  function renderSection(s) {
    const element=document.getElementById('panel-'+s.id);
    element.innerHTML=sectionBody(s)+(s.notes??[]).map(noteParagraph).join('');
    setupTables(element,s.id);
  }
  function setupTables(element,key) {
    window.RPentTables.setup(element,{key,language:lang,icon:asset('arrow-down.svg'),download:asset('download.svg')});
  }
  function renderCosts() {
    const element=document.getElementById('time-token-costs');
    const groups=costGroups;
    const methodOrder=new Map(d.configurations.map((c,i)=>[c.id,i]));
    element.innerHTML=`<div class="section-layout"><details class="module-directory" open><summary>${t('environments')}</summary><nav class="module-nav" aria-label="${h(t('costs'))}">${groups.map(([id,name])=>`<a href="#costs-${id}">${name}</a>`).join('')}</nav></details><div class="section-results">${groups.map(([id,name])=>{
      const rows=d.cost_results.filter(r=>r.benchmark_id===id).sort((a,b)=>(methodOrder.get(a.configuration_id)??Infinity)-(methodOrder.get(b.configuration_id)??Infinity)||a.id.localeCompare(b.id));
      const tokenMetrics=['total_output_tokens','mean_total_tokens'].filter(key=>rows.some(r=>r[key]!=null));
      return `<section id="costs-${id}" class="cost-group" data-cost-group="${id}"><div class="section-inner"><details data-module="costs-${id}"${collapsedModules.has('costs-'+id)?'':' open'}><summary class="module-header"><h3>${name}</h3></summary><div class="module-content"><div class="table-wrap" tabindex="0" role="region" aria-label="${name} ${h(t('costs'))}"><table data-sort-key="costs-${id}" data-sortable="false" data-highlight-best="false"><thead><tr><th>${t('method')}</th><th class="rate">${t('meanTime')}</th>${tokenMetrics.map(key=>`<th class="rate">${t(key)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>{
        return `<tr data-cost-record="${r.id}" data-method="${r.configuration_id}"><td><div class="matrix-label"><span>${h(label(r))}<small> ${h(detailLabel(r))}</small></span></div></td><td class="rate" data-value="${r.mean_elapsed_seconds??''}">${r.mean_elapsed_seconds===null?'—':r.mean_elapsed_seconds.toLocaleString('en-US')}</td>${tokenMetrics.map(key=>`<td class="rate" data-value="${r[key]??''}">${r[key]==null?'—':r[key].toLocaleString('en-US',{maximumFractionDigits:0})}</td>`).join('')}</tr>`;
      }).join('')}</tbody></table></div></div></details></div></section>`;
    }).join('')}</div></div>`;
    setupTables(element,'costs');
  }
  function render() {
    document.querySelectorAll('details[data-module]').forEach(el=>{
      if(el.open)collapsedModules.delete(el.dataset.module);else collapsedModules.add(el.dataset.module);
    });
    document.host.lang=lang==='en'?'en':'zh-CN';
    document.querySelectorAll('[data-i18n]').forEach(el=>el.textContent=t(el.dataset.i18n));
    document.querySelectorAll('[data-i18n-label]').forEach(el=>el.setAttribute('aria-label',t(el.dataset.i18nLabel)));
    const language=document.getElementById('language');language.textContent=lang==='en'?'中文':'English';language.lang=lang==='en'?'zh-CN':'en';language.setAttribute('aria-label',lang==='en'?'切换为中文':'Switch to English');
    document.querySelector('#performance .module-nav').innerHTML=d.benchmarks.map(b=>`<a href="#${h(b.id)}">${h(b.name)}</a>`).join('');
    document.getElementById('leaderboards').innerHTML=d.benchmarks.map(s=>`<section id="${s.id}" class="benchmark-band"><div class="benchmark-shell"><details data-module="${s.id}"${collapsedModules.has(s.id)?'':' open'}><summary class="module-header"><h3 class="benchmark-wordmark">${h(s.name)}</h3></summary><div class="module-content"><div id="panel-${s.id}"></div></div></details></div></section>`).join('');
    d.benchmarks.forEach(renderSection);renderCosts();updateSection();observeNavigation();
  }
  function downloadCSV(rows, name) {
    const keys=['record_id','benchmark','view','method','model','planner','perception_model','reasoning','effort','success_rate_percent','successes','episodes','status','evaluation_note'];
    const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';
    const body=[keys.join(','),...rows.map(r=>{
      const c=configs.get(r.configuration_id),v=views.get(r.view_id);
      return [r.id,v.benchmark_id,r.view_id,label(r),c.model,c.backend,c.perception_model,c.reasoning,c.effort,r.rate,r.successes,r.episodes,r.status,tr(r.evaluation_note)].map(quote).join(',');
    })].join('\r\n');
    const url=URL.createObjectURL(new Blob(['\uFEFF'+body],{type:'text/csv;charset=utf-8'}));
    const a=globalThis.document.createElement('a');a.href=url;a.download=name+'.csv';(document.body??document).append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),3000);
    document.getElementById('announcement').textContent=t('saved');
  }
  function closeMenus(focus=false) {
    document.querySelectorAll('[data-menu][aria-expanded="true"]').forEach(button=>{document.getElementById('menu-'+button.dataset.menu).hidden=true;button.setAttribute('aria-expanded','false');if(focus)button.focus();});
  }
  const main=document.getElementById('main');
  themeTarget.dataset.separatePages=Boolean(options.section).toString();
  let compact=null;
  const syncNavHeight=()=>{
    const next=main.getBoundingClientRect().width<820;
    main.dataset.compact=String(next);
    if(compact!==next){
      document.querySelectorAll('.module-directory').forEach(directory=>directory.open=!next);
      compact=next;
    }
    const directory=document.getElementById(activeSection).querySelector('.module-directory');
    themeTarget.style.setProperty('--directory-height',next?directory.getBoundingClientRect().height+'px':'0px');
  };
  const navObserver=new ResizeObserver(syncNavHeight);
  function observeNavigation(){
    navObserver.disconnect();navObserver.observe(main);
    document.querySelectorAll('.module-directory').forEach(directory=>{directory.open=!compact;navObserver.observe(directory);});
    syncNavHeight();
  }
  function markModule(id){
    if(id)activeModules.set(activeSection,id);
    document.querySelectorAll('.module-nav a').forEach(a=>{
      if(a.hash.slice(1)===activeModules.get(activeSection))a.setAttribute('aria-current','location');
      else a.removeAttribute('aria-current');
    });
  }
  function updateSection(){
    for(const id of ['performance','time-token-costs'])document.getElementById(id).hidden=id!==activeSection;
    document.getElementById('page-section-heading').textContent=t(activeSection==='performance'?'performance':'costs');
    document.querySelectorAll('.section-nav a').forEach(a=>{
      if(a.hash.slice(1)===activeSection)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');
    });
    document.querySelector('.skip-link').href='#'+activeSection;
    markModule();
  }
  function revealTarget(target) {
    if(target.matches('.benchmark-band,.cost-group')){
      const module=target.querySelector('details[data-module]');
      if(module){module.open=true;collapsedModules.delete(module.dataset.module);}
    }
    for(let node=target.parentElement;node;node=node.parentElement){
      if(node.matches('details'))node.open=true;
    }
    const group=target.closest('.benchmark-band,.cost-group');
    if(group)markModule(group.id);
  }
  document.addEventListener('toggle',e=>{
    const module=e.target;
    if(module.matches('.module-directory')){syncNavHeight();return;}
    if(!module.matches('details[data-module]')||!module.isConnected)return;
    if(module.open)collapsedModules.delete(module.dataset.module);else collapsedModules.add(module.dataset.module);
    closeMenus();
    if(!module.open&&module.contains(document.activeElement)){
      requestAnimationFrame(()=>{
        const summary=module.querySelector('summary');
        if(summary.getBoundingClientRect().top<parseFloat(getComputedStyle(summary).top))summary.scrollIntoView({block:'start'});
      });
    }
  },true);
  document.addEventListener('change',e=>{
    const select=e.target.closest('select[data-section]');if(!select)return;
    const id=select.dataset.section;selection.set(id,select.value);renderSection(d.benchmarks.find(s=>s.id===id));
    document.getElementById('view-'+id).focus({preventScroll:true});
    try{const url=new URL(location.href);url.searchParams.set('view',selection.get(id));url.searchParams.set('lang',lang);history.replaceState(null,'',url);}catch{/* file previews need no history support */}
  });
  document.addEventListener('click',e=>{
    const anchor=e.target.closest('a[href^="#"]');
    if(anchor&&document.getElementById(normalizeAnchor(anchor.hash))){
      e.preventDefault();navigate(normalizeAnchor(anchor.hash),true);
      if(anchor.closest('.note-reference'))document.getElementById(normalizeAnchor(anchor.hash)).focus({preventScroll:true});
    }
    const summary=e.target.closest('.module-header');
    if(summary)markModule(summary.parentElement.dataset.module);
    const langButton=e.target.closest('#language');if(langButton){lang=lang==='en'?'zh':'en';render();try{const u=new URL(location.href);u.searchParams.set('lang',lang);history.replaceState(null,'',u);}catch{}return;}
    const menu=e.target.closest('[data-menu]');if(menu){const wasOpen=menu.getAttribute('aria-expanded')==='true';closeMenus();if(!wasOpen){menu.setAttribute('aria-expanded','true');const panel=document.getElementById('menu-'+menu.dataset.menu);panel.hidden=false;panel.querySelector('button').focus();}return;}
    const action=e.target.closest('[data-action]');if(action){
      if(action.dataset.action==='csv')downloadCSV(allRows(views.get(selection.get(action.dataset.section))).filter(r=>r.rate!==null),selection.get(action.dataset.section));
      if(action.dataset.action==='print')window.print();
      if(action.dataset.action==='table'){const el=document.querySelector(`[data-details="${action.dataset.section}"]`);el.open=true;el.querySelector('summary').focus({preventScroll:true});}
      closeMenus(action.dataset.action!=='table');return;
    }
    if(!e.target.closest('.chart-menu'))closeMenus();
  });
  document.addEventListener('keydown',e=>{
    if(e.key==='Escape')closeMenus(true);
    const item=e.target.closest('[role="menuitem"]');
    if(item&&['ArrowDown','ArrowUp','Home','End'].includes(e.key)){
      e.preventDefault();const items=[...item.parentElement.children],i=items.indexOf(item);
      items[e.key==='Home'?0:e.key==='End'?items.length-1:(i+(e.key==='ArrowDown'?1:items.length-1))%items.length].focus();
    }
  });
  function navigate(id,push=false){
    const target=document.getElementById(id);
    if(!target)return;
    const owner=target.closest('#performance,#time-token-costs');
    if(owner)activeSection=owner.id;
    updateSection();revealTarget(target);
    if(compact)document.querySelectorAll('.module-directory').forEach(directory=>directory.open=false);
    syncNavHeight();closeMenus();
    if(push){
      try{const url=new URL(location.href);url.searchParams.set('section',activeSection);url.searchParams.set('lang',lang);url.hash=id;if(activeSection==='time-token-costs')url.searchParams.delete('view');history.pushState(null,'',url);}catch{/* file previews need no history support */}
    }
    requestAnimationFrame(()=>target.scrollIntoView({block:'start',behavior:push&&!window.matchMedia('(prefers-reduced-motion: reduce)').matches?'smooth':'instant'}));
  }
  function restoreLocation(){
    const params=new URLSearchParams(location.search);
    const nextLang=params.get('lang');
    if(nextLang==='en'||nextLang==='zh')lang=nextLang;
    const view=params.get('view');
    const owner=d.benchmarks.find(s=>s.views.some(v=>v.id===view));
    if(owner)selection.set(owner.id,view);
    const id=normalizeAnchor(location.hash);
    activeSection=options.section??sectionForAnchor(id)??(params.get('section')==='time-token-costs'?'time-token-costs':'performance');
    render();
    if(id)navigate(id);else if(owner&&activeSection==='performance')navigate(owner.id);
  }
  restoreLocation();
  window.addEventListener('hashchange',restoreLocation);
  window.addEventListener('popstate',restoreLocation);
  }
  window.RPentLeaderboard={mount};
})();
