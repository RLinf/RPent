/* Numeric sorting moves complete rows; task/seed identities never move independently. */
(() => {
  'use strict';
  const saved = new Map();
  const collator = new Intl.Collator('en', {numeric:true});
  const valueOf = cell => {
    if (!cell) return null;
    if (cell.dataset.value !== undefined) return cell.dataset.value === '' ? null : Number(cell.dataset.value);
    const text=cell.textContent.trim();
    const fraction=text.match(/^(\d+)\/(\d+)$/);
    if(fraction) return Number(fraction[1])/Number(fraction[2]);
    const number=text.match(/^(\d[\d,]*(?:\.\d+)?)(?:%|\s+\([\d.]+%\))?$/);
    return number ? Number(number[1].replaceAll(',','')) : null;
  };
  function sort(table, column) {
    const state=table._rpentSort;
    state.column=column;saved.set(state.key,column);
    const ascending=column===state.idColumn||state.direction==='ascending';
    const rows=[...table.tBodies[0].rows].filter(row=>!row.dataset.total);
    rows.sort((a,b)=>{
      const av=valueOf(a.cells[column]),bv=valueOf(b.cells[column]);
      return av===null ? bv===null?collator.compare(a.dataset.sortId,b.dataset.sortId):1
        : bv===null?-1:(ascending?av-bv:bv-av)||collator.compare(a.dataset.sortId,b.dataset.sortId);
    });
    const total=table.tBodies[0].querySelector('[data-total]');
    rows.forEach(row=>table.tBodies[0].insertBefore(row,total));
    state.headers.forEach((th,i)=>{
      if(state.columns.includes(i))th.setAttribute('aria-sort',i===column?(ascending?'ascending':'descending'):'none');
    });
    const maximum=rows.length?valueOf(rows[0].cells[column]):null;
    for(const row of rows){
      for(const cell of row.cells)cell.classList.remove('metric-best');
      if(state.highlightBest&&column!==state.idColumn&&maximum!==null&&valueOf(row.cells[column])===maximum)row.cells[column].classList.add('metric-best');
    }
  }
  function exportText(cell) {
    const copy=cell.cloneNode(true);
    copy.querySelectorAll('.note-reference').forEach(marker=>marker.remove());
    const text=copy.textContent.trim();
    return text==='—'?'':text;
  }
  function exportTable(table) {
    const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';
    const state=table._rpentSort;
    const rows=[state.headers.map(th=>th.dataset.label),...[...table.tBodies[0].rows].map(row=>[...row.cells].map(exportText))];
    const url=URL.createObjectURL(new Blob(['\uFEFF'+rows.map(row=>row.map(quote).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}));
    const a=document.createElement('a');a.href=url;a.download=state.key.replaceAll(/[^a-zA-Z0-9-]/g,'-')+'.csv';a.click();
    setTimeout(()=>URL.revokeObjectURL(url),3000);
  }
  function setup(scope, options={}) {
    const zh=(options.language??document.documentElement.lang).startsWith('zh');
    scope.querySelectorAll('table:not(.coverage)').forEach((table,index)=>{
      if(table._rpentSort||!table.tHead||!table.tBodies[0])return;
      const headers=[...table.tHead.rows[0].cells];
      const idColumn=headers.findIndex(th=>/^(ID|#)$/i.test(th.textContent.trim()));
      const candidates=headers.map((th,i)=>({th,i})).filter(({th,i})=>(i>0||i===idColumn)&&!/^s\d+$/i.test(th.textContent.trim())&&[...table.tBodies[0].rows].some(r=>valueOf(r.cells[i])!==null));
      if(!candidates.length)return;
      const columns=candidates.map(x=>x.i);
      const preferred=candidates.find(({i})=>i===idColumn) ?? candidates.find(({th})=>/Overall|总体/i.test(th.textContent))
        ?? candidates.find(({th})=>/Mean|均值|Success rate|成功率/i.test(th.textContent)) ?? candidates[0];
      const key=table.dataset.sortKey??(options.key??scope.id??'results')+'-'+index;
      [...table.tBodies[0].rows].forEach((row,i)=>{
        row.dataset.sortId??=row.dataset.method??row.dataset.tableRecord??row.dataset.astraTask??row.dataset.roboTask??row.dataset.paperTasks??row.cells[0].textContent.trim()+'-'+i;
        if(/^(Total|合计)$/i.test(row.cells[0].textContent.trim()))row.dataset.total='true';
      });
      const direction=table.dataset.sortDirection??'descending';
      const sortable=table.dataset.sortable!=='false';
      table._rpentSort={key,columns,headers,idColumn,column:preferred.i,direction,highlightBest:table.dataset.highlightBest!=='false'};
      headers.forEach((th,i)=>{
        th.dataset.label=th.textContent.trim();
        if(!sortable||!columns.includes(i))return;
        const button=document.createElement('button');button.type='button';button.className='sort-heading';
        button.textContent=th.dataset.label;button.title=(i===idColumn?(zh?'按任务 ID 升序：':'Sort task ID ascending: '):direction==='ascending'?(zh?'按此列升序：':'Sort ascending: '):(zh?'按此列降序：':'Sort descending: '))+th.dataset.label;
        const img=document.createElement('img');img.src=options.icon;img.alt='';img.width=12;img.height=12;
        button.append(img);button.addEventListener('click',()=>sort(table,i));th.replaceChildren(button);
      });
      if(sortable){const savedColumn=saved.get(key);sort(table,columns.includes(savedColumn)?savedColumn:preferred.i);}
      if(options.download){
        const button=document.createElement('button');button.type='button';button.className='table-download icon-button';
        button.title=zh?'下载当前排序 CSV':'Download sorted CSV';button.setAttribute('aria-label',button.title);
        const image=document.createElement('img');image.src=options.download;image.alt='';button.append(image);
        button.addEventListener('click',()=>exportTable(table));table.parentElement.before(button);
      }
    });
  }
  window.RPentTables={setup,valueOf};
})();
