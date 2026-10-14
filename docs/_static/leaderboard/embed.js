/* The docs mount is optional; ordinary documentation pages are unaffected. */
(() => {
  'use strict';
  const scriptBase = new URL('.', document.currentScript.src);
  async function mount() {
    const host = document.getElementById('rpent-interactive-leaderboard');
    if (!host || host.dataset.loaded) return;
    const hash=location.hash.replace(/^#(?:native-)?/,'');
    const params=new URLSearchParams(location.search);
    const requested=hash==='time-token-costs'||hash.startsWith('costs-')?'time-token-costs':
      hash&&hash!=='top'?'performance':
      params.get('section')==='time-token-costs'?'time-token-costs':null;
    const section=host.dataset.section;
    if(section==='index'||(section&&requested&&requested!==section)){
      const target=requested??'performance';
      const path=target==='performance'?host.dataset.performanceUrl:host.dataset.costsUrl;
      if(path){
        const url=new URL(path,location.href);url.search=location.search;url.hash=location.hash;
        url.searchParams.delete('section');location.replace(url.href);return;
      }
    }
    try {
      const response = await fetch(host.dataset.resultsUrl);
      if (!response.ok) throw new Error(`Results HTTP ${response.status}`);
      const data = await response.json();
      const templateResponse = await fetch(new URL('embed.html', scriptBase));
      if (!templateResponse.ok) throw new Error(`Template HTTP ${templateResponse.status}`);
      const markup = await templateResponse.text();
      const detached = document.createElement('div');
      detached.style.display = 'none';
      const root = detached.attachShadow({mode: 'open'});
      root.innerHTML = markup;
      const style = document.createElement('link');
      style.rel = 'stylesheet';style.href = new URL('embed.css', scriptBase);
      const loaded = new Promise((resolve, reject) => {style.onload = resolve;style.onerror = reject;});
      root.prepend(style);
      host.append(detached);
      await loaded;
      detached.style.removeProperty('display');
      window.RPentLeaderboard.mount(root, data, {language:host.dataset.language??'en',assetBase:scriptBase.href,section});
      host.querySelector('.rpent-static-leaderboard')?.remove();
      host.dataset.loaded = 'true';
    } catch (error) {
      host.querySelectorAll(':scope > div:not(.rpent-static-leaderboard)').forEach(el => el.remove());
      // The compact native summary stays readable when interactive assets fail.
      host.dataset.loadError = 'true';
      console.warn('RPent interactive results unavailable; showing the summary.', error);
    } finally {
      clearTimeout(host.loadingTimeout);
      delete host.dataset.loading;
      host.removeAttribute('aria-busy');
    }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount, {once:true});
  else mount();
})();
