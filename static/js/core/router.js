import {features, lessons, routeForHash} from './lesson-registry.js';
import {loadPage} from './lesson-loader.js';

export function startRouter(container, navigation, status) {
  const allRoutes = [...lessons, ...features];
  for (const route of allRoutes) {
    const link = document.createElement('a');
    link.href = route.hash;
    link.className = route.kind === 'lesson' ? 'lesson-link' : 'feature-link';
    link.textContent = route.kind === 'lesson' ? `${route.id}  ${route.title}` : route.title;
    navigation.appendChild(link);
  }

  let controller = null;
  let cleanup = () => {};
  let generation = 0;
  async function navigate() {
    controller?.abort();
    cleanup();
    cleanup = () => {};
    const current = ++generation;
    controller = new AbortController();
    const route = routeForHash(location.hash);
    if (route.id === '00' && location.hash && location.hash !== route.hash) {
      // 不存在的章节返回起点，同时修正地址栏，避免刷新后仍看到错误链接。
      history.replaceState(null, '', route.hash);
    }
    if (route.legacy) {
      location.assign(route.legacy);
      return;
    }
    document.title = `${route.title} · 从像素开始`;
    for (const link of navigation.querySelectorAll('a')) {
      if (link.hash === route.hash) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    }
    status.hidden = false;
    status.textContent = `正在打开「${route.title}」…`;
    container.replaceChildren();
    try {
      const mounted = await loadPage(route, container, controller.signal);
      if (current !== generation || controller.signal.aborted) {
        mounted();
        return;
      }
      cleanup = mounted;
      status.hidden = true;
      if (route.focus === 'settings') {
        const settings = container.querySelector('#settings');
        if (settings) {
          settings.focus({preventScroll: true});
          settings.scrollIntoView({block: 'start'});
        }
      } else {
        container.focus({preventScroll: true});
        window.scrollTo({top: 0, behavior: 'auto'});
      }
    } catch (error) {
      if (current !== generation || error.name === 'AbortError') return;
      status.hidden = false;
      status.textContent = error.message || '课程暂时无法打开，请刷新网页。';
      const retry = document.createElement('button');
      retry.type = 'button';
      retry.textContent = '重新尝试';
      retry.addEventListener('click', navigate, {once: true});
      container.replaceChildren(retry);
    }
  }
  window.addEventListener('hashchange', navigate);
  navigate();
  return () => {
    generation++;
    controller?.abort();
    cleanup();
    window.removeEventListener('hashchange', navigate);
  };
}
