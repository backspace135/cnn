import {mountSteps} from '../components/stepper.js';

// 课程片段来自项目自身；不把学员输入或 API 文本当成 HTML 注入页面。
export async function loadPage(entry, container, signal) {
  const response = await fetch(entry.url, {signal});
  if (!response.ok) throw new Error(`找不到这一页（${response.status}）。请重新打开课程。`);
  const documentPart = new DOMParser().parseFromString(await response.text(), 'text/html');
  if (documentPart.querySelector('script, iframe')) throw new Error('课程页面含有不受支持的脚本。');
  if (signal.aborted) return () => {};
  container.replaceChildren(...documentPart.body.childNodes);

  const cleanups = [];
  if (entry.kind === 'lesson') {
    const article = container.querySelector('.lesson');
    if (!article) throw new Error('课程内容不完整，请刷新后重试。');
    cleanups.push(mountSteps(article, entry.id));
    if (entry.activity) {
      const module = await import(`/static/js/activities/${entry.activity}.js`);
      if (signal.aborted) return () => cleanups.forEach(cleanup => cleanup());
      for (const slot of article.querySelectorAll(`[data-activity="${entry.activity}"]`)) {
        slot.classList.add('activity-host');
        const cleanup = module.mount(slot);
        if (typeof cleanup === 'function') cleanups.push(cleanup);
      }
    }
  } else {
    const module = await import(entry.module);
    if (signal.aborted) return () => {};
    const cleanup = module.mount(container, {focus: entry.focus, signal});
    if (typeof cleanup === 'function') cleanups.push(cleanup);
  }
  return () => cleanups.forEach(cleanup => cleanup());
}
