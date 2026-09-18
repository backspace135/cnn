import {request} from '../../core/api.js';
import {readSettings, resetSettings, renderControls} from './controls.js';
import {renderCharts} from './charts.js';
import {createResults} from './results.js';
import {loadModelInfo, loadExperiments} from './details.js';

const STATUS = {idle: '尚未开始', running: '训练中', paused: '已暂停', done: '训练完成',
  cancelled: '已停止', error: '训练出错'};

function duration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return '—';
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
}

export function mount(root, {focus} = {}) {
  const $ = id => root.querySelector(`#${id}`);
  const abort = new AbortController();
  const {signal} = abort;
  const renderResults = createResults(root);
  let state = null, pending = false, timer = null, disposed = false, actionError = false, modelNotice = false;
  let historyPending = false, nextHistoryAt = 0, historyAttempts = 0;

  function refreshExperiments(force = false) {
    if (disposed || historyPending || (!force && (historyAttempts >= 10 || Date.now() < nextHistoryAt))) return;
    historyPending = true;
    historyAttempts++;
    nextHistoryAt = Date.now() + 1500;
    // done 可能先于本地历史落盘；有限次数、单飞重试，离页时取消。
    void loadExperiments(root, request, signal).finally(() => { historyPending = false; });
  }

  function notice(message) {
    $('notice').textContent = message;
    $('notice').hidden = !message;
  }

  function render(next) {
    const previous = state;
    state = next;
    const status = next.status || 'idle';
    $('status').textContent = STATUS[status] || '状态未知';
    const sizes = next.dataset_sizes;
    $('dataset').textContent = sizes
      ? `训练图片 ${Number(sizes.train).toLocaleString()} 张 · 验证图片 ${Number(sizes.validation).toLocaleString()} 张 · 独立测试图片 ${Number(sizes.test).toLocaleString()} 张`
      : '图片数据尚未载入；请先到“开始使用”准备 MNIST，再回本页重试。';
    if (status === 'error') notice(`训练出错：${next.error || '请展开训练日志检查原因。'}`);
    else if (!actionError && !modelNotice) notice('');
    $('deviceLabel').textContent = next.config?.device === 'cuda'
      ? `CUDA · ${next.hardware?.gpu_name || 'NVIDIA GPU'}`
      : next.config?.device === 'cpu' ? 'CPU' : '检测中…';
    const loss = Number(next.train_loss_cur);
    $('loss').textContent = next.train_loss_cur == null || !Number.isFinite(loss) ? '—' : loss.toFixed(4);
    const accuracies = Array.isArray(next.epochs_val_acc) ? next.epochs_val_acc : [];
    $('accuracy').textContent = accuracies.length ? `${Number(accuracies.at(-1)).toFixed(2)}%` : '—';
    $('time').textContent = `${duration(next.elapsed_sec)} / ${duration(next.eta_sec)}`;
    const total = Number(next.epoch_total) * Number(next.steps_per_epoch);
    const percent = total > 0 ? Math.min(100, Math.max(0, 100 * (Number(next.step) || 0) / total)) : 0;
    $('progressBar').value = percent;
    $('progress').textContent = `第 ${next.epoch_current || 0} / ${next.epoch_total || 0} 轮 · 已更新 ${next.step || 0} 步 · 完成 ${percent.toFixed(0)}%`;
    renderControls(root, next, pending, !previous);
    renderCharts(root, next);
    renderResults(next);
    if (status === 'done') {
      if (!previous || previous.status !== 'done') {
        historyAttempts = 0;
        nextHistoryAt = 0;
      }
      refreshExperiments();
    }
  }

  // 一个状态请求完成后再等 700ms；不会因网络变慢而叠加请求。
  async function poll() {
    if (disposed) return;
    try {
      const next = await request('/api/state', {signal});
      if (!disposed) render(next);
    } catch (error) {
      if (!disposed) {
        $('status').textContent = '连接中断，正在重试…';
        if (!actionError && !modelNotice) notice(`暂时无法取得训练状态：${error.message}。将自动重试。`);
      }
    } finally {
      if (!disposed) timer = setTimeout(poll, 700);
    }
  }

  async function onClick(event) {
    const button = event.target.closest('button[data-action]');
    if (!button || !root.contains(button) || pending || button.disabled) return;
    const action = button.dataset.action;
    if (action === 'refresh-history') {
      refreshExperiments(true);
      return;
    }
    let body;
    if (action === 'apply') {
      const settings = readSettings(root);
      if (!settings) return;
      body = {action: 'reset', ...settings};
      $('settingsMessage').textContent = '正在应用设置并从头训练…';
    } else if (action === 'reset') {
      body = {action: 'reset', ...resetSettings(root, state?.config?.device_mode || 'auto')};
    } else body = {action};
    pending = true; actionError = false; modelNotice = false; notice('');
    if (state) renderControls(root, state, pending, false);
    else root.querySelectorAll('button[data-action]').forEach(item => { item.disabled = true; });
    try {
      const modelAction = action === 'save' || action === 'load';
      const data = await request(modelAction ? '/api/model' : '/api/control',
        {method: 'POST', body, signal});
      if (disposed) return;
      if (data?.ok === false) throw new Error(data.msg || data.error?.message || '操作未成功');
      if (modelAction) {
        modelNotice = true;
        notice(action === 'save'
          ? '模型已保存到本地 data/checkpoints/latest.pt。'
          : '已加载本地存档，可用于手写识别；这不是训练断点续训。');
      }
      if (action === 'apply') $('settingsMessage').textContent = '已应用设置，正在重新训练。';
    } catch (error) {
      if (disposed) return;
      actionError = true;
      notice(`操作失败：${error.message}。检查设置或连接后可再次点击重试。`);
      if (action === 'apply') $('settingsMessage').textContent = '应用失败，请检查提示后重试。';
    } finally {
      if (!disposed) {
        pending = false;
        if (state) renderControls(root, state, pending, false);
        else root.querySelectorAll('button[data-action]').forEach(item => { item.disabled = false; });
      }
    }
  }

  root.addEventListener('click', onClick);
  if (focus === 'lab' || focus === 'settings') {
    $('settings').focus({preventScroll: true});
    $('settings').scrollIntoView();
  }
  void poll();
  refreshExperiments(true);
  void loadModelInfo(root, request, signal);
  return () => {
    disposed = true;
    clearTimeout(timer);
    abort.abort();
    root.removeEventListener('click', onClick);
  };
}
