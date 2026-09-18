const DEFAULTS = {epochs: 6, lr: 0.001, batch_size: 64, dropout: 0.3};

export function readSettings(root) {
  const $ = id => root.querySelector(`#${id}`);
  for (const id of ['epochs', 'lr', 'batch', 'dropout']) {
    if (!$(id).reportValidity()) return null;
  }
  return {epochs: Number($('epochs').value), lr: Number($('lr').value),
    batch_size: Number($('batch').value), dropout: Number($('dropout').value),
    device_mode: $('device').value};
}

export function resetSettings(root, deviceMode) {
  const $ = id => root.querySelector(`#${id}`);
  $('epochs').value = DEFAULTS.epochs;
  $('lr').value = DEFAULTS.lr;
  $('batch').value = DEFAULTS.batch_size;
  $('dropout').value = DEFAULTS.dropout;
  $('device').value = deviceMode;
  return {...DEFAULTS, device_mode: deviceMode};
}

export function renderControls(root, state, pending, firstState) {
  const $ = id => root.querySelector(`#${id}`);
  const status = state.status || 'idle';
  if (firstState && state.config) {
    const config = state.config;
    $('epochs').value = config.epochs ?? DEFAULTS.epochs;
    $('lr').value = config.lr ?? DEFAULTS.lr;
    $('batch').value = config.batch_size ?? DEFAULTS.batch_size;
    $('dropout').value = config.dropout ?? DEFAULTS.dropout;
    $('device').value = config.device_mode || 'auto';
  }
  const hardware = state.hardware || {};
  $('device').querySelector('[value="cuda"]').disabled = !hardware.cuda_available;
  $('deviceHint').textContent = hardware.cuda_available
    ? `可用显卡：${hardware.gpu_name || 'NVIDIA GPU'}。选择后重新训练才会生效。`
    : hardware.cuda_reason || 'CUDA 不可用；可以选择 CPU 模式。';
  const running = status === 'running', paused = status === 'paused';
  const start = root.querySelector('[data-action="start"]');
  const resume = root.querySelector('[data-action="resume"]');
  start.hidden = paused;
  resume.hidden = !paused;
  start.disabled = pending || Boolean(state.model_loaded) || !['idle', 'paused'].includes(status);
  resume.disabled = pending || !paused || Boolean(state.model_loaded);
  root.querySelector('[data-action="pause"]').disabled = pending || !running;
  root.querySelector('[data-action="stop"]').disabled = pending || !(running || paused);
  for (const action of ['reset', 'apply', 'save', 'load']) {
    root.querySelector(`[data-action="${action}"]`).disabled = pending;
  }
}
