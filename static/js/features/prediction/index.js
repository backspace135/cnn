import {request} from '../../core/api.js';
import {setupBoard} from './board.js';
import {preprocess, renderPreview} from './preprocess.js';
import {showPrediction} from './results.js';

export function mount(root) {
  const $ = id => root.querySelector(`#${id}`);
  const board = $('drawCanvas');
  const brush = $('brushSize');
  const upload = $('uploadDigit');
  const invert = $('invertImage');
  const preview = $('inputPreview');
  const button = $('predictButton');
  const result = $('predictResult');
  const probabilities = $('predictProbs');
  const listeners = new AbortController();
  const objectURLs = new Set();
  let uploaded = null;
  let pending = null;
  let version = 0;
  let uploadVersion = 0;
  let decoding = false;
  let active = true;

  function invalidate() {
    version++;
    pending?.abort();
    pending = null;
    button.disabled = decoding;
    probabilities.replaceChildren();
    const context = preview.getContext('2d');
    context.fillStyle = '#000';
    context.fillRect(0, 0, 28, 28);
  }

  function updatePreview() {
    try {
      renderPreview(preview, preprocess(uploaded || board, Boolean(uploaded && invert.checked)));
      return true;
    } catch (error) {
      result.textContent = error.message;
      return false;
    }
  }

  const drawing = setupBoard(board, brush, () => {
    uploaded = null;
    uploadVersion++;
    decoding = false;
    upload.value = '';
    invalidate();
    result.textContent = '已修改画板；点击识别查看结果。';
  }, listeners.signal);
  invalidate();
  board.addEventListener('pointerup', () => {
    if (active && updatePreview()) result.textContent = '输入预览已更新；确认笔画后再点击识别。';
  }, {signal: listeners.signal});

  $('clearDraw').addEventListener('click', () => {
    uploaded = null;
    uploadVersion++;
    decoding = false;
    upload.value = '';
    drawing.clear();
    invalidate();
    result.textContent = '已清空，请重新输入。';
  }, {signal: listeners.signal});

  invert.addEventListener('change', () => {
    invalidate();
    if (decoding) result.textContent = '正在读取图片，完成后可按新的反色设置识别。';
    else if (updatePreview()) result.textContent = '反色设置已更新；请先检查输入预览，再点击识别。';
  }, {signal: listeners.signal});

  upload.addEventListener('change', async event => {
    const file = event.target.files?.[0];
    if (!file) return;
    uploaded = null;
    const current = ++uploadVersion;
    decoding = false;
    invalidate();
    if (!['image/png', 'image/jpeg'].includes(file.type) || file.size > 5 * 1024 * 1024) {
      result.textContent = '请选择不超过 5 MB 的 PNG/JPEG 图片。';
      return;
    }
    decoding = true;
    button.disabled = true;
    const url = URL.createObjectURL(file);
    objectURLs.add(url);
    try {
      const image = new Image();
      image.src = url;
      await image.decode();
      if (!active || current !== uploadVersion) return;
      if (image.width * image.height > 4_000_000) throw new Error('图片解码尺寸过大，请先缩小。');
      const source = root.ownerDocument.createElement('canvas');
      source.width = image.width;
      source.height = image.height;
      source.getContext('2d').drawImage(image, 0, 0);
      uploaded = source;
      if (updatePreview()) result.textContent = '图片已在本地读取；请先检查输入预览，再点击识别。';
    } catch (error) {
      if (active && current === uploadVersion) {
        uploaded = null;
        result.textContent = error.message || '图片读取失败。';
      }
    } finally {
      URL.revokeObjectURL(url);
      objectURLs.delete(url);
      if (active && current === uploadVersion) {
        decoding = false;
        button.disabled = false;
      }
    }
  }, {signal: listeners.signal});

  button.addEventListener('click', async () => {
    const current = version;
    button.disabled = true;
    let controller;
    try {
      const pixels = preprocess(uploaded || board, Boolean(uploaded && invert.checked));
      renderPreview(preview, pixels);
      result.textContent = '正在识别…';
      controller = new AbortController();
      pending = controller;
      const data = await request('/api/predict', {method: 'POST', body: {pixels}, signal: controller.signal});
      if (!active || current !== version || controller.signal.aborted) return;
      if (!data?.ok) throw new Error(data?.error?.message || data?.msg || '识别失败');
      showPrediction(result, probabilities, data);
    } catch (error) {
      if (active && current === version && !controller?.signal.aborted) {
        result.textContent = error.message || '识别失败，请稍后重试。';
      }
    } finally {
      if (pending === controller) pending = null;
      if (active && current === version) button.disabled = false;
    }
  }, {signal: listeners.signal});

  return () => {
    active = false;
    version++;
    uploadVersion++;
    listeners.abort();
    pending?.abort();
    objectURLs.forEach(url => URL.revokeObjectURL(url));
    objectURLs.clear();
    drawing.cleanup();
  };
}
