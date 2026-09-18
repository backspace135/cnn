/* 只在浏览器中处理原图；向本地 Flask 发送 28×28 的数字矩阵。 */
(() => {
  const $ = id => document.getElementById(id);
  let drawing = false;
  let uploaded = null;
  const board = document.getElementById('drawCanvas');
  const ctx = board.getContext('2d', {willReadFrequently: true});
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 280, 280);
  ctx.strokeStyle = '#fff'; ctx.lineCap = 'round'; ctx.lineJoin = 'round';

  function position(event) {
    const rect = board.getBoundingClientRect();
    return [(event.clientX - rect.left) * board.width / rect.width,
            (event.clientY - rect.top) * board.height / rect.height];
  }
  board.addEventListener('pointerdown', event => {
    board.setPointerCapture(event.pointerId);
    uploaded = null;
    drawing = true; ctx.beginPath(); ctx.moveTo(...position(event));
    ctx.lineWidth = Number($('brushSize').value);
    ctx.lineTo(...position(event)); ctx.stroke();
  });
  board.addEventListener('pointermove', event => {
    if (!drawing) return;
    ctx.lineTo(...position(event));ctx.stroke();
  });
  for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    board.addEventListener(name, () => { drawing = false; });
  }
  $('clearDraw').addEventListener('click', () => {
    uploaded = null; ctx.fillStyle = '#000'; ctx.fillRect(0, 0, 280, 280);
    $('uploadDigit').value = ''; $('predictResult').textContent = '已清空，请重新输入。';
    $('predictProbs').replaceChildren();
  });
  $('uploadDigit').addEventListener('change', async event => {
    const file = event.target.files[0];
    if (!file) return;
    if (!['image/png', 'image/jpeg'].includes(file.type) || file.size > 5 * 1024 * 1024) {
      $('predictResult').textContent = '请选择不超过 5 MB 的 PNG/JPEG 图片。';return;
    }
    const url = URL.createObjectURL(file);
    try {
      const image = new Image(); image.src = url; await image.decode();
      if (image.width * image.height > 4_000_000) throw new Error('图片解码尺寸过大，请先缩小。');
      const source = document.createElement('canvas');
      source.width = image.width; source.height = image.height;
      source.getContext('2d').drawImage(image, 0, 0);
      uploaded = source;
      $('predictResult').textContent = '图片已在本地读取；点击识别查看预处理结果。';
    } catch (error) { uploaded = null; $('predictResult').textContent = error.message; }
    finally { URL.revokeObjectURL(url); }
  });

  function preprocess() {
    const source = uploaded || board;
    const raw = source.getContext('2d', {willReadFrequently: true}).getImageData(0, 0, source.width, source.height).data;
    const inverse = uploaded && $('invertImage').checked;
    const mono = new Float32Array(source.width * source.height);
    let left = source.width, right = -1, top = source.height, bottom = -1;
    for (let y = 0; y < source.height; y++) for (let x = 0; x < source.width; x++) {
      const i = (y * source.width + x) * 4;
      const alpha = raw[i + 3] / 255;
      const gray = (raw[i] * .299 + raw[i + 1] * .587 + raw[i + 2] * .114) / 255;
      const value = (inverse ? 1 - gray : gray) * alpha;
      mono[y * source.width + x] = value;
      if (value > .15) { left = Math.min(left, x); right = Math.max(right, x);
        top = Math.min(top, y); bottom = Math.max(bottom, y); }
    }
    if (right < left || bottom < top) throw new Error('输入为空，请写一个数字或选择图片。');
    const cropped = document.createElement('canvas');
    cropped.width = right - left + 1; cropped.height = bottom - top + 1;
    const pixels = cropped.getContext('2d').createImageData(cropped.width, cropped.height);
    for (let y = 0; y < cropped.height; y++) for (let x = 0; x < cropped.width; x++) {
      const i = (y * cropped.width + x) * 4;
      const v = Math.round(mono[(y + top) * source.width + x + left] * 255);
      pixels.data[i] = pixels.data[i + 1] = pixels.data[i + 2] = v; pixels.data[i + 3] = 255;
    }
    cropped.getContext('2d').putImageData(pixels, 0, 0);
    const target = document.createElement('canvas'); target.width = target.height = 28;
    const tctx = target.getContext('2d', {willReadFrequently: true});
    tctx.fillStyle = '#000';tctx.fillRect(0, 0, 28, 28);
    const scale = 20 / Math.max(cropped.width, cropped.height);
    const w = cropped.width * scale, h = cropped.height * scale;
    tctx.drawImage(cropped, (28 - w) / 2, (28 - h) / 2, w, h);
    const out = tctx.getImageData(0, 0, 28, 28).data;
    const matrix = Array.from({length: 28}, (_, y) => Array.from({length: 28}, (_, x) =>
      Math.round(out[(y * 28 + x) * 4] / 255 * 10000) / 10000));
    pixToCanvas($('inputPreview'), matrix.map(row => row.map(value => value * 255)));
    return matrix;
  }

  $('predictButton').addEventListener('click', async () => {
    const button = $('predictButton'); button.disabled = true;
    try {
      const pixels = preprocess();
      const response = await fetch('/api/predict', {method: 'POST',
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify({pixels})});
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error(data.error?.message || data.msg || '识别失败');
      $('predictResult').textContent = `预测：${data.pred} · 最高概率 ${(data.prob * 100).toFixed(1)}%。` +
        (data.trained ? '概率并不保证一定正确。' : '当前模型尚未训练，这只是随机初始化的判断。');
      const box = $('predictProbs');box.replaceChildren();
      data.probs.forEach((value, digit) => {
        const row = document.createElement('div');row.className = 'srow';
        const label = document.createElement('span');label.textContent = digit;
        const barbox = document.createElement('div');barbox.className = 'barbox';
        const bar = document.createElement('div');bar.className = 'bar';
        bar.style.width = (100 * value).toFixed(1) + '%';barbox.appendChild(bar);
        const pct = document.createElement('span');pct.className = 'pct';pct.textContent = (100 * value).toFixed(1) + '%';
        row.append(label, barbox, pct);box.appendChild(row);
      });
    } catch (error) { $('predictResult').textContent = error.message; }
    finally { button.disabled = false; }
  });

  window.modelAction = async action => {
    try {
      const response = await fetch('/api/model', {method: 'POST',
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action})});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error?.message || result.msg || '操作失败');
      $('actionNotice').textContent = action === 'save' ? '模型已保存到本地 data/checkpoints/latest.pt。' :
        '已从本地存档加载模型。可到「手写识别」检验；这不是训练断点续训。';
    } catch (error) { $('actionNotice').textContent = '模型操作失败：' + error.message; }
    $('actionNotice').hidden = false;
  };
})();
