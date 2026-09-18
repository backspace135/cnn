function pixels(canvas, matrix, size) {
  if (!Array.isArray(matrix) || matrix.length !== size || matrix.some(row => !Array.isArray(row) || row.length !== size)) return false;
  const context = canvas.getContext('2d');
  if (!context) return false;
  canvas.width = size; canvas.height = size;
  const image = context.createImageData(size, size);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const gray = Math.round(Math.max(0, Math.min(255, Number(matrix[y][x]) || 0)));
    const i = 4 * (y * size + x);
    image.data[i] = image.data[i + 1] = image.data[i + 2] = gray;
    image.data[i + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  return true;
}

function confusion(box, matrix) {
  if (!Array.isArray(matrix) || matrix.length !== 10 || matrix.some(row => !Array.isArray(row) || row.length !== 10)) {
    box.textContent = '完成一轮训练后显示。';
    return;
  }
  const table = document.createElement('table');
  const caption = table.createCaption(); caption.textContent = '行：真实数字；列：预测数字（验证集样本数）';
  const headings = table.insertRow();
  const corner = document.createElement('th'); corner.textContent = '真实 / 预测'; headings.appendChild(corner);
  for (let i = 0; i < 10; i++) {
    const th = document.createElement('th'); th.scope = 'col'; th.textContent = i; headings.appendChild(th);
  }
  matrix.forEach((row, digit) => {
    const tr = table.insertRow();
    const th = document.createElement('th'); th.scope = 'row'; th.textContent = digit; tr.appendChild(th);
    row.forEach((value, prediction) => {
      const td = tr.insertCell(); td.textContent = String(value);
      if (digit === prediction) td.className = 'diagonal';
    });
  });
  box.replaceChildren(table);
}

export function createResults(root) {
  const $ = id => root.querySelector(`#${id}`);
  let featureVersion = null, matrixKey = null;
  return state => {
    const maps = Array.isArray(state.fmaps) ? state.fmaps : [];
    if (!maps.length) {
      if (featureVersion !== null) $('fmaps').textContent = '完成一轮训练后显示。';
      featureVersion = null;
    } else {
      const version = `${state.generation}:${state.fmaps_version}`;
      if (version !== featureVersion) {
        const cells = document.createDocumentFragment();
        maps.forEach((map, i) => {
          const cell = document.createElement('div');
          const canvas = document.createElement('canvas');
          canvas.setAttribute('aria-label', `卷积核 ${i + 1} 的特征图`);
          if (!pixels(canvas, map, 14)) return;
          const label = document.createElement('small'); label.textContent = `核 ${i + 1}`;
          cell.append(canvas, label); cells.appendChild(cell);
        });
        $('fmaps').replaceChildren(cells);
        featureVersion = version;
      }
    }
    const samples = Array.isArray(state.samples) ? state.samples : [];
    if (!samples.length) {
      $('samples').textContent = '暂时没有验证样例；请检查图片数据是否已准备好。';
      $('smax').replaceChildren();
      $('smaxTitle').textContent = '完成一轮训练后显示概率；最高分也不保证一定正确。';
    } else {
      const grid = document.createDocumentFragment();
      samples.forEach(sample => {
        const cell = document.createElement('div'); cell.className = 'training-sample';
        const canvas = document.createElement('canvas');
        canvas.setAttribute('aria-label', `真实数字 ${sample.true} 的验证图片`);
        if (!pixels(canvas, sample.pixels, 28)) return;
        const label = document.createElement('span');
        label.textContent = sample.pred == null ? `真实 ${sample.true} · 待预测` :
          `真实 ${sample.true} · 预测 ${sample.pred} · ${(Number(sample.prob) * 100).toFixed(0)}%`;
        if (sample.pred != null && sample.pred !== sample.true) cell.classList.add('wrong');
        cell.append(canvas, label); grid.appendChild(cell);
      });
      $('samples').replaceChildren(grid);
      const first = samples[0];
      if (first.pred == null || !Array.isArray(first.probs) || !first.probs.length) {
        $('smaxTitle').textContent = '样例已载入；完成一轮训练后显示十类概率。';
        $('smax').replaceChildren();
      } else {
        $('smaxTitle').textContent = `真实 ${first.true}，预测 ${first.pred}；最高分不保证一定正确。`;
        const rows = document.createDocumentFragment();
        first.probs.forEach((probability, digit) => {
          const row = document.createElement('div'); row.className = 'training-prob';
          const label = document.createElement('span'); label.textContent = digit;
          const meter = document.createElement('meter'); meter.min = 0; meter.max = 1;
          meter.value = Math.max(0, Math.min(1, Number(probability) || 0));
          meter.setAttribute('aria-label', `预测为 ${digit} 的概率`);
          const percent = document.createElement('span'); percent.textContent = `${(meter.value * 100).toFixed(1)}%`;
          row.append(label, meter, percent); rows.appendChild(row);
        });
        $('smax').replaceChildren(rows);
      }
    }
    const nextKey = JSON.stringify(state.confusion || []);
    if (nextKey !== matrixKey) { confusion($('confusionMatrix'), state.confusion); matrixKey = nextKey; }
    $('testResult').textContent = Number.isFinite(state.test_acc) && Number.isFinite(state.test_loss)
      ? `完整训练后的独立测试：准确率 ${state.test_acc.toFixed(2)}%，损失 ${state.test_loss.toFixed(4)}。不要反复按测试成绩挑参数。`
      : '独立测试集只在完整训练后评估一次，不能用它反复挑参数。';
    $('log').textContent = Array.isArray(state.log) && state.log.length ? state.log.join('\n') : '等待训练开始…';
  };
}
