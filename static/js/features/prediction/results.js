export function showPrediction(result, box, data) {
  result.textContent = `预测：${data.pred} · 最高概率 ${(data.prob * 100).toFixed(1)}%。` +
    (data.trained ? '概率并不保证一定正确。' : '当前模型尚未训练，这只是随机初始化的判断。');
  box.replaceChildren();
  data.probs.forEach((value, digit) => {
    const row = box.ownerDocument.createElement('div');
    row.className = 'probability-row';
    row.title = `数字 ${digit}：${(100 * value).toFixed(1)}%`;
    const label = box.ownerDocument.createElement('span');
    label.textContent = digit;
    const track = box.ownerDocument.createElement('div');
    track.className = 'probability-track';
    const bar = box.ownerDocument.createElement('div');
    bar.className = 'probability-bar';
    bar.style.width = (100 * value).toFixed(1) + '%';
    track.appendChild(bar);
    const percent = box.ownerDocument.createElement('span');
    percent.className = 'probability-value';
    percent.textContent = (100 * value).toFixed(1) + '%';
    row.append(label, track, percent);
    box.appendChild(row);
  });
}
