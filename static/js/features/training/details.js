export async function loadModelInfo(root, request, signal) {
  const box = root.querySelector('#modelLayers');
  try {
    const data = await request('/api/model-info', {signal});
    if (signal.aborted) return;
    if (!Array.isArray(data.layers) || !data.layers.length) {
      box.textContent = '当前没有可显示的模型层信息。'; return;
    }
    const table = document.createElement('table');
    const header = table.insertRow();
    for (const title of ['层', '输出形状（首维为每批图片数）', '可学习参数']) {
      const th = document.createElement('th'); th.scope = 'col'; th.textContent = title; header.appendChild(th);
    }
    data.layers.forEach(layer => {
      const row = table.insertRow();
      for (const value of [layer.layer, Array.isArray(layer.shape) ? layer.shape.join(' × ') : '—',
        Number(layer.parameters).toLocaleString()]) row.insertCell().textContent = value ?? '—';
    });
    const summary = document.createElement('p'); summary.textContent = data.summary || '';
    box.replaceChildren(table, summary);
  } catch (error) {
    if (!signal.aborted) box.textContent = `模型结构暂时无法读取：${error.message}。请检查数据后重新打开本页。`;
  }
}

export async function loadExperiments(root, request, signal) {
  const box = root.querySelector('#experimentHistory');
  try {
    const data = await request('/api/experiments', {signal});
    if (signal.aborted) return;
    const entries = Array.isArray(data.experiments) ? data.experiments : [];
    if (!entries.length) { box.textContent = '暂无完整运行。完成一次训练后再来比较。'; return; }
    const table = document.createElement('table');
    const header = table.insertRow();
    for (const title of ['运行', '轮数', '学习率', '批大小', '丢弃率', '验证准确率', '验证损失']) {
      const th = document.createElement('th'); th.scope = 'col'; th.textContent = title; header.appendChild(th);
    }
    entries.slice().reverse().forEach(entry => {
      const row = table.insertRow(), config = entry.config || {};
      for (const value of [entry.id, config.epochs, config.lr, config.batch_size, config.dropout,
        Number.isFinite(entry.validation_acc) ? `${entry.validation_acc.toFixed(2)}%` : '—',
        Number.isFinite(entry.validation_loss) ? entry.validation_loss.toFixed(4) : '—']) {
        row.insertCell().textContent = value ?? '—';
      }
    });
    box.replaceChildren(table);
  } catch (error) {
    if (!signal.aborted) box.textContent = `实验记录暂时无法读取：${error.message}。请稍后重新打开本页。`;
  }
}
