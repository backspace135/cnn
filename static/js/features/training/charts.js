const SVG = 'http://www.w3.org/2000/svg';

function node(parent, tag, attributes, label) {
  const element = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  if (label !== undefined) element.textContent = String(label);
  parent.appendChild(element);
  return element;
}

// 横轴是步数或轮数；每种指标各用一张图，避免双轴误导。
function lineChart(svg, x, series, accuracy = false) {
  svg.replaceChildren();
  const values = series.flatMap(item => item.values).filter(Number.isFinite);
  const left = 44, right = 468, top = 16, bottom = 180;
  if (!x.length || !values.length) {
    node(svg, 'text', {x: 240, y: 105, fill: '#566477', 'text-anchor': 'middle'},
      accuracy ? '完成第一轮训练后显示准确率' : '开始训练后显示损失');
    return;
  }
  const min = 0, max = accuracy ? 100 : Math.max(1, ...values) * 1.08;
  const pointX = i => left + (right - left) * (x.length === 1 ? 0.5 : (x[i] - x[0]) / (x[x.length - 1] - x[0] || 1));
  const pointY = value => bottom - (bottom - top) * (value - min) / (max - min);
  for (let tick = 0; tick <= 4; tick++) {
    const value = max * tick / 4, y = pointY(value);
    node(svg, 'line', {x1: left, x2: right, y1: y, y2: y, stroke: '#d9e0ea'});
    node(svg, 'text', {x: left - 6, y: y + 4, fill: '#566477', 'text-anchor': 'end', 'font-size': 11},
      accuracy ? `${value.toFixed(0)}%` : value.toFixed(value < 1 ? 2 : 1));
  }
  for (const i of new Set([0, Math.floor((x.length - 1) / 2), x.length - 1])) {
    node(svg, 'text', {x: pointX(i), y: 201, fill: '#566477', 'text-anchor': 'middle', 'font-size': 11}, x[i]);
  }
  for (const item of series) {
    let segment = [];
    const flush = () => {
      if (segment.length) {
        const d = segment.map(([px, py], i) => `${i ? 'L' : 'M'}${px.toFixed(2)} ${py.toFixed(2)}`).join(' ');
        const path = node(svg, 'path', {d, fill: 'none', stroke: item.color, 'stroke-width': item.width,
          'stroke-linecap': 'round', 'stroke-linejoin': 'round'});
        node(path, 'title', {}, item.name);
        const hit = node(svg, 'path', {d, fill: 'none', stroke: 'transparent', 'stroke-width': 14});
        node(hit, 'title', {}, `${item.name}，详细数值见下方数据表`);
        if (segment.length === 1) node(svg, 'circle', {cx: segment[0][0], cy: segment[0][1], r: 4, fill: item.color});
        segment = [];
      }
    };
    x.forEach((_, i) => {
      if (Number.isFinite(item.values[i])) segment.push([pointX(i), pointY(item.values[i])]);
      else flush();
    });
    flush();
  }
}

function bars(svg, values) {
  svg.replaceChildren();
  if (!values.length) {
    node(svg, 'text', {x: 240, y: 95, fill: '#566477', 'text-anchor': 'middle'}, '完成一轮训练后显示各类准确率');
    return;
  }
  values.slice(0, 10).forEach((value, digit) => {
    const pct = Math.max(0, Math.min(100, Number(value) || 0));
    const height = pct * 1.2, x = 22 + digit * 44;
    const bar = node(svg, 'rect', {x, y: 150 - height, width: 26, height, rx: 4, fill: '#355ed3'});
    node(bar, 'title', {}, `数字 ${digit}：${pct.toFixed(1)}%`);
    node(svg, 'text', {x: x + 13, y: 170, fill: '#566477', 'text-anchor': 'middle', 'font-size': 11}, digit);
  });
}

function dataTable(captionText, headings, rows) {
  const table = document.createElement('table');
  table.createCaption().textContent = captionText;
  const header = table.insertRow();
  headings.forEach(label => {
    const th = document.createElement('th'); th.scope = 'col'; th.textContent = label; header.appendChild(th);
  });
  rows.forEach(values => {
    const row = table.insertRow();
    values.forEach(value => { row.insertCell().textContent = value ?? '—'; });
  });
  return table;
}

export function renderCharts(root, state) {
  const $ = id => root.querySelector(`#${id}`);
  const history = Array.isArray(state.loss_history) ? state.loss_history : [];
  const stride = Math.max(1, Math.ceil(history.length / 600));
  const x = [], raw = [], smooth = [];
  for (let i = 0; i < history.length; i += stride) {
    x.push(state.loss_steps?.[i] ?? i); raw.push(history[i]); smooth.push(state.smooth_loss?.[i] ?? null);
  }
  lineChart($('lossChart'), x, [
    {name: '逐批损失', values: raw, color: '#8a97ad', width: 1.5},
    {name: '平滑损失', values: smooth, color: '#355ed3', width: 2.4},
  ]);
  const validation = Array.isArray(state.epochs_val_acc) ? state.epochs_val_acc : [];
  lineChart($('accChart'), validation.map((_, i) => i + 1), [
    {name: '训练准确率', values: state.epochs_train_acc || [], color: '#8a97ad', width: 2},
    {name: '验证准确率', values: validation, color: '#355ed3', width: 2.4},
  ], true);
  bars($('accBars'), Array.isArray(state.class_acc) ? state.class_acc : []);
  const classes = Array.isArray(state.class_acc) ? state.class_acc : [];
  const table = $('chartTable');
  if (!history.length && !validation.length && !classes.length) {
    table.textContent = '开始训练后显示每批损失、每轮准确率和各数字准确率。';
    return;
  }
  const lastPoints = x.map((step, i) => [step,
    Number.isFinite(raw[i]) ? raw[i].toFixed(4) : '—',
    Number.isFinite(smooth[i]) ? smooth[i].toFixed(4) : '—']).slice(-30);
  const tables = [];
  if (lastPoints.length) tables.push(dataTable('最近 30 个损失数据点（长曲线已抽样）',
    ['步数', '每批损失', '平滑损失'], lastPoints));
  if (validation.length) tables.push(dataTable('每轮训练与验证准确率',
    ['轮数', '训练集', '验证集'], validation.map((value, i) => [i + 1,
      Number.isFinite(state.epochs_train_acc?.[i]) ? `${state.epochs_train_acc[i].toFixed(2)}%` : '—',
      Number.isFinite(value) ? `${value.toFixed(2)}%` : '—'])));
  if (classes.length) tables.push(dataTable('各数字验证准确率', ['数字', '准确率'],
    classes.slice(0, 10).map((value, i) => [i, Number.isFinite(value) ? `${value.toFixed(2)}%` : '—'])));
  table.replaceChildren(...tables);
}
