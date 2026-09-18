/* 教学用小矩阵：真实训练仍由 train.py 的 PyTorch CNN 完成。 */
(() => {
  const byId = id => document.getElementById(id);
  let result = [], steps = [], current = -1;

  function parseMatrix(text, name) {
    const rows = text.trim().split(/\n/).map(line => line.trim().split(/\s+/).map(Number));
    if (!rows.length || rows.length > 8 || rows.some(row => row.length !== rows[0].length ||
        !row.length || row.length > 8 || row.some(value => !Number.isFinite(value)))) {
      throw new Error(name + '必须是尺寸不超过 8×8 的整齐数字矩阵。');
    }
    return rows;
  }

  function calculate() {
    const input = parseMatrix(byId('convInput').value, '输入');
    const kernel = parseMatrix(byId('convKernel').value, '卷积核');
    if (kernel.length !== 3 || kernel.some(row => row.length !== 3)) {
      throw new Error('本实验请使用 3×3 卷积核。');
    }
    const stride = Number(byId('convStride').value);
    const padding = Number(byId('convPadding').value);
    if (!Number.isInteger(stride) || stride < 1 || stride > 3 ||
        !Number.isInteger(padding) || padding < 0 || padding > 2) {
      throw new Error('步长需为 1–3，填充需为 0–2 的整数。');
    }
    const height = Math.floor((input.length + 2 * padding - 3) / stride) + 1;
    const width = Math.floor((input[0].length + 2 * padding - 3) / stride) + 1;
    if (height < 1 || width < 1 || height * width > 100) {
      throw new Error('卷积核比输入大，或输出过大；请调整步长/填充。');
    }
    result = Array.from({length: height}, () => Array(width).fill(null));
    steps = [];
    for (let r = 0; r < height; r++) for (let c = 0; c < width; c++) {
      const products = [];
      for (let kr = 0; kr < 3; kr++) for (let kc = 0; kc < 3; kc++) {
        const y = r * stride + kr - padding, x = c * stride + kc - padding;
        const pixel = y < 0 || y >= input.length || x < 0 || x >= input[0].length ? 0 : input[y][x];
        products.push([pixel, kernel[kr][kc]]);
      }
      steps.push({r, c, products});
    }
    current = -1;
    render();
  }

  function render() {
    const grid = byId('convResult'); grid.replaceChildren();
    result.forEach((row, r) => {
      const line = document.createElement('div'); line.className = 'matrix-line';
      row.forEach((value, c) => {
        const cell = document.createElement('span'); cell.className = 'matrix-cell';
        if (steps[current]?.r === r && steps[current]?.c === c) cell.classList.add('active');
        cell.textContent = value == null ? '?' : String(value);
        line.appendChild(cell);
      });
      grid.appendChild(line);
    });
    byId('convNext').disabled = current >= steps.length - 1;
  }

  function next() {
    if (current + 1 >= steps.length) return;
    const step = steps[++current];
    const sum = step.products.reduce((total, [pixel, weight]) => total + pixel * weight, 0);
    result[step.r][step.c] = sum;
    byId('convEquation').textContent = `输出 (${step.r + 1},${step.c + 1})：` +
      step.products.map(([pixel, weight]) => `${pixel}×${weight}`).join(' + ') + ` = ${sum}。`;
    render();
  }

  document.addEventListener('DOMContentLoaded', () => {
    byId('convReset').addEventListener('click', () => {
      try { calculate(); byId('convEquation').textContent = '已重新计算。按「下一格」开始。'; }
      catch (error) { byId('convEquation').textContent = error.message; }
    });
    byId('convNext').addEventListener('click', next);
    calculate();
  });
})();
