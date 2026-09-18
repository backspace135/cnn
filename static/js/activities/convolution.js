/* 小矩阵仅用于教学演示；与训练模型无关。 */
export function parseMatrix(text, name) {
  const rows = text.trim().split(/\n/).map(line => line.trim().split(/\s+/).map(Number));
  if (!text.trim() || rows.length > 8 || rows.some(row => !row.length || row.length > 8 ||
      row.length !== rows[0].length || row.some(value => !Number.isFinite(value))) ||
      text.trim().split(/\n/).some(line => !line.trim())) {
    throw new Error(`${name}必须是尺寸不超过 8×8 的整齐数字矩阵。`);
  }
  return rows;
}

export function computeConvolution(input, kernel, stride = 1, padding = 0) {
  if (!Array.isArray(input) || !input.length || input.length > 8 ||
      !Array.isArray(input[0]) || !input[0].length || input[0].length > 8 ||
      input.some(row => !Array.isArray(row) || row.length !== input[0].length ||
        row.some(value => typeof value !== 'number' || !Number.isFinite(value)))) {
    throw new Error('输入必须是尺寸不超过 8×8 的整齐数字矩阵。');
  }
  if (!Array.isArray(kernel) || kernel.length !== 3 ||
      kernel.some(row => !Array.isArray(row) || row.length !== 3 ||
        row.some(value => typeof value !== 'number' || !Number.isFinite(value)))) {
    throw new Error('本实验请使用 3×3 数字卷积核。');
  }
  if (!Number.isInteger(stride) || stride < 1 || stride > 3 ||
      !Number.isInteger(padding) || padding < 0 || padding > 2) {
    throw new Error('步长需为 1–3，填充需为 0–2 的整数。');
  }
  const height = Math.floor((input.length + 2 * padding - 3) / stride) + 1;
  const width = Math.floor((input[0].length + 2 * padding - 3) / stride) + 1;
  if (height < 1 || width < 1 || height * width > 100) {
    throw new Error('卷积核比输入大，或输出过大；请调整步长/填充。');
  }
  const steps = [];
  for (let r = 0; r < height; r++) for (let c = 0; c < width; c++) {
    const products = [];
    for (let kr = 0; kr < 3; kr++) for (let kc = 0; kc < 3; kc++) {
      const y = r * stride + kr - padding, x = c * stride + kc - padding;
      const pixel = y < 0 || y >= input.length || x < 0 || x >= input[0].length ? 0 : input[y][x];
      products.push([pixel, kernel[kr][kc]]);
    }
    steps.push({r, c, products, sum: products.reduce((total, [pixel, weight]) => total + pixel * weight, 0)});
  }
  return {height, width, steps};
}

export function mount(slot) {
  const input = [[1, 2, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]];
  const initialKernel = [[1, 0, -1], [0, 1, 0], [1, 0, 0]];
  const root = document.createElement('div');
  const add = (parent, tag, text) => {
    const element = document.createElement(tag);
    element.textContent = text;
    parent.appendChild(element);
    return element;
  };

  add(root, 'h3', '先算第一格：对应位置相乘，再相加');
  add(root, 'p', '固定的 3×3 输入与卷积核（实际为不翻转核的互相关）：');
  const example = add(root, 'pre',
    `输入：\n${input.slice(0, 3).map(row => row.slice(0, 3).join('  ')).join('\n')}\n\n卷积核：\n${initialKernel.map(row => row.join('  ')).join('\n')}`);
  example.style.overflowX = 'auto';
  const fixedStep = computeConvolution(input.slice(0, 3).map(row => row.slice(0, 3)), initialKernel).steps[0];
  add(root, 'p', `第一格：${fixedStep.products.map(([pixel, weight]) => `${pixel}×(${weight})`).join(' + ')} = ${fixedStep.sum}。`);
  add(root, 'p', '现在让同一个核逐格向右、向下滑动；超出输入的格子用 0 填充。');
  const inputGrid = add(root, 'div', '');
  inputGrid.setAttribute('role', 'group');
  inputGrid.setAttribute('aria-label', '固定输入矩阵，高亮显示当前卷积窗口');
  inputGrid.style.display = 'grid';
  inputGrid.style.gridTemplateColumns = `repeat(${input[0].length}, minmax(2rem, 3rem))`;
  inputGrid.style.gap = '.25rem';
  const inputCells = input.flatMap((row, r) => row.map((value, c) => {
    const cell = add(inputGrid, 'span', String(value));
    cell.setAttribute('aria-label', `第 ${r + 1} 行第 ${c + 1} 列：${value}`);
    cell.style.padding = '.35rem';
    cell.style.textAlign = 'center';
    cell.style.border = '1px solid currentColor';
    return cell;
  }));
  const kernelLabel = add(root, 'label', '卷积核（每行空格分隔）');
  const kernelField = add(kernelLabel, 'textarea', initialKernel.map(row => row.join(' ')).join('\n'));
  kernelField.rows = 3;
  kernelField.style.display = 'block';
  kernelField.setAttribute('aria-label', '卷积核矩阵');
  const strideLabel = add(root, 'label', '步长 ');
  const strideField = add(strideLabel, 'input', '');
  strideField.type = 'number'; strideField.min = '1'; strideField.max = '3'; strideField.value = '1';
  const paddingLabel = add(root, 'label', ' 填充 ');
  const paddingField = add(paddingLabel, 'input', '');
  paddingField.type = 'number'; paddingField.min = '0'; paddingField.max = '2'; paddingField.value = '0';
  const reset = add(root, 'button', '重新计算'); reset.type = 'button';
  const next = add(root, 'button', '下一格'); next.type = 'button';
  const explanation = add(root, 'p', '');
  explanation.setAttribute('role', 'status');
  const output = add(root, 'div', '');
  output.setAttribute('role', 'group');
  output.setAttribute('aria-label', '卷积输出矩阵，问号表示尚未计算');
  slot.appendChild(root);

  let calculation, current = -1, stepStride = 1, stepPadding = 0;
  function render() {
    output.replaceChildren();
    if (!calculation) {
      inputCells.forEach(cell => { cell.style.backgroundColor = ''; });
      next.disabled = true;
      return;
    }
    for (let r = 0; r < calculation.height; r++) {
      const line = add(output, 'div', ''); line.className = 'matrix-line';
      for (let c = 0; c < calculation.width; c++) {
        const index = r * calculation.width + c;
        const cell = add(line, 'span', index <= current ? String(calculation.steps[index].sum) : '?');
        cell.className = 'matrix-cell';
        if (index === current) cell.classList.add('active');
      }
    }
    inputCells.forEach((cell, index) => {
      const r = Math.floor(index / input[0].length), c = index % input[0].length;
      const step = calculation.steps[current];
      const y = step ? step.r * stepStride - stepPadding : -3;
      const x = step ? step.c * stepStride - stepPadding : -3;
      cell.style.backgroundColor = r >= y && r < y + 3 && c >= x && c < x + 3 ? 'rgba(98, 128, 218, .25)' : '';
    });
    next.disabled = current >= calculation.steps.length - 1;
  }
  for (const field of [kernelField, strideField, paddingField]) {
    field.addEventListener('input', () => {
      calculation = null;
      current = -1;
      explanation.textContent = '参数已修改，请按「重新计算」。';
      render();
    });
  }
  reset.addEventListener('click', () => {
    try {
      const kernel = parseMatrix(kernelField.value, '卷积核');
      if (!strideField.value.trim() || !paddingField.value.trim()) {
        throw new Error('步长和填充均需填写整数。');
      }
      stepStride = Number(strideField.value);
      stepPadding = Number(paddingField.value);
      calculation = computeConvolution(input, kernel, stepStride, stepPadding);
      current = -1;
      explanation.textContent = '已重新计算。按「下一格」观察每一步。';
    } catch (error) {
      calculation = null;
      current = -1;
      explanation.textContent = error.message;
    }
    render();
  });
  next.addEventListener('click', () => {
    if (!calculation || current + 1 >= calculation.steps.length) return;
    const step = calculation.steps[++current];
    explanation.textContent = `输出 (${step.r + 1},${step.c + 1})：` +
      step.products.map(([pixel, weight]) => `${pixel}×(${weight})`).join(' + ') + ` = ${step.sum}。`;
    render();
  });
  reset.click();
  return () => root.remove();
}
