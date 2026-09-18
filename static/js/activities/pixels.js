/* 点击灰度小格，观察图片怎样变成 0～1 的数字矩阵。 */
export function nextPixelValue(value) {
  return value === 0 ? 0.5 : value === 0.5 ? 1 : 0;
}

export function mount(slot) {
  const size = 5;
  const values = Array.from({length: size}, () => Array(size).fill(0));
  const root = document.createElement('div');
  const instruction = document.createElement('p');
  instruction.textContent = '点击小格（也可按 Enter 或空格），依次切换 0（黑）、0.5（灰）、1（白）。右侧数字表同步变化。';
  root.appendChild(instruction);

  const layout = document.createElement('div');
  layout.style.display = 'flex';
  layout.style.flexWrap = 'wrap';
  layout.style.gap = '1.5rem';
  root.appendChild(layout);
  const grid = document.createElement('div');
  grid.setAttribute('role', 'group');
  grid.setAttribute('aria-label', '可编辑的五行五列灰度像素');
  grid.style.display = 'grid';
  grid.style.gridTemplateColumns = `repeat(${size}, 2.5rem)`;
  grid.style.gap = '.2rem';
  layout.appendChild(grid);

  const table = document.createElement('table');
  const caption = document.createElement('caption');
  caption.textContent = '对应的数字矩阵（0～1）';
  table.appendChild(caption);
  const body = document.createElement('tbody');
  table.appendChild(body);
  layout.appendChild(table);
  for (let r = 0; r < size; r++) {
    const row = document.createElement('tr');
    body.appendChild(row);
    for (let c = 0; c < size; c++) {
      const button = document.createElement('button');
      button.type = 'button';
      button.style.width = '2.5rem';
      button.style.height = '2.5rem';
      button.style.border = '1px solid currentColor';
      button.style.cursor = 'pointer';
      grid.appendChild(button);
      const number = document.createElement('td');
      number.style.padding = '.3rem';
      number.style.textAlign = 'center';
      row.appendChild(number);
      const update = () => {
        const value = values[r][c];
        const gray = Math.round(value * 255);
        button.style.backgroundColor = `rgb(${gray}, ${gray}, ${gray})`;
        button.setAttribute('aria-label', `第 ${r + 1} 行第 ${c + 1} 列，${value === 0.5 ? '灰' : value === 0 ? '黑' : '白'}，数值 ${value}`);
        number.textContent = String(value);
      };
      button.addEventListener('click', () => {
        values[r][c] = nextPixelValue(values[r][c]);
        update();
      });
      update();
    }
  }
  slot.appendChild(root);
  return () => root.remove();
}
