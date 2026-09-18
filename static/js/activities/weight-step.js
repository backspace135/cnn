// 固定的小算例：只试着改权重、比较误差，不需要导数。
export function mount(slot) {
  const title = document.createElement('h3');
  title.textContent = '试一小步：怎样让结果更接近目标？';
  const intro = document.createElement('p');
  intro.textContent = '固定输入 x = 2、目标 = 4；输出 = x × w，平方误差 = (输出 − 目标) × (输出 − 目标)。每次只改权重 w。';
  const result = document.createElement('p');
  result.setAttribute('aria-live', 'polite');
  const comparison = document.createElement('p');
  const down = document.createElement('button');
  down.type = 'button';
  down.textContent = 'w −0.5，试试看';
  const up = document.createElement('button');
  up.type = 'button';
  up.textContent = 'w +0.5，试试看';
  const reset = document.createElement('button');
  reset.type = 'button';
  reset.textContent = '从头再试';
  let weight = 1;
  function errorFor(value) {
    const difference = 2 * value - 4;
    return difference * difference;
  }
  function render() {
    result.textContent = `x = 2，w = ${weight}，输出 = 2 × ${weight} = ${2 * weight}，目标 = 4，平方误差 = ${errorFor(weight)}。`;
  }
  function tryStep(amount) {
    const previous = errorFor(weight);
    weight += amount;
    render();
    const current = errorFor(weight);
    comparison.textContent = `这一步：平方误差从 ${previous} 变为 ${current}，${current < previous ? '更接近目标' : current > previous ? '更远离目标' : '距离不变'}。你还可以反方向试。`;
  }
  const tryDown = () => tryStep(-0.5);
  const tryUp = () => tryStep(0.5);
  function startOver() {
    weight = 1;
    render();
    comparison.textContent = '先猜一猜：权重加一点，还是减一点，会让误差变小？';
  }
  down.addEventListener('click', tryDown);
  up.addEventListener('click', tryUp);
  reset.addEventListener('click', startOver);
  slot.replaceChildren(title, intro, result, comparison, down, up, reset);
  startOver();
  return () => {
    down.removeEventListener('click', tryDown);
    up.removeEventListener('click', tryUp);
    reset.removeEventListener('click', startOver);
    slot.replaceChildren();
  };
}
