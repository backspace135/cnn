// 只用传入的容器绘制示意，不接入真实模型或训练数据。
export function mount(slot) {
  const title = document.createElement('h3');
  title.textContent = '一张图怎样变成十个分数？';
  const path = document.createElement('p');
  path.textContent = '教学示意：28×28 像素 → 14×14 特征格 → 7×7 特征格 → 数字 0～9 各一个分数。缩小的尺寸只是帮助理解，不代表每个网络都这样设计。';
  const hint = document.createElement('p');
  hint.textContent = '下面是虚构的示意分数，不是真实预测；分数最高只是模型会猜的数字，不保证猜对。';
  const scores = document.createElement('ol');
  scores.setAttribute('aria-live', 'polite');
  const guess = document.createElement('p');
  const entries = Array.from({ length: 10 }, (_, digit) => {
    const item = document.createElement('li');
    scores.append(item);
    return item;
  });
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = '换一组示意分数';
  const examples = [
    [2, 1, 3, 8, 2, 1, 4, 2, 3, 1],
    [1, 2, 2, 1, 3, 2, 1, 2, 8, 2],
  ];
  let current = 0;
  function render() {
    entries.forEach((item, digit) => {
      item.textContent = `${digit}：${examples[current][digit]} 分`;
    });
    const highest = examples[current].indexOf(Math.max(...examples[current]));
    guess.textContent = `这组最高是数字 ${highest}，只是示意猜测，不代表一定猜对。`;
  }
  function next() {
    current = (current + 1) % examples.length;
    render();
  }
  button.addEventListener('click', next);
  slot.replaceChildren(title, path, hint, scores, guess, button);
  render();
  return () => {
    button.removeEventListener('click', next);
    slot.replaceChildren();
  };
}
