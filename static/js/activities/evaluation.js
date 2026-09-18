// 所有数字均为教学示意：四张纸条不是 MNIST 样本，也不是模型运行记录。
export function mount(slot) {
  const title = document.createElement('h3');
  title.textContent = '训练时见过，与没见过的例子';
  const notice = document.createElement('p');
  notice.textContent = '教学示意：以下是虚构的训练/验证小样本，不是真实模型的准确率或训练记录。';
  const explanation = document.createElement('p');
  explanation.textContent = '训练样本用于练习；验证样本留作检查新题。准确率 = 猜对的张数 ÷ 总张数 × 100%。';
  const groups = ['训练', '验证'].map(name => {
    const section = document.createElement('section');
    const heading = document.createElement('h4');
    heading.textContent = `${name}样本（4 张）`;
    const list = document.createElement('ul');
    const accuracy = document.createElement('p');
    section.append(heading, list, accuracy);
    return { section, list, accuracy };
  });
  const insight = document.createElement('p');
  insight.setAttribute('aria-live', 'polite');
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = '换一种示意猜法';
  const answers = [0, 1, 2, 3];
  // 第一组只对训练题表现好；第二组让新题的表现也变好。
  const guesses = [
    [[0, 1, 2, 3], [0, 0, 0, 0]],
    [[0, 1, 2, 0], [0, 1, 2, 0]],
  ];
  let mode = 0;
  function render() {
    groups.forEach((group, groupIndex) => {
      const predictions = guesses[mode][groupIndex];
      let correct = 0;
      const rows = answers.map((answer, index) => {
        const row = document.createElement('li');
        const guess = predictions[index];
        if (guess === answer) correct += 1;
        row.textContent = `第 ${index + 1} 张：答案 ${answer}，示意猜测 ${guess}（${guess === answer ? '对' : '错'}）`;
        return row;
      });
      group.list.replaceChildren(...rows);
      group.accuracy.textContent = `准确率：${correct}/4 = ${correct * 25}%`;
    });
    insight.textContent = mode === 0
      ? '训练题 100% 也不能说明新题一定好；请比较验证题的准确率。'
      : '对比两组示意：即使训练题少对一张，验证题也可能多对。';
  }
  function change() {
    mode = (mode + 1) % guesses.length;
    render();
  }
  button.addEventListener('click', change);
  slot.replaceChildren(title, notice, explanation, groups[0].section, groups[1].section, insight, button);
  render();
  return () => {
    button.removeEventListener('click', change);
    slot.replaceChildren();
  };
}
