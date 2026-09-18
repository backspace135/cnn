// 只记录自己观察到的现象；浏览器禁用存储时仍可在当前挂载期间记笔记。
export function mount(slot) {
  const key = 'cnn-concept-experiment-notes-v1';
  let notes = [];
  try {
    const saved = JSON.parse(localStorage.getItem(key) || '[]');
    if (Array.isArray(saved)) {
      notes = saved.filter(item => item && typeof item.condition === 'string'
        && typeof item.before === 'string' && typeof item.after === 'string'
        && typeof item.observation === 'string');
    }
  } catch (_) {
    // 隐私模式或损坏的数据不应阻止填写。
  }
  const title = document.createElement('h3');
  title.textContent = '一次只改变一个条件';
  const hint = document.createElement('p');
  hint.textContent = '先固定其他条件，只选下方一个条件，写下改动前后与观察。这是你的学习笔记，不是模型自动生成的实验记录。';
  const form = document.createElement('form');
  const conditionLabel = document.createElement('label');
  conditionLabel.textContent = '本次唯一改变的条件：';
  const condition = document.createElement('select');
  for (const name of ['学习率', '训练轮数', '批大小', '丢弃率']) {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    condition.append(option);
  }
  conditionLabel.append(condition);
  const beforeLabel = document.createElement('label');
  beforeLabel.textContent = '原来的值：';
  const before = document.createElement('input');
  before.type = 'text';
  before.required = true;
  beforeLabel.append(before);
  const afterLabel = document.createElement('label');
  afterLabel.textContent = '改成的值：';
  const after = document.createElement('input');
  after.type = 'text';
  after.required = true;
  afterLabel.append(after);
  const observationLabel = document.createElement('label');
  observationLabel.textContent = '你观察到了什么？';
  const observation = document.createElement('textarea');
  observation.required = true;
  observationLabel.append(observation);
  const save = document.createElement('button');
  save.type = 'submit';
  save.textContent = '保存观察笔记';
  const status = document.createElement('p');
  status.setAttribute('role', 'status');
  const list = document.createElement('ol');
  form.append(conditionLabel, beforeLabel, afterLabel, observationLabel, save);
  function render() {
    list.replaceChildren(...notes.map(note => {
      const item = document.createElement('li');
      item.textContent = `${note.condition}：${note.before} → ${note.after}；观察：${note.observation}`;
      return item;
    }));
  }
  function onSubmit(event) {
    event.preventDefault();
    const from = before.value.trim();
    const to = after.value.trim();
    const observationText = observation.value.trim();
    if (!from || !to || !observationText || from === to) {
      status.textContent = '请填写不同的前后值，以及你的观察。';
      return;
    }
    notes.push({ condition: condition.value, before: from, after: to, observation: observationText });
    render();
    try {
      localStorage.setItem(key, JSON.stringify(notes));
      status.textContent = '已保存到本浏览器；换设备不会同步。';
    } catch (_) {
      status.textContent = '已记在当前页面；浏览器存储不可用，关闭或刷新后可能丢失。';
    }
    before.value = '';
    after.value = '';
    observation.value = '';
  }
  form.addEventListener('submit', onSubmit);
  slot.replaceChildren(title, hint, form, status, list);
  render();
  return () => {
    form.removeEventListener('submit', onSubmit);
    slot.replaceChildren();
  };
}
