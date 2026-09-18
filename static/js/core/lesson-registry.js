// 所有课程链接只在这里定义，避免章节与侧栏各维护一套顺序。
export const lessons = [
  {id: '00', title: '先把课程打开', file: '00-start.html'},
  {id: '01', title: '图片是一格格数字', file: '01-pixels.html', activity: 'pixels'},
  {id: '02', title: '用小窗口找线索', file: '02-patterns.html', activity: 'convolution'},
  {id: '03', title: '把线索组成网络', file: '03-network.html', activity: 'network'},
  {id: '04', title: '让机器一点点学', file: '04-learning.html', activity: 'weight-step'},
  {id: '05', title: '检查它是否真会了', file: '05-evaluation.html', activity: 'evaluation'},
  {id: '06', title: '自己写、自己试', file: '06-try-it.html', activity: 'experiment'},
].map(lesson => ({...lesson, hash: `#/learn/${lesson.id}`,
  url: `/static/lessons/${lesson.file}`, kind: 'lesson'}));

export const features = [
  {id: 'training', title: '训练观察', hash: '#/training', kind: 'feature',
    url: '/static/pages/training.html', module: '/static/js/features/training/index.js'},
  {id: 'prediction', title: '手写识别', hash: '#/prediction', kind: 'feature',
    url: '/static/pages/prediction.html', module: '/static/js/features/prediction/index.js'},
  {id: 'lab', title: '调参实验', hash: '#/lab', kind: 'feature',
    url: '/static/pages/training.html', module: '/static/js/features/training/index.js', focus: 'settings'},
];

const aliases = {intro: '01', model: '03', gloss: '04', theory: '02',
  live: 'training', predict: 'prediction', lab: 'lab'};

export function routeForHash(hash) {
  const raw = (hash || '').replace(/^#\/?/, '');
  const target = aliases[raw] || raw;
  if (target.startsWith('learn/')) {
    return lessons.find(lesson => lesson.id === target.slice(6)) || lessons[0];
  }
  return lessons.find(lesson => lesson.id === target) ||
    features.find(feature => feature.id === target) || lessons[0];
}

export function lessonIndex(id) {
  return lessons.findIndex(lesson => lesson.id === id);
}
