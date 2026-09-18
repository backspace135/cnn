const KEY = 'cnn-course-position-v1';

// 进度只是一种方便；浏览器拒绝存储时，学习功能不能跟着失效。
export function readPosition(lessonId) {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) || '{}');
    return Number.isInteger(saved[lessonId]) && saved[lessonId] >= 0 ? saved[lessonId] : 0;
  } catch (_) {
    return 0;
  }
}

export function savePosition(lessonId, index) {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) || '{}');
    localStorage.setItem(KEY, JSON.stringify({...saved, [lessonId]: index}));
  } catch (_) {
    // 私密模式或禁用存储时仍可继续学习。
  }
}
