import {readPosition, savePosition} from '../core/progress.js';

export function mountSteps(article, lessonId) {
  const steps = [...article.querySelectorAll('.lesson-step')];
  if (!steps.length) return () => {};

  const progress = document.createElement('p');
  progress.className = 'step-progress';
  progress.setAttribute('role', 'status');
  const controls = document.createElement('div');
  controls.className = 'step-controls';
  const previous = document.createElement('button');
  previous.type = 'button';
  previous.textContent = '上一步';
  const next = document.createElement('button');
  next.type = 'button';
  next.className = 'primary';
  next.textContent = '下一步';
  controls.append(previous, progress, next);
  article.append(controls);

  let current = Math.min(readPosition(lessonId), steps.length - 1);
  function show(index, moveFocus = false) {
    current = Math.max(0, Math.min(index, steps.length - 1));
    steps.forEach((step, position) => { step.hidden = position !== current; });
    progress.textContent = `第 ${current + 1} 步，共 ${steps.length} 步`;
    previous.disabled = current === 0;
    next.textContent = current === steps.length - 1 ? '本课已读完' : '下一步';
    next.disabled = current === steps.length - 1;
    savePosition(lessonId, current);
    if (moveFocus) {
      const heading = steps[current].querySelector('h2, h3, h4');
      if (heading) { heading.tabIndex = -1; heading.focus(); }
      else steps[current].scrollIntoView({block: 'start'});
    }
  }
  const back = () => show(current - 1, true);
  const forward = () => show(current + 1, true);
  previous.addEventListener('click', back);
  next.addEventListener('click', forward);

  function answer(event) {
    const button = event.target.closest('button[data-correct]');
    if (!button || !article.contains(button)) return;
    const quiz = button.closest('[data-quiz]') || button.parentElement;
    let feedback = quiz.querySelector('.answer-feedback');
    if (!feedback) {
      feedback = document.createElement('p');
      feedback.className = 'answer-feedback';
      feedback.setAttribute('role', 'status');
      quiz.appendChild(feedback);
    }
    const correct = button.dataset.correct === 'true';
    feedback.classList.toggle('is-correct', correct);
    feedback.classList.toggle('is-incorrect', !correct);
    feedback.textContent = button.dataset.explanation ||
      (correct ? '答对了，可以继续下一步。' : '再想一想，看看上面的例子后再试一次。');
  }
  article.addEventListener('click', answer);
  show(current);
  return () => {
    previous.removeEventListener('click', back);
    next.removeEventListener('click', forward);
    article.removeEventListener('click', answer);
  };
}
