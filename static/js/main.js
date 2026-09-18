import {startRouter} from './core/router.js';

const content = document.getElementById('courseView');
const navigation = document.getElementById('courseNav');
const status = document.getElementById('routeStatus');

// 页内跳转不能占用路由 hash；否则键盘跳到正文会把当前课切回第 00 课。
document.querySelector('.skip-link').addEventListener('click', event => {
  event.preventDefault();
  content.focus();
});
startRouter(content, navigation, status);
