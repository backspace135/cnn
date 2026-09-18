// 所有请求只发往同一台本地 Flask 服务；各功能页共享错误语义。
export async function request(path, {method = 'GET', body, signal} = {}) {
  const options = {method, signal};
  if (body !== undefined) {
    options.headers = {'Content-Type': 'application/json'};
    options.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error('无法连接本地课程服务。请检查启动窗口是否还开着。');
  }
  let data;
  try {
    data = await response.json();
  } catch (_) {
    throw new Error('服务返回了无法读取的结果。请刷新网页后重试。');
  }
  if (!response.ok || data?.ok === false) {
    throw new Error(data?.error?.message || data?.msg || data?.message || `请求未成功（${response.status}）。`);
  }
  return data;
}
