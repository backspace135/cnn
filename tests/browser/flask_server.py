"""独立浏览器测试使用的本地 Flask 服务夹具；无需 MNIST 或前端构建。"""
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from unittest.mock import patch
from urllib.error import URLError
from urllib.request import urlopen
from werkzeug.serving import make_server

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def local_flask_server():
    """绑定 127.0.0.1 的随机空闲端口，等首页可访问后交给测试。"""
    import server

    # 固定模拟未下载数据的首次使用场景，既不读取本机 MNIST，也不启动训练。
    with ExitStack() as patches:
        patches.enter_context(patch.object(server, 'trainer', None))
        patches.enter_context(patch.object(server, 'Trainer', side_effect=FileNotFoundError('测试环境未下载 MNIST')))
        httpd = make_server('127.0.0.1', 0, server.create_app())
        thread = Thread(target=httpd.serve_forever, name='browser-test-flask', daemon=True)
        base_url = f'http://127.0.0.1:{httpd.server_port}'
        try:
            thread.start()
            # 等待真实 HTTP 响应而不是猜测进程启动时间；超时会明确失败。
            deadline = monotonic() + 5
            while True:
                try:
                    with urlopen(base_url + '/', timeout=0.5) as response:
                        if response.status == 200:
                            response.read()  # 读完文件响应再断开，避免服务线程留下未关闭的流。
                            break
                        raise AssertionError(f'首页健康检查返回 HTTP {response.status}')
                except URLError as exc:
                    if monotonic() >= deadline:
                        raise AssertionError(f'本地 Flask 服务未在 5 秒内就绪：{base_url}') from exc
                    sleep(0.05)
            yield base_url
        finally:
            if thread.is_alive():
                httpd.shutdown()
                thread.join(timeout=5)
            httpd.server_close()
            if thread.is_alive():
                raise RuntimeError('本地 Flask 服务未能在 5 秒内关闭')
