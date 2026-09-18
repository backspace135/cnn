# 独立浏览器冒烟测试

浏览器测试位于 `tests/browser/`，使用 Python `unittest` + Playwright (Chromium)，通过随机空闲端口启动仅绑定 `127.0.0.1` 的 Flask 服务。夹具模拟“尚未下载 MNIST”，检查 00–06 课、章节历史与自测、卷积互动、窄屏布局、训练/识别接口和缺数据提示；**不会下载数据、训练模型或构建 Node 前端**。测试结束会关闭浏览器、隔离上下文和本地 HTTP 服务。

## Windows PowerShell：安装

以下命令在项目根目录运行，建议使用 Python 3.11 或更新版本。所有 Python 包仅安装到项目的 `libs/`；Chromium 安装到 Git 已忽略的项目内 `data/playwright-browsers/`，不使用用户级浏览器缓存。首次安装需要联网。

```powershell
python -m pip install -r requirements.txt --target libs --upgrade
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu --target libs --upgrade
python -m pip install -r requirements-dev.txt --target libs --upgrade
$env:PYTHONPATH = (Join-Path (Get-Location) 'libs')
$env:PLAYWRIGHT_BROWSERS_PATH = (Join-Path (Get-Location) 'data\playwright-browsers')
python -m playwright install chromium
```

如果已按项目说明运行过 `install.bat`，前两条 pip 命令可以省略。`PYTHONPATH` 使 `python -m playwright` 找到安装在 `libs/` 的 Python 包；Playwright 安装浏览器时自带运行所需的驱动，不需要安装 Node 或执行前端构建。新开 PowerShell 会话时，重新设置上述两个环境变量再安装浏览器；测试脚本在未显式设置浏览器路径时也会默认使用项目内缓存。

## 运行

```powershell
# 基础单元/API/静态契约测试：不会扫描或导入 tests/browser/。
python -m unittest discover -s tests -v

# 需要先安装 Playwright 和 Chromium；单独运行真实浏览器冒烟测试。
$env:PLAYWRIGHT_BROWSERS_PATH = (Join-Path (Get-Location) 'data\playwright-browsers')
python -m unittest discover -s tests/browser -p 'test_*.py' -v
```

`tests/browser/` 故意不放 `__init__.py`：标准 `unittest discover -s tests -v` 不递归扫描这个非包目录，所以仅运行基础测试时无需安装 Playwright；显式指定 `-s tests/browser` 才会导入浏览器测试。基础测试仍需要项目原有的 Flask、NumPy 和 PyTorch 依赖。少数 JavaScript 纯计算测试在开发机有 Node 时执行，没有 Node 时明确标为跳过；**学员运行网页、课程和 Python 后端不需要安装 Node**。浏览器测试不需要 `data/mnist.npz`，缺数据时训练页显示下载提示是预期行为。

若提示缺少 `torch`、`flask` 或 `playwright`，请确认使用了同一 Python 解释器并已按上面命令安装到 `libs/`；这属于测试环境尚未就绪。若提示 Chromium 可执行文件不存在，请确认浏览器缓存环境变量与安装时一致，再运行 `python -m playwright install chromium`。测试不访问互联网，但真实浏览器/操作系统依赖仍需在本机安装成功。
