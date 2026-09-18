# Windows 零基础启动指南

本页从打开文件夹开始。**首次安装 Python 和项目依赖需要联网**；MNIST 数据可以等学到真实训练时再联网下载。没有 MNIST 也能先打开课程并学习前面的互动章节。只用普通 Windows 电脑的 CPU 即可，不需要显卡、编程知识或微积分。

每一步按「操作 → 预期 → 失败时怎么办」检查。项目文件夹指同时能看到 `README.md`、`install.bat`、`start.bat` 和 `download_data.py` 的文件夹，不是里面的 `docs` 文件夹。

## 1. 找到项目文件夹

- **操作**：按 `Win + E` 打开文件资源管理器，找到下载或解压后的项目。如果拿到的是 ZIP 压缩包，先右键选择「全部解压缩」，再打开解压出的文件夹；若最外层还有一层同名文件夹，就继续进入，直到能看到上面四个文件。Windows 可能隐藏扩展名，`install.bat` 可能只显示成 `install`；可在资源管理器的「查看 → 显示 → 文件扩展名」打开显示。
- **预期**：同一个文件夹里有 `install.bat`、`start.bat`、`download_data.py`，也有 `docs`、`examples` 等文件夹。
- **失败时怎么办**：如果只看到一个 ZIP，先解压；如果看到了 `windows-start.md`，说明你还在 `docs`，退回上一层；如果没看到 `install.bat`，搜索下载位置或重新取得完整项目，别在压缩包预览里运行。

## 2. 安装 Python 并检查 PATH

**PATH** 是 Windows 寻找程序的目录列表；安装 Python 后把它加进 PATH，后面的 `.bat` 才能通过 `python` 找到它。

- **操作**：访问 [Python 官方 Windows 下载页](https://www.python.org/downloads/windows/)，下载适合 Windows 的 Python 3.11 或更新版本安装程序（本项目曾在 3.13 下使用）。运行安装程序时勾选 **Add python.exe to PATH**（或 **Add Python to PATH**），再选择安装。安装完毕后关闭旧终端；按 `Win` 键，输入「PowerShell」，打开一个**新的** PowerShell 窗口，输入：

  ```powershell
  python --version
  ```

- **预期**：看到类似 `Python 3.13.x` 的版本号（具体小版本可以不同），而不是跳到商店。完成安装时 Windows 若提示「Disable path length limit」，可按安装程序提示处理。
- **失败时怎么办**：若提示找不到 `python`，先重新打开 PowerShell 再试；仍失败时重新运行 Python 安装程序，选修改安装并启用 PATH。若自动打开 Microsoft Store，去 Windows 设置里的「应用执行别名」关闭 `python.exe` / `python3.exe` 的商店别名，再开新终端检查。若显示旧版本，检查是否安装了多个 Python，并让即将运行 `install.bat` 的 `python` 指向 3.11 或更新版本。不要以为只安装 PyTorch 或只下载本项目就可以省略 Python。

## 3. 在项目文件夹安装依赖

- **操作**：保持网络连接，在文件资源管理器的项目文件夹里**双击 `install.bat`**。它会在本项目的 `libs/` 安装 Flask、NumPy 和 CPU 版 PyTorch；无需自行创建 Python 虚拟环境，也无需管理员权限。窗口执行时不要提前关闭。
- **预期**：窗口先显示 `[1/2] Installing Flask and NumPy into libs...`，再显示 `[2/2] Installing CPU PyTorch into libs...`，成功后依次显示 `Dependencies installed. Next run: start.bat` 和 `Before training: python download_data.py`。先按下一步启动课程；MNIST 数据等到第 5 步、开始真实训练之前再下载。按提示关闭窗口即可；安装所需时间取决于网络。
- **失败时怎么办**：看到 `Python was not found`，返回第 2 步检查 PATH；看到 `Installation failed`，确认网络可访问依赖下载站、磁盘空间足够、Python 版本正确，然后重新双击安装脚本。若窗口一闪而过，在资源管理器项目文件夹的地址栏输入 `powershell` 并回车，在新打开的窗口运行 `.\install.bat` 查看错误。不要手动移动 `libs/`：启动脚本会从本项目的文件夹找依赖。

## 4. 启动课程并打开浏览器

- **操作**：回到项目文件夹，双击 `start.bat`。保持弹出的命令窗口打开；启动脚本会尝试打开默认浏览器。如未自动打开，在浏览器的**地址栏**输入 <http://127.0.0.1:5000>（不是在搜索框里搜索）。不要双击 `static/index.html`：课程需要本机服务提供网页文件。
- **预期**：浏览器打开中文课程，能看到「从像素开始」入口和 **00–06 章节目录**，从「00 开始使用」开始学习。`127.0.0.1` 指你自己的电脑，不是外部学习网站。若尚未下载 MNIST，启动窗口会提示缺少 `data\mnist.npz`，**仍会打开课程**：先学 00–03 和 04 的玩具互动，真实训练、识别等数据相关功能等第 5 步准备好数据再使用。网页不会自动开始训练。
- **失败时怎么办**：若提示 `Python was not found`，回第 2 步；提示 `Dependencies are missing`，回第 3 步并确认 Python 没换版本；**仅提示缺少 MNIST 不表示启动失败**。浏览器显示「无法访问此网站」时，先稍等并刷新，检查启动窗口有没有报错、是否保持打开且地址端口是 `5000`。若提示端口已占用，先关闭其他占用 5000 的本项目窗口再双击；若不能关闭，在项目文件夹的 PowerShell 中先运行 `$env:PORT='5001'`，再运行 `.\start.bat`，然后打开 <http://127.0.0.1:5001>。结束时关闭命令窗口即可停止本地服务，网页也会停止工作。

## 5. 到训练课再下载手写数字数据

到了 **04 让机器学习** 的真实训练入口，先完成无需数据的权重小练习；当页面提示数据尚未准备、准备开始真实训练时，再做本步。`install.bat` **不会**自动生成 `data/mnist.npz`，首次下载要联网。可以保持课程网页的启动窗口打开，另开一个 PowerShell 窗口下载。

- **操作**：回到资源管理器里的项目文件夹，单击窗口顶部**地址栏**（显示文件夹位置的长条），输入 `powershell` 并按回车。在**新打开**的 PowerShell 输入下面的命令并按回车：

  ```powershell
  python download_data.py
  ```

- **预期**：显示四个 MNIST 文件的下载或缓存校验消息，最后出现 `[完成] 训练 60,000、测试 10,000 张 → ...\data\mnist.npz`。`data` 文件夹和 `mnist.npz` 由脚本创建；首次运行要联网，后来已有完整数据时本地课程不必重新联网下载。返回网页，若数据状态尚未更新，刷新页面，再到训练入口开始训练。CPU 首次完整训练可能需要较长时间，不要求达到固定成绩。
- **失败时怎么办**：提示找不到 `download_data.py`，说明 PowerShell 不在项目文件夹；回第 1 步，从能看到 `install.bat` 的文件夹地址栏重新打开。提示缺少 `numpy`，回第 3 步安装依赖。网络超时或下载失败时检查连接后重新运行同一命令：脚本会验证已有缓存，损坏的原始下载会重下。若磁盘空间不足，先腾出空间再试。**不要**把四个原始 `.gz` 文件误当成最后所需的 `mnist.npz`。下载完成但网页仍提示缺数据时，确认下载与启动使用的是同一项目文件夹，刷新网页或重启 `start.bat`。


## 接下来读什么

按 [学习指南](learning-guide.md) 在网页内动手；用 [实验记录](experiments.md) 在网页上做单变量比较。`examples/` 的脚本和 [Python 与源码](advanced/python-and-source.md)、[数学进阶](advanced/math.md) 都是**选修**，第一次使用无需打开代码编辑器或学习导数。
