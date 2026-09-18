# 零基础 CNN 课程改造：详细实施计划

日期：2026-09-19
状态：用户已批准实施；本文记录预定步骤，自动化结果与未进行的人工验收须分别报告。
设计依据：[课程与代码整理设计](../specs/2026-09-19-zero-basics-course-redesign-design.md)。

## 执行约束与验证约定

- 目标读者：Windows 上完全没有计算机、机器学习及微积分基础的读者。网页主线能在不写代码、不推导导数的前提下完成；数学和源码是选修。
- 学员仍用 `install.bat`、`start.bat`、项目内 `libs/` 与 Flask；不需要 Node、前端构建器或 Playwright。只在开发环境安装可选浏览器测试工具，第三方依赖仍放项目内，不全局安装。
- **测试环境前置条件**：先在实际执行目录确认 Python 与 Flask、NumPy、PyTorch 可导入；缺依赖时按现有 `install.bat` 将核心依赖装入**该目录**的 `libs/`，再跑测试。独立 git worktree 不会自动带上主目录被忽略的 `libs/`，不得把其导入失败误报为代码回归。开发专用 Playwright 是核心依赖之外的另一步。记录每次测试的“运行数／通过数／跳过数／失败数”；主工作区改造前实测 22 项均显示 `ok`，其他隔离环境应以各自实际输出为准。
- 保留现有 `/api/*` 字段与错误语义、存档格式、CPU/CUDA 选择、训练线程及 `Trainer` 可替换／可继承的接口。训练数值算法和模型结构不在本计划内。
- 改动遵循“**先写能失败的测试 → 确认针对缺失行为失败 → 最小实现 → 运行针对性测试 → 完整回归**”。每个里程碑能启动；正式首页在切换前保持旧版可用。新增注释用解释意图的中文。
- 改造前已经运行 `python -m unittest discover -s tests -v`：**22 项通过**。这不是后续阶段的测试结论。执行计划时每阶段记录实际通过、失败或未运行；不要把浏览器或人工验收的缺席写成通过。
- 本计划不授权提交或推送。可以把不共享文件的课程编写、浏览器测试、后端辅助模块委派给独立执行者；共享入口 `index.html`、导航注册表和 `train.py` 的集成由同一负责人顺序完成，避免并行写入冲突。

以下“RED”表示先运行相关测试并确认**因目标行为尚不存在而失败**；若失败原因是环境或旧问题，先诊断而不是继续写实现。“GREEN”表示通过该测试与相关回归。命令均在仓库根目录执行。

## 阶段一：锁定旧行为，建立验证基础

### 任务 1：记录外部契约，给拆页预留测试方式

**文件**：`tests/test_frontend_contract.py`、新增 `tests/test_course_contract.py` 的可复用解析辅助；必要时读取 `static/index.html`、`static/app.js`、`static/lessons.js`、`static/predict.js`。
**准备**：列出并用当前页面可通过的测试固定旧锚点 `#intro/#model/#gloss/#theory/#live/#predict/#lab`、现有 DOM 控件、脚本加载方式与 API 请求形状。为多文件页面编写 HTML 链接／ID 解析辅助，并用小型测试夹具验证辅助本身；此任务是现状刻画，不伪造一个“失败测试”。
**验证**：`python -m unittest discover -s tests -v` 保持全绿。针对新课程的失败用例由任务 4 和各课任务在对应实现前添加，不让持续集成长期红灯。

### 任务 2：建立开发专用的真实浏览器测试夹具

**文件**：新增 `requirements-dev.txt`、`tests/browser/` 下的独立 `unittest` 用例与 Flask 启动夹具、`docs/advanced/testing.md`；按需要更新 `.gitignore` 忽略本地浏览器缓存。
**准备／验证**：此任务是测试基础设施，不把缺少 Playwright 当成产品行为的 RED。先写访问现有 Flask 首页、确认可读的浏览器冒烟用例和可模拟 `/api/health` 的夹具；在开发环境安装并实际运行，使测试通过。把 Python Playwright 装入项目内开发依赖位置，浏览器二进制放在项目内被忽略的开发缓存；文档给出 Windows 安装／运行命令，不修改学员的 `install.bat`。夹具选空闲本地端口，等待服务就绪，退出时关闭子进程；不依赖真实 MNIST、大规模训练或外网。
**命令**：开发者在 Windows 的命令提示符执行 `python -m pip install --target libs -r requirements-dev.txt`，设置 `PYTHONPATH=%CD%\libs` 与 `PLAYWRIGHT_BROWSERS_PATH=%CD%\libs\playwright-browsers` 后执行 `python -m playwright install chromium`，再运行 `python -m unittest discover -s tests/browser -v`；把可复制的完整命令写进 `docs/advanced/testing.md`。基础 `python -m unittest discover -s tests -v` 在未装 Playwright 时也能运行（浏览器用例独立放置，不让基础发现过程导入 Playwright）。缺少浏览器开发工具时单独报告“未运行”，不静默跳过，也不冒充基础测试失败。

### 任务 3：让缺少 MNIST 的学员先打开课程

**文件**：`start.bat`、`tests/test_learning.py`，必要时增加启动脚本测试；`server.py` 仅在现有行为不足时修改。
**RED**：固定无数据首页可访问、`/api/health` 明确返回数据不可用、训练入口返回可理解的错误；为启动脚本的数据前置检查增加回归覆盖，确认脚本不应在启动服务器前因无数据退出。
**GREEN**：去掉 `start.bat` 的强制数据检查，改为提示“现在可以学前面课程；到训练步骤再下载数据”，仍保留 Python、依赖及端口错误的明确提示。不要移动或删除用户已有的数据文件来模拟缺数据；使用隔离环境或静态脚本检查，真实 Windows 无数据启动另做人工冒烟。
**验证**：`python -m unittest discover -s tests -p "test_learning.py" -v` 与完整 Python 测试；Windows 手动运行 `start.bat`，分别核查有／无数据的入口。

## 阶段二：搭建可逐课替换的网页基础

### 任务 4：建立独立的课程暂存入口与路由

**文件**：新增临时 `static/course.html`、`static/js/main.js`、`static/js/core/lesson-registry.js`、`router.js`、`lesson-loader.js`；`tests/test_course_contract.py`、浏览器导航测试。
**RED**：测试直接打开章节地址、前进／后退、刷新、未知章节回到可操作的开始页，以及快速切课不会把上一课的异步结果插入当前课。先固定 `training`、`prediction`、`lab` 路由键的可达性；旧 `static/index.html` 在此阶段保持原样。
**GREEN**：创建课程壳和 00—06 注册表，按当前 hash 装入静态 HTML；加载活动时才动态 `import()` 对应模块。每次导航取消尚未完成的装载，并调用前一页的清理函数。训练／识别／调参路由暂时显式跳转到仍可用的旧首页 `#live/#predict/#lab`，任务 13、14 再替换为新页面挂载；`/static/course.html` 仅供迁移期间测试，不作为正式第二套课程入口。
**验证**：`python -m unittest discover -s tests -p "test_course_contract.py" -v`、浏览器导航用例和完整 Python 回归；确认旧 `/` 仍正常。

### 任务 5：抽出共用步骤、术语、反馈与本机进度

**文件**：`static/js/components/stepper.js`、`feedback.js`、`glossary.js`；`static/js/core/progress.js`；`static/css/base.css`、`layout.css`、`lesson.css`；相关浏览器测试。
**RED**：测试键盘可达、焦点可见、回答错误给解释、跳课／上一步／下一步，以及 `localStorage` 不可用或抛错时仍可开始学习。
**GREEN**：控件通过明确的 `mount(container, options)` 与清理接口组合；用语义化按钮和文本节点渲染动态数据。本机进度只用于恢复位置，不拦截跳课或依赖浏览器存储才能运行。小屏导航与内容纵向堆叠。
**验证**：浏览器键盘、存储禁用和约 400px 宽视口用例；开发者目检长文字没有横向滚动。

## 阶段三：逐课写内容和可验证互动

每一课先写文本／互动的可检查目标，再实现教材，不能只把旧段落搬进新文件。课程内容测试检查结构、链接和交互边界；“读者是否听懂”留给人工验收。

### 任务 6：00 课与电脑零基础 Windows 启动指引

**文件**：`static/lessons/00-start.html`、`docs/windows-start.md`、`README.md`、开始页浏览器用例。
**RED**：测试首页能找到从 Windows 文件夹到启动课程的入口；零数据状态能看见可执行的下一步而不是空白；README 链接均指向存在的文件／页面。
**GREEN**：按“做什么 → 会看到什么 → 没看到怎么办”写 Python 安装与检查、项目文件夹、`install.bat`、`start.bat`、浏览器地址、下载数据的时机、常见端口／PATH／依赖错误。避免让尚未打开网页的人必须先看网页才能完成安装；网页 00 课负责回顾与确认，文档负责入门前步骤。README 缩为入口而非重复整本讲义。
**验证**：静态链接测试、浏览器无数据流程、Windows 新手按文档实际走一遍。

### 任务 7：01 课，把数字图片变成像素表

**文件**：`static/lessons/01-pixels.html`、`static/js/activities/pixels.js`，必要时独立纯计算模块；内容契约与浏览器用例。
**RED**：点击方格改变对应数字；相同操作可恢复；能区分一个像素、小矩阵、28×28 图，不要求先懂 `float32` 或四维数组。
**GREEN**：从小方格开始，让学员先预测再观察 0—1 灰度数值；用一段短释义区分“人眼看到的图”和“输入给程序的数字”。代码与 NCHW 放进进阶链接。
**验证**：确定输入的数值映射用例、浏览器点击／键盘操作及正确／错误反馈。

### 任务 8：02 课，手算第一格再滑动模板

**文件**：`static/lessons/02-patterns.html`、`static/js/activities/convolution.js`、已有 `static/lessons.js` 中可复用的纯逻辑；边界与浏览器测试。
**RED**：固定小矩阵和模板，测试第一格每次乘法与求和及完整输出；测试尺寸或数值非法输入不导致无说明的异常；浏览器能先看第一格而非立即遇到步长／填充公式。
**GREEN**：先展示不能编辑的第一格演示，再允许逐格移动和改参数；之后才介绍池化的“保留局部重要线索”直觉。移植旧计算时保持数值结果一致，移除旧全局依赖前保留旧页面可用。
**验证**：数值边界用例、逐格操作浏览器测试、旧前端契约回归。

### 任务 9：03 课，把线索连成网络

**文件**：`static/lessons/03-network.html`、必要时 `static/js/activities/network.js`；课程结构和浏览器用例。
**RED**：测试 28→14→7 的示意每个阶段都有短解释，输出的十个分数不会被文案写成“必然正确”的概率。
**GREEN**：用交互图示连接卷积线索、池化汇集、分类输出；把 Dropout、BatchNorm、Softmax 和通道等内容放到首次接触处的简释或进阶页，正文不一次堆术语。
**验证**：浏览器步骤／解释呈现、课程术语审阅。

### 任务 10：04 课，用数字试探解释训练

**文件**：`static/lessons/04-learning.html`、`static/js/activities/weight-step.js`、训练入口链接与测试。
**RED**：给定权重、输入和目标，调整前后的预测与误差能用四则运算验证；连续点按不会把计算状态弄乱；没有 MNIST 时玩具例子仍可操作。
**GREEN**：先让学员猜一个小步是更好还是更差，再引入标签、损失、更新和轮次；数学导数仅进阶链接。实际训练按钮指向稍后迁移的训练功能，不承诺每次参数调整都会提高准确率。
**验证**：数值用例、浏览器反馈、缺数据下该课照常学习。

### 任务 11：05 课，检查模型是否学会

**文件**：`static/lessons/05-evaluation.html`、`static/js/activities/evaluation.js`（若需要）、评估页测试。
**RED**：固定正确／错误样例能计算可解释的准确率；明确标注的教学示例能对照“参与训练的图片”和“未参与训练的图片”，而不能把现有验证集样例误称训练样例；训练、验证、最终测试角色明确，错误样例能被选中并显示解释。
**GREEN**：用少量数字算一次指标，再接现有评估可视化和真实验证错例；训练／验证对照若使用静态小样本，必须显著标出“教学示意，不是当前模型的真实训练记录”。说明验证样本没有参与当前模型训练、测试集不用于反复调参；输出高分也可能出错。
**验证**：确定性指标用例、浏览器交互、文案审阅。

### 任务 12：06 课，手写识别与单变量实验

**文件**：`static/lessons/06-try-it.html`、`docs/experiments.md`、课程检查测试；实际画板复用第 14 项的预测功能。
**RED**：测试主线能找到手写识别与只改一个参数的观察入口，能记录“改了什么／看到了什么”；不能把某次偶然结果写成保证。实际识别页面的无模型行为留到任务 14 测试。
**GREEN**：给一次无需改代码的对照练习，提供“改了什么／看到什么／可能还有什么原因”的记录框；严格控制训练／验证样本数的比较、修改源码等另标进阶。暂存课程中需要真实训练／识别的按钮暂时通往仍可用的旧首页相应锚点；任务 13—15 完成时再切到新功能页，不出现失效按钮。
**验证**：暂存课程到旧功能的浏览器跳转、内容检查；人工审阅练习不要求隐含统计知识。

## 阶段四：迁移现有训练与识别功能，再切正式入口

### 任务 13：完整迁移训练观察、调参与模型操作

这项分成三个独立的 RED／GREEN 小循环：**13a** 状态、控制、曲线与样例；**13b** 参数／设备设置及实验历史；**13c** 模型结构与保存／加载。每个循环先测本组行为，再迁移并跑完整 Python 回归；第三组完成后才把 `training/lab` 路由指向新页。

**文件**：`static/pages/training.html`、`static/js/core/api.js`、`static/js/features/training/` 中的控制／状态／图表／模型详情／实验记录／存档模块、`static/css/training.css`；`static/app.js` 中相应旧逻辑作为行为参考；课程注册表与浏览器测试。
**RED**：逐项列出旧页入口并模拟成功、失败、响应延迟：训练开始／暂停／恢复／停止／重置及配置，epochs/lr/batch/dropout 与 CPU/CUDA 选择，状态曲线／样例／特征图、模型结构（`/api/model-info`）、实验历史（`/api/experiments`）、保存／加载模型（`POST /api/model` 的 `save/load`）。检查 700ms 串行轮询、控制不重入、特征图更新去重、离页不再请求、API 失败可重试。
**GREEN**：`api.js` 统一错误规范；训练控制只发原有 `/api/control` 字段，模型存取及详情继续使用现有端点与请求形状；视图接受状态快照而不直接发状态轮询，轮询由页面控制器独占并在卸载时停止。把注册表中的 `training` 和 `lab` 从旧首页回退改成挂载／卸载新训练页，`lab` 明确定位到参数实验区域。暂存课程能进入、返回训练页，空状态有解释和操作入口。
**验证**：浏览器逐项验证旧训练功能清单、从课程进入／离开的生命周期与请求格式；`python -m unittest discover -s tests -p "test_device_api.py" -v` 和完整回归。

### 任务 14：拆画板、输入预处理与预测结果

**文件**：`static/pages/prediction.html`、`static/js/features/prediction/` 下的画板／预处理／结果模块、`static/css/prediction.css`；从旧 `static/predict.js` 迁移行为；浏览器测试。
**RED**：空画板、清空、绘制、28×28 数值矩阵、请求与响应格式、无模型及 API 错误；测试暂存课程进入 `prediction` 路由后页面挂载，离开时事件监听解除，重新进入不会叠加处理器；验证移动端画板和键盘可达操作。
**GREEN**：画板预处理与结果呈现分开，原本跨 `app.js`／`predict.js` 的 Canvas 工具变为显式导入；按钮只在当前页面挂载时绑定，不依赖 `window.modelAction`。把注册表中的 `prediction` 从旧首页回退改为新页挂载／卸载，06 课可从暂存课程进入并返回。预测结果说明“高分不保证正确”。
**验证**：浏览器画板与结果用例、`python -m unittest discover -s tests -p "test_learning.py" -v`、完整回归。

### 任务 15：以课程壳替换首页并清理旧前端

**文件**：`static/index.html`、`static/course.html`、`static/app.js`、`static/lessons.js`、`static/predict.js`、`static/style.css`，新路由／CSS 与 `tests/test_frontend_contract.py`。
**RED**：切换前增加首页直接打开新课程、旧 hash 深链可到对应内容、浏览器后退和页面刷新、无内联事件／全局处理器的用例；用旧页面功能清单检查模型结构、实验历史、参数设置与设备选择、存档操作、训练图表／样例及画板都有可操作的新入口。
**GREEN**：只在任务 13、14 的新页路由、挂载和旧功能清单均已通过测试后，把暂存壳切成正式 `/`；把旧 `#intro/#model/#gloss/#theory/#live/#predict/#lab` 映射到最接近的新章节／功能页，其中 `#lab` 必须落到训练页的可修改参数区域而非仅显示曲线。一次清除不再引用的旧脚本、样式和暂存文件。更新旧静态契约测试，使其核对新入口及所有片段 ID 的并集，而非只检查单页 HTML。
**验证**：所有 Python 测试、全部浏览器测试、手动逐一打开旧深链；确认 Flask 静态路径有效、无脚本 404，浏览器控制台无异常。新入口仅保留一套权威课程导航。

## 阶段五：在契约测试保护下整理 Python 后端

### 任务 16：抽设备选择辅助逻辑

**文件**：新增 `training/__init__.py`、`training/device.py`，调整 `train.py`，扩充 `tests/test_training_devices.py`。
**RED**：固定 CPU、不可用 CUDA、失败后配置与模型不变、动态 mock 探测，以及 `from train import hardware_info, resolve_device` 仍可调用。
**GREEN**：只移动设备计算与探测，`train.py` 转发或重新导出原入口；保留 `paths` 优先注入 `libs/` 的导入顺序、重置前校验及用户可理解的错误。
**验证**：`python -m unittest discover -s tests -p "test_training_devices.py" -v` 和完整回归。

### 任务 17：抽存档校验与文件读写

**文件**：`training/checkpoint.py`、`train.py`、`tests/test_learning.py`。
**RED**：固定存／取往返、损坏／版本不兼容存档不会替换现有模型、失败不破坏当前训练状态，并检查既有存档字段及路径行为。
**GREEN**：只抽无状态的格式校验与原子文件读写；`Trainer` 继续持有模型、锁及同名公开方法，在原锁边界调用辅助逻辑；不改格式版本或异常语义。
**验证**：`python -m unittest discover -s tests -p "test_learning.py" -v` 和完整回归。

### 任务 18：抽预测纯计算与输入校验

**文件**：`training/inference.py`、`train.py`、`tests/test_learning.py`。
**RED**：覆盖 28×28 像素合法／非法值、预测返回字段、`no_grad`／评估模式、预测前后权重与模型状态不变，以及 `server.Trainer` 替身仍可使用。
**GREEN**：将可独立计算部分移入辅助函数；`Trainer.predict()` 仍控制 `model_lock`、调用顺序和状态查询。若不能保证原行为，缩小抽出范围，不移动训练线程／状态所有权。
**验证**：预测单测、`python -m unittest discover -s tests -v`，以及浏览器手写预测流程。

## 阶段六：进阶资料、完整验收与交付

### 任务 19：把讲义和源码练习移到明确的选修路径

**文件**：`docs/learning-guide.md`、`docs/advanced/math.md`、`docs/advanced/python-and-source.md`、现有 `examples/` 的说明、`README.md`、`docs/experiments.md`；课程链接测试。
**RED**：从 README、每课、进阶资料出发检查链接及章节编号一致，主线不以脚本、导数或 NCHW 为入门前提。
**GREEN**：把旧讲义改为课程与选修索引，不与网页重复两套相互矛盾的课表；现有示例按 01—04 顺序解释用途，公式从具体四则运算例子接入，实验单区分入门观察和严谨进阶比较。
**验证**：链接测试、编辑审阅逐课新词、数值例子与限定措辞。

### 任务 20：最终回归与真实新手走查

**文件**：按测试结果修复相关模块；补 `docs/advanced/testing.md` 中的最终执行记录模板。
**自动验证**：运行 `python -m unittest discover -s tests -v`；开发依赖可用时运行 `python -m unittest discover -s tests/browser -v`。针对服务页面与脚本请求检查本地资源均能加载、无外部前端请求；针对 400px 宽、键盘、缺数据、数据已备好、训练暂停／重置、预测空输入与有效输入、存／取档逐一核对。
**人工验收**：在 Windows 新手路径从文件资源管理器与 Python 安装开始，独立完成启动、算出卷积第一格、解释一组训练／验证结果、画一个数字并做一次单变量比较；记录卡住的位置及修改后的复测结果。不能用“自动化测试通过”替代“学员看得懂”。
**交付**：列出实际完成的任务、完整测试输出摘要、浏览器测试是否运行及原因、人工走查结果、仍未解决的限制。只在用户要求时提交或推送。

## 关键实施检查点

1. **启动可学**：无 MNIST 时正式入口仍可打开，Windows 文档足以让初学者抵达课程；Python 原测试保持通过。
2. **课程可学**：00—06 的文字与互动在暂存入口通过浏览器测试；每课有目标、预测、动手、反馈、复述和下一步。
3. **功能可用**：训练、评估、画板、存档通过原 API 工作；旧锚点仍能导航，新首页取代旧版且不再有隐式全局。
4. **结构可维护**：前端按课程／组件／功能拆分；后端设备、存档、推理的可独立部分在辅助模块，`Trainer` 保留状态所有权与兼容入口。
5. **验收真实**：Python、浏览器和新手人工走查分别给出证据与未运行项，不承诺某次训练固定达到某个准确率。
