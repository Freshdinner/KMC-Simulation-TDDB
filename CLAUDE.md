# Claude Code Context
**重要上下文提示**
1. 请优先加载参考 @PROJECT_CONTEXT_TDDB.md（相对路径：./PROJECT_CONTEXT_TDDB.md）作为项目背景

## 输出风格
中文提示词的问题使用简体中文回答
做根因修复，用正确逻辑直接覆盖，不要打补丁，禁止写否定的否定，直接写肯定性的描述。
要求语义流畅清晰，把商业黑话和表演性技术腔改回普通动作（如抓手、收窄、兜住、落盘、硬帐、插桩、胀成、咬住、钉子等），让普通人能够理解。
保留术语和系统主语，不要改成口语闲聊体。不要改变原意

## 环境
- Python 3.8
- numpy 固定在 1.24.x 及以下（1.25 起不再支持 3.8），scipy 固定在 1.10.x
  及以下（1.11 起不再支持 3.8）——写进 pyproject.toml 的版本约束里，不要
  让依赖自动升级到不兼容版本
- 依赖用 pyproject.toml 管理
- 核心库：numpy, scipy, matplotlib；CLI 用 typer；测试用 pytest（列在基础
  依赖里，固定在 8.3.x 及以下，8.4 起不再支持 3.8）

## 单位约定
- 内部计算全部用 SI 单位（J, m, K, C, s）
- 配置文件（YAML）用器件物理惯用单位（eV, nm, cm⁻³），只在 io.py 的读入
  边界做换算，换算系数一律从 `scipy.constants` 取，不手写字面量

## 变更记录、提交、决策记录

沿用 `PROJECT_CONTEXT.md` 的规范：

- 完成一次逻辑完整的修改后（不是每个中间步骤），在 `CHANGELOG.md` 的 [未发布] 节
  追加条目：改了什么、为什么、涉及哪个 T/R 编号或物理文档、pytest 是否通过。
- `git commit`，message 格式 `<模块>: <一句话>`（如 `rates/generation: 实现 T1 生成率 G`），
  `user.name: appetizingbrunch`、`user.email: appetizingbrunch@proton.me`。
- 达成物理里程碑（某 T 编号首次验证通过、单条 I-t 首次跑通、I-t 斜率首次与实验吻合、
  首个 Weibull 分布）用 `git tag` 打版本号。
- 改动影响到 `README.md` 描述的事实（命令、配置格式、当前范围）时覆盖式更新对应段落。
- **决策记录**：TDDB 的设计取舍沿用项目 ADR 约定（见 `docs/adr/zaz-stack-decisions.md`
  的格式与"superseded by ADR-NNN"标注），建议记在 `docs/adr/tddb-decisions.md`，
  把本文件第 6 节的每条决策落成一条 ADR，便于后续追溯与推翻。

## 代码规范 
- `rates/` 下每个函数写成纯函数：输入物理量和参数，返回速率数值，不依赖
  全局状态、不做 I/O
- 所有函数签名带类型标注
- 可视化函数（`visualization.py`）只接受已经算好的数据，返回 matplotlib
  的 `Figure`，内部不调用 `plt.show()`、不保存文件、不触发仿真——保存还是
  显示由调用方决定

## 常用命令

沿用现有（TDDB 阶段照跑）：

- 测试：`pytest tests/ -v`
- 稳态单点（内层对拍、Phase 0 本底）：`python -m kmc_leakage run --config examples/zro2_tddb.yaml`
- I-V/I-T 扫描：`python -m kmc_leakage sweep --config examples/zro2_tddb.yaml`
- 可视化：`python -m kmc_leakage plot --config examples/zro2_tddb.yaml --kind defects`（或 `bands`）

新增（`tddb.py` 与 CLI 子命令落地后补全，与现有 `run/sweep/plot` 并列）：TDDB 瞬态
运行、I-t / t_BD 系综、I-t / Weibull 出图的命令。落地时同步更新本节与 `CLAUDE.md`
的命令段。
