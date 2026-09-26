# Claude Code Context
**重要上下文提示**
1. 请优先加载参考 @PROJECT_CONTEXT.md（相对路径：./PROJECT_CONTEXT.md）作为项目背景

## 输出风格
把输出的文本按说人话规则轻改，保留术语和系统主语，不要改成口语闲聊体。

## 代码规范 
- `rates/` 下每个函数写成纯函数：输入物理量和参数，返回速率数值，不依赖
  全局状态、不做 I/O
- 所有函数签名带类型标注
- 公开函数的 docstring 标注对应 `docs/rate_equations.md` 里的编号，例如
  `"""R2: 电极→陷阱弹性俘获，见 docs/rate_equations.md。"""`
- 可视化函数（`visualization.py`）只接受已经算好的数据，返回 matplotlib
  的 `Figure`，内部不调用 `plt.show()`、不保存文件、不触发仿真——保存还是
  显示由调用方决定

## 测试要求
- 每新增一个速率函数，至少配一个验证测试：优先对照已知解析极限（比如 PF
  公式高场极限应退化为简单指数近似、零场极限应退化为 ν·exp(-ED/k_BT)），
  没有解析解的用物理约束（比如 R4/R5 之间的细致平衡关系）
- R1 拆分出的"热电子发射"子部分，需要和 Richardson-Schottky 解析公式
  （见 rate_equations.md）做对比，作为独立检验
- 按机制、按缺陷位置分解统计后，各分量之和必须等于总电流——这是天然的
  一致性检验，务必写成测试
- 新模块合并前必须跑通 `pytest tests/ -v`，不接受"看起来能跑"作为完成
  标准，要贴出测试通过的输出

## 常用命令
- 测试：`pytest tests/ -v`
- 单点仿真：`python -m kmc_leakage run --config examples/single_run.yaml`
- I-V/I-T 扫描：`python -m kmc_leakage sweep --config examples/iv_sweep.yaml`
- 可视化：`python -m kmc_leakage plot --config examples/single_run.yaml --kind defects`
  （`--kind` 取 `defects` 或 `bands`，后续新增绘图项在这里扩展）

## 变更记录与提交规范
完成一次逻辑完整的修改后(不是每个中间步骤):
1. 在 CHANGELOG.md 的 [未发布] 节追加条目:改了什么、为什么、涉及哪个 R 编号或物理文档、pytest 是否通过
2. git commit,message 格式 `<模块>: <一句话>`,如 `rates/r7: 实现 Miller-Abrahams 跳跃模型`, user.name: "appetizingbrunch" , user.email: "appetizingbrunch@gmail.com" 
3. 达成一个物理里程碑(某个 R 编号首次验证通过、新增缺陷分布模型等)时用 git tag 打版本号
4. 如果改动影响了 README.md 里描述的事实(命令、配置格式、依赖版本、当前范围),同步覆盖式更新对应段落,不要在 README 里追加变更说明。
完整变更历史见 CHANGELOG.md