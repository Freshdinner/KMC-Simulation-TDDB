# ADR-038 的验证数据

- `kmc_baseline.py` → `kmc_head.npz` / `kmc_new.npz`：漏电流侧 KMC（五个单点）与 iv_sweep 扫描的逐位快照，前者用
  `git archive HEAD`（d0cd0d8）的代码，后者用改动后的工作树，131 个数组逐位相同。
- `mf_terminal_check.py`：Phase 4 终态（`../phase4/p4_thermal_s{1..4}.npz`）上的电流连续、放热记账与零偏压检查。
  `terminal_head_s*.log` 是改动前，`terminal_v1_s*.log` 是 ADR-038 的做法。
- `terminal_constant_boundary_s*.log`、`pump_by_pair_class_constant_boundary_s1_*.log`：中间版本（簇与簇外之间的边界
  跃迁一律取平均场常数速率，即用户给的示例做法）的结果，ADR-038"为什么簇–孤立对要按组态求速率"一段引用；
  那一版代码没有提交。
- `mf_vs_kmc_redo.py` → `redo_{head,new}_*.csv`，`redo_summary.py`：ADR-031 对拍在新平均场下重做（KMC 值取
  `../../9.29/tddb_phase2_mf_vs_kmc.csv`）。
