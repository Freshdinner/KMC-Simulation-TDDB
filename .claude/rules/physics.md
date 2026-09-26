---
paths:
  - "src/kmc_leakage/rates/**"
  - "src/kmc_leakage/electrostatics.py"
  - "tests/test_rates_limits.py"
  - "tests/test_electrostatics.py"
---

# 物理公式规格（按需引入）

改这里的文件时，以下规格文档自动加入上下文：

@../../docs/physics_model.md

@../../docs/rate_equations.md

## 关键提醒
- ε_opt 只用于 PF 发射（R8, rates/pf_emission.py），κ 只用于泊松方程求解
  （electrostatics.py），不要混用
- **不考虑镜像力势垒降低**：势垒按零场高度 E_B 处理，界面处能带保持直线，
  WKB 积分用未经圆化的直线/分段线性势垒——ε_opt 出现在 R8 以外的任何速率
  公式里都是写错了

