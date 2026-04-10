# SPY Value-at-Risk Forecasting

使用多种方法预测 SPDR S&P 500 ETF (SPY) 日对数收益率的 Value-at-Risk（τ = 1%, 5%, 10%）。

## 方法

| 类别 | 模型 | 说明 |
|------|------|------|
| 分位数回归 | Quantile Regression | 线性 QR + HAR-RV-J 特征 |
| GARCH | GJR-GARCH-t | 杠杆项 + t 分布 |
| 深度学习 | MLP | Chronopoulos et al. (2024, JFE) Deep Quantile Estimator |
| 深度学习 | LSTM | Buczynski & Chlebus (2023) GARCHNet 架构 |

所有模型采用 500 天滚动窗口，pinball loss 训练，违约率评估。

## 目录结构

```
v2/
├── README.md              # 本文件
├── report.md              # 综合报告（含图表引用）
├── run_all.py             # 统一运行脚本（--group 控制）
├── plot_results.py        # 可视化绘图脚本
├── models/                # 模型实现
│   ├── garch.py           #   GARCH 族
│   ├── historical.py      #   历史模拟法
│   ├── lstm.py            #   LSTM 分位数回归
│   ├── nn.py              #   MLP 分位数回归
│   ├── parametric.py      #   参数法
│   └── quantile_reg.py    #   线性分位数回归
├── utils/                 # 工具函数
│   ├── data.py            #   数据加载与特征构建
│   ├── backtest.py        #   回测评估
│   └── training.py        #   公共训练框架（pinball loss + early stopping）
├── tests/                 # 测试套件（103 个测试）
├── results/               # 实验产出
│   ├── *.csv              #   VaR 预测结果
│   ├── *_meta.json        #   模型元数据
│   ├── *_search_log.csv   #   超参搜索日志
│   ├── checkpoints/       #   模型权重
│   └── *.png              #   可视化图表
└── spy_data.csv -> ../spy_data.csv  # 数据（符号链接）
```

## 环境配置

**前提**：Python >= 3.11，推荐使用 [uv](https://docs.astral.sh/uv/)。

```bash
# 1. 克隆项目
git clone <repo-url> && cd <repo-dir>

# 2. 创建虚拟环境并安装依赖
uv venv .venv
uv pip install -e ".[dev]"

# 3. 验证环境
.venv/bin/python -c "import torch, arch, statsmodels; print('OK')"
```

如果不使用 uv，也可以用 pip：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 复现步骤

### Step 1：运行测试（约 2 分钟）

```bash
.venv/bin/pytest v2/tests -v
```

应输出 `103 passed`。

### Step 2：运行全部模型（约 100 分钟 CPU）

```bash
# 运行全部（按顺序执行 traditional → garch500 → mlp → lstm）
.venv/bin/python v2/run_all.py

# 或按组分别运行：
.venv/bin/python v2/run_all.py --group traditional   # ~1 min
.venv/bin/python v2/run_all.py --group garch500      # ~5 min
.venv/bin/python v2/run_all.py --group mlp           # ~30 min
.venv/bin/python v2/run_all.py --group lstm          # ~60 min
```

产出写入 `v2/results/`（CSV 预测结果 + JSON 元数据 + 模型 checkpoints）。

> **注意**：超参搜索不包含在默认运行中（耗时过长）。如需复现搜索过程：
> ```bash
> .venv/bin/python v2/run_all.py --group search       # ~数小时
> ```

### Step 3：生成可视化图表（约 10 秒）

```bash
.venv/bin/python v2/plot_results.py
```

生成 5 张 PNG 到 `v2/results/`：

| 图表 | 文件 | 报告章节 |
|------|------|----------|
| EDA 时序+分布 | `eda_returns.png` | §2 |
| 违约率柱状图 | `violation_rates.png` | §5.1 |
| VaR 时序分面 | `var_timeseries_tau0.05.png` | §5.2 |
| VaR 叠加对比 | `var_overlay_tau0.01.png` | §5.2 |
| 分位数交叉 | `quantile_crossing.png` | §5.3 |

### Step 4：查看报告

报告为 Markdown 格式，图表通过相对路径引用：

```bash
# 直接阅读
cat v2/report.md

# 或用 Markdown 预览工具打开（如 VS Code、Obsidian）
```

## 完整复现命令（一键）

```bash
# 从零开始完整复现
uv venv .venv && uv pip install -e ".[dev]" \
  && .venv/bin/pytest v2/tests -v \
  && .venv/bin/python v2/run_all.py \
  && .venv/bin/python v2/plot_results.py
```

## 核心结果

| 模型 | τ=1% | τ=5% | τ=10% |
|------|------|------|-------|
| GJR-GARCH-t | 1.40% | 6.28% | 10.92% |
| Quantile Regression | 1.82% | 5.10% | 9.57% |
| MLP | 1.64% | 4.42% | 8.45% |
| **LSTM** | **1.09%** | **5.17%** | **9.18%** |
| 理论值 | 1.0% | 5.0% | 10.0% |

详细分析见 `report.md`。
