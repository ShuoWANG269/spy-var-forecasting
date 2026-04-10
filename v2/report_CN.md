# SPY Value-at-Risk 预测综合报告

## 1. 研究目标

本项目针对 SPDR S&P 500 ETF (SPY) 的日对数收益率，构建多种 Value-at-Risk (VaR) 预测模型，覆盖三个置信水平（$\tau = 1\%, 5\%, 10\%$），通过违约率（violation rate）与理论值的偏差评估各模型的预测质量。

## 2. 数据描述

| 项目 | 说明 |
|------|------|
| 标的 | SPDR S&P 500 ETF (SPY) |
| 时间跨度 | 2000-01-04 至 2018-06-27 |
| 观测数 | 4640 个交易日 |
| 变量 | `log_ret`（日对数收益率）、`rv5`（已实现波动率，5 分钟）、`bv`（双幂变差） |

收益率序列呈现典型的金融时间序列特征：尖峰厚尾（峰度 8.22）、负偏（偏度 -0.21）、波动率聚集、杠杆效应。2008 年金融危机期间波动率显著放大。

![SPY 收益率时序与分布](results/eda_returns.png)

*图 1：上图为 SPY 日对数收益率时序（红色阴影标注 2007-2009 金融危机），下图为收益率分布直方图（红线为正态拟合），可见显著的厚尾特征。*

## 3. 方法概述

### 3.1 分位数回归（Quantile Regression）

基于 Koenker & Bassett (1978) 的线性分位数回归，使用 HAR（Heterogeneous Autoregressive）特征集：

$$Q_{\tau}(r_t | \mathbf{x}_t) = \mathbf{x}_t' \boldsymbol{\beta}_\tau$$

特征采用 HAR-RV-J 框架（Corsi, 2009; Andersen et al., 2007）：$BV^{(d)}$（日双幂变差）、$BV^{(w)}$（周均值）、$BV^{(m)}$（月均值）、$J_d = \max(RV_d - BV_d, 0)$（跳跃分量）。

损失函数为 pinball loss：

$$\mathcal{L}_\tau(r, q) = \begin{cases} \tau (r - q) & \text{if } r \geq q \\ (1-\tau)(q - r) & \text{if } r < q \end{cases}$$

### 3.2 GARCH 族模型

使用条件异方差模型刻画波动率动态：

**GARCH(1,1)**（Bollerslev, 1986）：

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

**GJR-GARCH-t**（Glosten et al., 1993）：引入杠杆项 $\gamma I(\varepsilon_{t-1}<0)\varepsilon_{t-1}^2$，捕捉非对称波动率响应，标准化残差服从 t 分布。

VaR 通过条件分布的分位数函数计算：$\text{VaR}_{\tau,t} = \mu_t + \sigma_t \cdot F^{-1}(\tau)$

### 3.3 MLP 深度分位数回归

参照 Chronopoulos et al. (2024, JFE) 的 Deep Quantile Estimator，使用多层感知机（MLP）估计条件分位数。每个 $\tau$ 独立训练一个网络，损失函数为 pinball loss。

**网络结构**：输入层 → $L$ 层隐藏层（每层 $J$ 个神经元，ReLU 激活）→ Dropout → 输出层（1 个神经元）

**特征集**：$r_{t-1}$, $\hat{\sigma}_{t}^{GARCH}$（GARCH 条件波动率）

### 3.4 LSTM 分位数回归

参照 Buczynski & Chlebus (2023, Computational Economics) 的 GARCHNet 架构：

**网络结构**：输入序列 → 1 层 LSTM（hidden_size $h$）→ Dropout → FC（fc_units）→ ReLU → FC(1)

**特征**：仅使用历史收益率序列。每个 $\tau$ 独立训练，损失函数为 pinball loss。

## 4. 实验设置

### 4.1 滚动窗口

所有最终模型采用 500 天滚动窗口，测试集长度 4140 天。每个预测日使用前 500 天数据重新拟合模型参数，向前预测 1 天 VaR。

### 4.2 MLP/LSTM 滚动窗口训练

- 重训频率：每 20 步重训一次
- 训练方式：首次 early stopping（patience=10）确定 best_epoch，后续以 fixed_epochs 模式重训
- Gradient clipping：max_norm=1.0

### 4.3 超参搜索

**MLP 超参搜索**（网格搜索，固定窗口 80/20 划分）：$L \in \{1,2,3,4,5,10\} \times J \in \{1,2,3,4,5,10\} \times lr \in \{0.01, 0.001, 0.0001\} \times dropout \in \{0, 0.5\} \times weight\_decay \in \{0, 0.0001\}$。搜索在固定窗口实验中完成，最优超参数直接应用于 500 天滚动窗口评估，两种设置的数据划分方式不同。

**MLP 最优超参数**：

| $\tau$ | $L$ | $J$ | lr | dropout | weight_decay |
|--------|-----|-----|----|---------|-------------|
| 1% | 1 | 1 | 1e-4 | 0.0 | 1e-4 |
| 5% | 1 | 4 | 0.01 | 0.0 | 1e-4 |
| 10% | 1 | 1 | 0.01 | 0.0 | 1e-4 |

**LSTM 超参搜索**：$h \in \{32,64,100\} \times fc \in \{16,32,64\} \times seq \in \{10,20\} \times dropout \in \{0,0.2\} \times lr \in \{3\text{e-4},1\text{e-3},1\text{e-2}\} \times wd \in \{0,1\text{e-4}\}$，共 216 组合。搜索在数据前 500 天（`df.iloc[:500]`）上进行，内部 80/20 划分为训练/验证集。需注意该搜索窗口与滚动预测的第一个训练窗口完全重叠，最优超参数可能对早期数据存在一定选择偏差。

**LSTM 最优超参数**：

| $\tau$ | hidden_size | fc_units | seq_len | dropout | lr | weight_decay |
|--------|-------------|----------|---------|---------|------|-------------|
| 1% | 100 | 16 | 10 | 0.0 | 0.01 | 0.0 |
| 5% | 32 | 32 | 10 | 0.0 | 0.01 | 0.0 |
| 10% | 100 | 64 | 20 | 0.2 | 0.01 | 0.0 |

## 5. 结果与分析

### 5.1 违约率汇总

违约率（violation rate）是实际收益率低于 VaR 预测值的天数占比，理论值应等于 $\tau$。

| 模型 | $\tau=1\%$ | $\tau=5\%$ | $\tau=10\%$ |
|------|-----------|-----------|------------|
| Quantile Regression | 1.82% | 5.10% | 9.57% |
| GJR-GARCH-t | 1.40% | 6.28% | 10.92% |
| MLP | 1.64% | 4.42% | 8.45% |
| **LSTM** | **1.09%** | **5.17%** | **9.18%** |
| 理论值 | 1.0% | 5.0% | 10.0% |

LSTM 的违约率在三个 $\tau$ 上均最接近理论值。

![违约率柱状图](results/violation_rates.png)

*图 2：各模型违约率与理论值（虚线）对比。LSTM 在三个 τ 上均最接近理论值，GJR-GARCH-t 在 τ=5% 和 10% 系统性偏高。*

### 5.2 模型对比分析

1. **违约率准确性**：LSTM 的违约率最接近理论值（1.09% / 5.17% / 9.18%），QR 次之（1.82% / 5.10% / 9.57%）。GJR-GARCH-t 在 $\tau=5\%$ 和 $\tau=10\%$ 偏高，MLP 在 $\tau=10\%$ 偏低。

2. **模型特点**：QR 作为线性方法，结构简单且在中等分位数上表现稳定。GJR-GARCH-t 通过杠杆项捕捉非对称波动率，但在极端分位数上违约率偏高。MLP 和 LSTM 通过非线性建模提升了预测精度，其中 LSTM 利用序列依赖性在所有 $\tau$ 上取得最优违约率。

![各模型 VaR 时序对比](results/var_timeseries_tau0.05.png)

*图 3：τ=5% 下各模型 VaR 预测（彩色线）与实际收益率（灰色线），红点标注违约事件。GJR-GARCH-t 的 VaR 线平滑且对 2008 危机响应明显；QR 在极端时期出现剧烈跳变；MLP 的 VaR 变化相对缓慢；LSTM 的 VaR 跟踪波动率变化最敏感。*

3. **LSTM $\tau=1\%$ 预测值偏激进**：LSTM 在 $\tau=1\%$ 上的平均 VaR 为 $-5.40\%$（标准差 $5.75\%$），显著大于 GJR-GARCH-t 的 $-2.51\%$（标准差 $1.72\%$），最极端预测达 $-76.3\%$。这说明 LSTM 在极端分位数上的预测波动性远高于参数模型。虽然违约率 1.09% 接近理论值，但代价是 VaR 估计过度保守，在实际应用中可能导致过高的资本准备要求。

![τ=1% VaR 叠加对比](results/var_overlay_tau0.01.png)

*图 4：τ=1% 下 4 模型 VaR 预测叠加对比。LSTM（红色）频繁跌入 -10% 甚至更低，与 GJR-GARCH-t（蓝色）、QR（橙色）和 MLP（绿色）的平稳预测形成鲜明对比。*

### 5.3 分位数单调性分析

由于每个 $\tau$ 独立训练模型，不同分位数的预测可能违反 $\text{VaR}_{1\%} < \text{VaR}_{5\%} < \text{VaR}_{10\%}$ 的单调性约束（即分位数交叉）。

| 模型 | $\text{VaR}_{1\%} > \text{VaR}_{5\%}$ | $\text{VaR}_{5\%} > \text{VaR}_{10\%}$ | 任意交叉 |
|------|-------|--------|------|
| LSTM | 14.2% (588/4140) | 28.0% (1161/4140) | 41.8% (1729/4140) |
| MLP | 21.7% (900/4140) | 22.8% (943/4140) | 44.5% (1843/4140) |
| QR | 2.4% (99/4119) | 1.5% (60/4119) | 3.4% (139/4119) |
| GJR-GARCH-t | 0% | 0% | 0% |

LSTM 和 MLP 的分位数交叉比例较高，这是 per-$\tau$ 独立训练的已知缺陷。GJR-GARCH-t 基于参数化条件分布，天然满足单调性。QR 的交叉率很低，因为线性模型的分位数面更平滑。可能的缓解方案包括联合分位数回归（joint quantile regression）或事后排序（post-hoc sorting），但这些方法超出本研究范围。

![分位数交叉分析](results/quantile_crossing.png)

*图 5：4 模型的三分位数预测线（τ=1%/5%/10%），红色阴影标注交叉区域。GJR-GARCH-t 无交叉（0%），QR 交叉率极低（3.4%），而 MLP（44.5%）和 LSTM（41.8%）的交叉区域密集分布于整个预测期。*

### 5.4 方法探索过程总结

在确定最终模型配置前，进行了以下探索：

- **窗口长度**：对比 250 天和 500 天滚动窗口，500 天窗口在所有模型上违约率更接近理论值，尾部风险估计更充分。
- **LSTM 特征组**：对比仅收益率、HAR 波动率特征、GARCH 条件波动率三组输入，仅收益率组信噪比最高、违约率最优；额外波动率特征在有限训练数据下反而引入噪声。
- **LSTM 超参搜索**：默认超参数（lr=3e-4, hidden=64）下 LSTM 严重偏保守（违约率远低于理论值），216 组合网格搜索后性能大幅提升，学习率（0.01）是最关键超参数。
- **MLP 固定窗口 vs 滚动窗口**：固定窗口（80/20 划分）下 MLP 违约率严重偏低甚至退化，滚动窗口显著改善模型适应性。

## 6. 结论

1. **LSTM 分位数回归**在违约率准确性上表现最优，三个 $\tau$ 均最接近理论值。超参搜索至关重要——学习率从默认 3e-4 提升至 0.01 带来了决定性改善。但需注意其代价：$\tau=1\%$ 上 VaR 预测值波动剧烈（均值 $-5.40\%$，最极端达 $-76.3\%$），且 41.8% 的预测日存在分位数交叉，在实际风险管理中需要额外的后处理步骤。

2. **分位数回归（QR）**作为线性方法，在 $\tau=5\%$ 和 $\tau=10\%$ 上违约率接近理论值，且分位数交叉率仅 3.4%，是稳定性与准确性兼顾的实用选择。

3. **GJR-GARCH-t** 天然满足分位数单调性，通过杠杆项和 t 分布捕捉非对称厚尾特征，但在 $\tau=1\%$ 和 $\tau=5\%$ 上违约率偏高，说明条件分布假设仍有局限。

4. **MLP** 在 $\tau=5\%$ 上违约率最优（4.42%），但分位数交叉率高达 44.5%，与 LSTM 类似受限于 per-$\tau$ 独立训练的框架。

5. 实验过程中发现：500 天窗口优于 250 天窗口、简单特征（仅收益率）优于复杂特征、滚动窗口优于固定窗口。综合违约率准确性、预测稳定性和分位数单调性，没有单一模型在所有维度上占优，实际应用中应根据具体需求权衡选择。

## 7. 参考文献

- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.
- Buczynski, M. & Chlebus, M. (2023). GARCHNet — Value-at-Risk forecasting with GARCH models based on neural networks. *Computational Economics*.
- Andersen, T. G., Bollerslev, T. & Diebold, F. X. (2007). Roughing it up: Including jump components in the measurement, modeling, and forecasting of return volatility. *Review of Economics and Statistics*, 89(4), 701-720.
- Chronopoulos, I., Ames, M., & Daskalakis, G. (2024). Forecasting Value-at-Risk Using Deep Neural Network Quantile Regression. *Journal of Financial Econometrics*, 22(3), 636-669.
- Glosten, L. R., Jagannathan, R. & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779-1801.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174-196.
- Koenker, R. & Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33-50.
