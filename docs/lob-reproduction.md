# LOB 模型架构级重新实现

这一模块用统一接口重新实现 DeepLOB、C(TABL)、BiN 和 DAIN 的核心结构，并提供一个受 HLOB 启发的图模型。目标是把不同方法的归纳偏置拆开、写成可测试代码；它不宣称复刻论文的完整数据处理、训练设置或结果。

## 统一接口

所有模块接收浮点张量：

```text
(batch, time, features)
```

- `DeepLOB`、`CTABL` 和 `HLOBInspired` 输出 `(batch, num_classes)` logits。
- `BiN` 和 `DAIN` 是输入归一化层，输出形状与输入相同。
- 模型不在内部执行 softmax；训练时可直接配合 `CrossEntropyLoss`。
- `CTABL` 和 `BiN` 需要在构造时固定 `time_steps`。

示例：

```python
import torch

from ai_research_portfolio.lob_models import BiN, CTABL

x = torch.randn(16, 100, 40)
normalizer = BiN(feature_dim=40, time_steps=100)
model = CTABL(feature_dim=40, time_steps=100, num_classes=3)

logits = model(normalizer(x))
assert logits.shape == (16, 3)
```

## 各模型保留了什么

| 模型 | 本仓库保留的核心机制 | 没有覆盖的部分 |
|---|---|---|
| DeepLOB | 沿 LOB 字段共享的卷积、多尺度时间卷积、LSTM | 原论文的逐层宽度、通道数、初始化与训练协议 |
| C(TABL) | 特征轴与时间轴可分离的双线性映射、时间注意力、固定注意力矩阵对角线 | 原论文所有层宽、max-norm 训练细节和数据采样协议 |
| BiN | 分别沿时间轴、特征轴做样本级标准化，再用非负权重合并 | 原论文与 TABL 联合训练的完整超参数 |
| DAIN | 自适应 shift、scale、gate 三阶段；支持独立优化器参数组 | 原论文骨干网络、各子层学习率和数据集设置 |
| HLOBInspired | 固定图上的节点消息传递、节点池化、LSTM | MI 建图、TMFG、高阶单纯形卷积及 HLOB 论文结构 |

`HLOBInspired` 的名称刻意保留 Inspired。它只验证关系型盘口建模这一思路，不应写成 HLOB 复现。默认链式图仅供 smoke test；正式实验应从训练集估计邻接矩阵，并将其作为构造参数传入。验证集和测试集不得参与建图。

## 可核验测试

`tests/test_lob_baselines.py` 覆盖：

- 五类模块的输入、输出形状；
- 前向传播后梯度是否存在且为有限值；
- C(TABL) 注意力是否沿时间轴归一化；
- C(TABL) 时间混合矩阵的对角线是否固定为 `1 / T`；
- BiN 两个分支是否分别沿对应轴完成标准化，混合权重是否非负；
- DAIN 的 scale 是否为正、gate 是否落在 `[0, 1]`，三个参数组是否互不重叠；
- 图邻接矩阵是否对称、含自环并完成对称归一化；
- 输入维度不匹配时是否立即报错。

运行方式：

```bash
pytest tests/test_lob_baselines.py
```

## 复现状态

| 模型 | 架构代码 | Shape test | Gradient test | 论文协议训练 | 论文结果对齐 |
|---|---:|---:|---:|---:|---:|
| DeepLOB | 已完成 | 已完成 | 已完成 | 未执行 | 未验证 |
| C(TABL) | 已完成 | 已完成 | 已完成 | 未执行 | 未验证 |
| BiN | 已完成 | 已完成 | 已完成 | 未执行 | 未验证 |
| DAIN | 已完成 | 已完成 | 已完成 | 未执行 | 未验证 |
| HLOBInspired | 已完成 | 已完成 | 已完成 | 不适用 | 不适用 |

仓库目前没有论文协议下的 Benchmark 数字。测试通过只能证明张量接口、梯度和局部数学约束成立，不能证明预测效果与论文一致。后续若加入 FI-2010 实验，应同时固定数据版本、时间切分、标签 horizon、归一化统计范围、随机种子和评价脚本，再报告 macro-F1 的均值与标准差。

## 数据与评测边界

公开仓库不包含券商或交易所原始逐笔数据、模型权重和基于非公开数据生成的逐样本预测。建议把评测分为两层：

1. `paper protocol`：分别还原每篇论文的输入、数据切分和超参数，用于检验实现忠实度。
2. `controlled protocol`：统一信息窗口、切分、随机种子和训练预算，用于比较模型归纳偏置。

两类结果不能混在同一排名表中。HLOB 还依赖按股票、按训练期估计的盘口关系图；没有合法可用的逐日 LOB 数据时，只保留合成数据和架构测试。

## 来源与实现声明

代码根据论文公开描述独立编写，没有复制本地留存的第三方 LOBFrame 源码。历史复现代码缺失，因此这些文件属于新的 clean-room reimplementation，不是原实习期间代码的恢复版。

方法参考：

1. Zhang, Zohren & Roberts, *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books*, 2019。
2. Tran et al., *Temporal Attention-Augmented Bilinear Network for Financial Time-Series Data Analysis*, 2019。
3. Tran et al., *Data Normalization for Bilinear Structures in High-Frequency Financial Time-Series*, 2021。
4. Passalis et al., *Deep Adaptive Input Normalization for Time Series Forecasting*, 2020。
5. Briola, Bartolucci & Aste, *HLOB — Information Persistence and Structure in Limit Order Books*, 2024。
