# AI Research Portfolio

彭德东｜中山大学岭南学院金融学主修、统计学辅修

这是一份面向 AI 与前沿科技投资岗位的公开研究作品集。仓库集中展示三类能力：前沿方法复现与评测、可审计 Agent 工具链开发，以及将技术差异转化为产品和投资判断。

## 作品导航

| 模块 | 研究问题 | 可核验产出 |
|---|---|---|
| [DeepGELOB](docs/deepgelob.md) | 标准价量 LOB 是否遗漏队列组成、订单老化与事件强度信息？ | 论文状态说明、语义分组模型、标签审计、合成数据 smoke test |
| [LOB Reproduction Lab](docs/lob-reproduction.md) | DeepLOB、C(TABL)、BiN、DAIN 与 HLOB 分别引入了什么归纳偏置？ | 统一输入接口、架构级重新实现、shape/gradient tests、复现边界说明 |
| [Research Agent](docs/research-agent.md) | 如何把研究流程拆成可暂停、可复核、可恢复的工作流？ | LangGraph `StateGraph` 适配器、人工审批门、离线 deterministic demo、测试 |
| [Agent 技术与投资研究样稿](docs/agent-investment-sample.md) | Agent 价值链中，哪些环节更可能形成产品壁垒？ | 技术栈地图、评测框架、商业化假设、尽调问题清单 |

## 与岗位要求的对应关系

- **跟踪论文、开源与 Benchmark：** 将模型按问题假设、数据协议、评价口径和适用边界拆解，而不是只复述网络结构。
- **复现前沿方法：** 用统一接口重新实现多类 LOB 模型，并通过合成数据测试前向传播、梯度和输出形状。缺失的历史代码均标为 `re-implementation`，不冒充原始版本。
- **Agent 工具链：** 将问题拆解、实验规划、代码修改、运行、指标收集和复盘组织成显式状态图；高风险动作进入人工审批节点。
- **技术转商业判断：** 研究样稿从模型能力延伸到部署约束、评测成本、客户集成、单位经济性与尽调验证问题。

## 快速运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
python examples/run_lob_smoke_test.py
python examples/run_research_agent.py
```

`Research Agent` 的基础演示不调用外部模型，也不需要 API Key；它使用 deterministic adapters 展示工作流与审计记录。若要接入 LangGraph：

```bash
pip install -e '.[agent,test]'
python examples/run_langgraph_agent.py
```

## 仓库结构

```text
.
├── docs/                         # 项目说明与原创研究样稿
├── examples/                     # 可直接运行的离线示例
├── paper/                        # DeepGELOB 论文样稿
├── src/ai_research_portfolio/
│   ├── lob_models/               # DeepGELOB 与模型重新实现
│   └── research_agent/           # 研究 Agent 状态、节点和执行器
└── tests/                        # shape、梯度、路由和审计测试
```

## 公开边界与研究诚信

1. 仓库不包含券商、交易所或课题组的原始数据、客户材料、模型权重及非公开实验记录。
2. DeepGELOB 代码保留可公开的模型、目标构造和评测核心；数据接入层改用明确 schema 与合成样例。
3. DAIN、HLOB 等丢失的历史代码按论文重新实现，属于架构级复现。仓库不宣称复刻原论文全部训练设置，也不补造历史结果。
4. 研究样稿由作者基于公开资料独立整理，不是课题组报告的删减版；其中商业判断属于待验证假设。
5. 仓库中的代码采用 MIT License；论文与研究样稿保留作者署名及各自说明。

## 联系方式

- 彭德东
- Email：peng_dedong@qq.com
