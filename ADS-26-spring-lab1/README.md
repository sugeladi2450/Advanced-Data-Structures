# Lab1：图数据库与 AI Skill 集成

## 1. 背景与动机

### 1.1 软件依赖

软件依赖（dependency）广泛存在于各类生态中，是**自动安装与版本管理**的核心工具：

- **Debian/Ubuntu**：`apt`、`dpkg` 管理的包依赖（本 Lab 使用的数据来源）
- **Python**：PyPI + `pip` / `uv` 的包依赖
- **Rust**：crates.io + Cargo 的 crate 依赖
- **Node.js**：npm / yarn 的 package 依赖
- **Java**：Maven Central、Gradle 等

现代操作系统和开发环境中的**自动安装与版本管理**，正是依赖于社区维护的庞大包依赖关系图来实现正确的安装顺序和版本约束；同时，**软件供应链安全分析**（判断软件是否间接依赖有漏洞或恶意的包）也是软件依赖关系的重要应用场景。

因此，本 Lab 选用 **Debian 软件包依赖图** 作为数据，在自建的图数据库上做查询与推理。

### 1.2 图数据库

**图数据库**以顶点（Node）和边（Edge）为核心抽象，针对图上的遍历、最短路径、子图匹配、中心性等操作做了存储与执行优化，是关系型与文档型数据库的重要补充。

**主流开源图数据库：**

- **Neo4j**：属性图模型，Cypher 查询语言，常用于知识图谱、推荐、风控。
- **MemGraph**：兼容 Cypher 的实时图数据库，以内存为主、支持持久化，侧重流式图分析与低延迟查询，常用于实时推荐、欺诈检测。
- **TigerGraph**：强调实时图分析与并行计算。

虽然三种数据库的核心特性不同，但它们的共同点是都具备**持久化能力**，并且支持图的很多**基础操作**（如邻接遍历、路径查询、图算法等）。

对这些场景，图数据库通常需要提供：**高效的邻接遍历与索引**、**图上的算法算子**（BFS/DFS、最短路径、PageRank 等）、**持久化与事务**（含 WAL/快照等恢复机制），以及**可扩展的存储与查询接口**。

本 Lab 从「内存图 + WAL(Write-Ahead-Log) + Snapshot + 基本图算法」入手，正是这些能力的简化实践。

### 1.3 Skills 与渐进式披露

**Skill** 是继 Tool Calling、MCP（Model Context Protocol）之后的一种常见抽象：把「能做什么、怎么调」写成结构化的说明（如 Markdown + 函数描述），供 Agent 在需要时查阅并调用。

这符合 AI 领域里的一种常见思路：**给 Agent 一张地图，而不是一本 1000 页的说明书**——按需披露、按任务选能力，也是 OpenClaw 等 Agent 框架能够整合大量工具并保持可扩展的重要原因。

在本 Lab 中，**Skill** 具体指：将图数据库的查询能力封装成 LLM 可调用的工具，并配合 Skill 的 Markdown 文档与 Agent 循环，让大模型根据用户的自然语言任务自动选择工具、多轮调用、汇总答案。通过这种**渐进式披露**，Agent 可以按任务选工具、多轮调用并汇总结果，而无需在 prompt 中塞入全部实现细节。

---

## 2. 任务总览

本 Lab 分四个阶段，建议按顺序完成（与 `test_phase1.py`～`test_phase4.py` 一一对应）：

1. **Phase 1：内存图结构**  
   实现仅存在于内存中的图数据库核心：插入/删除节点与边、查询邻居与属性。不涉及落盘与恢复。

2. **Phase 2：持久化与恢复**  
   在 Phase 1 的基础上增加预写式日志与快照，实现落盘与断电恢复能力。

3. **Phase 3：图遍历算法**  
   在图库上实现最短路径、全部依赖（传递闭包）、**按层依赖**（限制层数以防查询时延爆炸）等软件依赖场景下需要且常见的算法，供后续 Skill 层调用。

4. **Phase 4：Skill 集成**  
   在图后端和 `skills/graph_query_skills.py` 中实现函数与 description，让 LLM 通过已提供的 Skill 文档与 Agent 循环自动完成依赖调查等复杂任务。

数据使用统一的图数据文件 `data/graph.json`（已提供），内含 Debian 软件包节点与依赖边。

---

## 3. 项目结构与数据说明

### 3.1 项目文件结构

```
ADS-26-spring-lab1/
├── graph/
│   └── graph_db.py          ← 需实现（Phase 1–3）
├── data/
│   ├── graph.json           ← 已提供，勿修改
│   ├── deps_answer.json     ← 已提供，测试答案
│   └── load_graph_json.py   ← 需实现（Phase 1，整图加载）
├── skills/
│   ├── graph_query_skills.py  ← 需实现（Phase 4）
│   ├── registry.py            ← 已提供，勿修改
│   ├── runtime.py             ← 已提供，勿修改
│   └── graph-query-skills/
│       └── SKILL.md           ← 已提供，Skill 说明文档
├── agent/
│   └── loop.py              ← 已提供；需在此设置 API_KEY
├── tests/
│   ├── test_phase1.py       ← Phase 1 测试
│   ├── test_phase2.py       ← Phase 2 测试
│   ├── test_phase3.py       ← Phase 3 测试
│   └── test_phase4.py       ← Phase 4 测试
└── report.md                ← 需提交
```

### 3.2 数据说明

- **graph.json**：已完成 Debian 软件包依赖关系的提取与简化，置于 `data/` 目录下。你最终建立的图便基于该数据。
- **格式**：
  - `nodes`（包名列表）
  - `edges`（`[{"from": "pkg1", "to": "pkg2", "weight": 1.0}, ...]`）。为简化实现，本 Lab **省略边的 type 属性**，仅保留边权 `weight`（默认统一为 1.0）。

---

## 4. 任务与实现要求

### Phase 0：环境配置

- **安装依赖**：在项目根目录执行（二选一）  
  - Mac / Linux：`uv sync && source .venv/bin/activate`  
  - Windows（CMD 或 PowerShell）：`uv sync`，然后执行 `.venv\Scripts\activate`
- **API key**：Phase 4 的 Agent 测试需调用 LLM。在 [交大 API 申请页](https://form.sjtu.edu.cn/infoplus/form/net_ai_api_apply/start?locale=zh) 获取 API key 后，在 `agent/loop.py` 中设置 `API_KEY`。

---

### Phase 1：内存图结构（约 30%）

**需修改文件：**
- `graph/graph_db.py`：实现 `GraphDB` 类的核心接口
- `data/load_graph_json.py`：实现整图加载函数

本阶段只实现**完全驻留内存**的图结构，不实现持久化。`GraphDB.__init__` 已提供内存结构声明，**本阶段无需修改 `__init__`**，直接实现各操作方法即可。

**核心接口：**

**重要：** 以下函数声明**不得修改**，修改函数签名和返回值语义可能会导致测试报错：

- `insert_node(node_id, properties)`: 插入节点，可附带属性。O(1)。
- `insert_edge(from_id, to_id, weight)`: 插入有向边。
- `delete_node(node_id)`: 删除节点及其关联的所有边。
- `delete_edge(from_id, to_id)`: 删除指定的边。
- `has_node(node_id)`: 判断节点是否存在。
- `get_node_properties(node_id)`: 获取节点属性（不存在返回 None）。
- `get_neighbors(node_id)`: 查询指定节点的所有出边邻居 `[(to_id, weight), ...]`。

**实现提示：**
- 推荐同时维护出边表 `_adj`（`from_id → {to_id: weight}`）和入边反向索引 `_radj`（`to_id → {from_id, ...}`）。`delete_node` 需要同时清理出边和入边，若没有反向索引则须遍历全部节点寻找入边，复杂度为 O(V + E)；有反向索引后可降至 O(出度 + 入度)。

**测试：** `test_phase1.py` 验证插入、删除、查询及整图加载后的邻接查询。

---

### Phase 2：持久化与恢复（约 30%）

**需修改文件：**
- `graph/graph_db.py`：在 Phase 1 基础上增加 Write-Ahead-Log(WAL) 写入、Snapshot 生成、故障恢复逻辑

使用 **WAL (Write-Ahead Log) + Snapshot** 实现落盘与断电恢复。

**路径约定（不可修改）：**
- **WAL 路径**：`{db_dir}/graph.log`（即 `graph_db` 内常量 `WAL_FILENAME = "graph.log"`）。
- **Snapshot 路径**：`{db_dir}/snapshot.dat`（即 `SNAPSHOT_FILENAME = "snapshot.dat"`）。

**实现要求：**

1. **WAL（预写式日志）**
   - 任何修改操作在更新内存之前，先以 **Append-Only** 方式写入 `graph.log`。
   - 日志格式**由你自行设计**，只要 `recover()` 能够正确解析并重放即可。

2. **Snapshot（快照）**
   - 当日志条数达到阈值（如 1000 条，见 `graph_db.WAL_THRESHOLD`）时，触发快照，将整个图序列化到 `snapshot.dat`。
   - 快照成功后清空 `graph.log`。

3. **恢复（Recovery）**
   - 重启时先读 `snapshot.dat` 恢复基础状态，再逐行重放 `graph.log` 恢复到最新状态。

**Phase 2 的实现步骤建议：**
1. 实现 `_write_wal(record)` —— Append-Only 写入 + 计数 + 触发阈值检查。
2. 在 Phase 1 的各写操作（`insert_node`、`insert_edge`、`delete_node`、`delete_edge`）中，更新内存前先调用 `_write_wal(...)`。
3. 实现 `_maybe_snapshot()` —— 达到阈值时调用 `save_snapshot()`。
4. 实现 `save_snapshot()` —— 序列化内存图、写文件、清空 WAL。
5. 实现 `recover()` —— 加载 snapshot（若存在），再逐行重放 WAL。
6. 实现 `close()` —— 调用 `save_snapshot()`。
7. 在 `__init__` 末尾取消注释 `self.recover()`。

**重要：** 以下函数声明**不得修改**，修改函数签名和返回值语义可能会导致测试报错：
- `GraphDB.__init__(self, db_dir)`：初始化时从 snapshot + WAL 恢复
- `GraphDB.close(self)`：关闭时触发 snapshot
- `GraphDB.save_snapshot(self)`：手动触发 snapshot
- `GraphDB.recover(self)`：从 snapshot + WAL 恢复
- `GraphDB.clear_memory(self)`（已提供）：模拟断电，清空内存中的图数据
- `GraphDB.wal_exists(self)`（已提供）：检查 WAL 文件是否存在
- `GraphDB.snapshot_exists(self)`（已提供）：检查 snapshot 文件是否存在

**测试：** `test_phase2.py` 检查 WAL 写入、snapshot 生成与清空、关闭后恢复、模拟断电后 `recover()` 等。

---

### Phase 3：图遍历算法（约 20%）

**需修改文件：**
- `graph/graph_db.py`：在 Phase 2 基础上增加图遍历算法接口

在图库上实现以下接口（供 Phase 4 的 Skill 调用）：

**重要：** 以下函数声明**不得修改**，修改函数签名和返回值语义可能会导致测试报错：

- **`find_shortest_path(start_id, target_id)`**：最短跳数路径。
  - 返回格式：`[(from, to, weight), ...]` 的边列表，或 `None`（无路径），或 `[]`（起点等于终点）。
  
- **`get_all_dependencies(node_id)`**：从该节点出发能到达的**全部依赖节点**（传递闭包）。
  - 返回格式：`[依赖包名, ...]` 列表。
  
- **`get_dependencies_by_layer(node_id, max_layers)`**（**必须提供**）：按层返回依赖。
  - 这是软件依赖场景中相当重要的函数，因为不按照依赖深度查询所有相关包可能导致查询开销爆炸。
  - 第 1 层为直接依赖，第 2 层为间接依赖，以此类推。
  - **最多返回 `max_layers` 层**，不限制层数会导致查询开销爆炸。
  - 返回格式：`{1: [包名, ...], 2: [...], ...}`。

**测试：** `test_phase3.py` 验证最短路径、全部依赖、按层依赖等算法实现的正确性。

---

### Phase 4：Skill 集成（约 10%）

你需要在这个阶段理解 Skill 的工作原理，代码实现并不多。

**需修改文件：**
- `skills/graph_query_skills.py`：实现四个工具函数的registry 及 description

将图数据库查询封装为 LLM 可调用的 Skills，让 Agent 通过「观察–思考–行动」完成较复杂自然语言任务（如漏洞依赖调查）。

**框架已提供（无需修改）：** Skill 的 Markdown 文件（`skills/graph-query-skills/SKILL.md`）、Agent 循环（`agent/loop.py`）、Skill 注册器（`skills/registry.py`）与执行机制（`skills/runtime.py`）。

**你需要做的：**

在 `skills/graph_query_skills.py` 中补全四个工具的注册与函数实现，使每个工具调用图后端对应方法。调用时由 SkillRuntime 解析 Agent 的 tool_call 并执行你注册的函数，无需你实现解析与执行逻辑。

**提示：** 注册时的 `name` 与 `parameters` 的字段名须与函数声明一致。

**重要：** `get_neighbors` 示例中的注册格式（`@registry.register(...)` 写法、函数签名与返回语义）建议保持不变，并作为其余三个工具的参考模板。

以下函数**已提供**，**不得修改**：
- `set_graph_backend(backend)`（已提供）：注入图后端供工具调用
- `get_graph_backend()`（已提供）：获取当前注入的图后端

**测试：** `test_phase4.py` 验证 Agent 在给定漏洞库与核心包列表下，能正确调用工具并输出依赖该漏洞库的核心包。Skill 的 Markdown（`skills/graph-query-skills/SKILL.md`）和 Agent 循环已写好，理解「手册 + 渐进式披露」如何让 LLM 自动完成自然语言任务即可。

---

## 5. 如何运行测试

确保已激活虚拟环境（见 Phase 0），然后在项目根目录执行：

```bash
# 运行单个阶段的测试
pytest tests/test_phase1.py -v
pytest tests/test_phase2.py -v
pytest tests/test_phase3.py -v
pytest tests/test_phase4.py -v

# 运行全部测试
pytest tests/ -v

# 运行某个具体测试函数
pytest tests/test_phase1.py::test_insert_node_and_has_node -v
```

建议**按阶段顺序**运行测试，确保每个阶段全部通过后再进入下一阶段。Phase 1 的测试通过是后续阶段的前提。

---

## 6. 评分与提交

### 评分标准

| 项目    | 占比   | 说明 |
|---------|--------|------|
| Phase 1 | 约 30% | 内存图结构：插入、删除、查询 |
| Phase 2 | 约 30% | WAL + Snapshot 持久化与故障恢复 |
| Phase 3 | 约 20% | 图遍历算法（最短路径、全部依赖、按层依赖） |
| Phase 4 | 约 10% | Skill 集成（图查询封装为工具，LLM 完成依赖调查） |
| 报告    | 约 10% | 设计思路与反思，见下方报告要求 |

### 提交方式

- 运行 `./submit.sh <学号>` 生成 zip，提交代码与 `report.md`。

---

## 7. 报告要求

在项目根目录创建 `report.md`，使用 Markdown 撰写。

报告内容**只需要回答** `docs/report_questions.md` 中的所有问题即可（可按题号组织）。

建议：
- 回答尽量结合你在 `graph/graph_db.py`、`skills/graph_query_skills.py` 中的实现。
- 文字清晰、逻辑自洽即可，不要求固定模板。
