# Lab0 Warmup：用 AI Skill 在数据结构上实现更高层接口

## 1. Lab 目标

本 Lab 目标是让你理解一个最小 Agent + Tool Calling 框架，并通过补全 Skill 提示词，让 LLM 能在多轮工具调用中完成任务。

你需要完成 3 个部分：

1. 补全 `agent/loader.py` 中 `load_all_skills()`；
2. 通过 `tests/test_add.py`；
3. 补全两个 skill 文件并通过剩余测试：
   - `skills/graph_skill/SKILL.md` -> `tests/test_graph.py`
   - `skills/range_query_skill/SKILL.md` -> `tests/test_kv.py`

## 2. 当前仓库结构

```text
agent/
  loader.py      # 加载 skills/*/SKILL.md 内容并拼接成 system prompt # TODO
  loop.py        # Agent loop：LLM 与工具多轮交互
  registry.py    # 工具注册表与 schema
  runtime.py     # 执行 tool_call

tools/
  add_tool.py    # add(a, b)
  graph_tool.py  # neighbors(node)
  kv_get_tool.py # kv_get(key)
  __init__.py    # 自动加载全部工具

skills/
  add_skill/SKILL.md
  graph_skill/SKILL.md          # TODO
  range_query_skill/SKILL.md    # TODO

examples/
  example_graph.py
  example_kv.py

tests/
  test_add.py
  test_graph.py
  test_kv.py
```

## 3. 环境准备

建议使用 `uv`：

```bash
uv sync
source .venv/bin/activate
```

API 配置：

1. 申请 API key [https://form.sjtu.edu.cn/infoplus/form/net_ai_api_apply/start?locale=zh](https://form.sjtu.edu.cn/infoplus/form/net_ai_api_apply/start?locale=zh)
2. 在 `agent/loop.py` 中设置 `API_KEY`；

## 4. 任务细节

### 任务 A：补全 `load_all_skills()`

文件：`agent/loader.py`

函数应实现：

- 扫描 `skills/*/SKILL.md`（兼容 `skill.md`）；
- 读取文本内容；
- 使用 `\n\n---\n\n` 拼接所有 skill 文本并返回。

该步骤完成后，`add_skill` 才会进入 system prompt，被 Agent 正确利用。

### 任务 B：通过 `test_add.py`

测试逻辑：向 Agent 发送 `"Add 1 and 2"`，期望输出包含 `"3"`。

```bash
pytest tests/test_add.py
```

### 任务 C：补全 graph skill 并通过 `test_graph.py`

文件：`skills/graph_skill/SKILL.md`

你需要写清楚：

- 何时使用 `neighbors`；
- 如何从 `A` 出发逐层探索；
- 如何避免重复访问；
- 如何输出最终答案（应包含最远节点 `G`）。

运行测试：

```bash
pytest tests/test_graph.py
```

### 任务 D：补全 range query skill 并通过 `test_kv.py`

文件：`skills/range_query_skill/SKILL.md`

你需要写清楚：

- 何时使用 `kv_get`；
- 对区间 `[1, 10]` 的每个整数 key 逐个调用；
- 保留返回非空值的 key；
- 最终按升序输出逗号分隔字符串：`1,3,5,7,10`。

运行测试：

```bash
pytest tests/test_kv.py
```

## 5. 提交要求

- 代码：确保上述 3 个测试可通过（允许模型随机性，建议保留测试截图/日志）；
- 报告：完成 `report.md`，回答 `docs/report_questions.md` 中问题；
- 打包：运行 `./submit.sh <学号>`。

## 6. QA

[LAB0 QA](https://my.feishu.cn/docx/Fx1odSUWDooZoYxL3NwcJFtJn5b?from=from_copylink)
