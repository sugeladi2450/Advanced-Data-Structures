---
name: graph-query-skills
description: Implements and registers graph query skills (get_neighbors, find_shortest_path, get_all_dependencies, get_dependencies_by_layer) under the project skills/ folder. Use when adding or changing LLM-callable graph tools, skill descriptions, or prompts for dependency or vulnerability analysis.
---

# Graph Query Skills (skills/)

## 使用方式（Agent / 模型）

请先阅读本 Skill 说明，再根据用户任务调用下方列出的对应工具完成依赖调查。可用工具：`get_neighbors`、`find_shortest_path`、`get_all_dependencies`、`get_dependencies_by_layer`。对漏洞调查类任务，需对每个核心包至少调用一次工具确认是否依赖漏洞库，全部调查完成后再输出最终答案。

---

## Where Things Live

- **Registration, implementation, and description**: All in `skills/`, not in tests or agent.
- **Registry**: `skills/registry.py` — `SkillRegistry`, single `registry` instance.
- **Runtime**: `skills/runtime.py` — `SkillRuntime(registry)` runs a tool call by name + JSON arguments.
- **Graph tools and prompts**: `skills/graph_query_skills.py` — backend injection, four registered tools, and prompt strings.

The graph backend is **injected** via `set_graph_backend(backend)`. Tests or the agent set the backend (e.g. GraphDB) before calling the LLM.

## Backend Contract

The graph backend used by skills must expose:

- `get_neighbors(node: str) -> List[Tuple[str, float]]`
- `find_shortest_path(start: str, target: str) -> ...`
- `get_all_dependencies(node: str) -> ...`
- `get_dependencies_by_layer(node: str, max_layers: int) -> ...`

Skills in `graph_query_skills.py` **delegate** to these backend methods (no BFS/algorithm implementation in the skills layer).

## Registry Pattern

Use the decorator to register name, description, and parameters (OpenAI-style schema) in one place:

```python
from skills.registry import registry

@registry.register(
    name="get_neighbors",
    description="查询某个软件包的直接依赖（出边邻居）。返回 [(邻居包名, 边权重), ...] 列表。",
    parameters={
        "type": "object",
        "properties": {
            "node": {"type": "string", "description": "软件包名称（如 adduser、libc6）"},
        },
        "required": ["node"],
    },
)
def get_neighbors(node: str) -> List[Tuple[str, float]]:
    return _neighbors(node)
```

Description and parameters are the LLM-facing tool metadata; keep them in the same file as the implementation.

## The Four Tools (and Descriptions)

| Tool | Description (for LLM) | Parameters |
|------|------------------------|------------|
| **get_neighbors** | 查询某个软件包的直接依赖（出边邻居）。返回 [(邻居包名, 边权重), ...] 列表。 | `node` (string) |
| **find_shortest_path** | 查找从源软件包到目标软件包的最短跳数路径（BFS）。返回 [(源包, 目标包, 权重), ...] 的边列表。若无路径则返回 None。 | `start`, `target` (string) |
| **get_all_dependencies** | 查询某个软件包的全部依赖（传递闭包），即直接与间接依赖的并集。返回 [依赖包名, ...] 列表。 | `node` (string) |
| **get_dependencies_by_layer** | 查询某个软件包按层划分的依赖：第1层为直接依赖，第2层为直接依赖的直接依赖，以此类推。返回 {1: [包名, ...], 2: [...], ...}，最多到 max_layers 层。 | `node` (string), `max_layers` (integer) |

Implementations in `graph_query_skills.py` delegate to the backend:

- `get_neighbors(node)` → `backend.get_neighbors(node)`.
- `find_shortest_path(start, target)` → `backend.find_shortest_path(start, target)`.
- `get_all_dependencies(node)` → `backend.get_all_dependencies(node)`.
- `get_dependencies_by_layer(node, max_layers)` → `backend.get_dependencies_by_layer(node, max_layers)`.

## Runtime

```python
from skills.runtime import SkillRuntime
from skills.registry import registry

runtime = SkillRuntime(registry)
result = runtime.run({"name": "find_shortest_path", "arguments": '{"start": "adduser", "target": "libc6"}'})
```

## Prompts

Task prompts (e.g. vulnerability investigation) are defined by the **caller** (e.g. in tests), not in `skills/graph_query_skills.py`. The skills module only provides tool registration and implementation; tests write the prompt text.

## Checklist for Changes

- Add or change tools only in `skills/graph_query_skills.py`.
- Keep each tool's `description` and `parameters` in the same `@registry.register` call.
- Implement graph algorithms in the **graph backend** (e.g. `graph_db.py`); skills only call backend methods.
- Define task prompts in the caller (e.g. tests), not in `graph_query_skills.py`.
