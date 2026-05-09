"""
Part 4：图查询 Skill —— 将图库接口封装为 LLM 可调用的工具。

本文件当前只提供 get_neighbors 的完整注册示例。
其余三个工具（find_shortest_path / get_all_dependencies / get_dependencies_by_layer）
由学生仿照示例补全注册与实现。

提示：工具实现可直接调用 Phase 3 中实现的 backend 的对应方法，不在本模块内实现算法。
"""

from typing import Any, Dict, List, Optional, Tuple

from skills.registry import registry

_graph_backend: Optional[Any] = None


def set_graph_backend(backend: Any) -> None:
    """注入图后端，供后续工具调用。"""
    global _graph_backend
    _graph_backend = backend


def get_graph_backend() -> Optional[Any]:
    """获取当前图后端；未注入时为 None。"""
    return _graph_backend


# ---------------------------------------------------------------------------
# 注册示例：get_neighbors（其余 3 个工具请按同样模式补全）
# ---------------------------------------------------------------------------

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
    """功能：查询 node 的直接依赖。返回：[(to_id, weight), ...]，无后端时返回 []。"""
    backend = get_graph_backend()
    if backend is None:
        return []
    return backend.get_neighbors(node)


# ---------------------------------------------------------------------------
# TODO(学生补全)：仿照 get_neighbors，为以下三个函数补全 @registry.register(...) 与实现
# ---------------------------------------------------------------------------

@registry.register(
    name="find_shortest_path",
    description="查找从源软件包到目标软件包的最短跳数路径（BFS）。返回 [(源包, 目标包, 权重), ...] 的边列表；若无路径则返回 None。",
    parameters={
        "type": "object",
        "properties": {
            "start": {"type": "string", "description": "起始软件包名称"},
            "target": {"type": "string", "description": "目标软件包名称"},
        },
        "required": ["start", "target"],
    },
)
def find_shortest_path(
    start: str, target: str
) -> Optional[List[Tuple[str, str, float]]]:
    """功能：查找 start -> target 最短路径。返回：边列表或 None，无后端时返回 None。"""
    backend = get_graph_backend()
    if backend is None:
        return None
    return backend.find_shortest_path(start, target)


@registry.register(
    name="get_all_dependencies",
    description="查询某个软件包的全部依赖（传递闭包），即直接与间接依赖的并集。返回 [依赖包名, ...] 列表。",
    parameters={
        "type": "object",
        "properties": {
            "node": {"type": "string", "description": "软件包名称"},
        },
        "required": ["node"],
    },
)
def get_all_dependencies(node: str) -> List[str]:
    """功能：查询 node 的全部依赖。返回：[dep_id, ...]，无后端时返回 []。"""
    backend = get_graph_backend()
    if backend is None:
        return []
    return backend.get_all_dependencies(node)


@registry.register(
    name="get_dependencies_by_layer",
    description="查询某个软件包按层划分的依赖：第1层为直接依赖，第2层为直接依赖的直接依赖，以此类推。返回 {1: [包名, ...], 2: [...], ...}，最多到 max_layers 层。",
    parameters={
        "type": "object",
        "properties": {
            "node": {"type": "string", "description": "软件包名称"},
            "max_layers": {
                "type": "integer",
                "description": "最多查询的依赖层数，必须为正整数",
                "minimum": 1,
            },
        },
        "required": ["node", "max_layers"],
    },
)
def get_dependencies_by_layer(
    node: str, max_layers: int = 2
) -> Dict[int, List[str]]:
    """功能：按层查询 node 的依赖。返回：{1: [dep, ...], 2: [...]}，无后端时返回 {}。"""
    backend = get_graph_backend()
    if backend is None:
        return {}
    return backend.get_dependencies_by_layer(node, int(max_layers))


