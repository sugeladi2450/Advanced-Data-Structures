"""
从 graph.json 读取图数据并插入 GraphDB。

graph.json 格式：nodes（包名列表）、edges（[{"from":"pkg1","to":"pkg2","weight":1.0}, ...]）。
为简化图库实现，边仅保留 weight，无 type 属性。本模块提供 read_graph_json；load_graph_json_into_db 由学生实现。
"""

import json
from pathlib import Path
from typing import List, Tuple

from graph.graph_db import GraphDB


def read_graph_json(path: str | Path) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """
    读取 graph.json，返回 (nodes, edges)。
    nodes: 包名列表。edges: (from, to, weight) 元组列表，weight 为 float。
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = list(data.get("nodes", []))
    edges = []
    for e in data.get("edges", []):
        edges.append((e["from"], e["to"], float(e.get("weight", 1.0))))
    return (nodes, edges)


def load_graph_json_into_db(db_dir: str, graph_json_path: str | Path) -> GraphDB:
    """
    将 graph.json 中的节点与边写入 GraphDB，并返回该 GraphDB 实例。

    功能：
    - 使用 read_graph_json(graph_json_path) 得到 nodes 与 edges（每条边为 (from, to, weight)）。
    - 创建 GraphDB(db_dir)，依次 insert_node 每个节点（无属性或空属性），
      再依次 insert_edge(from, to, weight) 每条边（使用 graph.json 中的 weight，通常为 1.0）。
    返回：创建并填充后的 GraphDB 实例。
    """
    nodes, edges = read_graph_json(graph_json_path)
    db = GraphDB(db_dir)
    for node in nodes:
        db.insert_node(node)
    for from_id, to_id, weight in edges:
        db.insert_edge(from_id, to_id, weight)
    return db
