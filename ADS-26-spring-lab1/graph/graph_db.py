"""
图数据库核心：内存图 + WAL + Snapshot 持久化。

Phase 1: 内存图结构 —— insert_node, insert_edge, delete_node, delete_edge,
         get_neighbors, has_node, get_node_properties。
Phase 2: WAL (Write-Ahead Log) + Snapshot 持久化与故障恢复。
Phase 3: 图遍历与依赖查询 —— find_shortest_path, get_all_dependencies, get_dependencies_by_layer。

WAL 与 Snapshot 路径由本模块统一提供，不可修改：
- WAL 文件：{db_dir}/graph.log
- Snapshot 文件：{db_dir}/snapshot.dat
"""

import json
import os
from collections import deque
from typing import Dict, List, Optional, Tuple


# WAL 条数达到此阈值时触发自动 snapshot（具体行为见 _maybe_snapshot）
WAL_THRESHOLD = 1000

# 路径常量（测试与恢复逻辑依赖此路径，不可修改）
WAL_FILENAME = "graph.log"
SNAPSHOT_FILENAME = "snapshot.dat"


class GraphDB:
    """
    内存图数据库。内部数据结构：
    - _nodes: node_id -> 属性字典
    - _adj:   from_id -> { to_id -> weight }   出边邻接表
    - _radj:  to_id   -> { from_id, ... }       反向索引，用于 delete_node 高效删入边
    """

    def __init__(self, db_dir: str) -> None:
        """
        初始化图数据库目录与内存结构。

        Phase 1 阶段：内存结构已在下方声明，无需修改此函数，直接实现各内存操作即可。
        Phase 2 TODO：完成 recover() 后，取消最后一行注释以启用启动恢复。
        """
        self._db_dir = db_dir
        os.makedirs(db_dir, exist_ok=True)
        self._snapshot_path = os.path.join(db_dir, SNAPSHOT_FILENAME)
        self._wal_path = os.path.join(db_dir, WAL_FILENAME)

        # 内存结构（已提供，Phase 1 直接使用）
        self._nodes: Dict[str, dict] = {}
        self._adj: Dict[str, Dict[str, float]] = {}
        self._radj: Dict[str, set] = {}
        self._wal_count: int = 0  # WAL 条数计数，达到阈值时触发自动 snapshot

        self.recover()

    # ------------------------------------------------------------------
    # Phase 1: 内存图操作（接口与返回格式不得更改，测试依赖）
    # ------------------------------------------------------------------

    def insert_node(self, node_id: str, properties: Optional[dict] = None) -> None:
        """
        插入节点；可带属性 properties（字典）。若节点已存在，可合并/覆盖属性。O(1)。
        返回：无。
        """
        record = {
            "op": "insert_node",
            "node_id": node_id,
            "properties": properties,
        }
        self._write_wal(record)
        self._apply_insert_node(node_id, properties)
        self._maybe_snapshot()

    def insert_edge(self, from_id: str, to_id: str, weight: float = 1.0) -> None:
        """
        插入有向边 (from_id → to_id, weight)。若端点不存在则自动创建（无属性）。
        返回：无。
        """
        record = {
            "op": "insert_edge",
            "from_id": from_id,
            "to_id": to_id,
            "weight": float(weight),
        }
        self._write_wal(record)
        self._apply_insert_edge(from_id, to_id, weight)
        self._maybe_snapshot()

    def delete_node(self, node_id: str) -> None:
        """
        删除节点及其所有关联边（出边与入边）。若节点不存在则不操作。
        返回：无。

        提示：同时维护 _adj（出边表）与 _radj（入边反向索引），可将删除复杂度控制在
        O(出度 + 入度) 而非 O(V + E)——这正是图数据库维护反向索引的核心原因。
        """
        if node_id not in self._nodes:
            return
        record = {"op": "delete_node", "node_id": node_id}
        self._write_wal(record)
        self._apply_delete_node(node_id)
        self._maybe_snapshot()

    def delete_edge(self, from_id: str, to_id: str) -> None:
        """
        删除有向边 (from_id → to_id)。若边不存在则不操作。
        返回：无。
        """
        if to_id not in self._adj.get(from_id, {}):
            return
        record = {"op": "delete_edge", "from_id": from_id, "to_id": to_id}
        self._write_wal(record)
        self._apply_delete_edge(from_id, to_id)
        self._maybe_snapshot()

    def has_node(self, node_id: str) -> bool:
        """判断节点是否存在。返回：True / False。"""
        return node_id in self._nodes

    def get_node_properties(self, node_id: str) -> Optional[dict]:
        """获取节点属性字典；节点不存在时返回 None。"""
        if node_id not in self._nodes:
            return None
        return dict(self._nodes[node_id])

    def get_neighbors(self, node_id: str) -> List[Tuple[str, float]]:
        """
        查询节点的所有出边邻居及边权。
        返回：[(to_id, weight), ...]；节点不存在或无出边时返回 []。
        """
        if node_id not in self._nodes:
            return []
        return list(self._adj.get(node_id, {}).items())

    # ------------------------------------------------------------------
    # Phase 2: WAL + Snapshot 持久化（路径已固定，行为需与测试一致）
    # ------------------------------------------------------------------

    def _write_wal(self, record: dict) -> None:
        """
        将一条日志记录以 Append-Only 方式追加写入 WAL 文件，递增 _wal_count，
        写后调用 _maybe_snapshot() 检查是否需要触发自动快照。

        记录格式由你自行设计，只要 recover() 能正确解析即可。
        Phase 1 阶段此方法保持 pass（空操作）。
        Phase 2 TODO：在此实现完整 WAL 写入逻辑，并在各写操作中调用。
        """
        os.makedirs(self._db_dir, exist_ok=True)
        with open(self._wal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")
            f.flush()
        self._wal_count += 1

    def _maybe_snapshot(self) -> None:
        """若 _wal_count 达到 WAL_THRESHOLD，自动调用 save_snapshot()。"""
        if self._wal_count >= WAL_THRESHOLD:
            self.save_snapshot()

    def save_snapshot(self) -> None:
        """
        将当前内存图序列化写入 snapshot 文件（self._snapshot_path），
        成功后清空 WAL 文件并将 _wal_count 置 0。
        返回：无。
        """
        os.makedirs(self._db_dir, exist_ok=True)
        snapshot = {
            "nodes": self._nodes,
            "adj": self._adj,
        }
        tmp_path = self._snapshot_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self._snapshot_path)

        with open(self._wal_path, "w", encoding="utf-8") as f:
            f.truncate(0)
            f.flush()
            os.fsync(f.fileno())
        self._wal_count = 0

    def recover(self) -> None:
        """
        从磁盘恢复：先加载 snapshot（若存在），再逐行重放 WAL 文件中的记录。
        返回：无。
        """
        self.clear_memory()

        if self.snapshot_exists():
            with open(self._snapshot_path, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
            self._nodes = {
                str(node_id): dict(properties or {})
                for node_id, properties in snapshot.get("nodes", {}).items()
            }
            self._adj = {
                str(from_id): {
                    str(to_id): float(weight)
                    for to_id, weight in neighbors.items()
                }
                for from_id, neighbors in snapshot.get("adj", {}).items()
            }
            self._rebuild_reverse_index()

        if not os.path.isfile(self._wal_path):
            return

        replayed = 0
        with open(self._wal_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._apply_record(json.loads(line))
                replayed += 1
        self._wal_count = replayed

    def snapshot_exists(self) -> bool:
        """检查 snapshot 文件是否存在且非空。"""
        return os.path.isfile(self._snapshot_path) and os.path.getsize(self._snapshot_path) > 0

    def wal_exists(self) -> bool:
        """检查 WAL 文件是否存在且非空。"""
        return os.path.isfile(self._wal_path) and os.path.getsize(self._wal_path) > 0

    def clear_memory(self) -> None:
        """
        模拟断电：清空内存中的图与 WAL 计数，不删除磁盘上的 WAL/snapshot 文件。
        """
        self._nodes.clear()
        self._adj.clear()
        self._radj.clear()
        self._wal_count = 0

    def close(self) -> None:
        """
        关闭时保证数据落盘。
        Phase 2 TODO：将下方 pass 替换为 save_snapshot() 调用。
        """
        self.save_snapshot()

    # ------------------------------------------------------------------
    # Phase 3: 图查询算法（接口与返回格式不得更改，测试依赖）
    # ------------------------------------------------------------------

    def find_shortest_path(
        self, start_id: str, target_id: str
    ) -> Optional[List[Tuple[str, str, float]]]:
        """
        最短跳数路径算法。
        返回：[(from, to, weight), ...] 边列表；起终点相同返回 []；
              无路径或节点不存在返回 None。
        路径需连续：path[i][1] == path[i+1][0]，path[0][0] == start_id，path[-1][1] == target_id。
        """
        if start_id == target_id:
            return [] if start_id in self._nodes else None
        if start_id not in self._nodes or target_id not in self._nodes:
            return None

        queue = deque([start_id])
        visited = {start_id}
        previous: Dict[str, Tuple[str, float]] = {}

        while queue:
            current = queue.popleft()
            for neighbor, weight in self._adj.get(current, {}).items():
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                previous[neighbor] = (current, weight)
                if neighbor == target_id:
                    return self._build_path(start_id, target_id, previous)
                queue.append(neighbor)

        return None

    def get_all_dependencies(self, node_id: str) -> List[str]:
        """
        传递闭包：从 node_id 出发能到达的全部依赖节点（直接 + 间接）。
        返回：[dep_id, ...]；节点不存在时返回 []。
        """
        if node_id not in self._nodes:
            return []

        dependencies: List[str] = []
        visited = {node_id}
        queue = deque([node_id])

        while queue:
            current = queue.popleft()
            for neighbor in self._adj.get(current, {}):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                dependencies.append(neighbor)
                queue.append(neighbor)

        return dependencies

    def get_dependencies_by_layer(
        self, node_id: str, max_layers: int = 5
    ) -> Dict[int, List[str]]:
        """
        按层返回依赖：第 1 层为直接依赖，第 2 层为直接依赖的直接依赖，以此类推，
        最多返回 max_layers 层。
        返回：{1: [dep, ...], 2: [...], ...}；节点不存在时返回 {}。
        """
        if node_id not in self._nodes or max_layers <= 0:
            return {}

        layers: Dict[int, List[str]] = {}
        visited = {node_id}
        frontier = [node_id]

        for layer_num in range(1, max_layers + 1):
            next_frontier: List[str] = []
            for current in frontier:
                for neighbor in self._adj.get(current, {}):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
            if not next_frontier:
                break
            layers[layer_num] = next_frontier
            frontier = next_frontier

        return layers

    def _apply_insert_node(self, node_id: str, properties: Optional[dict] = None) -> None:
        self._nodes.setdefault(node_id, {})
        self._adj.setdefault(node_id, {})
        self._radj.setdefault(node_id, set())
        if properties:
            self._nodes[node_id].update(properties)

    def _apply_insert_edge(self, from_id: str, to_id: str, weight: float = 1.0) -> None:
        self._apply_insert_node(from_id)
        self._apply_insert_node(to_id)
        self._adj.setdefault(from_id, {})[to_id] = float(weight)
        self._radj.setdefault(to_id, set()).add(from_id)

    def _apply_delete_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            return

        for to_id in list(self._adj.get(node_id, {})):
            if to_id in self._radj:
                self._radj[to_id].discard(node_id)

        for from_id in list(self._radj.get(node_id, set())):
            if from_id in self._adj:
                self._adj[from_id].pop(node_id, None)

        self._nodes.pop(node_id, None)
        self._adj.pop(node_id, None)
        self._radj.pop(node_id, None)

    def _apply_delete_edge(self, from_id: str, to_id: str) -> None:
        if from_id in self._adj:
            self._adj[from_id].pop(to_id, None)
        if to_id in self._radj:
            self._radj[to_id].discard(from_id)

    def _apply_record(self, record: dict) -> None:
        op = record.get("op")
        if op == "insert_node":
            self._apply_insert_node(record["node_id"], record.get("properties"))
        elif op == "insert_edge":
            self._apply_insert_edge(record["from_id"], record["to_id"], record.get("weight", 1.0))
        elif op == "delete_node":
            self._apply_delete_node(record["node_id"])
        elif op == "delete_edge":
            self._apply_delete_edge(record["from_id"], record["to_id"])
        else:
            raise ValueError(f"Unknown WAL operation: {op!r}")

    def _rebuild_reverse_index(self) -> None:
        self._radj = {node_id: set() for node_id in self._nodes}
        for from_id, neighbors in list(self._adj.items()):
            self._nodes.setdefault(from_id, {})
            self._radj.setdefault(from_id, set())
            for to_id in neighbors:
                self._nodes.setdefault(to_id, {})
                self._radj.setdefault(to_id, set()).add(from_id)
                self._adj.setdefault(to_id, {})

    def _build_path(
        self,
        start_id: str,
        target_id: str,
        previous: Dict[str, Tuple[str, float]],
    ) -> List[Tuple[str, str, float]]:
        path: List[Tuple[str, str, float]] = []
        current = target_id
        while current != start_id:
            parent, weight = previous[current]
            path.append((parent, current, weight))
            current = parent
        path.reverse()
        return path
