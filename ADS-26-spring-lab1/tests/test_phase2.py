"""
Phase 2 测试：WAL + Snapshot 持久化与故障恢复。

测试流程：
1. 插入数据 → 检查 snapshot/WAL 文件是否生成。
2. 模拟断电（clear_memory）→ 从 snapshot + WAL 恢复 → 验证查询正确。
3. 触发 snapshot 阈值 → 验证 WAL 被清空。
"""

import json
import os
import tempfile
import pytest

from graph.graph_db import GraphDB, WAL_THRESHOLD
from data.load_graph_json import load_graph_json_into_db


@pytest.fixture
def db_dir():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "graph_db")


def test_wal_written_on_mutation(db_dir):
    """写操作后 WAL 文件应存在且非空（格式不限，能被 recover() 正确解析即可）。"""
    db = GraphDB(db_dir)
    db.insert_node("a", {"version": "1.0"})
    db.insert_edge("a", "b", 1.0)
    assert db.wal_exists()
    db.close()


def test_snapshot_on_close(db_dir):
    """close() 应生成 snapshot 文件。"""
    db = GraphDB(db_dir)
    db.insert_node("a")
    db.insert_edge("a", "b", 1.0)
    db.close()
    assert os.path.isfile(os.path.join(db_dir, "snapshot.dat"))


def test_recovery_from_snapshot(db_dir):
    """关闭后重新打开（从 snapshot 恢复），数据应完整。"""
    db = GraphDB(db_dir)
    db.insert_node("adduser", {"desc": "add users"})
    db.insert_edge("adduser", "passwd", 1.0)
    db.insert_edge("passwd", "libc6", 1.0)
    db.close()

    db2 = GraphDB(db_dir)
    assert db2.has_node("adduser") is True
    assert db2.has_node("passwd") is True
    assert db2.has_node("libc6") is True
    assert "passwd" in dict(db2.get_neighbors("adduser"))
    props = db2.get_node_properties("adduser")
    assert props["desc"] == "add users"
    db2.close()


def test_crash_recovery_snapshot_plus_wal(db_dir):
    """
    模拟断电恢复：
    1. 插入数据并 save_snapshot。
    2. 再追加一些操作（只写 WAL，不 snapshot）。
    3. clear_memory 模拟断电。
    4. recover 从 snapshot + WAL 恢复。
    5. 验证所有数据完整。
    """
    db = GraphDB(db_dir)
    db.insert_node("a")
    db.insert_edge("a", "b", 1.0)
    db.save_snapshot()

    db.insert_edge("b", "c", 2.0)
    db.insert_node("d", {"role": "leaf"})
    assert db.wal_exists()

    db.clear_memory()
    assert db.has_node("a") is False

    db.recover()
    assert db.has_node("a") is True
    assert db.has_node("b") is True
    assert db.has_node("c") is True
    assert db.has_node("d") is True
    assert "b" in dict(db.get_neighbors("a"))
    assert "c" in dict(db.get_neighbors("b"))
    assert db.get_node_properties("d") == {"role": "leaf"}
    db.close()


def test_snapshot_clears_wal(db_dir):
    """save_snapshot 后 WAL 应被清空。"""
    db = GraphDB(db_dir)
    db.insert_node("a")
    db.insert_edge("a", "b", 1.0)
    assert db.wal_exists()
    db.save_snapshot()
    wal_path = os.path.join(db_dir, "graph.log")
    with open(wal_path, "r") as f:
        content = f.read()
    assert content == ""
    db.close()


def test_auto_snapshot_on_threshold(db_dir):
    """WAL 达到阈值时应自动触发 snapshot。"""
    db = GraphDB(db_dir)
    for i in range(WAL_THRESHOLD):
        db.insert_node(f"node_{i}")
    assert db.snapshot_exists()
    wal_path = os.path.join(db_dir, "graph.log")
    with open(wal_path, "r") as f:
        lines = f.readlines()
    assert len(lines) < WAL_THRESHOLD
    db.close()


def test_delete_then_recover(db_dir):
    """删除操作也应通过 WAL 持久化，恢复后删除效果仍在。"""
    db = GraphDB(db_dir)
    db.insert_edge("a", "b", 1.0)
    db.insert_edge("b", "c", 2.0)
    db.save_snapshot()

    db.delete_edge("a", "b")
    db.delete_node("c")

    db.clear_memory()
    db.recover()

    assert db.has_node("a") is True
    assert db.has_node("b") is True
    assert db.has_node("c") is False
    assert dict(db.get_neighbors("a")) == {}
    db.close()


# ---------------------------------------------------------------------------
# 全局插入 + 持久化综合检查：graph.json 整图写入后 WAL/snapshot/恢复 是否都正确
# ---------------------------------------------------------------------------

def _data_dir():
    return os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture
def graph_json_path():
    return os.path.join(_data_dir(), "graph.json")


@pytest.fixture
def deps_answer_path():
    return os.path.join(_data_dir(), "deps_answer.json")


def test_full_graph_persistence(db_dir, graph_json_path, deps_answer_path):
    """
    插入整个 data/graph.json 后综合检查持久化逻辑：
    1) 插入过程中 WAL 写入，达到阈值后自动 snapshot，close() 时再次 snapshot 并清空 WAL；
    2) 关闭后 snapshot 文件存在且非空，WAL 已被清空；
    3) 重新打开从 snapshot 恢复，数据完整（10 个包 layer1 与 deps_answer 一致）；
    4) 模拟断电：clear_memory 后 recover()，再次校验数据一致。
    """
    # 1) 整图插入（会写 WAL，超过 WAL_THRESHOLD 会触发 _maybe_snapshot）
    db = load_graph_json_into_db(db_dir, graph_json_path)
    assert db.snapshot_exists(), "大规模插入后应已触发 snapshot（WAL 达阈值）"
    db.close()

    # 2) 关闭后：snapshot 存在且有效，WAL 被 close() 里的 save_snapshot 清空
    snapshot_path = os.path.join(db_dir, "snapshot.dat")
    wal_path = os.path.join(db_dir, "graph.log")
    assert os.path.isfile(snapshot_path) and os.path.getsize(snapshot_path) > 0, "close 后应有有效 snapshot"
    with open(wal_path, "r", encoding="utf-8") as f:
        wal_after_close = f.read()
    assert wal_after_close.strip() == "", "close() 时 save_snapshot 应清空 WAL"

    # 3) 重新打开，从 snapshot 恢复，校验 10 个包的 layer1（get_neighbors）
    db2 = GraphDB(db_dir)
    with open(deps_answer_path, "r", encoding="utf-8") as f:
        answer = json.load(f)
    for pkg in answer["packages"]:
        want = set(answer["dependencies"].get(pkg, {}).get("layer1", []))
        got = {name for name, _ in db2.get_neighbors(pkg)}
        assert got == want, (
            f"恢复后包 {pkg} layer1 不一致. 多余: {got - want!r}, 缺失: {want - got!r}"
        )
    db2.close()

    # 4) 模拟断电：同一目录再次打开后 clear_memory + recover，再校验一次
    db3 = GraphDB(db_dir)
    db3.clear_memory()
    assert not db3.has_node(answer["packages"][0]), "clear_memory 后内存应为空"
    db3.recover()
    for pkg in answer["packages"]:
        want = set(answer["dependencies"].get(pkg, {}).get("layer1", []))
        got = {name for name, _ in db3.get_neighbors(pkg)}
        assert got == want, (
            f"recover() 后包 {pkg} layer1 不一致. 多余: {got - want!r}, 缺失: {want - got!r}"
        )
    db3.close()
