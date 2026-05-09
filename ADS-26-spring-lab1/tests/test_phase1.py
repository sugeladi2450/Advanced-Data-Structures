"""
Part 1 测试：内存图结构 —— 插入、删除、查询、属性。

验证 GraphDB 的嵌套哈希表实现是否正确。
"""

import json
import os
import tempfile
import pytest

from graph.graph_db import GraphDB
from data.load_graph_json import load_graph_json_into_db


@pytest.fixture
def db_dir():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "graph_db")


def test_insert_node_and_has_node(db_dir):
    """insert_node 后 has_node 应返回 True。"""
    db = GraphDB(db_dir)
    db.insert_node("adduser", {"desc": "add users"})
    db.insert_node("passwd")
    assert db.has_node("adduser") is True
    assert db.has_node("passwd") is True
    assert db.has_node("nonexistent") is False
    db.close()


def test_node_properties(db_dir):
    """节点可附带属性，且可查询。"""
    db = GraphDB(db_dir)
    db.insert_node("libc6", {"version": "2.36", "priority": "required"})
    props = db.get_node_properties("libc6")
    assert props is not None
    assert props["version"] == "2.36"
    assert props["priority"] == "required"
    assert db.get_node_properties("nonexistent") is None
    db.close()


def test_insert_edge_and_get_neighbors(db_dir):
    """insert_edge 后 get_neighbors 应返回正确的邻居与权重。"""
    db = GraphDB(db_dir)
    db.insert_node("a")
    db.insert_edge("a", "b", 1.0)
    db.insert_edge("a", "c", 2.5)
    neighbors = db.get_neighbors("a")
    neighbor_dict = dict(neighbors)
    assert "b" in neighbor_dict
    assert neighbor_dict["b"] == 1.0
    assert "c" in neighbor_dict
    assert neighbor_dict["c"] == 2.5
    assert db.get_neighbors("b") == []
    db.close()


def test_insert_edge_auto_creates_nodes(db_dir):
    """insert_edge 时若节点不存在，应自动创建。"""
    db = GraphDB(db_dir)
    db.insert_edge("x", "y", 3.0)
    assert db.has_node("x") is True
    assert db.has_node("y") is True
    db.close()


def test_delete_edge(db_dir):
    """delete_edge 后该边应不存在。"""
    db = GraphDB(db_dir)
    db.insert_edge("a", "b", 1.0)
    db.insert_edge("a", "c", 2.0)
    db.delete_edge("a", "b")
    neighbors = db.get_neighbors("a")
    neighbor_dict = dict(neighbors)
    assert "b" not in neighbor_dict
    assert "c" in neighbor_dict
    db.close()


def test_delete_node(db_dir):
    """delete_node 应删除节点及其所有关联边（出边和入边）。"""
    db = GraphDB(db_dir)
    db.insert_edge("a", "b", 1.0)
    db.insert_edge("b", "c", 2.0)
    db.insert_edge("c", "a", 3.0)
    db.delete_node("b")
    assert db.has_node("b") is False
    assert db.has_node("a") is True
    assert db.has_node("c") is True
    assert dict(db.get_neighbors("a")) == {}
    assert "b" not in dict(db.get_neighbors("c"))
    db.close()


def test_delete_nonexistent(db_dir):
    """删除不存在的节点或边不应报错。"""
    db = GraphDB(db_dir)
    db.insert_node("a")
    db.delete_node("nonexistent")
    db.delete_edge("a", "nonexistent")
    assert db.has_node("a") is True
    db.close()


def _data_dir():
    return os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture
def graph_json_path():
    return os.path.join(_data_dir(), "graph.json")


@pytest.fixture
def deps_answer_path():
    return os.path.join(_data_dir(), "deps_answer.json")


def test_full_graph_insert_and_layer1_neighbors(db_dir, graph_json_path, deps_answer_path):
    """
    插入整个 data/graph.json 到图库，对 10 个包用 get_neighbors 查直接依赖（layer1），
    与 data/deps_answer.json 的 layer1 比对，验证大规模插入与邻边查询是否正确。
    """
    db = load_graph_json_into_db(db_dir, graph_json_path)
    try:
        with open(deps_answer_path, "r", encoding="utf-8") as f:
            answer = json.load(f)
        packages = answer["packages"]
        expected = answer["dependencies"]

        for pkg in packages:
            want_set = set(expected.get(pkg, {}).get("layer1", []))
            got_set = {name for name, _ in db.get_neighbors(pkg)}
            assert got_set == want_set, (
                f"包 {pkg} layer1（直接依赖）与 get_neighbors 不一致. "
                f"多余: {got_set - want_set!r}, 缺失: {want_set - got_set!r}"
            )
    finally:
        db.close()
