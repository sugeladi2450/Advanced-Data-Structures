"""
Part 3 测试：BFS 最短路径与依赖查询。

全部使用 data/graph.json：插入图库后验证 find_shortest_path、get_all_dependencies、
以及按层依赖与 data/deps_answer.json 比对（至少 layer1/2/3）。
"""

import json
import os
import tempfile
import pytest

from graph.graph_db import GraphDB
from data.load_graph_json import load_graph_json_into_db


def _data_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture
def db_dir():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "graph_db")


@pytest.fixture
def graph_json_path():
    return os.path.join(_data_dir(), "graph.json")


@pytest.fixture
def deps_answer_path():
    return os.path.join(_data_dir(), "deps_answer.json")


def test_find_shortest_path(db_dir, graph_json_path):
    """从 graph.json 加载图，验证 find_shortest_path 返回格式、路径连续性及多跳路径。"""
    db = load_graph_json_into_db(db_dir, graph_json_path)
    try:
        # 单跳：0ad -> libc6（0ad 直接依赖 libc6）
        path = db.find_shortest_path("0ad", "libc6")
        assert path is not None, "0ad -> libc6 应存在路径"
        assert len(path) >= 1, "路径至少一条边"
        assert path[0][0] == "0ad" and path[-1][1] == "libc6", "路径起点、终点应对"
        for edge in path:
            assert len(edge) == 3, f"每条边应为 (from, to, weight)，得到 {edge}"
            assert edge[2] >= 0, f"边权重应非负，得到 {edge[2]}"
        for i in range(len(path) - 1):
            assert path[i][1] == path[i + 1][0], f"路径应连续：边{i}终点与边{i+1}起点一致"

        # 多跳：389-ds -> libc6（经 layer2 才到 libc6）
        path2 = db.find_shortest_path("389-ds", "libc6")
        assert path2 is not None, "389-ds -> libc6 应存在路径"
        assert len(path2) >= 2, "389-ds 到 libc6 至少 2 跳"
        assert path2[0][0] == "389-ds" and path2[-1][1] == "libc6"
        for edge in path2:
            assert len(edge) == 3 and edge[2] >= 0
        for i in range(len(path2) - 1):
            assert path2[i][1] == path2[i + 1][0]
    finally:
        db.close()


def test_get_all_dependencies(db_dir, graph_json_path, deps_answer_path):
    """从 graph.json 加载图，10 个包的全部依赖应包含 deps_answer 中各层出现的依赖。"""
    db = load_graph_json_into_db(db_dir, graph_json_path)
    try:
        with open(deps_answer_path, "r", encoding="utf-8") as f:
            answer = json.load(f)
        for pkg in answer["packages"]:
            got_names = set(db.get_all_dependencies(pkg))
            want_by_layer = answer["dependencies"].get(pkg, {})
            for key in ("layer1", "layer2", "layer3", "layer4", "layer5"):
                for dep in want_by_layer.get(key, []):
                    assert dep in got_names, f"包 {pkg} 的 get_all_dependencies 应包含 {dep}（来自 {key}）"
    finally:
        db.close()


def test_no_path(db_dir):
    """无路径时 find_shortest_path 返回 None。"""
    db = GraphDB(db_dir)
    db.insert_node("a")
    db.insert_node("b")
    assert db.find_shortest_path("a", "b") is None
    db.close()


def test_self_path(db_dir):
    """起点等于终点时返回空路径。"""
    db = GraphDB(db_dir)
    db.insert_node("a")
    assert db.find_shortest_path("a", "a") == []
    db.close()


# ---------------------------------------------------------------------------
# 按层依赖：与 data/deps_answer.json 比对
# ---------------------------------------------------------------------------

MIN_LAYERS_TO_CHECK = 3
MAX_LAYERS_IN_ANSWER = 5


def test_deps_vs_answer_file(db_dir, graph_json_path, deps_answer_path):
    """
    从 graph.json 插入图库，对 10 个包查询按层依赖（至少三层分开记录），
    结果需与 deps_answer.json 一致。
    """
    db = load_graph_json_into_db(db_dir, graph_json_path)
    try:
        with open(deps_answer_path, "r", encoding="utf-8") as f:
            answer = json.load(f)
        packages = answer["packages"]
        expected = answer["dependencies"]

        for pkg in packages:
            got_layers = db.get_dependencies_by_layer(pkg, max_layers=MAX_LAYERS_IN_ANSWER)
            want_pkg = expected.get(pkg, {})

            for layer_num in range(1, MIN_LAYERS_TO_CHECK + 1):
                key = f"layer{layer_num}"
                want_set = set(want_pkg.get(key, []))
                got_set = set(got_layers.get(layer_num, []))
                assert got_set == want_set, (
                    f"包 {pkg} {key}: 依赖集合不一致. "
                    f"多余: {got_set - want_set!r}, 缺失: {want_set - got_set!r}"
                )

            for layer_num in range(MIN_LAYERS_TO_CHECK + 1, MAX_LAYERS_IN_ANSWER + 1):
                key = f"layer{layer_num}"
                if key not in want_pkg:
                    continue
                want_set = set(want_pkg[key])
                got_set = set(got_layers.get(layer_num, []))
                assert got_set == want_set, (
                    f"包 {pkg} {key}: 依赖集合不一致. "
                    f"多余: {got_set - want_set!r}, 缺失: {want_set - got_set!r}"
                )
    finally:
        db.close()
