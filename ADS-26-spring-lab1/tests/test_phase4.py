"""
Part 3.2 测试：Skill 集成。

- test_agent_uses_skills：漏洞依赖调查（需 data/graph.json，调用 LLM）；技能已注册，backend 通过 set_graph_backend 注入。
"""

import os
import tempfile
from pathlib import Path

from skills.registry import registry
import skills.graph_query_skills  # noqa: F401 — 触发 @registry.register
from skills.graph_query_skills import set_graph_backend
from data.load_graph_json import load_graph_json_into_db

# 漏洞调查任务 prompt：在测试中写死，不放在 skills/（skills 由实验者自己实现，只提供工具）
PROMPT_VULNERABILITY_INVESTIGATION = """
你是一个高级依赖分析 Agent。
已知底层库 X 存在高危漏洞。现在有五个核心业务包：[A, B, C, D, E]。
请你调查这五个包是否（直接或间接）依赖了 X。

你必须对五个核心包逐一使用工具进行调查，不得遗漏任一包（每个包至少调用一次工具确认是否依赖 X）。
全部五个包都调查完成后，再输出最终答案。
重要：最终回复必须只包含一行——依赖于漏洞库 X 的核心包包名，包名之间用英文逗号隔开。
若没有任何核心包依赖 X，则只输出「无」。不要输出其他解释、换行或多余文字。
"""

VULN_X = "389-ds-base-libs"
CORE_FIVE = ["389-ds", "389-ds-base", "0ad", "0ad-data","389-ds-base-dev"]
ANSWERS = ["389-ds", "389-ds-base", "389-ds-base-dev"]

def _parse_agent_dependent_list(result: str, core_five: list) -> set:
    """从 Agent 输出中解析「依赖 X 的包」：按英文逗号分割，只保留在 core_five 中的包名，严格匹配。"""
    if not result or not result.strip():
        return set()
    # 按逗号、换行分割，strip，只保留与 core_five 完全一致的 token
    core_set = set(core_five)
    tokens = []
    for part in result.replace("\n", ",").split(","):
        t = part.strip()
        if t in core_set:
            tokens.append(t)
    return set(tokens)


def _load_skill_markdown() -> str:
    """Load graph-query skill from skills/ (read markdown first, then call functions)."""
    repo_root = Path(__file__).resolve().parent.parent
    skill_path = repo_root / "skills" / "graph-query-skills" / "SKILL.md"
    return skill_path.read_text(encoding="utf-8")


def test_agent_uses_skills():
    """
    真机测试：高级依赖分析 Agent。
    先加载 skills/graph-query-skills/SKILL.md 注入为 system，模型先读 skill 再根据任务调用对应工具（与 Cursor/Codex 一致）。
    用图先算 ground truth（哪些核心包依赖 X），再让 LLM 调查；要求 LLM 只输出依赖 X 的包名（英文逗号分隔）。
    验证时严格字符串匹配：解析出的包名集合必须与 ground truth 一致。
    """
    from skills.runtime import SkillRuntime
    from agent.loop import agent_loop

    data_dir = Path(__file__).resolve().parent.parent / "data"
    graph_json_path = data_dir / "graph.json"
    skill_markdown = _load_skill_markdown()

    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "graph_db")
        db = load_graph_json_into_db(db_path, graph_json_path)
        set_graph_backend(db)
        runtime = SkillRuntime(registry)

        X = VULN_X
        core_five = CORE_FIVE

        # expected_dependents = _core_packages_that_depend_on(db, core_five, X)
        expected_dependents = ANSWERS
        print("\n========== 漏洞依赖调查 ==========")
        print(f"X（漏洞库）: {X}")
        print(f"核心包 [A,B,C,D,E]: {core_five}")
        print(f"ground truth（依赖 X 的核心包）: {expected_dependents}")
        print("====================================\n")

        user_content = (
            PROMPT_VULNERABILITY_INVESTIGATION
            + f"\n\n请完成调查：\n"
            f"- 底层库 X = {X}（存在高危漏洞）\n"
            f"- 五个核心业务包：[{', '.join(core_five)}]\n"
            f"请调查后，最终只输出依赖于 {X} 的核心包包名，包名之间用英文逗号隔开；若没有则只输出「无」。"
        )
        messages = [{"role": "user", "content": user_content}]
        result = agent_loop(registry, runtime, messages, skill_markdown=skill_markdown)

        assert result is not None, "Agent 未返回结果"
        parsed = _parse_agent_dependent_list(result, core_five)
        # 若 LLM 输出「无」且未输出任何核心包名，视为空集合
        if "无" in result and not any(p in result for p in core_five):
            parsed = set()
        
        assert parsed == set(expected_dependents), (
            f"依赖 X 的包名集合与 ground truth 不一致。\n"
            f"  ground truth: {expected_dependents}\n"
            f"  Agent 解析结果: {parsed}\n"
            f"  Agent 原始输出: {result!r}"
        )
        print(f"验证通过。依赖 {X} 的包: {parsed}")
        db.close()
