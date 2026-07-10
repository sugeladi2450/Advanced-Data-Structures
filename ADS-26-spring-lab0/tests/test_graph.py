import tools  # noqa: F401 — loads all tools into registry

from agent.registry import registry
from agent.runtime import SkillRuntime
from agent.loop import agent_loop


def test_agent_finds_farthest_node_from_a() -> None:
    runtime = SkillRuntime(registry)

    messages = [{"role": "user", "content": "Find the farthest node from A"}]

    result = agent_loop(registry, runtime, messages)

    assert isinstance(result, str)
    assert "G" in result
