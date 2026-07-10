import tools  # noqa: F401 — loads all tools into registry

from agent.registry import registry
from agent.runtime import SkillRuntime
from agent.loop import agent_loop


def test_agent_performs_add() -> None:
    runtime = SkillRuntime(registry)

    messages = [{"role": "user", "content": "Add 1 and 2"}]

    result = agent_loop(registry, runtime, messages)

    assert isinstance(result, str)
    assert "3" in result
