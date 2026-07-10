import tools  # noqa: F401 — loads all tools into registry

from agent.registry import registry
from agent.runtime import SkillRuntime
from agent.loop import agent_loop


def test_agent_performs_kv_range_query() -> None:
    runtime = SkillRuntime(registry)

    messages = [{"role": "user", "content": "range query the kv store from 1 to 10"}]

    result = agent_loop(registry, runtime, messages)

    assert isinstance(result, str)
    assert "1,3,5,7,10" in result
