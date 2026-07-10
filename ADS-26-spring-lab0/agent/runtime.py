import json
from typing import Any
from agent.registry import SkillRegistry


class SkillRuntime:
    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def run(self, tool_call: dict) -> Any:
        name = tool_call["name"]
        args = json.loads(tool_call["arguments"])

        func = self.registry.skills[name]

        result = func(**args)

        return result
