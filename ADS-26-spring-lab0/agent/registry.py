from typing import Callable, Dict, Any


class SkillRegistry:
    def __init__(self):
        self.skills: Dict[str, Callable[..., Any]] = {}
        self.schemas: Dict[str, dict] = {}

    def register(self, name: str, description: str, parameters: dict):
        def decorator(func):
            self.skills[name] = func

            self.schemas[name] = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }

            return func

        return decorator


registry = SkillRegistry()
