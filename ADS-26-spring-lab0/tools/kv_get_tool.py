from typing import Optional

from agent.registry import registry
from examples.example_kv import example_kv_store


@registry.register(
    name="kv_get",
    description="Get the value of a key in the KV store if it exists else return None",
    parameters={
        "type": "object",
        "properties": {
            "key": {
                "type": "integer",
            }
        },
        "required": ["key"],
    },
)
def kv_get(key: int) -> Optional[str]:
    return example_kv_store.get(key)
