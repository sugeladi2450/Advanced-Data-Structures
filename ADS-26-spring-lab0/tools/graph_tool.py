from typing import List

from agent.registry import registry
from examples.example_graph import example_graph


@registry.register(
    name="neighbors",
    description="Get the neighbors of a node in the graph",
    parameters={
        "type": "object",
        "properties": {
            "node": {
                "type": "string",
            }
        },
        "required": ["node"],
    },
)
def neighbors(node: str) -> List[str]:
    return example_graph.get(node, [])
