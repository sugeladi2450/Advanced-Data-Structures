from agent.registry import registry


@registry.register(
    name="add",
    description="Add two numbers",
    parameters={
        "type": "object",
        "properties": {
            "a": {
                "type": "number",
            },
            "b": {
                "type": "number",
            },
        },
        "required": ["a", "b"],
    },
)
def add(a: float, b: float) -> float:
    return a + b
