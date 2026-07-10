from openai import OpenAI

from agent.registry import SkillRegistry
from agent.runtime import SkillRuntime
from agent.loader import load_all_skills

API_KEY = ""
BASE_URL = "https://models.sjtu.edu.cn/api/v1"
MODEL = "qwen3coder"
TEMPERATURE = 0

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


def agent_loop(registry: SkillRegistry, runtime: SkillRuntime, messages: list[dict]):
    all_skills = load_all_skills()
    if all_skills:
        messages = [{"role": "system", "content": all_skills}] + messages

    tools = list(registry.schemas.values())

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            temperature=TEMPERATURE,
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

        tool_call = msg.tool_calls[0]

        print(f"Tool Call: {tool_call}")  # for debugging

        result = runtime.run(
            {"name": tool_call.function.name, "arguments": tool_call.function.arguments}
        )

        messages.append(msg)

        messages.append(
            {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)}
        )
