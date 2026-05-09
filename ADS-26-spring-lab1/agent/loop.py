import json
import os
import re
import time
from typing import List, Dict, Any
from openai import OpenAI, APIConnectionError
from skills.registry import SkillRegistry
from skills.runtime import SkillRuntime

# TODO: 在 https://form.sjtu.edu.cn/infoplus/form/net_ai_api_apply/start?locale=zh 获取 API key 并填入
API_KEY = os.getenv("SJTU_API_KEY", "<your-api-key>")
BASE_URL = "https://models.sjtu.edu.cn/api/v1"
MODEL = "deepseek-chat"
TEMPERATURE = 0

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=60.0,
)


def extract_fake_tool_calls(content: str) -> List[Dict[str, Any]]:
    """
    针对 DeepSeek-R1 的幻觉：从文本中提取 Markdown 格式的 JSON 工具调用。
    """
    if not content:
        return []
    
    # 匹配 ```json { ... } ``` 块
    pattern = r"```json\s*(\{.*?\})\s*```"
    matches = re.findall(pattern, content, re.DOTALL)
    
    results = []
    for m in matches:
        try:
            data = json.loads(m)
            # 兼容模型可能输出的 action/action_input 格式
            if "action" in data and "action_input" in data:
                results.append({
                    "id": f"fake_{int(time.time())}_{len(results)}",
                    "function": {
                        "name": data["action"],
                        "arguments": json.dumps(data["action_input"])
                    }
                })
            # 兼容模型直接输出函数名和参数的格式
            elif "name" in data and "arguments" in data:
                results.append({
                    "id": f"fake_{int(time.time())}_{len(results)}",
                    "function": {
                        "name": data["name"],
                        "arguments": json.dumps(data["arguments"]) if isinstance(data["arguments"], dict) else data["arguments"]
                    }
                })
        except:
            continue
    return results


def agent_loop(
    registry: SkillRegistry,
    runtime: SkillRuntime,
    messages: list[dict],
    skill_markdown: str | None = None,
):
    """
    Agent Loop：
    1. 支持多工具并行调用。
    2. 自动重试网络连接错误。
    3. 容错处理模型幻觉（正则提取工具调用）。
    """
    MAX_STEPS = 10
    MAX_RETRIES = 3
    tools = list(registry.schemas.values())
    step = 0
    
    system_prompt = (
        "你是一个专业的图数据库分析助手。\n"
        "你的任务是根据提供的 Skill 工具来解决用户的图查询问题。\n\n"
        "规则：\n"
        "1. **多步规划**：复杂的任务通常需要多次调用工具。不要急于给出最终答案。\n"
        "2. **并行调用**：如果可以同时查询多个节点，请在一轮中发起多个 tool_calls。\n"
        "3. **持续探索**：如果当前工具返回的结果不足以回答问题，请根据新线索继续调用工具。\n"
        "4. **标准格式**：请务必使用标准的 tool_calls 字段发起调用，不要在回复正文中手动编写 JSON 块。\n"
    )

    if skill_markdown:
        system_msg = {
            "role": "system",
            "content": system_prompt + "可用工具说明：\n" + skill_markdown,
        }
        run_messages = [system_msg] + messages
    else:
        run_messages = messages

    while step < MAX_STEPS:
        step += 1
        print(f"\n[Step {step}] 调用 LLM (model={MODEL}) ...")
        
        response = None
        for retry in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=run_messages,
                    tools=tools,
                    temperature=TEMPERATURE,
                )
                break
            except (APIConnectionError, Exception) as e:
                print(f"  [Retry {retry+1}] API 请求失败: {type(e).__name__}: {e}")
                if retry == MAX_RETRIES - 1:
                    print("  达到最大重试次数，尝试跳过本轮或报错。")
                    raise
                time.sleep(2 ** retry) # 指数退避

        msg = response.choices[0].message
        run_messages.append(msg)

        # 获取工具调用（标准格式）
        final_tool_calls = []
        if msg.tool_calls:
            final_tool_calls = msg.tool_calls
        
        # 容错：如果标准的 tool_calls 为空，尝试从 content 中提取伪造的调用
        if not final_tool_calls and msg.content:
            fake_calls = extract_fake_tool_calls(msg.content)
            if fake_calls:
                print(f"  [Debug] 检测到幻觉工具调用，尝试手动解析并执行...")
                from types import SimpleNamespace
                for c in fake_calls:
                    # 将字典模拟成 SimpleNamespace 以便后续统一访问 .function.name
                    tc = SimpleNamespace(**c)
                    tc.function = SimpleNamespace(**tc.function)
                    final_tool_calls.append(tc)

        if not final_tool_calls:
            print(f"[Step {step}] LLM 返回最终回答（无 tool_call），结束。")
            print(f"\n===== Agent Final Answer =====\n{msg.content}\n==============================\n")
            return msg.content

        print(f"[Step {step}] 模型请求调用 {len(final_tool_calls)} 个工具")
        
        # 处理本轮所有的工具调用
        for tool_call in final_tool_calls:
            name = tool_call.function.name
            args = tool_call.function.arguments
            print(f"  - 执行 skill: {name}, 参数: {args}")

            try:
                # 统一执行接口
                result = runtime.run({"name": name, "arguments": args})
                print(f"    结果: {result}")
                
                run_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False)
                })
            except Exception as e:
                print(f"    执行出错: {e}")
                run_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": f"Error: {str(e)}"
                })

        print(f"[Step {step}] 已将所有 tool 结果写入 messages，继续下一轮 LLM 调用。")

    return "达到最大步数限制，未能完成任务。"
