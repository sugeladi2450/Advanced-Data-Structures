---
name: add-tool
description: Teaches the LLM how to use the add tool for numeric addition. Use when the user asks to add numbers, sum values, or perform addition operations.
---

# Add Tool 使用指南

## 何时使用

当用户要求做**加法运算**（如 "Add 1 and 2"、"计算 3+5"、"sum of a and b"）时，应使用 `add` 工具，而不是在文本中猜测或输出占位符。

## 使用方法

1. **调用 add 工具**：传入两个数字参数 `a` 和 `b`
   - 示例：`add(a=1, b=2)` 返回 `3`
   - 参数类型：`a` 和 `b` 均为 number

2. **获取工具返回结果后**：将结果**直接作为最终回答**，不要输出额外文本。

3. **禁止行为**：
   - 不要在回复中输出 `<tool_call>` 等占位符
   - 不要重复发起不必要的工具调用
   - 不要在得到数字结果后再次调用 add

## 工作流程

```
用户请求加法 → 调用 add(a, b) → 收到数字结果 → 直接返回该数字（仅此而已）
```

## 示例

**输入**：Add 1 and 2. Use the add tool. Return only the result.

**正确流程**：
1. 调用 `add(a=1, b=2)` → 得到 `3`
2. 最终回复：`3`（仅这一行）

**错误示例**：
- 回复 `3\n<tool_call>` ❌
- 得到 3 后再次调用 add ❌
- 回复 "The result is 3" 而非仅 "3"（若要求仅返回结果）
