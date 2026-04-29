"""
裁判模型逻辑2：逐条判断
逐条处理 messages，只给当前消息 + 规则来判断是否冗余。
跳过不含 tool_call 且不是 tool 结果的消息。
"""

from openai import OpenAI
import json


def read_messages(json_file_path):
    """读取 JSON 文件中的 simulations 列表"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('simulations', [])


def read_tasks(json_file_path):
    """读取 JSON 文件中的 tasks 列表，提取每条任务的 purpose"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tasks = data.get('tasks', [])
    return [t.get('description', {}).get('purpose', '') for t in tasks]


def has_tool_call_or_result(message):
    """判断消息是否包含 tool_call 或是 tool 的结果"""
    role = message.get('role', '')
    tool_calls = message.get('tool_calls')
    if role == 'tool':
        return True
    if role == 'assistant' and tool_calls is not None and len(tool_calls) > 0:
        return True
    return False


def judge_single_message(message, task_purpose=""):
    """
    逐条判断逻辑：只给当前消息 + 规则 + 任务目标，判断该消息是否冗余
    """
    base_url = "https://api.deepseek.com"
    model_name = "deepseek-v4-pro"

    sysprompt = '''你是一个负责寻找轨迹冗余步骤的裁判，检查轨迹中由assistant发起的toolcall动作以及tool执行结果是否冗余，判断逻辑为是否包含以下3点任意一点：
1. 无效工具调用（例如因为网络问题，工具调用没有返回正常结果）
2. 重复步骤（工具重复调用，且结果没有变化）
3. 不在ground truth action范围内，且没有给ground truth action执行提供参数
4. 偏离任务目标'''

    purpose_section = f"""
任务目标（ground truth）：
{task_purpose}
""" if task_purpose else ""

    prompt = f"""请根据以下单条轨迹消息判断是否冗余：
{purpose_section}
当前消息：
{json.dumps(message, ensure_ascii=False, indent=2)}

请严格依据上述信息，判断：**该消息是否为冗余动作**

请仅以如下 JSON 格式输出结果，不要包含任何额外文字、解释或 Markdown：

{{"is_redundant": true, "reason": "简要说明判断依据"}}

如果不是冗余动作则输出：
{{"is_redundant": false, "reason": "简要说明判断依据"}}"""

    client_roma = OpenAI(
        base_url=base_url,
        api_key="",
        default_headers={"Content-Type": "application/json",
                         "csb-token": ""},
    )

    messages_list = [
        {"role": "system", "content": sysprompt},
        {"role": "user", "content": prompt}
    ]

    completion = client_roma.chat.completions.create(
        model=model_name,
        messages=messages_list,
        max_tokens=4000,
        stream=False
    )

    ai_response = completion.choices[0].message.content
    return ai_response


def main():
    all = read_messages('airline_resultstest.json')
    tasks = read_tasks('airline_resultstest.json')
    print(f"共有 {len(all)} 条轨迹，{len(tasks)} 个任务目标")

    all_results = []
    for m in range(0, 1):
        messages = all[m].get('messages', [])
        task_purpose = tasks[m] if m < len(tasks) else ""
        print(f"\n=== 轨迹 {m}，共有 {len(messages)} 条消息 ===")
        if task_purpose:
            print(f"任务目标: {task_purpose[:100]}...")

        trajectory_result = {
            "trajectory_index": m,
            "message_count": len(messages),
            "task_purpose": task_purpose,
            "judged_messages": []
        }

        for i, msg in enumerate(messages):
            if not has_tool_call_or_result(msg):
                continue

            print(f"  判断消息 {i} (role={msg.get('role')})...")
            result = judge_single_message(msg, task_purpose)
            print(f"    AI 响应: {result}")

            parsed_result = None
            try:
                parsed_result = json.loads(result)
            except json.JSONDecodeError:
                print("    无法解析 JSON 结果")

            trajectory_result["judged_messages"].append({
                "message_index": i,
                "role": msg.get('role'),
                "raw_response": result,
                "parsed_result": parsed_result
            })

        all_results.append(trajectory_result)

    output_file = '../one_one_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n所有结果已保存到 {output_file}")


if __name__ == "__main__":
    main()