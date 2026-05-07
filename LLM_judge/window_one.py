"""
裁判模型逻辑3：滑动窗口判断
逐条处理 messages，给当前消息 + 前后各3条上下文（共最多7条）来判断是否冗余。
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
    """读取 JSON 文件中的 tasks 列表，返回 {task_id: purpose} 映射"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tasks = data.get('tasks', [])
    return {t.get('task_id'): t.get('description', {}).get('purpose', '') for t in tasks}


def has_tool_call_or_result(message):
    """判断消息是否包含 tool_call 或是 tool 的结果"""
    role = message.get('role', '')
    tool_calls = message.get('tool_calls')
    if role == 'tool':
        return True
    if role == 'assistant' and tool_calls is not None and len(tool_calls) > 0:
        return True
    return False


def judge_message_with_window(target_message, window_messages, target_index_in_window, task_purpose=""):
    """
    滑动窗口判断逻辑：给当前消息 + 前后各3条上下文，判断当前消息是否冗余
    """
    base_url = ""
    model_name = ""

    sysprompt = '''You are a judge responsible for identifying redundant steps in the trajectory. Your task is to examine the "toolcall" actions initiated by the assistant and the corresponding execution results of the "tools" to determine if they are redundant. The criteria for judgment include any of the following three points:
    1. Toolcalls and their results should be considered redundant if they are not essential steps for completing the objective(Mark the suspicious ones as redundant).
    2. Invalid tool invocation (for example, due to network issues, the tool invocation did not return a normal result)
    3. Repeated steps (repeated tool invocation with no change in the result)
    for example, the messages includes:{
    { "role": "assistant",XXX,"tool_calls": AB,"turn_idx": 4},
    {XXX"role": "tool","requestor": "assistant","turn_idx": 5,XXX},
    { "role": "user",XXX,"tool_calls": CD,"turn_idx": 6},          
    }, 
    You should mark 4 and 5 simultaneously. If the "toolcall" mentioned in 4 is considered redundant, then you should do so.
    You don't need to consider "toolcall" in 6 because 6 was not initiated by the assistant,You also don't need to consider the messages initiated by the assistant either, but those where the toolcall is empty.'''

    purpose_section = f"""
    Task objective：
    {task_purpose}
    """ if task_purpose else ""

    window_json = json.dumps(window_messages, ensure_ascii=False, indent=2)

    prompt = f"""Please evaluate the task completion based on the following information:
    {purpose_section}
    The message list in the window (each dictionary represents one step, please pay special attention to the one marked as 【Current judgment message】):
    {window_json}
    
    The current judgment message is the message with index {target_index_in_window} in the window.
    
    Please strictly base your judgment on the above information: **Does this trajectory contain redundant actions?** 

    Please output the result only in the following JSON format and do not include any additional text, explanations or Markdown:

    {{"is_redundant": true, "reason": "A brief explanation of the basis for the judgment."}}

    If there are no redundant actions, output:
    {{"is_redundant": false, "reason": "A brief explanation of the basis for the judgment."}}"""

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
    all = read_messages('telecom_results.json')
    tasks = read_tasks('telecom_results.json')

    all_results = []
    for m in range(0, len(all)):
        messages = all[m].get('messages', [])
        task_purpose = tasks.get(all[m]['task_id'], "")


        trajectory_result = {
            "trajectory_index": m,
            "message_count": len(messages),
            "task_purpose": task_purpose,
            "judged_messages": []
        }

        for i, msg in enumerate(messages):
            if not has_tool_call_or_result(msg):
                continue

            # 取前后各3条，包含当前消息
            start = max(0, i - 3)
            end = min(len(messages), i + 4)
            window = messages[start:end]
            target_index_in_window = i - start

            result = judge_message_with_window(msg, window, target_index_in_window, task_purpose)

            parsed_result = None
            try:
                parsed_result = json.loads(result)
            except json.JSONDecodeError:
                print("    JSON error")

            trajectory_result["judged_messages"].append({
                "message_index": i,
                "role": msg.get('role'),
                "window_range": [start, end],
                "raw_response": result,
                "parsed_result": parsed_result
            })

        all_results.append(trajectory_result)

    output_file = ''
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nsave to {output_file}")


if __name__ == "__main__":
    main()
