
"""
裁判模型逻辑1：全入全出
将整条轨迹一次性输入，一次性输出所有冗余步骤的判断结果
"""

from openai import OpenAI
import json


def read_messages(json_file_path):
    """读取 JSON 文件中的 messages 列表"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('simulations', [])


def read_tasks(json_file_path):
    """读取 JSON 文件中的 tasks 列表，提取每条任务的 purpose"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tasks = data.get('tasks', [])
    return [t.get('description', {}).get('purpose', '') for t in tasks]


def judge_redundancy_all_in_all_out(messages, task_purpose=""):
    """
    全入全出逻辑：一次性输入整条轨迹，输出所有冗余步骤
    """
    base_url = "https://api.openai-proxy.org/v1"
    model_name = "gpt-4o-2024-08-06"

#     sysprompt = '''你是一个负责寻找轨迹冗余步骤的裁判，检查轨迹中由assistant发起的toolcall动作以及tool执行结果是否冗余，判断逻辑为是否包含以下3点任意一点：
# 1. 重复步骤、无效工具调用
# 2. 不改变环境状态
# 3. 偏离任务目标'''
    sysprompt = '''你是一个负责寻找轨迹冗余步骤的裁判，检查轨迹中由assistant发起的toolcall动作以及tool执行结果是否冗余，判断逻辑为是否包含以下3点任意一点：
1. 无效工具调用（例如因为网络问题，工具调用没有返回正常结果）
2. 重复步骤（工具重复调用，且结果没有变化）
3. 不在ground truth action范围内，且没有给ground truth action执行提供参数
4. 偏离任务目标'''

    purpose_section = f"""
任务目标（ground truth）：
{task_purpose}
""" if task_purpose else ""

    prompt = f"""请根据以下信息进行任务完成度评估：
{purpose_section}
- 整条轨迹信息（messages里的每个字典代表一步，一步可能是用户输入，模型思考，toolcall）：
{messages}

请严格依据上述信息，判断：**该轨迹中是否包含冗余动作**

请仅以如下 JSON 格式输出结果，不要包含任何额外文字、解释或 Markdown：

{{"冗余动作在messages里的位置（位置序号从0开始）": [1, 2, 3], "reason": "简要说明判断依据"}}

如果没有冗余动作则输出：
{{"冗余动作在messages里的位置（位置序号从0开始）": [], "reason": "简要说明判断依据"}}"""

    client_roma = OpenAI(
        base_url=base_url,
        api_key="sk-emC4sdL85VD55BsnpHwcQs7GIyBz0ZpNzZ3XKvqLXpqLqk15",
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
    all = read_messages('airline_results.json')
    tasks = read_tasks('airline_results.json')
    print(f"共有 {len(all)} 条轨迹，{len(tasks)} 个任务目标")

    all_results = []
    for m in range(0, len(all)):
        messages = all[m]['messages']
        task_purpose = tasks[m] if m < len(tasks) else ""
        print(f"\n=== 轨迹 {m}，共有 {len(messages)} 条消息 ===")
        if task_purpose:
            print(f"任务目标: {task_purpose[:100]}...")

        result = judge_redundancy_all_in_all_out(messages, task_purpose)
        print("AI 响应:", result)

        parsed_result = None
        try:
            parsed_result = json.loads(result)
        except json.JSONDecodeError:
            print("无法解析 JSON 结果")

        all_results.append({
            "trajectory_index": m,
            "message_count": len(messages),
            "task_purpose": task_purpose,
            "raw_response": result,
            "parsed_result": parsed_result
        })

    output_file = '../all_all_airline_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n所有结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
