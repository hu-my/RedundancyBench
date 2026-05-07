
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


def response_parser(response, model_name='gpt-4o'):
    if '```json' in response:
        response = response.split('json\n')[-1][:-4]

    return response



def read_tasks(json_file_path):
    """读取 JSON 文件中的 tasks 列表，提取每条任务的 purpose"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tasks = data.get('tasks', [])
    return {t.get('task_id'): t.get('description', {}).get('purpose', '') for t in tasks}


def filter_messages(messages):
    new_messages = []
    for msg in messages:
        if msg['role'] == 'tool':
            new_messages.append(msg)
        else: # 'role' = assistant or user
            new_msg = {}
            new_msg['role'] = msg['role']
            new_msg['content'] = msg['content']
            new_msg['tool_calls'] = msg['tool_calls']
            new_msg['turn_idx'] = msg['turn_idx']
            new_msg['timestamp'] = msg['timestamp']
            new_msg['usage'] = msg['usage']
            new_messages.append(new_msg)
    return new_messages


def judge_redundancy_all_in_all_out(messages, task_purpose=""):
    """
    全入全出逻辑：一次性输入整条轨迹，输出所有冗余步骤
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

    prompt = f"""Please evaluate the task completion based on the following information:
{purpose_section}
- The entire trajectory information (each dictionary in messages represents a step, and a step could be user input, model thinking, or tool call):
{messages}

Please strictly base your judgment on the above information: **Does this trajectory contain redundant actions?** 

Please output the result only in the following JSON format and do not include any additional text, explanations or Markdown:

{{"Redundant actions' positions in messages (position index starts from 0)": [1, 2, 3], "reason": "Briefly explain the basis for judgment"}}

If there are no redundant actions, output:
{{"Positions of redundant actions in messages (position index starts from 0)": [], "reason": "Briefly explain the basis for the judgment"}}"""

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
    all = read_messages('airline_results.json')
    tasks = read_tasks('airline_results.json')


    all_results = []
    for m in range(0, len(all)):
        messages = filter_messages(all[m]['messages'])
        task_purpose = tasks.get(all[m]['task_id'], "")



        result = judge_redundancy_all_in_all_out(messages, task_purpose)
        print("AI:", result)

        parsed_result = None
        try:
            parsed_result = json.loads(response_parser(result))
        except json.JSONDecodeError:
            print("error")

        all_results.append({
            "trajectory_index": m,
            "message_count": len(messages),
            "task_purpose": task_purpose,
            "raw_response": result,
            "parsed_result": parsed_result
        })

    output_file = 'dsv4pro_all_all_airline_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nsave to {output_file}")


if __name__ == "__main__":
    main()
