import json
from vllm import LLM, SamplingParams
from datetime import datetime
import re


def get_json_data(path):
    """ 读取json文件

    Args:
        path: 文件路径

    Returns:
        data: json数据
    """ 
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_bracket_content(s):
    """
    从给定的字符串s中抽出有用的部分，即推理生成的id列表
    Args:
        s (str): 输入的字符串

    Returns:
        str: 输出的字符串
    """
    match = re.search(r'\[.*?\]', s)  # 使用正则表达式匹配 '[...]' 的内容
    if match:
        return match.group(0)  # 提取匹配的部分
    return None  # 如果没有找到，返回 None

def save_json_data(data, path):
    """保存json文件到指定路径

    Args:
        data (_type_): 要保存的数据
        path (_type_): 保存文件的路径
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    start_time = datetime.now()
    cnt=0
    llm = LLM(model='../../user_data/rule_sft_lora_merge_qwen_model/',gpu_memory_utilization=0.9,max_model_len=20000)

    sampling_params = SamplingParams(
        temperature=0.9,  # 控制模型输出的多样性
        top_p=0.9,        # Top-p采样
        max_tokens=80    # 生成的最大 token 数量
    )
    
    input_file="../../user_data/rule_data/test_data.json"
    data=get_json_data(input_file)
    new_data=[]
    for item in data:
        cnt+=1
        rule_id=item["rerank_rule_ids"]
        input ="你是一个专业的智能问答助手，现在我给你一个问题和它可能回答选项（A,B,C,D），以及可能对回答问题有帮助的规则列表。"
        input += item["question_text"]
        input += " \n以下是规则列表:" + str(rule_id) + "。每条规则的ID和文本对应如下，规则之间用换行符分开 ："
        input += item["rerank_rule_text"]
        input += "你需要思考该题的正确回答选项（无需返回答案），并告诉我你选择它参考的规则列表（按照python list形式返回参考的规则id，规则id按照该规则对回答问题的帮助大小由先到后排列）"

        outputs = llm.generate(input, sampling_params)
        # 6. 打印输出结果
        prediction=outputs[0].outputs[0].text  # 输出模型生成的文本
        prediction=extract_bracket_content(prediction)
        prediction = prediction.replace("'", '"')
        prediction = json.loads(prediction)

        prediction2 = list(map(str,prediction))

        item["prediction"]= prediction2

        if 'retrieve_rule_ids' in item:
            del item['retrieve_rule_ids']
        new_data.append(item)
        
    output_file="../../user_data/rule_data/test_with_predict_rule2.json"
    save_json_data(new_data, output_file)

    end_time = datetime.now()
    elapsed_time = end_time - start_time  # 获取 timedelta 对象

    print(f"vllm_generate_rule.py程序运行时间: {elapsed_time}")