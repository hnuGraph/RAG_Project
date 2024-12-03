import re
import os
import json
import random

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

def save_json_data(data, path):
    """保存json文件到指定路径

    Args:
        data (_type_): 要保存的数据
        path (_type_): 保存文件的路径
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def data_with_rule_text(data, rule_data):
    
    """
    为data中的检索的规则列表生成对应的文本列表
    
    Args:
        data (list): 原始数据
        rule_data (list): 规则数据
    
    Returns:
        list: 合并后的数据
    """

    rule_dict = {rule['rule_id']: rule['rule_text'] for rule in rule_data}

    new_data = []
    for item in data:
        prediction_rule_ids=item['prediction'][:5]
        rule_text=""
        for rule in prediction_rule_ids:
            rule_text += "规则ID: " + str(rule) + "， 规则文本：" + rule_dict[str(rule)] + "\n "
        item["predict_rule_id_text"] = rule_text
        new_data.append(item)
    return new_data    

def generate_formatted_data(data):
    
    """
    生成llama factory需要的格式化的数据
    
    Args:
        data (list): 原始数据
    Returns:
        list: 格式化后的数据
    """
    new_data=[]
    prompt = '你是一个专业的智能问答助手，现在我向你提出问题，并给出问题的可能回答选项（A,B,C,D）,以及回答问题所需的规则。请你根据题意和规则，一步一步地思考和计算，注意不要利用你自己的知识。最后，请返回你认为最符合题意的回答选项（仅输出A或B或C或D，无其余额外的文字内容输出）'
    for item in data:
        instruction = item['question_text'] + "\n 以下是所有规则：\n" + item["predict_rule_id_text"]
        output=""
        new_item = {
           'question_id': item["question_id"],
           'instruction': instruction,
           'input': prompt, 
           'output':  output
        }
        new_data.append(new_item)
    return new_data

if __name__ == "__main__":
    folder_path = '../../user_data/option_data'  

    # 创建多级文件夹
    os.makedirs(folder_path, exist_ok=True) 
    print(f"文件夹 '{folder_path}' 已创建")

    rule_path="../../user_data/data_processed/rules_all.json"
    rule_data=get_json_data(rule_path)

    # 生成测试数据并格式化
    test_data_path="../../user_data/rule_data/test_with_predict_rule2.json"
    test_data=get_json_data(test_data_path)
    test_data_with_rule_text = data_with_rule_text(test_data,rule_data)
    save_json_data(test_data_with_rule_text ,"../../user_data/option_data/test_data2.json")
    test_data_formatted = generate_formatted_data(test_data_with_rule_text)
    save_json_data(test_data_formatted,"../LLaMA-Factory/data/option_inference2.json")
    





