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
        new_selected_rule_id = []
        rule_id=item["rule_id"] 
        rerank_rule_ids=item["rerank_rule_ids"].copy()
        for rule in rule_id:
            rule=int(rule)
            if rule in rerank_rule_ids:
            # 如果在 rerank_rule_ids 中，先移除它
                rerank_rule_ids.remove(rule)
            # 把golden rule添加到新的列表最前面
            new_selected_rule_id.append(rule)
        # 将剩下的 rerank_rule_ids 的值追加到新列表中
        new_selected_rule_id.extend(rerank_rule_ids)
        # 保证新列表只包含前10个值
        new_selected_rule_id = new_selected_rule_id[:5]
        item["rerank_rule_ids_true_first"] = new_selected_rule_id

        rerank_rule_ids_true_first_shuffle=new_selected_rule_id.copy()  # 创建副本
        random.shuffle(rerank_rule_ids_true_first_shuffle)
        
        rule_text=""
        for rule in rerank_rule_ids_true_first_shuffle:
            rule_text += "规则ID: " + str(rule) + "， 规则文本：" + rule_dict[str(rule)] + "\n "
        item["rerank_rule_ids_true_first_shuffle"] = rerank_rule_ids_true_first_shuffle
        item["rerank_rule_text_true_first_shuffle"] = rule_text
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
        instruction = item['question_text'] + "\n 以下是所有规则：\n" + item["rerank_rule_text_true_first_shuffle"]
        output=item["answer"]
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

    # 生成训练数据并格式化
    dev_data_path="../../user_data/data_processed/dev_with_topk_rerank_rules.json"
    dev_data=get_json_data(dev_data_path)
    dev_data_with_rule_text = data_with_rule_text(dev_data,rule_data)
    save_json_data(dev_data_with_rule_text ,"../../user_data/option_data/train_data.json")
    train_data_formatted = generate_formatted_data(dev_data_with_rule_text)
    save_json_data(train_data_formatted,"../LLaMA-Factory/data/option_train.json")





