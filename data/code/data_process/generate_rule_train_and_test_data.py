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

def merge_data(data1, data2):     
    """ 合并两个数据

    Args:
        data1 (list): 数据1
        data2 (list): 数据2

    Returns:
        list: 合并的数据
    """
    new_data=[]
    for item in data1:
        new_data.append(item)

    for item in data2:
        new_data.append(item)
    return new_data


def data_with_rule_text(data, rule_data, is_train):
    
    """
    为data中的检索的规则列表生成对应的文本列表
    
    Args:
        data (list): 原始数据
        rule_data (list): 规则数据
        is_train (bool): 是否为训练数据
    
    Returns:
        list: 合并后的数据
    """

    rule_dict = {rule['rule_id']: rule['rule_text'] for rule in rule_data}
    new_data = []
    for item in data:
        new_selected_rule_id = []
        if is_train==True:
            rule_id=item["rule_id"] 
            rerank_rule_ids=item["rerank_rule_ids"].copy()
            print("rerank_rule_ids:",rerank_rule_ids)
            for rule in rule_id:
                if rule in rerank_rule_ids:
                # 如果在 rerank_rule_ids 中，先移除它
                    rerank_rule_ids.remove(rule)
                # 把golden rule添加到新的列表最前面
                new_selected_rule_id.append(rule)
            # 将剩下的 rerank_rule_ids 的值追加到新列表中
            new_selected_rule_id.extend(rerank_rule_ids)
            # 保证新列表只包含前10个值
            new_selected_rule_id = new_selected_rule_id[:10]
            item["rerank_rule_ids_true_first"] = new_selected_rule_id

            rerank_rule_ids_true_first_shuffle=new_selected_rule_id.copy()  # 创建副本
            random.shuffle(rerank_rule_ids_true_first_shuffle)
            rule_text=""
            for rule in rerank_rule_ids_true_first_shuffle:
                rule_text += "规则ID: " + str(rule) + "， 规则文本：" + rule_dict[str(rule)] + "\n "
            item["rerank_rule_ids_true_first_shuffle"] = rerank_rule_ids_true_first_shuffle
            item["rerank_rule_text_true_first_shuffle"] = rule_text
            new_data.append(item)
        else:
            rerank_rule_ids=item['rerank_rule_ids']
            rule_text=""
            for rule in rerank_rule_ids:
                rule_text += "规则ID: " + str(rule) + "， 规则文本：" + rule_dict[str(rule)] + "\n "
            item["rerank_rule_text"] = rule_text
            new_data.append(item)
    return new_data    

def generate_formatted_data(data, is_train):
    
    """
    生成llama factory需要的格式化的数据
    
    Args:
        data (list): 原始数据
        is_train (bool): 是否为训练数据
    
    Returns:
        list: 格式化后的数据
    """
    new_data=[]
    for item in data:
        input ="你是一个专业的智能问答助手，现在我给你一个问题和它可能回答选项（A,B,C,D），以及对回答问题有帮助的规则列表。"
        input += item["question_text"]
        
        if is_train==True:
            rule_list= item["rerank_rule_ids_true_first_shuffle"]
            input += " \n以下是规则列表:" + str(rule_list) + "。每条规则的ID和文本对应如下，规则之间用换行符分开 ："
            input += item["rerank_rule_text_true_first_shuffle"]
            input += "你需要思考该题的正确回答选项（无需返回答案），并告诉我你选择它参考的规则列表（按照python list形式返回参考的规则id，规则id按照该规则对回答问题的帮助大小由先到后排列)。"
            output=str(item["rerank_rule_ids_true_first"])
        else :
            rule_list= item['rerank_rule_ids']
            input += " \n以下是规则列表:" + str(rule_list) + "。每条规则的ID和文本对应如下，规则之间用换行符分开 ："
            input += item["rerank_rule_text"]
            input += "你需要思考该题的正确回答选项（无需返回答案），并告诉我你选择它参考的规则列表（按照python list形式返回参考的规则id，规则id按照该规则对回答问题的帮助大小由先到后排列)。"
            output=""
        new_item = {
            "instruction": "" ,
            "input": input,
            "output": output 
        }
        new_data.append(new_item)
    return new_data


if __name__ == "__main__":
    folder_path = '../../user_data/rule_data'  

    # 创建多级文件夹
    os.makedirs(folder_path, exist_ok=True) 
    print(f"文件夹 '{folder_path}' 已创建")

    rule_path="../../user_data/data_processed/rules_all.json"
    rule_data=get_json_data(rule_path)

    # 生成训练数据并格式化
    dev_data_path="../../user_data/data_processed/dev_with_topk_rerank_rules.json"
    dev_data=get_json_data(dev_data_path)
    dev_data_with_rule_text = data_with_rule_text(dev_data,rule_data,True)

    dev2_data_path="../../user_data/data_processed/dev2_with_topk_rerank_rules.json"
    dev2_data=get_json_data(dev2_data_path)
    dev2_data_with_rule_text = data_with_rule_text(dev2_data,rule_data,True)

    train_data= merge_data(dev_data_with_rule_text,dev2_data_with_rule_text)
    save_json_data(train_data,"../../user_data/rule_data/train_data.json")
    
    train_data_formatted = generate_formatted_data(train_data,True)
    save_json_data(train_data_formatted,"../LLaMA-Factory/data/rule_train.json")

    # 生成测试数据并格式化
    test_data_path="../../user_data/data_processed/test_with_topk_rerank_rules.json"
    test_data=get_json_data(test_data_path)
    test_data_with_rule_text = data_with_rule_text(test_data,rule_data,False)
    save_json_data(test_data_with_rule_text,"../../user_data/rule_data/test_data.json")

    test_data_formatted = generate_formatted_data(test_data_with_rule_text,False)
    save_json_data(test_data_formatted,"../LLaMA-Factory/data/rule_inference.json")
    





