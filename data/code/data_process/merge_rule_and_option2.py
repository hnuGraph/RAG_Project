import json

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
        data : 要保存的数据
        path : 保存文件的路径
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def merge_result(rule_data, option_data):     
    """ 合并rule和option

    Args:
        rule_data (list): rule数据
        option_data (list): option数据

    Returns:
        list: 合并的数据
    """
    new_data=[]
    for i in range(len(rule_data)):
        option_item=option_data[i]
        rule_item=rule_data[i]
        new_item={
            "question_id":rule_item["question_id"],
            "answer":option_item["prediction"],
            "rule_id":rule_item["prediction"]
        }
        new_data.append(new_item)
    return new_data 

if __name__ == "__main__":
    rule_path="../../user_data/rule_data/test_with_predict_rule2.json"
    option_path="../../user_data/option_data/test_with_predict_option2.json"
    rule_data=get_json_data(rule_path)
    option_data=get_json_data(option_path)
    result=merge_result(rule_data,option_data)
    save_json_data(result,"../../prediction_result/result2.json")