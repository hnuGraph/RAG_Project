import json
import random

def get_json_data(path):
    """ 读取json文件

    Args:
        path: 文件路径

    Returns:
        data: json数据
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_json_data(data, path):
    """保存json文件到指定路径

    Args:
        data : 要保存的数据
        path : 保存文件的路径
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def jsonl_to_json(jsonl_file, json_file):
    # 读取 JSONL 文件
    with open(jsonl_file, 'r') as f:
        jsonl_data = f.readlines()
    
    # 解析 JSONL 中的每一行并存储为 JSON 对象
    json_objects = [json.loads(line.strip()) for line in jsonl_data]
    
    # 将 JSON 对象列表转换为 JSON 数组，并保留Unicode字符
    json_array = json.dumps(json_objects, indent=4, ensure_ascii=False)
    
    # 将 JSON 数组写入 JSON 文件
    with open(json_file, 'w') as f:
        f.write(json_array)


def data_process(data):
    new_data=[]
    for item in data:
        prediction=item["predict"]
        first_char = prediction[0] if prediction else None
        # 如果第一个字符是 "A"、"B"、"C" 或 "D"，则直接返回
        if first_char in ["A", "B", "C", "D"]:
            prediction2=first_char
        else:
            prediction2=random.choice(["A", "B", "C", "D"])
        item["prediction"]=prediction2
        new_data.append(item)
    return new_data
        
if __name__ == "__main__":
    option_path="../LLaMA-Factory/saves/qwen2.5-7B/lora/sft_option_inference/generated_predictions.jsonl"
    option_path2="../LLaMA-Factory/saves/qwen2.5-7B/lora/sft_option_inference/generated_predictions.json"
    jsonl_to_json(option_path, option_path2)
    option_data=get_json_data(option_path2)
    option_data_processed=data_process(option_data)
    save_json_data(option_data_processed,"../../user_data/option_data/test_with_predict_option.json")