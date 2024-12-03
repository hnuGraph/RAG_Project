import json
import numpy as np
from transformers import AutoModel, LlamaTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
import os
import pickle
import torch
from datetime import datetime

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

def get_embedding(model, text):
    """获取输入的embedding

    Args:
        model : embedding模型
        text: 输入文本

    Returns:
       embedding: text的embedding
    """
    embedding = model.encode(text)
    return embedding

def get_rule_embeddigs(model, rule_path):
    
    """
    生成规则库的embedding
    
    Args:
        model: sentence_transformers的模型
        rule_path: 规则库的路径
    
    Returns:
        embeddings: 规则库的embedding
    """
    rule_data=get_json_data(rule_path)
    rule_embeddings_path = "../../user_data/data_processed/rule_embeddings.pkl"
    if os.path.exists(rule_embeddings_path):
        print("读取之间保存的rule_embeddings")
        with open(rule_embeddings_path, 'rb') as fin:
            embeddings = pickle.load(fin)
    else:
        topics =  ['国家安全生产监督管理总局危险化学品事故灾难应急预案', '国家地震应急预案', '国家防汛抗旱应急预案', '国家海上搜救应急预案', '国家处置民用航空器飞行事故应急预案', '矿山事故灾难应急预案', '国家突发公共事件医疗卫生救援应急预案', '国家重大海上溢油应急处置预案', '国家通信保障应急预案', '风暴潮、海浪、海啸和海冰灾害应急预案', '重大沙尘暴灾害应急预案', '国家突发公共卫生事件应急预案', '国家食品安全事故应急预案', '中国气象局气象灾害应急预案', '国家森林火灾应急预案', '国家网络安全事件应急预案', '全国草原火灾应急预案', '国家处置铁路行车事故应急预案', '国家核应急预案', '国家大面积停电事件应急预案', '国家城市轨道交通运营突发事件应急预案']
        begins = ['1', '49', '101', '153', '201', '241', '301', '334', '401', '431', '501', '548', '601', '642', '711', '753', '801', '836', '881', '920', '957']
        ends = ['48', '100', '152', '200', '240', '300', '333', '400', '430', '500', '547', '600', '641', '710', '752', '800', '835', '880', '919', '956', '1000']  
        
        topic_data = []
        for i, topic in enumerate(topics):
            topic_data.extend([topic for i in range(int(ends[i])-int(begins[i])+1)])
        
        for i, item in enumerate(rule_data):

            topic_text = topic_data[i]
            item['rule_text'] = f'主题：{topic_text}。内容：' + item['rule_text']  # 更新
        embeddings = np.vstack([get_embedding(model, rule['rule_text']) for rule in rule_data])
        embeddings = embeddings.astype('float32')
        with open(rule_embeddings_path, mode='wb') as f:
            pickle.dump(embeddings, f)
        print("rule_embeddings生成并保存完毕！")
    return embeddings


def retrieve_rule_ids(embeddings, model, query,rule_path, k):
    """ 获取规则集中，与query最相关的k个知识

    Args:
        embeddings : 所有规则的embedding集合
        model : embedding模型
        query : 问题文本
        k : 返回知识的个数

    Returns:
        ids: 规则集中与query最相关的k个知识的id列表
    """
    ids = []
    dimension = embeddings.shape[1]
    query_embedding = get_embedding(model,query)
    query_embedding = query_embedding.reshape(1, -1)  
    query_embedding = query_embedding.astype('float32')  
    index = faiss.IndexFlatL2(dimension) 
    index.add(embeddings)  
    distances, indices = index.search(query_embedding, k)
    rule_data = get_json_data(rule_path)
    rule_id=[rule['rule_id'] for rule in rule_data]
    for i in range(k):
        ids.append(rule_id[indices[0][i]])
    print("--------------------------")
    return ids

def get_topk_rue(input_path, embeddings, model,rule_path, k):
    """ 对输入文件中的每一个问题，找到规则集中与其最相近的top k个规则，返回更新后的数据
    Args:
        input_path: 输入文件路径
        k: 返回知识的个数
    Returns:
        data: 检索到规则集中与问题最相近的top k个规则后的数据
    """

    print(f"直接检索 {input_path} 中每个问题的top {k} 个有关规则")
    new_data = []
    print("embeddings shape")
    print(embeddings.shape)
    data = get_json_data(input_path)
    for item in data:
        query=item["question_text"]
        ids = retrieve_rule_ids(embeddings, model, query,rule_path, k)
        item["retrieve_rule_ids"]= ids
        new_data.append(item)
    return new_data

def rerank_cpm(data, rule_path):
    """
    对输入文件中的每一个问题，重新排序规则集中与其最相近的top k个规则

    Args:
        data: 输入数据
        rule_path: 规则集文件路径

    Returns:
        data: 重新排序后的数据
    """
    class MiniCPMRerankerLLamaTokenizer(LlamaTokenizer):
        def build_inputs_with_special_tokens(
                self, token_ids_0, token_ids_1=None
        ):
            """
            - single sequence: `<s> X </s>`
            - pair of sequences: `<s> A </s> B`

            Args:
                token_ids_0 (`List[int]`):
                    List of IDs to which the special tokens will be added.
                token_ids_1 (`List[int]`, *optional*):
                    Optional second list of IDs for sequence pairs.

            Returns:
                `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
            """

            if token_ids_1 is None:
                return super().build_inputs_with_special_tokens(token_ids_0)
            bos = [self.bos_token_id]
            sep = [self.eos_token_id]
            return bos + token_ids_0 + sep + token_ids_1

    model_name = "../../user_data/MiniCPM-Reranker"
    tokenizer = MiniCPMRerankerLLamaTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True,
                                                               attn_implementation="flash_attention_2",
                                                               torch_dtype=torch.float16).to("cuda")
    model.eval()

    @torch.no_grad()
    def rerank(input_query, input_docs):
        tokenized_inputs = tokenizer([[input_query, input_doc] for input_doc in input_docs], return_tensors="pt",
                                     padding=True, truncation='longest_first', max_length=1024)

        for k in tokenized_inputs:
            tokenized_inputs[k] = tokenized_inputs[k].to("cuda")

        outputs = model(**tokenized_inputs)
        score = outputs.logits
        return score.float().detach().cpu().numpy()

    rule_data = get_json_data(rule_path)
    rule_dict = {rule['rule_id']: rule['rule_text'] for rule in rule_data}
    print(f"对检索的规则重排序，返回重排序后的前10个规则")
    l=[]
    for item in tqdm(data):
        question = item['question_text']
        retrieve_ids = item["retrieve_rule_ids"]

        INSTRUCTION = "为这个问题检索相关规则。"
        passages = []
        for rule_id in retrieve_ids:
            passages.append(rule_dict[rule_id])

        scores = rerank(INSTRUCTION+question, passages)

        sorted_scores_list = sorted(enumerate(scores), key=lambda x: x[1])
        sorted_scores_index = [x[0] for x in sorted_scores_list]
        rerank_rule_list = [retrieve_ids[index] for index in sorted_scores_index[::-1]] 
        item['rerank_rule_ids'] = rerank_rule_list[:10]


    return data


if __name__ == "__main__":
    # 检索模型
    start_time = datetime.now()

    model = SentenceTransformer('../../user_data/MiniCPM-Embedding', trust_remote_code=True,model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.float16})

    rule_path="../../user_data/data_processed/rules_all.json"
    rule_embeddings=get_rule_embeddigs(model, rule_path)
    print("-"*50)

    # 处理 dev.json
    dev_data_path="../../raw_data/dev.json"
    dev_data_retrieve = get_topk_rue(dev_data_path,rule_embeddings,model,rule_path,20)
    dev_data_rerank = rerank_cpm(dev_data_retrieve,rule_path)
    save_json_data(dev_data_rerank, "../../user_data/data_processed/dev_with_topk_rerank_rules.json")
    print("-"*50)

    # 处理 复赛新增训练参考集.json 中的2996条带referened_rule的数据
    dev2_data_path="../../user_data/data_processed/dev2.json"
    dev2_data_retrieve = get_topk_rue(dev2_data_path,rule_embeddings,model,rule_path,20)
    dev2_data_rerank = rerank_cpm(dev2_data_retrieve,rule_path)
    save_json_data(dev2_data_rerank, "../../user_data/data_processed/dev2_with_topk_rerank_rules.json")
    print("-"*50)

    #处理 复赛测试集test.json
    test_data_path="../../raw_data/test.json"
    test_data_retrieve = get_topk_rue(test_data_path,rule_embeddings,model,rule_path,20)
    save_json_data(test_data_retrieve, "../../user_data/data_processed/test_with_topk_retrieve_rules.json")
    test_data_retrieve = get_json_data("../../user_data/data_processed/test_with_topk_retrieve_rules.json")
    test_data_rerank = rerank_cpm(test_data_retrieve,rule_path)
    save_json_data(test_data_rerank, "../../user_data/data_processed/test_with_topk_rerank_rules.json")

    end_time = datetime.now()
    elapsed_time = end_time - start_time  # 获取 timedelta 对象

    print(f"get_topk_rule.py程序运行时间: {elapsed_time}")



    
