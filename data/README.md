# 目录结构
    |--- raw_data 存放所有比赛数据集文件

    |--- user_data 存放用户数据文件，包括模型、数据等
 
        |--- data_processed 比赛数据集处理后的中间文件

        |--- MiniCPM-Embedding.zip 编码模型文件

        |--- MiniCPM-Reranker.zip 重排序模型文件

        |--- option_sft_lora.zip 答案生成模型的lora模块

        |--- qwen2.5-7B 基座模型

        |--- rule_sft_lora 规则排序模型的lora模块

        |--- rule_sft_lora_merge_qwen_model 规则排序模型的lora模块和qwen2.5-7B的合并模型(用我们训好的lora推理时，我们会生成此文件，用于vllm推理)

    |--- prediction_result 预测结果

    |--- code 下存放所有用户代码

        |--- data_process 下存放数据处理文件

            |--- data_preprocess.py  数据预处理，取出复赛新增训练集里的3000条有标签数据

            |--- merge_rule.py 合并rule1和rule2

            |--- get_topk_rule.py 为数据集中的每个问题检索知识库中的top10个相关规则

            |--- generate_rule_train_and_test_data.py 生成用于微调rule模型的训练数据，和生成用微调后的rule模型的推理测试集的相关rule的数据集

            |--- generate_option_train_data.py 生成用于微调option模型的训练数据

            |--- vllm_generate_rule.py 用vllm推理微调后的rule模型，生成的测试集上的rule list

            |--- vllm_generate_rule2.py 用vllm推理微调后的rule模型，生成的测试集上的rule list(用我们团队已经生成的模型推理)

            |--- generate_option_test_data.py 生成推理微调后的option模型的测试数据

            |--- generate_option_test_data2.py 生成推理微调后的option模型的测试数据(用我们团队已经生成的模型推理)

            |--- process_option_prediction.py 推理生成的option后处理

            |--- process_option_prediction2.py 推理生成的option后处理(用我们团队已经生成的模型推理)

            |--- merge_rule_and_option.py 合并推理生成的rule list和option，生成最终结果

            |--- merge_rule_and_option2.py 合并推理生成的rule list和option，生成最终结果(用我们团队已经生成的模型推理)

        |--- LLaMA-Factory 下存放微调框架

        |--- run1.sh 从训练到推理程序入口

        |--- run2.sh 直接使用我们的生成模型推理程序入口

# 运行
用户可通过执行/data/code下的脚本run1.sh，直接实现训练到推理过程，预测结果将存放在prediction_result/result.json下。

用户可通过执行/data/code下的脚本run2.sh，直接用我们团队已经生成的模型推理，预测结果将存放在prediction_result/result2.json下。

# 运行流程
执行/data/code下的脚本run1.sh流程：

1. 首先执行/data/code/data_process/data_preprocess.py,进行数据预处理，取出复赛新增训练集里的3000条有标签数据，保存到/data/user_data/data_processed/dev2.json

2. 执行/data/code/data_process/merge_rule.py,合并rule1和rule2,合并的1000条rule保存到/data/user_data/data_processed/rules_all.json

3. 执行/data/code/data_process/get_topk_rule.py,为数据集中的每个问题检索知识库中的top10个相关规则,其中dev.json数据的结果保存在../../user_data/data_processed/dev_with_topk_rerank_rules.json，复赛新增训练参考集.json数据的结果保存至../../user_data/data_processed/dev2_with_topk_rerank_rules.json，复赛测试集test.json数据集的结果保存至../../user_data/data_processed/test_with_topk_rerank_rules.json

4. 执行/data/code/data_process/generate_rule_train_and_test_data.py，合并dev.json和复赛新增训练参考集.json及其检索到的相关规则，生成用于微调rule模型的训练数据，保存至/data/user_data/rule_data/train_data.json，将其转换为符合llama_factory微调的格式，保存至/data/code/LLaMA-Factory/data/rule_train.json。生成用微调后的rule模型的推理测试集的相关rule的数据集，保存至/data/user_data/rule_data/test_data.json,将其转换为符合llama_factory微调的格式，保存至/data/code/LLaMA-Factory/data/rule_inference.json。

5. 执行/data/code/data_process/generate_option_train_data.py,生成用于微调option模型的训练数据,保存至/data/user_data/option_data/train_data.json，将其转换为符合llama_factory微调的格式，保存至/data/code/LLaMA-Factory/data/option_train.json

6. 执行/data/code/LLaMA-Factory下面的train_rule.yaml脚本，开始微调rule 模型，相关的lora参数保存在/data/code/LLaMA-Factory/saves/qwen2.5-7B/lora/sft_rule_train，然后执行merge_rule.yaml脚本，合并微调rule模型的lora参数及qwen2.5 7B模型，结果保存至/data/user_data/qwen_sft_rule_model/，用于后续vllm推理

7. 执行/data/code/LLaMA-Factory下面的train_option.yaml脚本，开始微调option 模型，相关的lora参数保存在/data/code/LLaMA-Factory/saves/qwen2.5-7B/lora/sft_option_train

8. 执行/data/code/data_process/vllm_generate_rule.py，用vllm推理微调后的rule模型，预测测试集上每个问题相关的top10个rule list，结果保存在/data/user_data/rule_data/test_with_predict_rule.json

9. 执行/data/code/data_process/generate_option_test_data.py，用上一步生成的测试集上每个问题相关的top10个rule list，生成要用微调后的option模型推理测试集上的option的数据,保存至/data/user_data/option_data/test_data.json，将其转换为符合llama_factory微调的格式，保存至/data/code/LLaMA-Factory/data/option_inference.json

10. 执行/data/code/LLaMA-Factory下面的test_option.yaml脚本，用微调后的option 模型推理测试集上每个问题的option，结果保存在/data/code/LLaMA-Factory/saves/qwen2.5-7B/lora/sft_option_inference/generated_predictions.jsonl

11. 执行/data/code/data_process/process_option_prediction.py,对上一步推理生成的option后处理，结果保存在/data/user_data/option_data/test_with_predict_option.json

12. 执行/data/code/data_process/merge_rule_and_option.py，合并测试集上推理生成的rule list和option，生成最终结果，结果保存在/data/prediction_result/result.json


执行/data/code下的脚本run2.sh流程：

1. 首先执行/data/code/data_process/data_preprocess.py,进行数据预处理，取出复赛新增训练集里的3000条有标签数据，保存到/data/user_data/data_processed/dev2.json

2. 执行/data/code/data_process/merge_rule.py,合并rule1和rule2,合并的1000条rule保存到/data/user_data/data_processed/rules_all.json

3. 执行/data/code/data_process/get_topk_rule.py,为数据集中的每个问题检索知识库中的top10个相关规则,其中dev.json数据的结果保存在../../user_data/data_processed/dev_with_topk_rerank_rules.json，复赛新增训练参考集.json数据的结果保存至../../user_data/data_processed/dev2_with_topk_rerank_rules.json，复赛测试集test.json数据集的结果保存至../../user_data/data_processed/test_with_topk_rerank_rules.json

4. 执行/data/code/data_process/generate_rule_train_and_test_data.py，合并dev.json和复赛新增训练参考集.json及其检索到的相关规则，生成用于微调rule模型的训练数据，保存至/data/user_data/rule_data/train_data.json，将其转换为符合llama_factory微调的格式，保存至/data/code/LLaMA-Factory/data/rule_train.json。生成用微调后的rule模型的推理测试集的相关rule的数据集，保存至/data/user_data/rule_data/test_data.json,将其转换为符合llama_factory微调的格式，保存至/data/code/LLaMA-Factory/data/rule_inference.json。

5. 执行/data/code/data_process/generate_option_train_data.py,生成用于微调option模型的训练数据,保存至/data/user_data/option_data/train_data.json，将其转换为符合llama_factory微调的格式，保存至/data/code/LLaMA-Factory/data/option_train.json

6. 执行/data/code/LLaMA-Factory下面的执行merge_rule2.yaml脚本，合并我们团队微调rule模型的lora参数及qwen2.5 7B模型，结果保存至/data/user_data/rule_sft_lora_merge_qwen_model/，用于后续vllm推理

7. 执行/data/code/data_process/vllm_generate_rule2.py，用vllm推理我们团队已经微调好的rule型（/data/user_data/rule_sft_lora_merge_qwen_model），预测测试集上每个问题相关的top10个rule list，结果保存在/data/user_data/rule_data/test_with_predict_rule2.json

8. 执行/data/code/data_process/generate_option_test_data2.py，用上一步生成的测试集上每个问题相关的top10个rule list，生成要用我们团队微调好的option模型推理测试集上的option的数据,保存至/data/user_data/option_data/test_data2.json，将其转换为符合llama_factory微调的格式，保存至/data/code/LLaMA-Factory/data/option_inference2.json

9. 执行/data/code/LLaMA-Factory下面的test_option2.yaml脚本，用我们团队微调好的option 模型推理测试集上每个问题的option，结果保存在/data/code/LLaMA-Factory/saves/qwen2.5-7B/lora/sft_option_inference2/generated_predictions.jsonl

10. 执行/data/code/data_process/process_option_prediction2.py,对上一步推理生成的option后处理，结果保存在/data/user_data/option_data/test_with_predict_option2.json

10. 执行/data/code/data_process/merge_rule_and_option2.py，合并测试集上推理生成的rule list和option，生成最终结果，结果保存在/data/prediction_result/result2.json



# 环境依赖
- 操作系统：CentOS Linux 7 (Core)
- CUDA: 12.1
- python: 3.11
- torch: 2.4.0
- transformers: 4.44.2
- datasets:	2.19.0
- accelerate: 0.34.2
- peft: 0.12.0
- trl: 0.9.6
- faiss-cpu: 1.9.0
- FlagEmbedding: 1.3.2
- flash-attn: 2.6.3
- pandas>=2.0.0
- scipy
- einops
- sentencepiece
- tiktoken
- protobuf
- uvicorn
- pydantic
- fastapi
- sse-starlette
- matplotlib>=3.7.0
- fire
- packaging
- pyyaml
- numpy<2.0.0
- av


