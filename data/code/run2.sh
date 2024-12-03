!/bin/bash
echo "进入data_process"
cd data_process
pwd
echo "数据预处理，取出复赛新增训练集里的3000条有标签数据"
python data_preprocess.py

echo "合并rule1和rule2"
python merge_rule.py

echo "为数据集中的每个问题检索知识库中的top10个相关规则"
python get_topk_rule.py

echo "生成用于微调rule模型的训练数据，和用于推理微调后的rule模型的测试数据"
python generate_rule_train_and_test_data.py

echo "生成用于微调option模型的训练数据"
python generate_option_train_data.py

echo "进入LLaMA-Factory"
cd ..
cd LLaMA-Factory

echo "开始合并rule模型(我们训练好的lora)"
merge_generate_start_time=$(date +%s)
llamafactory-cli export ./merge_rule2.yaml

merge_generate_end_time=$(date +%s)
# 计算总时间
merge_generate_total_time=$(( merge_generate_end_time - merge_generate_start_time ))
echo "rule模型 merge开始: $merge_generate_start_time "
echo "rule模型 merge结束: $merge_generate_end_time "
echo "rule模型 merge总时间: $merge_generate_total_time 秒"
echo "-----------------------------------------------------------"
echo "-----------------------------------------------------------"
sleep 10



echo "用vllm推理微调后的rule模型，生成的测试集上的rule list(队伍已经训练好的模型)"
python vllm_generate_rule2.py

echo "生成推理微调后的option模型的测试数据(队伍已经训练好的模型)"
python generate_option_test_data2.py

echo "进入LLaMA-Factory"
cd ..
cd LLaMA-Factory


echo "开始批量推理测试集的option"
test_generate_start_time=$(date +%s)
llamafactory-cli train ./test_option2.yaml

test_generate_end_time=$(date +%s)
# 计算总时间
test_generate_total_time=$(( test_generate_end_time - test_generate_start_time ))
echo "测试集的option批量推理开始: $test_generate_start_time "
echo "测试集的option批量推理结束: $test_generate_end_time "
echo "测试集的option批量推理总时间: $test_generate_total_time 秒"
echo "-----------------------------------------------------------"
echo "-----------------------------------------------------------"
sleep 10

echo "进入data_process"
cd ..
cd data_process
echo "推理生成的option后处理"
python process_option_prediction2.py
echo " 合并推理生成的rule list和option，生成最终结果"
python merge_rule_and_option2.py

echo "利用我们团队训练好的rule模型和option模型，推理过程完毕"