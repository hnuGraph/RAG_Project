import json

with open('../../raw_data/rules1.json', 'r', encoding='utf-8') as f:
    data1 = json.load(f)

with open('../../raw_data/rules2.json', 'r', encoding='utf-8') as f:
    data2 = json.load(f)

new_data=[]

for item in data1:
    new_data.append(item)

for item in data2:
    if item["rule_id"]=="977":
        continue
    new_data.append(item)

with open("../../user_data/data_processed/rules_all.json",'w',encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
