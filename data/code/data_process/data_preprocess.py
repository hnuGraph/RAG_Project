import json
import re

with open("../../raw_data/复赛新增训练参考集.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

new_data=[]
for item in data:
    if "refered_rules" not in item:
        continue
    if item["refered_rules"][0]=="977. ":
        continue
    rule_id = [re.match(r"(\d+)\.", rule).group(1) for rule in item["refered_rules"]]
    del item["refered_rules"]
    item["rule_id"]=rule_id
    new_data.append(item)
    


with open("../../user_data/data_processed/dev2.json",'w',encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
