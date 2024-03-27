import json

cate2charge = {
               "社会公共安全":["非法制造枪支罪", "非法买卖枪支罪", "非法持有枪支罪"],
               "国家对知识产权的管理秩序":['假冒注册商标罪', "销售假冒注册商标的商品罪"],
               "市场秩序":["合同诈骗罪", "非法经营罪"],
               "公民的人身、民主权利":["故意杀人罪", "故意伤害罪", "非法拘禁罪"],
               "财产权利":["抢劫罪", "诈骗罪", '敲诈勒索罪'],
               "社会公共秩序":['招摇撞骗罪', "聚众斗殴罪", "寻衅滋事罪"],
               "国家司法秩序":["窝藏、包庇罪", "掩饰、隐瞒犯罪所得罪"],
               "社会道德风尚和社会治安管理秩序":["组织卖淫罪", "协助组织卖淫罪", "容留卖淫罪", "介绍卖淫罪"]
               }
charge2ob = {}
for key, values in cate2charge.items():
    for v in values:
        charge2ob[v] = key


folder_names = ["all","easy","hard"]
file_names = ["train", "test", "dev"]

for folder in folder_names:
    for file in file_names:
        in_path = f"datasets/{folder}/{file}.json"
        out_path = f"datasets/{folder}/{file}_.json"
        with open(out_path, "w", encoding="utf-8") as fo:
            with open(in_path,"r", encoding="utf-8") as fi:
                for line in fi.readlines():
                    case = json.loads(line)
                    for label in case["labels"]: # 每个被告人的label
                        for item in label["sub+ob"]: # 每个被告人
                            item["object"] = charge2ob[label["charge"]]
                    fo.write(json.dumps(case, ensure_ascii=False)+"\n")
