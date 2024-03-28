import json

folders = ["all","easy","hard"]
file_names = ["train", "test", "dev"]

def def2name(names):
    defendant_names = ["<被告人A>", "<被告人B>", "<被告人C>", "<被告人D>", "<被告人E>", "<被告人F>"]
    return {a:b for a, b in zip(names, defendant_names)}

def anon(text, real_name2_fake_name):
    for rn, fn in real_name2_fake_name.items():
        text = text.replace(rn, fn)
    return text

for folder in folders:
    for fn in file_names:
        in_path = f"datasets/{folder}/{fn}.json"
        out_path = f"datasets_anonymized/{folder}/{fn}.json"
        with open(out_path, "w", encoding="utf-8") as fo:
            with open(in_path,"r", encoding="utf-8") as fi:
                for line in fi.readlines():
                    case = json.loads(line)
                    real_names = [label["subject"] for label in case["labels"]]
                    real_name2_fake_name = def2name(real_names) # 匿名
                    case["facts"] = anon(case["facts"], real_name2_fake_name) # fact
                    for label in case["labels"]:
                        label["subject"] = real_name2_fake_name[label["subject"]]
                        for so in label["sub+ob"]:
                            so["subjective"]=anon(so["subjective"], real_name2_fake_name)
                            so["objective"]['act']= anon(so["objective"]['act'], real_name2_fake_name) 
                            so["objective"]['res']= anon(so["objective"]['res'], real_name2_fake_name)
                    fo.write(json.dumps(case, ensure_ascii=False)+"\n")
