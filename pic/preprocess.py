import re, os
import json
import random
from dataset import Lang

class Processor():
    def __init__(self, lang):
        self.lang = lang
        self.sp_pattern_1 = re.compile(r"///")
        self.sp_pattern_2 = re.compile(r",|，|。|；|;|：|:|///")
        self.sp_pattern_3 = re.compile(r"[。，]另查")
        mask_list = [s[:-1] for s in self.lang.index2charge]
        mask_list.extend(["被告人", "被害人"])
        s = "|".join(mask_list)
        self.mask_pt = re.compile(s)

    def load_data(self, data_path):
        data = []
        with open(data_path, "r", encoding="utf-8") as fi:
            for line in fi:
                sample = json.loads(line)
                # 替换提示字符
                sample["facts"] = [sent for sent in self.sp_pattern_2.split(self.mask_pt.sub("", sample["facts"])) if sent != ""]
                # 将标注转化为句子索引
                for label in sample["labels"]:
                    for item in label["sub+ob"]:
                        item["objective"]["act"] = self.__get_sent_id(sample["facts"], item["objective"]["act"])
                        item["objective"]["res"] = self.__get_sent_id(sample["facts"], item["objective"]["res"])
                        item["subjective"] = self.__get_sent_id(sample["facts"], item["subjective"])

                data.append(sample)
        return data

    def __get_sent_id(self,facts, annos):
        annos = [self.mask_pt.sub("", anno) for anno in self.sp_pattern_2.split(annos) if anno != ""]
        anno_ids = []
        for anno in annos:
            if anno == "无":
                continue
            for idx, sent in enumerate(facts):
                if anno in sent and idx not in anno_ids:
                    anno_ids.append(idx)
                    break
        return anno_ids


