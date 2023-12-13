import re
import json
import copy
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class Lang:
    def __init__(self, data_path, charge_desc, cate2charge):
        self.cate2charge = cate2charge
        self.charge2cate = {}
        for cate, charges in self.cate2charge.items():
            for c in charges:
                self.charge2cate[c] = cate
        self.index2cate = list(self.cate2charge.keys())
        self.cate2index = {cate: idx for idx, cate in enumerate(self.index2cate)}

        self.index2charge = []
        self.charge2index = None
        self.stat(data_path)

        # # 去除数据集中不存在得charge
        # diff_charge = list(set(charge_desc.keys()).difference(set(self.index2charge)))
        # for d in diff_charge:
        #     charge_desc.pop(d)
        self.charge2desc = charge_desc

    def stat(self, data_path):
        # 统计标签
        with open(data_path, "r", encoding="utf-8") as fi:
            for line in fi:
                case = json.loads(line)
                charges = list(set([label["charge"] for label in case["labels"]]))
                for c in charges:
                    if c not in self.index2charge:
                        self.index2charge.append(c)
        self.charge2index = {charge: idx for idx, charge in enumerate(self.index2charge)}

class ClozeDataset(Dataset):
    def __init__(self, data, model_path, lang, charge_desc):
        self.lang = lang
        self.data = data
        self.charge_desc = charge_desc
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pt = re.compile(r"\[MASK\]")
        self.sp_token = ","
        self.sp_pt = re.compile(r",|\[SEP\]")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch): # for relevant sent extraction
        # [id, fact, dfds, grouped_dfds, charge_idx, sp_lens, relevant_sents]
        ids = [] # 数据id
        inputs = [] # fact
        dfds = [] # 被告人
        grouped_dfds = []  # 每个case涉及的被告人
        charge_idxs = []  # 指控索引
        grouped_charge_idxs = []  # 每个case涉及的指控索引
        grouped_cate_idxs = []
        pad_sp_lens = [] # 分割
        relevant_sents = []# 相关句子[{"subjective":[], "act":[], "res":[]}]
        mask_positions = [] # 掩码位置
        dfd_positions = []
        for sample in batch: # 为每个被告人构造一个sample
            # 被告人
            grouped_dfds.append([l["subject"] for l in sample["labels"]])
            grouped_charge_idxs.append([self.lang.charge2index[l["charge"]] for l in sample["labels"]])
            for label in sample["labels"]:
                # id
                ids.append(sample['id'])
                # fact
                ipt = [f"{label['subject']}构成[MASK]罪,"+self.sp_token.join(sample["facts"])]
                inputs.extend(ipt)
                # mask positions
                mask_positions.append(1+len(label["subject"])+2)
                # 被告人
                dfds.append(label["subject"])
                # 指控索引
                charge_idxs.append(self.lang.charge2index[label["charge"]])
                # 分割
                sp = self.tokenizer.decode(self.tokenizer(ipt)["input_ids"][0])
                sp = [s.split(" ") for s in self.sp_pt.split(sp)][:-1]
                for idx, s in enumerate(sp):
                    sp[idx] = [i for i in s if i != ""]
                sp_len = [len(s)+1 for s in sp]
                pad_sp_lens.append(sp_len)
                # 相关句子
                labeled_sent = {"subjective":[], "act":[], "res":[]}
                for so in label['sub+ob']:
                    labeled_sent["subjective"].extend([i+1 for i in so["subjective"]])
                    labeled_sent["act"].extend([i+1 for i in so["objective"]["act"]])
                    labeled_sent["res"].extend([i+1 for i in so["objective"]["res"]])
                relevant_sents.append(labeled_sent)
                # dfd positions
                dfd_positions.append(self.get_dfd_positions(label["subject"], sample["facts"]))
        enc_inputs = self.tokenizer(inputs, truncation=True, max_length=512, return_tensors="pt", padding=True)
        enc_desc = self.get_charge_desc()
        sent_lens = copy.deepcopy(pad_sp_lens)
        # 根据padding后的长度修改sp_len
        pad_len = enc_inputs['input_ids'].shape[1]
        for sp_len in pad_sp_lens:
            if sum(sp_len)<pad_len:
                sp_len[-1]+=pad_len-sum(sp_len)
        cate_idxs = [self.lang.cate2index[self.lang.charge2cate[self.lang.index2charge[c]]] for c in charge_idxs]
        for idxs in grouped_charge_idxs:
            grouped_cate_idxs.append([self.lang.cate2index[self.lang.charge2cate[self.lang.index2charge[c]]] for c in idxs])
        return ids, inputs, enc_inputs, enc_desc, dfds, grouped_dfds, charge_idxs, grouped_charge_idxs, cate_idxs, grouped_cate_idxs, sent_lens, pad_sp_lens, relevant_sents, mask_positions, dfd_positions

    def get_dfd_rel(self, grouped_relevant_sents):
        dfds_rel = []
        for rel_sents in grouped_relevant_sents:
            sent_ids = []
            for dic in rel_sents:# 某被告人的相关句子
                temp = []
                for k, v in dic.items():
                    temp.extend(v)
                sent_ids.append(temp)
            rels = []
            for i in range(len(sent_ids)-1):
                for j in range(i+1,len(sent_ids)):
                    if 2*len(set(sent_ids[i]).intersection(set(sent_ids[j])))/(len(set(sent_ids[i]))+len(set(sent_ids[j])))>=0.6:
                        rels.append((i,j))
            dfds_rel.append(rels)
        return dfds_rel

    def get_dfd_positions(self, dfd, fact):
        positions = []
        for idx, f in enumerate(fact):
            if dfd in f:
                positions.append(idx+1)
        return positions

    def get_charge_desc(self):
        descs = [self.charge_desc[c] for c in self.lang.index2charge]
        enc_descs = self.tokenizer(descs, truncation=True, max_length=512, return_tensors="pt", padding=True, add_special_tokens=False)
        return enc_descs
