import re,time
import json
import copy
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class Lang:
    def __init__(self, index2charge, charge_desc, cate2charge):
        self.index2charge = index2charge
        self.charge2index = {charge: idx for idx, charge in enumerate(self.index2charge)}

        # self.charge2desc = charge_desc
        self.cate2charge = cate2charge
        self.charge2cate = {}
        keys = []
        vals = []
        for cate, charges in self.cate2charge.items():
            for c in charges:
                if c not in self.index2charge:
                    keys.append(cate)
                    vals.append(c)
        for k,v in zip(keys, vals):
            # self.charge2desc.pop(v)
            self.cate2charge[k].remove(v)
        for cate, charges in self.cate2charge.items():
            for c in charges:
                self.charge2cate[c] = cate

        self.chaIdx2desc = [charge_desc[charge] for charge in self.index2charge]
        self.index2cate = list(self.cate2charge.keys())
        self.cate2index = {cate: idx for idx, cate in enumerate(self.index2cate)}

    def stat(self, data_path):
        # 统计标签
        with open(data_path, "r", encoding="utf-8") as fi:
            for line in fi:
                case = json.loads(line)
                charges = list(set([label["charge"] for label in case["labels"]]))
                for c in charges:
                    if c not in self.index2charge:
                        self.index2charge.append(c)


class ClozeDataset(Dataset):
    def __init__(self, data, model_path, lang):
        self.lang = lang
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sp_token = ","
        self.sp_token_id = self.tokenizer.get_vocab()[self.sp_token]
        # self.sp_pt = re.compile(r",|\[SEP\]")

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
        cate_idxs = []
        grouped_cate_idxs = []
        sp_lens = []# 分割长度
        relevant_sents = []# 相关句子
        mask_positions = [] # 掩码位置
        dfd_positions = []
        for sample in batch: # 为每个被告人构造一个sample
            # 被告人
            charge_names = [label["charge"] for label in sample["labels"]]
            grouped_dfds.append([label["subject"] for label in sample["labels"]])
            grouped_charge_idxs.append([self.lang.charge2index[c] for c in charge_names])
            grouped_cate_idxs.append([self.lang.cate2index[self.lang.charge2cate[c]] for c in charge_names])
            for label in sample["labels"]:
                # id
                ids.append(sample['id'])
                # fact
                f = f"{label['subject']}构成[MASK]罪,"+self.sp_token.join(sample["facts"])
                inputs.append(f)
                # mask positions
                mask_positions.append(len(label["subject"])+2)
                # 被告人
                dfds.append(label["subject"])
                # 指控索引
                charge_idxs.append(self.lang.charge2index[label["charge"]])
                # 指控类别
                cate_idxs.append(self.lang.cate2index[self.lang.charge2cate[label["charge"]]])
                # 分割长度
                enc_ipt = self.tokenizer(inputs[-1], truncation=True, max_length=512, add_special_tokens=False)["input_ids"]
                sp_len,pre = [], -1
                for idx, val in enumerate(enc_ipt):
                    if val == self.sp_token_id or idx == len(enc_ipt)-1:
                        sp_len.append(idx-pre)
                        pre = idx
                sp_lens.append(sp_len)
                # 相关句子
                labeled_sent = []
                for so in label['sub+ob']:
                    labeled_sent.extend([i+1 for i in so["subjective"]])
                    labeled_sent.extend([i+1 for i in so["objective"]["act"]])
                    labeled_sent.extend([i+1 for i in so["objective"]["res"]])
                labeled_sent = list(sorted(set(labeled_sent)))
                relevant_sents.append(labeled_sent)
                # dfd positions
                dfd_positions.append(self.get_dfd_positions(label["subject"], sample["facts"]))
        enc_inputs = self.tokenizer(inputs, truncation=True, max_length=512, return_tensors="pt", padding=True)
        # enc_desc = self.get_charge_desc()
        pad_sp_lens = copy.deepcopy(sp_lens)
        # 根据padding后的长度修改sp_len
        for l in pad_sp_lens:
            if sum(l)<enc_inputs['input_ids'].shape[1]:
                l[-1]+=enc_inputs['input_ids'].shape[1]-sum(l)

        return ids, inputs, enc_inputs, dfds, grouped_dfds, charge_idxs, grouped_charge_idxs, cate_idxs, grouped_cate_idxs, sp_lens, pad_sp_lens, relevant_sents, mask_positions, dfd_positions

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
        if len(positions)==0:
            print(fact)
        return positions

    def get_charge_desc(self):
        enc_descs = self.tokenizer(self.lang.chaIdx2desc, truncation=True, max_length=512, return_tensors="pt", padding=True, add_special_tokens=False)
        return enc_descs
