import re
import util
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class Base(nn.Module):
    def __init__(self, model_path, lang, device):
        super(Base, self).__init__()
        self.device = device
        self.lang = lang
        self.enc = AutoModel.from_pretrained(model_path)
        self.pred = nn.Linear(768, len(lang.index2charge)),


    def forward(self, enc_fact, mask_positions):
        enc_fact = {k: v.to(self.device) for k, v in enc_fact.items()}
        enc_fact = self.enc(**enc_fact)['last_hidden_state']
        ts = []
        for idx in range(len(mask_positions)):
            reps = torch.index_select(enc_fact[idx].squeeze(), 0, torch.tensor(mask_positions[idx]).to(self.device))
            ts.append(reps)
        ts = torch.concat(ts, dim=0)
        outputs = self.pred(ts)
        return outputs

class ElemExtractor(nn.Module):
    def __init__(self, model_path, lang, device):
        super(ElemExtractor, self).__init__()
        self.device = device
        self.lang = lang
        self.enc = AutoModel.from_pretrained(model_path)
        self.sent_pred = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, enc_inputs, pad_sp_lens):
        enc_input = {k: v.to(self.device) for k, v in enc_inputs.items()}
        outputs = self.enc(**enc_input)['last_hidden_state']
        # mask_em = self.select_mask_tensor(outputs, mask_positions)
        # group_score = self.group_pred(mask_em)
        # 得到句子的嵌入表示 [batch_size, sent_count, dim]
        sent_em = self.split_tensor(outputs, pad_sp_lens)
        # 预测每个句子的重要性 [batch_size, sent_count, 2]
        sent_score = self.sent_pred(sent_em)
        # 根据pred_score选择的句子预测其对应的指控 [batch_size, dim]
        return sent_score.view(sent_score.shape[0]*sent_score.shape[1], -1)

    def split_tensor(self, input, pad_sp_lens):
        max_len = max([len(s) for s in pad_sp_lens])
        splited_tensor = []
        for idx, sp_len in enumerate(pad_sp_lens):
            # [sent_count, dim]
            sample = torch.stack([torch.mean(i, dim=0) for i in torch.split(input[idx], sp_len)], 0)
            splited_tensor.append(F.pad(sample, (0, 0, 0, max_len-sample.shape[0]))) #left,right,top,bottom
        # [batch_size,sent_count, dim]
        return torch.stack(splited_tensor, dim=0)

    def select_mask_tensor(self, outputs, mask_positions):
        mask_tensors = []  # [MASK] tensor
        for idx in range(outputs.shape[0]):
            # 添加mask对应的tensor
            mask_tensors.append(torch.unsqueeze(outputs[idx][mask_positions[idx]], dim=0))
        return torch.concat(mask_tensors, dim=0)

    def sent_select(self, pred_score):
        """
        :param pred_score: [batch_size, sent_count, 2] 句子重要性预测分数
        :return:
        """
        # [batch_size, snent_cout]
        pred = pred_score.argmax(dim=2).cpu().tolist()
        # [batch_size, dim]
        selected_sents = []
        for b_idx, sent_idxs in enumerate(pred):
            indices = torch.LongTensor([i for i, v in enumerate(sent_idxs) if v==1]).to(self.device)
            selected_sents.append(torch.sum(torch.index_select(self.sent_em[b_idx], 0, indices), dim=0))
        # [batch_size, dim]
        return torch.stack(selected_sents, dim=0)

class GroupPred(nn.Module):
    def __init__(self, enc_path, lang, device):
        super(GroupPred, self).__init__()
        self.device = device
        self.lang = lang
        self.enc_path = enc_path
        self.GroupEnc = AutoModel.from_pretrained(enc_path)# predict
        self.GroupPred = nn.Sequential(
            nn.Linear(2*768, 768),
            nn.ReLU(),
            nn.Linear(768, len(lang.index2cate))
        )

    def forward(self, enc_inputs, mask_positions, pad_sp_lens, relevant_sents, dfd_positions):
        enc_inputs = {k: v.to(self.device) for k, v in enc_inputs.items()}
        outputs = self.GroupEnc(**enc_inputs)['last_hidden_state']
        mask_tensors = self.select_mask_tensor(outputs, mask_positions)
        elem_tensors = self.select_rel_sent_tensors(outputs, pad_sp_lens, relevant_sents, dfd_positions)
        g_scores = self.GroupPred(torch.concat([mask_tensors, elem_tensors], dim=1))
        return g_scores

    def select_mask_tensor(self, outputs, mask_positions):
        mask_tensors = []  # [MASK] tensor
        for idx in range(outputs.shape[0]):
            # 添加mask对应的tensor
            mask_tensors.append(torch.unsqueeze(outputs[idx][mask_positions[idx]], dim=0))
        return torch.concat(mask_tensors, dim=0)

    def select_rel_sent_tensors(self, outputs, pad_sp_lens, relevant_sents, dfd_positions):
        splited_tensors = self.split_tensor(outputs, pad_sp_lens)
        relevant_tensors = []
        for idx in range(len(relevant_sents)):
            indices = relevant_sents[idx]
            indices.append(dfd_positions[idx][0])
            indices = list(set(indices))
            tensors = torch.index_select(splited_tensors[idx], 0, torch.tensor(indices).to(self.device))
            relevant_tensors.append(tensors.mean(dim=0, keepdim=True))
        return torch.concat(relevant_tensors, dim=0)

    def split_tensor(self, input, pad_sp_lens):
        max_len = max([len(s) for s in pad_sp_lens])
        splited_tensor = []
        for idx, sp_len in enumerate(pad_sp_lens):
            # [sent_count, dim]
            sample = torch.stack([torch.mean(i, dim=0) for i in torch.split(input[idx], sp_len)], 0)
            splited_tensor.append(F.pad(sample, (0, 0, 0, max_len-sample.shape[0]))) #left,right,top,bottom
        # [batch_size,sent_count, dim]
        return torch.stack(splited_tensor, dim=0)

class SimcseModel(nn.Module):
    """Simcse有监督模型定义"""
    def __init__(self, pretrained_model: str, pooling: str):
        super(SimcseModel, self).__init__()
        # config = BertConfig.from_pretrained(pretrained_model)   # 有监督不需要修改dropout
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):

        # out = self.bert(input_ids, attention_mask, token_type_ids)
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]

        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]

        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]

        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)

class Reasoner(nn.Module):
    def __init__(self, module1, module2, enc_path, lang, device, topk=1, alpha=1.0, beta=1.0):
        super(Reasoner, self).__init__()
        self.device = device
        self.lang = lang
        self.enc_path = enc_path
        self.tokenizer = AutoTokenizer.from_pretrained(enc_path)
        self.module1 = module1# sent_level match score
        self.module1.eval()
        self.module2 = module2# predict
        self.module2.eval()
        self.topk = topk
        self.alpha = alpha # 控制charge prediction的分布
        self.beta = beta # 控制group prediction 的分布
        # self.threhold = threhold

    def forward(self, inputs, mask_positions, pad_sp_lens, relevant_sents, dfd_positions):
        enc_inputs = self.tokenizer(inputs, truncation=True, max_length=512, return_tensors="pt", padding=True)
        # enc_inputs = {k: v.to(self.device) for k, v in enc_inputs.items()}
        with torch.no_grad(): # get group scores
            g_scores = self.module1(enc_inputs, mask_positions, pad_sp_lens, relevant_sents, dfd_positions)
        tg_ids, tg_scores, gc_ids, charge_desc, rel_sents = self.get_match_pairs(g_scores, inputs, relevant_sents)
        pred_charges = self.reason(tg_ids, tg_scores, gc_ids, charge_desc, rel_sents)
        return pred_charges

    def get_match_pairs(self, g_scores, inputs, relevant_sents):
        g_scores = g_scores*self.beta # scale dist of group probe
        g_scores = torch.softmax(g_scores, dim=1)
        # g_ids = g_scores.argmax(dim=1).cpu().tolist()
        tg_scores, tg_ids = torch.topk(g_scores, self.topk)
        # 每个组内的指控id
        gc_ids = []
        for tg in tg_ids.tolist():
            g_charges = [self.lang.cate2charge[self.lang.index2cate[g]] for g in tg]
            temp = []
            for charge in g_charges:
                temp.append([self.lang.charge2index[c] for c in charge])
            gc_ids.append(temp)
        # c_ids对应的指控描述
        charge_desc = []
        for c_ids in gc_ids:
            temp = []
            for ids in c_ids:
                temp.append([self.lang.chaIdx2desc[id] for id in ids])
            charge_desc.append(temp)

        rel_sents = []
        for facts, ids in zip(inputs, relevant_sents):
            facts = facts.split(",")
            ids = sorted(list(set(ids)))
            rel_sents.append(",".join([facts[i] for i in ids]))
        # [batch_size, 2], [batch_size, 2], [batch_size, 2, c_num], [batch_size, 2, c_num], [batch_size]
        return tg_ids, tg_scores, gc_ids, charge_desc, rel_sents

    def reason(self, tg_ids, tg_scores, gc_ids, charge_desc, rel_sents):
        """
        :param tg_ids: [batch_size, 2]
        :param tg_scores: [batch_size, 2]
        :param gc_ids: [batch_size, 2, c_num]
        :param charge_desc: [batch_size, 2, c_num]
        :param rel_sents: [batch_size]
        :return:
        """
        logs = []
        pred_charges = []
        # 计算每个被告人的预测结果
        for g_ids, g_scores, c_ids, desc, sent in zip(tg_ids, tg_scores, gc_ids, charge_desc, rel_sents):
            """
            g_ids: [2]
            g_scores: [2]
            c_ids: [2, c_num]
            desc: [2, c_num]
            sent: [1]
            """

            desc_sp_lens = [len(d) for d in desc]
            descs = []
            for d in desc:
                descs.extend(d)
            enc_desc = self.tokenizer(descs, max_length=256, truncation=True, padding='max_length', return_tensors='pt')
            input_ids = enc_desc.get('input_ids').to(self.device)
            attention_mask = enc_desc.get('attention_mask').to(self.device)
            token_type_ids = enc_desc.get('token_type_ids').to(self.device)
            with torch.no_grad():
                desc_out = self.module2(input_ids, attention_mask, token_type_ids)
            enc_sent = self.tokenizer([sent], max_length=256, truncation=True, padding='max_length', return_tensors='pt')
            input_ids = enc_sent.get('input_ids').to(self.device)
            attention_mask = enc_sent.get('attention_mask').to(self.device)
            token_type_ids = enc_sent.get('token_type_ids').to(self.device)
            with torch.no_grad():
                sent_out = self.module2(input_ids, attention_mask, token_type_ids)
            sim = F.cosine_similarity(sent_out.unsqueeze(1), desc_out.unsqueeze(0), dim=-1)
            # 温度参数
            sim = torch.softmax(sim/self.alpha, dim=1)
            sims = torch.split(sim, desc_sp_lens, dim=1)
            verified_socre = []
            for g_prob, charges_prob in zip(g_scores, sims):
                # if g_prob.item() < self.threhold:
                #     g_prob = 0
                verified_socre.append(g_prob*charges_prob)
            verified_socre = torch.concat(verified_socre, dim=1)
            verified_ids = []
            for ids in c_ids:
                verified_ids.extend(ids)
            pred_charges.append(verified_ids[verified_socre.argmax(dim=1).item()])
            logs.append([g_ids.cpu().tolist(), g_scores.cpu().tolist(), c_ids, [s.cpu().tolist() for s in sims]])
        return pred_charges, logs

