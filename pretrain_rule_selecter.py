import json
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
import setting
from dataset import Lang
from model import SimcseModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

EPOCHS = 20
MAXLEN = 256
LR = 1e-5
BATCH_SIZE = 11
POOLING = 'pooler'   # choose in ['cls', 'pooler', 'last-avg', 'first-last-avg']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RoBERTa = './pretrained_files/roberta_wwm'
SAVE_PATH = 'pretrained_files/roberta_wwm_cse(0).pkl'
TRAIN_DATA = "./datasets/LNLI/nli_train.txt"
DEV_DATA = "./datasets/LNLI/nli_dev.txt"

def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as fi:
        for line in fi:
            data.append(json.loads(line))
    return data

class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text):
        return tokenizer(text, max_length=MAXLEN,
                         truncation=True, padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        # return self.data[index]
        return self.text_2_id(self.data[index])

class DevDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法
    """
    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        data = []
        label = []
        for b in batch:
            data.append(b[0])
            label.append(b[1])
        data = tokenizer(data, max_length=MAXLEN,truncation=True, padding='max_length', return_tensors='pt')
        return data, label

    def __getitem__(self, index: int):
        return self.data[index]


def simcse_sup_loss(y_pred: 'tensor') -> 'tensor':
    """有监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]
    """
    # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 3, 2, ...]
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    use_row = torch.where(y_true % 6 < 2)[0]
    y_true = (use_row - use_row % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 选取有效的行
    sim = torch.index_select(sim, 0, use_row)
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)

def eval(model, dataloader) -> float:
    """模型评估函数
    批量预测, 计算cos_sim, 转成numpy数组拼接起来, 一次性求spearman相关度
    """
    model.eval()
    y_preds = []
    y_trues = []
    with torch.no_grad():
        for source, label in dataloader:
            # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
            # real_batch_num = source.get('input_ids').shape[0]
            input_ids = source.get('input_ids').to(DEVICE)
            attention_mask = source.get('attention_mask').to(DEVICE)
            token_type_ids = source.get('token_type_ids').to(DEVICE)
            out = model(input_ids, attention_mask, token_type_ids)
            enc_desc = tokenizer(lang.chaIdx2desc, max_length=MAXLEN,truncation=True, padding='max_length', return_tensors='pt')
            input_ids = enc_desc.get('input_ids').to(DEVICE)
            attention_mask = enc_desc.get('attention_mask').to(DEVICE)
            token_type_ids = enc_desc.get('token_type_ids').to(DEVICE)
            desc = model(input_ids, attention_mask, token_type_ids)
            sim = F.cosine_similarity(out.unsqueeze(1), desc.unsqueeze(0), dim=-1)
            # sim = sim - torch.eye(out.shape[0], device=DEVICE) * 1e12
            y_pred = sim.argmax(dim=1).cpu().tolist()
            y_preds.extend(y_pred)
            y_trues.extend(label)
    acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
    precision = precision_score(y_true=y_trues, y_pred=y_preds, average="macro")
    recall = recall_score(y_true=y_trues, y_pred=y_preds, average="macro")
    f1 = f1_score(y_true=y_trues, y_pred=y_preds, average="macro")
    print(f"acc:{round(acc, 2)}  precision:{round(precision, 2)}  recall:{round(recall, 2)}  f1:{round(f1, 2)}")
    return sum([acc, precision, recall, f1])

def train(model, train_dl, dev_dl, optimizer) -> None:
    """模型训练函数
    """
    model.train()
    global best
    early_stop_batch = 0
    for batch_idx, source in enumerate(train_dl, start=1):
        # 维度转换 [batch, 6, seq_len] -> [batch * 6, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 6, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 6, -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 6, -1).to(DEVICE)
        # 训练
        out = model(input_ids, attention_mask, token_type_ids)
        loss = simcse_sup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 评估
        if batch_idx % 1 == 0:
            score = eval(model, dev_dl)
            model.train()
            if score>=best:
                torch.save(model, SAVE_PATH)
                best = score


if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained(RoBERTa)
    lang = Lang(setting.index2charge, setting.charge_desc, setting.cate2charge)
    # load data
    train_data = TrainDataset(load_data(TRAIN_DATA))
    dev_data = DevDataset(load_data(DEV_DATA))
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
    dev_dataloader = DataLoader(dev_data, batch_size=BATCH_SIZE, collate_fn=dev_data.collate_fn, shuffle=True)
    model = SimcseModel(pretrained_model=RoBERTa, pooling=POOLING)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    best = 0
    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch}")
        train(model, train_dataloader, dev_dataloader,  optimizer)