import os,json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import util
import copy
from preprocess import Processor
from model import ElemExtractor
import setting
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
from dataset import ClozeDataset, Lang
from torch.utils.data import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
transformers.logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def elem_metric(y_preds, y_trues, sent_lens):
    """
    Extraction Match at sent level and token level
    :param y_preds:
    :param y_trues:
    :param sp_lens:
    :return:
    """
    case_count = 0
    sent_score = 0.
    token_score = 0.
    for pred, truth, sent_len in zip(y_preds, y_trues, sent_lens): # 便利每个例子
        if truth == [0]*len(truth): # 标签全为0，即无效数据
            continue
        else:# 有效数据
            case_count += 1
            if pred == truth:
                sent_score += 1.0
                token_score += 1.0
                continue
            p_set = set([idx for idx, val in enumerate(pred) if val == 1])
            t_set = set([idx for idx, val in enumerate(truth) if val == 1])
            matched_sent_idx = list(p_set.intersection(t_set))
            if len(matched_sent_idx) == 0:
                continue
            # sentence level
            sent_score += 2*len(matched_sent_idx)/(len(t_set)+len(p_set)+0.0001)
            # token leval
            matched_token_count = 0
            totall_token_count = 0
            for sent_idx in t_set:
                totall_token_count += sent_len[sent_idx]
                if sent_idx in matched_sent_idx:
                    matched_token_count += sent_len[sent_idx]
            for sent_idx in p_set:
                if sent_idx >= len(sent_len):
                    continue
                totall_token_count += sent_len[sent_idx]
                if sent_idx in matched_sent_idx:
                    matched_token_count += sent_len[sent_idx]
            token_score += matched_token_count/(totall_token_count+0.0001)
    print(f"Elem: sent_match: {round(sent_score/case_count, 2)}   'token_match: {round(token_score/case_count,2)}")

def eval(model, data_loader):
    model.eval()
    with torch.no_grad():
        elem_trues = []
        elem_preds = []
        sent_lens = []
        for ids, inputs, enc_inputs, dfds, grouped_dfds, charge_idxs, grouped_charge_idxs, cate_idxs, grouped_cate_idxs, sp_lens, pad_sp_lens, relevant_sents, mask_positions, dfd_positions in data_loader:
            sent_score = model(enc_inputs,pad_sp_lens)
            max_len = max([len(s) for s in pad_sp_lens])
            elem_pred = sent_score.argmax(dim=1).cpu().tolist()
            elem_label = util.label_construct(relevant_sents, pad_sp_lens)
            elem_trues.extend([ts.tolist() for ts in torch.split(torch.tensor(elem_label), [max_len] * int(len(elem_label) / max_len))])
            elem_preds.extend([ts.tolist() for ts in torch.split(torch.tensor(elem_pred), [max_len] * int(len(elem_label) / max_len))])
            sent_lens.extend(sp_lens)
        elem_metric(y_preds=elem_preds, y_trues=elem_trues, sent_lens=sent_lens)

def train(seed, enc, data_path, params, index2charge, charge_desc, cate2charge):
    util.set_seed(seed)
    print("preparing dataset...")
    train_path, dev_path, test_path = f"./datasets/{data_path}/train.json",f"./datasets/{data_path}/dev.json",f"./datasets/{data_path}/test.json"
    lang = Lang(index2charge, charge_desc, cate2charge)
    processor = Processor(lang)
    train_data, dev_data, test_data = processor.load_data(train_path), processor.load_data(dev_path), processor.load_data(test_path)
    dataset_train = ClozeDataset(train_data, enc, lang,)
    dataset_dev = ClozeDataset(dev_data, enc, lang)
    dataset_test = ClozeDataset(test_data, enc, lang)
    train_data_loader = DataLoader(dataset_train, batch_size=params['batch_size'], collate_fn=dataset_train.collate_fn, shuffle=True)
    dev_data_loader = DataLoader(dataset_dev, batch_size=params['batch_size'], collate_fn=dataset_dev.collate_fn, shuffle=False)
    test_data_loader = DataLoader(dataset_test, batch_size=params['batch_size'], collate_fn=dataset_test.collate_fn, shuffle=False)
    model = ElemExtractor(enc, lang, device)
    model.to(device)
    # 定义损失函数，优化器，学习率调整器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(params=model.parameters(), lr=params['lr'])
    warmup_step = int(0.15*(len(train_data)/params["batch_size"])*params['epoch']+1)
    training_step = int((len(train_data)/params["batch_size"])*params['epoch']+1)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                   num_warmup_steps=warmup_step,
                                                                   num_training_steps=training_step,
                                                                   num_cycles=1)
    print("training model...")
    from tqdm import tqdm
    for epoch in range(params['epoch']):
        print(f"-------------------------------epoch:{epoch + 1}-------------------------------------")
        model.train()
        train_loss = 0
        for ids, inputs, enc_inputs, dfds, grouped_dfds, charge_idxs, grouped_charge_idxs, cate_idxs, grouped_cate_idxs, sp_lens, pad_sp_lens, relevant_sents, mask_positions, dfd_positions in train_data_loader:            # 梯度置零
            optimizer.zero_grad()
            sent_score = model(enc_inputs, pad_sp_lens)
            sent_label = util.label_construct(relevant_sents, pad_sp_lens)
            # group_loss = criterion(group_score, torch.tensor(cate_idxs).to(device))
            loss = criterion(sent_score, torch.tensor(sent_label).to(device))
            # loss = group_loss+sent_loss
            train_loss += loss.item()
            # 累计梯度
            loss.backward()
            # 梯度
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 更新梯度
            optimizer.step()
            # 更新学习率
            scheduler.step()
        print(f"train_loss:{round(train_loss/len(train_data_loader.dataset), 4)}")
        if epoch>4:
            torch.save(model, f"./outputs/models/fact_selector_{data_path}_{epoch}.pkl")
        print("Dev:")
        eval(model, dev_data_loader)
        print("Test:")
        eval(model, test_data_loader)

def main():
    params = setting.params
    enc = setting.enc
    index2charge = setting.index2charge
    charge_desc = setting.charge_desc
    cate2charge = setting.cate2charge
    print("Running ...")
    for seed in params["seeds"][:1]:
        print(f"set seed {seed}")
        for data_path in params["data_path"]:
            print(f"seed: {seed}\n"
                  f"data: {data_path}\n"
                  f"batch_size: {params['batch_size']}\n"
                  f"lr: {params['lr']}\n")
            if data_path == "hard":
                train(seed, enc, data_path, params, index2charge[:-1], charge_desc, copy.deepcopy(cate2charge))
            else:
                train(seed, enc, data_path, params, index2charge, charge_desc, copy.deepcopy(cate2charge))

if __name__=="__main__":
    main()