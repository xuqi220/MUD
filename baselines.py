import os,json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import util
import copy
from preprocess import Processor
from model import Reasoner, GroupPred
import setting
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
from dataset import ClozeDataset, Lang
from torch.utils.data import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
transformers.logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(seed,enc_path, data_path, params, index2charge, charge_desc, cate2charge):
    util.set_seed(seed)
    print("preparing dataset...")
    train_path, dev_path, test_path = f"./datasets/{data_path}/train.json",f"./datasets/{data_path}/dev.json",f"./datasets/{data_path}/test.json"
    lang = Lang(index2charge, charge_desc, cate2charge)
    processor = Processor(lang)
    train_data, dev_data, test_data = processor.load_data(train_path), processor.load_data(dev_path), processor.load_data(test_path)
    dataset_train = ClozeDataset(train_data, enc_path, lang,)
    dataset_dev = ClozeDataset(dev_data, enc_path, lang)
    dataset_test = ClozeDataset(test_data, enc_path, lang)
    train_data_loader = DataLoader(dataset_train, batch_size=params['batch_size'], collate_fn=dataset_train.collate_fn, shuffle=True)
    dev_data_loader = DataLoader(dataset_dev, batch_size=params['batch_size'], collate_fn=dataset_dev.collate_fn, shuffle=False)
    test_data_loader = DataLoader(dataset_test, batch_size=params['batch_size'], collate_fn=dataset_test.collate_fn, shuffle=False)
    model = GroupPred(enc_path, lang, device)
    model.to(device)
    # 定义损失函数，优化器，学习率调整器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(params=model.parameters(), lr=params['lr'])
    warmup_step = int(0.1*(len(train_data)/params["batch_size"])*params['epoch']+1)
    training_step = int((len(train_data)/params["batch_size"])*params['epoch']+1)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                   num_warmup_steps=warmup_step,
                                                                   num_training_steps=training_step,
                                                                   num_cycles=1)
    print("training model...")
    from tqdm import tqdm
    model_id = f"{seed}_{data_path}" # enc_name_model_name_data_name
    dev_report_file = open(f"./outputs/reports/{model_id}_dev.txt", "w", encoding="utf-8")  # 记录模型分类报告
    test_report_file = open(f"./outputs/reports/{model_id}_test.txt", "w", encoding="utf-8")  # 记录模型分类报告
    for epoch in range(params['epoch']):
        print(f"-------------------------------epoch:{epoch + 1}-------------------------------------")
        model.train()
        train_loss = 0
        for ids, inputs, enc_inputs, dfds, grouped_dfds, charge_idxs, grouped_charge_idxs, cate_idxs, grouped_cate_idxs, sp_lens, pad_sp_lens, relevant_sents, mask_positions, dfd_positions in tqdm(train_data_loader):
            # 梯度置零
            optimizer.zero_grad()
            group_scores = model(enc_inputs, mask_positions, pad_sp_lens, relevant_sents)
            loss = criterion(group_scores, torch.tensor(cate_idxs).to(device))
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
        print("Dev:")
        util.eval_group_prediction(model, dev_data_loader)
        print("Test:")
        util.eval_group_prediction(model, test_data_loader)

def main():
    params = setting.params
    enc_path = "./pretrained_files/roberta"
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
                train(seed, enc_path, data_path, params, index2charge[:-1], charge_desc, copy.deepcopy(cate2charge))
            else:
                train(seed, enc_path, data_path, params, index2charge, charge_desc, copy.deepcopy(cate2charge))

if __name__=="__main__":
    main()