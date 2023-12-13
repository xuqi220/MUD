import os,json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import util
from preprocess import Processor
from model import Base, BaseWP, BaseWE, BaseWEE, MyModel
import setting
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from dataset import ClozeDataset, Lang
from torch.utils.data import DataLoader
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
transformers.logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = setting.params
encs = setting.encs
charge_desc = setting.charge_desc
cate2charge = setting.cate2charge



def train(seed, enc_name, enc_path, model_name, data_path, params):
    print("preparing dataset...")
    train_path, dev_path, test_path = f"./datasets/{data_path}/train.json",f"./datasets/{data_path}/dev.json",f"./datasets/{data_path}/test.json"
    lang = Lang(train_path, charge_desc, cate2charge)
    processor = Processor(lang)
    train_data, dev_data, test_data = processor.load_data(train_path), processor.load_data(dev_path), processor.load_data(test_path)
    dataset_train = ClozeDataset(train_data, enc_path, lang, charge_desc)
    dataset_dev = ClozeDataset(dev_data, enc_path, lang, charge_desc)
    dataset_test = ClozeDataset(test_data, enc_path, lang, charge_desc)
    train_data_loader = DataLoader(dataset_train, batch_size=params['batch_size'], collate_fn=dataset_train.collate_fn, shuffle=True)
    dev_data_loader = DataLoader(dataset_dev, batch_size=params['batch_size'], collate_fn=dataset_dev.collate_fn, shuffle=False)
    test_data_loader = DataLoader(dataset_test, batch_size=params['batch_size'], collate_fn=dataset_test.collate_fn, shuffle=False)

    print("creating model...")
    if model_name == "Base":
        model = Base(enc_path, lang, device)
    if model_name == "BaseWP":
        model = BaseWP(enc_path, lang, device)
    if model_name == "BaseWE":
        model = BaseWE(enc_path, lang, device)
    if model_name == "BaseWEE":
        ee_path = f"./pretrained_files/EE/roberta_wwm_80_{data_path}.pkl"
        model = BaseWEE(enc_path, ee_path, lang, device)
    if model_name == "MyModel":
        ee_path = f"./pretrained_files/EE/roberta_wwm_80_{data_path}.pkl"
        model = MyModel(enc_path, ee_path, lang, device)
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
    model_id = f"{seed}_{enc_name}_{model_name}_{data_path}" # enc_name_model_name_data_name
    dev_report_file = open(f"./outputs/reports/{model_id}_dev.txt", "w", encoding="utf-8")  # 记录模型分类报告
    test_report_file = open(f"./outputs/reports/{model_id}_test.txt", "w", encoding="utf-8")  # 记录模型分类报告
    for epoch in range(params['epoch']):
        sp = f"-------------------------------epoch:{epoch + 1}-------------------------------------"
        print(sp)
        model.train()
        train_loss = 0
        for ids, inputs, enc_inputs, enc_desc, dfds, grouped_dfds, charge_idxs, grouped_charge_idxs, cate_idxs, grouped_cate_idxs, sent_lens, pad_sp_lens, relevant_sents, mask_positions, dfd_positions in tqdm(train_data_loader):
            # 梯度置零
            optimizer.zero_grad()
            if isinstance(model, Base): # baseline
                charge_scores = model(enc_inputs, mask_positions)
            if isinstance(model, BaseWP): # add dfd position
                charge_scores = model(enc_inputs, mask_positions, pad_sp_lens, dfd_positions)
            if isinstance(model, BaseWE): # add anno Elem
                charge_scores = model(enc_inputs, mask_positions, pad_sp_lens, relevant_sents)
            if isinstance(model, BaseWEE): # add Ext Elem
                charge_scores = model(enc_inputs, mask_positions, pad_sp_lens)
            if isinstance(model, MyModel): # add Ext Elem
                group_scores, charge_scores = model(enc_inputs, enc_desc, mask_positions, pad_sp_lens, dfd_positions)
            c_loss = criterion(charge_scores, torch.tensor(charge_idxs).to(device))
            g_loss = criterion(group_scores, torch.tensor(cate_idxs).to(device))
            loss = c_loss+g_loss
            train_loss+=loss.item()
            # 累计梯度
            loss.backward()
            # 梯度
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 更新梯度
            optimizer.step()
            # 更新学习率
            scheduler.step()
        print(f"train_loss:{round(train_loss/len(train_data_loader.dataset), 4)}")
        # torch.save(model, f"./outputs/models/{prefix}.pkl")
        util.evaluate(lang, model, epoch, dev_data_loader, dev_report_file, model_id, lang.charge2cate, mode="dev")
        util.evaluate(lang, model, epoch, test_data_loader, test_report_file, model_id, lang.charge2cate, mode="test")

def main():
    print("Running ...")
    for seed in params["seeds"][:5]:
        print(f"set seed {seed}")
        util.set_seed(seed)
        for enc_name, enc_path in encs.items():
            for data_path in params["data_path"]:
                for model_name in params["model_name"]:
                    print(f"enc_name: {enc_name}\n"
                          f"model_name: {model_name}\n"
                          f"data: {data_path}\n"
                          f"batch_size: {params['batch_size']}\n"
                          f"lr: {params['lr']}\n")
                    train(seed, enc_name, enc_path, model_name, data_path, params)


if __name__=="__main__":
    main()