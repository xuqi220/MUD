import os,json, time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import copy
from preprocess import Processor
from model import Reasoner
import setting
import torch
from dataset import ClozeDataset, Lang
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_sent_id(sent_score, pad_sp_lens, sp_len):
    sent_pred = sent_score.argmax(dim=1).cpu().tolist()
    max_len = max([len(s) for s in pad_sp_lens])
    sp_sent_pred = [ts.tolist() for ts in torch.split(torch.tensor(sent_pred),[max_len]*int(len(sent_pred)/max_len))]
    relevant_sents = []
    for i in range(len(sp_sent_pred)):
        temp = []
        for idx, val in enumerate(sp_sent_pred[i]):
            if val == 1 and idx < len(sp_len[i]):
                temp.append(idx)
        relevant_sents.append(temp)
    return relevant_sents

def reason(enc, data_path, index2charge, charge_desc, cate2charge, topk, alpha, beta, f, pattern):
    # util.set_seed(seed)
    # print("preparing dataset...")
    train_path, dev_path, test_path = f"./datasets/{data_path}/train.json",f"./datasets/{data_path}/dev.json",f"./datasets/{data_path}/test.json"
    lang = Lang(index2charge, charge_desc, cate2charge)
    processor = Processor(lang)
    train_data, dev_data, test_data = processor.load_data(train_path), processor.load_data(dev_path), processor.load_data(test_path)
    dataset_dev = ClozeDataset(dev_data, enc, lang)
    dataset_test = ClozeDataset(test_data, enc, lang)
    dev_data_loader = DataLoader(dataset_dev, batch_size=10, collate_fn=dataset_dev.collate_fn, shuffle=False)
    test_data_loader = DataLoader(dataset_test, batch_size=10, collate_fn=dataset_test.collate_fn, shuffle=False)
    fact_selector = torch.load(f"./pretrained_files/fact_selector_{data_path}.pkl") # fact selector
    group_selector = torch.load(f"./pretrained_files/group_selector_{data_path}.pkl") # 预测类别
    rule_selector = torch.load("./pretrained_files/rule_selector(9).pkl") # 预测指控
    reasoner = Reasoner(group_selector, rule_selector, enc, lang, device, topk=topk, alpha=alpha, beta=beta)
    reasoner.to(device)
    s_dev = start(reasoner, dev_data_loader, pattern,fact_selector)
    s_test = start(reasoner, test_data_loader, pattern,fact_selector)
    print(f"{data_path}_dev : {s_dev}\n{data_path}_test: {s_test}\n")
    f.write(f"{data_path}_dev : {s_dev}\n{data_path}_test: {s_test}\n")

def start(reasoner, data_loader, pattern, fact_selector):
    y_preds = []
    y_trues = []
    group_y_preds = []
    group_y_trues = []
    reasoner.eval()
    fact_selector.eval()
    logits = []
    with open("./sota_error.txt", "r", encoding="utf-8") as fi:
        sota_error_ids = [json.loads(line)[0] for line in fi.readlines()]
    with torch.no_grad():
        with open("./anna.txt", "w", encoding="utf-8") as fi:
            for ids, inputs, enc_inputs, dfds, grouped_dfds, charge_idxs, grouped_charge_idxs, cate_idxs, grouped_cate_idxs, sp_lens, pad_sp_lens, relevant_sents, mask_positions, dfd_positions in data_loader:
                if pattern == "ext":
                    sent_score = fact_selector(enc_inputs, pad_sp_lens)
                    relevant_sents = get_sent_id(sent_score, pad_sp_lens, sp_lens)
                preds, logs = reasoner(inputs, mask_positions, pad_sp_lens, relevant_sents, dfd_positions)
                y_preds.extend(preds)
                y_trues.extend(charge_idxs)
                pred_sp_len = [len(i) for i in grouped_charge_idxs]
                # grouped_y_pred = [ts.tolist() for ts in torch.split(torch.tensor(preds), sp_len)]
                group_y_preds.extend([ts.tolist() for ts in torch.split(torch.tensor(preds), pred_sp_len)])
                group_y_trues.extend(grouped_charge_idxs)
                for id, input, log, pred, label in zip(ids, inputs, logs, preds, charge_idxs):
                    if pred==label and label in sota_error_ids:
                        fi.write(json.dumps([id, input, log, pred, label], ensure_ascii=False)+"\n")
    acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
    precision = precision_score(y_true=y_trues, y_pred=y_preds, average="macro")
    recall = recall_score(y_true=y_trues, y_pred=y_preds, average="macro")
    f1 = f1_score(y_true=y_trues, y_pred=y_preds, average="macro")
    right_case = 0
    for i, j in zip(group_y_preds, group_y_trues):
        if i == j:
            right_case += 1
    acc_ = right_case / len(group_y_trues)
    s = f"acc:{round(acc, 3)}  precision:{round(precision, 3)}  recall:{round(recall, 3)}  f1:{round(f1, 3)}, acc_:{round(acc_, 3)}"
    return s

def main():
    params = setting.params
    enc = setting.enc
    index2charge = setting.index2charge
    charge_desc = setting.charge_desc
    cate2charge = setting.cate2charge
    print("Running ...")
    with open("./reason_records.txt", "w", encoding="utf-8") as f:
        for alpha in params["alpha"]:
            for beta in params["beta"]:
                for topk in params["topk"]:
                    logs = f"--------------------------------\nalpha: {alpha}\nbeta: {beta}\ntopk: {topk}\n"
                    print(logs)
                    f.write(logs)
                    for data_path in params["data_path"]:
                        if data_path == "hard":
                            reason(enc, data_path, index2charge[:-1], charge_desc, copy.deepcopy(cate2charge), topk, alpha, beta, f, params["pattern"])
                        else:
                            reason(enc, data_path, index2charge, charge_desc, copy.deepcopy(cate2charge), topk, alpha, beta, f, params["pattern"])
    print("ending")
if __name__=="__main__":
    main()