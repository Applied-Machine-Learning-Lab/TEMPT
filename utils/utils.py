import json
import random
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import numpy as np
import os
import logging
import time
import torch
from tensorboardX import SummaryWriter


class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


def multi_label_metric(y_gt, y_pred, y_prob):

    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(
                    2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(
                y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(
                y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    auc = roc_auc(y_gt, y_prob)
    p_1 = precision_at_k(y_gt, y_prob, k=1)
    p_3 = precision_at_k(y_gt, y_prob, k=3)
    p_5 = precision_at_k(y_gt, y_prob, k=5)
    f1 = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_pred)
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


def metric_report(logger, y_pred, y_true, therhold=0.5):
    y_prob = y_pred.copy()
    y_pred[y_pred > therhold] = 1
    y_pred[y_pred <= therhold] = 0

    acc_container = {}
    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(
        y_true, y_pred, y_prob)
    acc_container['jaccard'] = ja
    acc_container['f1'] = avg_f1
    acc_container['prauc'] = prauc

    # acc_container['jaccard'] = jaccard_similarity_score(y_true, y_pred)
    # acc_container['f1'] = f1(y_true, y_pred)
    # acc_container['auc'] = roc_auc(y_true, y_prob)
    # acc_container['prauc'] = precision_auc(y_true, y_prob)

    for k, v in acc_container.items():
        logger.info('%-10s : %-10.4f' % (k, v))

    return acc_container


def t2n(x):
    return x.detach().cpu().numpy()


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def set_seed(seed):
    '''Fix all of random seed for reproducible training'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_res(args, res, now_str, out_path):

    if not os.path.exists(out_path):
        with open(out_path, 'w+') as f:
            json.dump({}, f)
    
    with open(out_path, 'r') as f:
        res_dict = json.load(f)

    new_dict = {'model': args.model_name, 'hos_id': args.hos_id}
    new_dict.update(res)
    res_dict.update({now_str: new_dict})

    with open(out_path, 'w') as f:
        json.dump(res_dict, f)


def log_res_multi(args, res, now_str, out_path):

    if not os.path.exists(out_path):
        with open(out_path, 'w+') as f:
            json.dump({}, f)
    
    with open(out_path, 'r') as f:
        res_dict = json.load(f)
    
    for i, hos_id in enumerate(res.keys()):

        new_dict = {'model': args.model_name, 'hos_id': int(hos_id)}
        new_dict.update(res[hos_id])
        res_dict.update({now_str+'_'+str(i): new_dict})
    
    with open(out_path, 'w') as f:
        json.dump(res_dict, f)


def log_efficiency(best_epoch, train_time, fp_num, ap_num, args, now_str):

    out_path = args.out_exp.split('/')[-1]
    out_path = os.path.join('./log/efficiency', out_path)

    if not os.path.exists(out_path):
        with open(out_path, 'w+') as f:
            json.dump({}, f)
    
    with open(out_path, 'r') as f:
        res_dict = json.load(f)

    res = {'best_epoch': best_epoch,
           'train_time': train_time,
           'fp_num': fp_num,
           'ap_num':ap_num}
    new_dict = {'model': args.model_name, 'hos_id': args.hos_id}
    new_dict.update(res)
    res_dict.update({now_str: new_dict})

    with open(out_path, 'w') as f:
        json.dump(res_dict, f)
    



