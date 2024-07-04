import utility.metrics as metrics
from . import metrics
from .parser import parse_args
import multiprocessing
import heapq
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn import ensemble

from .loader_rcanddi import RCANDDI_loader

import threading
from threading import Lock, Thread
import time, os
import random as rd

cores = multiprocessing.cpu_count() // 2
args = parse_args()

data_generator = RCANDDI_loader(args=args, path=args.data_path + args.dataset)
batch_test_flag = False

ITEM_NUM = data_generator.n_drugs
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size

lock = Lock()
total_thread = 128


# 平衡标签
def get_label_score(adj_recovered, test_dict, test_neg_dict):
    label_true, score_predict = [], []
    for k, value in test_dict.items():
        for v in value:
            if k < v:
                label_true.append(1)
                score_predict.append(adj_recovered[k, v])

    for k, value in test_neg_dict.items():
        for v in value:
            if k < v:
                label_true.append(0)
                score_predict.append(adj_recovered[k, v])
    return np.array(label_true), np.array(score_predict)


# 不平衡标签
def get_label_score1(adj_recovered, test_dict):
    label_true, score_predict = [], []
    for i in range(1710):
        for j in range(1710):
            if i in data_generator.train_drug_dict.keys() and j in data_generator.train_drug_dict[i]:
                continue
            elif i not in test_dict or j not in test_dict[i]:
                label_true.append(0)
            else:
                label_true.append(1)
            score_predict.append(adj_recovered[i][j])
    return np.array(label_true), np.array(score_predict)


def test(sess, model, drugs_to_test, train_dict, test_dict, test_neg_dict, drop_flag=False, batch_test_flag=False):
    pool = multiprocessing.Pool(cores)
    # 2048
    u_batch_size = BATCH_SIZE * 2
    # 1545
    test_drugs = drugs_to_test
    n_test_drugs = len(test_drugs)
    # 1       // 整除：相除的整数部分，即向下取整
    n_drug_batchs = n_test_drugs // u_batch_size + 1

    for _ in range(n_drug_batchs):
        # item_batch = range(ITEM_NUM,2*ITEM_NUM)   0-1710
        item_batch = range(ITEM_NUM)
        feed_dict = data_generator.generate_test_feed_dict(model=model,
                                                           drug_batch=item_batch,
                                                           item_batch=item_batch,
                                                           drop_flag=drop_flag)
        # 1710,100    1710,100    1710,1710
        _, _, rate_batch = model.eval(sess, feed_dict=feed_dict)
        # 1710,1710
        rate_batch1 = rate_batch.reshape((-1, len(item_batch)))
        # 1710,1710
        rate_batch = rate_batch1 + rate_batch1.T
        # lable:(38340,)     rating:(38340,)
        label, rating = get_label_score(rate_batch, test_dict, test_neg_dict)

        rating = np.array(rating)
        label = np.array(label)
        result_all = evaluate(rating, label, 2)
        # {'aupr': 0.989067027219868, 'auc': 0.9883728603307658}
        print('banlance:', result_all)
        # (38340,)
        rating_sigmoid = 1 / (1 + (np.exp((-rating))))
        threshold = 0.9
        ##(38340,)
        pred_label = 1 * (rating_sigmoid > threshold)
        f1_micro = f1_score(label, pred_label, average='micro')
        f1_macro = f1_score(label, pred_label, average='macro')
        print('micro f1:%f,macro f1:%f' % (f1_micro, f1_macro))
        # micro f1:0.950183,macro f1:0.950182
        label, rating = get_label_score1(rate_batch, test_dict)
        rating = np.array(rating)
        label = np.array(label)
        result_all1 = evaluate(rating, label, 2)
        # no balance {'aupr': 0.7725939708241074, 'auc': 0.9883235753942539}
        print('no balance', result_all1)
    pool.close()
    return result_all1['aupr']

#
# def evaluate(pred_score, y_test, event_num):
#     result_all = {}
#     # y_one_hot = label_binarize(y_test, np.arange(event_num))
#
#     precision, recall, _ = precision_recall_curve(y_test, pred_score)
#     result_all['aupr'] = auc(recall, precision)  # aupr
#     # result_all['auc'] = roc_auc_score(y_one_hot, pred_score, average='micro')
#     result_all['auc'] = 0
#     return result_all

from sklearn.metrics import precision_score, recall_score, average_precision_score
def evaluate(pred_score, y_test, event_num):
    result_all = {}
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    precision, recall, _ = precision_recall_curve(y_test, pred_score)
    result_all['aupr'] = auc(recall, precision)  # AUPR
    result_all['auc'] = roc_auc_score(y_one_hot, pred_score, average='micro')  # AUC
    threshold = 0.9  # 设置阈值
    y_pred_binary = np.where(pred_score >= threshold, 1, 0)
    # Accuracy
    accuracy = np.mean(y_pred_binary == y_test)
    result_all['accuracy'] = accuracy

    # Precision
    precision = precision_score(y_test, y_pred_binary, average='macro', zero_division=0)
    result_all['precision'] = precision

    # Average Precision
    average_precision = average_precision_score(y_one_hot, pred_score, average='macro')
    result_all['average_precision'] = average_precision

    # Recall
    recall = recall_score(y_test, y_pred_binary, average='macro', zero_division=0)
    result_all['recall'] = recall
    # #############################################
    #
    # result_all['accuracy1'] = accuracy_score(y_test, y_pred_binary)
    # result_all['aupr1'] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    # result_all['auc1'] = roc_auc_score(y_one_hot, pred_score, average='micro')
    # result_all['f11'] = f1_score(y_test, y_pred_binary, average='macro')
    # result_all['precision1'] = precision_score(y_test, y_pred_binary, average='macro')
    # result_all['recall1'] = recall_score(y_test, y_pred_binary, average='macro')
    # ##################################################
    return result_all

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        order = np.lexsort((recall, precision))
        return auc(precision[order], recall[order])

    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)


def multiclass_precision_recall_curve(y_true, y_score):
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_score.ndim == 1:
        y_score = y_score.reshape((-1, 1))
    y_true_c = y_true.take([0], axis=1).ravel()
    y_score_c = y_score.take([0], axis=1).ravel()
    precision, recall, pr_thresholds = precision_recall_curve(y_true_c, y_score_c)
    return (precision, recall, pr_thresholds)
