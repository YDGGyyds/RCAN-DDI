import tensorflow as tf
from utility.helper import *
from utility.batch_test import *
from time import time
from RCANDDI import RCANDDI
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from collections import defaultdict

from model0 import Discriminator,Model0
import os
import sys
#
# 这行代码设置了环境变量TF_CPP_MIN_LOG_LEVEL的值为'2'，这将控制TensorFlow的日志输出级别。设置为'2'将禁用除错误消息之外的所有日志输出。
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
###################################新增#########################333333333333333333333



##################################新增#########################333333333333333333333
if __name__ == "__main__":
    args = parse_args()
    ################新增########################################3333333333
    n_topo_feats = args.n_topo_feats
    n_hid = args.n_hid
    n_out_feat = args.n_out_feat
    ################新增########################################3333333333
    max_aupr = 0.
    train_dict = data_generator.train_dict
    test_dict = data_generator.test_dict
    test_neg_dict = data_generator.test_neg_dict
    print('检索test，train数据完毕。')
    # 这两行代码设置了TensorFlow和NumPy的随机种子，以确保实验的可重复性。
    tf.set_random_seed(2021)
    np.random.seed(2021)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    """
    *********************************************************
    Load Data from data_generator function.
    """
    config = dict()
    config['n_drugs'] = data_generator.n_drugs
    config['n_relations'] = data_generator.n_relations
    # config['w_sp_matrix'] = data_generator.w_sp_matrix

    "Load the KG triplets."
    config['all_h_list'] = data_generator.all_h_list
    config['all_r_list'] = data_generator.all_r_list
    config['all_t_list'] = data_generator.all_t_list
    config['sparse_adj_list'] = data_generator.sparse_adj_list

    t0 = time()
    #############################新增###########################333333333


    #############################新增###########################333333
    model = RCANDDI(data_config=config, args=args)
    # 创建了一个 TensorFlow 的 Saver 对象，用于保存和恢复模型的权重和变量。
    saver = tf.compat.v1.train.Saver()
    # 创建 TensorFlow 1.x 会话配置的方法。
    config = tf.compat.v1.ConfigProto()
    # 这一行代码将 allow_growth 属性设置为 True。这意味着 TensorFlow 会话将动态分配 GPU 内存，根据需要增加内存，而不是一开始就分配所有可用的 GPU 内存。这有助于有效地管理 GPU 内存，并允许多个 TensorFlow 会话共享 GPU。
    config.gpu_options.allow_growth = True
    # 这一行代码创建了一个 TensorFlow 1.x 会话，并将上述配置应用于会话。这个会话将按照配置的方式使用 GPU，并允许 GPU 内存动态增长。
    sess = tf.compat.v1.Session(config=config)

    """
    *********************************************************
    Reload the model parameters to fine tune.
    """
    # 这行代码用于在 TensorFlow 1.x 中初始化全局变量。
    sess.run(tf.compat.v1.global_variables_initializer())
    print('without pretraining.')

    """
    *********************************************************
    Train.
    """
    # print('current margin:',model.margin)

    for epoch in range(args.epoch):
        t1 = time()
        loss, base_loss, kge_loss, reg_loss = 0., 0., 0., 0.
        # // 运算符：这是整数除法运算符，它会将结果四舍五入为最接近的整数。
        # n_batch = 338  它表示将训练数据集分成多少个小批次进行训练。
        n_batch = data_generator.n_train // args.batch_size + 1
        for idx in range(n_batch):
            btime = time()

            batch_data = data_generator.generate_train_batch()
            feed_dict = data_generator.generate_train_feed_dict(model, batch_data)

            _, batch_loss1, batch_base_loss, batch_kge_loss1, batch_reg_loss1, \
            batch_loss2, batch_kge_loss2, batch_reg_loss2 ,sa_array, ro_array,Dganloss= model.train(sess, feed_dict=feed_dict)

            loss = loss + batch_loss1 + batch_loss2
            base_loss += batch_base_loss
            kge_loss = batch_kge_loss1 + batch_kge_loss2 + kge_loss
            reg_loss = reg_loss + batch_reg_loss1 + batch_reg_loss2
            # breaky

        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f],%.5f' % (
            epoch, time() - t1, loss, base_loss, kge_loss, reg_loss,Dganloss)
        print(perf_str)
        if epoch % 5 == 0:
            drugs_to_test = list(data_generator.test_drug_dict.keys())
            ret = test(sess, model, drugs_to_test, train_dict, test_dict, test_neg_dict, drop_flag=False,
                    batch_test_flag=batch_test_flag)
    drugs_to_test = list(data_generator.test_drug_dict.keys())
    ret = test(sess, model, drugs_to_test, train_dict, test_dict, test_neg_dict, drop_flag=False,
               batch_test_flag=batch_test_flag)

