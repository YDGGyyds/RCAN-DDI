import math
from typing import Hashable
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim import lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
# from utils import get_adj_mat,load_feat,normalize_adj,preprocess_adj,get_train_test_set
import sys
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from time import gmtime, strftime,localtime
import numpy as np
# from torch_geometric.nn import GCNConv,GATConv
# from utils import get_adj_mat
import tensorflow as tf
'''Model definition'''




#Generator definition with WGAN style
class Generator(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(Generator,self).__init__()
        self.l1 = nn.Linear(num_inputs,80)
        self.l2 = nn.Linear(80,num_outputs)
        self.relu = F.relu
        self.dropout = nn.Dropout()
        
    def forward(self,x):
        x = self.l1(x)
        
        x = self.l2(x)
      
        return x


class MLP(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        # super(Generator,self).__init__()
        super().__init__()
        self.l1 = nn.Linear(num_inputs,80)
        self.l2 = nn.Linear(80,num_outputs)
        self.relu = F.relu
        self.dropout = nn.Dropout()
        
    def forward(self,x):
        x = self.l1(x)
        
        x = self.l2(x)
      
        return x

#WGAN Style discriminator
class Discriminator(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(Discriminator,self).__init__()
        self.l1 = nn.Linear(num_inputs,80)
        self.l2 = nn.Linear(80,num_outputs)
        self.relu = F.relu
        
    def forward(self,x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        # （1710,1）取均值
        y = x.mean(0)
        return y




class DNN(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(DNN,self).__init__()
        
        self.layers = nn.Sequential(nn.Linear(num_inputs,512),nn.ReLU(),nn.BatchNorm1d(512),nn.Dropout(0.3),
                      nn.Linear(512,256),nn.ReLU(),nn.BatchNorm1d(256),nn.Dropout(0.3),
                      nn.Linear(256,num_outputs)
                      )
    def forward(self,x):
        output = torch.sigmoid(self.layers(x))
        return output


class Model0(nn.Module):
    def __init__(self,num_nodes,gcn_outputs,gcn_hidden,num_outputs,attr_dim):
        super(Model0,self).__init__()

        self.g_t = MLP(gcn_outputs,num_outputs) #Projecting structural embedding to the ommonc space
        self.g_t2a = Generator(num_outputs,attr_dim) #Generator for structure to attribute
        self.g_a = MLP(attr_dim,num_outputs) #MLP to encoding drug attribute
        self.g_a2t = Generator(num_outputs,gcn_outputs) #Generator for attribute to structure

        # self.classifier = DNN(num_outputs * 4, 1)
        
    def forward(self,sa,ro,x,*args):
        # 初始化特征GAT更新特征（1710,64）->（1710，256） ->（1710，80）

        MLP_sa = self.g_t(sa)
        t2a_mtx = self.g_t2a(MLP_sa)
        MLP_ro = self.g_a(ro)
        a2t_mtx = self.g_a2t(MLP_ro)
        array = np.loadtxt('data_array.csv', delimiter=',')
        embedding = torch.hstack([MLP_sa, MLP_ro])

        embedding=embedding.detach().cpu().numpy()
        embedding=tf.convert_to_tensor(embedding)

        self.h_t = MLP_sa
        self.h_a = MLP_ro
        self.topo_emb = sa
        self.attr_emb = ro
        self.false_topo = a2t_mtx
        self.false_attr = t2a_mtx

        # X1=embedding[x[:, 0]]
        # X2=embedding[x[:, 1]]
        # X = torch.hstack([X1,X2])
        indices1 = x[:, 0]
        indices2 = x[:, 1]

        # 使用 tf.gather 获取对应的向量
        X1 = tf.gather(embedding, indices1)
        X2 = tf.gather(embedding, indices2)
        X = tf.concat([X1, X2], 1)

        output = DNN(X, 1)
        # output = self.classifier(X)
        # output = 1
        return t2a_mtx,a2t_mtx,t2a_mtx,a2t_mtx,output


def DNN(inputs, num_outputs):
    dense1 = tf.layers.dense(inputs, 512, activation=tf.nn.relu)
    batch_norm1 = tf.layers.batch_normalization(dense1)
    dropout1 = tf.layers.dropout(batch_norm1, rate=0.3)

    dense2 = tf.layers.dense(dropout1, 256, activation=tf.nn.relu)
    batch_norm2 = tf.layers.batch_normalization(dense2)
    dropout2 = tf.layers.dropout(batch_norm2, rate=0.3)

    output = tf.layers.dense(dropout2, num_outputs)

    return output
