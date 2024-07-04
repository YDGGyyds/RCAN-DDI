import tensorflow as tf
import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from model0 import Discriminator,Model0

# #############################3333333333333333333333
class RCANDDI(object):
    def __init__(self, data_config, args):
        self._parse_args(data_config, args)
        self._build_inputs()
        self.weights = self._build_weights()
        self._build_model_phase_I()
        self._build_loss_phase_I()
        self._build_model_phase_II()
        self._build_loss_phase_II()
        self._build_total_loss()
        self._statistics_params()

    def _parse_args(self, data_config, args):
        # argument settings
        self.model_type = 'rcanddi'

        self.n_drugs = data_config['n_drugs']
        # self.n_items = data_config['n_items']
        self.n_entities = data_config['n_drugs']
        self.n_relations = data_config['n_relations']

        self.all_h_list = data_config['all_h_list']
        self.all_r_list = data_config['all_r_list']
        self.all_t_list = data_config['all_t_list']

        self.lr = args.lr

        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)

        self.kge_dim = args.kge_size
        self.emb_dim = args.kge_size

        self.regs = eval(args.regs)
        self.verbose = args.verbose
        self.margin = args.margin
        self.B = args.B

        # 这里更新一下sparse_adj_list #1317*1317
        self.all_sparse_adj_list = data_config['sparse_adj_list']
        self.sparse_adj_list = self._sparse_adj_list_process()

    def _sparse_adj_list_process(self):
        sparse_adj_list = []
        for adj in self.all_sparse_adj_list:
            convert_adj = self._convert_sp_mat_to_sp_tensor(adj)  # 转tensor
            sparse_adj_list.append(convert_adj)
        return sparse_adj_list

    def _build_inputs(self):
        # placeholder definition
        self.drugs = tf.placeholder(tf.int32, shape=(None,))
        self.pos_drugs = tf.placeholder(tf.int32, shape=(None,))
        self.neg_drugs = tf.placeholder(tf.int32, shape=(None,))
        self.data_array = tf.placeholder(tf.int32, shape=(None,2))
        self.h = tf.placeholder(tf.int32, shape=[None], name='h')
        self.r = tf.placeholder(tf.int32, shape=[None], name='r')
        self.pos_t = tf.placeholder(tf.int32, shape=[None], name='pos_t')
        self.neg_t = tf.placeholder(tf.int32, shape=[None], name='neg_t')
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

    def _build_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()
        # 初始化实数部分和虚数部分的药物和实体的嵌入向量（1710,64）
        all_weights['re_drug_embed'] = tf.Variable(initializer([self.n_drugs, self.emb_dim]), name='re_drug_embed')
        all_weights['re_entity_embed'] = tf.Variable(initializer([self.n_entities, self.emb_dim]),
                                                     name='re_entity_embed')
        all_weights['im_drug_embed'] = tf.Variable(initializer([self.n_drugs, self.emb_dim]), name='im_drug_embed')
        all_weights['im_entity_embed'] = tf.Variable(initializer([self.n_entities, self.emb_dim]),
                                                     name='im_entity_embed')

        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())  # 初始化所有变量
        #     weights_values = sess.run(all_weights)  # 获取所有变量的数值
        #     print(weights_values)
        # print('using xavier initialization')
        # 初始化关系嵌入（86,46）
        all_weights['re_relation_embed'] = tf.Variable(initializer([self.n_relations, self.kge_dim]),
                                                       name='re_relation_embed')
        all_weights['im_relation_embed'] = tf.Variable(initializer([self.n_relations, self.kge_dim]),
                                                       name='im_relation_embed')
        # 关系映射的权重矩阵（86,2*64）
        all_weights['relation_mapping'] = tf.Variable(initializer([self.n_relations, 2 * self.kge_dim]),
                                                      name='relation_mapping')
        # 关系矩阵的权重矩阵（35,2*64,2*64）<tf.Variable 'relation_matrix:0' shape=(35, 128, 128) dtype=float32_ref>
        all_weights['relation_matrix'] = tf.Variable(initializer([self.B, self.kge_dim, self.kge_dim]),
                                                     name='relation_matrix')
        # 随机数组（86，35）：[[ 1.4886091   0.67601085 -0.41845137 ...  0.7281305  -0.38964808,   0.27889377], [ 0.0519002  -1.0447437  -0.16150753 ...  0.8703685   1.0032452,  -0.36644983], [ 1.1280516   0.7925838  -1.7508425  ...  1.2572469  -0.44170406,   0.5413351 ], ..., [-0.0614
        relation_initial = np.random.randn(self.n_relations, self.B).astype(np.float32)
        # 使用tf.Variable()函数创建了alpha变量，并将relation_initial作为初始值。它的名称为arb。
        all_weights['alpha'] = tf.Variable(relation_initial, name='arb')
        # （86,1710,1）
        relation_d_att = np.random.randn(self.n_relations, self.n_drugs, 1).astype(np.float32)
        relation_e_att = np.random.randn(self.n_relations, self.n_entities, 1).astype(np.float32)
        all_weights['relation_d_att'] = tf.Variable(relation_d_att, name='relation_d_att')
        all_weights['relation_e_att'] = tf.Variable(relation_e_att, name='relation_e_att')
        all_weights['att1'] = tf.Variable(initializer([1,64]), name='att1')
        all_weights['att2'] = tf.Variable(initializer([86,1,1]), name='att2')
        all_weights['att5'] = tf.Variable(initializer([3420,192]), name='att5')
        # list：2  [128,100]
        self.weight_size_list = [self.emb_dim * 2] + self.weight_size
        # (256,100)
        all_weights['W_mlp_0'] = tf.Variable(
            initializer([192, self.weight_size_list[1]]), name='W_mlp_0')
        # (1,100)
        all_weights['b_mlp_0'] = tf.Variable(
            initializer([1, self.weight_size_list[1]]), name='b_mlp_0')

        return all_weights

    def _build_model_phase_I(self):
        # 1710,100药物嵌入和实体嵌入
        self.da_embeddings, self.ea_embeddings ,self.drug_neigh, self.drug_embedding1 = self._create_bi_interaction_embed()
        # 从嵌入矩阵中查找指定索引的嵌入向量
        self.d_e = tf.nn.embedding_lookup(self.da_embeddings, self.drugs)
        self.pos_e = tf.nn.embedding_lookup(self.ea_embeddings, self.pos_drugs)
        self.neg_e = tf.nn.embedding_lookup(self.ea_embeddings, self.neg_drugs)

        # 所有pro drug的嵌入表示
        # self.all_pro_drug_embdding = tf.nn.embedding_lookup(self.ea_embeddings, self.all_pro_drug)
        # transpose_a=False表示不对self.d_e进行转置，transpose_b=True表示对self.pos_e进行转置
        # Yb公式6
        self.batch_predictions = tf.matmul(self.d_e, self.pos_e, transpose_a=False, transpose_b=True)



    def _build_model_phase_II(self):
        self.re_h_e, self.re_pos_t_e, self.re_neg_t_e, self.im_h_e, self.im_pos_t_e, self.im_neg_t_e, self.re_r_e, self.im_r_e = self._get_kg_inference_rotate(
            self.h, self.r, self.pos_t, self.neg_t)

    def _get_kg_inference_rotate(self, h, r, pos_t, neg_t):
        pi = 3.14159265358979323846
        re_embeddings = tf.concat([self.weights['re_drug_embed'], self.weights['re_entity_embed']], axis=0)
        re_embeddings = tf.expand_dims(re_embeddings, 1)
        im_embeddings = tf.concat([self.weights['im_drug_embed'], self.weights['im_entity_embed']], axis=0)
        im_embeddings = tf.expand_dims(im_embeddings, 1)

        re_h_e = tf.nn.embedding_lookup(re_embeddings, h)
        re_pos_t_e = tf.nn.embedding_lookup(re_embeddings, pos_t)
        re_neg_t_e = tf.nn.embedding_lookup(re_embeddings, neg_t)
        im_h_e = tf.nn.embedding_lookup(im_embeddings, h)
        im_pos_t_e = tf.nn.embedding_lookup(im_embeddings, pos_t)
        im_neg_t_e = tf.nn.embedding_lookup(im_embeddings, neg_t)

        re_h_e = tf.reshape(re_h_e, [-1, self.kge_dim])
        re_pos_t_e = tf.reshape(re_pos_t_e, [-1, self.kge_dim])
        re_neg_t_e = tf.reshape(re_neg_t_e, [-1, self.kge_dim])
        im_h_e = tf.reshape(im_h_e, [-1, self.kge_dim])
        im_pos_t_e = tf.reshape(im_pos_t_e, [-1, self.kge_dim])
        im_neg_t_e = tf.reshape(im_neg_t_e, [-1, self.kge_dim])

        relation = self.weights['re_relation_embed']
        relation = (tf.nn.l2_normalize(relation, dim=1) - 0.5) * pi

        r_e = tf.nn.embedding_lookup(relation, r)
        re_r_e = tf.cos(r_e)
        im_r_e = tf.sin(r_e)

        return re_h_e, re_pos_t_e, re_neg_t_e, im_h_e, im_pos_t_e, im_neg_t_e, re_r_e, im_r_e

    def _build_loss_phase_I(self):
        # 代码使用tf.multiply()函数对self.d_e和self.pos_e进行逐元素相乘，然后使用tf.reduce_sum()函数对结果沿着第1个维度（列）进行求和，得到正样本的分数pos_scores。
        pos_scores = tf.reduce_sum(tf.multiply(self.d_e, self.pos_e), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(self.d_e, self.neg_e), axis=1)
        # 代码计算了正则化项regularizer。它包括了self.d_e、self.pos_e、self.neg_e和self.weights['relation_matrix']的L2范数的和。
        regularizer = tf.nn.l2_loss(self.d_e) + tf.nn.l2_loss(self.pos_e) + tf.nn.l2_loss(self.neg_e) + \
                      tf.nn.l2_loss(self.weights['relation_matrix'])
        # 最后，将regularizer除以self.batch_size来进行归一化。
        regularizer = regularizer / self.batch_size

        # maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores -self.margin))
        # 该部分损失进行修改
        # 首先，代码定义了一个margin变量，用于控制正样本和负样本之间的间隔。
        # 接下来，代码使用tf.math.log()函数和tf.clip_by_value()函数计算了损失函数的一部分。首先，计算了margin-neg_scores的sigmoid函数的对数，然后使用tf.clip_by_value()函数将结果限制在一个较小的范围内，以避免出现无穷大或无穷小的值。然后，计算了pos_scores-margin的sigmoid函数的对数，并进行了相同的限制。最后，将这两部分相加得到maxi。
        margin = 1
        maxi = tf.math.log(tf.clip_by_value(tf.nn.sigmoid(margin - neg_scores), 1e-8, 1.0)) + tf.math.log(
            tf.clip_by_value(tf.nn.sigmoid(pos_scores - margin), 1e-8, 1.0))
        base_loss = tf.negative(tf.reduce_mean(maxi))
        # 接下来，代码计算了基本损失base_loss，通过对maxi取负数并求均值得到。
        self.base_loss = base_loss  # Lb
        # 在这段代码中，self.kge_loss被初始化为一个形状为[1]的浮点型常数张量，值为0.0。这个初始化的目的可能是为了在后续的代码中使用self.kge_loss，并在需要时对其进行更新。
        self.kge_loss = tf.constant(0.0, tf.float32, [1])
        self.reg_loss = self.regs[0] * regularizer  # 正则化项

        self.loss = self.base_loss + self.kge_loss + self.reg_loss +self.Dgan_loss
        # 使用Adam优化器tf.compat.v1.train.AdamOptimizer()，并指定学习率为self.lr，最小化总损失self.loss。
        # Optimization process.RMSPropOptimizer
        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _build_loss_phase_II(self):
        # def _get_kg_score(h_e, r_e, t_e):
        #     kg_score = tf.reduce_sum(tf.square((h_e + r_e - t_e)), 1, keep_dims=True)
        #     return kg_score
        # 用于计算知识图嵌入的分数
        def _get_kg_score(re_h_e, re_pos_t_e, im_h_e, im_pos_t_e, re_r_e, im_r_e):
            re_score = tf.multiply(re_h_e, re_r_e) - tf.multiply(im_h_e, im_r_e)
            im_score = tf.multiply(re_h_e, im_r_e) + tf.multiply(im_h_e, re_r_e)

            re_score = re_score - re_pos_t_e
            im_score = im_score - im_pos_t_e
            kg_score = tf.concat([re_score, im_score], axis=1)
            kg_score = tf.reduce_sum(tf.square((kg_score)), 1, keep_dims=True)
            kg_score = tf.negative(kg_score)

            return kg_score

        # 计算得分
        pos_kg_score = _get_kg_score(self.re_h_e, self.re_pos_t_e, self.im_h_e, self.im_pos_t_e, self.re_r_e,
                                     self.im_r_e)
        # self.prediction = tf.negative(_get_kg_score(self.h_e, self.r_e, self.pos_t_e))
        neg_kg_score = _get_kg_score(self.re_h_e, self.re_neg_t_e, self.im_h_e, self.im_neg_t_e, self.re_r_e,
                                     self.im_r_e)
        # loss
        maxi = tf.log(tf.clip_by_value(tf.nn.sigmoid(pos_kg_score - neg_kg_score - self.margin), 1e-8, 1.0))
        kg_loss = tf.negative(tf.reduce_mean(maxi))

        # loss2 ：the rank-based hinge loss
        # maxi = tf.maximum(0.,neg_kg_score + self.margin - pos_kg_score)
        # kg_loss = tf.reduce_mean(maxi)

        # loss3
        # maxi = tf.log(tf.clip_by_value(tf.nn.sigmoid(self.margin-neg_kg_score),1e-8,1.0)) + tf.log(tf.clip_by_value(tf.nn.sigmoid(pos_kg_score-self.margin),1e-8,1.0))
        # kg_loss = tf.negative(tf.reduce_mean(maxi))

        # loss4 ：the logistic-based loss
        # maxi = tf.log(1+tf.exp(tf.negative(pos_kg_score))) + tf.log(1+tf.exp(pos_kg_score))
        # kg_loss = tf.reduce_mean(maxi)

        kg_reg_loss = tf.nn.l2_loss(self.re_h_e) + tf.nn.l2_loss(self.re_pos_t_e) + \
                      tf.nn.l2_loss(self.im_h_e) + tf.nn.l2_loss(self.im_pos_t_e) + \
                      tf.nn.l2_loss(self.re_r_e) + tf.nn.l2_loss(self.im_r_e) + \
                      tf.nn.l2_loss(self.re_neg_t_e) + tf.nn.l2_loss(self.im_neg_t_e)
        kg_reg_loss = kg_reg_loss / self.batch_size

        self.kge_loss2 = kg_loss
        self.reg_loss2 = self.regs[1] * kg_reg_loss
        self.loss2 = self.kge_loss2 + self.reg_loss2

        # Optimization process.
        self.opt2 = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2)

    def _build_total_loss(self):
        self.total_loss = self.loss + self.loss2
        # self.total_loss = self.loss
        self.total_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.total_loss)

    def _create_bi_interaction_embed5(self):
        drug_embedding = tf.concat([self.weights['re_drug_embed'], self.weights['im_drug_embed']], axis=1)
        entity_embedding = tf.concat([self.weights['re_entity_embed'], self.weights['im_entity_embed']], axis=1)

        ego_embeddings = tf.concat([drug_embedding, entity_embedding], axis=0)
        all_embeddings = [ego_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)

        da_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_drugs, self.n_entities], 0)
        return da_embeddings, ea_embeddings

    def _create_bi_interaction_embed(self):
        pi = 3.14159265358979323846
        # 公式（3）
        # F(edj）1710,128
        drug_embedding1 = tf.concat([self.weights['re_drug_embed'], self.weights['im_drug_embed']], axis=1)
        entity_embedding1 = tf.concat([self.weights['re_entity_embed'], self.weights['im_entity_embed']], axis=1)
#1710drug_feature.csv文件第一列为id，第二列为我希望得到drug_embedding，格式为Tensor("concat:0", shape=(1710, 128), dtype=float32)它的
        # 读取 CSV 文件
        df = pd.read_csv("1710drug_feature.csv")

        # 提取第二列的内容并拆分为列表
        features = df['feature'].apply(lambda x: [float(val[:-1]) for val in x[1:-1].split(',')]).tolist()
        # 将列表转换为 TensorFlow 张量
        features_tensor = tf.constant(features, dtype=tf.float32)

        # 获取前64维的内容
        drug_embedding = tf.slice(features_tensor, [0, 0], [1710, 64])

        # # 在后64维填充0
        # padding = tf.zeros([1710, 64], dtype=tf.float32)
        # drug_embedding = tf.concat([drug_embedding, padding], axis=1)
        # 输出第一行内容
        # with tf.Session() as sess:
        #     result = sess.run(drug_embedding[0])
        #     print(result)
        entity_embedding = drug_embedding
        # 使用R-GCN方式定义关系矩阵\获得Mr
        relation_embedding = []
        for i in range(self.n_relations):
            weights = tf.reshape(self.weights['alpha'][i], [-1, 1, 1])
            relation_matrix_temp = self.weights['relation_matrix'] * weights
            relation_matrix_temp = tf.reduce_sum(relation_matrix_temp, axis=0)
            # 128,128
            relation_embedding.append(relation_matrix_temp)

        drug_neigh, entity_neigh = [], []
        for i in range(self.n_relations):
            # print(i)
            # edj，r=Mr*f  (1710,128)=(1710,128)*(128,128)
            r_entity_embedding = entity_embedding @ relation_embedding[i]
            # adj*dj，r    +entity_embedding公式3              <tf.Variable 'relation_e_att:0' shape=(86, 1710, 1) dtype=float32_ref>
            weight_entity_embedding = r_entity_embedding * self.weights['relation_e_att'][i]
            weight_entity_embedding = weight_entity_embedding + entity_embedding

            weight_entity_embedding = weight_entity_embedding
            # 接着，将加权的实体嵌入与原始实体嵌入相加，得到最终的实体邻居嵌入relation_u_neigh。
            # (1710,128):        86:(1710*1710)    *     (1710,128)
            relation_u_neigh = tf.sparse.sparse_dense_matmul(self.sparse_adj_list[i], weight_entity_embedding)
            # 计算了药物嵌入与关系嵌入的乘积r_drug_embedding。然后，将其与关系权重self.weights['relation_d_att'][i]相乘，得到加权的药物嵌入weight_drug_embedding。最后，将加权的药物嵌入赋值给weight_drug_embedding。
            r_drug_embedding = drug_embedding @ relation_embedding[i]
            weight_drug_embedding = r_drug_embedding * self.weights['relation_d_att'][i]
            # weight_drug_embedding = weight_drug_embedding + drug_embedding
            # weight_drug_embedding =  drug_embedding
            weight_drug_embedding = weight_drug_embedding
            relation_e_neigh = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(self.sparse_adj_list[i]),
                                                             weight_drug_embedding)

            # 最后，将计算得到的实体邻居嵌入relation_u_neigh添加到drug_neigh列表中，将药物邻居嵌入relation_e_neigh添加到entity_neigh列表中
            drug_neigh.append(relation_u_neigh)
            entity_neigh.append(relation_e_neigh)

        # drug_neigh = self.weights['att2'] * drug_neigh
        # entity_neigh = self.weights['att2'] * entity_neigh
        # 对drug_neigh沿着第一个维度进行求和操作(1710,128)
        drug_neigh = tf.reduce_sum(drug_neigh, 0)
        entity_neigh = tf.reduce_sum(entity_neigh, 0)
        # drug_neigh = drug_neigh + self.weights['att1'] * drug_embedding
        # entity_neigh = entity_neigh + self.weights['att1'] * entity_embedding

        333############################################################################
        ################新增########################################3333333333
        neigh_embed = tf.concat([drug_neigh, entity_neigh], axis=0)
        # (3420,128)
        ego_embeddings = tf.concat([drug_embedding1, entity_embedding1], axis=0)
        sa=neigh_embed
        ro=ego_embeddings
        # 假设x是一个TensorFlow张量
        init = tf.global_variables_initializer()
        data_array = self.data_array
        # 创建会话
        with tf.Session() as sess:
            sess.run(init)
            # 将张量转换为NumPy数组
            sa = sess.run(sa)
            ro = sess.run(ro)
            # data_array=sess.run(data_array)
        sa = torch.tensor(sa)
        ro = torch.tensor(ro)
        # data_array = torch.tensor(data_array)
        #################3gpu。############
        # sa = sa.to('cuda')
        # ro = ro.to('cuda')
        # data_array = data_array.to('cuda')
        n_topo_feats = 64
        n_hid = 128
        n_out_feat = 100

        model0 = Model0(1710, n_topo_feats, n_hid, n_out_feat,128)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        D_t2a = Discriminator(128, 1)
        D_a2t = Discriminator(n_topo_feats, 1)

        model0.to(device)
        D_t2a.to(device)
        D_a2t.to(device)
        # 二分类交叉熵损失函数 criterion
        criterion = nn.BCELoss().to(device)
        # 使用了 RMSprop 优化算法来定义三个不同模型的优化器 trainer_E、trainer_D_t2a 和 trainer_D_a2t，并设置学习率为 1e-3。
        trainer_E = torch.optim.RMSprop(model0.parameters(), lr=1e-3)
        trainer_D_t2a = torch.optim.RMSprop(D_t2a.parameters(), lr=1e-3)
        trainer_D_a2t = torch.optim.RMSprop(D_a2t.parameters(), lr=1e-3)
        # #############################新增###########################333333

        false_ro, false_sa ,MLP_sa,MLP_ro,Dgan_loss,output= update_E(model0, sa, ro, data_array ,D_t2a, D_a2t, criterion, trainer_E, device)

        update_D_t2a(model0, sa, ro,data_array, D_t2a, trainer_D_t2a, device)
        update_D_a2t(model0, sa, ro,data_array, D_a2t, trainer_D_a2t, device)

        # Dgan_loss = Dgan_loss.detach().numpy()
        #
        # # 将 NumPy 数组转换为 TensorFlow 类型张量
        # self.Dgan_loss = tf.convert_to_tensor(Dgan_loss, dtype=tf.float32)
        self.Dgan_loss=Dgan_loss
        self.output=output
        #######################################################################3

        # false = tf.concat([false_ro, false_sa], 1)


        # 创建会话
        # with tf.Session() as sess:
        #     sess.run(init)
        #     # 将张量转换为NumPy数组
        #     neigh_embed = sess.run(neigh_embed)
        #     ego_embeddings = sess.run(ego_embeddings)
        # print(type(neigh_embed))
        # print(type(false_sa))
        # neigh_embed = torch.tensor(neigh_embed)
        # ego_embeddings = torch.tensor(ego_embeddings)


        # 进行连接操作

        # neigh_embed = tf.concat([neigh_embed, false_sa], 1)
        # ego_embeddings = tf.concat([ego_embeddings, false_ro], 1)

        all_embeddings = []

        # side_embeddings = tf.nn.l2_normalize(neigh_embed, dim=1)
        side_embeddings = neigh_embed
        # (3420,256)

        side_embeddings0  = tf.concat([ego_embeddings, side_embeddings], 1)

        MLP_sa = MLP_sa.detach().cpu().numpy()
        MLP_sa = tf.convert_to_tensor(MLP_sa)
        MLP_ro = MLP_ro.detach().cpu().numpy()
        MLP_ro = tf.convert_to_tensor(MLP_ro)
        side_embeddings0 = tf.concat([sa, ro], 1)
        side_embeddings1  = tf.concat([MLP_ro,MLP_sa], 1)
        # side_embeddings01=tf.concat([side_embeddings0,side_embeddings1], 1)
        side_embeddings01 = side_embeddings0
        # side_embeddings01 = side_embeddings0 + self.weights['att5'] * side_embeddings1
        # (3420,100)=relu(((3420,256)*((256,100)+(1,100)))
        pre_embeddings = tf.nn.relu(
            tf.matmul(side_embeddings01, self.weights['W_mlp_0']) + self.weights['b_mlp_0'])
        pre_embeddings = tf.nn.dropout(pre_embeddings, 1 - self.mess_dropout[0])


        # normalize the distribution of embeddings.
        # norm_embeddings = tf.nn.l2_normalize(pre_embeddings, dim=1)
        all_embeddings += [pre_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)

        da_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_drugs, self.n_entities], 0)
        return da_embeddings, ea_embeddings, drug_neigh, drug_embedding1

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        if len(coo.data) == 0:
            return tf.SparseTensor([[1, 2]], [0.], coo.shape)  # 生成空的稀疏矩阵
        indices = np.mat([coo.row, coo.col]).transpose()
        sparse_result = tf.SparseTensor(indices, coo.data, coo.shape)
        return tf.sparse_reorder(sparse_result)  # 重新排序，不然会报--稀疏矩阵乱序错误

    # 统计模型的参数数量，并可选择性地打印出参数数量。
    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    # 执行模型的训练操作，并返回训练过程中的损失和其他相关信息。
    # - self.total_opt：模型的优化操作，用于更新模型的参数。- self.loss：总损失。- self.base_loss：基本损失。- self.kge_loss：知识图嵌入损失。- self.reg_loss：正则化损失。- self.loss2：第二个损失。- self.kge_loss2：第二个知识图嵌入损失。- self.reg_loss2：第二个正则化损失。
    def train(self, sess, feed_dict):
        return sess.run([self.total_opt, self.loss, self.base_loss, self.kge_loss, self.reg_loss, \
                         self.loss2, self.kge_loss2, self.reg_loss2, self.drug_neigh, self.drug_embedding1,self.Dgan_loss], feed_dict)

    def train_A(self, sess, feed_dict):
        return sess.run([self.opt2, self.loss2, self.kge_loss2, self.reg_loss2], feed_dict)

    def eval(self, sess, feed_dict):
        d_e = sess.run(self.d_e, feed_dict)
        pos_e = sess.run(self.pos_e, feed_dict)
        batch_predictions = sess.run(self.batch_predictions, feed_dict)
        return d_e, pos_e, batch_predictions

###################################新增#########################333333333333333333333
def update_E(net_E,sa, ro, data_array,D_t2a, D_a2t, loss, trainer_E, device):

    t2a_mtx, a2t_mtx ,MLP_sa,MLP_ro,output= net_E(sa, ro,data_array)

    # 用于将模型的梯度清零，以便于进行下一次的反向传播和参数更新。 有助于避免梯度累积导致训练不稳定或出现意外的梯度更新。
    trainer_E.zero_grad()

    one = torch.FloatTensor([1]).to(device)
    # Calcute adversarial loss
    mone = (one * -1).to(device)
    # 将模型设置为"评估模式"意味着将模型切换为推理模式，而不是训练模式。
    # 在评估模式下，模型不会进行梯度计算和参数更新，而是专注于对输入数据进行前向传播，生成预测结果。这通常用于在验证集或测试集上评估模型的性能。通过将模型设置为评估模式，可以确保模型在评估过程中不会进行不必要的参数更新，从而保持模型的状态稳定。
    D_a2t.eval()
    D_t2a.eval()
    fake_y_a2t = D_a2t(a2t_mtx)
    fake_y_a2t.backward(one)

    fake_y_t2a = D_t2a(t2a_mtx)
    fake_y_t2a.backward()
    # 使用优化器对网络模型进行参数更新，并清零网络模型的梯度。
    trainer_E.step()
    trainer_E.zero_grad()
    # Calcute prediction loss
    _, _, _, _,y_pred = net_E(sa, ro,data_array)
    # y_pred = y_pred.reshape(-1)
    labels = tf.constant([1, 0] * 1024, dtype=tf.int32)
    labels = tf.expand_dims(tf.cast(labels, tf.float32), axis=1)

    # labels = torch.tensor([1, 0] * 1024)
    # labels = labels.float()
    # 计算二分类交叉熵损失
    model_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=tf.cast(labels, tf.float32)))
    # 定义优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    # 执行反向传播和参数更新
    optimizer.minimize(model_loss)
    # model_loss = loss(y_pred, labels)
    # model_loss.backward(retain_graph=True)

    # trainer_E.step()
    trainer_E.step()
    return t2a_mtx, a2t_mtx,MLP_sa,MLP_ro,model_loss,output




def update_D_t2a(net_E, sa, ro,data_array, D_t2a, trainer_D_t2a, device):
    '''This function mainly used to optimize parameters of discriminator for topology-to-attribute'''

    clamp_lower = -0.01
    clamp_upper = 0.01
    # Perform gradient clip
    for p in D_t2a.parameters():
        p.data.clamp_(clamp_lower, clamp_upper)
    trainer_D_t2a.zero_grad()

    one = torch.FloatTensor([1]).to(device)
    mone = (one * -1).to(device)
    net_E.eval()
    t2a_mtx, _ , _ , _,_ = net_E(sa, ro,data_array)
    fake_y = D_t2a(t2a_mtx)
    fake_y.backward(mone)

    real_y = D_t2a(ro)
    real_y.backward(one)

    trainer_D_t2a.step()
    return


def update_D_a2t(net_E, sa, ro,data_array, D_a2t, trainer_D_a2t, device):

    clamp_lower = -0.01
    clamp_upper = 0.01
    # Perform gradient clip
    for p in D_a2t.parameters():
        p.data.clamp_(clamp_lower, clamp_upper)

    trainer_D_a2t.zero_grad()

    one = torch.FloatTensor([1]).to(device)
    mone = (one * -1).to(device)
    net_E.eval()
    _,a2t_mtx , _ , _,_  = net_E(sa, ro,data_array)
    fake_y = D_a2t(a2t_mtx)

    fake_y.backward(mone)
    real_y = D_a2t(sa)
    real_y.backward(one)

    trainer_D_a2t.step()
    return
#################################新增#########################333333333333333333333