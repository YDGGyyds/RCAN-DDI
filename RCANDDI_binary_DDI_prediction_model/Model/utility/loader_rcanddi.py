import numpy as np
from .load_data import Data
from time import time
import scipy.sparse as sp
import random as rd
import collections
import bisect
import tensorflow as tf
class RCANDDI_loader(Data):
    def __init__(self, args, path):
        super().__init__(args, path)
        # self.w_sp_matrix = self._get_w_sp_matrix()
        # generate the sparse adjacency matrices for drug-item interaction & relational kg data.
        #生成药物相互作用稀疏邻接矩阵 和 关系知识图谱数据
        self.adj_list, self.adj_r_list = self._get_relational_adj_list()
        self.n_relations = 86

        # generate the sparse laplacian matrices.
        #生成稀疏拉普拉斯矩阵
        self.lap_list = self._get_relational_lap_list()

        # generate the triples dictionary, key is 'head', value is '(tail, relation)'.
        #生成三元组字典，键为“head”，值为“（tail，relation）”。
        self.all_kg_dict,self.relation_relate_eneity = self._get_all_kg_dict()

        self.all_h_list, self.all_r_list, self.all_t_list = self._get_all_kg_data()

        self.sparse_adj_list = self._get_sparse_adj_list()

    def _get_sparse_adj_list(self):
        all_h_list = np.array(self.all_h_list)
        all_r_list = np.array(self.all_r_list)
        all_t_list = np.array(self.all_t_list)
        sparse_adj_list = []
        adj_size = self.n_entities+self.n_drugs
        #使用collections.Counter()函数统计了all_h_list中每个元素的出现次数，得到了一个计数器对象degree，用于记录每个药物的度数
        #按度降序排序
        degree = collections.Counter(all_h_list)
        for i in range(max(self.all_r_list)+1):
            r_position = np.where(all_r_list==i)
            h_position = all_h_list[r_position]
            t_position = all_t_list[r_position]
            h_position1, t_position1= [],[]
            values = []
            for h,t in zip(h_position,t_position):
                #对于每个关系r，成对的头实体和尾实体%1710
                h_position1.append(h%self.n_drugs)
                t_position1.append(t%self.n_drugs)
                #权重值=1/度数
                values.append(1/degree[h_position1[-1]])
            #COO格式的稀疏矩阵，，使用np.array(values)/2作为数据参数，(h_position1, t_position1)作为行索引和列索引参数，shape=(self.n_drugs, adj_size-self.n_drugs)指定矩阵的形状。
            sparse_adj_list.append(sp.coo_matrix((np.array(values)/2, (h_position1, t_position1)), shape=(self.n_drugs, adj_size-self.n_drugs)))
        return sparse_adj_list

    # def _get_w_sp_matrix(self):
    #     train_data = self.train_data
    #     train_row = train_data[:,0]
    #     train_col = train_data[:,1]
    #     values = np.array([1 for i in range(len(train_row))])
    #     #药物个数，特征个数
    #     n_drug = self.n_drugs
    #     n_feature = 0
    #     #先转稀疏矩阵
    #     ddi_sp = sp.coo_matrix((values,(train_row,train_col)),shape=(n_drug,n_drug))
    #     ddi_sp1 = sp.coo_matrix((values,(train_col,train_row)),shape=(n_drug,n_drug))
    #     #ddi的行列
    #     ddi_row = ddi_sp.row
    #     ddi_col = ddi_sp.col+n_drug
    #     ddi_row1 = ddi_sp1.row+n_drug
    #     ddi_col1 = ddi_sp1.col
    #     #最终的行列
    #     rows =  np.concatenate((ddi_row,ddi_row1))
    #     cols =  np.concatenate((ddi_col,ddi_col1))
    #     values = np.concatenate((ddi_sp.data,ddi_sp.data))
    #     #
    #     w_sp_matrix = sp.coo_matrix((values, (rows, cols)), shape=(n_drug*2 , n_drug*2))
    #     return w_sp_matrix


    def _get_relational_adj_list(self):
        t1 = time()
        #存储生成的稀疏邻接矩阵和关系列表
        adj_mat_list = []
        adj_r_list = []
        #用于将NumPy数组转换为稀疏邻接矩阵。它接受三个参数：np_mat表示输入的NumPy数组，row_pre和col_pre表示行和列的偏移量。
        def _np_mat2sp_adj(np_mat, row_pre, col_pre):
            n_all = self.n_drugs + self.n_entities
            # single-direction、
            #它首先计算稀疏矩阵的行、列和值
            a_rows = np_mat[:, 0] + row_pre
            a_cols = np_mat[:, 1] + col_pre
            #a_vals是一个列表，其中的每个元素都是浮点数1.0。列表的长度由a_rows的长度决定。
            a_vals = [1.] * len(a_rows)

            # b_rows = a_cols
            # b_cols = a_rows
            # b_vals = [1.] * len(b_rows)
            #然后使用sp.coo_matrix()函数创建稀疏邻接矩阵，并返回该矩阵。
            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            # b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))


            return a_adj
        #，用于将NumPy数组转换为稀疏邻接矩阵。它的功能与_np_mat2sp_adj()类似，但它还处理了特殊情况，通过连接行和列的方式生成了两个方向的邻接矩阵。
        def _np_mat2sp_adj_f(np_mat, row_pre, col_pre):
            n_all = self.n_drugs + self.n_entities
            # single-direction
            a_rows = np_mat[:, 0] + row_pre
            a_rows = np.concatenate((a_rows,a_rows-self.n_items))
            a_cols = np_mat[:, 1] + col_pre
            a_cols = np.concatenate((a_cols,a_cols))
            a_vals = [1.] * len(a_rows)

            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)


            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            return a_adj
        #这段代码调用了_np_mat2sp_adj()函数，将self.train_data作为输入，生成了一个稀疏邻接矩阵R。然后，将R添加到adj_mat_list列表中，并将关系索引0添加到adj_r_list列表中。
        R= _np_mat2sp_adj(self.train_data, row_pre=0, col_pre=self.n_drugs)
        adj_mat_list.append(R)
        adj_r_list.append(0)

        self.n_relations = len(adj_r_list)

        return adj_mat_list, adj_r_list

    def _get_relational_lap_list(self):
        def _bi_norm_lap(adj):
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def _si_norm_lap(adj):
            #adj:COO格式是一种常用的稀疏矩阵存储格式，它使用三个数组来表示非零元素的位置和值。（3420，3420）
            #先转化为邻接矩阵，计算每行的和，得到一个行和数组rowsum。（3420，1）
            rowsum = np.array(adj.sum(1))
            #计算rowsum每个元素的倒数，得到一个倒数数组d_inv。（3420，）
            d_inv = np.power(rowsum, -1).flatten()
            #将无穷大（rowsum=0）的元素设置为0。
            d_inv[np.isinf(d_inv)] = 0.
            #构建对角矩阵d_mat_inv，其中对角线元素为d_inv。
            d_mat_inv = sp.diags(d_inv)
            #计算归一化的邻接矩阵norm_adj，通过将d_mat_inv与邻接矩阵adj相乘。
            norm_adj = d_mat_inv.dot(adj)
            #将norm_adj转换为COO格式的稀疏矩阵，并返回
            return norm_adj.tocoo()

        if self.args.adj_type == 'bi':
            lap_list = [_bi_norm_lap(adj) for adj in self.adj_list]
            print('\tgenerate bi-normalized adjacency matrix.')
        else:
            lap_list = []
            for adj in self.adj_list:
                buffer = _si_norm_lap(adj)
                lap_list.append(buffer)
            # lap_list = [_si_norm_lap(adj) for adj in self.adj_list]
            print('\tgenerate si-normalized adjacency matrix.')
        return lap_list

    def  _get_all_kg_dict(self):
        def relation_mapping(head,tail,realtion):
            if head >= self.n_drugs:head -= self.n_drugs
            if tail >= self.n_drugs:tail -= self.n_drugs
            ddi_type = self.adj_multi[head][tail]
            return ddi_type - 1
        #都是从这里采样的，所以在构建all_kg_dict时，要把反应类型考虑进去
        #另外，这里为了进行有针对性的负采样，--》（只选取当前关系下尾部出现的实体作为负样本），所以需要构建一个字典，key为关系，value为头尾部
        all_kg_dict = collections.defaultdict(list)
        relation_relate_eneity = collections.defaultdict(list)
        for l_id, lap in enumerate(self.lap_list):
            rows = lap.row
            cols = lap.col
            for i_id in range(len(rows)):
                head = rows[i_id]
                tail = cols[i_id]
                relation = self.adj_r_list[l_id]
                # if relation == 0 or relation == 8:
                #     #在存在相互作用的情况下，将relation=0转换成对应的类型
                #     relation = relation_mapping(head,tail,relation)
                all_kg_dict[head].append((tail, relation))
                # #这里添加一部分，即D1部分的药物也需要存在属性内容，所以如果head>1316且relation在1-16之内的话
                # if head>1316 and relation>0 and relation<17:
                #      all_kg_dict[head-1317].append((tail, relation))
                relation_relate_eneity[relation].append(tail)
                relation_relate_eneity[relation].append(head)
        for k,v in relation_relate_eneity.items():
            relation_relate_eneity[k] = list(set(v))
        return all_kg_dict,relation_relate_eneity

    def _get_all_kg_data(self):
        #根据索引，从adj_multi获得r
        def relation_mapping(self,head,tail,realtion):
            if head >= self.n_drugs:head -= self.n_drugs
            if tail >= self.n_drugs:tail -= self.n_drugs
            ddi_type = self.adj_multi[head][tail]
            # if ddi_type == 1:return 0
            # else:
            #     return ddi_type + 14
            return ddi_type - 1
        def _reorder_list(org_list, order):
            new_list = np.array(org_list)
            new_list = new_list[order]
            return new_list

        all_h_list, all_t_list, all_r_list = [], [], []
        #将self.lap_list中每个元素的row、col和对应的adj_r_list的值提取出来，并分别添加到all_h_list、all_t_list和all_r_list列表中
        for l_id, lap in enumerate(self.lap_list):
            all_h_list += list(lap.row)#list：345830，self.lap_list第一列0，0，0，0，0，0...1709,1709.。。。
            all_t_list += list(lap.col)#list：345830，self.lap_list第二列3413,3406,3397...1716
            all_r_list += [self.adj_r_list[l_id]] * len(lap.row)#345830个0
        #检查all_h_list列表的长度是否与上述求和结果相等。如果相等，则断言通过，程序继续执行。如果不相等，则断言失败，会引发AssertionError异常。
        assert len(all_h_list) == sum([len(lap.data) for lap in self.lap_list])

        # resort the all_h/t/r/v_list,
        # ... since tensorflow.sparse.softmax requires indices sorted in the canonical lexicographic order
        #由于tensorflow.sparse.softmax需要按规范词典顺序排序的索引，打印排序索引
        print('\treordering indices...')
        org_h_dict = dict()
        #是将all_h_list、all_t_list和all_r_list中的元素重新组织到org_h_dict字典中。
        # org_h_dict字典的键是all_h_list中的元素，对应的值是一个包含两个列表的列表，其中第一个列表存储与键对应的all_t_list元素，第二个列表存储与键对应的all_r_list元素。
        for idx, h in enumerate(all_h_list):
            if h not in org_h_dict.keys():
                org_h_dict[h] = [[],[]]

            org_h_dict[h][0].append(all_t_list[idx])
            org_h_dict[h][1].append(all_r_list[idx])
        print('\treorganize all kg data done.')
        #对org_h_dict字典中的每个键对应的值进行排序，并将排序后的结果存储在sorted_h_dict字典中。
        sorted_h_dict = dict()
        for h in org_h_dict.keys():
            org_t_list, org_r_list = org_h_dict[h]
            sort_t_list = np.array(org_t_list)
            sort_order = np.argsort(sort_t_list)

            sort_t_list = _reorder_list(org_t_list, sort_order)
            sort_r_list = _reorder_list(org_r_list, sort_order)

            sorted_h_dict[h] = [sort_t_list, sort_r_list]
        print('\tsort meta-data done.')

        od = collections.OrderedDict(sorted(sorted_h_dict.items()))
        new_h_list, new_t_list, new_r_list = [], [], []
        #h，排序后的t和r
        for h, vals in od.items():
            new_h_list += [h] * len(vals[0])#345380，[0,0，..0,1,1...，1709]
            new_t_list += list(vals[0])#345380,[1715, 1732, 1760, 1765, 1767, 1783, 1789, 1791, 1800, 1810, 1816, 1820, 1829, 1830, 1845, 1854, 1861, 1868, 1869, 1884, 1889, 1890, 1892, 1896, 1911, 1914, 1921, 1930, 1936, 1943, 1949, 1986, 1990, 1993, 1995, 1998, 2009, 2022, 2027, 2051, 2058, 2069, 2078, 2082, 2086, 2096, 2110, 2112, 2145, 2154, 2158, 2163, 2182, 2183, 2186, 2188, 2195, 2203, 2209, 2213, 2221, 2228, 2236, 2243, 2249, 2258, 2265, 2272, 2273, 2275, 2281, 2292, 2298, 2300, 2306, 2341, 2344, 2366, 2371, 2395, 2401, 2405, 2408, 2412, 2430, 2437, 2439, 2440, 2442, 2455, 2475, 2485, 2504, 2539, 2548, 2549, 2593, 2608, 2637, 2641...
            new_r_list += list(vals[1])#345380,全0


        assert sum(new_h_list) == sum(all_h_list)
        assert sum(new_t_list) == sum(all_t_list)
        assert sum(new_r_list) == sum(all_r_list)
        print('\tsort all data done.')
        new_r_list1 = []
        for h,r,t in zip(new_h_list,new_r_list,new_t_list):
            new_r_list1.append(relation_mapping(self,h,t,r))
        return new_h_list, new_r_list1, new_t_list

    def generate_train_batch(self):
        drugs, pos_items, neg_items,relations,pos_tails, neg_tails = self._generate_train_cf_batch()

        batch_data = {}
        batch_data['drugs'] = drugs
        batch_data['pos_drugs'] = pos_items
        batch_data['neg_drugs'] = neg_items
####################################新增###################################
        data_list = []
        for i in range(len(drugs)):
            data_list.append([drugs[i], pos_items[i]])
            data_list.append([drugs[i], neg_items[i]])

        # 将数据结构转换为 NumPy 数组
        batch_data['data_array'] = np.array(data_list)
        data_array = batch_data['data_array']

        # 保存为 CSV 文件
        np.savetxt('data_array.csv', data_array, delimiter=',',fmt='%d')
        ####################################新增###################################
        batch_data['relations'] = relations
        batch_data['pos_tails'] = pos_tails
        batch_data['neg_tails'] = neg_tails

        return batch_data

    def generate_train_feed_dict(self, model, batch_data):
        feed_dict = {
            model.drugs: batch_data['drugs'],
            model.pos_drugs: batch_data['pos_drugs'],
            model.neg_drugs: batch_data['neg_drugs'],
            # model.all_pro_drug:batch_data['pro_drug'],
            model.data_array: batch_data['data_array'],
            model.h: batch_data['drugs'],
            model.r:batch_data['relations'],
            model.pos_t: batch_data['pos_tails'],
            model.neg_t: batch_data['neg_tails'],

            model.mess_dropout: eval(self.args.mess_dropout),
            # model.node_dropout: eval(self.args.node_dropout),
        }

        return feed_dict

    def generate_test_feed_dict(self, model, drug_batch, item_batch, drop_flag=True):
        h = []
        r = []
        t = []
        for u in drug_batch:
            #将u重复item_batch次（1710）
            h += [u for i in item_batch]
            #将0添加到列表r中
            r += [0 for i in item_batch]
            #将len(item_batch) + i添加到列表t中
            t += [len(item_batch)+i for i in item_batch]

        feed_dict ={
            model.drugs: drug_batch,
            model.h: h,
            model.pos_drugs: item_batch,
            model.pos_t: t,
            model.r : r,
            model.mess_dropout: [0.] * len(eval(self.args.layer_size)),
            # model.node_dropout: [0.] * len(eval(self.args.layer_size)),
        }

        return feed_dict

