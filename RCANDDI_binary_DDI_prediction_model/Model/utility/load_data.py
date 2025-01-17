import collections
import numpy as np
import random as rd
from collections import defaultdict


class Data(object):
    def __init__(self, args, path):
        self.path = path
        self.args = args

        self.batch_size = args.batch_size

        train_file = path + '/10_fold/train0.txt'
        test_file = path + '/10_fold/test0.txt'

        # kg_file = path + '/data/kg_final_1_t.txt'

        if args.dataset == 'deepddi_data':
            npz_file = path + '/deepddi.npz'
        else:
            npz_file = path + '/1317_drug_v4.npz'
        # npz_file='../Data/deepddi_data/deepddi.npz'
        data = np.load(npz_file, allow_pickle=True)
        data = data['data'].item()
        ##########################新增##取出drug_id和Adj_multi######################################
        # # 从data字典中获取Drug_id和Adj_multi的值
        # drug_id = data['Drug_id']
        # adj_multi = data['Adj_multi2']
        #
        # # 将Drug_id的值转换为字符串类型
        # drug_id = [str(x) for x in drug_id]
        #
        # # 将Adj_multi的值转换为浮点数类型
        # adj_multi = adj_multi.astype(float)
        #
        # # 将Drug_id的值保存到Drug_id.tab文件中
        # with open('Drug_id_1317.tab', 'w') as f:
        #     f.write('\n'.join(drug_id))
        # #
        # # 将Adj_multi的值保存到Adj_multi.tab文件中
        # np.savetxt('Adj_multi_1317.tab', adj_multi, delimiter='\t', fmt='%.6f')
        ##############################################################################33
        self.adj_multi = data['Adj_multi'].astype(np.int32)
        self.type_num = 86
        # self.df_mat = data['interaction_feature']#1316:638
        # ----------get number of drugs and items & then load rating data from train_file & test_file------------.
        self.n_train, self.n_test = 0, 0
        # train_file='../Data/deepddi_data/10_fold/train0.txt'
        self.train_data, self.train_drug_dict = self._load_ratings(train_file)
        self.test_data, self.test_drug_dict = self._load_ratings(test_file)
        self.exist_drugs = self.train_drug_dict.keys()
        self.train_dict, self.test_dict = self.get_train_test_dict(train_file, test_file)

        self._statistic_ratings()
        self.train_neg_dict, self.test_neg_dict = self.get_neg_sample()
        # ----------get number of entities and relations & then load kg data from kg_file ------------.
        # 获取实体和关系的数量 & 然后从kg_file加载kg数据
        self.n_entities = self.n_drugs

        self.batch_size_kg = 300
        self._print_data_info()
        # self.ratio_random = args.ratio_random

    def get_neg_sample(self):
        adj = np.copy(self.adj_multi)
        # 通过条件i > j来确定只对矩阵的下三角部分进行操作，下三角设置为-1，以避免重复设置相同的值。
        for i in range(len(self.adj_multi)):
            for j in range(len(self.adj_multi)):
                if i > j:
                    adj[i, j] = -1
        # 1. 这行代码通过使用np.where()函数找到adj矩阵中值为0的位置，并将这些位置转置为一个列表。这个列表neg_data包含了adj矩阵中值为0的位置的坐标。
        neg_data = np.transpose(np.where(adj == 0)).tolist()
        # 2.这行代码计算了总的负样本数量。根据代码中的注释，self.n_train表示训练样本的数量，self.n_test表示测试样本的数量。通过将训练样本数量和测试样本数量相加，然后除以2，得到了总的负样本数量。
        total_neg = (self.n_train + self.n_test) // 2
        # 3.函数从neg_data列表中随机选择total_neg个样本，并将选择的样本存储在neg_sample列表中。这样可以得到一个随机的负样本集合。
        neg_sample = rd.sample(neg_data, total_neg)
        # 4.这行代码从neg_sample列表中选择前self.n_train // 2 个样本作为训练集的负样本。根据代码中的注释，self.n_train表示训练样本的数量。所以这行代码选择了前一半的负样本作为训练集的负样本。
        train_neg_sample = neg_sample[:(self.n_train // 2)]
        # 5.这行代码从neg_sample列表中选择从self.n_train // 2 开始的剩余样本作为测试集的负样本。根据代码中的注释，self.n_train表示训练样本的数量。所以这行代码选择了从训练集之后的样本作为测试集的负样本。
        test_neg_sample = neg_sample[(self.n_train // 2):]
        # 这段代码的目的是将训练集和测试集的负样本中的节点信息存储在字典中，929：。。。。
        # 生成字典
        train_neg_dict, test_neg_dict = defaultdict(list), defaultdict(list)
        for p in train_neg_sample:
            train_neg_dict[p[0]].append(p[1])
            train_neg_dict[p[1]].append(p[0])
        for p in test_neg_sample:
            test_neg_dict[p[0]].append(p[1])
            test_neg_dict[p[1]].append(p[0])

        return train_neg_dict, test_neg_dict

    def get_train_test_dict(self, train_file, test_file):
        train_dict, test_dict = defaultdict(list), defaultdict(list)
        with open(train_file, 'r') as f1, open(test_file, 'r') as f2:
            line1 = f1.readline()
            while line1:
                line1 = line1.strip()  # 移除字符串两端的空白字
                line1 = list(map(int, line1.split(' ')))
                train_dict[line1[0]] = line1[1:]
                line1 = f1.readline()
            line2 = f2.readline()
            while line2:
                line2 = line2.strip()  # 移除字符串两端的空白字
                line2 = list(map(int, line2.split(' ')))
                test_dict[line2[0]] = line2[1:]
                line2 = f2.readline()
        return train_dict, test_dict

    # reading train & test interaction data.
    def _load_ratings(self, file_name):
        drug_dict = dict()
        inter_mat = list()

        lines = open(file_name, 'r').readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(' ')]

            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))

            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])

            if len(pos_ids) > 0:
                drug_dict[u_id] = pos_ids
        return np.array(inter_mat), drug_dict

    def _statistic_ratings(self):
        self.n_drugs = self.adj_multi.shape[0]
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)

    def _print_data_info(self):
        print('[n_drugs]=[%d]' % (self.n_drugs))
        print('[n_train, n_test]=[%d, %d]' % (self.n_train, self.n_test))
        print('[batch_size, batch_size_kg]=[%d, %d]' % (self.batch_size, self.batch_size_kg))

    def _generate_train_cf_batch(self):
        if self.batch_size <= self.n_drugs:
            # 从self.exist_drugs中随机选择self.batch_size个药物
            drugs = rd.sample(self.exist_drugs, self.batch_size)
        else:
            # 如果self.exist_drugs中的药物数量不足以满足self.batch_size，则会重复选择药物。
            drugs = [rd.choice(list(self.exist_drugs)) for _ in range(self.batch_size)]

        # 从给定药物的正样本列表中随机选择一个正样本，并将其作为列表返回
        def sample_pos_sample_for_d(d, num):
            pos_items = self.train_drug_dict[d]
            n_pos_items = len(pos_items)
            # pos_batch = []
            pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            return [pos_items[pos_id]]

        # 从给定药物的负样本列表中随机选择一个负样本，
        def sample_neg_sample_for_d(d, num):
            neg_items = []
            neg_i_id = rd.sample(self.train_neg_dict[d], 1)[0]
            neg_items.append(neg_i_id)
            return neg_items

        # def sample_neg_sample_for_d(d, num):
        #     neg_items = []
        #     while True:
        #         neg_i_id = np.random.randint(low=0, high=self.n_drugs,size=1)[0]#随机采取负样本
        #         if neg_i_id not in self.train_drug_dict[d] and neg_i_id not in neg_items:
        #             neg_items.append(neg_i_id)
        #             break
        #         # if len(neg_items) == num: break
        #     return neg_items

        pos_items, neg_items, relations = [], [], []
        pos_tails, neg_tails = [], []
        # length = [i for i in range(len(unsim_matrix))]
        for d in drugs:
            # 正样本
            pos_items += sample_pos_sample_for_d(d, 1)
            # 负样本
            neg_items += sample_neg_sample_for_d(d, 1)
            # 关系矩阵d行列表最后一个元素列对应内容-1
            relations.append(self.adj_multi[d, pos_items[-1]] - 1)
            # 尾实体=药物2+1710
            pos_tails.append(pos_items[-1] + self.n_drugs)
            neg_tails.append(neg_items[-1] + self.n_drugs)

        return drugs, pos_items, neg_items, relations, pos_tails, neg_tails