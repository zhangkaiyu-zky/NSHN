import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from model.graph.evaluate import *
import random


# Paper: Are graph augmentations necessary? simple graph contrastive learning for recommendation. SIGIR'22


class SimGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SimGCL, self).__init__(conf, training_set, test_set)#代表着用户的训练集和测试集
        args = OptionConf(self.config['SimGCL'])
        self.cl_rate = float(args['-lambda'])
        self.eps = float(args['-eps'])
        self.n_layers = int(args['-n_layer'])
        self.model = SimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers)#self.data点进去详细看一下。

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        recall_list=[]
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch  #正样本的选择还是选择的是交互数据，负样本的选择还是从用户未交互的数据中选择
                #**************************添加边缘丢失


                # 计算要丢弃的元素数量
                # percent_to_remove = 0.1
                # num_to_remove = int(len(user_idx) * percent_to_remove)
                #
                # # 随机选择要删除的元素
                # indices_to_remove = random.sample(range(len(user_idx)), num_to_remove)
                #
                # # 创建一个新用户列表，不包括要删除的元素
                # new_user_idx = [x for i, x in enumerate(user_idx) if i not in indices_to_remove]
                #
                # # 创建一个新项目列表，不包括要删除的元素
                # percent_to_remove = 0.1
                # num_to_remove1 = int(len(pos_idx) * percent_to_remove)
                #
                # # 随机选择要删除的元素
                # indices_to_remove = random.sample(range(len(pos_idx)), num_to_remove1)
                #
                # # 创建一个新用户列表，不包括要删除的元素
                # new_pos_idx = [x for i, x in enumerate(pos_idx) if i not in indices_to_remove]

                # print(new_user_idx )

                #**************************
                rec_user_emb, rec_item_emb,neg_embeddings = model()#要注意，这里并没有添加噪声，二十按照用户的正常嵌入进行选择，只有在对比损失函数InfoNCE中使用了


                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], neg_embeddings[neg_idx]
                #user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx],rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb,epoch)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx])
                # cl_loss = self.cl_rate * self.cal_cl_loss([new_user_idx, new_pos_idx])
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss  #这里l2正则没有添加负样本的嵌入，可以试一下添加负样本
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb,_ = self.model()
            #添加一个列表用来接收返回的各项指标
            list_measure=[]
            list_measure = self.fast_evaluation(epoch)
            num_list = [float(s.split(':')[-1]) for s in list_measure]
            recall_list.append(num_list[2])
            # print("这是用于可视化消融实验的评估指标")
            # print(list_measure)
        # print(recall_list)  #这个就是执行全部epoch后的recall值，用于消融实验。
        #把生成recall值的列表添加到文本中
        with open('recall_all_SimGCL_douban.txt', 'a') as file:
            file.write(str(recall_list) + '\n')

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb


    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        #添加边缘丢失用于对比学习


        user_view_1, item_view_1,_ = self.model(perturbed=True)#负样本也要输出
        user_view_2, item_view_2,_ = self.model(perturbed=True)
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], 0.2)

        return user_cl_loss + item_cl_loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb,_ = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()
    #添加




class SimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers):
        super(SimGCL_Encoder, self).__init__()
        self.data = data
        print("输出data数据")
        print(data)
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)#ego_embeddings:4119*64
        all_embeddings = []
        neg_embeddings=[]
        sum_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)#4119,4119
            all_embeddings.append(ego_embeddings)
            #"添加选择负样本的代码"

            if k==2:
                # 给负样本添加噪声
                # random_noise = torch.rand_like(ego_embeddings).cuda()
                # ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
                #结束
                neg_user_embeddings, neg_item_embeddings = torch.split(ego_embeddings, [self.data.user_num, self.data.item_num])
                neg_embeddings = neg_item_embeddings
                #print(neg_embeddings.shape)
            #结束
            else:
                if perturbed:
                    random_noise = torch.rand_like(ego_embeddings).cuda()
                    ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
                all_embeddings.append(ego_embeddings)#这种是三层GCN
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        # sum_embeddings = torch.stack(sum_embeddings, dim=1)
        # sum_embeddings = torch.mean(sum_embeddings, dim=1)

        # neg_user_embeddings, neg_item_embeddings = torch.split(sum_embeddings, [self.data.user_num, self.data.item_num])
        # neg_embeddings = neg_item_embeddings

        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        # neg_embeddings=item_all_embeddings
        #print(user_all_embeddings.shape)

        #添加边缘丢失

        return user_all_embeddings, item_all_embeddings,neg_embeddings
