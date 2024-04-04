import torch
import torch.nn.functional as F
import numpy as np


def bpr_loss(user_emb, pos_item_emb, neg_item_emb,epoch):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)

    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    # if (epoch>=5):
    #     orgin_tensor = pos_score - neg_score
    #     index_sorted_tensor = torch.argsort(orgin_tensor, 0, descending=True)  # 把数值大的评分放在前边
    #     new_tensor = orgin_tensor[index_sorted_tensor]
    #     # drop_rate = np.linspace(0, 0.1, 50)
    #
    #     number = int(len(new_tensor) * (1.00 - 0.05))
    #     sup_logits = new_tensor[:number]
    # else:
    #     sup_logits=pos_score-neg_score

    #添加去噪函数
    orgin_tensor = pos_score-neg_score
    index_sorted_tensor = torch.argsort(orgin_tensor, 0, descending=True)#把数值大的评分放在前边
    new_tensor = orgin_tensor[index_sorted_tensor]
    # drop_rate = np.linspace(0, 0.1, 50)

    number = int(len(new_tensor) * (1.00 - 0.05))
    sup_logits = new_tensor[:number]
    loss = -torch.log(10e-6 + torch.sigmoid(sup_logits))
    # loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))#10e-5效果还行

    return torch.mean(loss)

def triplet_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = F.relu(neg_score+1-pos_score)
    return torch.mean(loss)

def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)/emb.shape[0]
    return emb_loss * reg


def batch_softmax_loss(user_emb, item_emb, temperature):
    user_emb, item_emb = F.normalize(user_emb, dim=1), F.normalize(item_emb, dim=1)
    pos_score = (user_emb * item_emb).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(user_emb, item_emb.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    loss = -torch.log(pos_score / ttl_score+10e-6)
    return torch.mean(loss)


def InfoNCE(view1, view2, temperature, b_cos = True):
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score+10e-6)  #在这里修改一下损失函数
    return torch.mean(cl_loss)


def kl_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(kl)

def js_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    q = F.softmax(q_logit, dim=-1)
    kl_p = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    kl_q = torch.sum(q * (F.log_softmax(q_logit, dim=-1) - F.log_softmax(p_logit, dim=-1)), 1)
    return torch.mean(kl_p+kl_q)