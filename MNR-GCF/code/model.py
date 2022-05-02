"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.HeteGraph = self.dataset.getSparseHeteGraph()
        self.HomoGraph = self.dataset.getSparseHomoGraph()
        self.HeteDegreeWeight = self.dataset.getDegreeWeight()
        self.degree_he = self.HeteDegreeWeight
        self.degree_ho = 1 - self.HeteDegreeWeight
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob, G):
        if self.A_split:
            graph = []
            for g in G:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(G, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        embs_he = [all_emb]
        embs_ho = [all_emb]
        if self.config['dropout']:
            if self.training:
                # print("droping")
                he_g_droped = self.__dropout(self.keep_prob, self.HeteGraph)
                ho_g_droped = self.__dropout(self.keep_prob, self.HomoGraph)
            else:
                he_g_droped = self.HeteGraph
                ho_g_droped = self.HomoGraph
        else:
            he_g_droped = self.HeteGraph
            ho_g_droped = self.HomoGraph
        
        weight_ho = self.degree_ho
        weight_he = self.degree_he

        for layer in range(self.n_layers):

            # Homogeneous graph convolution
            if self.A_split:
                temp_emb = []
                for f in range(len(ho_g_droped)):
                    temp_emb.append(torch.sparse.mm(ho_g_droped[f], all_emb))
                side_emb_ho = torch.cat(temp_emb, dim=0)
                # all_emb = side_emb_ho + all_emb
            else:
                side_emb_ho = torch.sparse.mm(ho_g_droped, all_emb)
                # all_emb = side_emb_ho + all_emb

            # Heterogeneous graph convolution
            if self.A_split:
                temp_emb = []
                for f in range(len(he_g_droped)):
                    temp_emb.append(torch.sparse.mm(he_g_droped[f], all_emb))
                side_emb_he = torch.cat(temp_emb, dim=0)
            else:
                side_emb_he = torch.sparse.mm(he_g_droped, all_emb)
            
            all_emb = weight_ho * side_emb_ho + weight_he * side_emb_he

            weight_ho = weight_ho + 0.1 * torch.sum((all_emb * side_emb_ho), dim=1, keepdim=True)
            weight_he = weight_he + 0.1 * torch.sum((all_emb * side_emb_he), dim=1, keepdim=True)
            
            weight_ho = weight_ho / (weight_ho + weight_he)
            weight_he = 1 - weight_ho

            embs.append(all_emb)
            embs_he.append(side_emb_he)
            embs_ho.append(side_emb_ho)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        embs_he = torch.stack(embs_he, dim=1)
        light_out_he = torch.mean(embs_he, dim=1)
        users_he, items_he = torch.split(light_out_he, [self.num_users, self.num_items])

        embs_ho = torch.stack(embs_ho, dim=1)
        light_out_ho = torch.mean(embs_ho, dim=1)
        users_ho, items_ho = torch.split(light_out_ho, [self.num_users, self.num_items])

        return users, items, users_he, items_he, users_ho, items_ho
    
    def getUsersRating(self, users):
        all_users, all_items, users_he, items_he, users_ho, items_ho = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items, users_he, items_he, users_ho, items_ho = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        users_emb_he = users_he[users]
        users_emb_ho = users_ho[users]
        pos_emb_he = items_he[pos_items]
        pos_emb_ho = items_ho[pos_items]
        neg_emb_he = items_he[neg_items]
        neg_emb_ho = items_ho[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, users_emb_he, users_emb_ho, \
               pos_emb_he, pos_emb_ho, neg_emb_he, neg_emb_ho
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0,
        users_emb_he, users_emb_ho,
        pos_emb_he, pos_emb_ho,
        neg_emb_he, neg_emb_ho) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2) +
                         negEmb0.norm(2).pow(2))/float(len(users))
        aux_w = 0.05
        pos_scores = torch.mul(users_emb, pos_emb) + aux_w * torch.mul(users_emb_he, pos_emb_he) + \
                     aux_w * torch.mul(users_emb_ho, pos_emb_ho)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb) + aux_w * torch.mul(users_emb_he, neg_emb_he) + \
                     aux_w * torch.mul(users_emb_ho, neg_emb_ho)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
