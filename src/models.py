import os
import sys
import copy
import torch
import random
import numpy as np
from scipy.sparse import csr_matrix

import torch.nn as nn
import torch.nn.functional as F

class DeepFD(nn.Module):
    def __init__(self, features, feat_size, hidden_size, emb_size):
        super(DeepFD, self).__init__()
        self.features = features

        self.fc1 = nn.Linear(feat_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, emb_size)
        self.fc3 = nn.Linear(emb_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, feat_size)

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                # initialize all bias as zeros
                nn.init.constant_(param, 0.0)

    def forward(self, nodes_batch):
        feats = self.features[nodes_batch]
        x_en = F.relu_(self.fc1(feats))
        embs = F.relu_(self.fc2(x_en))
        x_de = F.relu_(self.fc3(embs))
        recon = F.relu_(self.fc4(x_de))
        return embs, recon

class Loss_DeepFD():
    def __init__(self, features, graph_simi, device, alpha, beta, gamma):
        self.features = features
        self.graph_simi = graph_simi
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.node_pairs = {}
        self.original_nodes_batch = None
        self.extended_nodes_batch = None

    def extend_nodes(self, nodes_batch, training_cps):
        self.original_nodes_batch = copy.deepcopy(nodes_batch)
        self.node_pairs = {}
        self.extended_nodes_batch = set(nodes_batch)

        for node in nodes_batch:
            cps = training_cps[node]
            self.node_pairs[node] = cps
            for cp in cps:
                self.extended_nodes_batch.add(cp[1])
        self.extended_nodes_batch = list(self.extended_nodes_batch)
        return self.extended_nodes_batch

    def get_loss(self, nodes_batch, embs_batch, recon_batch):
        # calculate loss_simi and loss+recon,
        # loss_reg is included in SGD optimizer as weight_decay
        loss_recon = self.get_loss_recon(nodes_batch, recon_batch)
        loss_simi = self.get_loss_simi(embs_batch)
        loss = loss_recon + self.alpha * loss_simi
        return loss

    def get_loss_simi(self, embs_batch):
        node2index = {n:i for i,n in enumerate(self.extended_nodes_batch)}
        simi_feat = []
        simi_embs = []
        for node, cps in self.node_pairs.items():
            for i, j in cps:
                simi_feat.append(torch.FloatTensor([self.graph_simi[i, j]]))
                dis_ij = (embs_batch[node2index[i]] - embs_batch[node2index[j]]) ** 2
                dis_ij = torch.exp(-dis_ij.sum())
                simi_embs.append(dis_ij.view(1))
        simi_feat = torch.cat(simi_feat, 0).to(self.device)
        simi_embs = torch.cat(simi_embs, 0)
        L = simi_feat * ((simi_embs - simi_feat) ** 2)
        return L.mean()

    def get_loss_recon(self, nodes_batch, recon_batch):
        feats_batch = self.features[nodes_batch]
        H_batch = (feats_batch * (self.beta - 1)) + 1
        assert feats_batch.size() == recon_batch.size() == H_batch.size()
        L = ((recon_batch - feats_batch) * H_batch) ** 2
        return L.mean()