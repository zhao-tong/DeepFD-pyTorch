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
    def __init__(self, feat_size, emb_size):
        super(DeepFD, self).__init__()

        self.fc1 = nn.Linear(feat_size, emb_size)
        self.fc2 = nn.Linear(emb_size, feat_size)

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                # initialize all bias as zeros
                nn.init.constant_(param, 0.0)

    def forward(self, feats):
        embs = F.relu_(self.fc1(feats))
        recon = F.relu_(self.fc2(embs))
        return embs, recon

class Loss_DeepFD():
    def __init__(self, features, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        self.features = features
        self.node_pairs = {}
        self.original_nodes_batch = None

    def extend_nodes(nodes_batch, training_cps):
        self.original_nodes_batch = copy.deepcopy(nodes_batch)
        self.node_pairs = {}
        extended_nodes_batch = set(nodes_batch)

        for node in nodes_batch:
            cps = training_cps[node]
            self.node_pairs[node] = cps
            for cp in cps:
                extended_nodes_batch.add(cp[1])
        extended_nodes_batch = list(extended_nodes_batch)
        return extended_nodes_batch

    def get_loss(nodes_batch, embs_batch, recon_batch, params):

        loss_recon = self.get_loss_recons(nodes_batch, recon_batch)
        loss_simi = self.get_loss_simi(embs_batch)
        loss_reg = self.get_loss_reg(params)

        loss = loss_recon + self.alpha * loss_simi + self.gamma * loss_reg

    def get_loss_simi(self, embs_batch):
        for node, cps in self.node_pairs.items():
            pass

    def get_loss_recon(self, nodes_batch, recon_batch):
        feats_batch = self.features[nodes_batch]
        pass

    def get_loss_reg(self, params):
        pass