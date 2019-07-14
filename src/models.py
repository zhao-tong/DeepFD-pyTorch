import os
import sys
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
        recons = F.relu_(self.fc2(embs))
        return embs, recons