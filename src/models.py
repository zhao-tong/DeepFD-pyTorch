import os
import sys
import torch
import random
import numpy as np
from scipy.sparse import csr_matrix

import torch.nn as nn
import torch.nn.functional as F

class DeepFD(nn.Module):

    def __init__(self, emb_size):
        super(Classification, self).__init__()

        self.fc1 = nn.Linear(emb_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                # initialize all bias as zeros
                nn.init.constant_(param, 0.0)

    def forward(self, embeds):
        x = F.elu_(self.fc1(embeds))
        x = F.elu_(self.fc2(x))
        logists = torch.log_softmax(x, 1)
        return logists