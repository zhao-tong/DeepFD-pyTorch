import os
import sys
import json
import math
import torch
import pickle
import random
import logging
import logging.config

import numpy as np
import torch.nn as nn

from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.metrics import *

def getLogger(name, out_path, config_dir):
    config_dict = json.load(open(config_dir + '/log_config.json'))

    config_dict['handlers']['file_handler']['filename'] = f'{out_path}/log-{name}.txt'
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger

def get_simi_single_iter(params):
    entries_batch, feats = params
    ii, jj = entries_batch.T
    simi = []
    for x in range(len(ii)):
        simi.append(get_simi(feats[ii[x]].toarray(), feats[jj[x]].toarray()))
    simi = np.asarray(simi)
    assert np.shape(ii) == np.shape(jj) == np.shape(simi)
    return ii, jj, simi

def get_simi(u1, u2):
    nz_u1 = u1.nonzero()[1]
    nz_u2 = u2.nonzero()[1]
    nz_inter = np.array(list(set(nz_u1) & set(nz_u2)))
    nz_union = np.array(list(set(nz_u1) | set(nz_u2)))
    if len(nz_inter) == 0:
        simi_score = 1 / (len(nz_union) + len(u1))
    elif len(nz_inter) == len(nz_union):
        simi_score = (len(nz_union) + len(u1) - 1) / (len(nz_union) + len(u1))
    else:
        simi_score = len(nz_inter) / len(nz_union)
    return float(simi_score)

def train_classification(Dl, args, logger, deepFD, device, max_vali_f1, epoch):
    pass

def train_model(Dl, args, logger, deepFD, model_loss, device, epoch):
    train_nodes = getattr(Dl, Dl.ds+'_train')
    np.random.shuffle(train_nodes)

    params = []
    for param in deepFD.parameters():
        if param.requires_grad:
            params.append(param)
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.gamma)
    optimizer.zero_grad()
    deepFD.zero_grad()

    batches = math.ceil(len(train_nodes) / args.b_sz)
    visited_nodes = set()
    training_cps = Dl.get_train()
    logger.info('sampled pos and neg nodes for each node in this epoch.')
    for index in range(batches):
        nodes_batch = train_nodes[index*args.b_sz:(index+1)*args.b_sz]
        nodes_batch = np.asarray(model_loss.extend_nodes(nodes_batch, training_cps))
        visited_nodes |= set(nodes_batch)

        embs_batch, recon_batch = deepFD(nodes_batch)
        loss = model_loss.get_loss(nodes_batch, embs_batch, recon_batch)

        logger.info(f'EP[{epoch}], Batch [{index+1}/{batches}], Loss: {loss.item():.4f}, Dealed Nodes [{len(visited_nodes)}/{len(train_nodes)}]')
        loss.backward()

        nn.utils.clip_grad_norm_(deepFD.parameters(), 5)
        optimizer.step()

        optimizer.zero_grad()
        deepFD.zero_grad()

    return deepFD