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