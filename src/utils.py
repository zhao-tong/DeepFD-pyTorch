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
from numba import guvectorize
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.metrics import f1_score, precision_recall_fscore_support, precision_recall_curve, roc_curve, average_precision_score

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