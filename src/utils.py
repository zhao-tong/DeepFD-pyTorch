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
from sklearn.cluster import OPTICS, DBSCAN, cluster_optics_dbscan

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

def _eval(labels, logists):
    pre, rec, f1, _ = precision_recall_fscore_support(labels, predicts, average='binary')
    fpr, tpr, _ = roc_curve(labels, logists, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    precisions, recalls, thresholds = precision_recall_curve(labels, logists, pos_label=1)
    pr_auc = metrics.auc(recalls, precisions)
    ap = average_precision_score(labels, logists)
    f1s = np.nan_to_num(2*precisions*recalls/(precisions+recalls))
    plt.step(recalls, precisions, color='b', alpha=0.2,
         where='post')
    step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
    best_comb = np.argmax(f1s)
    best_f1 = f1s[best_comb]
    best_pre = precisions[best_comb]
    best_rec = recalls[best_comb]
    best_threshold = thresholds[best_comb]
    results = {
        'h_pre': pre,
        'h_rec': rec,
        'h_f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'ap': ap,
        'pre': best_pre,
        'rec': best_rec,
        'f1': best_f1,
    }
    return results

def get_embeddings(deepFD, Dl):
    nodes = getattr(Dl, Dl.ds+'_train')
    b_sz = 500
    batches = math.ceil(len(nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = nodes[index*b_sz:(index+1)*b_sz]
        with torch.no_grad():
            embs_batch, _ = deepFD(nodes_batch)
        # print(embs_batch.size(), np.shape(nodes_batch))
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)
    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes)
    return embs.detach().cpu()

def save_embeddings(embs, out_path, outer_epoch):
    pickle.dump(embs, open(f'{out_path}/embs_ep{outer_epoch}.pkl', 'wb'))

def test_dbscan(Dl, args, logger, deepFD, epoch):
    logger.info('Testing with DBSCAN...')
    labels = getattr(Dl, Dl.ds+'_labels')
    features = get_embeddings(deepFD, Dl).numpy()
    save_embeddings(features, args.out_path, epoch)

    resultfile = f'{args.out_path}/results.txt'
    fa = open(resultfile, 'a')
    fa.write(f'====== Epoch {epoch} ======\n')
    # optics
    optics = OPTICS()
    optics.fit(features)
    logists = optics.labels_
    logists[logists >= 0] = 0
    logists[logists < 0] = 1
    logger.info('evaluating with optics')
    results = _eval(labels, logists)
    logger.info(' pre  \t rec  \t  f1  \t  ap  \tpr_auc\troc_auc\t h_pre\t h_rec\t h_f1 \n')
    logger.info('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(results['pre'],results['rec'],results['f1'],results['ap'],results['pr_auc'],results['roc_auc'],results['h_pre'],results['h_rec'],results['h_f1']))
    fa.write('OPTICS\n')
    fa.write(' pre  \t rec  \t  f1  \t  ap  \tpr_auc\troc_auc\t h_pre\t h_rec\t h_f1 \n')
    fa.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(results['pre'],results['rec'],results['f1'],results['ap'],results['pr_auc'],results['roc_auc'],results['h_pre'],results['h_rec'],results['h_f1']))
    # dbscan at 0.5
    logists_050 = cluster_optics_dbscan(reachability=optics.reachability_,core_distances=optics.core_distances_, ordering=optics.ordering_, eps=0.5)
    logists_050[logists_050 >= 0] = 0
    logists_050[logists_050 < 0] = 1
    logger.info('evaluating with dbscan at 0.5')
    results = _eval(labels, logists_050)
    logger.info(' pre  \t rec  \t  f1  \t  ap  \tpr_auc\troc_auc\t h_pre\t h_rec\t h_f1 \n')
    logger.info('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(results['pre'],results['rec'],results['f1'],results['ap'],results['pr_auc'],results['roc_auc'],results['h_pre'],results['h_rec'],results['h_f1']))
    fa.write('DBSCAN at 0.5\n')
    fa.write(' pre  \t rec  \t  f1  \t  ap  \tpr_auc\troc_auc\t h_pre\t h_rec\t h_f1 \n')
    fa.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(results['pre'],results['rec'],results['f1'],results['ap'],results['pr_auc'],results['roc_auc'],results['h_pre'],results['h_rec'],results['h_f1']))
    # dbscan at 2
    logists_200 = cluster_optics_dbscan(reachability=optics.reachability_, core_distances=optics.core_distances_, ordering=optics.ordering_, eps=2)
    logists_200[logists_200 >= 0] = 0
    logists_200[logists_200 < 0] = 1
    logger.info('evaluating with dbscan at 2')
    results = _eval(labels, logists_200)
    logger.info(' pre  \t rec  \t  f1  \t  ap  \tpr_auc\troc_auc\t h_pre\t h_rec\t h_f1 \n')
    logger.info('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(results['pre'],results['rec'],results['f1'],results['ap'],results['pr_auc'],results['roc_auc'],results['h_pre'],results['h_rec'],results['h_f1']))
    fa.write('DBSCAN at 2\n')
    fa.write(' pre  \t rec  \t  f1  \t  ap  \tpr_auc\troc_auc\t h_pre\t h_rec\t h_f1 \n')
    fa.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(results['pre'],results['rec'],results['f1'],results['ap'],results['pr_auc'],results['roc_auc'],results['h_pre'],results['h_rec'],results['h_f1']))
    fa.close()

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

        # stop when all nodes are trained
        if len(visited_nodes) == len(train_nodes):
            break

    return deepFD