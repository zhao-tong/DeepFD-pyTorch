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

def cls_evaluate(Dl, logger, cls_model, features, device, out_path, max_vali_f1, epoch):
    test_nodes = getattr(Dl, Dl.ds+'_cls_test')
    val_nodes = getattr(Dl, Dl.ds+'_cls_val')
    labels = getattr(Dl, Dl.ds+'_labels')

    embs = features[val_nodes]
    with torch.no_grad():
        logists = cls_model(embs)
    _, predicts = torch.max(logists, 1)
    labels_val = labels[val_nodes]
    assert len(labels_val) == len(predicts)
    logists = logists.cpu().numpy().T[1]
    logists = np.exp(logists)
    vali_results = _eval(labels_val, logists, predicts.cpu().numpy())
    logger.info('Epoch [{}], Validation F1: {:.6f}'.format(epoch, vali_results['f1']))
    if vali_results['f1'] > max_vali_f1:
        max_vali_f1 = vali_results['f1']
        embs = features[test_nodes]
        with torch.no_grad():
            logists = cls_model(embs)
        _, predicts = torch.max(logists, 1)
        labels_test = labels[test_nodes]
        assert len(labels_test) == len(predicts)
        logists = logists.cpu().numpy().T[1]
        logists = np.exp(logists)
        test_results = _eval(labels_test, logists, predicts.cpu().numpy())

        logger.info('Epoch [{}], Current best test F1: {:.6f}'.format(epoch, test_results['f1']))

        resultfile = f'{out_path}/result.txt'
        with open(resultfile, 'w') as fr:
            fr.write(f'Epoch {epoch}\n')
            fr.write('     \t pre  \t rec  \t  f1  \t  ap  \tpr_auc\troc_auc\t h_pre\t h_rec\t h_f1 \n')
            fr.write('vali:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(vali_results['pre'],vali_results['rec'],vali_results['f1'],vali_results['ap'],vali_results['pr_auc'],vali_results['roc_auc'],vali_results['h_pre'],vali_results['h_rec'],vali_results['h_f1']))
            fr.write('test:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(test_results['pre'],test_results['rec'],test_results['f1'],test_results['ap'],test_results['pr_auc'],test_results['roc_auc'],test_results['h_pre'],test_results['h_rec'],test_results['h_f1']))
    return max_vali_f1

def _eval(labels, logists, predicts):
    pre, rec, f1, _ = precision_recall_fscore_support(labels, predicts, average='binary')
    fpr, tpr, _ = roc_curve(labels, logists, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    precisions, recalls, thresholds = precision_recall_curve(labels, logists, pos_label=1)
    pr_auc = metrics.auc(recalls, precisions)
    ap = average_precision_score(labels, logists)
    f1s = np.nan_to_num(2*precisions*recalls/(precisions+recalls))
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
    return embs.detach()

def save_embeddings(embs, out_path, outer_epoch):
    pickle.dump(embs, open(f'{out_path}/embs_ep{outer_epoch}.pkl', 'wb'))

def train_classification(Dl, args, logger, deepFD, cls_model, device, max_vali_f1, outer_epoch, epochs=500):
    logger.info('Testing with MLP')
    cls_model.zero_grad()
    c_optimizer = torch.optim.SGD(cls_model.parameters(), lr=0.5)
    c_optimizer.zero_grad()
    b_sz = 100
    train_nodes = getattr(Dl, Dl.ds+'_cls_train')
    labels = getattr(Dl, Dl.ds+'_labels')
    features = get_embeddings(deepFD, Dl)
    save_embeddings(features.cpu().numpy(), args.out_path, outer_epoch)

    for epoch in range(epochs):
        # train_nodes = shuffle(train_nodes)
        np.random.shuffle(train_nodes)
        batches = math.ceil(len(train_nodes) / b_sz)

        for index in range(batches):
            nodes_batch = train_nodes[index*b_sz:(index+1)*b_sz]
            labels_batch = labels[nodes_batch]
            embs_batch = features[nodes_batch]
            logists = cls_model(embs_batch)
            loss = -torch.sum(logists[range(logists.size(0)), labels_batch], 0)
            loss /= len(nodes_batch)
            loss.backward()

            nn.utils.clip_grad_norm_(cls_model.parameters(), 5)
            c_optimizer.step()
            c_optimizer.zero_grad()
            cls_model.zero_grad()

        max_vali_f1 = cls_evaluate(Dl, logger, cls_model, features, device, args.out_path, max_vali_f1, 1000*outer_epoch+epoch)
    return max_vali_f1

def test_dbscan(Dl, args, logger, deepFD, epoch):
    logger.info('Testing with DBSCAN...')
    labels = getattr(Dl, Dl.ds+'_labels')
    features = get_embeddings(deepFD, Dl).cpu().numpy()
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
    results = _eval(labels, logists, logists)
    logger.info(' pre  \t rec  \t  f1  \t  ap  \tpr_auc\troc_auc\t h_pre\t h_rec\t h_f1')
    logger.info('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(results['pre'],results['rec'],results['f1'],results['ap'],results['pr_auc'],results['roc_auc'],results['h_pre'],results['h_rec'],results['h_f1']))
    fa.write('OPTICS\n')
    fa.write(' pre  \t rec  \t  f1  \t  ap  \tpr_auc\troc_auc\t h_pre\t h_rec\t h_f1 \n')
    fa.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(results['pre'],results['rec'],results['f1'],results['ap'],results['pr_auc'],results['roc_auc'],results['h_pre'],results['h_rec'],results['h_f1']))
    # dbscan at 0.5
    logists_050 = cluster_optics_dbscan(reachability=optics.reachability_,core_distances=optics.core_distances_, ordering=optics.ordering_, eps=0.5)
    logists_050[logists_050 >= 0] = 0
    logists_050[logists_050 < 0] = 1
    logger.info('evaluating with dbscan at 0.5')
    results = _eval(labels, logists_050, logists_050)
    logger.info(' pre  \t rec  \t  f1  \t  ap  \tpr_auc\troc_auc\t h_pre\t h_rec\t h_f1')
    logger.info('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(results['pre'],results['rec'],results['f1'],results['ap'],results['pr_auc'],results['roc_auc'],results['h_pre'],results['h_rec'],results['h_f1']))
    fa.write('DBSCAN at 0.5\n')
    fa.write(' pre  \t rec  \t  f1  \t  ap  \tpr_auc\troc_auc\t h_pre\t h_rec\t h_f1 \n')
    fa.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(results['pre'],results['rec'],results['f1'],results['ap'],results['pr_auc'],results['roc_auc'],results['h_pre'],results['h_rec'],results['h_f1']))
    # dbscan at 2
    logists_200 = cluster_optics_dbscan(reachability=optics.reachability_, core_distances=optics.core_distances_, ordering=optics.ordering_, eps=2)
    logists_200[logists_200 >= 0] = 0
    logists_200[logists_200 < 0] = 1
    logger.info('evaluating with dbscan at 2')
    results = _eval(labels, logists_200, logists_200)
    logger.info(' pre  \t rec  \t  f1  \t  ap  \tpr_auc\troc_auc\t h_pre\t h_rec\t h_f1')
    logger.info('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(results['pre'],results['rec'],results['f1'],results['ap'],results['pr_auc'],results['roc_auc'],results['h_pre'],results['h_rec'],results['h_f1']))
    fa.write('DBSCAN at 2\n')
    fa.write(' pre  \t rec  \t  f1  \t  ap  \tpr_auc\troc_auc\t h_pre\t h_rec\t h_f1 \n')
    fa.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(results['pre'],results['rec'],results['f1'],results['ap'],results['pr_auc'],results['roc_auc'],results['h_pre'],results['h_rec'],results['h_f1']))
    fa.close()

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