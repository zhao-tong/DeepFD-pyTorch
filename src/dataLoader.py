import os
import sys
import copy
import pickle
import pathlib
import numpy as np
from multiprocessing import Pool
from scipy.sparse import csr_matrix

from src.utils import *
from collections import defaultdict, Counter

class DataLoader():
    """ data loader """
    def __init__(self, args, logger):
        self.ds = args.dataSet
        self.args = args
        self.logger = logger
        self.file_paths = json.load(open(f'{args.config_dir}/{args.file_paths}'))
        self.load_dataSet(args.dataSet)

    def load_dataSet(self, dataSet):
        ds = dataSet
        graph_u2p_file = self.file_paths[ds]['graph_u2p']
        graph_simi_file = self.file_paths[ds]['graph_u2u_simi']
        labels_file = self.file_paths[ds]['labels']

        graph_u2p = pickle.load(open(graph_u2p_file, 'rb'))
        labels = pickle.load(open(labels_file, 'rb'))
        graph_u2p[graph_u2p > 0] = 1
        graph_u2u = graph_u2p @ graph_u2p.T
        if os.path.isfile(graph_simi_file):
            graph_simi = pickle.load(open(graph_simi_file, 'rb'))
            self.logger.info('loaded similarity graph from cache')
        else:
            graph_simi = np.zeros(np.shape(graph_u2u))
            nz_entries = []
            for i in range(np.shape(graph_u2u)[0]):
                for j in range(i+1, np.shape(graph_u2u)[0]):
                    nz_entries.append([i, j])
            self.logger.info(f'Calculating user-user similarity graph, {len(nz_entries)} edges to go...')
            sz = 1000
            n_batch = math.ceil(len(nz_entries) / sz)
            batches = np.array_split(nz_entries, n_batch)
            pool = Pool()
            results = pool.map(get_simi_single_iter, [(entries_batch, graph_u2p) for entries_batch in batches])
            results = list(zip(*results))
            row = np.concatenate(results[0])
            col = np.concatenate(results[1])
            dat = np.concatenate(results[2])
            for x in range(len(row)):
                graph_simi[row[x], col[x]] = dat[x]
                graph_simi[col[x], row[x]] = dat[x]
            pickle.dump(graph_simi, open(graph_simi_file, "wb"))
            self.logger.info('Calculated user-user similarity and saved it for catch.')

        assert len(labels) == np.shape(graph_u2p)[0] == np.shape(graph_u2u)[0]
        test_indexs_cls, val_indexs_cls, train_indexs_cls = self._split_data_cls(len(labels))

        setattr(self, dataSet+'_train', np.arange(np.shape(graph_u2p)[0]))
        setattr(self, dataSet+'_cls_test', test_indexs_cls)
        setattr(self, dataSet+'_cls_val', val_indexs_cls)
        setattr(self, dataSet+'_cls_train', train_indexs_cls)
        setattr(self, dataSet+'_u2p', graph_u2p)
        setattr(self, dataSet+'_u2u', graph_u2u)
        setattr(self, dataSet+'_simi', graph_simi)
        setattr(self, dataSet+'_labels', labels)

    def get_train(self):
        training_cps = defaultdict(list)
        g_u2u = getattr(self, self.ds+'_u2u')
        n = np.shape(g_u2u)[0]
        for i in range(n):
            line = g_u2u[i].toarray().squeeze()
            pos_pool = np.where(line != 0)[0]
            neg_pool = np.where(line == 0)[0]
            if len(pos_pool) <= 10:
                pos_nodes = pos_pool
            else:
                pos_nodes = np.random.choice(pos_pool, 10, replace=False)
            if len(neg_pool) <= 10:
                neg_nodes = neg_pool
            else:
                neg_nodes = np.random.choice(neg_pool, 10, replace=False)
            for pos_n in pos_nodes:
                training_cps[i].append((i, pos_n))
            for neg_n in neg_nodes:
                training_cps[i].append((i, neg_n))
        return training_cps

    def _split_data_cls(self, num_nodes, test_split=3, val_split=6):
        rand_indices = np.random.permutation(num_nodes)

        test_size = num_nodes // test_split
        val_size = num_nodes // val_split
        train_size = num_nodes - (test_size + val_size)

        test_indexs = rand_indices[:test_size]
        val_indexs = rand_indices[test_size:(test_size+val_size)]
        train_indexs = rand_indices[(test_size+val_size):]

        return test_indexs, val_indexs, train_indexs