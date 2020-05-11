import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import time
import math
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parameter import Parameter
from os.path import join as pjoin


# Data loader and reader
class GraphDataSiamese(torch.utils.data.Dataset):
    def __init__(self,
                 datareader04, datareader19,
                 fold_id,
                 split):
        self.fold_id = fold_id
        self.split = split
        self.rnd_state = datareader04.rnd_state
        self.set_fold(datareader04.data, datareader19.data, fold_id)

    def set_fold(self, data04, data19, fold_id):
        self.total = len(data04['targets'])
        self.N_nodes_max =max(data04['N_nodes_max'], data19['N_nodes_max'])
        self.n_classes = data04['n_classes']
        self.features_dim = data04['features_dim']
        self.idx = data04['splits'][fold_id][self.split]
        # use deepcopy to make sure we don't alter objects in folds
        #for 2004
        self.labels04 = copy.deepcopy([data04['targets'][i] for i in self.idx])
        self.adj_list04 = copy.deepcopy([data04['adj_list'][i] for i in self.idx])
        self.features_onehot04 = copy.deepcopy([data04['features_onehot'][i] for i in self.idx])
        print('%s: %d/%d' % (self.split.upper(), len(self.labels04), len(data04['targets'])))
        self.indices = np.arange(len(self.idx))  # sample indices for this epoch
        #for 2019
        self.labels19 = copy.deepcopy([data19['targets'][i] for i in self.idx])
        self.adj_list19 = copy.deepcopy([data19['adj_list'][i] for i in self.idx])
        self.features_onehot19 = copy.deepcopy([data19['features_onehot'][i] for i in self.idx])


    def pad(self, mtx, desired_dim1, desired_dim2=None, value=0):
        sz = mtx.shape
        assert len(sz) == 2, ('only 2d arrays are supported', sz)
        # if np.all(np.array(sz) < desired_dim1 / 3): print('matrix shape is suspiciously small', sz, desired_dim1)
        if desired_dim2 is not None:
            mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, desired_dim2 - sz[1])), 'constant', constant_values=value)
        else:
            mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, 0)), 'constant', constant_values=value)
        return mtx

    def nested_list_to_torch(self, data):
        if isinstance(data, dict):
            keys = list(data.keys())
        for i in range(len(data)):
            if isinstance(data, dict):
                i = keys[i]
            if isinstance(data[i], np.ndarray):
                data[i] = torch.from_numpy(data[i]).float()
            elif isinstance(data[i], list):
                data[i] = list_to_torch(data[i])
        return data

    def __len__(self):
        return len(self.labels04)

    def __getitem__(self, index):
        index = self.indices[index]
        #N_nodes_max = self.N_nodes_max
        #04
        N_nodes_04 = self.adj_list04[index].shape[0]
        graph_support_04 = np.zeros(self.N_nodes_max)
        graph_support_04[:N_nodes_04] = 1
        #19
        N_nodes_19 = self.adj_list19[index].shape[0]
        graph_support_19 = np.zeros(self.N_nodes_max)
        graph_support_19[:N_nodes_19] = 1

        return self.nested_list_to_torch(
            [self.pad(self.features_onehot04[index].copy(), self.N_nodes_max),  # node_features
             self.pad(self.adj_list04[index], self.N_nodes_max, self.N_nodes_max),  # adjacency matrix
             graph_support_04,  # mask with values of 0 for dummy (zero padded) nodes, otherwise 1
             N_nodes_04,
             int(self.labels04[index])]), self.nested_list_to_torch(
            [self.pad(self.features_onehot19[index].copy(), self.N_nodes_max),  # node_features
             self.pad(self.adj_list19[index], self.N_nodes_max, self.N_nodes_max),  # adjacency matrix
             graph_support_19,  # mask with values of 0 for dummy (zero padded) nodes, otherwise 1
             N_nodes_19,
             int(self.labels19[index])]) # convert to torch


class DataReader():
    '''
    Class to read the txt files containing all data of the dataset
    '''

    def __init__(self,
                 data_dir,  # folder with txt files
                 rnd_state=None,
                 use_cont_node_attr=False,
                 # use or not additional float valued node attributes available in some datasets
                 folds=10):

        self.data_dir = data_dir
        self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
        self.use_cont_node_attr = use_cont_node_attr
        files = os.listdir(self.data_dir)
        data = {}
        nodes, graphs = self.read_graph_nodes_relations(
            list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])
        data['features'] = self.read_node_features(list(filter(lambda f: f.find('node_labels') >= 0, files))[0],
                                                   nodes, graphs, fn=lambda s: int(s.strip()))
        data['adj_list'] = self.read_graph_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes, graphs)
        data['targets'] = np.array(self.parse_txt_file(list(filter(lambda f: f.find('graph_labels') >= 0, files))[0],
                                                       line_parse_fn=lambda s: int(float(s.strip()))))

        if self.use_cont_node_attr:
            data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0],
                                                   nodes, graphs,
                                                   fn=lambda s: np.array(list(map(float, s.strip().split(',')))))

        features, n_edges, degrees = [], [], []
        for sample_id, adj in enumerate(data['adj_list']):
            N = len(adj)  # number of nodes
            if data['features'] is not None:
                assert N == len(data['features'][sample_id]), (N, len(data['features'][sample_id]))
            n = np.sum(adj)  # total sum of edges
            assert n % 2 == 0, n
            n_edges.append(int(n/2))  # undirected edges, so need to divide by 2 n_edges.append(int(n/2))
            if not np.allclose(adj, adj.T):
                print(sample_id, 'not symmetric')
            degrees.extend(list(np.sum(adj, 1)))
            features.append(np.array(data['features'][sample_id]))

        # Create features over graphs as one-hot vectors for each node
        features_all = np.concatenate(features)
        features_min = features_all.min()
        features_dim = int(features_all.max() - features_min + 1)  # number of possible values

        features_onehot = []
        for i, x in enumerate(features):
            feature_onehot = np.zeros((len(x), features_dim))
            for node, value in enumerate(x):
                feature_onehot[node, value - features_min] = 1
            if self.use_cont_node_attr:
                feature_onehot = np.concatenate((feature_onehot, np.array(data['attr'][i])), axis=1)
            features_onehot.append(feature_onehot)

        if self.use_cont_node_attr:
            features_dim = features_onehot[0].shape[1]

        shapes = [len(adj) for adj in data['adj_list']]
        labels = data['targets']  # graph class labels
        labels -= np.min(labels)  # to start from 0
        # N_nodes_max = np.max(shapes)

        classes = np.unique(labels)
        n_classes = len(classes)

        if not np.all(np.diff(classes) == 1):
            print('making labels sequential, otherwise pytorch might crash')
            labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
            for lbl in range(n_classes):
                labels_new[labels == classes[lbl]] = lbl
            labels = labels_new
            classes = np.unique(labels)
            assert len(np.unique(labels)) == n_classes, np.unique(labels)

        print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % (
        np.mean(shapes), np.std(shapes), np.min(shapes), np.max(shapes)))
        print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % (
        np.mean(n_edges), np.std(n_edges), np.min(n_edges), np.max(n_edges)))
        print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' % (
        np.mean(degrees), np.std(degrees), np.min(degrees), np.max(degrees)))
        print('Node features dim: \t\t%d' % features_dim)
        print('N classes: \t\t\t%d' % n_classes)
        print('Classes: \t\t\t%s' % str(classes))
        #for lbl in classes:
        #    print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

        for u in np.unique(features_all):
            print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))

        N_graphs = len(labels)  # number of samples (graphs) in data
        assert N_graphs == len(data['adj_list']) == len(features_onehot), 'invalid data'

        # Create test sets first
        train_ids, test_ids = self.split_ids(np.arange(N_graphs), rnd_state=self.rnd_state, folds=folds)

        # Create train sets
        splits = []
        for fold in range(folds):
            splits.append({'train': train_ids[fold],
                           'test': test_ids[fold]})

        data['features_onehot'] = features_onehot
        data['targets'] = labels
        data['splits'] = splits
        data['N_nodes_max'] = np.max(shapes)  # max number of nodes
        data['features_dim'] = features_dim
        data['n_classes'] = n_classes

        self.data = data

    def split_ids(self, ids_all, rnd_state=None, folds=10):
        n = len(ids_all)
        ids = ids_all[rnd_state.permutation(n)]
        stride = int(np.ceil(n / float(folds)))
        test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
        assert np.all(
            np.unique(np.concatenate(test_ids)) == sorted(ids_all)), 'some graphs are missing in the test sets'
        assert len(test_ids) == folds, 'invalid test sets'
        train_ids = []
        for fold in range(folds):
            train_ids.append(np.array([e for e in ids if e not in test_ids[fold]]))
            assert len(train_ids[fold]) + len(test_ids[fold]) == len(
                np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'

        return train_ids, test_ids

    def parse_txt_file(self, fpath, line_parse_fn=None):
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
        return data

    def read_graph_adj(self, fpath, nodes, graphs):
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
        adj_dict = {}
        for edge in edges:
            node1 = int(edge[0].strip()) -1 # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip()) -1
            graph_id = nodes[node1]
            assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
            if graph_id not in adj_dict:
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = 1
            adj_dict[graph_id][ind2,ind1] = 1 # Modified to make my data symmetric - > to remove for other datasets

        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]

        return adj_list

    def read_graph_nodes_relations(self, fpath):
        graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)
            nodes[node_id] = graph_id
        graph_ids = np.unique(list(graphs.keys()))
        for graph_id in graphs:
            graphs[graph_id] = np.array(graphs[graph_id])
        return nodes, graphs

    def read_node_features(self, fpath, nodes, graphs, fn):
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
        node_features = {}
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]
            if graph_id not in node_features:
                node_features[graph_id] = [None] * len(graphs[graph_id])
            ind = np.where(graphs[graph_id] == node_id)[0]
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
            node_features[graph_id][ind[0]] = x
        node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return node_features_lst



if __name__ == '__main__':
    batch_size = 32
    dataset = 'ign_2004'
    dataset2 = 'ign_2004_shifted'
    model_name = 'gcn'  # 'gcn', 'unet'
    device = 'cpu'  # 'cuda', 'cpu'
    visualize = True
    shuffle_nodes = False
    n_folds = 10  # 10-fold cross validation
    seed = 111
    threads = 0

    print('Loading data')
    datareader = DataReader(data_dir='./data/%s/' % dataset.upper(),
                            rnd_state=np.random.RandomState(seed),
                            folds=n_folds,
                            use_cont_node_attr=True)
    datareader19 = DataReader(data_dir='./data/%s/' % dataset2.upper(),
                            rnd_state=np.random.RandomState(seed),
                            folds=n_folds,
                            use_cont_node_attr=True)

    acc_folds = []
    for fold_id in range(n_folds):
        print('\nFOLD', fold_id)
        loaders = []
        for split in ['train', 'test']:
            gdata = GraphDataSiamese(fold_id=fold_id,
                              datareader04=datareader, datareader19 = datareader19,
                              split=split)

            loader = torch.utils.data.DataLoader(gdata,
                                                 batch_size=batch_size,
                                                 shuffle=split.find('train') >= 0,
                                                 num_workers=threads)
            loaders.append(loader)
    train_loader = loaders[0]
    for batch_idx, data in enumerate(train_loader):
#            for i in range(len(data[0])):
             assert  np.allclose(data[0][4],data[1][4])


