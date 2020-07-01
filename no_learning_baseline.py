import argparse
from datetime import datetime
from data_loader_siamese import *
from knn_check import knn_distance_calculation, map_for_dataset
from index import BagOfNodesIndex
# NN layers and models



def cross_val_map_local(data04,data19, dims = 17):
    ''' a function to extract features from the graphs (attributes), then similarity search is perfromed using faiss
    Attributes: dataloaders for 2 distinc dates
    dims - num of dimensions for each node attribute
    Output: map@5 value'''
    features04 =np.empty((0, dims))
    features19 = np.empty((0, dims))
    gt04 =[]
    gt19 = []
    dist_graphs_19 = []      #to store the distinct graphs
    for i in range(len(data19.data['features_onehot'])):
        gt19 += [data19.data['targets'][i]]*len(data19.data['features_onehot'][i])        # label for each node
        features19 = np.vstack((features19, data19.data['features_onehot'][i]))
        dist_graphs_19 += [i]* len(data19.data['features_onehot'][i])
    dist_graphs_04 = []  #to store the distinct graphs
    for i in range(len(data04.data['features_onehot'])):
        gt04+=[data04.data['targets'][i]]*len(data04.data['features_onehot'][i])
        features04 = np.vstack((features04, data04.data['features_onehot'][i]))
        dist_graphs_04 += [i]* len(data04.data['features_onehot'][i])
    indexer = BagOfNodesIndex(dimension=features04.shape[1], N_CENTROIDS = 128)       #faiss similarity search
    indexer.train(features04, dist_graphs_04)
    unique_graphs = np.unique(dist_graphs_19)
    gt_gt19 = build_gt_voc(dist_graphs_19, gt19)       # matching between unique graphs and labels, needed for map calc
    gt_gt04 = build_gt_voc(dist_graphs_04, gt04)
    gt_19 = []
    knn_array = []  #returned most similar indexes
    for i in unique_graphs:
        query_features = features19[dist_graphs_19 == i]
        answer = indexer.search(query_features)
        #sorted(answer, key=lambda x: x[1], reverse=True)  # sort the array -> actually, not needed, looks like faiss does it anyway
        gt_19.append(gt_gt19[i])
        knn_array.append([gt_gt04[a] for a in answer[0][:args.N]])  # workaround for structure

    map = map_for_dataset(gt_19, knn_array)
    return map

def build_gt_voc(unique_graphs, gt):
    ''' return a vocabulary matching gt zone labes with graph labels'''
    gt_g = {}
    for i in range(len(unique_graphs)):
        if unique_graphs[i] not in gt_g:
            gt_g[unique_graphs[i]] = gt[i]
    return gt_g

if __name__ == '__main__':
    # Experiment parameters
    parser = argparse.ArgumentParser(description='contrastive_GCN')

    parser.add_argument('--first_dataset', type=str, default='ign_2004',
                        help='Name of dataset number 1, should correspond to the folder with data')
    parser.add_argument('--test_dataset', type=str, default='ign_2014',
                        help='Name of the matching dataset, should correspond to the folder with data')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input training batch-size')
    parser.add_argument('--threads', type=int, default=0,
                        help='num of threads')
    parser.add_argument('--n_folds', type=int, default=1,
                        help='n-fold cross validation, default is 2 folds - single check of best val accuracy')
    parser.add_argument('--seed', type=int, default=111,
                        help='seed for reproduction')
    parser.add_argument('--N', type=bool, default=5,help='N in map@N')

    args = parser.parse_args()
    datareader19 = DataReader(data_dir='./data/IGN_all_clean/%s/' % args.first_dataset.upper(),
                            rnd_state=np.random.RandomState(args.seed),
                            folds=args.n_folds,
                            use_cont_node_attr=True)


    datareader10 = DataReader(data_dir='./data/IGN_all_clean/%s/' % args.test_dataset.upper(),
                              rnd_state=np.random.RandomState(args.seed),
                              folds=args.n_folds,
                              use_cont_node_attr=True)

    start = time.time()
    map = cross_val_map_local(datareader19, datareader10)
    print(map)
    end = time.time()
    print('final map@N is %f time to query all files %f seconds.' % (map, end - start))
    print('it gives %f sec per query' %( (end - start)/len(datareader19.data['targets'])))