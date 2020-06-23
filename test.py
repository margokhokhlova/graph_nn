import argparse
from datetime import datetime
from data_loader_siamese import *
from knn_check import knn_distance_calculation, map_for_dataset
from models import GCN_unwrapped, GCN
from index import BagOfNodesIndex
# NN layers and models
if __name__ == '__main__':
    # Experiment parameters
    parser = argparse.ArgumentParser(description='contrastive_GCN')

    parser.add_argument('--first_dataset', type=str, default='ign_2004',
                        help='Name of dataset number 1, should correspond to the folder with data')
    parser.add_argument('--test_dataset', type=str, default='ign_2010',
                        help='Name of the matching dataset, should correspond to the folder with data')
    parser.add_argument('--emb_dim', type=int, default=256,
                        help='Feature output size (default: 128')
    parser.add_argument('--hidden_filters', type=list, default=[256, 512],
                        help='num of gcn layers')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='input training batch-size')
    parser.add_argument('--device', default='cpu',
                        help='cpu or cuda')
    parser.add_argument('--load-model', type=str, default='saved_model/Jun11_11-50-07_HP1908P001.pth',
                        help='Load model to resume training for (default None)')
    parser.add_argument('--device-id', type=int, default=0,
                        help='GPU device id (default: 0')
    parser.add_argument('--threads', type=int, default=0,
                        help='num of threads')
    parser.add_argument('--n_folds', type=int, default=1,
                        help='n-fold cross validation, default is 2 folds - single check of best val accuracy')
    parser.add_argument('--seed', type=int, default=111,
                        help='seed for reproduction')
    parser.add_argument('--features', type=str, default = 'local',
                         help='global or local scenario? global descriptors matching or local with a BOW model')
    parser.add_argument('--visualize', type=bool, default=True,
                        help='visualize the data')
    parser.add_argument('--N', type=bool, default=5,
                        help='N in map@N')

    args = parser.parse_args()
    datareader19 = DataReader(data_dir='./data/IGN_all_clean/%s/' % args.first_dataset.upper(),
                            rnd_state=np.random.RandomState(args.seed),
                            folds=args.n_folds,
                            use_cont_node_attr=True)


    datareader10 = DataReader(data_dir='./data/IGN_all_clean/%s/' % args.test_dataset.upper(),
                              rnd_state=np.random.RandomState(args.seed),
                              folds=args.n_folds,
                              use_cont_node_attr=True)



    # model definition
    if args.features =='local':
        model = GCN_unwrapped(in_features=datareader19.data['features_dim'],
                    out_features=args.emb_dim,  # loaders[0].dataset.n_classes
                    n_hidden=0,
                    filters=args.hidden_filters,
                    dropout=0,
                    adj_sq=False,
                    scale_identity=True).to(args.device)

        print('\nInitialize model')
        print(model)

        #
    else:
        model = GCN(in_features=datareader19.data['features_dim'],
                              out_features=args.emb_dim,  # loaders[0].dataset.n_classes
                              n_hidden=0,
                              filters=args.hidden_filters,
                              dropout=0.2,
                              adj_sq=False,
                              scale_identity=True).to(args.device)

        print('\nInitialize model')
        print(model)

    def calculate_features(train_loader, dims):
        'just returns an array of the global graph features and corresponding GT indexes'
        features2004 = np.empty((0, dims))
        features2019 = np.empty((0, dims))
        gt2004 = []
        gt2019 = []

        for batch_idx, data in enumerate(train_loader):
            for i in range(len(data[0])):
                data[0][i] = data[0][i].to(args.device)
                data[1][i] = data[1][i].to(args.device)
            gt2004.extend(list(data[0][4].numpy())) #save the GT values
            gt2019.extend(list(data[1][4].numpy()))
            output_2004 = model(data[0])
            output_2019 = model(data[1])
            features_2004 = output_2004.detach().cpu()  # .max(1, keepdim=True)[1]
            features_2019 = output_2019.detach().cpu()  # .max(1, keepdim=True)[1]
            features2004 = np.vstack((features2004, features_2004.data.numpy()))
            features2019 = np.vstack(
                (features2019, features_2019.data.numpy()))  # (features_2019.reshape(features_2019.shape[0], emb_dim))
        return np.array(features2004), np.array(features2019), gt2004, gt2019

    def unwrap_unmask(features, masks, gt, counter=0):
        ''' function transforms the features from 3 dims [B, N, F] to 2 dims [N*B,F], N is number of nodes, B is
         the batch size and F is feature dimention.
        masks - whether the node is real or padded
        gt - gt graph label
        The artificial nodes are removed (we use zero-padding to train the model to have a constanst node number)
        returns: new features, N*B ground truth labels and graph_id (just an array with specification of nodes beloging to graphs
        '''
        f =[] #an array to store the final features
        new_gt = []
        graph_id = []
        B,N,F = features.shape
        for i in range(B):
            feat = features[i]
            masks_graph = masks[i]
            for m in range(len(masks_graph)):
                if masks_graph[m]!=0:
                    f.append(feat[m,:])
                    new_gt.append(gt[i])
                    graph_id.append(counter)
            counter+=1
        return f, new_gt, graph_id, counter




    def calculate_features_local(train_loader, dims):
        'just returns an array of the global graph features and corresponding GT indexes'
        features2004 = np.empty((0, dims))
        features2019 = np.empty((0, dims))
        gt2004 = []
        gt2019 = []
        dist_graphs = []
        counter = 0
        for batch_idx, data in enumerate(train_loader):
            for i in range(len(data[0])):
                data[0][i] = data[0][i].to(args.device)
                data[1][i] = data[1][i].to(args.device)
            gt04 = data[0][4].numpy() # save the GT values
            gt19 = data[1][4].numpy()
            output_2004 = model(data[0])
            output_2019 = model(data[1])
            features_2004 = output_2004.detach().cpu().numpy()  # .max(1, keepdim=True)[1]
            features_2019 = output_2019.detach().cpu().numpy()  # .max(1, keepdim=True)[1]
            node_mask04 = data[0][2].numpy() # masks to remove the padding
            node_mask19 = data[1][2].numpy()
            # unwrap features and delete zero nodes
            features_2004, gt_2004, _, _= unwrap_unmask(features_2004,node_mask04, gt04)
            features_2019, gt_2019, graph_id, counter = unwrap_unmask(features_2019, node_mask19, gt19, counter)
            features2004 = np.vstack((features2004, features_2004))
            features2019 = np.vstack((features2019, features_2019))  # (features_2019.reshape(features_2019.shape[0], emb_dim))
            gt2004 += gt_2004
            gt2019 += gt_2019
            dist_graphs += graph_id
        return np.array(features2004), np.array(features2019), gt2004, gt2019, dist_graphs

    def cross_val_map(loaders):
        'calculates the features of the graphs and then the map value'
        emb04, emb19, gt04, gt19 = calculate_features(loaders, args.emb_dim)
        #map = knn_distance_calculation(query=emb04, database=emb19,gt_indexes2004 = gt04, gt_indexes2019=gt19, distance='cosine', N=args.N)
        indexer = BagOfNodesIndex(dimension=emb04.shape[1], N_CENTROIDS=256)
        indexer.train(emb04, gt04)
        unique_graphs = np.unique(gt19)
        gt_19 = []
        knn_array = []
        for i in unique_graphs:
            query_features = emb19[gt19 == i]
            answer = indexer.search(query_features)
            sorted(answer, key=lambda x: x[1], reverse=True)  # sort the array
            gt_19.append(i)
            knn_array.append(answer[0][:args.N])  # workaround for structure

        map = map_for_dataset(gt_19, knn_array)
        return map
    def cross_val_map_local(loaders):
        'calculates the features of the graphs and then the map value'
        emb04, emb19, gt04, gt19, dist_graphs = calculate_features_local(loaders, 512) #TODO make it as a parameter
        indexer =BagOfNodesIndex(dimension=emb04.shape[1], N_CENTROIDS = 128)
        indexer.train(emb04, gt04)
        unique_graphs = np.unique(dist_graphs)
        gt_g = build_gt_voc(dist_graphs, gt19)
        gt_19 = []
        knn_array = []
        for i in range(len(unique_graphs)):
            query_features = emb19[dist_graphs==unique_graphs[i]]
            answer = indexer.search(query_features)
            sorted(answer,key=lambda x: x[1], reverse = True) #sort the array
            gt_19.append(gt_g[unique_graphs[i]])
            knn_array.append(answer[0][:args.N])  # workaround for structure

        map = map_for_dataset(gt_19, knn_array)
        return map


    def build_gt_voc(unique_graphs, gt):
        ''' return a vocabulary matching gt zone labes with graph labels'''
        gt_g = {}
        for i in range(len(unique_graphs)):
            if unique_graphs[i] not in gt_g:
                gt_g[unique_graphs[i]] = gt[i]
        return gt_g
    if args.features =='global':
        model.load_state_dict(torch.load(args.load_model))
    else:
        pretrained_dict = torch.load(args.load_model)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    start = time.time()
    #final test accuracy calculation
    test_loaders = []
    for split in ['train', 'test']:
        gdata = GraphDataSiamese(fold_id=0,
                                 datareader04=datareader19, datareader19=datareader10, split = split)

        test_loader = torch.utils.data.DataLoader(gdata,
                                             batch_size=args.batch_size,
                                             # shuffle=split.find('train') >= 0,
                                             num_workers=args.threads)
        test_loaders.append(test_loader)
    if args.features == 'local':
        map = cross_val_map_local(test_loaders[1])
    else:
        map = cross_val_map(test_loaders[1])
    print(map)
    end = time.time()
    print('time to query all files %f seconds.' % (end - start))
    print('it gives %f sec per query' % ((end - start) / len(datareader19.data['targets'])))
