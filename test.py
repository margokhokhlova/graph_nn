import torch.utils
import torch.utils.data
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from data_loader_siamese import *
from loss import contrastive_loss
from knn_check import knn_distance_calculation
from models import GCN_unwrapped, GCN

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
    parser.add_argument('--features', type=str, default = 'global',
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
    # tensorboard


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

    def cross_val_map(loaders):
        'calculates the features of the graphs and then the map value'
        emb04, emb19, gt04, gt19 = calculate_features(loaders, args.emb_dim)
        map = knn_distance_calculation(query=emb04, database=emb19,gt_indexes2004 = gt04, gt_indexes2019=gt19, distance='cosine', N=args.N)
        return map

    model.load_state_dict(torch.load(args.load_model))
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

    map = cross_val_map(test_loaders[1])
    end = time.time()
    print('time to query all files %f seconds.' % (end - start))
