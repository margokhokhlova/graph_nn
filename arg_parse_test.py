import argparse
from datetime import datetime
import csv

parser = argparse.ArgumentParser(description='contrastive_GCN')


parser.add_argument('--first_dataset', type=str, default='ign_2004',
                    help='Name of dataset number 1, should correspond to the folder with data')
parser.add_argument('--second_dataset', type=str, default='ign_2019',
                    help='Name of the matching dataset, should correspond to the folder with data')
parser.add_argument('--feature-size', type=int, default=128,
                    help='Feature output size (default: 128')
parser.add_argument('--batch-size', type=int, default=35, metavar='N',
                    help='input training batch-size')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of training epochs (default: 150)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3')
parser.add_argument("--decay-lr", default=1e-6, action="store", type=float,
                    help='Learning rate decay (default: 1e-6')
parser.add_argument('--tau', default=0.5, type=float,
                    help='Tau temperature smoothing (default 0.5)')
parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: runs)')
parser.add_argument('--device', default='cpu',
                    help='cpu or cuda')
parser.add_argument('--load-model', type=str, default=None,
                    help='Load model to resume training for (default None)')
parser.add_argument('--device-id', type=int, default=0,
                    help='GPU device id (default: 0')
parser.add_argument('--threads', type=int, default=0,
                    help='num of threads')
parser.add_argument('--log_interval', type=int, default=10,
                    help='num of threads')
parser.add_argument('--n_folds', type=int, default=2,
                    help='n-fold cross validation, default is 2 folds')
parser.add_argument('--seed', type=int, default=111,
                    help='seed for reproduction')
parser.add_argument('--visualize', type=bool, default=True,
                    help='visualize the data')
args = parser.parse_args()
filename = args.log_dir + '/' + datetime.now().strftime("%Y%m%d-%H%M%S") +'.csv'
w = csv.writer(open(filename, "w"))
for key, val in args.__dict__.items():
    w.writerow([key, val])

# # Experiment parameters
# batch_size = 5
# threads = 0
# lr = 0.005
# epochs = 40
# log_interval = 10
# wdecay = 1e-4
# emb_dim = 128  # I am trying to get an embedding in the end
# dataset = 'ign_2004'
# dataset2 = 'ign_2019'
# model_name = 'gcn'  # 'gcn', 'unet'
# device = 'cpu'  # 'cuda', 'cpu'
# visualize = True
# shuffle_nodes = False
# n_folds = 2  # 10-fold cross validation
# seed = 111
# tau = 0.5