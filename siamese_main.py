import torch.utils
import torch.utils.data
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from data_loader_siamese import *
from loss import contrastive_loss
from knn_check import knn_distance_calculation

# NN layers and models
class GraphConv(nn.Module):
    '''
    Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017)
    Additional tricks (power of adjacency matrix and weight self connections) as in the Graph U-Net paper
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 activation=None,
                 adj_sq=False,
                 scale_identity=False):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.adj_sq = adj_sq
        self.activation = activation
        self.scale_identity = scale_identity

    def laplacian_batch(self, A):
        batch, N = A.shape[:2]
        if self.adj_sq:
            A = torch.bmm(A, A)  # use A^2 to increase graph connectivity
        I = torch.eye(N).unsqueeze(0).to(args.device)
        if self.scale_identity:
            I = 2 * I  # increase weight of self connections
        A_hat = A + I
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, data):
        x, A = data[:2]
        x = self.fc(torch.bmm(self.laplacian_batch(A), x))
        if self.activation is not None:
            x = self.activation(x)
        return (x, A)


class GCN(nn.Module):
    '''
    Baseline Graph Convolutional Network with a stack of Graph Convolution Layers and global pooling over nodes.
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 filters=[64, 64, 64],
                 n_hidden=0,
                 dropout=0.2,
                 adj_sq=False,
                 scale_identity=False):
        super(GCN, self).__init__()

        # Graph convolution layers
        self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                                out_features=f,
                                                activation=nn.ReLU(inplace=True),
                                                adj_sq=adj_sq,
                                                scale_identity=scale_identity) for layer, f in enumerate(filters)]))

        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(filters[-1], n_hidden))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]
        fc.append(nn.Linear(n_last, out_features))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        x = self.gconv(data)[0]
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes
        x = self.fc(x)
        return x



if __name__ == '__main__':
    # Experiment parameters
    parser = argparse.ArgumentParser(description='contrastive_GCN')

    parser.add_argument('--first_dataset', type=str, default='ign_2019',
                        help='Name of dataset number 1, should correspond to the folder with data')
    parser.add_argument('--second_dataset', type=str, default='ign_2014',
                        help='Name of the matching dataset, should correspond to the folder with data')
    parser.add_argument('--third_dataset', type=str, default='ign_2010',
                        help='Name of the matching dataset, should correspond to the folder with data')
    parser.add_argument('--testing_dataset', type=str, default='ign_2004',
                        help='Name of the matching dataset, should correspond to the folder with data')
    parser.add_argument('--emb_dim', type=int, default=256,
                        help='Feature output size (default: 128')
    parser.add_argument('--hidden_filters', type=list, default=[128, 256],
                        help='num of gcn layers')
    parser.add_argument('--batch-size', type=int, default=35, metavar='N',
                        help='input training batch-size')
    parser.add_argument('--epochs', type=int, default=45, metavar='N',
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
    parser.add_argument('--n_folds', type=int, default=1,
                        help='n-fold cross validation, default is 2 folds - single check of best val accuracy')
    parser.add_argument('--seed', type=int, default=111,
                        help='seed for reproduction')
    parser.add_argument('--visualize', type=bool, default=True,
                        help='visualize the data')

    args = parser.parse_args()
    filename = args.log_dir + '/' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.txt'

    w = open(filename, "w")
    for key, val in args.__dict__.items():
        w.write(str(key) +' ' +str(val) +'\n')
    print('torch', torch.__version__)

    print('Loading data')
    datareader19 = DataReader(data_dir='./data/%s/' % args.first_dataset.upper(),
                            rnd_state=np.random.RandomState(args.seed),
                            folds=args.n_folds,
                            use_cont_node_attr=True)
    datareader14 = DataReader(data_dir='./data/%s/' % args.second_dataset.upper(),
                              rnd_state=np.random.RandomState(args.seed),
                              folds=args.n_folds,
                              use_cont_node_attr=True)
    datareader10 = DataReader(data_dir='./data/%s/' % args.third_dataset.upper(),
                              rnd_state=np.random.RandomState(args.seed),
                              folds=args.n_folds,
                              use_cont_node_attr=True)
    datareader04 = DataReader(data_dir='./data/%s/' % args.testing_dataset.upper(),
                              rnd_state=np.random.RandomState(args.seed),
                              folds=args.n_folds,
                              use_cont_node_attr=True)
    # tensorboard
    writer = SummaryWriter()

    # model definition

    model = GCN(in_features=datareader19.data['features_dim'],
                out_features=args.emb_dim,  # loaders[0].dataset.n_classes
                n_hidden=0,
                filters=args.hidden_filters,
                dropout=0.2,
                adj_sq=False,
                scale_identity=True).to(args.device)

    print('\nInitialize model')
    print(model)

    c = 0
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        c += p.numel()
    print('N trainable parameters:', c)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.decay_lr,
        betas=(0.5, 0.999))

    scheduler = lr_scheduler.MultiStepLR(optimizer, [10, 15], gamma=0.1)
    best_test_loss = 10000 # to keep track of the validation_loss parameter

    def train(train_loader, epoch):
        scheduler.step()
        model.train()
        loss_func = contrastive_loss(tau=args.tau)
        start = time.time()
        train_loss, n_samples = 0, 0
        for batch_idx, data in enumerate(train_loader):
            for i in range(len(data[0])):
                data[0][i] = data[0][i].to(args.device)
                data[1][i] = data[1][i].to(args.device)
            optimizer.zero_grad()
            output_2004 = model(data[0])
            output_2019 = model(data[1])
            loss = loss_func(output_2004, output_2019)
            loss.backward()
            optimizer.step()
            time_iter = time.time() - start
            train_loss += loss.item() * len(output_2004)
            n_samples += len(output_2004)
            if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
                    epoch, n_samples, len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples,
                    time_iter / (batch_idx + 1)))
                w.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f} \n'.format(
                    epoch, n_samples, len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples,
                    time_iter / (batch_idx + 1)))
        train_loss /= n_samples
        return train_loss/len(output_2004)

    #             break

    def test(test_loader, epoch):
        model.eval()
        start = time.time()
        loss_fn = contrastive_loss(tau=args.tau)
        test_loss, correct, n_samples = 0, 0, 0
        for batch_idx, data in enumerate(test_loader):
            for i in range(len(data[0])):
                data[0][i] = data[0][i].to(args.device)
                data[1][i] = data[1][i].to(args.device)
            optimizer.zero_grad()
            output_2004 = model(data[0])
            output_2019 = model(data[1])
            loss = loss_fn(output_2004, output_2019)
            test_loss += loss.item()
            n_samples += len(output_2004)
        #         pred = output.detach().cpu().max(1, keepdim=True)[1]

        #         correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()
        test_loss /= n_samples
        print('Test set (epoch {}): Average loss: {:.4f}\n'.format(epoch,
                                                                 test_loss))
        global best_test_loss
        if test_loss<best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'saved_model/siamese_gcn.pth') # save the best parameters
        return test_loss


    def calculate_features(train_loader, dims):
        features2004 = np.empty((0, dims))
        features2019 = np.empty((0, dims))
        for batch_idx, data in enumerate(train_loader):
            for i in range(len(data[0])):
                data[0][i] = data[0][i].to(args.device)
                data[1][i] = data[1][i].to(args.device)
            output_2004 = model(data[0])
            output_2019 = model(data[1])
            features_2004 = output_2004.detach().cpu()  # .max(1, keepdim=True)[1]
            features_2019 = output_2019.detach().cpu()  # .max(1, keepdim=True)[1]
            features2004 = np.vstack((features2004, features_2004.data.numpy()))
            features2019 = np.vstack(
                (features2019, features_2019.data.numpy()))  # (features_2019.reshape(features_2019.shape[0], emb_dim))
        return np.array(features2004), np.array(features2019)
    def cross_val_map(loaders):
        emb04, emb19 = calculate_features(loaders, args.emb_dim)
        map = knn_distance_calculation(query=emb04, database=emb19, distance='cosine', N=5)
        return map





    # first round of training
    for fold_id in range(args.n_folds):
        print('\nFOLD', fold_id)
        loaders = []
        loaders_val = []
        for split in ['train', 'test']:
            gdata = GraphDataSiamese(fold_id=fold_id,
                                     datareader04=datareader19, datareader19=datareader14,
                                     split=split)
            gdata_val = GraphDataSiamese(fold_id=fold_id,
                                     datareader04=datareader19, datareader19=datareader10,
                                     split=split) #second validation set to calculate map

            loader = torch.utils.data.DataLoader(gdata,
                                                 batch_size=args.batch_size,
                                                 # shuffle=split.find('train') >= 0,
                                                 num_workers=args.threads)
            loader_val = torch.utils.data.DataLoader(gdata_val,
                                                 batch_size=args.batch_size,
                                                 # shuffle=split.find('train') >= 0,
                                                 num_workers=args.threads)
            loaders.append(loader)
            loaders_val.append(loader_val)
        for epoch in range(args.epochs):
            tr_l =train(loaders[1], epoch)
            ts_l = test(loaders_val[1], epoch)
            writer.add_scalar('Loss/train', tr_l, epoch)
            writer.add_scalar('Loss/val', ts_l, epoch)
            map = cross_val_map(loaders[1])
            w.write('epoch' + str(epoch) + 'map on train set is ' + str(map) + '\n')
            writer.add_scalar('map/train', map, epoch)
            map = cross_val_map(loaders_val[1])
            w.write('epoch'+str(epoch)+'map on val set is ' + str(map) + '\n')
            writer.add_scalar('map/val', map, epoch)
            for param_group in optimizer.param_groups:
                lr  = param_group['lr']
            writer.add_scalar('lr', lr, epoch)
    # second round of training using the second dataset:
    #lr_scheduler.base_lrs = [args.lr]
    for epoch in range(args.epochs, args.epochs*2):
        tr_l = train(loaders_val[1], epoch)
        ts_l = test(loaders[1], epoch)
        map = cross_val_map(loaders[1])
        writer.add_scalar('Loss/train', tr_l, epoch)
        writer.add_scalar('Loss/val', ts_l, epoch)
        map = cross_val_map(loaders[1])
        w.write('epoch' + str(epoch) + 'map on train set is ' + str(map) + '\n')
        writer.add_scalar('map/train', map, epoch)
        map = cross_val_map(loaders_val[1])
        w.write('epoch'+str(epoch)+'map on val set is ' + str(map) + '\n')
        writer.add_scalar('map/val', map, epoch)
        for param_group in optimizer.param_groups:
            lr  = param_group['lr']
        writer.add_scalar('lr', lr, epoch)


    model.load_state_dict(torch.load('saved_model/siamese_gcn.pth'))

    #final test accuracy calculation
    test_loaders = []
    for split in ['train', 'test']:
        gdata = GraphDataSiamese(fold_id=0,
                                 datareader04=datareader04, datareader19=datareader10, split = split)

        test_loader = torch.utils.data.DataLoader(gdata,
                                             batch_size=args.batch_size,
                                             # shuffle=split.find('train') >= 0,
                                             num_workers=args.threads)
        test_loaders.append(test_loader)

    map = cross_val_map(test_loaders[1])
    w.write('final map precision is ' + str(map)+'\n')
    w.close()