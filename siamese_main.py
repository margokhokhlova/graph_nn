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

from data_loader_siamese import *
from loss import contrastive_loss


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
        I = torch.eye(N).unsqueeze(0).to(device)
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
    batch_size = 5
    threads = 0
    lr = 0.005
    epochs = 40
    log_interval = 10
    wdecay = 1e-4
    emb_dim = 128  # I am trying to get an embedding in the end
    dataset = 'ign_2004'
    dataset2 = 'ign_2019'
    model_name = 'gcn'  # 'gcn', 'unet'
    device = 'cpu'  # 'cuda', 'cpu'
    visualize = True
    shuffle_nodes = False
    n_folds = 2  # 10-fold cross validation
    seed = 111
    tau = 0.5
    print('torch', torch.__version__)


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
                                     datareader04=datareader, datareader19=datareader19,
                                     split=split)

            loader = torch.utils.data.DataLoader(gdata,
                                                 batch_size=batch_size,
                                                 # shuffle=split.find('train') >= 0,
                                                 num_workers=threads)
            loaders.append(loader)

    if model_name == 'gcn':
        model = GCN(in_features=loaders[0].dataset.features_dim,
                    out_features=emb_dim,  # loaders[0].dataset.n_classes
                    n_hidden=0,
                    filters=[64, 64, 64],
                    dropout=0.2,
                    adj_sq=False,
                    scale_identity=False).to(device)

    print('\nInitialize model')
    print(model)
    c = 0
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        c += p.numel()
    print('N trainable parameters:', c)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=wdecay,
        betas=(0.5, 0.999))

    scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 30], gamma=0.1)


    def train(train_loader, epoch):
        scheduler.step()
        model.train()
        loss_func = contrastive_loss(tau=tau)
        start = time.time()
        train_loss, n_samples = 0, 0
        for batch_idx, data in enumerate(train_loader):
            for i in range(len(data[0])):
                data[0][i] = data[0][i].to(device)
                data[1][i] = data[1][i].to(device)
            optimizer.zero_grad()
            output_2004 = model(data[0])
            output_2019 = model(data[1])
            loss = loss_func(output_2004, output_2019)
            loss.backward()
            optimizer.step()
            time_iter = time.time() - start
            train_loss += loss.item() * len(output_2004)
            n_samples += len(output_2004)
            if batch_idx % log_interval == 0 or batch_idx == len(train_loader) - 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
                    epoch, n_samples, len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples,
                    time_iter / (batch_idx + 1)))


    #             break

    def test(test_loader, epoch):
        model.eval()
        start = time.time()
        loss_fn = contrastive_loss(tau=tau)
        test_loss, correct, n_samples = 0, 0, 0
        for batch_idx, data in enumerate(test_loader):
            for i in range(len(data[0])):
                data[0][i] = data[0][i].to(device)
                data[1][i] = data[1][i].to(device)
            optimizer.zero_grad()
            output_2004 = model(data[0])
            output_2019 = model(data[1])
            loss = loss_fn(output_2004, output_2019, tau=tau)
            test_loss += loss.item()
            n_samples += len(output_2004)
        #         pred = output.detach().cpu().max(1, keepdim=True)[1]

        #         correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()

        time_iter = time.time() - start

        test_loss /= n_samples

        # acc = 100. * correct / n_samples
        print('Test set (epoch {}): Average loss: {:.4f}\n'.format(epoch,
                                                                   test_loss))
        return test_loss


    def calculate_features(train_loader):
        features2004 = np.empty((0, 128))
        features2019 = np.empty((0, 128))
        for batch_idx, data in enumerate(train_loader):
            for i in range(len(data[0])):
                data[0][i] = data[0][i].to(device)
                data[1][i] = data[1][i].to(device)
            output_2004 = model(data[0])
            output_2019 = model(data[1])
            features_2004 = output_2004.detach().cpu()  # .max(1, keepdim=True)[1]
            features_2019 = output_2019.detach().cpu()  # .max(1, keepdim=True)[1]
            features2004 = np.vstack((features2004, features_2004.data.numpy()))
            features2019 = np.vstack(
                (features2019, features_2019.data.numpy()))  # (features_2019.reshape(features_2019.shape[0], emb_dim))
        return np.array(features2004), np.array(features2019)


    for epoch in range(epochs):
        train(loaders[0], epoch)
        test(loaders[0], epoch)