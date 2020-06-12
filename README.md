# Graph convolutional networks

## Overview

This repo was forked from bknyaz to test the possibility to use GCN with contrastive loss for graph mathing. New jupyter demo notebook was added along with a modified dataloader to handle a scenario, where a pair of graphs should have close embeddings.  NT-Xent loss was used to train the global graph embeddings.
I use graphs extracted from geo spatial data and try to match them across time.
An example of my data is shown below:
![Alt text](https://github.com/margokhokhlova/graph_nn/blob/master/figs/4_periods2.png?raw=true "same geo zone across time")

Current results:
parameters
F =[256,512], dims = 256, 15 epochs
Map@5  train 2019-2004 0,40
Map@5 val 2014-2004  0,45
Map@5 test 2010-2004  0.483181





