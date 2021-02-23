#!/usr/bin/env python
# coding: utf-8

# In[14]:


"""
Title: Graph Convolutional Networks in Pytorch
Author: Thomas Kipf
Date: February 25, 2019
Availability: https://github.com/tkipf/pygcn
"""
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


# In[15]:


import torch.nn as nn
import torch.nn.functional as F

class VanillaGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):#, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        #self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x#F.log_softmax(x, dim=1)


# In[2]:


import numpy as np
import pandas as pd


# In[ ]:


#load in adjacency matrix
adj = pd.read_csv(adj_filpath)
features = featureDaySpace(5, 0)
features = torch.FloatTensor(np.array(features))


# In[3]:


sample_adj = np.array([[1, 0, 1],
                      [0, 1, 1],
                      [0, 0, 1]])


# In[12]:


#normalize adjacency matrix
D = np.diag(sample_adj.sum(1))
adj = np.linalg.inv(D**(1/2)).dot(sample_adj).dot(D**(1/2))


# In[17]:


#num_days = 5
#NUM_EPOCHS = 100
#model = VanillaGCN(nfeat=4 * num_days,
            #nhid=16,
            #nclass=1)


# In[18]:


import torch.optim as optim


# In[ ]:


optimizer = optim.Adam(model.parameters(),
                       lr=.001)
crit = nn.BCEWithLogitsLoss()


# In[ ]:


def accuracy(preds, labels):
    
    preds = torch.round(torch.sigmoid(preds))

    acc = torch.round((preds == labels).sum().float() / labels.shape[0] * 100)
    
    return acc


# In[ ]:


#training loop
for e in range(NUM_EPOCHS):
    #use 70% of days for training
    epoch_loss = 0
    for i in range(int(127 * .7)):
        features, labels = featureDaySpace(num_days, i)
        features = torch.FloatTensor(np.array(features))
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = crit(output, labels)
        acc_train = accuracy(output, labels)
        loss_train.backward()
        optimizer.step()
        epoch_loss += train_loss.item()
    epoch_loss /= int(127 * .7)
    print(f'Loss for epoch {e}: {epoch_loss}')


# In[ ]:


#test loop
total_test_loss = 0
total_test_acc = 0
for i in range(int(127 * .7) + 1, 127):
    features, labels = featureDaySpace(num_days, i)
    features = torch.FloatTensor(np.array(features))
    model.eval()
    output = model(features, adj)
    loss_test = crit(output, labels)
    acc_test = accuracy(output, labels)
    total_test_loss += loss_test.item()
    total_test_acc = acc_test.item()
total_test_loss /= (127 - (int(127 * .7) + 1))
total_test_acc /= (127 - (int(127 * .7) + 1))

